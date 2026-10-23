"""MoE EP backward, bf16, routed + 1 shared expert: dout on the token's rank -> dx, the router score grads, dW1 / dW2 of
the local experts and the shared expert's three weight grads. bench_forward_shared_expert's shapes, seeds and
protocol (bench_all); every arm runs its forward once at setup (routing, handle, saved activations) and the backward
alone is timed.

Arms:
  mok         Mixture-of-Kittens' fused bf16 backward (functional.backward on the setup forward's schedule + context,
              shared expert inside; one arm mok@N per --mok-comm-sms value = its bwd_num_comm_sms)
  deepep_te   DeepEP dispatch of dout on the forward's handle (the combine's adjoint), the kernels of TE
              GroupedLinear's fc2 and fc1 backwards called explicitly (its grouped GEMM: dgrad NN, wgrad NT; TE
              dswiglu) on the setup forward's saved activations, the score grads reduced to their tokens by DeepEP's
              combine (its top-k weight reduction), DeepEP combine of the dx rows (the dispatch's adjoint); the
              shared expert as dense bf16 GEMMs. (TE modules free their saved inputs after one backward, so an
              autograd graph cannot be replayed; the explicit chain is the same kernels without autograd's overhead.)
  quack_intra_swap   our fused stages in reverse (see _quack_intra_swap): dout published + our prologue, the intra-SM
              pull + swap-AB GEMM with W2^T (the fc2 dgrad, dout rows landed packed per expert beside it), TE dswiglu
              on the forward's preact, the intra-SM combine kernel with W1^T on dh (the fc1 dgrad, each token's dx
              slice stored into the origins' slots) + reduce_over_k; the score grads over the forward's slots; the
              weight grads as torch's K-grouped GEMM over the packed rows; the shared expert as dense bf16 GEMMs.

--ref_check: every grad vs an fp32 autograd reference (per expert), max rel err < 2^-6 (as the forward benches);
--r2r_check: two backward calls on the same setup forward, bitwise. Cells: TFLOP/s (us) on the backward's GEMM FLOPs,
2 x the forward's (routed fc1 + fc2 + the shared expert over rank 0's rows). mok needs an EP size of 4, 8, ...

  torchrun --nproc_per_node=4 bench_backward.py mok deepep_te quack_intra_swap --model dsv3_e128 --t-local 1024 --ref_check --r2r_check
"""
import argparse
import os

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from bench_dispatch import MODEL_CONFIGS, REPS, WARM_S, WINDOW_S, arm_stats, bench_all
from bench_forward import DispatchMode, SymmMemManager, expert_weights, swiglu_cat

GRADS = ("dx", "dscores", "dw1", "dw2", "dw1s", "dw2s")  # (T, H), (T, K), (E_local, 2I, H), (E_local, H, I), (2I, H), (H, I)


def reference_backward(x, topk_idx, scores, dout, E, I, rank, E_local, w1s, w2s, group):
    """fp32 autograd per expert: dx and the score grads of the local tokens over every expert; dW of the local
    experts over every rank's tokens (all-gathered); the shared expert."""
    T, H = x.shape
    W = dist.get_world_size(group)

    def gathered(t):
        g = [torch.empty_like(t) for _ in range(W)]
        dist.all_gather(g, t, group=group)
        return torch.cat(g)

    x32, dout32 = x.float(), dout.float()
    scores32 = scores.clone().requires_grad_()
    dx = torch.zeros(T, H, dtype=torch.float32, device=x.device)
    for e in range(E):
        t, k = (topk_idx == e).nonzero(as_tuple=True)
        if t.numel() == 0:
            continue
        w1, w2 = (w.float() for w in expert_weights(e, H, I, x.device))
        xt = x32[t].requires_grad_()
        loss = (scores32[t, k, None] * (swiglu_cat(xt @ w1.T) @ w2.T) * dout32[t]).sum()
        loss.backward()  # scores32.grad accumulates
        dx.index_add_(0, t, xt.grad)
    x_g, topk_g, scores_g, dout_g = (gathered(t) for t in (x, topk_idx, scores, dout))
    dw1 = torch.zeros(E_local, 2 * I, H, dtype=torch.float32, device=x.device)
    dw2 = torch.zeros(E_local, H, I, dtype=torch.float32, device=x.device)
    for i in range(E_local):
        e = rank * E_local + i
        t, k = (topk_g == e).nonzero(as_tuple=True)
        if t.numel() == 0:
            continue
        w1, w2 = (w.float().requires_grad_() for w in expert_weights(e, H, I, x.device))
        loss = (scores_g[t, k, None] * (swiglu_cat(x_g[t].float() @ w1.T) @ w2.T) * dout_g[t].float()).sum()
        dw1[i], dw2[i] = torch.autograd.grad(loss, (w1, w2))
    xs, w1s32, w2s32 = x32.clone().requires_grad_(), w1s.float().requires_grad_(), w2s.float().requires_grad_()
    (swiglu_cat(xs @ w1s32.T) @ w2s32.T * dout32).sum().backward()
    return {"dx": dx + xs.grad, "dscores": scores32.grad, "dw1": dw1, "dw2": dw2, "dw1s": w1s32.grad, "dw2s": w2s32.grad}


def _mok(model, x, topk_idx, scores, dout, w1_cat, w2, w1s, w2s, rank, W, T_local, args, ws):
    """One arm per --mok-comm-sms value (mok@N): its own MoKConfig (bwd_num_comm_sms = N) and workspace, the setup
    forward's schedule + context replayed by every backward call."""
    from mok import functional

    H, I, K, E_local, dev = model.H, model.I, model.topk, model.E // W, x.device
    topk64 = topk_idx.to(torch.int64).contiguous()
    gate, up = w1_cat[:, :I].contiguous(), w1_cat[:, I:].contiguous()  # (E_local, I, H) each; w2 (E_local, H, I)
    gate_s, up_s = w1s[:I].contiguous(), w1s[I:].contiguous()
    arms = {}
    for n in args.mok_comm_sms.split(","):
        cfg = functional.MoKConfig(fwd_num_comm_sms=24, bwd_num_comm_sms=int(n))
        space = functional.get_workspace(cfg, dist.group.WORLD, device=dev, num_local_tokens=T_local, hidden_size=H, topk=K)
        schedule = functional.build_schedule(space, cfg, topk64, num_local_experts=E_local)
        _, ctx = functional.forward(cfg, space, schedule, x, scores, gate_s, up_s, w2s, gate, up, w2)
        out = {}

        def run(cfg=cfg, space=space, schedule=schedule, ctx=ctx, out=out):
            out["g"] = functional.backward(cfg, space, schedule, ctx, dout, x, scores, gate_s, up_s, w2s, gate, up, w2)

        def grads(out=out):
            d_x, d_s, d_gate, d_up, d_down, d_sg, d_su, d_sd = out["g"]
            return {"dx": d_x, "dscores": d_s, "dw1": torch.cat((d_gate, d_up), 1), "dw2": d_down, "dw1s": torch.cat((d_sg, d_su)), "dw2s": d_sd}

        arms[f"mok@{int(n)}"] = (run, grads)
    return arms


def _deepep_te(model, x, topk_idx, scores, dout, w1_cat, w2, w1s, w2s, rank, W, T_local, args, ws):
    """The explicit chain of GroupedLinear's backward kernels (TE modules free their saved inputs after one
    backward, so a retained autograd graph cannot be replayed): TE's grouped GEMM dgrad (NN) and wgrad (NT) on the
    setup forward's saved activations, TE dswiglu, the shared expert as dense bf16 GEMMs."""
    import deep_ep
    import transformer_engine.pytorch  # noqa: F401  (loads the torch extension)
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.cpp_extensions import general_grouped_gemm

    H, I, K, E, E_local, dev = model.H, model.I, model.topk, model.E, model.E // W, x.device
    bf16 = torch.bfloat16
    buffer = deep_ep.ElasticBuffer(dist.group.WORLD, num_max_tokens_per_rank=T_local, hidden=H, num_topk=K, explicitly_destroy=True)
    num_sms = buffer.get_theoretical_num_sms(E, K)
    common = dict(num_sms=num_sms, num_max_tokens_per_rank=T_local, num_experts=E, expert_alignment=1)
    recv_x, _, recv_w, handle, _ = buffer.dispatch(
        x=x, topk_idx=topk_idx.to(deep_ep.topk_idx_t), topk_weights=scores, do_expand=True, do_cpu_sync=True, **common
    )
    n, m_splits = int(handle.num_expanded_tokens), handle.num_recv_tokens_per_expert_list
    w1_l, w2_l = list(w1_cat), list(w2)  # per expert (2I, H), (H, I)
    split = lambda t: list(torch.split(t, m_splits))
    gemm = lambda A, B, out, layout: general_grouped_gemm(A, B, out, [None] * E_local, bf16, single_output=len(out) == 1, layout=layout, m_splits=m_splits, grad=True)
    # the forward's saved activations: fc1 preact h, postact a, the scores onto the rows a_s (fc2's input); the shared expert's
    xr, wr = recv_x[:n], recv_w[:n].to(bf16)
    h = torch.empty(n, 2 * I, dtype=bf16, device=dev)
    gemm(w1_l, split(xr), [h], "TN")
    a = tex.swiglu(h, None)
    a_s = a * wr[:, None]
    hs = x @ w1s.T
    a_sh = tex.swiglu(hs, None)
    da_s, dxr = torch.empty(n, I, dtype=bf16, device=dev), torch.empty(n, H, dtype=bf16, device=dev)
    dw1, dw2 = torch.empty_like(w1_cat), torch.empty_like(w2)
    out = {}

    def run():
        dy = buffer.dispatch(x=dout, handle=handle, do_expand=True, do_cpu_sync=False, **common)[0][:n]  # the combine's adjoint
        dys = split(dy)
        gemm(w2_l, dys, [da_s], "NN")  # fc2 dgrad: da_s = dy @ W2
        gemm(split(a_s), dys, list(dw2), "NT")  # fc2 wgrad: dW2_e = dy_e^T @ a_s_e
        dw = torch.bmm(da_s[:, None, :], a[:, :, None]).view(-1).float()  # the score grad of every row
        dh = tex.dswiglu(da_s * wr[:, None], h, None)
        gemm(w1_l, split(dh), [dxr], "NN")  # fc1 dgrad: dxr = dh @ W1
        gemm(split(xr), split(dh), list(dw1), "NT")  # fc1 wgrad: dW1_e = dh_e^T @ xr_e
        dhs = tex.dswiglu(dout @ w2s, hs, None)  # the shared expert
        out["dw2s"], out["dw1s"] = dout.T @ a_sh, dhs.T @ x
        dx, out["dscores"] = buffer.combine(dxr, handle=handle, topk_weights=dw, num_sms=num_sms)[:2]  # the dispatch's adjoint
        out["dx"] = dx + dhs @ w1s

    grads = lambda: {"dx": out["dx"], "dscores": out["dscores"], "dw1": dw1, "dw2": dw2, "dw1s": out["dw1s"], "dw2s": out["dw2s"]}
    return {"deepep_te": (run, grads)}


def _quack_intra_swap(model, x, topk_idx, scores, dout, w1_cat, w2, w1s, w2s, rank, W, T_local, args, ws):
    """Our fused stages in reverse (quack_intra_swap's kernels, bench_forward's prologue / walk / slots):
      1. dscores = <dout, y slot> per (token, k) over the forward's combine slots (torch bmm, local);
      2. fused dispatch + fc2 dgrad: dout published, the prologue (the walk over the dout buffers), the intra-SM
         pull + swap-AB GEMM with W2^T (plain epilogue) -> da rows packed per expert, the landed dout rows beside;
      3. da * score, TE dswiglu on the forward's preact h -> dh;
      4. fused fc1 dgrad + combine: the intra-SM combine kernel with W1^T on dh, each token's dx slice stored into
         the origins' slots, barrier, reduce_over_k with unit weights -> dx;
      5. wgrad: torch's K-grouped GEMM over the packed rows (dW2 = dy^T a, dW1 = dh^T x); the shared expert as
         dense bf16 GEMMs.
    The transposed weight copies are made at setup (per step in training, not per backward)."""
    import transformer_engine.pytorch  # noqa: F401
    import transformer_engine_torch as tex
    from combine.combine_push import make_combine_buffers
    from dispatch.prologue import make_prologue
    from quack.cute_dsl_utils import get_device_capacity, get_max_active_clusters, torch2cute_dtype_map
    from quack.gemm_default_epi import GemmDefaultEpiMixin
    from quack.gemm_tvm_ffi_utils import make_scheduler_args, make_varlen_args
    from reduce_over_k import reduce_over_k

    import bench_forward as bf

    H, I, K, E_local, dev = model.H, model.I, model.topk, model.E // W, x.device
    bf16, dtype = torch.bfloat16, torch2cute_dtype_map[x.dtype]
    TK, max_rows = W * T_local * K, bf.routed_rows(topk_idx, rank, W, E_local, ws.ep_group)
    clusters = get_max_active_clusters(bf.CLUSTER[0] * bf.CLUSTER[1], get_device_capacity(dev))
    desc = lambda: (torch.empty(torch.cuda.get_device_properties(dev).multi_processor_count * 16, dtype=torch.int64, device=dev),) if bf.SM100 else ()
    buffers = make_combine_buffers(T_local, K, H, bf16, ws.ep_group)
    varlen = lambda cu: make_varlen_args(None, None, None, cu_seqlens_n=cu)
    # --- the forward, once: x landed packed per expert (the fused gated GEMM1's pull), its postact a, the preact h
    fn1, Cls1, block_rows = bf.compile_gemm1_gated_swap(dtype, model, W, T_local, intra_pull=H, gather_B=False)
    ws.x_symm.copy_(x)
    prologue, meta, walk1 = make_prologue(topk_idx, ws.x_peer_bufs, rank, W, T_local, K, E_local, max_rows, block_rows, buffers.slot_base, ws.ep_group)
    prologue()
    cu, n = meta.cu, int(meta.cu[E_local])
    rows = torch.empty(max_rows, H, dtype=bf16, device=dev)
    a = torch.empty(max_rows, I, dtype=bf16, device=dev)
    sched1 = make_scheduler_args(clusters, 8, prologue.sem1, dispatch_args=(walk1.flags, walk1.first_src_row, meta.a_idx))
    fn1(bf.interleave_gate_up(w1_cat), rows, None, None, Cls1.EpilogueArguments(mAuxOut=a.T), sched1, varlen(cu), None, None, walk1.args, *desc())
    fn_h, Cls_h, _ = bf.compile_gemm1_gated_swap(dtype, model, W, T_local, intra_pull=0, gather_B=False, activation=None, gated=False)
    h = torch.empty(max_rows, 2 * I, dtype=bf16, device=dev)
    flags_full, sem_h = torch.full_like(walk1.flags, 1 << 30), torch.zeros(1, dtype=torch.int32, device=dev)
    sched_h = make_scheduler_args(clusters, 8, sem_h, dispatch_args=(flags_full, walk1.first_src_row, meta.a_idx))
    fn_h(w1_cat, rows, None, None, Cls_h.EpilogueArguments(mAuxOut=h.T), sched_h, varlen(cu), None, None)
    fn2, _, _ = bf.compile_gemm2_swap(dtype, model, W, T_local, intra_push=H)
    y = torch.empty(max_rows, H, dtype=bf16, device=dev)  # D's shape for the combine kernel; never written
    flags2 = torch.zeros(E_local, -(-W * T_local // block_rows), dtype=torch.int32, device=dev)
    first2 = torch.tensor([((rank + 1) % W) * TK // W], dtype=torch.int32, device=dev)
    epi2 = GemmDefaultEpiMixin.EpilogueArguments(add_to_output=None, rounding_mode=None)
    combine_tail = (buffers.slot_base, T_local * K, meta.slot_addr)
    sched2 = make_scheduler_args(clusters, 8, prologue.sem2, combine_args=(flags2, meta.s_scatter, first2))
    fn2(w2, a, y.T, None, epi2, sched2, varlen(cu), None, None, *combine_tail, *desc())  # y into the slots
    ws.o_hdl.barrier(channel=1)
    torch.cuda.synchronize()
    slots_y = buffers.slots.clone()  # the forward's y per (token, k): dscores' operand (the backward refills the slots)
    scores_g = torch.empty(TK, dtype=scores.dtype, device=dev)
    dist.all_gather_into_tensor(scores_g, scores.view(-1), group=ws.ep_group)
    s_rows = scores_g[meta.s_scatter[:n].long()].to(bf16)  # each packed row's score
    a_s = a[:n] * s_rows[:, None]  # dW2's operand: dy^T a with the score on a
    hs = x @ w1s.T
    a_sh = tex.swiglu(hs, None)
    # --- the backward's operands
    dout_symm = symm_mem.empty(T_local, H, dtype=bf16, device=dev)
    hd = symm_mem.rendezvous(dout_symm, group=ws.ep_group)
    dout_peers = [hd.get_buffer(q, (T_local, H), bf16) for q in range(W)]
    prologue_d, meta_d, walk_d = make_prologue(topk_idx, dout_peers, rank, W, T_local, K, E_local, max_rows, block_rows, buffers.slot_base, ws.ep_group)
    fn_dg2, Cls_dg2, _ = bf.compile_gemm1_gated_swap(dtype, model, W, T_local, intra_pull=H, gather_B=False, activation=None, gated=False)
    w2t, w1t = w2.transpose(1, 2).contiguous(), w1_cat.transpose(1, 2).contiguous()  # (E, I, H), (E, H, 2I): K-major
    rows_d = torch.empty(max_rows, H, dtype=bf16, device=dev)  # dout rows packed per expert
    da = torch.empty(max_rows, I, dtype=bf16, device=dev)
    sched_dg2 = make_scheduler_args(clusters, 8, prologue_d.sem1, dispatch_args=(walk_d.flags, walk_d.first_src_row, meta_d.a_idx))
    rt_dg2 = (Cls_dg2.EpilogueArguments(mAuxOut=da.T), sched_dg2, varlen(meta_d.cu))
    sched_dg1 = make_scheduler_args(clusters, 8, prologue_d.sem2, combine_args=(flags2, meta_d.s_scatter, first2))
    rt_dg1 = (epi2, sched_dg1, varlen(meta_d.cu))
    ones, dx_r = torch.ones(T_local, K, dtype=scores.dtype, device=dev), torch.empty(T_local, H, dtype=bf16, device=dev)
    offs = cu[1 : E_local + 1].to(torch.int32)
    out = {}

    def run():
        out["dscores"] = torch.bmm(slots_y, dout[:, :, None]).view(T_local, K).float()  # 1
        dout_symm.copy_(dout)  # 2
        prologue_d()
        fn_dg2(w2t, rows_d, None, None, *rt_dg2, None, None, walk_d.args, *desc())
        dh = tex.dswiglu(da[:n] * s_rows[:, None], h[:n], None)  # 3
        fn2(w1t, dh, y[:n].T, None, *rt_dg1, None, None, *combine_tail, *desc())  # 4: dx rows into the origins' slots
        ws.o_hdl.barrier(channel=1)
        reduce_over_k(buffers.slots, ones, dx_r)
        out["dw2"] = torch._grouped_mm(rows_d[:n].t(), a_s, offs)  # 5: dW2_e = dy_e^T a_e (the score on a)
        out["dw1"] = torch._grouped_mm(dh.t(), rows[:n], offs)  # dW1_e = dh_e^T x_e
        dhs = tex.dswiglu(dout @ w2s, hs, None)  # the shared expert
        out["dw2s"], out["dw1s"] = dout.T @ a_sh, dhs.T @ x
        out["dx"] = dx_r + dhs @ w1s

    return {"quack_intra_swap": (run, lambda: dict(out))}


BASELINES = {"mok": _mok, "deepep_te": _deepep_te, "quack_intra_swap": _quack_intra_swap}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("baselines", nargs="+", choices=list(BASELINES))
    p.add_argument("--model", choices=list(MODEL_CONFIGS), default="dsv3")
    p.add_argument("--t-local", type=int, default=8192)
    p.add_argument("--mok-comm-sms", type=str, default="28", help="mok: bwd_num_comm_sms, one arm per comma-separated value (its benchmark default 28)")
    p.add_argument("--ref_check", action="store_true")
    p.add_argument("--r2r_check", action="store_true")
    args = p.parse_args()

    rank, W = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    dev = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(dev)
    dist.init_process_group("nccl", device_id=dev)
    model = MODEL_CONFIGS[args.model]
    H, I, E, K = model
    assert E % W == 0
    T_local, E_local = args.t_local, E // W
    g = torch.Generator(device=dev).manual_seed(123 + rank)
    x = torch.randn(T_local, H, generator=g, device=dev, dtype=torch.bfloat16)
    topk_idx = torch.rand(T_local, E, generator=g, device=dev).topk(K, dim=-1).indices.to(torch.int32)
    scores = torch.rand(T_local, K, generator=g, device=dev).softmax(-1)
    dout = torch.randn(T_local, H, generator=g, device=dev, dtype=torch.bfloat16)
    local = [expert_weights(e, H, I, dev) for e in range(rank * E_local, (rank + 1) * E_local)]
    w1_cat = torch.stack([w1 for w1, _ in local])  # (E_local, 2I, H) rows [gate; up]
    w2 = torch.stack([w2 for _, w2 in local])  # (E_local, H, I)
    del local
    w1s, w2s = expert_weights(E, H, I, dev)  # the shared expert
    rows0 = (topk_idx.flatten() // E_local == 0).sum()  # rank 0's rows: every rank's pairs routed to its experts
    dist.all_reduce(rows0)
    total_m = int(rows0)
    ws = SymmMemManager(None, dev)._get_or_alloc(T_local, H, K, E_local, torch.bfloat16, DispatchMode.RANK_DEDUP_DISPATCH_TRITON)
    hdl = ws.x_hdl  # bench_all's barrier
    ref = reference_backward(x, topk_idx, scores, dout, E, I, rank, E_local, w1s, w2s, dist.group.WORLD) if args.ref_check else None

    arms = {}
    for name in args.baselines:
        arms.update(BASELINES[name](model, x, topk_idx, scores, dout, w1_cat, w2, w1s, w2s, rank, W, T_local, args, ws))
    checks = {name: {} for name in arms}
    for name, (run, grads) in arms.items():
        run()
        torch.cuda.synchronize()
        if args.ref_check:
            got = grads()
            errs = {k: float((got[k].float() - ref[k]).abs().max() / ref[k].abs().max()) for k in GRADS}
            checks[name]["ref"] = (max(errs.values()) < 2**-6, errs)
        if args.r2r_check:
            first = {k: v.clone() for k, v in grads().items()}
            run()
            torch.cuda.synchronize()
            checks[name]["r2r"] = all(torch.equal(first[k], v) for k, v in grads().items())
    ms = bench_all({name: run for name, (run, _) in arms.items()}, hdl)
    unstable = [None] * W
    dist.all_gather_object(unstable, [k for k, v in ms.items() if arm_stats(v)[3]])
    rerun = [k for k in arms if any(k in u for u in unstable)]
    if rerun:
        again = bench_all({k: arms[k][0] for k in rerun}, hdl)
        ms.update({k: v for k, v in again.items() if not arm_stats(v)[3]})

    flops = 2 * (2 * total_m * H * 2 * I + 2 * total_m * H * I + 2 * T_local * H * 3 * I)  # dgrad + wgrad of every GEMM
    stats = {k: arm_stats(v) for k, v in ms.items()}
    torch.cuda.synchronize()
    gathered = [None] * W
    dist.all_gather_object(gathered, (stats, checks))
    if rank == 0:
        med = {k: max((s[k] for s, _ in gathered), key=lambda t: t[0]) for k in ms}
        print(
            f"<bf16 MoE EP backward + shared expert {args.model} (H={H} I={I} E={E} top-{K}) | W={W} T_local={T_local} "
            f"total_m={total_m} (rank 0) | warm {WARM_S * 1e3:.0f} ms, {REPS} x {WINDOW_S * 1e3:.0f} ms blocks, rerun={rerun}>"
        )
        for k, (m, lo, hi, _) in med.items():
            flag = "  UNSTABLE" if any(s[k][3] for s, _ in gathered) else ""
            print(f"  {k + ':':<20} {m * 1e3:7.0f} us  [{lo * 1e3:.0f}-{hi * 1e3:.0f}]  {flops / m / 1e9:5.0f} TFLOP/s{flag}")
        for name in arms:
            if args.ref_check:
                ok = all(c[name]["ref"][0] for _, c in gathered)
                errs = {k: max(c[name]["ref"][1][k] for _, c in gathered) for k in GRADS}
                print(f"  {name} ref: {'ok' if ok else 'MISMATCH'} " + " ".join(f"{k} {v:.2e}" for k, v in errs.items()))
            if args.r2r_check:
                print(f"  {name} r2r: {'ok' if all(c[name]['r2r'] for _, c in gathered) else 'MISMATCH'}")
    dist.barrier()
    os._exit(0)


if __name__ == "__main__":
    main()
