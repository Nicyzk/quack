"""bench_forward.py's MoE EP forward with one shared expert (DeepSeek-V3's: a dense SwiGLU MLP of intermediate I on
every token, summed into the routed output), plus Mixture-of-Kittens as a baseline. bench_forward's protocol, shapes,
seeds, checks and reporting; every arm computes routed + shared inside its timed call.

Arms:
  mok              Mixture-of-Kittens (~/mixture-of-kittens, `mok.functional`): schedule build (top-k all-gather) +
                   the fused dispatch / MLP / combine megakernel with the shared expert inside; output = routed + shared.
                   One arm mok@N/M per --mok-comm-sms value N (its fwd_num_comm_sms; even, 4-52 recommended)
                   crossed with --mok-minibatch value M (its minibatch_size).
  mok_mxfp8        the same with the routed weights prequantized to MXFP8 (mok quantizes x in-kernel after the bf16
                   dispatch; combine and the shared expert stay bf16).
  megamoe          DeepGEMM's megamoe with num_shared_experts=1 (shared MLP inside the kernel).
  megamoe_mxfp8    the same in its fp8xfp8 mode: routed and shared weights and x MXFP8 at setup; the timed call copies
                   x, its scales and the shared expert's copy of the scales (the kernel's block layout).
  quack_* / sonic  bench_forward's arms + the shared expert as two dense quack GEMMs (gemm_act swiglu, gemm) and an
  / deepep_*       add into the arm's output, all inside the timed call (unfused).
  quack_intra_swap_mxfp8  + the shared expert as two dense quack MXFP8 GEMMs on the arm's prequantized x (weights
                   quantized at setup, the postact by a separate to_mx in the timed call) (unfused).
Reference: bench_forward.reference + swiglu(x W1s^T) W2s^T (reference_mxfp8 + the quantized shared expert for the
MXFP8 arms).

  torchrun --nproc_per_node=W bench_forward_shared_expert.py mok megamoe quack_intra_swap --t-local 1024 --ref_check --no-canary
"""
import argparse
import itertools
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

import bench_forward as bf
from bench_dispatch import MODEL_CONFIGS, REPS, WARM_S, WINDOW_S, Kernel, arm_stats, bench_all, r2r_check
from bench_forward import DispatchMode, SymmMemManager, _NOOP, expert_weights, megamoe_fp8_weights, megamoe_fp8_x, quantize_mx, ref_check, swiglu_cat
from quack.blockscaled.operand import BlockScaledOperand
from quack.gemm_interface import gemm, gemm_act
from sonicmoe.functional.distributed import build_rank_dedup_a_idx, compute_dispatch_metadata
from sonicmoe.functional.ep import _ag_routing_decision, _build_consumer_metadata


def reference_shared(x, topk_idx, scores, E, I, w1s, w2s):
    """bench_forward.reference (routed, fp32) + the shared expert on every token."""
    out = bf.reference(x, topk_idx, scores, E, I)
    return out + (swiglu_cat(x @ w1s.T) @ w2s.T).float()


def reference_shared_mxfp8(x, topk_idx, scores, E, I, w1s, w2s):
    """bench_forward.reference_mxfp8 + the shared expert with x, its weights and its postact quantized the same way
    (the postact through bf16 first: the unfused gemm_act stores it in bf16 before the separate quantize)."""
    out = bf.reference_mxfp8(x, topk_idx, scores, E, I)
    _, _, x_dq = quantize_mx(x)
    _, _, w1_dq = quantize_mx(w1s)
    _, _, w2_dq = quantize_mx(w2s)
    g, u = (x_dq @ w1_dq.T).chunk(2, dim=-1)
    _, _, p_dq = quantize_mx((F.silu(g) * u).to(torch.bfloat16))
    return out + p_dq @ w2_dq.T


def reference_shared_megamoe_mxfp8(x, topk_idx, scores, E, I, w1s, w2s):
    """bench_forward.reference_megamoe_mxfp8 + the shared expert with the same rounding (no score)."""
    out = bf.reference_megamoe_mxfp8(x, topk_idx, scores, E, I)
    dq = bf.dequant_megamoe
    g, u = (dq(x) @ dq(w1s).T).to(torch.bfloat16).float().chunk(2, dim=-1)
    return out + dq(F.silu(g) * u) @ dq(w2s).T


def _mok(model, x, topk_idx, scores, w1_cat, w1_il, w2, ws, rank, W, T_local, args, w1s, w2s, mxfp8=False):
    """One arm per --mok-comm-sms / --mok-mxfp8-comm-sms value x --mok-minibatch value (mok@N/M), each with its own
    MoKConfig and workspace."""
    from mok import functional, ops

    H, I, K, E_local, dev = model.H, model.I, model.topk, model.E // W, x.device
    topk64 = topk_idx.to(torch.int64).contiguous()
    gate, up = w1_cat[:, :I].contiguous(), w1_cat[:, I:].contiguous()  # (E_local, I, H) each; w2 (E_local, H, I)
    if mxfp8:  # routed weights prequantized (data, scale); x, shared expert and combine stay bf16 (mok's MXFP8 mode)
        gate, up, w2 = (ops.mxfp8_quantize(w, True, False)[:2] for w in (gate, up, w2))
    gate_s, up_s = w1s[:I].contiguous(), w1s[I:].contiguous()
    arms = {}
    sms = (args.mok_mxfp8_comm_sms if mxfp8 else args.mok_comm_sms).split(",")
    for n, mb in itertools.product(sms, args.mok_minibatch.split(",")):
        cfg = functional.MoKConfig(fwd_num_comm_sms=int(n), minibatch_size=int(mb))
        space = functional.get_workspace(cfg, dist.group.WORLD, device=dev, num_local_tokens=T_local, hidden_size=H, topk=K)
        out = [torch.zeros(T_local, H, dtype=torch.bfloat16, device=dev)]

        def run(cfg=cfg, space=space, out=out):
            schedule = functional.build_schedule(space, cfg, topk64, num_local_experts=E_local)
            out[0], _ = functional.forward(cfg, space, schedule, x, scores, gate_s, up_s, w2s, gate, up, w2)

        name = f"{'mok_mxfp8' if mxfp8 else 'mok'}@{int(n)}/{int(mb)}"
        arms[name] = Kernel(run, _NOOP, lambda out=out: out[0].float(), lambda out=out: out[0].zero_(), lambda ref: ref)
    return arms


def _megamoe_shared(model, x, topk_idx, scores, w1_cat, w1_il, w2, ws, rank, W, T_local, args, w1s, w2s):
    import deep_gemm

    H, I, E, K, dev = model.H, model.I, model.E, model.topk, x.device
    buf = deep_gemm.get_symm_buffer_for_mega_moe(ws.ep_group, E, T_local, K, H, I, mma_type="bf16xbf16", num_shared_experts=1)
    l1, l2 = deep_gemm.transform_weights_for_mega_moe(w1_cat, w2)
    l1s, l2s = deep_gemm.transform_weights_for_mega_moe(w1s, w2s)
    idx64 = topk_idx.to(torch.int64)
    y = torch.empty(T_local, H, dtype=torch.bfloat16, device=dev)

    def run():
        buf.x[:T_local].copy_(x)
        buf.topk_idx[:T_local].copy_(idx64)
        buf.topk_weights[:T_local].copy_(scores)
        deep_gemm.bf16_mega_moe(y=y, l1_weights=l1, l2_weights=l2, sym_buffer=buf, shared_l1_weights=l1s, shared_l2_weights=l2s)

    return {"megamoe": Kernel(run, buf.destroy, y.float, y.zero_, lambda ref: ref)}


def _megamoe_shared_mxfp8(model, x, topk_idx, scores, w1_cat, w1_il, w2, ws, rank, W, T_local, args, w1s, w2s):
    import deep_gemm
    from deep_gemm.utils import align

    H, I, E, K, dev = model.H, model.I, model.E, model.topk, x.device
    buf = deep_gemm.get_symm_buffer_for_mega_moe(ws.ep_group, E, T_local, K, H, I, mma_type="fp8xfp8", num_shared_experts=1)
    l1, l2 = deep_gemm.transform_weights_for_mega_moe(megamoe_fp8_weights(w1_cat), megamoe_fp8_weights(w2))

    def shared_weights(w):  # 2-D weights take the per-token cast's packed scales, strided for its TMA
        q, sf = megamoe_fp8_x(w)
        sf_tma = torch.empty_strided(sf.shape, (1, align(sf.shape[0], 4)), dtype=sf.dtype, device=dev)
        return q, sf_tma.copy_(sf)

    l1s, l2s = deep_gemm.transform_weights_for_mega_moe(shared_weights(w1s), shared_weights(w2s))
    # The shared fc1 reads x's scales in the kernel's block layout: token m of block b at row
    # b * align(block_m, 128) + (m // 128) * 128 + (m % 32) * 4 + (m % 128) // 32, k-blocks strided by the row count
    block_m = deep_gemm.get_block_m_for_mega_moe(W, E, buf.num_max_tokens_per_rank, T_local, K, "fp8xfp8")
    m = torch.arange(T_local, device=dev)
    b, mi = m // block_m, m % block_m
    sf_rows = b * align(block_m, 128) + (mi // 128) * 128 + (mi % 32) * 4 + (mi % 128) // 32
    xq, xs = megamoe_fp8_x(x)  # x arrives quantized; the copies are timed
    sf_shared = torch.zeros(buf.shared_l1_acts_sf.shape, dtype=torch.int32, device=dev).index_copy_(0, sf_rows, xs)
    idx64 = topk_idx.to(torch.int64)
    y = torch.empty(T_local, H, dtype=torch.bfloat16, device=dev)

    def run():
        buf.x[:T_local].copy_(xq)
        buf.x_sf[:T_local].copy_(xs)
        buf.shared_l1_acts_sf.copy_(sf_shared)
        buf.topk_idx[:T_local].copy_(idx64)
        buf.topk_weights[:T_local].copy_(scores)
        deep_gemm.fp8_fp4_mega_moe(y=y, l1_weights=l1, l2_weights=l2, sym_buffer=buf, shared_l1_weights=l1s, shared_l2_weights=l2s)

    ref = lambda ref_bf16: reference_shared_megamoe_mxfp8(x, topk_idx, scores, E, I, w1s, w2s)
    return {"megamoe_mxfp8": Kernel(run, buf.destroy, y.float, y.zero_, ref)}


def _with_dense_shared_mxfp8(kerns, x, topk_idx, scores, E, w1s, w2s, I):
    """bench_forward's MXFP8 arm + the shared expert as two dense quack MXFP8 GEMMs on the arm's prequantized x (the
    postact quantized in the timed call) and an add into the arm's output."""
    T, H = x.shape
    h = torch.empty(T, I, dtype=torch.bfloat16, device=x.device)
    y_s = torch.empty(T, H, dtype=torch.bfloat16, device=x.device)
    w1s_il = w1s.view(2, I, H).transpose(0, 1).reshape(2 * I, H).contiguous()
    w1s_t, w2s_t = BlockScaledOperand.quantize(w1s_il, "mxfp8").T, BlockScaledOperand.quantize(w2s, "mxfp8").T
    ref = lambda ref_bf16: reference_shared_mxfp8(x, topk_idx, scores, E, I, w1s, w2s)
    out = {}
    for name, kern in kerns.items():
        target = kern.read_out.__self__
        assert isinstance(target, torch.Tensor) and target.shape == (T, H), name
        parts = kern.run.parts
        x_mx = BlockScaledOperand.from_parts(parts.xq, bf.blocked_sf(parts.xs), "mxfp8")

        def run(kern=kern, target=target, x_mx=x_mx):
            kern.run()
            gemm_act(x_mx, w1s_t, activation="swiglu", postact_out=h, store_preact=False)
            gemm(BlockScaledOperand.quantize(h, "mxfp8"), w2s_t, out=y_s)
            target.add_(y_s)

        out[name] = kern._replace(run=run, ref=ref, label=kern.label + " + dense shared (quack mxfp8)")
    return out


def _output_of(kern):
    """() -> the arm's bf16 output tensor: read_out is the tensor's bound .float (quack, megamoe) or a lambda over an
    out dict refilled per call (sonic, deepep: out["o"])."""
    ro = kern.read_out
    if hasattr(ro, "__self__"):
        return lambda: ro.__self__
    holder = next(c.cell_contents for c in ro.__closure__ if isinstance(c.cell_contents, dict))
    return lambda: holder["o"]


def _with_dense_shared(kerns, x, w1s, w2s, I):
    """bench_forward's arms + the shared expert as two dense quack GEMMs and an add into the arm's output, timed."""
    T, H = x.shape
    h = torch.empty(T, I, dtype=torch.bfloat16, device=x.device)
    y_s = torch.empty(T, H, dtype=torch.bfloat16, device=x.device)
    w1s_il = w1s.view(2, I, H).transpose(0, 1).reshape(2 * I, H).contiguous()  # gate j, up j interleaved: quack's gated pairing
    w1s_t, w2s_t = w1s_il.T, w2s.T  # (H, 2I), (I, H): gemm's (K, N)
    out = {}
    for name, kern in kerns.items():
        target = _output_of(kern)  # resolved per call: sonic / deepep allocate their output inside run

        def run(kern=kern, target=target):
            kern.run()
            gemm_act(x, w1s_t, activation="swiglu", postact_out=h, store_preact=False)
            gemm(h, w2s_t, out=y_s)
            target().add_(y_s)

        out[name] = kern._replace(run=run, label=kern.label + " + dense shared (quack)")
    return out


def _dense_shared(model, x, topk_idx, scores, w1_cat, w1_il, w2, ws, rank, W, T_local, args, w1s, w2s):
    """The shared expert alone (timing-only): the two dense quack GEMMs _with_dense_shared adds to every arm, so
    mok minus this is an upper bound on mok's routed part."""
    I, dev = model.I, x.device
    out = torch.zeros(T_local, model.H, dtype=torch.bfloat16, device=dev)
    k = _with_dense_shared({"dense_shared": Kernel(_NOOP, _NOOP, out.float, out.zero_, lambda ref: ref)}, x, w1s, w2s, I)
    return {"dense_shared": k["dense_shared"]._replace(label=" (shared expert alone, timing-only)")}


BASELINES = {
    "mok": _mok,
    "mok_mxfp8": lambda *a: _mok(*a, mxfp8=True),
    "megamoe": _megamoe_shared,
    "megamoe_mxfp8": _megamoe_shared_mxfp8,
    "dense_shared": _dense_shared,
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("baselines", nargs="+", choices=sorted(set(BASELINES) | set(bf.BASELINES)))
    p.add_argument("--model", choices=list(MODEL_CONFIGS), default="dsv3")
    p.add_argument("--t-local", type=int, default=8192)
    p.add_argument("--comm-sms", type=int, default=16, help="quack_inter: SMs of the pull / push partition")
    p.add_argument("--mok-comm-sms", type=str, default="24", help="mok: fwd_num_comm_sms, one arm per comma-separated value (mok recommends even 4-52; its bf16 benchmark default 24)")
    p.add_argument("--mok-mxfp8-comm-sms", type=str, default="36", help="mok_mxfp8: fwd_num_comm_sms, one arm per comma-separated value (its mxfp8 benchmark default 36)")
    p.add_argument("--mok-minibatch", type=str, default="4096", help="mok / mok_mxfp8: minibatch_size, one arm per comma-separated value crossed with the comm SMs (mok's default 4096; its README: tune in 2048-16384, a multiple of 256 dividing the 131072 macrobatch)")
    p.add_argument("--ref_check", action="store_true")
    p.add_argument("--r2r_check", action="store_true")
    p.add_argument("--no-canary", action="store_true")
    p.add_argument("--sonic-metadata", action="store_true")
    args = p.parse_args()

    rank, W = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    dev = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(dev)
    dist.init_process_group("nccl", device_id=dev)
    model = MODEL_CONFIGS[args.model]
    H, I, E, topk = model
    assert E % W == 0
    T_local, E_local = args.t_local, E // W
    ws = SymmMemManager(None, dev)._get_or_alloc(T_local, H, topk, E_local, torch.bfloat16, DispatchMode.RANK_DEDUP_DISPATCH_TRITON)
    g = torch.Generator(device=dev).manual_seed(123 + rank)
    x = torch.randn(T_local, H, generator=g, device=dev, dtype=torch.bfloat16)
    topk_idx = torch.rand(T_local, E, generator=g, device=dev).topk(topk, dim=-1).indices.to(torch.int32)
    scores = torch.rand(T_local, topk, generator=g, device=dev).softmax(-1)
    local = [expert_weights(e, H, I, dev) for e in range(rank * E_local, (rank + 1) * E_local)]
    w1_cat = torch.stack([w1 for w1, _ in local])  # (E_local, 2I, H) rows [gate; up]
    w2 = torch.stack([w2 for _, w2 in local])  # (E_local, H, I)
    del local
    w1s, w2s = expert_weights(E, H, I, dev)  # the shared expert (expert id E: the same weights on every rank)
    w1_il = (
        w1_cat.view(E_local, 2, I, H).transpose(1, 2).reshape(E_local, 2 * I, H).contiguous()
        if not args.no_canary or any(b.startswith("quack") or b == "sonic" for b in args.baselines) else None
    )
    topk_idx_g = _ag_routing_decision(ws, topk_idx)
    meta = compute_dispatch_metadata(topk_idx_g, my_rank=rank, E_local=E_local, emit_combine=False)
    consumer = _build_consumer_metadata(
        expert_indices=meta["expert_local_padded"], token_indices=meta["a2a_token_indices"], TK=W * T_local * topk,
        E_local=E_local, s_reverse_idx_symm=ws.s_rev_symm,
    )
    total_m = int(consumer["expert_frequency_offset"][E_local])
    del topk_idx_g, meta, consumer
    ref = reference_shared(x, topk_idx, scores, E, I, w1s, w2s) if args.ref_check else None
    shared = (model, x, topk_idx, scores, w1_cat, w1_il, w2, ws, rank, W, T_local, args)

    with torch.no_grad():
        canary = _with_dense_shared(bf.BASELINES["sonic"](*shared), x, w1s, w2s, I) if "sonic" in args.baselines or not args.no_canary else None
        kerns = {}
        for name in args.baselines:
            if name == "sonic":
                kerns.update(canary)
            elif name in BASELINES:
                kerns.update(BASELINES[name](*shared, w1s, w2s))
            elif name == "quack_intra_swap_mxfp8":
                kerns.update(_with_dense_shared_mxfp8(bf.BASELINES[name](*shared), x, topk_idx, scores, E, w1s, w2s, I))
            else:
                kerns.update(_with_dense_shared(bf.BASELINES[name](*shared), x, w1s, w2s, I))
        if "sonic" not in kerns and canary is not None:
            kerns["sonic"] = canary["sonic"]
        del shared, w1_cat

        checks = {name: {} for name in kerns}
        for name, kern in kerns.items():
            if args.ref_check:
                checks[name]["ref"] = ref_check(kern, ref)
            if args.r2r_check:
                checks[name]["r2r"] = r2r_check(kern)

        arms = {name: kern.run for name, kern in kerns.items()}
        ms = bench_all(arms, ws.x_hdl)
        unstable = [None] * W
        dist.all_gather_object(unstable, [k for k, v in ms.items() if arm_stats(v)[3]])
        rerun = [k for k in arms if any(k in u for u in unstable)]
        if rerun:
            again = bench_all({k: arms[k] for k in dict.fromkeys(rerun + [k for k in ("sonic",) if k in arms])}, ws.x_hdl)
            ms.update({k: v for k, v in again.items() if not arm_stats(v)[3]})

    flops = 2 * total_m * H * I * 3 + 2 * T_local * H * I * 3  # routed GEMM1 (N = 2I) + GEMM2, + the shared expert
    stats = {k: arm_stats(v) for k, v in ms.items()}
    torch.cuda.synchronize()
    gathered = [None] * W
    dist.all_gather_object(gathered, (stats, checks))
    if rank == 0:
        med = {k: max((s[k] for s, _ in gathered), key=lambda t: t[0]) for k in ms}
        print(
            f"<bf16 MoE EP forward + 1 shared expert {args.model} (H={H} I={I} E={E} top-{topk}) | W={W} T_local={T_local} "
            f"total_m={total_m} (rank 0) | timed: x publish -> out, router outside | warm {WARM_S * 1e3:.0f} ms, "
            f"{REPS} x {WINDOW_S * 1e3:.0f} ms blocks, rerun={rerun}>"
        )
        for k, (m, lo, hi, _) in med.items():
            mean = sum(s[k][0] for s, _ in gathered) / W
            flag = "  UNSTABLE" if any(s[k][3] for s, _ in gathered) else ""
            print(
                f"  {k + kerns[k].label + ':':<60} {m * 1e3:7.0f} us  [{lo * 1e3:.0f}-{hi * 1e3:.0f}]  "
                f"{flops / m / 1e9:5.0f} TFLOP/s  rank-mean {mean * 1e3:.0f} us{flag}"
            )
        for name in kerns:
            if "ref" in checks[name]:
                ok = all(c[name]["ref"][0] for _, c in gathered)
                err = max(c[name]["ref"][1] for _, c in gathered)
                print(f"  {name} ref: {'ok' if ok else 'MISMATCH'} max rel err {err:.3e}")
            if "r2r" in checks[name]:
                ok = all(c[name]["r2r"][0] for _, c in gathered)
                delta = max(c[name]["r2r"][1] for _, c in gathered)
                print(f"  {name} r2r: {'bitwise' if ok else 'MISMATCH'} max|delta|={delta:.3e}")
    for kern in kerns.values():
        kern.free()
    del kerns, arms
    torch.cuda.synchronize()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
