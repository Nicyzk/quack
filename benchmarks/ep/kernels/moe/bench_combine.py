"""Run EP combine baselines through one interface (sibling of bench_dispatch.py, same protocol).

Every arm reduces GEMM2's expert-major output y (random rows here: the combine only reads it) into out
(T_local, H) on each token's origin rank, each token's K rows weighted by its router scores. Shape from --model
(MODEL_CONFIGS), bf16, uniform routing (distinct top-k per token, seeded per rank); routing and combine metadata are
sonic's, built once outside the timed region; W comes from the launch. Correctness = max relative error of each
arm's out vs an fp32 reference (the methods round differently), the counters of our arms bitwise.

  sonic       sonic-moe PR54 rank-dedup combine, triton (rank_dedup_combine_triton: producer pre-reduce, barrier,
              destination gather): the reference method and the contention canary
  sonic_a2a   sonic's routed combine (a2a_combine_triton: K rows per token pulled from the owning peers, fp32 sum)
  deepep      DeepEP V2 elastic combine (~/DeepEP) on --sms SMs, routed rows, its own reduce epilogue
  quack_inter combine/combine_push.py: the dispatch pull run the other way, 1 CTA/SM on --sms, this rank's y rows
              pushed to slot (token, k) of their origin rank's (T_local, K, H) buffer with a per-token counter
              release-add; the SM's smem full of rows in flight (16 warps on B300), the inter-SM leg
  quack_intra the same push with 4 warps / 56 KB stage per SM: what the transfer warps look like inside the GEMM
  dst_reduce  the destination reduce every push placement pays: fp32 sum over a token's K slots (torch), the leg
              to add to quack_* for a whole combine

Each factory returns a Kernel: run / free plus read_out (out as fp32, and the counters for our arms), zero and
ref (the fp32 reference -> what read_out must match). --ref_check / --r2r_check work generically; the chosen
baselines are timed together by bench_dispatch.bench_all (its fixed protocol), sonic as the canary, a barrier-only
arm as the per-call floor.

Usage:
    PYTHONPATH=~/quack:~/sonic-moe-pr54:~/DeepEP torchrun --nproc_per_node=8 --standalone bench_combine.py quack_inter deepep --ref_check
    ... bench_combine.py quack_inter quack_intra dst_reduce --sms 24
"""
import os

os.environ.setdefault("TRITON_CACHE_AUTOTUNING", "1")  # sonic's triton autotune choices persist across launches

import argparse
import sys

import torch
import torch.distributed as dist
from sonicmoe.distributed_utils import DispatchMode, SymmMemManager
from sonicmoe.functional.distributed import build_rank_dedup_a_idx, compute_dispatch_metadata
from sonicmoe.functional.distributed.ep_combine import a2a_combine_triton, rank_dedup_combine_triton
from sonicmoe.functional.ep import _ag_routing_decision, _build_consumer_metadata, _build_single_row

from bench_dispatch import (
    _NOOP,
    MODEL_CONFIGS,
    REPS,
    WARM_S,
    WINDOW_S,
    Kernel,
    SMStream,
    _run_synced,
    _tup,
    arm_stats,
    bench_all,
    r2r_check,
)


def _rel_err(a, b):
    return float((a.float() - b.float()).abs().max() / b.float().abs().max())


def ref_check(kern, ref):
    """(ok, max rel err) of kern's fresh output vs its reference: float outputs within 2^-7 (bf16 rounding of the
    methods' different summation orders), integer outputs (our counters) bitwise."""
    _run_synced(kern)
    ok, err = True, 0.0
    for got, want in zip(_tup(kern.read_out()), _tup(kern.ref(ref))):
        if got.is_floating_point():
            e = _rel_err(got, want)
            ok, err = ok and e < 2 ** -7, max(err, e)
        else:
            ok = ok and torch.equal(got, want)
    return ok, err


# Every factory: (model, y, scores, scores_g, topk_idx_g, meta, ws, sm_stream, rank, W, T_local, args) -> Kernel.
# y: this rank's expert-major GEMM2 output (total_m rows of ws.y_symm); scores: (T_local, K) this rank's router
# scores, scores_g: every rank's, flat; the rest as bench_dispatch (meta with sonic's combine metadata).
def _sonic(model, y, scores, scores_g, topk_idx_g, meta, ws, sm_stream, rank, W, T_local, args):
    out = torch.empty(T_local, model.H, dtype=torch.bfloat16, device=y.device)

    def run():
        rank_dedup_combine_triton(
            ws.y_symm, ws.s_rev_symm, scores_g, meta["peer_present_mask"], ws.partial_combine_buf, out,
            K=model.topk, T_local=T_local, group=ws.ep_group,
            partial_combine_hdl=ws.partial_combine_hdl, partial_combine_peer_bufs=ws.partial_combine_peer_bufs,
            my_rank=rank,
            mine_slot_idx=meta["mine_slot_idx"], mine_count=meta["mine_count"], combine_contrib_C=meta["combine_contrib_C"],
            combine_work_list=meta["combine_work_list_multi"], combine_work_count=meta["combine_work_count_multi"],
            combine_single_k=meta["combine_single_k"],
            y_peer_bufs=ws.y_peer_bufs, s_reverse_peer_bufs=ws.s_rev_peer_bufs, single_row=meta["single_row"],
        )

    return Kernel(run, _NOOP, lambda: out.float(), out.zero_, lambda ref: ref)


def _sonic_a2a(model, y, scores, scores_g, topk_idx_g, meta, ws, sm_stream, rank, W, T_local, args):
    out = torch.empty(T_local, model.H, dtype=torch.bfloat16, device=y.device)

    def run():
        a2a_combine_triton(
            ws.y_symm, ws.s_rev_symm, meta["my_dst_rank"], scores, out, K=model.topk, group=ws.ep_group,
            peer_y_base=ws.a2a_peer_y_base, peer_s_base=ws.a2a_peer_s_base, my_rank=rank,
        )

    return Kernel(run, _NOOP, lambda: out.float(), out.zero_, lambda ref: ref)


def _deepep(model, y, scores, scores_g, topk_idx_g, meta, ws, sm_stream, rank, W, T_local, args):
    """DeepEP V2 elastic combine (~/DeepEP) in expand mode with multiple reduction off, i.e. the routed pattern:
    one row per (token, expert) slot crosses the wire, summed at the origin by DeepEP's reduce epilogue. The
    expanded send buffer is filled from y through the shared global slot id, scores folded in (expand mode takes
    no weights); num_sms = the --sms partition's count (DeepEP's bandwidth-model default when --sms is unset)."""
    sys.path.insert(0, os.path.expanduser("~/DeepEP"))
    import deep_ep

    E, K, dev = model.E, model.topk, y.device
    buffer = deep_ep.ElasticBuffer(
        ws.ep_group, num_max_tokens_per_rank=T_local, hidden=model.H, allow_multiple_reduction=False,
        explicitly_destroy=True,
    )
    num_sms = sm_stream.sms if args.sms else buffer.get_theoretical_num_sms(E, K)
    idx64 = topk_idx_g[rank].to(deep_ep.topk_idx_t)
    weights = torch.ones(idx64.shape, dtype=torch.float32, device=dev)
    common = dict(num_sms=num_sms, num_max_tokens_per_rank=T_local, num_experts=E, expert_alignment=1)
    recv_x, _, _, handle, _ = buffer.dispatch(
        x=torch.zeros(T_local, model.H, dtype=torch.bfloat16, device=dev), topk_idx=idx64, topk_weights=weights,
        do_expand=True, do_cpu_sync=True, **common,
    )
    torch.cuda.synchronize()
    # recv_src_metadata[i] = (global token id, _, expanded row of slot 0..K-1 or -1 if that expert is elsewhere).
    n = int(handle.num_recv_tokens)
    md = handle.recv_src_metadata[:n].reshape(n, -1).to(torch.int64)
    slot = md[:, 2 : 2 + K]
    pos = (md[:, :1] * K + torch.arange(K, device=dev)).masked_select(slot >= 0)  # global slot id
    rows = slot.masked_select(slot >= 0)
    y_send = torch.zeros(recv_x.shape[0], model.H, dtype=torch.bfloat16, device=dev)
    y_send[rows] = (y[ws.s_rev_symm[pos].long()].float() * scores_g[pos, None]).to(torch.bfloat16)
    res = {}

    def run():
        res["out"] = buffer.combine(x=y_send, handle=handle, num_sms=num_sms)[0]

    run()  # DeepEP allocates out per call: one call so read_out / zero have a tensor before the first check
    torch.cuda.synchronize()
    return Kernel(
        run, buffer.destroy, lambda: res["out"].float(), lambda: res["out"].zero_(), lambda ref: ref,
        label=f" (deepep_sms={num_sms})",
    )


def _quack(model, y, scores, scores_g, topk_idx_g, meta, ws, sm_stream, rank, W, T_local, args, mode):
    """Our push (combine/combine_push.py) over the torch-built walk, mode "inter" / "intra" (see the module
    header), on --sms, into symmetric slots + per-token counters. Output = the fp32 sum over each token's K slots
    (the destination reduce every push placement pays, timed as dst_reduce) and the counters (K per token)."""
    from combine.combine_push import make_combine_buffers, make_combine_push, make_combine_walk

    K, E_local = model.topk, model.E // W
    buffers = make_combine_buffers(T_local, K, model.H, torch.bfloat16, ws.ep_group)
    ready = torch.ones(E_local, 1, dtype=torch.int32, device=y.device)  # standalone: every row is ready
    walk, pos = make_combine_walk(topk_idx_g, ws.s_rev_symm, meta["expert_frequency_offset"], rank, E_local, buffers, ready, y.shape[0])
    y_send = (y.float() * scores_g[pos, None]).to(torch.bfloat16)  # scores folded in, as a fused epilogue would
    push = make_combine_push(y_send, mode, sm_stream.sms)(walk, 1)
    return Kernel(
        push, _NOOP, lambda: (buffers.slots.float().sum(1), buffers.counters),
        lambda: (buffers.slots.zero_(), buffers.counters.zero_()), lambda ref: (ref, buffers.targets),
    )


def _dst_reduce(model, y, scores, scores_g, topk_idx_g, meta, ws, sm_stream, rank, W, T_local, args):
    """The local leg after every push: fp32 sum over a token's K slots, bf16 out (HBM-bound; content irrelevant)."""
    slots = torch.randn(T_local, model.topk, model.H, dtype=torch.bfloat16, device=y.device)
    out = torch.empty(T_local, model.H, dtype=torch.bfloat16, device=y.device)
    return Kernel(lambda: torch.sum(slots, 1, out=out), _NOOP, lambda: out.float(), out.zero_, None)


BASELINES = {
    "sonic": _sonic,
    "sonic_a2a": _sonic_a2a,
    "deepep": _deepep,
    "quack_inter": lambda *a: _quack(*a, "inter"),
    "quack_intra": lambda *a: _quack(*a, "intra"),
    "dst_reduce": _dst_reduce,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("baselines", nargs="+", choices=list(BASELINES))
    p.add_argument("--model", choices=list(MODEL_CONFIGS), default="dsv3", help="shape: H, I, E, top-k (MODEL_CONFIGS)")
    p.add_argument("--t-local", type=int, default=8192)
    p.add_argument(
        "--sms", type=int, default=0,
        help="SMs the transport gets: a green-context partition for the timed loop and DeepEP's num_sms "
        "(0 = the whole GPU; DeepEP's own default)",
    )
    p.add_argument("--ref_check", action="store_true", help="each arm's fresh output (zeroed first) vs the fp32 reference")
    p.add_argument("--r2r_check", action="store_true", help="run each arm twice; bit-identity of the two outputs")
    args = p.parse_args()

    # Process group, shape, symm-mem workspace, routing and metadata, as bench_dispatch (plus sonic's combine
    # metadata: emit_combine), then y, the router scores and the fp32 reference (built once, outside the timed region).
    rank, W = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    dev = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(dev)
    dist.init_process_group("nccl", device_id=dev)
    model = MODEL_CONFIGS[args.model]
    H, _, E, topk = model
    assert E % W == 0, f"{args.model}: E={E} not divisible by W={W}"
    T_local, E_local = args.t_local, E // W
    sm_stream = SMStream(dev.index, args.sms if args.sms else 1 << 30)
    ws = SymmMemManager(None, dev)._get_or_alloc(
        T_local, H, topk, E_local, torch.bfloat16, DispatchMode.RANK_DEDUP_DISPATCH_TRITON
    )
    g = torch.Generator(device=dev).manual_seed(123 + rank)
    x = torch.randn(T_local, H, generator=g, device=dev, dtype=torch.bfloat16)
    topk_idx = torch.rand(T_local, E, generator=g, device=dev).topk(topk, dim=-1).indices.to(torch.int32)
    ws.x_symm.copy_(x)
    topk_idx_g = _ag_routing_decision(ws, topk_idx)
    ws.x_hdl.barrier()
    meta = compute_dispatch_metadata(topk_idx_g, my_rank=rank, E_local=E_local, emit_combine=True)
    meta["single_row"] = _build_single_row(topk_idx_g, meta["combine_single_k"], rank, E_local)  # rank-dedup gather
    consumer = _build_consumer_metadata(
        expert_indices=meta["expert_local_padded"], token_indices=meta["a2a_token_indices"], TK=W * T_local * topk,
        E_local=E_local, s_reverse_idx_symm=ws.s_rev_symm,
    )
    meta["recv_pos"] = consumer["s_reverse_local"]
    meta["s_scatter_idx"] = consumer["s_scatter_idx"]
    meta["expert_frequency_offset"] = consumer["expert_frequency_offset"]
    meta["x_idx_expanded_remap_for_rank_dedup"] = build_rank_dedup_a_idx(
        dst_rank_flat=meta["dst_rank_flat"], s_reverse_local=consumer["s_reverse_local"],
        rank_dedup_recv_pos=meta["rank_dedup_recv_pos"], my_rank=rank, out=ws.x_idx_expanded_remap_for_rank_dedup_buf,
    )
    total_m = int(meta["expert_frequency_offset"][E_local])
    y = ws.y_symm[:total_m]
    y.copy_(torch.randn(total_m, H, generator=g, device=dev, dtype=torch.bfloat16))
    # Router scores: fp32 softmax over the token's K logits, as sonic's TC softmax-topk router emits; every rank's
    # gathered flat, indexed by the global slot id (sonic's _all_gather_topk_scores).
    scores = torch.rand(T_local, topk, generator=g, device=dev).softmax(-1)
    scores_g = torch.empty(W * T_local * topk, dtype=scores.dtype, device=dev)
    dist.all_gather_into_tensor(scores_g, scores.view(-1), group=ws.ep_group)
    ws._ensure_a2a_peer_base()
    ws._ensure_partial_combine_buf()
    ws.o_hdl.barrier()
    # Reference in fp32, following sonic's A2A combine: slot (t, k) is row s_rev[pos] of peer my_dst_rank[t, k].
    dst = meta["my_dst_rank"].view(-1)
    pos = rank * T_local * topk + torch.arange(T_local * topk, device=dev)
    ref = torch.zeros(T_local, H, dtype=torch.float32, device=dev)
    for q in range(W):
        sel = (dst == q).nonzero().flatten()
        rows = ws.s_rev_peer_bufs[q][pos[sel]].long()
        ref.index_add_(0, sel // topk, ws.y_peer_bufs[q][rows].float() * scores.view(-1)[sel, None])
    shared = (model, y, scores, scores_g, topk_idx_g, meta, ws, sm_stream, rank, W, T_local, args)

    canary = _sonic(*shared)
    kerns = {name: canary if name == "sonic" else BASELINES[name](*shared) for name in args.baselines}

    checks = {name: {} for name in kerns}
    for name, kern in kerns.items():
        if args.ref_check and kern.ref is not None:
            checks[name]["ref"] = ref_check(kern, ref)
        if args.r2r_check:
            checks[name]["r2r"] = r2r_check(kern)

    arms = {name: kern.run for name, kern in kerns.items()}
    if "sonic" not in arms:
        arms["sonic"] = canary.run  # timed next to the baselines: the contention canary
    arms["barrier"] = _NOOP  # the floor inside every call: the symm-mem barrier plus the enqueue
    with sm_stream.ctx():
        ms = bench_all(arms, ws.o_hdl)
        # Unstable arms get one rerun next to the canary; a stable rerun replaces them. The rerun set is the union
        # over ranks: every rank must call the same collectives.
        unstable = [None] * W
        dist.all_gather_object(unstable, [k for k, v in ms.items() if arm_stats(v)[3]])
        rerun = [k for k in arms if any(k in u for u in unstable)]
        if rerun:
            again = bench_all({k: arms[k] for k in dict.fromkeys(rerun + ["sonic"])}, ws.o_hdl)
            ms.update({k: v for k, v in again.items() if k in rerun and not arm_stats(v)[3]})

    # Wire rows for this rank's routing: routed = off-rank (token, k) slots; dedup = present off-rank peers.
    present = meta["peer_present_mask"]  # (W, T_local)
    rows_routed = int((meta["my_dst_rank"] != rank).sum())
    rows_dedup = int(present.sum() - present[rank].sum())
    nbytes = {k: rows_routed * H * 2 for k in ms}
    nbytes["sonic"] = rows_dedup * H * 2
    nbytes["dst_reduce"] = T_local * (topk + 1) * H * 2  # HBM: slots read + out written
    stats = {k: arm_stats(v) for k, v in ms.items()}
    gathered = [None] * W
    dist.all_gather_object(gathered, (stats, checks))
    if rank == 0:
        print(
            f"<bf16 EP combine {args.model} (H={H} E={E} top-{topk}) | W={W} T_local={T_local} "
            f"sms={sm_stream.sms}/{sm_stream.total} | wire routed {rows_routed} rows ({rows_routed * H * 2 / 1e6:.0f} MB), "
            f"dedup {rows_dedup} rows | warm {WARM_S * 1e3:.0f} ms, {REPS} x {WINDOW_S * 1e3:.0f} ms blocks, rerun={rerun}>"
        )
        for k in ms:
            med, lo, hi, _ = max((g[k] for g, _ in gathered), key=lambda t: t[0])  # slowest rank sets the collective
            mean = sum(g[k][0] for g, _ in gathered) / W  # cross-rank mean of the medians (sonic's number)
            label = kerns[k].label if k in kerns else ""
            unit = "TB/s" if k == "dst_reduce" else "GB/s"
            rate = "" if k == "barrier" else f"{nbytes[k] / med / (1e9 if unit == 'TB/s' else 1e6):.{1 if unit == 'TB/s' else 0}f} {unit}"
            flag = "  UNSTABLE" if any(g[k][3] for g, _ in gathered) else ""
            print(
                f"  {k + label + ':':<28} {med * 1e3:7.0f} us  [{lo * 1e3:.0f}-{hi * 1e3:.0f}]  {rate:>10}  "
                f"rank-mean {mean * 1e3:.0f} us{flag}"
            )
        for name in kerns:
            for what in checks[name]:
                ok = all(c[name][what][0] for _, c in gathered)
                delta = max(c[name][what][1] for _, c in gathered)
                print(f"  {name} {what}: {'ok' if ok else 'MISMATCH'} max rel err={delta:.3e}")
    for kern in kerns.values():
        kern.free()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
