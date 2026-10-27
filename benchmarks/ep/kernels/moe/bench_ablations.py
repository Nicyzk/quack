"""Tile walk order ablations of the fused MoE stages (results/2026-10-19/final_results.md, "MoE tile walk order
ablations").

walk (one GPU): the bare swap-AB gated GEMM1 (bench_forward's compile_gemm1_gated_swap, gate flags pre-satisfied)
over rank 0's rows of an emulated W-rank EP at uniform routing: E/W experts, T_local * top-k rows, each expert's rows
in W equal slices, slice k = source rank k. Arms: expert = expert-major, slices ascending; expert_ring = expert-major,
slices rotated to start at rank 1 (the production order); stage = source-rank-major (Comet's layer order: one GEMM
group per (stage, expert), stages in ring order, the weights replicated per stage so the HBM traffic is the real
schedule's, in which no expert's weights survive in L2 across the other E/W - 1 experts between its stages). The
stage arm's rows are the expert arm's permuted, and its postact must match row for row. Interleaved single launches,
median and min.

  CUDA_VISIBLE_DEVICES=0 python bench_ablations.py walk --model dsv4_pro --t-local 4096 --emulated-w 2,4,8

src-order (W ranks, torchrun): bench_forward's quack_intra_swap pipeline built twice, with each expert's source ranks
walked from the peer after this rank (ring, production) or from rank 0 (ascending): the pull walk, GEMM1's tile
order and GEMM2's slot-store order all follow it (bench_forward --src-order). dispatch+gemm1 (prologue + GEMM1) and
gemm2+combine (GEMM2 + barrier + reduce) timed as stages beside the whole forward, bench_all's protocol, both forwards
ref-checked.

  torchrun --nproc_per_node=4 bench_ablations.py src-order --model dsv4_pro --t-local 4096 --ref_check

src-order-fc1 (W ranks, torchrun): the quantized fused dispatch + fc1 stage (bench_new_kernels.fc1_stage_quantized,
--format nvfp4 / mxfp8, --activation) built twice, ring vs ascending source order (make_prologue's first_src_rank):
the fused stage and the pull alone per order, the GEMM1 alone once, both fused stages ref-checked.

  torchrun --nproc_per_node=8 bench_ablations.py src-order-fc1 --model dsv4_pro --t-local 4096 --format nvfp4 --activation geglu --ref_check
"""
import argparse
import os
import statistics

import cutlass
import torch
import torch.distributed as dist

import bench_forward as bf
from bench_dispatch import MODEL_CONFIGS, REPS, WARM_S, WINDOW_S, arm_stats, bench_all
from bench_forward import CLUSTER, DispatchMode, SymmMemManager, expert_weights, ref_check
from dispatch.pull import make_dispatch_pull
from quack.gemm_tvm_ffi_utils import make_scheduler_args, make_varlen_args


def walk(args):
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    model = MODEL_CONFIGS[args.model]
    H, I, E, K = model
    T_local, dtype = args.t_local, cutlass.BFloat16
    g = torch.Generator(device=dev).manual_seed(123)
    i32 = lambda vals: torch.tensor(vals, dtype=torch.int32, device=dev)
    print(f"<GEMM1 tile walk {args.model} (H={H} I={I} E={E} top-{K}) | T_local={T_local} | {args.rounds} rounds of single launches>")
    for W in [int(v) for v in args.emulated_w.split(",")]:
        E_local = E // W
        fn, Cls, block_rows = bf.compile_gemm1_gated_swap(dtype, model, W, T_local, intra_pull=0, gather_B=False)
        # rank 0's rows per expert at uniform routing; --full-blocks: one whole token block per slice instead, so
        # the stage arm pays no tile quantization and the difference is the weight re-streaming alone
        rows_e = W * block_rows if args.full_blocks else T_local * K // E_local
        slice_rows = rows_e // W
        assert args.full_blocks or rows_e * E_local == T_local * K, (W, rows_e)
        assert slice_rows * W == rows_e, (W, rows_e)
        total_m = rows_e * E_local
        clusters = bf.get_max_active_clusters(CLUSTER[0] * CLUSTER[1], bf.get_device_capacity(dev))
        rows = torch.randn(total_m, H, generator=g, device=dev, dtype=torch.bfloat16)
        w1_il = bf.interleave_gate_up(torch.randn(E_local, 2 * I, H, generator=g, device=dev, dtype=torch.bfloat16) * 0.02)
        # row keys: slice k of expert e plays source rank k, its rows at k * T_local + ... (ascending inside the expert)
        j = torch.arange(rows_e, device=dev)
        key_e = (j // slice_rows) * T_local + j % slice_rows
        keys = key_e.repeat(E_local).to(torch.int32)
        # stage order: stages 1, 2, ..., W-1, 0 (rank 0's ring), experts inside; group (s, e) = expert e's slice s
        ring = [(s + 1) % W for s in range(W)]
        q = torch.arange(W * total_m // W, device=dev)  # the stage arm's row index
        grp, t = q // slice_rows, q % slice_rows
        s = torch.tensor(ring, device=dev)[grp // E_local]
        perm = (grp % E_local) * rows_e + s * slice_rows + t  # stage row -> expert row
        rows_stage = rows[perm].contiguous()
        w1_stage = w1_il.unsqueeze(0).expand(W, E_local, 2 * I, H).reshape(W * E_local, 2 * I, H)
        keys_stage = keys[perm].contiguous()
        outs = {}

        def arm(name, cu, keys, first_src_row, w, b):
            ncols = -(-int((cu[1:] - cu[:-1]).max()) // block_rows)
            flags = torch.full((cu.numel() - 1, ncols), 1 << 30, dtype=torch.int32, device=dev)  # never waits
            sem = torch.zeros(1, dtype=torch.int32, device=dev)
            a = torch.empty(b.shape[0], I, dtype=torch.bfloat16, device=dev)
            outs[name] = a
            sched = make_scheduler_args(clusters, 8, sem, dispatch_args=(flags, i32([first_src_row]), keys))
            rt = (Cls.EpilogueArguments(mAuxOut=a.T), sched, make_varlen_args(None, None, None, cu_seqlens_n=cu))
            return lambda: (sem.zero_(), fn(w, b, None, None, *rt, None, None))

        cu_e = i32([e * rows_e for e in range(E_local + 1)])
        cu_s = i32([g_ * slice_rows for g_ in range(W * E_local + 1)])
        arms = {
            "expert": arm("expert", cu_e, keys, 0, w1_il, rows),
            "expert_ring": arm("expert_ring", cu_e, keys, T_local, w1_il, rows),
            "stage": arm("stage", cu_s, keys_stage, 0, w1_stage, rows_stage),
        }
        for run in arms.values():
            for _ in range(3):
                run()
        torch.cuda.synchronize()
        err = {k: (outs[k].float() - outs["expert"].float()).abs().max().item() for k in ("expert_ring",)}
        err["stage"] = (outs["stage"].float() - outs["expert"][perm].float()).abs().max().item()
        times = {k: [] for k in arms}
        names = list(arms)
        for r in range(args.rounds):
            for k in names[r % len(names):] + names[: r % len(names)]:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                arms[k]()
                end.record()
                end.synchronize()
                times[k].append(start.elapsed_time(end) * 1e3)
        flops = 2 * total_m * H * 2 * I
        print(f"W={W} E_local={E_local} rows={total_m} ({rows_e} per expert, {slice_rows} per slice) tile_n={block_rows}")
        for k in names:
            med, lo = statistics.median(times[k]), min(times[k])
            print(f"  {k:12s} {flops / med / 1e6:7.0f} TFLOP/s ({med:6.0f} us)  min {lo:6.0f} us  max|diff vs expert| {err.get(k, 0.0):.3g}")
        del rows, rows_stage, w1_il, w1_stage, outs, arms
        torch.cuda.empty_cache()


def src_order(args):
    rank, W = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    dev = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(dev)
    dist.init_process_group("nccl", device_id=dev)
    model = MODEL_CONFIGS[args.model]
    H, I, E, K = model
    T_local, E_local = args.t_local, E // W
    args.sonic_metadata = False
    ws = SymmMemManager(None, dev)._get_or_alloc(T_local, H, K, E_local, torch.bfloat16, DispatchMode.RANK_DEDUP_DISPATCH_TRITON)
    g = torch.Generator(device=dev).manual_seed(123 + rank)
    x = torch.randn(T_local, H, generator=g, device=dev, dtype=torch.bfloat16)
    topk_idx = torch.rand(T_local, E, generator=g, device=dev).topk(K, dim=-1).indices.to(torch.int32)
    scores = torch.rand(T_local, K, generator=g, device=dev).softmax(-1)
    local = [expert_weights(e, H, I, dev) for e in range(rank * E_local, (rank + 1) * E_local)]
    w1_cat = torch.stack([w1 for w1, _ in local])
    w2 = torch.stack([w2 for _, w2 in local])
    del local
    w1_il = w1_cat.view(E_local, 2, I, H).transpose(1, 2).reshape(E_local, 2 * I, H).contiguous()
    ref = bf.reference(x, topk_idx, scores, E, I) if args.ref_check else None
    shared = (model, x, topk_idx, scores, w1_cat, w1_il, w2, ws, rank, W, T_local, args)

    with torch.no_grad():
        kerns, arms, wire = {}, {}, {}
        call = lambda *steps: (lambda: [s() for s in steps])
        for order in ("ring", "ascending"):
            args.src_order = order
            kerns[order] = bf.BASELINES["quack_intra_swap"](*shared)["quack_intra_swap"]
            run = kerns[order].run
            st = lambda *names, run=run: [run.steps[run.step_names.index(n)] for n in names]
            arms[f"dispatch+gemm1 {order}"] = call(*st("prologue", "gemm1"))  # the prologue resets the flags and semaphores
            arms[f"gemm2+combine {order}"] = call(run.parts.sem2.zero_, *st("gemm2", "barrier", "reduce"))
            arms[f"forward {order}"] = run
            # the transport alone: our pull on every SM over the same walk (bench_stages' pull alone bound)
            pull = make_dispatch_pull(ws.x_peer_bufs, rank, "intra", None, run.parts.rows, scatter=True)[0](run.parts.walk1)
            arms[f"pull alone {order}"] = call(*st("prologue"), pull)
            wire[order] = int(run.parts.walk1.num_rows.item()) * H * 2
        checks = {name: ref_check(kern, ref) for name, kern in kerns.items()} if args.ref_check else {}
        total_m = int(kerns["ring"].run.parts.cu[-1].item())
        for fn in arms.values():
            fn()
        torch.cuda.synchronize()
        dist.barrier()
        ms = bench_all(arms, ws.x_hdl)
        unstable = [None] * W
        dist.all_gather_object(unstable, [k for k, v in ms.items() if arm_stats(v)[3]])
        rerun = [k for k in arms if any(k in u for u in unstable)]
        if rerun:
            again = bench_all({k: arms[k] for k in rerun}, ws.x_hdl)
            ms.update({k: v for k, v in again.items() if not arm_stats(v)[3]})

    flops1, flops2 = 2 * total_m * H * 2 * I, 2 * total_m * H * I
    unit = {  # (work, name): the stage's GEMM FLOPs, or the pull's wire bytes
        k: (wire[k.split()[-1]], "GB/s") if k.startswith("pull") else
        ((flops1 if k.startswith("dispatch") else flops2 if k.startswith("gemm2") else flops1 + flops2), "TFLOP/s")
        for k in arms
    }
    stats = {k: arm_stats(v) for k, v in ms.items()}
    gathered = [None] * W
    dist.all_gather_object(gathered, (stats, checks))
    if rank == 0:
        med = {k: max((s[k] for s, _ in gathered), key=lambda t: t[0]) for k in ms}
        print(
            f"<source rank order {args.model} (H={H} I={I} E={E} top-{K}) | W={W} T_local={T_local} total_m={total_m} (rank 0) "
            f"| warm {WARM_S * 1e3:.0f} ms, {REPS} x {WINDOW_S * 1e3:.0f} ms blocks, rerun={rerun}>"
        )
        for k, (m, lo, hi, _) in med.items():
            flag = "  UNSTABLE" if any(s[k][3] for s, _ in gathered) else ""
            work, name = unit[k]
            rate = work / m / (1e6 if name == "GB/s" else 1e9)  # m in ms
            print(f"  {k:24s} {rate:7.0f} {name} ({m * 1e3:6.0f} us)  [{lo * 1e3:.0f}, {hi * 1e3:.0f}]{flag}")
        for order in checks:
            print(f"  forward {order}: ref " + " ".join(f"{c}" for _, c in gathered for c in [c[order]]))


def src_order_fc1(args):
    import bench_new_kernels as bnk
    from quack.activation import gate_fn_map

    rank, W = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    dev = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(dev)
    dist.init_process_group("nccl", device_id=dev)
    model = MODEL_CONFIGS[args.model]
    H, I, E, K = model
    T_local, E_local = args.t_local, E // W
    fmt, act = args.format, args.activation
    args.sonic_metadata = False
    ws = SymmMemManager(None, dev)._get_or_alloc(T_local, H, K, E_local, torch.bfloat16, DispatchMode.RANK_DEDUP_DISPATCH_TRITON)
    g = torch.Generator(device=dev).manual_seed(123 + rank)
    x = torch.randn(T_local, H, generator=g, device=dev, dtype=torch.bfloat16)
    topk_idx = torch.rand(T_local, E, generator=g, device=dev).topk(K, dim=-1).indices.to(torch.int32)
    scores = torch.rand(T_local, K, generator=g, device=dev).softmax(-1)
    local = [expert_weights(e, H, I, dev) for e in range(rank * E_local, (rank + 1) * E_local)]
    w1_cat = torch.stack([w1 for w1, _ in local])
    w2 = torch.stack([w2 for _, w2 in local])
    del local
    w1_il = w1_cat.view(E_local, 2, I, H).transpose(1, 2).reshape(E_local, 2 * I, H).contiguous()
    shared = (model, x, topk_idx, scores, w1_cat, w1_il, w2, ws, rank, W, T_local, args)

    with torch.no_grad():
        arms, posts, keep = {}, {}, []
        block_rows = bf.compile_gemm1_gated_swap(cutlass.BFloat16, model, W, T_local, intra_pull=H, gather_B=False)[2]
        desc = torch.empty(torch.cuda.get_device_properties(dev).multi_processor_count * 16, dtype=torch.int64, device=dev)
        for order in ("ring", "ascending"):
            args.src_order = order
            run = bf.BASELINES["quack_intra_swap"](*shared)["quack_intra_swap"].run  # the bf16 pipeline: its landed rows in this order
            pt = run.parts
            total_m = int(pt.cu[-1].item())
            first = (rank + 1) % W if order == "ring" else 0
            fmt_arms, fmt_posts, pinned = bnk.fc1_stage_quantized(
                fmt, act, model, x, topk_idx, w1_cat, ws, rank, W, T_local, block_rows, pt.clusters, desc, pt.rows[:total_m],
                first_src_rank=first,
            )
            keep.append((pinned, run))
            for name, fn in fmt_arms.items():
                if name.startswith("gemm1 alone"):
                    if order == "ring":
                        arms[name] = fn  # the same GEMM over the landed rows, whatever their order
                    continue
                arms[f"{name} {order}"] = fn
            posts[f"dispatch+gemm1 {act} {fmt} {order}"] = fmt_posts[f"dispatch+gemm1 {act} {fmt}"]
        gemm_flops = 2 * total_m * H * (2 * I if act in gate_fn_map else I)
        for fn in arms.values():  # every kernel once before the timed calls
            fn()
        torch.cuda.synchronize()
        dist.barrier()
        checks = {}
        if args.ref_check:
            for name, (a, want_fn) in posts.items():
                a.zero_()
                torch.cuda.synchronize()
                dist.barrier()
                arms[name]()
                torch.cuda.synchronize()
                dist.barrier()
                want = want_fn()
                err = ((a[:total_m].float() - want).abs().max() / want.abs().max()).item()
                checks[name] = (err < 2**-6, err)
        ms = bench_all(arms, ws.x_hdl)
        unstable = [None] * W
        dist.all_gather_object(unstable, [k for k, v in ms.items() if arm_stats(v)[3]])
        rerun = [k for k in arms if any(k in u for u in unstable)]
        if rerun:
            again = bench_all({k: arms[k] for k in rerun}, ws.x_hdl)
            ms.update({k: v for k, v in again.items() if not arm_stats(v)[3]})

    stats = {k: arm_stats(v) for k, v in ms.items()}
    gathered = [None] * W
    dist.all_gather_object(gathered, (stats, checks))
    if rank == 0:
        med = {k: max((s[k] for s, _ in gathered), key=lambda t: t[0]) for k in ms}
        print(
            f"<source rank order, {fmt} fc1 {act} {args.model} (H={H} I={I} E={E} top-{K}) | W={W} T_local={T_local} "
            f"total_m={total_m} (rank 0) | warm {WARM_S * 1e3:.0f} ms, {REPS} x {WINDOW_S * 1e3:.0f} ms blocks, rerun={rerun}>"
        )
        for k, (m, lo, hi, _) in med.items():
            flag = "  UNSTABLE" if any(s[k][3] for s, _ in gathered) else ""
            rate = "" if k.startswith("pull") else f"{gemm_flops / m / 1e9:5.0f} TFLOP/s"
            print(f"  {k + ':':<36} {m * 1e3:7.0f} us  [{lo * 1e3:.0f}-{hi * 1e3:.0f}]  {rate}{flag}")
        for name in checks:
            ok = all(c[name][0] for _, c in gathered)
            err = max(c[name][1] for _, c in gathered)
            print(f"  {name} ref: {'ok' if ok else 'MISMATCH'} max rel err {err:.3e}")
    dist.barrier()
    os._exit(0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=("walk", "src-order", "src-order-fc1"))
    p.add_argument("--model", choices=list(MODEL_CONFIGS), default="dsv4_pro")
    p.add_argument("--t-local", type=int, default=4096)
    p.add_argument("--emulated-w", type=str, default="2,4,8", help="walk: the EP sizes emulated, one table row each")
    p.add_argument("--rounds", type=int, default=20, help="walk: rounds of one launch per arm, arm order rotated")
    p.add_argument("--full-blocks", action="store_true", help="walk: one whole token block per slice (W blocks per expert) instead of the real rows")
    p.add_argument("--ref_check", action="store_true", help="src-order: both whole forwards vs the reference")
    p.add_argument("--format", choices=("nvfp4", "mxfp8"), default="nvfp4", help="src-order-fc1: the stage's operand format")
    p.add_argument("--activation", default="geglu", help="src-order-fc1: the stage's activation")
    args = p.parse_args()
    {"walk": walk, "src-order": src_order, "src-order-fc1": src_order_fc1}[args.mode](args)


if __name__ == "__main__":
    main()
