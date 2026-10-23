"""Comet (ByteDance flux) MoE AG + scatter grouped GEMM, the GEMM1 baseline, under flux's own env (torch 2.4.1; the
wheel only imports there), so it runs as its own torchrun after bench_fused_dispatch_gemm.py on the same GPUs:

  LD_PRELOAD=/scratch/gpfs/TRIDAO/zy4586/envs/ipc_attr_shim.so \\
  /scratch/gpfs/TRIDAO/zy4586/envs/flux/bin/torchrun --nproc_per_node=4 --standalone -- h100_flux_moe_runner.py --t-local 8192

Same shape (dsv3), seeds, routing and weights as the bench (seed 123 + rank, torch.rand top-k routing, w1 (E_local, 2I, H)):
the header prints total_m for the cross-check. tp = ep = W: every rank holds its T_local input tokens and E / W experts
at full width, flux all-gathers every token (no dedup) and scatters each (token, expert) pair to its expert's rows.
Timing mirrors bench_dispatch.bench_all (200 ms warm-up and 500 ms windows sized by a 20-call probe, 5 blocks, 100
chunks, rank-MAX'd counts) with a NCCL all-reduce of one element in place of the symm-mem barrier; flux's own
per-call op.clear_buffers() (its test's protocol) is inside the timed call. Cells: slowest rank's median, TFLOP/s on
GEMM1's FLOPs 2 * total_m * H * 2I. --ref_check compares the output rows of every local expert against torch.
"""
import argparse
import os
import statistics
import time

import flux
import torch
import torch.distributed as dist

try:  # the NCCL channel floor every reference runs with (kernels/misc/nccl_env.py)
    from kernels.misc import nccl_env  # noqa: F401
except ImportError:  # flux env without duckpond on PYTHONPATH: import it by path
    import importlib.util as _ilu

    _spec = _ilu.spec_from_file_location(
        "nccl_env", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../misc/nccl_env.py")
    )
    _spec.loader.exec_module(_ilu.module_from_spec(_spec))

MODELS = {  # H, I, E, top-k (bench_dispatch.MODEL_CONFIGS)
    "dsv3": (7168, 2048, 256, 8),
    "dsv4_pro": (7168, 3072, 384, 6),
    "kimi_k2": (7168, 2048, 384, 8),
    "glm_5_2": (6144, 2048, 256, 8),
    "qwen3_5_397b": (4096, 1024, 512, 10),
}
WARM_S, WINDOW_S, REPS, SAMPLES = 0.2, 0.5, 5, 100


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(MODELS), default="dsv3")
    p.add_argument("--t-local", type=int, default=8192)
    p.add_argument("--sm-margin", type=int, default=0, help="flux: SMs left free by its GEMM")
    p.add_argument("--ref_check", action="store_true")
    args = p.parse_args()
    rank, W = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    dev = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(dev)
    dist.init_process_group("nccl")
    H, I, E, topk = MODELS[args.model]
    T_local, E_local = args.t_local, E // W
    g = torch.Generator(device=dev).manual_seed(123 + rank)
    x = torch.randn(T_local, H, generator=g, device=dev, dtype=torch.bfloat16)
    topk_idx = torch.rand(T_local, E, generator=g, device=dev).topk(topk, dim=-1).indices.to(torch.int32)
    w1 = torch.randn(E_local, 2 * I, H, generator=g, device=dev, dtype=torch.bfloat16) * 0.02  # (E_local, N, K)

    # Routing in flux's terms: every rank's choices in rank order, global splits, stable scatter index (rows of an
    # expert in (token, k) order).
    choosed = torch.empty(W, T_local, topk, dtype=torch.int32, device=dev)
    dist.all_gather_into_tensor(choosed, topk_idx)
    choosed = choosed.reshape(W * T_local, topk)
    splits_gpu = torch.bincount(choosed.reshape(-1), minlength=E).to(torch.int32)
    order = torch.argsort(choosed.reshape(-1), stable=True)
    scatter_index = torch.empty(W * T_local * topk, dtype=torch.int32, device=dev)
    scatter_index[order] = torch.arange(W * T_local * topk, dtype=torch.int32, device=dev)
    scatter_index = scatter_index.reshape(W * T_local, topk)
    first = rank * E_local
    total_m = int(splits_gpu[first : first + E_local].sum())
    out = torch.empty(total_m, 2 * I, dtype=torch.bfloat16, device=dev)

    tp_group = dist.group.WORLD
    ep_group = dist.new_group(list(range(W)))  # ep = tp = W: experts across every rank, full width each
    flux.init_flux_shm(tp_group)
    tp_env = flux.DistEnvTPWithEP(tp_group=tp_group, nnodes=1, ep_group=ep_group)
    moe_args = flux.MoeArguments(
        max_ntokens=W * T_local, hidden=H, ffn_hidden=2 * I, nexperts=E, topk=topk,
        input_dtype=torch.bfloat16, output_dtype=torch.bfloat16,
    )
    op = flux.GemmGroupedV3AGScatter(tp_env=tp_env, moe_args=moe_args)

    def run():
        op.clear_buffers()
        op.forward(
            inputs_shard=x, weights=w1, splits_gpu=splits_gpu, scatter_index=scatter_index, outputs_buf=out,
            fast_accum=False, sm_margin=args.sm_margin,
        )

    ref_ok = None
    if args.ref_check:
        out.zero_()
        run()
        torch.cuda.synchronize()
        xg = torch.empty(W * T_local, H, dtype=torch.bfloat16, device=dev)
        dist.all_gather_into_tensor(xg, x)
        tok = order // topk  # global token of every expert-sorted row
        start = int(splits_gpu[:first].sum())
        err = 0.0
        for e in range(E_local):
            n = int(splits_gpu[first + e])
            rows = xg[tok[start : start + n]]
            ref = (rows @ w1[e].T).float()
            got = out[start - int(splits_gpu[:first].sum()) : start - int(splits_gpu[:first].sum()) + n].float()
            err = max(err, ((got - ref).abs().max() / (ref.abs().max() + 1e-6)).item())
            start += n
        ref_ok = err

    # bench_dispatch.bench_all's protocol with a one-element all-reduce as the per-call sync.
    sync = torch.zeros(1, device=dev)
    for _ in range(2):
        dist.all_reduce(sync)
        run()
    dist.barrier()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        dist.all_reduce(sync)
        run()
    torch.cuda.synchronize()
    per_call = torch.tensor([(time.perf_counter() - t0) / 20], device=dev)
    dist.all_reduce(per_call, op=dist.ReduceOp.MAX)
    s = per_call.item()
    warm, iters = max(10, int(WARM_S / s)), max(20, int(WINDOW_S / s))
    blocks = []
    for _ in range(REPS):
        for _ in range(warm):
            dist.all_reduce(sync)
            run()
        dist.barrier()
        chunk = max(1, iters // SAMPLES)
        events = []
        for i in range(iters):
            if i % chunk == 0:
                events.append(torch.cuda.Event(enable_timing=True))
                events[-1].record()
            dist.all_reduce(sync)
            run()
        events.append(torch.cuda.Event(enable_timing=True))
        events[-1].record()
        torch.cuda.synchronize()
        bounds = list(range(0, iters, chunk)) + [iters]
        blocks.append([events[j].elapsed_time(events[j + 1]) / (bounds[j + 1] - bounds[j]) for j in range(len(bounds) - 1)])
    meds = [statistics.median(b) for b in blocks]
    stats = (statistics.median(meds), min(meds), max(meds), total_m, ref_ok)
    gathered = [None] * W
    dist.all_gather_object(gathered, stats)
    if rank == 0:
        m, lo, hi, _, _ = max(gathered, key=lambda t: t[0])  # slowest rank sets the collective
        mean = sum(g[0] for g in gathered) / W
        flops = 2 * total_m * H * 2 * I
        print(
            f"<flux moe_ag_scatter {args.model} | W={W} T_local={T_local} total_m={[g[3] for g in gathered]} "
            f"all-gather {(W - 1) * T_local * H * 2 / 1e6:.0f} MB per rank | sm_margin {args.sm_margin} | warm {WARM_S * 1e3:.0f} ms, "
            f"{REPS} x {WINDOW_S * 1e3:.0f} ms blocks>"
        )
        print(f"  flux: {m * 1e3:7.0f} us  [{lo * 1e3:.0f}-{hi * 1e3:.0f}]  {flops / m / 1e9:5.0f} TFLOP/s  rank-mean {mean * 1e3:.0f} us")
        if args.ref_check:
            print(f"  flux ref: max rel err {max(g[4] for g in gathered):.3e}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
