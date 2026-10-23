"""Comet (ByteDance flux) MoE GEMM2 + top-k reduce + scatter (GemmGroupedV3GatherRS, nvshmem), the fused GEMM2 +
combine baseline, under flux's env (torch 2.4.1) as its own torchrun, like h100_flux_moe_runner.py:

  LD_PRELOAD=/scratch/gpfs/TRIDAO/zy4586/envs/ipc_attr_shim.so NVSHMEM_BOOTSTRAP=UID \\
  /scratch/gpfs/TRIDAO/zy4586/envs/flux/bin/torchrun --nproc_per_node=4 --standalone -- h100_flux_combine_runner.py --t-local 8192

Same shape and routing as bench_fused_gemm_combine.py (seed 123 + rank, torch.rand top-k, fp32 softmax scores over
the token's K logits). tp = 1, ep = W: this rank's GEMM2 input is its experts' rows in the global expert-sorted order
(flux's routing_idx = the stable scatter index of every (token, k) pair), weights (E_local, H, I), the router score
of each row as output_vec_scale; the op multiplies, top-k reduces and scatters the rows to their token's rank, so its
output is this rank's (T_local, H) combined tokens and its time includes the reduce: compare with our fused + reduce.
Timing mirrors bench_dispatch.bench_all with a one-element NCCL all-reduce as the per-call sync. --ref_check compares
the output against torch (per-expert GEMM, scale, sum over k, all-reduce, this rank's slice).
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
    logits = torch.rand(T_local, E, generator=g, device=dev)
    scores, topk_idx = logits.topk(topk, dim=-1)
    scores = torch.softmax(scores, dim=-1)  # fp32 over the token's K logits, as sonic's router
    topk_idx = topk_idx.to(torch.int32)
    w2 = torch.randn(E_local, H, I, generator=g, device=dev, dtype=torch.bfloat16) * 0.02  # (E_local, N, K)

    choosed = torch.empty(W, T_local, topk, dtype=torch.int32, device=dev)
    dist.all_gather_into_tensor(choosed, topk_idx)
    scores_g = torch.empty(W, T_local, topk, dtype=torch.float32, device=dev)
    dist.all_gather_into_tensor(scores_g, scores)
    flat = choosed.reshape(-1)
    splits = torch.bincount(flat, minlength=E).to(torch.int32)
    order = torch.argsort(flat, stable=True)  # expert-sorted (token, k) pairs
    routing_idx = torch.empty(flat.numel(), dtype=torch.int32, device=dev)
    routing_idx[order] = torch.arange(flat.numel(), dtype=torch.int32, device=dev)
    first = rank * E_local
    m_start, total_m = int(splits[:first].sum()), int(splits[first : first + E_local].sum())
    mine = order[m_start : m_start + total_m]  # my experts' rows: global flat (token, k) ids
    a = torch.randn(total_m, I, generator=g, device=dev, dtype=torch.bfloat16)  # GEMM2 input rows
    scale = scores_g.reshape(-1)[mine].contiguous()

    flux.init_flux_shm(dist.group.WORLD)
    op = flux.GemmGroupedV3GatherRS(E, W * T_local * topk, H, topk, rank, W, 1, W)  # tp 1, ep W

    def run():
        return op.forward_gather_rs(
            a, w2, splits.cpu(), routing_idx, output_vec_scale=scale, fast_accum=False, sm_margin=args.sm_margin
        )

    ref_err = None
    if args.ref_check:
        out = run()
        torch.cuda.synchronize()
        ref = torch.zeros(W * T_local, H, dtype=torch.float32, device=dev)  # per token, not per (token, k) pair
        s = 0
        for e in range(E_local):
            n = int(splits[first + e])
            ref.index_add_(0, mine[s : s + n] // topk, (a[s : s + n] @ w2[e].T).float() * scale[s : s + n, None])
            s += n
        dist.all_reduce(ref)
        ref = ref[rank * T_local : (rank + 1) * T_local]
        ref_err = ((out.float() - ref).abs().max() / (ref.abs().max() + 1e-6)).item()

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
    gathered = [None] * W
    dist.all_gather_object(gathered, (statistics.median(meds), min(meds), max(meds), total_m, ref_err))
    if rank == 0:
        m, lo, hi, _, _ = max(gathered, key=lambda t: t[0])  # slowest rank sets the collective
        mean = sum(g[0] for g in gathered) / W
        flops = 2 * total_m * H * I
        print(
            f"<flux gather_rs (GEMM2 + top-k reduce + scatter) {args.model} | W={W} T_local={T_local} "
            f"total_m={[g[3] for g in gathered]} | sm_margin {args.sm_margin} | warm {WARM_S * 1e3:.0f} ms, {REPS} x {WINDOW_S * 1e3:.0f} ms blocks>"
        )
        print(f"  flux: {m * 1e3:7.0f} us  [{lo * 1e3:.0f}-{hi * 1e3:.0f}]  {flops / m / 1e9:5.0f} TFLOP/s  rank-mean {mean * 1e3:.0f} us")
        if args.ref_check:
            print(f"  flux ref: max rel err {max(g[4] for g in gathered):.3e}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
