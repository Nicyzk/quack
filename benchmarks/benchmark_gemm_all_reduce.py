# Copyright (c) 2026, QuACK team.
"""GEMM + all-reduce benchmark: the AllReduce operators (quack/distributed/reduce_scatter.py two-shot,
all_reduce_lamport.py one-shot) vs the cuBLAS + NCCL chain, on the paper's decode sites: M = 64 tokens, K
sharded K/world_size per rank, every rank ends with the full (64, N).

Methods:
  quack_ar_2shot  gemm() with dist_op=AllReduce: each rank reduces one stripe of every tile and commits it into
                  every rank's D by multimem store (world_size <= 4 on the 64-row tile: a 16-row stripe floor)
  quack_ar_1shot  gemm() with dist_op=AllReduceLamport: the data is the flag, every rank reduces whole tiles
  nccl_seq        torch matmul then dist.all_reduce (the reference, timed last as the contention canary)

Every quack method on the 64x64 one-CTA tile (the paper's decode tile; quack's tuner never proposes tile_M 64),
swizzle 1 / 8, the best reported. Protocol = quack's autotuner methodology made collective
(quack.bench.bench_utils.bench_l2_rotate_dist): cloned operand sets that turn L2 over, one captured 200-call
window warmed by replaying itself, cross-rank max, --rounds rounds.

  torchrun --nproc_per_node=8 benchmarks/benchmark_gemm_all_reduce.py
"""

import argparse
import math
import os

import torch
import torch.distributed as dist

from quack.bench.bench_utils import bench_l2_rotate_dist, l2_rotate_arg_sets_dist
from quack.distributed.all_reduce_lamport import AllReduceOneShotRunner
from quack.distributed.reduce_scatter import AllReduceRunner
from quack.gemm import gemm as quack_gemm

DEFAULT_SHAPES = [
    (4096, 4096),  # Llama 8B out
    (4096, 14336),  # Llama 8B down
    (8192, 8192),  # Llama 70B out
    (8192, 28672),  # Llama 70B down
    (16384, 16384),  # Llama 405B out
    (16384, 53248),  # Llama 405B down
]  # (N, K) global
TILE = 64  # the decode tile: 64x64, one CTA
SWIZZLES = (1, 8)


def _print0(rank, *args):
    if rank == 0:
        print(*args, flush=True)


def _check(rank, name, d, ref, tol) -> bool:
    err = (d.float() - ref).abs().max().item()
    flag = torch.tensor([1.0 if err > tol else 0.0], device=d.device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    if flag.item() > 0:
        _print0(
            rank,
            f"  [{name}] SKIP: correctness check failed (err={err:.3e} "
            f"tol={tol:.3e}) — not timing a wrong result",
        )
        return False
    _print0(rank, f"  [{name}] correctness OK (err={err:.2e})")
    return True


def bench_shape(m, n, k, dtype, rank, world_size, device, args):
    k_local = k // world_size
    torch.manual_seed(1000 + rank)
    a = torch.randn(m, k_local, dtype=dtype, device=device) / math.sqrt(k)
    b = torch.randn(n, k_local, dtype=dtype, device=device) / math.sqrt(k)
    sets = l2_rotate_arg_sets_dist(a, b)

    # fp32 reference for loose correctness checks: the full-K product
    ref = a.float() @ b.float().T
    dist.all_reduce(ref)
    tol = ref.abs().max().item() * 5e-2 + 1e-2
    _print0(rank, f"M={m} N={n} K={k}")
    results = {}

    def best_of(name, runner):
        """Record (ms, swizzle) of the best round at the best correct swizzle."""
        best = None
        for sw in SWIZZLES:

            def run(x, y, sw=sw):
                quack_gemm(
                    x,
                    y,
                    runner.d,
                    None,
                    None,
                    TILE,
                    TILE,
                    1,
                    1,
                    max_swizzle_size=sw,
                    dist_op=runner.op,
                    dist_op_args=runner.prepare(),
                )

            run(a, b)
            torch.cuda.synchronize(device)
            if not _check(rank, f"{name} sw{sw}", runner.d, ref, tol):
                continue
            ms = min(bench_l2_rotate_dist(run, sets) for _ in range(args.rounds))
            if best is None or ms < best[0]:
                best = (ms, f"sw{sw}")
        if best is not None:
            results[name] = best

    if (
        world_size <= 4
    ):  # the 64-row tile's stripes are 64 / world_size rows; the comm frame needs 16
        best_of("quack_ar_2shot", AllReduceRunner(m, n, dtype, TILE, TILE))
    best_of("quack_ar_1shot", AllReduceOneShotRunner(m, n, dtype, TILE))
    out = torch.empty(m, n, dtype=dtype, device=device)

    def run_nccl_seq(x, y):  # the reference, last
        torch.matmul(x, y.T, out=out)
        dist.all_reduce(out)

    results["nccl_seq"] = (
        min(bench_l2_rotate_dist(run_nccl_seq, sets) for _ in range(args.rounds)),
        "",
    )
    for name, (ms, cfg) in results.items():
        _print0(rank, f"  {name:<15} {ms * 1e3:8.1f} us  {cfg}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shapes",
        type=str,
        default=";".join(",".join(map(str, s)) for s in DEFAULT_SHAPES),
        help="N,K;... (global)",
    )
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank % torch.cuda.device_count())
    device = torch.device("cuda", rank % torch.cuda.device_count())
    dist.init_process_group("nccl", device_id=device)
    dtype = getattr(torch, args.dtype)
    if rank == 0:
        print(f"GEMM+AR world_size={world_size} {args.dtype}, 64x64 one-CTA tile, best round")
    for shape in args.shapes.split(";"):
        n, k = map(int, shape.split(","))
        bench_shape(args.m, n, k, dtype, rank, world_size, device, args)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
