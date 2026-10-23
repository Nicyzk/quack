"""Reduce-scatter harness + benchmark.

KERNELS maps a name to a factory reduce_scatter(dtype, out, inp, rank, world) -> fn(), a nullary
callable baked to the fixed symmetric buffers (graph replay or plain launcher — the kernel's
choice) that the bench, the gate, and a PKB solution all consume identically. rs_io allocates
the standard symmetric buffers; busbw is the shared metric. test_rs.py imports KERNELS/rs_io.

Usage: torchrun --nproc_per_node=<world> bench_rs.py <kernel> [--m M --n N]  # simple | multimem | tma
"""
import argparse
import os
import sys

import torch
import torch.distributed as dist
import cutlass

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # package-qualified: hosts own bare `utils`
from kernels.misc.utils import (
    torchrun_init_nvshmem as init_nvshmem,
    torchrun_finalize_nvshmem as finalize_nvshmem,
    do_bench_all,
    clean_print,
)
import nvshmem.core

from kernels.rs.reduce_scatter_simple import reduce_scatter as _simple
from kernels.rs.reduce_scatter_multimem import reduce_scatter as _multimem
from kernels.rs.reduce_scatter_tma import reduce_scatter as _tma


def _nccl(dtype, out, inp, rank, world):
    def rs():
        dist.reduce_scatter_tensor(out, inp)
    return rs


KERNELS = {"simple": _simple, "multimem": _multimem, "tma": _tma, "nccl": _nccl}


def rs_io(M, N, world, dtype=torch.bfloat16):
    """Standard symmetric reduce-scatter buffers: inp (M, N), out (M // world, N)."""
    inp = nvshmem.core.tensor((M, N), dtype=dtype)
    out = nvshmem.core.tensor((M // world, N), dtype=dtype)
    return inp, out


def busbw(ms, M, N, world):
    """Reduce-scatter bus bandwidth (GB/s) from per-call time in ms."""
    return 2 * M * N * (world - 1) / world / (ms * 1e-3) / 1e9


def main():
    p = argparse.ArgumentParser()
    p.add_argument("kernel", choices=[k for k in KERNELS if k != "nccl"])
    p.add_argument("--m", type=int, default=4096)
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--k", type=int, default=None, help="ignored; accepted for GEMM+RS interface parity")
    p.add_argument("--skip_ref_check", action="store_true")
    args = p.parse_args()

    init_nvshmem()
    rank, world = dist.get_rank(), dist.get_world_size()
    M, N = args.m, args.n
    assert M % world == 0
    inp, out = rs_io(M, N, world)
    torch.manual_seed(rank)  # rank-dependent data so peer mixups can't cancel
    inp.normal_()

    rs = KERNELS[args.kernel](cutlass.BFloat16, out, inp, rank, world)
    ncc = KERNELS["nccl"](cutlass.BFloat16, out, inp, rank, world)

    if not args.skip_ref_check:
        ref = torch.empty_like(out)
        dist.reduce_scatter_tensor(ref, inp)
        rs()
        torch.cuda.synchronize()
        # f32-acc vs NCCL stepwise-bf16 differ by ~2 ULP of the summands; real bugs are O(1)
        torch.testing.assert_close(out.float(), ref.float(), atol=0.25, rtol=1e-2)
        clean_print("Ref check PASSED", print_once=True)

    dist.barrier(device_ids=[rank])
    gen_ms = do_bench_all(rs, warmup=10, rep=30)  # factory-owned graph replay; NCCL eager
    dist.barrier(device_ids=[rank])
    ncc_ms = do_bench_all(ncc, warmup=10, rep=30)

    clean_print(f"<BF16 RS | {args.kernel} | world={world} | {M}x{N}>", print_once=True)
    clean_print(f"{args.kernel:>8}: {gen_ms * 1e3:8.1f} us | busBW={busbw(gen_ms, M, N, world):6.1f} GB/s", print_once=True)
    clean_print(f"    NCCL: {ncc_ms * 1e3:8.1f} us | busBW={busbw(ncc_ms, M, N, world):6.1f} GB/s", print_once=True)

    nvshmem.core.free_tensor(inp)
    nvshmem.core.free_tensor(out)
    finalize_nvshmem()


if __name__ == "__main__":
    main()
