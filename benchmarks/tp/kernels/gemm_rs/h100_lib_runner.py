"""GEMM+RS baselines that only run on H100 (sm90): ByteDance flux and
Triton-distributed. Their wheels live in their own conda envs
(/scratch/gpfs/TRIDAO/zy4586/envs/{flux,triton-dist-main}), so
bench_baselines.py shells out to this runner under the matching env's
torchrun (run_h100_lib); it can also be launched directly:

  /scratch/gpfs/TRIDAO/zy4586/envs/flux/bin/torchrun --nproc_per_node=4 \
      h100_lib_runner.py flux --m 8192 --n 7168 --k 16384
  /scratch/gpfs/TRIDAO/zy4586/envs/triton-dist-main/bin/torchrun --nproc_per_node=4 \
      h100_lib_runner.py triton_dist --m 8192 --n 7168 --k 16384

Direct invocation needs "--" before the script (torchrun abbrev-matches --m
otherwise) and the env run_h100_lib sets: flux LD_PRELOAD=envs/ipc_attr_shim.so
(torch's from_blob rejects IPC peer buffers), triton_dist LD_LIBRARY_PATH to
the env's nvidia/nvshmem/lib.

Same shape semantics as bench_baselines: K is the GLOBAL contraction dim
(local_K = K/world), each rank computes (M, N) x (N, local_K)^T partials and
reduce-scatters to an (M/world, N) slab; A,B ~ N(0, 1/sqrt(K));
per-rank FLOPs = 2*M*N*local_K. Both libraries take the weight as (N,
local_K) row-major, so the cuBLAS+NCCL reference computes A @ W.T then
reduce_scatter_tensor. Timing-only, like the 82 arm (correctness is covered
by each repo's own tests); timed with do_bench_l2_rotate_all in eager mode
(neither library's forward is proven CUDA-graph-capturable) against the
reference on the same rotation sets.
"""
import argparse
import math
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "misc"))
from bench_rotate import do_bench_l2_rotate_all, l2_rotate_count, try_capture  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("lib", choices=["flux", "triton_dist"])
    p.add_argument("--m", type=int, required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--k", type=int, required=True)
    args = p.parse_args()
    M, N, K = args.m, args.n, args.k

    cap = torch.cuda.get_device_capability()
    if cap != (9, 0):
        sys.exit(f"{args.lib} baseline is H100 (sm90)-only; this GPU is sm{cap[0]}{cap[1]}")

    if args.lib == "flux":
        dist.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        tp_group = dist.new_group(list(range(dist.get_world_size())), backend="nccl")
    else:
        from triton_dist.utils import initialize_distributed

        tp_group = initialize_distributed()

    rank, world = dist.get_rank(), dist.get_world_size()
    assert M % world == 0 and K % world == 0
    local_K = K // world
    torch.manual_seed(1111 + rank)  # bench_baselines' seed
    std = 1.0 / math.sqrt(K)
    A = (torch.randn(M, local_K, dtype=torch.bfloat16, device="cuda") * std)
    B = (torch.randn(N, local_K, dtype=torch.bfloat16, device="cuda") * std)

    if args.lib == "flux":
        import flux

        flux.init_flux_shm(tp_group)
        op = flux.GemmRS(tp_group, 1, M, N, A.dtype, A.dtype)
        fn = lambda a, b: op.forward(a, b)  # noqa: E731
        ctx = None
        configs = [{}]
    else:
        from triton_dist.kernels.nvidia import create_gemm_rs_context, gemm_rs

        rs_stream = torch.cuda.Stream(priority=-1)
        ctx = create_gemm_rs_context(M, N, rank, world, world, A.dtype, rs_stream)
        # gemm_rs takes B as [K_local, N] (a view of our (N, K_local) weight, as their test); its
        # autotune space = GEMM config x fuse_scatter x persistent; both settings timed, best reported
        fn = lambda a, b, autotune=False: gemm_rs(a, b.t(), ctx, autotune=autotune)  # noqa: E731
        configs = [{"autotune": False}, {"autotune": True}]

    n_sets = l2_rotate_count(A, B)
    arg_sets = [(A.clone(), B.clone()) for _ in range(n_sets)]
    kwarg_sets = [{}] * n_sets

    out = torch.empty(M // world, N, dtype=A.dtype, device="cuda")

    def ref_fn(a, b):
        full = torch.matmul(a, b.t())
        dist.reduce_scatter_tensor(out, full)
        return out

    # Graph-captured when the library's forward captures; GEMM+RS at these sizes is
    # throughput-bound, so a non-capturable library is timed eager and labeled (the strict
    # no-eager rule belongs to the latency-bound GEMM+AR table).
    sweep = {}
    for cfg in configs:
        dist.barrier()
        label = " ".join(f"{k}={v}" for k, v in cfg.items()) or "default"
        # flux's forward drives its own streams unjoined ("capturing stream has unjoined
        # work"); probing it under its torch 2.4 leaves the allocator in a failed-capture
        # state, so it is not probed
        captured = args.lib != "flux" and try_capture(fn, arg_sets[0], cfg)
        if not captured:
            label += " (eager-timed: not graph-capturable)"
        sweep[label] = do_bench_l2_rotate_all(fn, arg_sets, kwarg_sets, cfg, capture=captured)
    best = min(sweep, key=sweep.get)
    ms = sweep[best]
    dist.barrier()
    ref_ms = do_bench_l2_rotate_all(ref_fn, arg_sets, kwarg_sets, {})

    if rank == 0:
        def tf(v):
            return 2 * M * N * local_K / (v * 1e9)

        print("=" * 80, flush=True)
        print(f"<BF16 GEMM+RS {args.lib} | world={world} | {M}x{K}x{N} | L2-rotate>", flush=True)
        for label, v in sweep.items():
            print(f"  {args.lib} {label}: {v:.3f} ms | {tf(v):.2f} TFLOp/s", flush=True)
        print(f"  {args.lib + ':':<14} {ms:.3f} ms | {tf(ms):.2f} TFLOp/s  (best: {best})", flush=True)
        print(f"  {'cuBLAS+NCCL:':<14} {ref_ms:.3f} ms | {tf(ref_ms):.2f} TFLOp/s", flush=True)

    dist.barrier()
    if args.lib == "triton_dist":
        from triton_dist.utils import finalize_distributed

        ctx.finalize()
        finalize_distributed()
    else:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
