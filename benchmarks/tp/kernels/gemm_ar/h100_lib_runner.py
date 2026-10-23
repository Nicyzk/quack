"""GEMM+AllReduce baseline that only runs on H100 (sm90): Triton-distributed.

flux is NOT here — it ships AllGather+GEMM and GEMM+ReduceScatter but no fused
GEMM+AllReduce op. Triton-distributed's wheel lives in its own conda env
(/scratch/gpfs/TRIDAO/zy4586/envs/triton-dist-main), so gemm_ar/bench_baselines.py
shells out to this runner under that env's torchrun (run_h100_lib); it can also
be launched directly:

  /scratch/gpfs/TRIDAO/zy4586/envs/triton-dist-main/bin/torchrun --nproc_per_node=8 \
      -- h100_lib_runner.py triton_dist --m 8192 --n 8192 --k 8192

Direct invocation needs "--" before the script (torchrun abbrev-matches --m
otherwise); run_h100_lib sets LD_LIBRARY_PATH to the env's nvidia/nvshmem/lib.

Same shape semantics as bench_baselines: K is the GLOBAL contraction dim
(local_K = K/world), each rank computes (M, N) x (N, local_K)^T partials and
all-reduces to the full (M, N) on every rank; A,B ~ N(0, 1/sqrt(K)); per-rank
FLOPs = 2*M*N*local_K. cuBLAS+NCCL reference computes A @ B.T then all_reduce.
Timing-only (correctness is covered by the repo's own tests); graph-captured under
do_bench_l2_rotate_all like every other arm, against the reference on the same
rotation sets; a forward that does not capture is flagged, never eager-timed.
GemmARLayer is built the way triton_dist's own TP layer builds it (low-latency
fused kernel at M <= 256, result left in the symmetric buffer) with NUM_COMM_SMS
swept; the best setting is reported.
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
    p.add_argument("lib", choices=["triton_dist"])
    p.add_argument("--m", type=int, required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--k", type=int, required=True)
    args = p.parse_args()
    M, N, K = args.m, args.n, args.k
    cap = torch.cuda.get_device_capability()
    if cap != (9, 0):
        sys.exit(f"{args.lib} baseline is H100 (sm90)-only; this GPU is sm{cap[0]}{cap[1]}")

    from triton_dist.utils import initialize_distributed, finalize_distributed
    from triton_dist.layers.nvidia import GemmARLayer

    tp_group = initialize_distributed()
    rank, world = dist.get_rank(), dist.get_world_size()
    assert M % world == 0 and K % world == 0
    local_K = K // world
    torch.manual_seed(1111 + rank)  # bench_baselines' seed
    std = 1.0 / math.sqrt(K)
    A = torch.randn(M, local_K, dtype=torch.bfloat16, device="cuda") * std  # [M, local_K]
    B = torch.randn(N, local_K, dtype=torch.bfloat16, device="cuda") * std  # [N, local_K]

    # GemmARLayer(tp_group, max_M, N, K, in_dtype, out_dtype); forward([M,local_K],[N,local_K])->[M,N].
    # Built as their own TP layer builds it (triton_dist/layers/nvidia/tp_attn.py): the single
    # fused low-latency kernel at M <= 256, the result left in the symmetric buffer (no copy
    # out), NUM_COMM_SMS swept like PK's comm SMs (their picks: 16 at small M, 4 at large).
    use_ll = M <= 256
    comm_sms = (8, 16, 24, 32) if use_ll else (4, 8, 16)
    layers = {
        c: GemmARLayer(tp_group, M, N, K, torch.bfloat16, torch.bfloat16,
                       use_ll_kernel=use_ll, copy_to_local=False, NUM_COMM_SMS=c)
        for c in comm_sms
    }
    fn = lambda a, b, comm_sms=comm_sms[0]: layers[comm_sms].forward(a, b)  # noqa: E731

    n_sets = l2_rotate_count(A, B)
    arg_sets = [(A.clone(), B.clone()) for _ in range(n_sets)]
    kwarg_sets = [{}] * n_sets

    out = torch.empty(M, N, dtype=A.dtype, device="cuda")

    def ref_fn(a, b):
        torch.matmul(a, b.t(), out=out)
        dist.all_reduce(out)
        return out

    dist.barrier()
    # Every arm in the tables is graph-captured (the production decode regime; eager
    # timing would count each arm's launches instead): no capture, no number.
    if not try_capture(fn, arg_sets[0]):
        if rank == 0:
            print(f"<BF16 GEMM+AR {args.lib} | {M}x{K}x{N} | NOT GRAPH-CAPTURABLE: not timed>", flush=True)
        dist.barrier()
        for layer in layers.values():
            layer.finalize()
        finalize_distributed()
        sys.exit(2)
    sweep = {}
    for c in comm_sms:
        dist.barrier()
        sweep[c] = do_bench_l2_rotate_all(fn, arg_sets, kwarg_sets, {"comm_sms": c})
    best_c = min(sweep, key=sweep.get)
    ms = sweep[best_c]
    dist.barrier()
    ref_ms = do_bench_l2_rotate_all(ref_fn, arg_sets, kwarg_sets, {})

    if rank == 0:
        def tf(v):
            return 2 * M * N * local_K / (v * 1e9)

        kern = "ll" if use_ll else "persistent"
        print("=" * 80, flush=True)
        print(f"<BF16 GEMM+AR {args.lib} | world={world} | {M}x{K}x{N} | L2-rotate>", flush=True)
        for c in comm_sms:
            print(f"  {args.lib} {kern} comm_sms={c}: {sweep[c]:.4f} ms | {tf(sweep[c]):.2f} TFLOp/s", flush=True)
        print(f"  {args.lib + ':':<14} {ms:.4f} ms | {tf(ms):.2f} TFLOp/s  (best: {kern} comm_sms={best_c})",
              flush=True)
        print(f"  {'cuBLAS+NCCL:':<14} {ref_ms:.4f} ms | {tf(ref_ms):.2f} TFLOp/s", flush=True)

    dist.barrier()
    for layer in layers.values():
        layer.finalize()
    finalize_distributed()


if __name__ == "__main__":
    main()
