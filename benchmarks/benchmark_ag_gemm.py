"""
Rotating all-gather + dense GEMM benchmark using quack.gemm.gemm().

A is sharded by rows (M) across TP ranks; B is replicated. Each rank computes the
full (M, N) output in world_size steps, rotating through shards (rank+j)%world_size.
Remote A-shards are copied from NVSHMEM symmetric memory on copy_stream; the gemm
waits in-kernel via quack's load_a_flag until copy_stream signals that shard.

Usage:
    torchrun --nproc_per_node=8 benchmarks/benchmark_ag_gemm.py \
        --mnkl 8192,8192,8192,1 --tile_shape_mnk 256,256 --cluster_shape_mnk 2,1
"""

import math
import os

import numpy as np
import torch
import torch.distributed as dist
import nvshmem.core
from cuda.bindings import driver
from cuda.core.experimental import Device
import triton.runtime as triton_runtime
from triton.testing import _summarize_statistics

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.runtime import from_dlpack

from benchmark_gemm import _torch_dtype, parse_arguments
from quack.gemm import gemm as quack_gemm


# ---- distributed bench helpers ------------------------------------------------------


def clean_print(*args, **kwargs):
    # No collectives -> can't deadlock; ranks just print (output may interleave).
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    print_once = kwargs.pop("print_once", False)
    if print_once and local_rank != 0:
        return
    kwargs.setdefault("flush", True)
    prefix = () if print_once else (f"[Rank {local_rank}]",)
    print(*prefix, *args, **kwargs)


def torchrun_init_nvshmem():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dev = Device(local_rank)
    dev.set_current()
    dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    num_ranks = dist.get_world_size()
    uid = nvshmem.core.get_unique_id(empty=(local_rank != 0))
    uid_tensor = torch.from_numpy(uid._data.view(np.uint8).copy()).cuda()
    dist.broadcast(uid_tensor, src=0)
    dist.barrier()
    uid._data[:] = uid_tensor.cpu().numpy().view(uid._data.dtype)
    nvshmem.core.init(
        device=dev, uid=uid, rank=local_rank, nranks=num_ranks, initializer_method="uid"
    )


def torchrun_finalize_nvshmem():
    nvshmem.core.finalize()
    dist.destroy_process_group()


def do_bench_all(fn, warmup=25, rep=100, grad_to_none=None, quantiles=None, return_mode="mean"):
    """Adapted from Triton do_bench. Issues with do_bench for multi-GPU scenarios:
      (1) Triton interprets warmup/rep as ms, and derives the count per-rank from a local estimate,
          so under skew ranks can disagree on the estimate and a collective fn() deadlocks.
      (2) Using fixed warmup/rep counts across ranks can cause throttling for longer vs shorter kernels.
    Solution: interpret warmup/rep as ms, derive the count per-rank from a local estimate,
    then sync the counts across ranks."""
    assert return_mode in ["min", "max", "mean", "median", "all"]

    di = triton_runtime.driver.active.get_device_interface()

    fn()
    di.synchronize()

    cache = triton_runtime.driver.active.get_empty_cache_for_benchmark()

    # Estimate the runtime of the function
    start_event = di.Event(enable_timing=True)
    end_event = di.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        triton_runtime.driver.active.clear_cache(cache)
        fn()
    end_event.record()
    di.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5

    # compute number of warmup and repeat
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))

    # sync counts across ranks so a collective fn() can't diverge and deadlock
    if dist.is_initialized():
        c = torch.tensor([n_warmup, n_repeat], device="cuda", dtype=torch.int64)
        dist.all_reduce(c, op=dist.ReduceOp.MAX)
        n_warmup, n_repeat = int(c[0].item()), int(c[1].item())

    start_event = [di.Event(enable_timing=True) for i in range(n_repeat)]
    end_event = [di.Event(enable_timing=True) for i in range(n_repeat)]
    # Warm-up
    for _ in range(n_warmup):
        fn()
    # Benchmark
    for i in range(n_repeat):
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        # we clear the L2 cache before each run
        triton_runtime.driver.active.clear_cache(cache)
        # record time of `fn`
        start_event[i].record()
        fn()
        end_event[i].record()
    # Record clocks
    di.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_event, end_event)]
    return _summarize_statistics(times, quantiles, return_mode)


# ---- cross-GPU barrier ---------------------------------------------------------------
# Cross-GPU barrier kernel: multicast release-increment, spin until the local
# flag reaches world_size (>= wait), then atomically consume world_size tokens.
#
# Reuse-safe with a single instance: release needs world_size*phase cumulative
# tokens, a phase p+1 token cannot exist before every rank issued phase p, and
# early tokens bank in the counter. (The old CAS==world_size reset wedged when
# a fast rank's next-phase +1 landed before a slow rank's CAS fired.)
#
# No acquire fence: the rendezvous orders execution, and cross-GPU visibility
# comes from the grid boundary (producer kernel commits to L2 before the
# barrier) plus fresh uncached peer reads -- verified by the race tests.


class BarrierAllKernel:
    @cute.jit
    def __call__(self, flag, flag_mc, world_size: cutlass.Constexpr, stream):
        self.kernel(flag, flag_mc, world_size).launch(
            grid=[1, 1, 1], block=[32, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(self, flag: cute.Tensor, flag_mc: cute.Tensor, world_size: cutlass.Constexpr):
        tidx = cute.arch.thread_idx()[0]
        if tidx == 0:
            utils.distributed.multimem_red_add1(flag_mc.iterator, scope="sys", order="release")
            utils.distributed.spin_lock_ld_lt_relaxed_wait(
                flag.iterator,
                expected_val=world_size,
                scope="sys",
            )
            cute.arch.atomic_add(
                flag.iterator.llvm_ptr, cutlass.Int32(-world_size), sem="relaxed", scope="sys"
            )


def make_barrier(world_size):
    """Allocate a flag, compile the kernel; returns (barrier_fn, free_fn)."""
    flag = nvshmem.core.tensor((1,), dtype=torch.int32)
    flag.fill_(0)
    flag_mc = nvshmem.core.get_multicast_tensor(nvshmem.core.Teams.TEAM_NODE, flag)
    fd = from_dlpack(flag, enable_tvm_ffi=True)
    fmd = from_dlpack(flag_mc, enable_tvm_ffi=True)
    # TVM-FFI env stream (quack style): launches on the current stream, captured into the graph.
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    fn = cute.compile(BarrierAllKernel(), fd, fmd, world_size, stream, options="--enable-tvm-ffi")

    def free():
        nvshmem.core.free_tensor(flag_mc)
        nvshmem.core.free_tensor(flag)

    return lambda: fn(flag, flag_mc), free


def make_peer_tensor(shape, dtype):
    """Symmetric tensor + per-rank peer views (no multicast)."""
    t = nvshmem.core.tensor(shape, dtype=dtype)
    peers = [nvshmem.core.get_peer_tensor(t, r) for r in range(dist.get_world_size())]
    return t, peers


# ---- benchmark -----------------------------------------------------------------------


def run(args):
    m, n, k, l = args.mnkl
    tile_M, tile_N = args.tile_shape_mnk[:2]
    tile_K = args.tile_shape_mnk[2] if len(args.tile_shape_mnk) == 3 else None
    cluster_M, cluster_N, cluster_K = args.cluster_shape_mnk
    warmup, repeats = args.warmup_iterations, args.iterations
    ab_dtype = _torch_dtype(args.ab_dtype) if args.ab_dtype is not None else torch.bfloat16
    d_dtype = _torch_dtype(args.d_dtype)

    from quack.cute_dsl_utils import get_device_capacity

    persistent = args.persistent
    if get_device_capacity(torch.device("cuda"))[0] in [10, 11]:
        persistent = True

    print("Running Dense GEMM with:")
    print(f"mnkl: {args.mnkl}")
    print(f"Tile Shape MNK: {args.tile_shape_mnk}, Cluster Shape MNK: {args.cluster_shape_mnk}")
    print(f"Warmup iterations: {warmup}")
    print(f"Iterations: {repeats}")
    print(f"Skip reference checking: {args.skip_ref_check}")

    torch.manual_seed(1111)
    device = "cuda"
    rank = int(os.environ["LOCAL_RANK"])
    local_rank = rank
    world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    assert m % world_size == 0, f"m={m} must be divisible by world_size={world_size}"
    m_per_rank = m // world_size

    # A-shard (l, m_per_rank, k) in symmetric memory; B (l, n, k) replicated.
    a_shard = torch.randn(l, m_per_rank, k, dtype=torch.bfloat16, device=device) / math.sqrt(k)
    a_symm, a_peers = make_peer_tensor((l, m_per_rank, k), torch.bfloat16)
    a_symm.copy_(a_shard)
    B = torch.randn(l, n, k, dtype=ab_dtype, device=device) / (k**0.5)

    a_local = [
        torch.empty(l, m_per_rank, k, dtype=torch.bfloat16, device=device)
        for _ in range(world_size)
    ]
    d_local = [
        torch.empty(l, m_per_rank, n, dtype=torch.bfloat16, device=device)
        for _ in range(world_size)
    ]
    a_local[rank].copy_(a_shard)  # local shard (j==0); remote shards arrive via copy

    # Per-shard gate flag: copy_stream signals, gemm waits. Local (intra-rank).
    gate_a_flags = torch.zeros(world_size, dtype=torch.int32, device=device)

    def fn(a, d):
        quack_gemm(
            a, B, d,
            C=None, tile_count_semaphore=None,
            tile_M=tile_M, tile_N=tile_N, tile_K=tile_K,
            cluster_M=cluster_M, cluster_N=cluster_N, cluster_K=cluster_K,
            persistent=persistent,
            max_swizzle_size=args.max_swizzle_size,
        )

    def fn_gated_a_load(a, d, flag):
        quack_gemm(
            a, B, d,
            C=None, tile_count_semaphore=None,
            tile_M=tile_M, tile_N=tile_N, tile_K=tile_K,
            cluster_M=cluster_M, cluster_N=cluster_N, cluster_K=cluster_K,
            persistent=persistent,
            max_swizzle_size=args.max_swizzle_size,
            load_a_flag=flag,
        )

    g = torch.cuda.CUDAGraph()
    copy_stream = torch.cuda.Stream(local_rank)
    gemm_stream = torch.cuda.Stream(local_rank)
    bar_a, free_a = make_barrier(world_size)
    bar_b, free_b = make_barrier(world_size)

    with torch.cuda.graph(g):
        capture_stream = torch.cuda.current_stream()
        # Reset gate flags each iter (non-consuming wait).
        driver.cuMemsetD32Async(
            gate_a_flags.data_ptr(), 0, world_size, capture_stream.cuda_stream
        )
        bar_a()
        gemm_stream.wait_stream(capture_stream)
        # quack_gemm uses the current stream, so put gemms on gemm_stream here.
        with torch.cuda.stream(gemm_stream):
            for j in range(world_size):
                offset = (rank + j) % world_size
                if offset == rank:
                    fn(a_local[offset], d_local[offset])
                else:
                    fn_gated_a_load(
                        a_local[offset], d_local[offset], gate_a_flags[offset : offset + 1]
                    )
        copy_stream.wait_stream(capture_stream)
        for j in range(world_size):
            offset = (rank + j) % world_size
            if offset != rank:
                nbytes = a_local[offset].numel() * a_local[offset].element_size()
                driver.cuMemcpyDtoDAsync(
                    a_local[offset].data_ptr(),
                    a_peers[offset].data_ptr(),
                    nbytes,
                    copy_stream.cuda_stream,
                )
                # signal: flag=1 after the copy completes
                driver.cuMemsetD32Async(
                    gate_a_flags[offset].data_ptr(), 1, 1, copy_stream.cuda_stream
                )
        capture_stream.wait_stream(copy_stream)
        capture_stream.wait_stream(gemm_stream)
        bar_b()

    # Warm up d_local, then validate vs all-gather + bmm.
    g.replay()
    torch.cuda.synchronize()
    dist.barrier()

    if not args.skip_ref_check:
        gathered = [torch.empty_like(a_shard) for _ in range(world_size)]
        dist.all_gather(gathered, a_shard)
        ref = torch.bmm(torch.cat(gathered, dim=1), B.mT)
        got = torch.cat([d_local[offset] for offset in range(world_size)], dim=1)
        torch.testing.assert_close(got.float(), ref.float(), atol=args.tolerance, rtol=1e-3)
        clean_print("Ref check PASSED", print_once=True)

    flops = 2 * m * n * k * l
    kernel_ms = do_bench_all(g.replay, warmup=warmup, rep=repeats)

    # cuBLAS + NCCL all-gather baseline.
    a_full = torch.empty(l, m, k, dtype=ab_dtype, device=device)
    d_full = torch.empty(l, m, n, dtype=ab_dtype, device=device)

    def cublas_ag_gemm():
        shards = [a_full[:, i * m_per_rank : (i + 1) * m_per_rank, :] for i in range(world_size)]
        dist.all_gather(shards, a_shard)
        torch.bmm(a_full, B.mT, out=d_full)

    dist.barrier()
    baseline_ms = do_bench_all(cublas_ag_gemm, warmup=warmup, rep=repeats)

    clean_print(f"<AG+GEMM | world_size={world_size} | {m}x{k}x{n}>", print_once=True)
    clean_print(
        f"  rotating quack: {kernel_ms:.3f} ms | {flops / (kernel_ms * 1e9):.2f} TFLOp/s",
        print_once=True,
    )
    clean_print(
        f"  cuBLAS+NCCL AG: {baseline_ms:.3f} ms | {flops / (baseline_ms * 1e9):.2f} TFLOp/s",
        print_once=True,
    )

    free_a()
    free_b()
    for r in range(world_size):
        if r != rank:
            nvshmem.core.free_tensor(a_peers[r])
    nvshmem.core.free_tensor(a_symm)


if __name__ == "__main__":
    args = parse_arguments()
    torchrun_init_nvshmem()
    run(args)
    clean_print("PASS", print_once=True)
    torchrun_finalize_nvshmem()
