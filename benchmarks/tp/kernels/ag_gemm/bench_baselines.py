"""Run external-library AllGather+GEMM baselines through one interface.

These are NOT our kernels -- they are third-party / upstream implementations we
benchmark against (sibling of gemm_rs/bench_baselines.py, same protocol). Each
is registered in BASELINES with a tiny spec describing how it must be launched:

  quack      AllGatherRunner (reverse-ring CE push) + flag-gated persistent GEMM
             ~/quack/quack/distributed/all_gather_gemm.py  (branch intra-sm)
  tk         ThunderKittens all_gather_matmul
             ~/ThunderKittens/kernels/parallel/ag_gemm  (NCCL init; needs GPU env)
  cutlass    cutlass distributed_all_gather_gemm_blackwell (gated GEMMs + CE pull)
             ~/cutlass/examples/python/CuTeDSL/cute/blackwell/kernel/distributed  (torch symm_mem)
  asynctp    torch.ops.symm_mem.fused_all_gather_matmul (async-TP decomposition)
  82         cutlass 82_blackwell_distributed_gemm -- prebuilt C++ binary
             ~/tmp/cutlass-82 branch 82-tp4  (subprocess; self-manages TP=4)
  flux       ByteDance flux AGKernel -- H100 (sm90)-only, own conda env
             (subprocess to h100_lib_runner.py under that env's torchrun)
  triton_dist Triton-distributed ag_gemm -- H100 (sm90)-only, own conda env
             (subprocess, same runner)

In-process arms run under torchrun and are timed against an NCCL all-gather +
cuBLAS GEMM reference. 82 self-manages its GPUs and prints its own numbers, so
we just shell out to it (bf16 + randn patches; timing-only, no ref/r2r).

M is the gathered dim, sharded across ranks (shard_m = M/world); N is the
rank-LOCAL free dim (column-parallel weight shard); K is the full contraction
dim. A,B ~ N(0, 1/sqrt(K)). Per-rank FLOPs = 2*M*N*K; every rank's output is
the full (M, N).

Consistency models differ (same trap as quack's benchmark_all_gather_gemm):
quack stages the local shard into its symmetric buffer EVERY call (the
per-iteration producer dependency); tk / cutlass / 82 stage it once at setup,
so only their cross-rank transport is in the timed loop.

Each in-process factory returns a Kernel: run/free plus the rank's A M-shard
and B, and read_out/zero handles, so --ref_check / --r2r_check work generically
for any kernel (see ref_check/r2r_check). 82 is an external binary we only
time, so its --ref_check / --r2r_check are N/A.

Shapes come from --model / --site (utils.MODEL_CONFIGS, SITES): the site's global N is
split by the world size (n_local = N/TP), K is the model's hidden dim.

Usage:
    torchrun --nproc_per_node=8 bench_baselines.py quack     --model llama3_70b --site qkv --m 8192
    GPU=B300 torchrun --nproc_per_node=8 bench_baselines.py tk --model llama3_70b --site qkv --m 8192
    torchrun --nproc_per_node=8 bench_baselines.py cutlass --model llama3_70b --site qkv --m 8192 --ref_check
    bench_baselines.py 82 --model llama3_70b --site qkv --m 8192   # plain python; binary handles TP=8
"""
import argparse
import math
import os
import re
import subprocess
import sys
import time
from collections import namedtuple

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "misc"))
from bench_rotate import do_bench_l2_rotate_all, l2_rotate_arg_sets, l2_rotate_count  # noqa: E402
from utils import (  # noqa: E402
    MODEL_CONFIGS,
    SITES,
    clean_print,
    do_bench_counts,
    parse_quack_configs,
    quack_config_label,
    torchrun_init,
    torchrun_finalize,
    tune_quack_tile,
)

_NOOP = lambda: None  # noqa: E731
# quack config candidates (tile x swizzle x scheduler); main overrides from --quack_configs
quack_configs = parse_quack_configs()

# run/free: timed call + cleanup. A: the rank's M-shard (M//world, K). B: the rank's
# (N, K) weight shard. read_out: the full (M, N) output as fp32. zero: reset the output.
# sweep: optional [(label, run)] tuning variants -- each is timed, best reported.
# calls: iterations one run() performs (graph arms replay several); ms is divided by it.
# raw: the device-resident output run() writes (tensor, band list, or holder dict) --
# epilogue-composing benches (bench_llama_ag) read it without read_out's fp32 copy.
# fn / operands / arg_sets: the rotate protocol's per-set launch and what it rotates
# (cloned operands, or prebuilt per-set args for arms that own their operand buffers).
Kernel = namedtuple("Kernel", ["run", "free", "A", "B", "read_out", "zero", "sweep", "calls",
                               "raw", "fn", "operands", "arg_sets"],
                    defaults=[None, 1, None, None, None, None])


def _quack(dtype, M, N, K, rank, world):
    """quack overlapped AG+GEMM: AllGatherRunner (CE push) + flag-gated GEMM,
    zero-copy + CUDA-graph capture (benchmark_all_gather_gemm.py's honest
    static-buffer comparator: eager runs pay a 44-74 us/iter host-enqueue tax).
    Both double-buffer slots are prefilled at setup and gather() reads
    next_local_slot() -- staging out of the timed loop like the other arms.
    One graph captures GRAPH_CALLS calls (even count: the double-buffer parity
    advances per call and must be graph-invariant); run() replays it and
    Kernel.calls tells main to divide. Config = the first --quack_configs
    candidate (the rotate path sweeps them all); shard_m must be a multiple of
    tile_M*cluster_M."""
    from quack.distributed import AllGatherRunner
    from quack.gemm import gemm as quack_gemm

    shard_m = M // world
    A = torch.randn(shard_m, K, dtype=dtype, device="cuda") / math.sqrt(K)
    B = torch.randn(N, K, dtype=dtype, device="cuda") / math.sqrt(K)
    D = torch.empty(M, N, dtype=dtype, device="cuda")
    runner = AllGatherRunner(shard_m, K, dtype)
    tile_m, tile_n, cluster_m, cluster_n, swizzle, clc = quack_configs[0]

    def ag_gemm():
        with runner.gather(runner.next_local_slot()) as (a_full, ag_args):
            quack_gemm(a_full, B, D, None, None, tile_m, tile_n, cluster_m, cluster_n,
                       is_dynamic_persistent=clc, max_swizzle_size=swizzle, ag_args=ag_args)

    for _ in range(len(runner.bufs)):
        runner.next_local_slot().copy_(A)
        runner.last_parity ^= 1  # two flips = restored; prefills both slots
    torch.cuda.synchronize()
    for _ in range(3):  # compiles + symm rendezvous must precede capture
        ag_gemm()
    torch.cuda.synchronize()
    dist.barrier()
    GRAPH_CALLS = 4
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(GRAPH_CALLS):
            ag_gemm()

    def run(_keep=(runner, B, D)):
        g.replay()

    return Kernel(run, _NOOP, A, B, lambda: D.float(), D.zero_, None, GRAPH_CALLS, raw=D)


def _tk_import(world):
    """Load TK ag_gemm _C for this world size. NUM_DEVICES is compile-time in
    ag_gemm_b200.cu, so _C is per-world: _C_nd{2,4} are cross-built alongside the
    default _C (nd4). Falls back to _C if no per-world build exists."""
    import glob
    import importlib.util

    d = os.path.expanduser("~/ThunderKittens/kernels/parallel/ag_gemm")
    cand = glob.glob(os.path.join(d, f"_C_nd{world}.*.so")) or glob.glob(os.path.join(d, "_C.*.so"))
    assert cand, f"no TK ag_gemm _C for world={world}; build _C_nd{world} (NUM_DEVICES={world})"
    spec = importlib.util.spec_from_file_location("_C", cand[0])
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.TKParallelTensor, m.all_gather_matmul


def _tk(dtype, M, N, K, rank, world):
    """ThunderKittens all_gather_matmul. Local shard staged into the multicast
    A tensor at setup (TK's own benchmark methodology). num_comm_sms is swept
    over TK benchmark.py's grid; the best value is reported."""
    TKParallelTensor, all_gather_matmul = _tk_import(world)

    gpu = os.environ.get("GPU")
    assert gpu in ("B200", "B300", "H100"), "set GPU=B200|B300|H100 for tk"
    shard_m = M // world
    A = torch.randn(shard_m, K, dtype=torch.bfloat16, device="cuda") / math.sqrt(K)
    # H100 wants B as (K, N); Blackwell wants (N, K).
    b_shape = (K, N) if gpu == "H100" else (N, K)
    B = torch.randn(*b_shape, dtype=torch.bfloat16, device="cuda") / math.sqrt(K)

    A_tk = TKParallelTensor((M, K), dtype=torch.bfloat16,
                            local_rank=rank, local_world_size=world, multicast=True)
    A_tk.data_.zero_()
    A_tk.data_[rank * shard_m:(rank + 1) * shard_m].copy_(A)
    C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    # 2 MB is TK's minimum multicast object size
    barrier = TKParallelTensor((2, 1024, 1024), dtype=torch.int,
                               local_rank=rank, local_world_size=world, multicast=True)
    barrier.data_.zero_()
    def make_run(num_comm_sms):
        return lambda: all_gather_matmul(A_tk, B, C, barrier, num_comm_sms)

    # TK benchmark.py's sweep grid; kern.run (used by checks) fixed at 16
    sweep = [(f"num_comm_sms={v}", make_run(v)) for v in (2, 4, 8, 16, 32, 64)]
    B_shard = B if gpu != "H100" else B.T
    return Kernel(make_run(16), _NOOP, A, B_shard, lambda: C.float(), C.zero_, sweep, raw=C)


def _blackwell(dtype, M, N, K, rank, world, A=None, B=None, swizzle=1):
    """cutlass distributed_all_gather_gemm_blackwell (gated GEMMs + CE pull):
    per-source-rank A/C bands, one ungated + world-1 flag-gated GEMMs on a gemm
    stream, a copy stream CE-pulling peers' shards then setting flags, a leading
    device-barrier/flag-reset kernel; one iteration = one CUDA-graph replay.
    Local shard sits in the symm buffer from setup on (static-buffer model).

    Line-for-line mirror of upstream run() (same names/order, diff against it)
    with four deltas: (1) operands -- upstream gives every rank IDENTICAL
    small-int data (manual_seed(1111) + int fill), we use this bench's per-rank
    randn/sqrt(K); (2) torchrun's existing process group instead of its own
    init/destroy; (3) the tail -- graph replay handed to do_bench_counts/ref_check
    instead of its event loop and _compare; (4) raster swizzle -- the example
    fixes the scheduler's default (1), a subclass threads `swizzle` in. Config =
    the example's documented bench command: --mma_tiler_mn 256,128
    --cluster_shape_mn 2,1 --use_2cta_instrs --use_tma_store, acc fp32.
    A/B: optional (M//world, K)/(N, K) bf16 operand overrides."""
    sys.path.insert(0, os.path.expanduser(
        "~/cutlass/examples/python/CuTeDSL/cute/blackwell/kernel/distributed"))
    import cuda.bindings.driver as cuda
    from cuda.bindings import driver
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as utils
    import cutlass.torch as cutlass_torch
    import torch.distributed._symmetric_memory as symm_mem
    from cutlass.cute.runtime import from_dlpack
    from distributed_all_gather_gemm_blackwell import (
        PersistentDenseGemmKernel,
        ReleaseFlagSet,
        SyncNvlDevices,
    )

    acc_dtype = cutlass.Float32
    mma_tiler_mn, cluster_shape_mn = (256, 128), (2, 1)
    use_2cta_instrs = True
    use_tma_store = True
    ab_dtype = c_dtype = cutlass.BFloat16

    local_rank = rank  # module global in upstream's run(); single-node == rank
    world_size = world
    m, n, k, l = M, N, K, 1
    m_per_step = m // world_size
    if A is None:
        A = torch.randn(m_per_step, k, dtype=torch.bfloat16, device="cuda") / math.sqrt(k)
    if B is None:
        B = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") / math.sqrt(k)

    def _create_tensors(l, m_per_iteration, n, k, a_major, b_major, c_major,
                        ab_dtype, c_dtype, num_steps, local_rank):
        # delta: upstream builds these via manual_seed(1111)+cutlass_torch.matrix
        # (identical int data on every rank); we use the bench's randn operands
        a_torch_cpu = A.unsqueeze(-1)
        b_torch_cpu = B.unsqueeze(-1)
        c_torch_cpu = torch.zeros(m_per_iteration, n, 1, dtype=torch.bfloat16, device="cuda")

        a_local_list = []
        a_local_torch_list = []
        c_local_list = []
        c_local_torch_list = []
        for m_dim_offset in range(num_steps):
            a_temp = a_torch_cpu.clone()
            a_temp.zero_()
            if m_dim_offset == local_rank:
                a_temp.copy_(a_torch_cpu)
            a_local, a_local_torch = cutlass_torch.cute_tensor_like(
                a_temp, ab_dtype, is_dynamic_layout=True, assumed_align=16
            )
            a_local_list.append(a_local)
            a_local_torch_list.append(a_local_torch)
            c_temp = c_torch_cpu.clone().zero_()
            c_local, c_local_torch = cutlass_torch.cute_tensor_like(
                c_temp, c_dtype, is_dynamic_layout=True, assumed_align=16
            )
            c_local_list.append(c_local)
            c_local_torch_list.append(c_local_torch)
        a_torch_unique = torch.empty(
            a_torch_cpu.shape, device="cuda", dtype=a_torch_cpu.dtype
        )
        a_torch_unique.copy_(a_torch_cpu)
        a_tensor_symm = symm_mem.empty(
            a_torch_cpu.shape, device="cuda", dtype=a_torch_cpu.dtype
        )
        a_tensor_symm.copy_(a_local_torch_list[local_rank])
        a_tensor_symm_hdl = symm_mem.rendezvous(a_tensor_symm, dist.group.WORLD)
        a_tensor_symm_hdl.barrier(channel=0)

        b_tensor, b_torch = cutlass_torch.cute_tensor_like(
            b_torch_cpu, ab_dtype, is_dynamic_layout=True, assumed_align=16
        )

        gate_a_flags = torch.zeros(num_steps, device="cuda", dtype=torch.int32)

        device_arrival_counters = torch.tensor([0], dtype=torch.int32, device="cuda")
        device_arrival_counters_symm = symm_mem.empty(
            1, device="cuda", dtype=torch.int32
        )
        device_arrival_counters_symm.copy_(device_arrival_counters)
        device_arrival_counters_hdl = symm_mem.rendezvous(
            device_arrival_counters_symm, dist.group.WORLD
        )
        device_arrival_counters_hdl.barrier(channel=0)

        return (
            a_local_list,
            a_local_torch_list,
            a_torch_unique,
            a_tensor_symm_hdl,
            b_tensor,
            b_torch,
            c_local_list,
            c_local_torch_list,
            a_torch_cpu,
            b_torch_cpu,
            c_torch_cpu,
            gate_a_flags,
            device_arrival_counters_hdl,
        )

    torch_stream = torch.cuda.current_stream()
    current_stream = cuda.CUstream(torch_stream.cuda_stream)

    (
        a_local_list,
        a_local_torch_list,
        a_torch_unique,
        a_tensor_symm_hdl,
        b_tensor,
        b_torch,
        c_local_list,
        c_local_torch_list,
        a_torch_cpu,
        b_torch_cpu,
        c_torch_cpu,
        gate_a_flags,
        device_arrival_counters_hdl,
    ) = _create_tensors(
        l, m_per_step, n, k, "k", "k", "n", ab_dtype, c_dtype, world_size, rank
    )

    class SwizzledGemmKernel(PersistentDenseGemmKernel):
        # upstream's _compute_grid verbatim, plus the swizzle_size argument
        def _compute_grid(self, c, cta_tile_shape_mnk, cluster_shape_mn, max_active_clusters):
            c_shape = cute.slice_(cta_tile_shape_mnk, (None, None, 0))
            gc = cute.zipped_divide(c, tiler=c_shape)
            num_ctas_mnl = gc[(0, (None, None, None))].shape
            cluster_shape_mnl = (*cluster_shape_mn, 1)
            tile_sched_params = utils.PersistentTileSchedulerParams(
                num_ctas_mnl, cluster_shape_mnl, swizzle
            )
            grid = utils.StaticPersistentTileScheduler.get_grid_shape(
                tile_sched_params, max_active_clusters
            )
            return tile_sched_params, grid

    gemm = SwizzledGemmKernel(
        acc_dtype,
        use_2cta_instrs,
        mma_tiler_mn,
        cluster_shape_mn,
        use_tma_store,
        gated_a_load=False,
    )
    gemm_with_gated_a_load = SwizzledGemmKernel(
        acc_dtype,
        use_2cta_instrs,
        mma_tiler_mn,
        cluster_shape_mn,
        use_tma_store,
        gated_a_load=True,
    )

    can_implement = gemm.can_implement(a_local_list[0], b_tensor, c_local_list[0])
    if not can_implement:
        raise ValueError(
            f"The current config which is invalid/unsupported: use_2cta_instrs = {use_2cta_instrs}, "
            f"mma_tiler_mn = {mma_tiler_mn}, cluster_shape_mn = {cluster_shape_mn}, "
            f"use_tma_store = {use_tma_store}"
        )
    max_active_clusters = utils.HardwareInfo().get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1]
    )
    num_steps = world_size
    # flag_offset is a Constexpr: one compiled instance per step. Upstream also
    # compiles ungated instances for j>0 that the capture never calls -- kept.
    compiled_gemm_list = []
    compiled_gemm_with_gated_a_load_list = []
    for j in range(num_steps):
        m_dim_offset = (rank + j) % num_steps
        compiled_gemm_list.append(
            cute.compile(
                gemm,
                a_local_list[m_dim_offset],
                b_tensor,
                c_local_list[m_dim_offset],
                max_active_clusters,
                current_stream,
                flag_offset=m_dim_offset,
                gate_a_flags=from_dlpack(gate_a_flags),
            )
        )
        if j > 0:
            compiled_gemm_with_gated_a_load_list.append(
                cute.compile(
                    gemm_with_gated_a_load,
                    a_local_list[m_dim_offset],
                    b_tensor,
                    c_local_list[m_dim_offset],
                    max_active_clusters,
                    current_stream,
                    flag_offset=m_dim_offset,
                    gate_a_flags=from_dlpack(gate_a_flags),
                )
            )

    device_arrival_counters_ptrs = torch.tensor(
        device_arrival_counters_hdl.buffer_ptrs, device="cuda", dtype=torch.int64
    )
    sync_nvl_devices_instance = SyncNvlDevices(world_size)
    compiled_sync_nvl_devices = cute.compile(
        sync_nvl_devices_instance,
        device_idx=rank,
        device_arrival_counters=from_dlpack(device_arrival_counters_ptrs),
        iteration_flags=from_dlpack(gate_a_flags),
        stream=current_stream,
    )

    release_flag_set_instance = ReleaseFlagSet()
    compiled_release_flag_set = cute.compile(
        release_flag_set_instance,
        gate_a_flags=from_dlpack(gate_a_flags),
        step=cutlass.Int32(0),
        stream=current_stream,
    )

    g = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream(local_rank)
    copy_stream = torch.cuda.Stream(local_rank)
    gemm_stream = torch.cuda.Stream(local_rank)

    with torch.cuda.graph(g, stream=capture_stream):
        compiled_sync_nvl_devices(
            rank,
            from_dlpack(device_arrival_counters_ptrs),
            from_dlpack(gate_a_flags),
            cuda.CUstream(capture_stream.cuda_stream),
        )
        gemm_stream.wait_stream(capture_stream)
        for j in range(num_steps):
            m_dim_offset = (local_rank + j) % num_steps
            if j == 0:  # gemm process local data directly
                compiled_gemm_list[j](
                    a_local_list[m_dim_offset],
                    b_tensor,
                    c_local_list[m_dim_offset],
                    cuda.CUstream(gemm_stream.cuda_stream),
                    gate_a_flags=from_dlpack(gate_a_flags),
                )
            else:  # gemm process remote data with gated a load
                compiled_gemm_with_gated_a_load_list[j - 1](
                    a_local_list[m_dim_offset],
                    b_tensor,
                    c_local_list[m_dim_offset],
                    cuda.CUstream(gemm_stream.cuda_stream),
                    gate_a_flags=from_dlpack(gate_a_flags),
                )

        copy_stream.wait_stream(capture_stream)
        for j in range(num_steps):
            m_dim_offset = (local_rank + j) % num_steps
            if m_dim_offset != local_rank:
                copy_bytes = (
                    a_local_torch_list[m_dim_offset].element_size()
                    * a_local_torch_list[m_dim_offset].numel()
                )
                dst_address = a_local_torch_list[m_dim_offset].data_ptr()
                src_address = a_tensor_symm_hdl.buffer_ptrs[m_dim_offset]
                driver.cuMemcpyDtoDAsync(
                    dst_address,
                    src_address,
                    copy_bytes,
                    copy_stream.cuda_stream,
                )
                compiled_release_flag_set(
                    from_dlpack(gate_a_flags),
                    cutlass.Int32(m_dim_offset),
                    cuda.CUstream(copy_stream.cuda_stream),
                )

        capture_stream.wait_stream(copy_stream)
        capture_stream.wait_stream(gemm_stream)

    # delta: upstream's replay warmup + event timing + _compare + destroy
    # replaced by handing the graph to the shared do_bench_counts / ref_check.
    # The graph holds raw pointers only -- keep every captured buffer and
    # compiled module alive for the Kernel's lifetime (upstream keeps them in
    # run()'s scope; a factory must pin them explicitly).
    keep = (a_local_list, a_local_torch_list, a_torch_unique, a_tensor_symm_hdl,
            b_tensor, b_torch, gate_a_flags, device_arrival_counters_hdl,
            device_arrival_counters_ptrs, compiled_gemm_list,
            compiled_gemm_with_gated_a_load_list, compiled_sync_nvl_devices,
            compiled_release_flag_set)

    def run(_keep=keep):
        g.replay()

    def read_out():
        return torch.cat(c_local_torch_list, dim=0)[:, :, 0].float()

    def zero():
        for c_local_torch in c_local_torch_list:
            c_local_torch.zero_()

    return Kernel(run, _NOOP, A, B, read_out, zero, raw=c_local_torch_list)


def _asynctp(dtype, M, N, K, rank, world):
    """PyTorch async-TP: micro-pipelined all_gather(A) @ B via symmetric memory
    (the decomposition torchtitan's enable_async_tensor_parallel applies;
    private API). Gathers along dim 0 of the (shard_m, K) shard."""
    import torch.distributed._symmetric_memory as symm

    gn = dist.group.WORLD.group_name
    symm.enable_symm_mem_for_group(gn)
    shard_m = M // world
    A = torch.randn(shard_m, K, dtype=torch.bfloat16, device="cuda") / math.sqrt(K)
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / math.sqrt(K)
    B_kn = B.T
    res = {}

    def run():
        _, outs = torch.ops.symm_mem.fused_all_gather_matmul(A, [B_kn], 0, gn)
        res["out"] = outs[0]

    # output is freshly allocated each call, so zero is a no-op
    return Kernel(run, _NOOP, A, B, lambda: res["out"].float(), _NOOP, raw=res)


def _quack_l2_rotate_inputs(dtype, M, N, K, rank, world):
    """(fn, arg_sets) + sweep for do_bench_l2_rotate_all — quack's clone step
    plus the chain fn(b, d). B/D are the rotating L2-turnover payload; A rides
    the runner's double-buffer and is CE-rewritten every call (hot-A is
    production-real: the GEMM reads it right after the push lands). n_sets
    forced even so a captured round leaves the double-buffer parity
    graph-invariant. The config sweep (tile x swizzle x scheduler, see
    --quack_configs) shares arg_sets and the runner (as tk's num_comm_sms sweep)."""
    from quack.distributed import AllGatherRunner
    from quack.gemm import gemm as quack_gemm

    shard_m = M // world
    A = torch.randn(shard_m, K, dtype=dtype, device="cuda") / math.sqrt(K)
    B = torch.randn(N, K, dtype=dtype, device="cuda") / math.sqrt(K)
    D = torch.empty(M, N, dtype=dtype, device="cuda")
    # Lockstep: nothing paces the ranks in an isolated loop and quack's cross-call schedule drifts
    # (150 vs 135 us at qkv-8B TP=4); it gives quack the per-call sync every other arm already has.
    runner = AllGatherRunner(shard_m, K, dtype, capture_lockstep=True)

    def make_fn(cfg):
        tile_m, tile_n, cluster_m, cluster_n, swizzle, clc = cfg

        def fn(b, d):
            with runner.gather(runner.next_local_slot()) as (a_full, ag_args):
                quack_gemm(a_full, b, d, None, None, tile_m, tile_n, cluster_m, cluster_n,
                           is_dynamic_persistent=clc, max_swizzle_size=swizzle, ag_args=ag_args)
        return fn

    arg_sets = l2_rotate_arg_sets(B, D)
    for _ in range(len(runner.bufs)):
        runner.next_local_slot().copy_(A)
        runner.last_parity ^= 1  # two flips = restored; prefills both slots
    torch.cuda.synchronize()
    sweep = [(quack_config_label(c), make_fn(c)) for c in quack_configs]
    return sweep[0][1], arg_sets, (runner, A), sweep


def _nccl_l2_rotate_inputs(dtype, M, N, K, rank, world):
    """(fn, arg_sets) for the _nccl reference under do_bench_l2_rotate_all:
    cloned (B, D) per set; A_full is regathered (rewritten) every call and
    stays shared."""
    shard_m = M // world
    A = torch.randn(shard_m, K, dtype=torch.bfloat16, device="cuda") / math.sqrt(K)
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / math.sqrt(K)
    A_full = torch.empty(M, K, dtype=torch.bfloat16, device="cuda")
    D = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")

    def fn(b, d):
        dist.all_gather_into_tensor(A_full, A)
        torch.matmul(A_full, b.T, out=d)

    return fn, l2_rotate_arg_sets(B, D), (A, A_full)


def _legs_l2_rotate_inputs(dtype, M, N, K, rank, world):
    """The bound legs under do_bench_l2_rotate_all: cuBLAS GEMM = the solo
    per-rank (M, n_local, K) GEMM on a pre-gathered A; CE AG = bare
    cudaMemcpyAsync pushes of the A shard into each peer's symmetric buffer
    (the runner's reverse ring order), unpaced. main reports the slower leg as
    the bound (perfect-overlap floor)."""
    import torch.distributed._symmetric_memory as symm_mem
    from cuda.bindings import runtime
    from quack.distributed import AllGatherRunner

    shard_m = M // world
    A = torch.randn(shard_m, K, dtype=dtype, device="cuda") / math.sqrt(K)
    B = torch.randn(N, K, dtype=dtype, device="cuda") / math.sqrt(K)
    A_full = torch.empty(M, K, dtype=dtype, device="cuda")
    dist.all_gather_into_tensor(A_full, A)
    D = torch.empty(M, N, dtype=dtype, device="cuda")
    gn = dist.group.WORLD.group_name
    symm_mem.enable_symm_mem_for_group(gn)
    a_symm = symm_mem.empty(M, K, dtype=dtype, device="cuda")
    a_symm[rank * shard_m:(rank + 1) * shard_m].copy_(A)
    hdl = symm_mem.rendezvous(a_symm, gn)
    shard_bytes = A.numel() * A.element_size()
    src = a_symm.data_ptr() + rank * shard_bytes
    d2d = runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice
    runner = AllGatherRunner(shard_m, K, dtype)
    for _ in range(len(runner.bufs)):
        runner.next_local_slot().copy_(A)
        runner.last_parity ^= 1  # two flips = restored; prefills both slots
    torch.cuda.synchronize()

    def gemm_leg(b, d):
        torch.matmul(A_full, b.T, out=d)

    def push_leg(b, d):
        stream = torch.cuda.current_stream().cuda_stream
        for step in range(1, world):
            dst = (rank - step) % world
            (err,) = runtime.cudaMemcpyAsync(hdl.buffer_ptrs[dst] + rank * shard_bytes, src, shard_bytes, d2d, stream)
            assert err == runtime.cudaError_t.cudaSuccess, err

    # Bare pushes do not drift (95 us unpaced at the qkv-8B shard, TP=4, 2026-09-20 = the table's floor); the
    # runner's per-peer 4-byte flag copies do (135 us unpaced, 116 with a barrier every 4 calls on the push
    # stream, and that pacing alone costs bare pushes 8-11 us). Unused: the runner's own transport cost, not the bound.
    pace, n_calls, ev_sends = 4, [0], torch.cuda.Event()

    def push_leg_runner(b, d):
        with runner.gather(runner.next_local_slot()):
            pass
        n_calls[0] += 1
        if n_calls[0] % pace == 0:
            with torch.cuda.stream(runner.push_stream):  # behind this call's sends; channels 0/1 are the runner's
                runner.handle.barrier(channel=2)
            ev_sends.record(runner.push_stream)  # rejoin: the runner's own join ran before the barrier
            torch.cuda.current_stream().wait_event(ev_sends)

    sweep = [("cuBLAS GEMM", gemm_leg), ("CE AG", push_leg)]
    return gemm_leg, l2_rotate_arg_sets(B, D), (A, A_full, a_symm, hdl, runner), sweep


def _tk_l2_rotate_inputs(dtype, M, N, K, rank, world):
    """(fn, arg_sets) + sweep for the tk arm under do_bench_l2_rotate_all.
    all_gather_matmul is functional in (B, C), so both rotate; the
    num_comm_sms sweep returns per-candidate fns sharing one arg_sets (the
    historical no-per-iteration-C-zero caveat carries over)."""
    TKParallelTensor, all_gather_matmul = _tk_import(world)

    gpu = os.environ.get("GPU")
    assert gpu in ("B200", "B300", "H100"), "set GPU=B200|B300|H100 for tk"
    shard_m = M // world
    A = torch.randn(shard_m, K, dtype=torch.bfloat16, device="cuda") / math.sqrt(K)
    b_shape = (K, N) if gpu == "H100" else (N, K)
    B = torch.randn(*b_shape, dtype=torch.bfloat16, device="cuda") / math.sqrt(K)

    A_tk = TKParallelTensor((M, K), dtype=torch.bfloat16,
                            local_rank=rank, local_world_size=world, multicast=True)
    A_tk.data_.zero_()
    A_tk.data_[rank * shard_m:(rank + 1) * shard_m].copy_(A)
    C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    barrier = TKParallelTensor((2, 1024, 1024), dtype=torch.int,
                               local_rank=rank, local_world_size=world, multicast=True)
    barrier.data_.zero_()

    def make_fn(num_comm_sms):
        return lambda b, c: all_gather_matmul(A_tk, b, c, barrier, num_comm_sms)

    sweep = [(f"num_comm_sms={v}", make_fn(v)) for v in (2, 4, 8, 16, 32, 64)]
    return make_fn(16), l2_rotate_arg_sets(B, C), (A_tk, barrier, A), sweep


def _blackwell_l2_rotate_inputs(dtype, M, N, K, rank, world):
    """(fn, arg_sets) + sweep for the cutlass arm under do_bench_l2_rotate_all
    with capture=False: the arm's iteration is itself a captured graph (its
    line-for-line upstream mirror stays untouched), so rotation = one arm
    instance per cloned B (via the factory's B override, sharing A) and
    fn(i) replays that instance's graph. Replay can't record into an
    outer capture, hence the eager-timed mode. The raster swizzle 1/8 sweep
    = a second instance list over the same B clones."""
    shard_m = M // world
    A = torch.randn(shard_m, K, dtype=torch.bfloat16, device="cuda") / math.sqrt(K)
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / math.sqrt(K)
    D = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")  # sizing only

    Bs = [B.clone() for _ in range(l2_rotate_count(B, D))]
    arms = {s: [_blackwell(dtype, M, N, K, rank, world, A=A, B=b, swizzle=s) for b in Bs]
            for s in (1, 8)}

    def make_fn(swizzle):
        return lambda i: arms[swizzle][i].run()

    sweep = [(f"swizzle={s}", make_fn(s)) for s in (1, 8)]
    return make_fn(8), [(i,) for i in range(len(Bs))], (A, arms), sweep


def _asynctp_l2_rotate_inputs(dtype, M, N, K, rank, world):
    """(fn, arg_sets) for the asynctp arm under do_bench_l2_rotate_all.
    fused_all_gather_matmul is functional — outputs are allocated at
    capture time per recorded call, so only B rotates (graph capture +
    replay-recompute verified bit-exact 2026-08-20)."""
    import torch.distributed._symmetric_memory as symm

    gn = dist.group.WORLD.group_name
    symm.enable_symm_mem_for_group(gn)
    shard_m = M // world
    A = torch.randn(shard_m, K, dtype=torch.bfloat16, device="cuda") / math.sqrt(K)
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / math.sqrt(K)
    D = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")  # sizing only

    def fn(b, d):
        torch.ops.symm_mem.fused_all_gather_matmul(A, [b.T], 0, gn)

    return fn, l2_rotate_arg_sets(B, D), (A,)


def _te_ub(dtype, M, N, K, rank, world):
    """Transformer Engine Userbuffers AG+GEMM overlap (misc/te_ub.py): te.Linear column-parallel with
    ub_overlap_ag, the "ring_exchange" P2P ring on 1 SM beside the cuBLASLt GEMM; the output is allocated per
    call. B rides the module's weight; is_first_microbatch=False skips its per-call workspace refresh."""
    from te_ub import ag_linear, init_ub

    shard_m = M // world
    A = torch.randn(shard_m, K, dtype=torch.bfloat16, device="cuda") / math.sqrt(K)
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / math.sqrt(K)
    init_ub(M, K, world)
    lin = ag_linear(K, N, world, B)
    res = {}

    def fn(a, d):
        res["out"] = lin(a, is_first_microbatch=False)

    with torch.no_grad():
        lin(A)  # the first call: TE's one-time weight workspace
    return Kernel(lambda: fn(A, None), _NOOP, A, lin.weight.data, lambda: res["out"].float(), _NOOP, raw=res, fn=fn)


def _te_ub_l2_rotate_inputs(dtype, M, N, K, rank, world):
    """(fn, arg_sets) for the te_ub arm: the A shard rotates (it is copied into the Userbuffers buffer every call),
    B stays the module's weight (as asynctp's single rotating operand)."""
    kern = _te_ub(dtype, M, N, K, rank, world)
    D = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")  # sizing only
    return kern.fn, l2_rotate_arg_sets(kern.A, D), (kern,)


def _nccl(dtype, M, N, K, rank, world):
    """NCCL all-gather + cuBLAS GEMM reference."""
    shard_m = M // world
    A = torch.randn(shard_m, K, dtype=torch.bfloat16, device="cuda") / math.sqrt(K)
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / math.sqrt(K)
    A_full = torch.empty(M, K, dtype=torch.bfloat16, device="cuda")
    D = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")

    def run():
        dist.all_gather_into_tensor(A_full, A)
        torch.matmul(A_full, B.T, out=D)

    return Kernel(run, _NOOP, A, B, lambda: D.float(), D.zero_)


BASELINES = {
    "quack":       {"factory": _quack},
    "tk":          {"factory": _tk},
    "cutlass":     {"factory": _blackwell},
    "asynctp":     {"factory": _asynctp},
    "te_ub":       {"factory": _te_ub},
    "legs":        {"factory": None},  # bound legs (cuBLAS GEMM, CE AG); timing-only
    "82":          {"binary": True},
    "flux":        {"h100_lib": True},
    "triton_dist": {"h100_lib": True},
}

H100_LIB_ENVS = {
    "flux": "/scratch/gpfs/TRIDAO/zy4586/envs/flux",
    "triton_dist": "/scratch/gpfs/TRIDAO/zy4586/envs/triton-dist-main",  # upstream main 7908e4e, 2026-09-18
}


def run_h100_lib(lib, M, N, K):
    """flux / Triton-distributed only build for H100 (sm90) and live in their
    own conda envs, so shell out to h100_lib_runner.py under that env's
    torchrun (plain-python invocation, like 82; world = visible GPU count)."""
    cap = torch.cuda.get_device_capability()
    if cap != (9, 0):
        sys.exit(f"{lib} baseline is H100 (sm90)-only; this GPU is sm{cap[0]}{cap[1]}")
    torchrun = os.path.join(H100_LIB_ENVS[lib], "bin", "torchrun")
    runner = os.path.join(os.path.dirname(os.path.abspath(__file__)), "h100_lib_runner.py")
    env = os.environ.copy()
    if lib == "flux":
        # report IPC peer buffers as local so torch's from_blob device check passes
        env["LD_PRELOAD"] = "/scratch/gpfs/TRIDAO/zy4586/envs/ipc_attr_shim.so"
        # flux's loader looks in python/flux/lib, the build installs into lib64
        env["LD_LIBRARY_PATH"] = os.path.expanduser("~/flux/python/flux/lib64") + ":" + env.get(
            "LD_LIBRARY_PATH", ""
        )
    else:  # nvshmem's host lib is not on the loader path; env/lib = the libstdc++ (>= GLIBCXX_3.4.30) libtriton links
        env["LD_LIBRARY_PATH"] = ":".join([
            os.path.join(H100_LIB_ENVS[lib], "lib/python3.12/site-packages/nvidia/nvshmem/lib"),
            os.path.join(H100_LIB_ENVS[lib], "lib"), env.get("LD_LIBRARY_PATH", "")])
    # "--": torchrun abbrev-matches --m/--n/--k to its own flags without it
    subprocess.run([torchrun, f"--nproc_per_node={torch.cuda.device_count()}", "--", runner,
                    lib, f"--m={M}", f"--n={N}", f"--k={K}"], check=True, env=env)


def tflops(ms, M, N, K):
    return 2 * M * N * K / (ms * 1e9)


def _run_synced(kern):
    """Zero, barrier, run, sync, barrier -- one fresh cross-rank-consistent output."""
    kern.zero()
    torch.cuda.synchronize()
    dist.barrier()
    kern.run()
    torch.cuda.synchronize()
    dist.barrier()


def ref_check(kern, rank, world):
    """Max rel err of kern's full (M, N) output vs an fp32 reference
    (all-gather A M-shards, GEMM against the rank's own B)."""
    _run_synced(kern)
    got = kern.read_out()
    A_shards = [torch.empty_like(kern.A) for _ in range(world)]
    dist.all_gather(A_shards, kern.A)
    A_full = torch.cat(A_shards, dim=0).float()  # (M, K)
    ref = A_full @ kern.B.float().T              # (M, N)
    denom = ref.abs()
    mask = denom > 1e-3
    return ((got - ref).abs()[mask] / denom[mask]).max().item() if mask.any() else 0.0


def r2r_check(kern):
    """Run twice; return (bit-identical, max |delta|) of the rank's output."""
    _run_synced(kern)
    o1 = kern.read_out().clone()
    _run_synced(kern)
    o2 = kern.read_out().clone()
    return torch.equal(o1, o2), (o1 - o2).abs().max().item()


def run_82(M, N, K):
    """Shell out to the prebuilt cutlass 82 binary (timing-only; self-manages
    TP=4 on the first 4 visible GPUs). Build: ~/tmp/cutlass-82 branch 82-tp4
    (bf16 + --data_dist/--skip_verify + L2-rotation patches, sm103a-native), schedule
    AllGather1D_TilingCD_RotatingA; the RS-schedule TP4 binary is preserved
    next to it as *.rs_tp4. Its --n is the GLOBAL N; scaling our rank-local N
    by TP keeps per-rank work at 2*M*N*K like the other arms."""
    # TP is compile-time (using TP = _N); per-TP binaries .ag_tp{2,4} next to the build
    tp = torch.cuda.device_count()
    binary = os.environ.get("BIN", os.path.expanduser(
        "~/tmp/cutlass-82/build/examples/82_blackwell_distributed_gemm/"
        f"82_blackwell_distributed_gemm.ag_tp{tp}"))
    print(f"<82_blackwell bf16 AG+GEMM | {M}x{K}x{N} local (global N {N * tp}) | L2-rotate (in-binary)>",
          flush=True)
    # the binary's harness rotates cloned operand sets in its timed loop
    # (quack _pick_l2_rotate_count semantics) with one event pair + cross-device max
    p = subprocess.run([binary, f"--m={M}", f"--n={N * tp}", f"--k={K}",
                        "--data_dist=randn", "--skip_verify",
                        "--warmup-iterations=100", "--iterations=500"],
                       check=True, capture_output=True, text=True)
    print(p.stdout, end="", flush=True)
    ms = float(re.search(r"Avg runtime: ([\d.]+) ms", p.stdout).group(1))
    print(f"  {'82:':<14} {ms:.3f} ms | {tflops(ms, M, N, K):.2f} TFLOp/s", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("baseline", choices=list(BASELINES))
    p.add_argument("--model", choices=list(MODEL_CONFIGS), required=True, help="shape source (MODEL_CONFIGS)")
    p.add_argument("--site", choices=("qkv", "gate_up"), required=True, help="column-parallel GEMM (SITES)")
    p.add_argument("--m", type=int, required=True, help="gathered rows (2048 prefill, 8192 training)")
    p.add_argument("--ref_check", action="store_true",
                   help="Max rel err of the kernel & reference vs an fp32 full GEMM.")
    p.add_argument("--r2r_check", action="store_true",
                   help="Run kernel & reference twice; report bit-identity and max |delta|.")
    p.add_argument("--eager", action="store_true",
                   help="Time kernel & reference eager (capture=False), as the flux/triton_dist runners do.")
    p.add_argument("--rounds", type=int, default=1,
                   help="re-time the winning config N-1 more times back-to-back (self-shadow only); "
                        "the reference is timed once, last, as the contention canary")
    p.add_argument("--quack_configs", default=None,
                   help="quack sweep: 'tm,tn,cm,cn[,sw,static|clc];...' (default: quack's autotuner "
                        "winner + 256x256/256x512 c(2,1) x swizzle {1,8} x {static,clc})")
    args = p.parse_args()
    n_global, K = SITES[args.site](MODEL_CONFIGS[args.model])
    tp = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))  # 82 runs without torchrun
    assert n_global % tp == 0, f"{args.model} {args.site}: N={n_global} not divisible by TP={tp}"
    M, N = args.m, n_global // tp
    global quack_configs
    quack_configs = parse_quack_configs(args.quack_configs)

    spec = BASELINES[args.baseline]
    if spec.get("binary"):
        run_82(M, N, K)
        if args.ref_check or args.r2r_check:
            print("  (82: ref/r2r check N/A -- external binary, timing-only)", flush=True)
        return
    if spec.get("h100_lib"):
        run_h100_lib(args.baseline, M, N, K)
        if args.ref_check or args.r2r_check:
            print(f"  ({args.baseline}: ref/r2r check N/A -- external env, timing-only)", flush=True)
        return

    torchrun_init()
    rank, world = dist.get_rank(), dist.get_world_size()
    assert M % world == 0
    if args.quack_configs is None and args.baseline == "quack":
        tile = tune_quack_tile(M, N, K)  # winner + the always-run fallbacks, x swizzle {1,8} x {static,clc}
        shard = M // world  # AG geometry: shard boundaries must land on whole scheduler clusters
        cta_m = lambda tm, cm: tm // 2 if cm % 2 == 0 and tm in (128, 256) else tm  # 2-CTA MMA
        tiles = [t for t in (tile, (256, 256, 2, 1), (256, 512, 2, 1))
                 if shard % (cta_m(t[0], t[2]) * t[2]) == 0]
        assert tiles, f"no quack config with tile_m*cluster_m dividing the {shard}-row shard"
        quack_configs = list(dict.fromkeys(parse_quack_configs(";".join(
            ",".join(map(str, t)) for t in tiles))))
        clean_print(f"  quack tuner: {tile[0]}x{tile[1]} c({tile[2]},{tile[3]})"
                    + ("" if len(tiles) == 3 else f" (tiles beyond the {shard}-row shard dropped)"),
                    print_once=True)

    # Default protocol: the quack-autotuner L2-cold rotate methodology
    # (do_bench_l2_rotate_all) for rotate-capable arms, so configs are
    # picked and reported under one regime; arms without a
    # *_l2_rotate_inputs helper fall back to counts and say so. Checks
    # always run through the plain factories (rotate is timing-only).
    # capture=False for arms whose iteration is itself a captured graph.
    ROTATE_INPUTS = {
        "quack":   (_quack_l2_rotate_inputs, True),
        "tk":      (_tk_l2_rotate_inputs, True),
        "cutlass": (_blackwell_l2_rotate_inputs, False),
        "asynctp": (_asynctp_l2_rotate_inputs, True),
        "te_ub":   (_te_ub_l2_rotate_inputs, True),
        "legs":    (_legs_l2_rotate_inputs, True),
    }
    rotate_inputs, rotate_capture = ROTATE_INPUTS.get(args.baseline, (None, True))
    rotate_capture = rotate_capture and not args.eager
    need_plain = args.ref_check or args.r2r_check or rotate_inputs is None
    assert not (need_plain and spec["factory"] is None), "legs: timing-only, no checks"
    kern = spec["factory"](torch.bfloat16, M, N, K, rank, world) if need_plain else None
    ref = _nccl(torch.bfloat16, M, N, K, rank, world) if need_plain else None

    checks = []
    if args.ref_check:
        checks.append(("rel err", f"{ref_check(kern, rank, world):.4e} (|ref|>1e-3)",
                       f"{ref_check(ref, rank, world):.4e}"))
    if args.r2r_check:
        kb, kd = r2r_check(kern)
        rb, rd = r2r_check(ref)
        checks.append(("r2r", f"bit-identical={kb} max|Δ|={kd:.4e}",
                       f"bit-identical={rb} max|Δ|={rd:.4e}"))

    sweep_ms = []
    if rotate_inputs is not None:
        from quack.autotuner import _gpu_warmup

        proto = "L2-rotate" if rotate_capture else "L2-rotate (eager-timed)"
        out = rotate_inputs(torch.bfloat16, M, N, K, rank, world)
        fn, arg_sets, _keep = out[:3]
        sweep = out[3] if len(out) > 3 else None
        ref_fn, ref_arg_sets, _ref_keep = _nccl_l2_rotate_inputs(torch.bfloat16, M, N, K, rank, world)
        _gpu_warmup()
        kwarg_sets = [{}] * len(arg_sets)
        if sweep:
            for label, fn_v in sweep:
                dist.barrier()
                sweep_ms.append((label, do_bench_l2_rotate_all(
                    fn_v, arg_sets, kwarg_sets, {}, capture=rotate_capture)))
            pick = max if args.baseline == "legs" else min  # legs: the bound is the slower leg
            best_label, ms = pick(sweep_ms, key=lambda r: r[1])
        else:
            dist.barrier()
            ms = do_bench_l2_rotate_all(fn, arg_sets, kwarg_sets, {}, capture=rotate_capture)
        dist.barrier()
        rounds = []  # (arm_ms, {leg: ms} | None) per extra round, winner config only.
        # Rounds run back-to-back right after the sweep; the ref chain is timed ONCE at the
        # end (its ~100 ms full-power window depresses the next timing's clocks ~5% at
        # heavy shapes -- the 82 binary self-times in isolation, so arm rounds must too).
        best_fn = dict(sweep)[best_label] if sweep_ms else fn
        for _ in range(args.rounds - 1):
            legs = None
            if args.baseline == "legs":  # every leg per round; the bound is the slower one
                legs = {}
                for label, fn_v in sweep:
                    dist.barrier()
                    legs[label] = do_bench_l2_rotate_all(fn_v, arg_sets, kwarg_sets, {}, capture=rotate_capture)
                v = max(legs.values())
            else:
                dist.barrier()
                v = do_bench_l2_rotate_all(best_fn, arg_sets, kwarg_sets, {}, capture=rotate_capture)
            rounds.append((v, legs))
        dist.barrier()
        ref_ms = do_bench_l2_rotate_all(ref_fn, ref_arg_sets, [{}] * len(ref_arg_sets), {},
                                        capture=rotate_capture)
    else:
        proto = "counts — not rotate-capable yet"
        rounds = []
        if kern.sweep:
            for label, run in kern.sweep:
                dist.barrier()
                time.sleep(1.0)  # prevent throttling
                sweep_ms.append((label, do_bench_counts(run, calls_per_inv=kern.calls)))
            best_label, ms = min(sweep_ms, key=lambda r: r[1])
        else:
            dist.barrier()
            time.sleep(1.0)  # prevent throttling
            ms = do_bench_counts(kern.run, calls_per_inv=kern.calls)
        dist.barrier()
        time.sleep(1.0)
        ref_ms = do_bench_counts(ref.run)

    clean_print("=" * 80, print_once=True)
    clean_print(f"<BF16 AG+GEMM {args.baseline} | world={world} | {M}x{K}x{N} (N local) | {proto}>",
                print_once=True)
    for label, v_ms in sweep_ms:
        clean_print(f"  {args.baseline + ' ' + label + ':':<22} {v_ms:.3f} ms | {tflops(v_ms, M, N, K):.2f} TFLOp/s", print_once=True)
    if sweep_ms:
        kern_name = (f"bound ({best_label})" if args.baseline == "legs"
                     else f"{args.baseline} best ({best_label})")
    else:
        kern_name = args.baseline
    clean_print(f"  {kern_name + ':':<14} {ms:.3f} ms | {tflops(ms, M, N, K):.2f} TFLOp/s", print_once=True)
    clean_print(f"  {'NCCL+cuBLAS:':<14} {ref_ms:.3f} ms | {tflops(ref_ms, M, N, K):.2f} TFLOp/s", print_once=True)
    for i, (v, legs) in enumerate(rounds, 1):
        for label, lv in (legs or {}).items():
            clean_print(f"  {args.baseline + ' ' + label + f' round{i}:':<22} {lv:.3f} ms | {tflops(lv, M, N, K):.2f} TFLOp/s", print_once=True)
        clean_print(f"  {args.baseline + f' round{i}:':<22} {v:.3f} ms | {tflops(v, M, N, K):.2f} TFLOp/s", print_once=True)
    for label, kv, rv in checks:
        clean_print(f"  {args.baseline + ' ' + label + ':':<22} {kv}", print_once=True)
        clean_print(f"  {'NCCL+cuBLAS ' + label + ':':<22} {rv}", print_once=True)

    if kern is not None:
        kern.free()
        ref.free()
    dist.barrier()
    torchrun_finalize()


if __name__ == "__main__":
    main()
