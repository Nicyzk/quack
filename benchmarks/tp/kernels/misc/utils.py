import datetime
import math
import os
from collections import namedtuple
import numpy as np

import torch
import torch.distributed as dist

try:  # the NCCL channel floor every reference runs with (see nccl_env.py)
    from kernels.misc import nccl_env  # noqa: F401
except ImportError:
    import nccl_env  # noqa: F401  (imported by path from kernels/misc)

import triton.runtime as triton_runtime
from triton.testing import _summarize_statistics

import nvshmem.core

try:
    from cuda.core.experimental import Device
except ModuleNotFoundError:  # cuda-core >= 1.0 promoted the API to stable
    from cuda.core import Device

import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack


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
        # we don't want `fn` to accumulate gradient values
        # if it contains a backward pass. So we clear the
        # provided gradients
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

def do_bench_counts(fn, warmup=100, rep=500, calls_per_inv=1):
    """ms per iteration: `warmup` untimed iterations, then ONE timed
    back-to-back block of `rep` iterations (single event pair, no L2
    clearing) — benchmark_all_gather_gemm's burst timing, solo. Counts are
    iterations; an fn() performing calls_per_inv of them runs counts/that
    many times. Cross-rank max, so every rank reports the fleet time."""
    n_warm = max(1, warmup // calls_per_inv)
    n_rep = max(1, rep // calls_per_inv)
    for _ in range(n_warm):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    fn()  # un-timed invocation absorbs the post-barrier skew
    s_ev = torch.cuda.Event(enable_timing=True)
    e_ev = torch.cuda.Event(enable_timing=True)
    s_ev.record()
    for _ in range(n_rep):
        fn()
    e_ev.record()
    torch.cuda.synchronize()
    ms = s_ev.elapsed_time(e_ev) / (n_rep * calls_per_inv)
    t = torch.tensor([ms], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


def capture_graph(fn, warmup=3):
    """Capture fn's launches into a CUDA graph, returning replay (agent/cuda_graph_kernels.md).
    fn must launch on torch's current stream: tvm-ffi env stream with a live torch tensor arg,
    or a CUstream re-fetched inside fn."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g.replay


def create_mc_tensor(torch_tensor_cpu, dtype, leading_dim, is_dynamic_layout=True):
    # nvshmem allocs are contiguous: alloc in descending-stride (base) order and permute
    # back so the symmetric tensor matches the cpu view's strides (l > 1 breaks otherwise).
    ndim = torch_tensor_cpu.dim()
    base_order = sorted(range(ndim), key=lambda d: -torch_tensor_cpu.stride(d))
    inv = [base_order.index(d) for d in range(ndim)]
    base = nvshmem.core.tensor(
        tuple(torch_tensor_cpu.shape[d] for d in base_order), dtype=torch_tensor_cpu.dtype
    )
    torch_tensor_gpu = base.permute(inv)
    torch_tensor_gpu.copy_(torch_tensor_cpu)
    torch_tensor_gpu_mc = nvshmem.core.get_multicast_tensor(
        nvshmem.core.Teams.TEAM_NODE, base
    ).permute(inv)
    peer_torch_tensors = [
        nvshmem.core.get_peer_tensor(base, rank).permute(inv)
        for rank in range(dist.get_world_size())
    ]
    cute_peer_tensors = [from_dlpack(t) for t in peer_torch_tensors]
    cute_tensor_mc = from_dlpack(torch_tensor_gpu_mc, assumed_align=16)
    if is_dynamic_layout:
        cute_tensor_mc = cute_tensor_mc.mark_layout_dynamic(leading_dim=leading_dim)
    cute_tensor = from_dlpack(torch_tensor_gpu, assumed_align=16)
    cute_tensor.element_type = dtype
    if is_dynamic_layout:
        cute_tensor = cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)
    cute_tensor = cutlass_torch.convert_cute_tensor(torch_tensor_gpu, cute_tensor, dtype, is_dynamic_layout=is_dynamic_layout)
    return cute_tensor, cute_tensor_mc, torch_tensor_gpu, torch_tensor_gpu_mc, peer_torch_tensors, cute_peer_tensors

def create_tensors_randn(l, m, n, k, a_major, b_major, c_major, ab_dtype, c_dtype, world_size=1):
    """create_tensors variant that initializes A, B ~ N(0, 1/sqrt(k * world_size)) in fp32, then casts.
    k is per-rank K; world_size > 1 normalizes by global K = k * world_size (as if unsharded).
    """
    torch.manual_seed(1111)
    inv_sqrt_k = 1.0 / math.sqrt(k * world_size)
    torch_ab_dtype = cutlass_torch.dtype(ab_dtype)

    a_torch_cpu = (cutlass_torch.matrix(l, m, k, a_major == "m", ab_dtype)
                   .to(torch.float32).normal_().mul_(inv_sqrt_k).to(torch_ab_dtype))
    b_torch_cpu = (cutlass_torch.matrix(l, n, k, b_major == "n", ab_dtype)
                   .to(torch.float32).normal_().mul_(inv_sqrt_k).to(torch_ab_dtype))
    c_torch_cpu = cutlass_torch.matrix(l, m, n, c_major == "m", c_dtype)

    a_tensor, _ = cutlass_torch.cute_tensor_like(
        a_torch_cpu, ab_dtype, is_dynamic_layout=True, assumed_align=16)
    b_tensor, _ = cutlass_torch.cute_tensor_like(
        b_torch_cpu, ab_dtype, is_dynamic_layout=True, assumed_align=16)

    c_tensor, c_tensor_mc, c_torch_gpu, c_torch_gpu_mc, c_peer_torch_tensors, c_peer_tensors = create_mc_tensor(
        c_torch_cpu, c_dtype, (1 if c_major == "n" else 0), is_dynamic_layout=True)

    return (
        a_tensor, b_tensor, c_tensor, c_tensor_mc,
        a_torch_cpu, b_torch_cpu, c_torch_cpu,
        c_torch_gpu, c_torch_gpu_mc,
        c_peer_torch_tensors, c_peer_tensors,
    )

def torchrun_init():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    assert world_size == local_world_size, "multi-node runs are not supported"
    assert rank == local_rank, "multi-node runs are not supported"
    dist.init_process_group(
        backend="nccl",
        device_id=local_rank,
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(hours=1),  # peers idle through rank 0's first-compile tune
    )
    
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.random.manual_seed(local_rank)

    return local_rank, local_world_size

def torchrun_finalize():
    dist.destroy_process_group()

def import_nvshmem():
    """Import nvshmem.core + load the nvshmem_host lib (pip hints on failure); returns the module."""
    from cuda.pathfinder import load_nvidia_dynamic_lib
    try:
        import nvshmem.core
    except ImportError:
        raise ImportError("nvshmem4py required: pip install nvshmem4py-cu12 (or -cu13), >=0.1.3") from None
    try:
        load_nvidia_dynamic_lib("nvshmem_host")
    except RuntimeError:
        raise ImportError("nvidia-nvshmem lib required: pip install nvidia-nvshmem-cu12 (or -cu13)") from None
    return nvshmem


def torchrun_init_nvshmem(dist_initialized=False):
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dev = Device(local_rank)
    dev.set_current()
    if not dist_initialized:
        dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    num_ranks = dist.get_world_size()
    uid = nvshmem.core.get_unique_id(empty=(local_rank != 0))
    uid_bytes = uid._data.view(np.uint8).copy()
    uid_tensor = torch.from_numpy(uid_bytes).cuda()
    dist.broadcast(uid_tensor, src=0)
    dist.barrier()
    uid._data[:] = uid_tensor.cpu().numpy().view(uid._data.dtype)
    nvshmem.core.init(device=dev, uid=uid, rank=local_rank, nranks=num_ranks, initializer_method="uid")

# Cleanup callbacks run (in reverse) by torchrun_finalize_nvshmem, so factories that allocate
# symmetric memory internally (barrier flags, multicast views) don't leak into finalize errors.
_finalizers = []

def on_finalize(fn):
    _finalizers.append(fn)

def run_finalizers():
    for fn in reversed(_finalizers):
        try:
            fn()
        except Exception:
            pass  # tolerate already-freed tensors
    _finalizers.clear()

def torchrun_finalize_nvshmem():
    run_finalizers()
    nvshmem.core.finalize()
    dist.destroy_process_group()

def make_barrier_flags(num_flags):
    bf_torch = nvshmem.core.tensor((num_flags,), dtype=torch.int32)
    bf_torch.fill_(0)
    bf_torch_mc = nvshmem.core.get_multicast_tensor(nvshmem.core.Teams.TEAM_NODE, bf_torch)
    bf = from_dlpack(bf_torch).mark_layout_dynamic()
    bf_mc = from_dlpack(bf_torch_mc).mark_layout_dynamic()
    on_finalize(lambda: (nvshmem.core.free_tensor(bf_torch_mc), nvshmem.core.free_tensor(bf_torch)))
    return bf_torch, bf_torch_mc, bf, bf_mc

def make_peer_tensor(shape, dtype):
    """Symmetric NVSHMEM tensor + per-rank peer views (no multicast)."""
    t = nvshmem.core.tensor(shape, dtype=dtype)
    peers = [nvshmem.core.get_peer_tensor(t, r) for r in range(dist.get_world_size())]
    on_finalize(lambda: nvshmem.core.free_tensor(t))
    return t, peers

ModelConfig = namedtuple("ModelConfig", ["hidden", "q_heads", "kv_heads", "head_dim", "intermediate"])
MODEL_CONFIGS = {
    "llama3_8b": ModelConfig(4096, 32, 8, 128, 14336),
    "llama3_70b": ModelConfig(8192, 64, 8, 128, 28672),
    "mistral_large2": ModelConfig(12288, 96, 8, 128, 28672),
    "llama3_405b": ModelConfig(16384, 128, 8, 128, 53248),
}
# TP GEMM sites as GLOBAL (N, K): column-parallel qkv / gate_up (AG+GEMM: the harness takes
# N/TP per rank), row-parallel out / down (GEMM+RS: K/TP per rank).
SITES = {
    "qkv": lambda c: ((c.q_heads + 2 * c.kv_heads) * c.head_dim, c.hidden),
    "gate_up": lambda c: (2 * c.intermediate, c.hidden),
    "out": lambda c: (c.hidden, c.hidden),
    "down": lambda c: (c.hidden, c.intermediate),
}

QUACK_TILE_CANDIDATES = ((256, 256, 2, 1), (256, 512, 2, 1))


def parse_quack_configs(spec=None, clc_ok=True):
    """quack GEMM config candidates for a fused arm's sweep, as a list of
    (tile_m, tile_n, cluster_m, cluster_n, swizzle, clc). ``spec`` = ';'-separated
    entries 'tm,tn,cm,cn' (expanded over swizzle {1,8} x {static,clc}) or
    'tm,tn,cm,cn,sw,static|clc' (exact). None = QUACK_TILE env (the per-cell tuner
    winner), if set, plus QUACK_TILE_CANDIDATES, all expanded. clc_ok=False keeps
    static only (epi_reduce_mode requires static persistence)."""
    scheds = (False, True) if clc_ok else (False,)
    if spec is None:
        tiles = []
        if os.environ.get("QUACK_TILE"):
            tiles.append(tuple(int(x) for x in os.environ["QUACK_TILE"].split(",")))
        tiles += [t for t in QUACK_TILE_CANDIDATES if t not in tiles]
        entries = [",".join(map(str, t)) for t in tiles]
    else:
        entries = [e for e in spec.split(";") if e]
    out = []
    for e in entries:
        f = e.split(",")
        tile = tuple(int(x) for x in f[:4])
        if len(f) == 4:
            out += [(*tile, sw, clc) for sw in (1, 8) for clc in scheds]
        else:
            clc = f[5] == "clc"
            assert clc_ok or not clc, f"config {e}: clc not supported by this arm"
            out.append((*tile, int(f[4]), clc))
    return out


def quack_config_label(cfg):
    tm, tn, cm, cn, sw, clc = cfg
    return f"{tm}x{tn} c({cm},{cn}) sw{sw} {'clc' if clc else 'static'}"


def tune_quack_tile(m, n, k, dtype=torch.bfloat16):
    """The campaign's tile pick, sonic's method on the plain GEMM: rank 0 runs quack's
    stock autotuner (gemm_interface with tuned=True, static scheduler, disk-cached
    by shape) on D = A @ B^T at this rank's local shape, A (m, k) and B (n, k), and
    broadcasts the winner's (tile_m, tile_n, cluster_m, cluster_n). A private RNG
    keeps the arms' operand draws untouched."""
    tile = torch.zeros(4, dtype=torch.int64, device="cuda")
    if dist.get_rank() == 0:
        from quack.gemm_interface import gemm, gemm_tuned

        g = torch.Generator(device="cuda").manual_seed(0)
        a = torch.randn(m, k, dtype=dtype, device="cuda", generator=g) / math.sqrt(k)
        b = torch.randn(n, k, dtype=dtype, device="cuda", generator=g) / math.sqrt(k)
        gemm(a, b.T, out=torch.empty(m, n, dtype=dtype, device="cuda"), dynamic_scheduler=False)
        c = gemm_tuned.best_config.kwargs["config"]
        tile.copy_(torch.tensor([c.tile_m, c.tile_n, c.cluster_m, c.cluster_n]))
    dist.broadcast(tile, src=0)
    return tuple(tile.tolist())


def clean_print(*args, **kwargs):
    # No collectives -> can't deadlock; ranks just print (output may interleave).
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    print_once = kwargs.pop("print_once", False)
    if print_once and local_rank != 0:
        return
    kwargs.setdefault("flush", True)
    prefix = () if print_once else (f"[Rank {local_rank}]",)
    print(*prefix, *args, **kwargs)
