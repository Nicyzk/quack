"""Rank-synced L2-cold rotation timing, importable from any conda env.

torch + torch.distributed only -- no quack/triton/nvshmem/cutlass imports --
so the H100-only external-library runners (ag_gemm and gemm_rs
h100_lib_runner.py, launched under the flux / triton-dist envs) can share
the exact driver that bench_baselines.py uses; l2_rotate_count /
l2_rotate_arg_sets size the operand sets for every harness.
"""
import time
from collections import namedtuple

import torch
import torch.distributed as dist

try:  # the NCCL channel floor every reference runs with (see nccl_env.py)
    from kernels.misc import nccl_env  # noqa: F401
except ImportError:
    import nccl_env  # noqa: F401  (imported by path from kernels/misc)


# Referenced-from: ~/quack/quack/bench/bench_utils.py (verbatim; for envs without quack)
def _pick_l2_rotate_count(
    args, kwargs, target_ratio: int = 3, min_buffers: int = 4, max_buffers: int = 16
):
    if not torch.cuda.is_available():
        return min_buffers
    tensor_bytes = sum(
        a.numel() * a.element_size() for a in args if isinstance(a, torch.Tensor)
    ) + sum(v.numel() * v.element_size() for v in kwargs.values() if isinstance(v, torch.Tensor))
    if tensor_bytes == 0:
        return min_buffers
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    l2_size = getattr(props, "L2_cache_size", 50 * 1024 * 1024)  # torch<2.5 lacks it; 50MB=H100
    n_by_l2 = (target_ratio * l2_size + tensor_bytes - 1) // tensor_bytes
    free_bytes, _ = torch.cuda.mem_get_info()
    n_by_mem = max(1, int(free_bytes * 0.5) // tensor_bytes)
    return max(min_buffers, min(max_buffers, min(n_by_l2, n_by_mem)))


# Referenced-from: ~/quack/quack/bench/bench_utils.py (verbatim)
def _clone_l2_rotate_inputs(args, kwargs, n_buffers: int):
    arg_sets = []
    kwarg_sets = []
    for _ in range(n_buffers):
        arg_sets.append(tuple(a.clone() if isinstance(a, torch.Tensor) else a for a in args))
        kwarg_sets.append(
            {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
        )
    return arg_sets, kwarg_sets


# Referenced-from: ~/quack/quack/autotuner.py (verbatim; for envs without quack)
def _gpu_warmup(duration_ms=200):
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    target = duration_ms / 1000
    t0 = time.time()
    while time.time() - t0 < target:
        for _ in range(100):
            a = a @ a
        torch.cuda.synchronize()


def l2_rotate_count(*tensors):
    """Operand-set count for the rotate protocol: quack's pick, forced even (a
    captured round leaves double-buffered runners parity-invariant) and
    rank-MIN'd (a collective fn must never disagree on launch counts)."""
    n = max(4, _pick_l2_rotate_count(tensors, {}) & ~1)
    if dist.is_initialized():
        c = torch.tensor([n], device="cuda", dtype=torch.int64)
        dist.all_reduce(c, op=dist.ReduceOp.MIN)
        n = int(c.item())
    return n


def l2_rotate_arg_sets(*tensors):
    """quack's clone step: l2_rotate_count cloned copies of the L2-turnover operands."""
    return [tuple(t.clone() for t in tensors) for _ in range(l2_rotate_count(*tensors))]


def do_bench_l2_rotate_all(
    fn,
    arg_sets,
    kwarg_sets,
    extra_kwargs,
    warmup_target_ms: float = 200.0,
    n_timed_calls: int = 200,
    quantiles=None,
    capture=True,
):
    """Collective, rank-synced port of quack's _bench_cuda_graph_l2_rotate
    (the autotuner's L2-cold methodology; signature and call shape kept
    identical). ``fn`` is called as ``fn(*arg_sets[i], **kwarg_sets[i],
    **extra_kwargs)`` once per recorded launch; arms put their rotating
    non-symm inputs (B/D -- the L2-turnover bytes) in ``arg_sets`` and bake
    the rest (runner, symm workspaces; rewritten every call) into ``fn``'s
    closure. Collective deltas vs quack: rank barriers around
    probe/capture/replay, the warm-up count MAX-synced across ranks (a
    collective fn must never disagree on launch counts), and the result is
    the cross-rank max. The warm-up is ~warmup_target_ms of the arm's own
    load right before its window: with capture on, the graph is captured
    first and replayed for the warm-up, and the timed replay follows on the
    same stream with no host gap (an eager warm-up before the capture left
    the GPU idle through the capture and the clock recovered before the
    window). Keep len(arg_sets) even so a captured round leaves double-buffered
    runners parity-invariant. Arms whose compiled kernels bake a CUstream
    must be BUILT on the capturing stream. capture=False times an eager
    back-to-back round-robin in one event pair instead, for arms that
    CANNOT be captured: a host-blocking op, or an iteration that is itself
    a captured graph (replay can't record into an outer capture). Rotation
    and L2-cold semantics are identical, but the numbers are NOT: every
    iteration pays the replay/launch boundary (~3 us device-side, nsys
    2026-08-24) and flag-polling collectives absorb the peer's boundary
    into their wait phase — at decode sizes this near-doubles the reading.
    Prefer restructuring the arm so the whole window captures (per-call
    phase advance, parity-invariant rounds); callers passing capture=False
    must label the reported number as eager-timed."""
    n_sets = len(arg_sets)
    # Round timed-call count to a multiple of n_sets for even L2 turnover.
    rounds_timed = max(1, n_timed_calls // n_sets)
    total_timed_calls = rounds_timed * n_sets

    # A few priming launches so the probe doesn't catch first-launch
    # driver / kernel-load overhead.
    for _ in range(3):
        fn(*arg_sets[0], **kwarg_sets[0], **extra_kwargs)
    torch.cuda.synchronize()
    dist.barrier()

    # Probe a single launch to estimate per-call ms; size the warm-up to
    # ``warmup_target_ms`` of GPU work regardless of kernel cost.
    probe_start = torch.cuda.Event(enable_timing=True)
    probe_end = torch.cuda.Event(enable_timing=True)
    probe_start.record()
    fn(*arg_sets[0], **kwarg_sets[0], **extra_kwargs)
    probe_end.record()
    torch.cuda.synchronize()
    est_ms = max(probe_start.elapsed_time(probe_end), 1e-3)

    def rank_max(n):  # every rank must issue the same count: fn() may be a collective
        c = torch.tensor([n], device="cuda", dtype=torch.int64)
        dist.all_reduce(c, op=dist.ReduceOp.MAX)
        return int(c.item())

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    if capture:
        # Capture first, then warm up by replaying the graph and time the next
        # replay on the same stream with no host gap in between: the window opens
        # at the clock the warm-up settled to. (Warming eagerly and capturing
        # afterwards left the GPU idle for the capture and instantiation, tens
        # of ms, and the clock recovered before the window.)
        timed_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(timed_graph):
            for _ in range(rounds_timed):
                for i in range(n_sets):
                    fn(*arg_sets[i], **kwarg_sets[i], **extra_kwargs)
        torch.cuda.synchronize()
        n_warm_replays = rank_max(max(1, int(-(-warmup_target_ms // (est_ms * total_timed_calls)))))
        dist.barrier()
        for _ in range(n_warm_replays):
            timed_graph.replay()
        start_evt.record()
        timed_graph.replay()
        end_evt.record()
    else:
        n_warmup_calls = rank_max(max(50, int(warmup_target_ms / est_ms)))
        for i in range(n_warmup_calls):
            fn(*arg_sets[i % n_sets], **kwarg_sets[i % n_sets], **extra_kwargs)
        torch.cuda.synchronize()
        dist.barrier()
        start_evt.record()
        for _ in range(rounds_timed):
            for i in range(n_sets):
                fn(*arg_sets[i], **kwarg_sets[i], **extra_kwargs)
        end_evt.record()
    torch.cuda.synchronize()
    t = torch.tensor([start_evt.elapsed_time(end_evt) / total_timed_calls], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    ms = t.item()

    if quantiles:
        return [ms for _ in quantiles]
    return ms


# One arm contract for every harness, mirroring the autotuner's call shape:
# fn is invoked as fn(*arg_sets[i], **kwarg_sets[i], **extra_kwargs).
#   args      fn's canonical operand tuple; when arg_sets is None, bench_arm
#             clones it via _clone_l2_rotate_inputs (the L2-turnover payload).
#   arg_sets  set explicitly by arms whose operands are library-owned views
#             (repeated allocate_tensors sets etc.) — the declared deviation
#             from generic cloning; kwarg_sets defaults to [{}] per set.
#   configs   extra_kwargs dicts swept over (the autotuner's
#             config.all_kwargs()); None means one config, {}.
#   read_out  () -> fp32 output for checks (run set 0 first, see run0).
#   zero      reset output state between check runs.
#   free      teardown (symm/nvshmem buffers); None = nothing to free.
#   calls     iterations one fn() performs (a capture=False arm replaying a
#             pre-built graph does several); reported ms is divided by it.
Arm = namedtuple("Arm", ["fn", "args", "arg_sets", "kwarg_sets", "configs",
                         "read_out", "zero", "free", "calls", "keep"],
                 defaults=[(), None, None, None, None, None, None, 1, None])


def run0(arm, **extra_kwargs):
    """One eager call on set 0 / the canonical operands (for checks)."""
    if arm.arg_sets is not None:
        kw = arm.kwarg_sets[0] if arm.kwarg_sets else {}
        arm.fn(*arm.arg_sets[0], **kw, **extra_kwargs)
    else:
        arm.fn(*arm.args, **extra_kwargs)


def bench_arm(arm, capture=True):
    """Autotuner.benchmark's sequence for one Arm: _gpu_warmup, clone the
    L2-turnover operands unless the arm owns its sets, then one
    do_bench_l2_rotate_all window per config. Returns [(label, ms)] in
    config order; ms is the cross-rank max, divided by arm.calls."""
    _gpu_warmup()
    if arm.arg_sets is None:
        arg_sets, kwarg_sets = _clone_l2_rotate_inputs(
            arm.args, {}, l2_rotate_count(*arm.args))
    else:
        arg_sets = arm.arg_sets
        kwarg_sets = arm.kwarg_sets or [{}] * len(arg_sets)
    out = []
    for cfg in arm.configs or [{}]:
        if dist.is_initialized():
            dist.barrier()
        label = " ".join(f"{k}={v}" for k, v in cfg.items())
        out.append((label, do_bench_l2_rotate_all(arm.fn, arg_sets, kwarg_sets, cfg,
                                                  capture=capture) / arm.calls))
    return out


def try_capture(fn, args, kwargs=None):
    """Collective probe: True when fn captures into a CUDA graph and the graph replays
    on EVERY rank (a library doing host-side work per call fails here); all ranks
    agree on the verdict, so a mixed outcome never splits a captured collective."""
    kwargs = kwargs or {}
    ok = True
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fn(*args, **kwargs)  # warm (lazy inits must not land in the graph)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn(*args, **kwargs)
        g.replay()
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001
        print(f"  capture failed: {type(e).__name__}: {str(e)[:120]}", flush=True)
        ok = False
    if dist.is_initialized():
        flag = torch.tensor([int(ok)], device="cuda")
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        ok = bool(flag.item())
    return ok
