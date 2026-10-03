"""The EP benches' harness (benchmarks/benchmark_ep_forward*.py): model shapes, the Kernel record, SM partitions,
the timing protocol with its checks, CUDA-graph capture, and the fused-combine GEMM2's compile glue.

The protocol is fixed, not flags, so two runs are comparable: per arm and block, WARM_S of warm-up at the arm's own
load right before a WINDOW_S timed block, REPS blocks with the arm order rotated per block, the block timed in
SAMPLES chunks. An arm is unstable when a block median is off the arm's median by more than SPREAD, or a block's
p90 / median exceeds TAIL (co-tenant burst or straggler rank)."""
import contextlib
import copy
import statistics
import time
from collections import namedtuple

import cutlass
import cutlass.cute as cute
import torch
import torch.distributed as dist
from cuda.bindings import driver
from cutlass import Int32, Int64
from quack.compile_utils import make_fake_tensor
from quack.cute_dsl_utils import get_device_capacity, get_max_active_clusters
from quack.gemm_default_epi import GemmDefaultEpiMixin, GemmDefaultSm90, GemmDefaultSm100
from quack.gemm_sm100_ep_intra_combine import GemmSm100 as GemmSm100IntraCombine
from quack.gemm_tvm_ffi_utils import (
    make_fake_gemm_tensors,
    make_fake_scheduler_args,
    make_fake_varlen_args,
    make_scheduler_args,
    make_varlen_args,
)
from quack.rounding import RoundingMode
from sonicmoe.functional.distributed.collectives import _check
from sonicmoe.functional.reduction_over_k_gather import token_gather_and_sum_varlen_K_triton

ModelConfig = namedtuple("ModelConfig", ["H", "I", "E", "topk"])  # hidden, MoE intermediate, experts, top-k
MODEL_CONFIGS = {
    "dsv3": ModelConfig(7168, 2048, 256, 8),
    "dsv3_e128": ModelConfig(7168, 2048, 128, 8),  # dsv3 shapes, half the experts: varies experts/rank alone
    "kimi_k2": ModelConfig(7168, 2048, 384, 8),
    "qwen3_235b": ModelConfig(4096, 1536, 128, 8),
    "mixtral_8x22b": ModelConfig(6144, 16384, 8, 2),
    "olmoe": ModelConfig(2048, 1024, 64, 8),
}
_NOOP = lambda: None  # noqa: E731

# run/free: timed call + cleanup. read_out: () -> the output to compare, a tensor or a tuple of tensors. zero: reset
# it. ref: sonic's receive rows (n_recv, H) -> what read_out must equal (None: timing-only arm). label: suffix for
# the report line. halves: for an arm made of two parts, the list of (start, end, start, end) events run() records
# per call.
Kernel = namedtuple("Kernel", ["run", "free", "read_out", "zero", "ref", "label", "halves"], defaults=[None, None, None, "", None])


class SMStream:
    """Torch stream confined to ~num_sms SMs via a green context; the whole GPU if num_sms >= total.
    Copied from sonic-moe PR #54 benchmarks/moe_sm_tflops.py (SMStream); the count rounds up to the
    device's co-scheduling alignment, so read .sms for what was granted."""

    def __init__(self, device_index: int, num_sms: int):
        _check(driver.cuInit(0))
        self.device_index = device_index
        self.dev = _check(driver.cuDeviceGet(device_index))
        res = _check(driver.cuDeviceGetDevResource(self.dev, driver.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM))
        self.total = int(res.sm.smCount)
        self.rest = None  # SMs the split left over (see complement)
        if num_sms >= self.total:
            self.sms, self.stream, self._gctx = self.total, None, None
            return
        align = int(res.sm.smCoscheduledAlignment) or 8
        n = max(align, ((num_sms + align - 1) // align) * align)
        groups, _nb, self.rest = _check(driver.cuDevSmResourceSplitByCount(1, res, 0, n))
        self.sms, self.stream, self._gctx = self._green(groups[0])

    def _green(self, part):
        desc = _check(driver.cuDevResourceGenerateDesc([part], 1))
        gctx = _check(driver.cuGreenCtxCreate(desc, self.dev, driver.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM))
        cu_stream = _check(driver.cuGreenCtxStreamCreate(gctx, driver.CUstream_flags.CU_STREAM_NON_BLOCKING, 0))
        return int(part.sm.smCount), torch.cuda.ExternalStream(int(cu_stream), device=torch.device("cuda", self.device_index)), gctx

    def complement(self):
        """Stream on the SMs this partition left over, disjoint from it (two kernels co-run, one on each)."""
        other = copy.copy(self)
        other.sms, other.stream, other._gctx = self._green(self.rest)
        other.rest = None
        return other

    def ctx(self):
        return torch.cuda.stream(self.stream) if self.stream is not None else contextlib.nullcontext()


WARM_S, WINDOW_S, REPS, SAMPLES = 0.2, 0.5, 5, 100
SPREAD, TAIL = 0.03, 1.10


def bench_all(fns, hdl):
    """sonic's EP timing (benchmarks/distributed/bench-ep-nvlink.py bench_fn) under the protocol above. Every call
    is {symm-mem barrier, fn}. A 20-call probe per arm (host-timed, rank-MAX'd, after 2 untimed calls) sizes its
    warm-up and window as fixed call counts equal on every rank (a time-based loop would desync the barriers). Returns
    {name: [[chunk ms, ...] per block]} per rank; see arm_stats."""
    counts = {}
    for k, fn in fns.items():
        for _ in range(2):  # an arm's first calls may compile or autotune (triton): keep them out of the probe
            hdl.barrier()
            fn()
        dist.barrier()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            hdl.barrier()
            fn()
        torch.cuda.synchronize()
        per_call = torch.tensor([(time.perf_counter() - t0) / 20], device="cuda")
        dist.all_reduce(per_call, op=dist.ReduceOp.MAX)
        s = per_call.item()
        counts[k] = (max(10, int(WARM_S / s)), max(20, int(WINDOW_S / s)))
    ms = {k: [] for k in fns}
    names = list(fns)
    for rep in range(REPS):
        shift = rep % len(names)  # rotate the order per block: no arm always inherits the same predecessor's clocks
        for k in names[shift:] + names[:shift]:
            fn, (warm, iters) = fns[k], counts[k]
            for _ in range(warm):
                hdl.barrier()
                fn()
            dist.barrier()
            chunk = max(1, iters // SAMPLES)
            events = []
            for i in range(iters):
                if i % chunk == 0:
                    events.append(torch.cuda.Event(enable_timing=True))
                    events[-1].record()
                hdl.barrier()
                fn()
            events.append(torch.cuda.Event(enable_timing=True))
            events[-1].record()
            torch.cuda.synchronize()
            bounds = list(range(0, iters, chunk)) + [iters]
            ms[k].append([events[j].elapsed_time(events[j + 1]) / (bounds[j + 1] - bounds[j]) for j in range(len(bounds) - 1)])
    return ms


def arm_stats(blocks):
    """(median, low, high, unstable) of an arm: the median of its block medians and their range, and whether a
    block median is off by more than SPREAD or a block's p90 / median exceeds TAIL."""
    meds = [statistics.median(b) for b in blocks]
    med = statistics.median(meds)
    tails = [sorted(b)[int(0.9 * (len(b) - 1))] / statistics.median(b) for b in blocks]
    unstable = max(abs(m - med) for m in meds) > SPREAD * med or max(tails) > TAIL
    return med, min(meds), max(meds), unstable


def _tup(v):
    return v if isinstance(v, tuple) else (v,)


def _run_synced(kern):
    """Zero, barrier, run, sync, barrier: one fresh output, no rank still pushing into a buffer being zeroed."""
    kern.zero()
    torch.cuda.synchronize()
    dist.barrier()
    kern.run()
    torch.cuda.synchronize()
    dist.barrier()


def _compare(a, b):
    """(bitwise equal, max |delta|) over matching tuples of tensors."""
    pairs = list(zip(_tup(a), _tup(b)))
    return all(torch.equal(x, y) for x, y in pairs), max((x.float() - y.float()).abs().max().item() for x, y in pairs)


def ref_check(kern, rows):
    """(bitwise equal, max |delta|) of kern's fresh output vs its reference from sonic's rows."""
    _run_synced(kern)
    return _compare(kern.read_out(), kern.ref(rows))


def r2r_check(kern):
    """Run twice; (bit-identical, max |delta|) of the two outputs."""
    _run_synced(kern)
    first = tuple(t.clone() for t in _tup(kern.read_out()))
    _run_synced(kern)
    return _compare(kern.read_out(), first)


def n_recv(meta, rank):
    """Rows this rank receives (sonic's dedup: each hit token once per receiving rank)."""
    return int(meta["pair_count"][:, rank].sum())


def capture_graph(fn, warmup=3):
    """Capture fn's launches into a CUDA graph, returning replay. fn must launch on torch's current stream: tvm-ffi
    env stream with a live torch tensor arg, or a CUstream re-fetched inside fn."""
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


# --- GEMM2 (the fused-combine GEMM) compile glue -------------------------------------------------------------------
# The M cluster is 256 rows either way (the walk's BLOCK_ROWS): 2-CTA 256x256 on SM100, 128x256 x cluster 2 on H100.
# 256x256 (two tmem accumulator stages) over 256x512 (one stage, epilogue exposed): GEMM2 alone -5 %, intra -2 %.
SM100 = get_device_capacity(torch.device("cuda"))[0] >= 10
TILE, CLUSTER = ((256, 256), (2, 1)) if SM100 else ((128, 256), (2, 1))


class GemmDefaultSm100IntraCombine(GemmDefaultEpiMixin, GemmSm100IntraCombine):
    pass


def compile_gemm2(dtype, tile=TILE, cluster=CLUSTER, intra_push=0, epilogue_store=False):
    """(fn, block_rows, cta_tile_n): the GEMM compiled against symbolic shapes (varlen M, combine scheduler
    argument, static persistence); block_rows = the M cluster (CTA tile M x cluster M), the unit the push waits on.
    intra_push = the hidden size: the intra-SM kernel (transfer warps pushing rows of that width, five walk tensors
    after the stream); epilogue_store: its variant whose epilogue warps store the rows into the slots themselves
    (after the stream: the slot base table and the slots per rank)."""
    if intra_push:
        assert SM100, "the intra-SM kernel is SM100 only"
        gemm = GemmDefaultSm100IntraCombine(
            cutlass.Float32, dtype, tile, (*cluster, 1), use_clc_persistence=False, intra_combine_push=intra_push,
            combine_epilogue_store=epilogue_store,
        )
        i32 = lambda: make_fake_tensor(Int32, (cute.sym_int(),), leading_dim=0, divisibility=1)
        i64 = lambda: make_fake_tensor(Int64, (cute.sym_int(),), leading_dim=0, divisibility=1)
        walk_fakes = (None, None, None, None, None, i64(), Int32(0)) if epilogue_store else (i32(), i64(), i64(), i64(), i32())
    elif SM100:
        gemm = GemmDefaultSm100(cutlass.Float32, dtype, tile, (*cluster, 1), use_clc_persistence=False)
        walk_fakes = ()
    else:  # H100: quack's SM90 kernel publishes the same combine flags from its epilogue
        gemm = GemmDefaultSm90(cutlass.Float32, dtype, tile, (*cluster, 1))
        walk_fakes = ()
    gemm.b_transposed = gemm.a_transposed = gemm.cd_transposed = False  # as quack's compile_gemm_kernel sets
    gemm.cd_packed = None
    mA, mB, mD, mC, m, n, k, l = make_fake_gemm_tensors(dtype, dtype, dtype, None, "k", "k", "n", "n", varlen_m=True)
    epi = GemmDefaultEpiMixin.EpilogueArguments(add_to_output=False, rounding_mode=RoundingMode.RN)
    sched = make_fake_scheduler_args(False, False, l, has_combine=True)
    varlen = make_fake_varlen_args(True, False, False, m)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    fn = cute.compile(
        gemm, mA, mB, mD, mC, epi, sched, varlen, stream, None, None, *walk_fakes, options="--enable-tvm-ffi"
    )
    return fn, gemm.cta_tile_shape_mnk[0] * cluster[0], gemm.cta_tile_shape_mnk[1]


def gemm2_args(cu_seqlens_m, flags, slot_of_row, first_src_row, cluster=CLUSTER, sms=None):
    """(epilogue, scheduler, varlen) launch arguments; flags (E_local, max M clusters) int32; slot_of_row (rows,)
    int32 = each D row's global slot id and first_src_row (1,) int32 = the peer after this rank's first slot: the
    scheduler starts each expert's clusters there (the push's destination rotation); sms = the SMs the GEMM runs on
    (its resident clusters; default: the whole device)."""
    epi_rt = GemmDefaultEpiMixin.EpilogueArguments(add_to_output=None, rounding_mode=None)
    size = cluster[0] * cluster[1]
    clusters = get_max_active_clusters(size, get_device_capacity(flags.device)) if sms is None else sms // size
    sched_rt = make_scheduler_args(clusters, 8, None, combine_args=(flags, slot_of_row, first_src_row))
    return epi_rt, sched_rt, make_varlen_args(cu_seqlens_m, None, None)


def _gather_and_sum(buffers, scores, T_local, K, H):
    """(out, reduce): sonic's gather-and-sum over the (T_local, K, H) slots with the scores, into out (T_local, H)."""
    dev = scores.device
    out = torch.empty(T_local, H, dtype=torch.bfloat16, device=dev)
    slots, w = buffers.slots.view(T_local * K, H), scores.view(-1)
    perm = torch.arange(T_local * K, dtype=torch.int32, device=dev)
    offset = torch.arange(0, T_local * K + 1, K, dtype=torch.int32, device=dev)
    return out, lambda: token_gather_and_sum_varlen_K_triton(slots, w, out, perm, offset, T_local, K, H, False)
