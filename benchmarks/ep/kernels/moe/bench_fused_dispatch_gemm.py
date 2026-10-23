"""Run fused dispatch + GEMM1 baselines through one interface (bench_dispatch.py's protocol and structure).

GEMM1 h = a[a_idx] @ w1[expert]^T over the rows sonic's rank-dedup dispatch lands (sonic's receive layout and
metadata, as in bench_dispatch.py), with the same quack GEMM kernel in every arm (quack branch ep's varlen gather-A
GEMM with the dispatch scheduler: expert-major order, source order inside, each tile gated on flags[expert, cid_m]
reaching its row count; compiled below like fused_gemm_combine/gemm_combine_epi_push.py, fake tensors at compile,
torch tensors at launch), so the fusion is the only variable. The tile config comes from quack's autotuner run
over this kernel at this shape (gemm1_tuned: rank 0 sweeps, disk-cached by shape under quack's cache, the winner
broadcast); the walk's M cluster is derived from the compiled kernel, so there is one source of truth for it. The
tuner's single-window pick can be contention-poisoned (and is disk-cached), and a kernel sharing SMs or A traffic
with the transfer can prefer a bigger tile than the bare GEMM, so every arm also runs on the 2-CTA fallbacks
256x512 and 256x256, each as its own arm `<name>@tile`; the report marks the best per baseline. Shape from --model,
bf16, uniform routing; metadata built once outside the timed region.

  quack_inter  our pull (dispatch/pull.py, inter mode, over the device walk) on --comm-sms SMs alongside the
               gated GEMM on the rest, two disjoint green-context partitions; the walk build (whole GPU) before the fork
  quack_intra  the GEMM carrying 4 transfer warps per CTA that pull the walk themselves
               (fused_gemm_dispatch/gemm_sm100_intra_dispatch_pull.py), on all SMs, on the same tile config
  quack_inter_swap  as quack_inter with bench_forward's swap-AB varlen-N gated GEMM1 (weights on M, tokens on N in
               even-split blocks of the swap tile, B gathered from the pool: SM100 only); its output is the SwiGLU
               postact, so its check is the tolerance one (bench_combine's)
  quack_intra_swap  bench_forward's prologue kernel (routing metadata + the walk, one launch) then the swap-AB GEMM1
               whose producer warps pull the walk into rows packed per expert (gemm_sm100/sm90_intra_dispatch_pull.py),
               on all SMs, SM100 and SM90
  sonic        no fusion: sonic's dispatch on the whole GPU, then the GEMM with its flags pre-filled (never waits)
  legs         the bound legs alone, each back-to-back: `legs/dispatch` (sonic's dispatch) and `legs/gemm` (the GEMM
               on all SMs, flags pre-filled); sum − max(legs) = the gap an overlap can recover

--presatisfied adds a `<name>/presat` arm per baseline: the same arrangement (same streams, partitions and power
budget) with the flags held at target, so the GEMM never waits and the transfer still runs. For quack_inter each
half is bracketed by its own events on its own stream, giving t_comm and t_gemm under the shared budget and the
bound = max(t_gemm, t_comm); quack_intra's halves live inside one kernel, so its pre-satisfied time is the bound
itself; sonic's halves are sequential. Timing the halves this way, rather than each leg alone, keeps the power
throttling of the co-running halves in the bound.

Each factory returns {arm name: Kernel} (bench_dispatch.Kernel: run / free / read_out / zero / ref / label / halves).
A fused arm's read_out is (h, pool rows, flags) and its ref maps (h_ref, sonic's rows) to that, so --ref_check /
--r2r_check work generically; zero() clears the pool too, so an early read would show. A presat arm's GEMM may read
rows before they land, so it is timing-only.

Usage:
    PYTHONPATH=~/quack:~/sonic-moe-pr54 torchrun --nproc_per_node=8 --standalone bench_fused_dispatch_gemm.py \\
        quack_inter quack_intra sonic --comm-sms 8 --presatisfied --ref_check
"""
import os

os.environ.setdefault("TRITON_CACHE_AUTOTUNING", "1")  # sonic's triton autotune choices persist across launches

import argparse
import glob
import importlib.util
import statistics
import sys
import time
from dataclasses import replace

import cutlass
import cutlass.cute as cute
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # the kernels package root
from kernels.misc import nccl_env  # noqa: E402, F401  (NCCL channel floor for every reference)
from quack.autotuner import AutotuneConfig, autotune
from quack.cute_dsl_utils import get_device_capacity, get_max_active_clusters, torch2cute_dtype_map
from quack.gemm_config import config_supports, get_all_configs
from quack.gemm_default_epi import GemmDefaultEpiMixin, GemmDefaultSm90, GemmDefaultSm100
from quack.gemm_tvm_ffi_utils import (
    make_fake_gemm_tensors,
    make_fake_scheduler_args,
    make_fake_varlen_args,
    make_scheduler_args,
    make_varlen_args,
)
from quack.rounding import RoundingMode
from sonicmoe.distributed_utils import DispatchMode, SymmMemManager
from sonicmoe.functional.distributed import build_rank_dedup_a_idx, compute_dispatch_metadata, rank_dedup_dispatch_triton
from sonicmoe.functional.ep import _ag_routing_decision, _build_consumer_metadata

import bench_forward as bf  # the swap-AB kernels (compile_gemm1_gated_swap, interleave_gate_up, swiglu_cat)
from bench_combine import ref_check as ref_check_tol
from bench_dispatch import MODEL_CONFIGS, REPS, WARM_S, WINDOW_S, Kernel, SMStream, _NOOP, arm_stats, bench_all, n_recv, r2r_check, ref_check
from combine.combine_push import make_combine_buffers
from dispatch.prologue import make_prologue
from dispatch.pull import make_dispatch_pull
from dispatch.walk import flag_targets, make_dispatch_walk
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../all"))
from gemm_dispatch_pull import DispatchArguments, DispatchPull, fake_dispatch_arguments  # noqa: E402
from gemm_sm100 import GemmSm100 as GemmSm100Comm  # noqa: E402  # quack main's kernel + the comm object surface


class GemmDefaultSm100Comm(GemmDefaultEpiMixin, GemmSm100Comm):
    pass


_compiled = {}


def compile_gemm1(dtype, config, intra_pull=0, smem_cut=0):
    """(fn, block_rows): the GEMM compiled once per (dtype, config's tile and cluster, intra_pull, smem_cut) with
    fake tensors (varlen M, gather A, dispatch scheduler), and its M cluster (CTA tile M x cluster M: the unit one
    flag gates, what the walk is built on). intra_pull = the hidden size: the intra-SM kernel (transfer warps pulling
    rows of that width, the walk's WalkArguments after the SF tensors); smem_cut: bytes taken off the smem budget the kernel
    plans its pipeline stages with (read once, at compile)."""
    tile, cluster = (config.tile_m, config.tile_n), (config.cluster_m, config.cluster_n, 1)
    key = (dtype, tile, cluster, config.pingpong, intra_pull, smem_cut)
    if key not in _compiled:
        sm100 = get_device_capacity(torch.device("cuda"))[0] >= 10
        if intra_pull:
            assert sm100, "the intra-SM kernel is SM100 only"
            gemm = GemmDefaultSm100Comm(
                cutlass.Float32, dtype, tile, cluster, gather_A=True, use_clc_persistence=False,
                comm=DispatchPull(pull_bytes=intra_pull * dtype.width // 8),
            )
            walk_fakes = (fake_dispatch_arguments(dtype),)
        elif sm100:
            gemm = GemmDefaultSm100(cutlass.Float32, dtype, tile, cluster, gather_A=True, use_clc_persistence=False)
            walk_fakes = ()
        else:  # H100: quack's SM90 kernel carries the same dispatch gate in its AB-load warps
            gemm = GemmDefaultSm90(cutlass.Float32, dtype, tile, cluster, pingpong=config.pingpong, gather_A=True)
            walk_fakes = ()
        gemm.b_transposed = gemm.a_transposed = gemm.cd_transposed = False  # as quack's compile_gemm_kernel sets
        gemm.cd_packed = None
        mA, mB, mD, mC, m, n, k, l = make_fake_gemm_tensors(
            dtype, dtype, dtype, None, "k", "k", "n", "n", varlen_m=True, gather_A=True
        )
        epi = GemmDefaultEpiMixin.EpilogueArguments(add_to_output=False, rounding_mode=RoundingMode.RN)
        sched = make_fake_scheduler_args(False, False, l, has_dispatch=True)
        varlen = make_fake_varlen_args(True, False, True, m)
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        full = cutlass.utils.get_smem_capacity_in_bytes
        cutlass.utils.get_smem_capacity_in_bytes = lambda arch: full(arch) - smem_cut
        try:
            fn = cute.compile(
                gemm, mA, mB, mD, mC, epi, sched, varlen, stream, None, None, *walk_fakes, options="--enable-tvm-ffi"
            )
        finally:
            cutlass.utils.get_smem_capacity_in_bytes = full
        _compiled[key] = (fn, gemm.cta_tile_shape_mnk[0] * gemm.cluster_shape_mnk[0])
    return _compiled[key]


def gemm1_args(a, cu_seqlens_m, a_idx, flags, first_src_row, config, sms=None):
    """(epilogue, scheduler, varlen) runtime arguments; flags (E_local, max M clusters) int32 and first_src_row (1,)
    int32 as make_dispatch_walk's; sms = SM count of the partition the launch runs on (persistent grid = its
    clusters; default: the whole device). The intra-SM kernel needs every CTA resident: the walk is strided over
    all of them."""
    epi_rt = GemmDefaultEpiMixin.EpilogueArguments(add_to_output=None, rounding_mode=None)
    cluster_size = config.cluster_m * config.cluster_n
    clusters = get_max_active_clusters(cluster_size, get_device_capacity(a.device)) if sms is None else sms // cluster_size
    sched_rt = make_scheduler_args(clusters, config.max_swizzle_size, None, dispatch_args=(flags, first_src_row))
    return epi_rt, sched_rt, make_varlen_args(cu_seqlens_m, None, a_idx)


def _gemm1_candidates(configs, named_args, **kwargs):
    """The autotuner's candidates: quack's configs for this device that our kernels take (gather A + varlen M:
    cluster_n 1, no swap_ab; static persistence, the intra-SM kernel's requirement; no TMA gather; no split-K):
    tile_m 128 / 256, tile_n 16 .. 512, cluster_m 1 / 2, 25 on B300."""
    cap = get_device_capacity(named_args["a"].device)[0]

    def ok(c):
        return (
            c.device_capacity == cap and config_supports(c, gather_A=True, varlen_m=True)
            and not c.is_dynamic_persistent and not c.use_tma_gather and not c.swap_ab and c.split_k in (None, 1)
        )

    return [ac for ac in configs if ok(ac.kwargs["config"])]


def _do_bench(fn, quantiles):
    """The autotuner's per-config timing under the protocol: a probe sizes WARM_S of warm-up at the config's own
    load and one WINDOW_S window; a single-GPU GEMM, so no barriers. Returns ms per call, one value per quantile."""
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    per_call = (time.perf_counter() - t0) / 5
    for _ in range(max(10, int(WARM_S / per_call))):
        fn()
    iters = max(20, int(WINDOW_S / per_call))
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return [start.elapsed_time(end) / iters] * len(quantiles)


_targets = {}  # per M cluster: the pre-filled flags for the shape being tuned


@autotune(
    configs=[AutotuneConfig(config=c) for c in get_all_configs()],
    prune_configs_by={"early_config_prune": _gemm1_candidates},
    do_bench=_do_bench,
)
def gemm1_tuned(a, w1, h, a_idx, cu_seqlens_m, first_src_row, *, config):
    """One launch of the GEMM on landed rows with `config` and its flags pre-filled; quack's autotuner picks the
    config for these tensors' shapes (see _gemm1_candidates, _do_bench) and reports it as .best_config."""
    fn, block_rows = compile_gemm1(torch2cute_dtype_map[a.dtype], config)
    if block_rows not in _targets:
        cols = -(-a.shape[0] // block_rows)  # an expert reads each pool row at most once: its M clusters <= the pool's
        _targets[block_rows] = flag_targets(cu_seqlens_m, torch.zeros(cu_seqlens_m.numel() - 1, cols, dtype=torch.int32, device=a.device), block_rows)
    rt = gemm1_args(a, cu_seqlens_m, a_idx, _targets[block_rows], first_src_row, config)
    fn(a, w1, h, None, *rt, None, None)


def _sonic_dispatch(ws, meta, recv, rank, K):
    """sonic's rank-dedup dispatch into recv, nullary: the call is sonic-moe PR #54 sonicmoe/functional/ep.py:95
    _do_dispatch's RANK_DEDUP branch, without the backward / autograd, AG / A2A and inter-node paths."""

    def dispatch():
        rank_dedup_dispatch_triton(
            x_symm=ws.x_symm,
            dst_rank_flat=meta["dst_rank_flat"],
            pair_present_mask=meta["pair_present_mask"],
            rank_dedup_recv_pos=meta["rank_dedup_recv_pos"],
            recv_packed=recv,
            K=K,
            group=ws.ep_group,
            peer_bufs=ws.x_peer_bufs,
            my_rank=rank,
        )

    return dispatch


def _walk(model, topk_idx_g, meta, ws, rank, W, block_rows):
    """The device walk for our pull and the GEMM's scheduler: (build, walk, a_idx, cu, total_m)."""
    cu = meta["expert_frequency_offset"]
    total_m = int(cu[model.E // W])
    a_idx = meta["x_idx_expanded_remap_for_rank_dedup"][:total_m]
    build, walk = make_dispatch_walk(
        topk_idx_g, meta["s_scatter_idx"], meta["recv_pos"], meta["pair_offset"][(rank + 1) % W, rank : rank + 1], rank, a_idx, cu, ws.x_peer_bufs,
        block_rows,
    )
    return build, walk, a_idx, cu, total_m


# Every factory: (model, x, topk_idx_g, meta, ws, w1, config, rank, W, T_local, args) -> {arm: Kernel};
# w1 (E_local, 2I, H); config: the tuned quack GemmConfig.
def _sonic(model, x, topk_idx_g, meta, ws, w1, config, rank, W, T_local, args):
    """No fusion: sonic's dispatch into recv on the whole GPU, then the GEMM on recv with its flags pre-filled."""
    dev, dtype = x.device, torch2cute_dtype_map[x.dtype]
    cu = meta["expert_frequency_offset"]
    total_m = int(cu[model.E // W])
    a_idx = meta["x_idx_expanded_remap_for_rank_dedup"][:total_m]
    n = n_recv(meta, rank)
    recv = torch.empty(W * T_local, model.H, dtype=torch.bfloat16, device=dev)
    h = torch.empty(total_m, 2 * model.I, dtype=torch.bfloat16, device=dev)
    fn, block_rows = compile_gemm1(dtype, config)
    flags = flag_targets(cu, torch.zeros(model.E // W, -(-W * T_local // block_rows), dtype=torch.int32, device=dev), block_rows)
    first_src_row = meta["pair_offset"][(rank + 1) % W, rank : rank + 1]
    rt = gemm1_args(recv, cu, a_idx, flags, first_src_row, config)
    gemm = lambda: fn(recv, w1, h, None, *rt, None, None)
    dispatch = _sonic_dispatch(ws, meta, recv, rank, model.topk)

    def run():
        dispatch()
        gemm()

    arms = {"sonic": Kernel(run, _NOOP, lambda: (h, recv[:n]), lambda: (h.zero_(), recv.zero_()), lambda r: r)}
    if args.presatisfied:  # nothing to pre-satisfy: the same sequence, each half bracketed by events
        halves = []

        def presat():
            ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            ev[0].record()
            dispatch()
            ev[1].record()
            ev[2].record()
            gemm()
            ev[3].record()
            halves.append(ev)

        arms["sonic/presat"] = Kernel(presat, _NOOP, halves=halves)
    return arms


def _fork_arm(full, comm, rest, build, pull, gemm):
    """fork(prologue, halves=None): build + prologue on the whole GPU (stream full), then the pull on comm and the
    GEMM on rest, joined; halves: record each half's events into it."""

    def fork(prologue, halves=None):
        cur = torch.cuda.current_stream()
        full.wait_stream(cur)
        with torch.cuda.stream(full):
            build()
            prologue()
        comm.stream.wait_stream(full)
        rest.stream.wait_stream(full)
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)] if halves is not None else None
        with comm.ctx():
            if ev:
                ev[0].record()
            pull()
            if ev:
                ev[1].record()
        with rest.ctx():
            if ev:
                ev[2].record()
            gemm()
            if ev:
                ev[3].record()
        cur.wait_stream(comm.stream)
        cur.wait_stream(rest.stream)
        if ev:
            halves.append(ev)

    return fork


def _quack_inter(model, x, topk_idx_g, meta, ws, w1, config, rank, W, T_local, args):
    """Inter-SM: the walk build on the whole GPU, then the pull (inter mode) on the --comm-sms partition and the gated
    GEMM on its complement, forked on the two partitions' streams and joined."""
    dev, dtype = x.device, torch2cute_dtype_map[x.dtype]
    fn, block_rows = compile_gemm1(dtype, config)
    build, walk, a_idx, cu, total_m = _walk(model, topk_idx_g, meta, ws, rank, W, block_rows)
    n = n_recv(meta, rank)
    comm = SMStream(dev.index, args.comm_sms)
    assert comm.stream is not None, "--comm-sms must leave SMs for the GEMM"
    rest = comm.complement()
    full = torch.cuda.Stream(device=dev)  # the build: metadata work on the whole GPU
    run_pull, pool = make_dispatch_pull(ws.x_peer_bufs, rank, "inter", comm.sms)
    pull = run_pull(walk, zero_counters=False)  # the fork's prologue owns the flags: zeroed, or held at target
    h = torch.empty(total_m, 2 * model.I, dtype=torch.bfloat16, device=dev)
    rt = gemm1_args(pool, cu, a_idx, walk.flags, walk.first_src_row, config, sms=rest.sms)
    gemm = lambda: fn(pool, w1, h, None, *rt, None, None)
    fork = _fork_arm(full, comm, rest, build, pull, gemm)
    fused = lambda: (h, pool[:n], walk.flags)
    zero = lambda: (h.zero_(), pool.zero_(), walk.flags.zero_())
    label = f" (comm {comm.sms}, gemm {rest.sms} SMs)"
    # the fused arm zeroes the flags before the fork: the GEMM must not see last call's full flags
    arms = {"quack_inter": Kernel(lambda: fork(walk.flags.zero_), _NOOP, fused, zero, lambda r: (r[0], r[1], walk.targets), label=label)}
    if args.presatisfied:  # flags held at target before the fork: the GEMM never waits, the pull still runs
        halves = []
        arms["quack_inter/presat"] = Kernel(lambda: fork(lambda: walk.flags.copy_(walk.targets), halves), _NOOP, label=label, halves=halves)
    return arms


def _quack_intra(model, x, topk_idx_g, meta, ws, w1, config, rank, W, T_local, args):
    """Intra-SM: the walk build, then the GEMM whose CTAs carry 4 transfer warps pulling the walk, on all SMs. The
    pre-satisfied arm is the same kernel with the flags held at target (its halves are one kernel)."""
    dev, dtype = x.device, torch2cute_dtype_map[x.dtype]
    fn, block_rows = compile_gemm1(dtype, config, intra_pull=model.H)
    build, walk, a_idx, cu, total_m = _walk(model, topk_idx_g, meta, ws, rank, W, block_rows)
    n = n_recv(meta, rank)
    pool = torch.empty(W * T_local, model.H, dtype=torch.bfloat16, device=dev)
    h = torch.empty(total_m, 2 * model.I, dtype=torch.bfloat16, device=dev)
    rt = gemm1_args(pool, cu, a_idx, walk.flags, walk.first_src_row, config)
    launch = lambda: fn(pool, w1, h, None, *rt, None, None, DispatchArguments(walk.args, pool))

    def run():
        build()
        walk.flags.zero_()
        launch()

    def presat():
        build()
        walk.flags.copy_(walk.targets)
        launch()

    fused = lambda: (h, pool[:n], walk.flags)
    zero = lambda: (h.zero_(), pool.zero_(), walk.flags.zero_())
    arms = {"quack_intra": Kernel(run, _NOOP, fused, zero, lambda r: (r[0], r[1], walk.targets))}
    if args.presatisfied:
        arms["quack_intra/presat"] = Kernel(presat, _NOOP)
    return arms


def _swap_tag(block_rows):
    return f"swap tile {256 if bf.SM100 else 128}x{block_rows}"


def _quack_inter_swap(model, x, topk_idx_g, meta, ws, w1, config, rank, W, T_local, args):
    """Inter-SM, swap-AB: the walk build (even-split blocks of the swap tile) on the whole GPU, then the pull (inter
    mode) on the --comm-sms partition and bench_forward's gated swap-AB varlen-N GEMM1 (tokens on N, B gathered from
    the pool by a_idx, dynamic persistence) on its complement, forked and joined. The tile is the swap rule's, not
    the tuned config's. SM100 only (gather_B)."""
    assert bf.SM100, "quack_inter_swap gathers B from the pool (gather_B): SM100 only"
    dev, dtype = x.device, torch2cute_dtype_map[x.dtype]
    fn, Cls, block_rows = bf.compile_gemm1_gated_swap(dtype, model, W, T_local)
    cu = meta["expert_frequency_offset"]
    total_m = int(cu[model.E // W])
    a_idx = meta["x_idx_expanded_remap_for_rank_dedup"][:total_m]
    build, walk = make_dispatch_walk(
        topk_idx_g, meta["s_scatter_idx"], meta["recv_pos"], meta["pair_offset"][(rank + 1) % W, rank : rank + 1], rank, a_idx, cu,
        ws.x_peer_bufs, block_rows, even_split=True,
    )
    n = n_recv(meta, rank)
    comm = SMStream(dev.index, args.comm_sms)
    assert comm.stream is not None, "--comm-sms must leave SMs for the GEMM"
    rest = comm.complement()
    full = torch.cuda.Stream(device=dev)  # the build: metadata work on the whole GPU
    run_pull, pool = make_dispatch_pull(ws.x_peer_bufs, rank, "inter", comm.sms)
    pull = run_pull(walk, zero_counters=False)
    w1_il8 = bf.interleave_gate_up(w1)
    a = torch.empty(total_m, model.I, dtype=torch.bfloat16, device=dev)  # the SwiGLU postact (the kernel writes a.T)
    sem = torch.zeros(1, dtype=torch.int32, device=dev)  # dynamic persistence's tile counter, zeroed per call
    sched = make_scheduler_args(rest.sms // (bf.CLUSTER[0] * bf.CLUSTER[1]), 8, sem, dispatch_args=(walk.flags, walk.first_src_row))
    rt = (Cls.EpilogueArguments(mAuxOut=a.T), sched, make_varlen_args(None, None, None, cu_seqlens_n=cu, B_idx=a_idx))
    gemm = lambda: fn(w1_il8, pool, None, None, *rt, None, None)
    fork = _fork_arm(full, comm, rest, build, pull, gemm)
    fused = lambda: (a, pool[:n], walk.flags)
    zero = lambda: (a.zero_(), pool.zero_(), walk.flags.zero_())
    ref = lambda r: (bf.swiglu_cat(r[0]), r[1], walk.targets)
    label = f" ({_swap_tag(block_rows)}, comm {comm.sms}, gemm {rest.sms} SMs)"
    arms = {"quack_inter_swap": Kernel(lambda: fork(lambda: (walk.flags.zero_(), sem.zero_())), _NOOP, fused, zero, ref, label=label)}
    if args.presatisfied:  # flags held at target before the fork: the GEMM never waits, the pull still runs
        halves = []
        arms["quack_inter_swap/presat"] = Kernel(
            lambda: fork(lambda: (walk.flags.copy_(walk.targets), sem.zero_()), halves), _NOOP, label=label, halves=halves
        )
    return arms


def _quack_intra_swap(model, x, topk_idx_g, meta, ws, w1, config, rank, W, T_local, args):
    """Intra-SM, swap-AB: bench_forward's prologue kernel (top-k all-gather, routing metadata and the even-split walk
    in one launch; it zeroes the flags and the semaphore) then the gated swap-AB varlen-N GEMM1 whose producer warps
    pull the walk into rows packed per expert, on all SMs (SM100 and SM90). The rows are the pool's in cu order, so
    they are checked as pool[a_idx]; the pre-satisfied arm holds the flags at target after the prologue."""
    dev, dtype = x.device, torch2cute_dtype_map[x.dtype]
    H, I, K, E_local = model.H, model.I, model.topk, model.E // W
    fn, Cls, block_rows = bf.compile_gemm1_gated_swap(dtype, model, W, T_local, intra_pull=H, gather_B=False)
    total_m = int(meta["expert_frequency_offset"][E_local])
    max_rows = -(-total_m // 512) * 512
    buffers = make_combine_buffers(T_local, K, H, torch.bfloat16, ws.ep_group)  # the prologue also fills GEMM2's slot addresses
    prologue, pmeta, walk = make_prologue(
        topk_idx_g[rank].contiguous(), ws.x_peer_bufs, rank, W, T_local, K, E_local, max_rows, block_rows, buffers.slot_base, ws.ep_group
    )
    w1_il8 = bf.interleave_gate_up(w1)
    rows = torch.empty(max_rows, H, dtype=torch.bfloat16, device=dev)  # B rows packed per expert (cu order)
    a = torch.empty(max_rows, I, dtype=torch.bfloat16, device=dev)
    clusters = get_max_active_clusters(bf.CLUSTER[0] * bf.CLUSTER[1], get_device_capacity(dev))
    sched = make_scheduler_args(clusters, 8, prologue.sem1, dispatch_args=(walk.flags, walk.first_src_row, pmeta.a_idx))
    rt = (Cls.EpilogueArguments(mAuxOut=a.T), sched, make_varlen_args(None, None, None, cu_seqlens_n=pmeta.cu))
    # per-CTA copies of the weight tensormaps (the SM100 kernel patches L2 promotion to 256B): 128 B slots
    desc = (torch.empty(torch.cuda.get_device_properties(dev).multi_processor_count * 16, dtype=torch.int64, device=dev),) if bf.SM100 else ()
    launch = lambda: fn(w1_il8, rows, None, None, *rt, None, None, walk.args, *desc)
    prologue()
    torch.cuda.synchronize()
    assert int(pmeta.cu[E_local]) == total_m, "the prologue's routing tables differ from sonic's"
    targets = flag_targets(pmeta.cu, torch.zeros_like(walk.flags), block_rows, True)
    a_idx = pmeta.a_idx[:total_m].long()

    def run():
        prologue()
        launch()

    def presat():
        prologue()
        walk.flags.copy_(targets)
        launch()

    fused = lambda: (a[:total_m], rows[:total_m], walk.flags)
    zero = lambda: (a.zero_(), rows.zero_(), walk.flags.zero_())
    ref = lambda r: (bf.swiglu_cat(r[0]), r[1][a_idx], targets)
    arms = {"quack_intra_swap": Kernel(run, _NOOP, fused, zero, ref, label=f" ({_swap_tag(block_rows)})")}
    if args.presatisfied:
        arms["quack_intra_swap/presat"] = Kernel(presat, _NOOP)
    return arms


def _legs(model, x, topk_idx_g, meta, ws, w1, config, rank, W, T_local, args):
    """The bound legs, each alone and back-to-back (timing-only): sonic's dispatch into recv, and the GEMM on recv
    with its flags pre-filled on all SMs."""
    dev, dtype = x.device, torch2cute_dtype_map[x.dtype]
    cu = meta["expert_frequency_offset"]
    total_m = int(cu[model.E // W])
    a_idx = meta["x_idx_expanded_remap_for_rank_dedup"][:total_m]
    recv = torch.empty(W * T_local, model.H, dtype=torch.bfloat16, device=dev)
    h = torch.empty(total_m, 2 * model.I, dtype=torch.bfloat16, device=dev)
    fn, block_rows = compile_gemm1(dtype, config)
    flags = flag_targets(cu, torch.zeros(model.E // W, -(-W * T_local // block_rows), dtype=torch.int32, device=dev), block_rows)
    first_src_row = meta["pair_offset"][(rank + 1) % W, rank : rank + 1]
    rt = gemm1_args(recv, cu, a_idx, flags, first_src_row, config)
    return {
        "legs/dispatch": Kernel(_sonic_dispatch(ws, meta, recv, rank, model.topk), _NOOP),
        "legs/gemm": Kernel(lambda: fn(recv, w1, h, None, *rt, None, None), _NOOP),
    }


def _tk(model, x, topk_idx_g, meta, ws, w1, config, rank, W, T_local, args):
    """ThunderKittens' fused dispatch + expert GEMM (~/ThunderKittens/kernels/parallel/moe_dispatch_gemm, H100 only;
    H, E and top-k hard-coded to DeepSeek-V3, NUM_DEVICES and the GEMM width compile-time: `GPU=H100 make ND={W}
    I={2I}` there builds _C_nd{W}_i{2I}, so it does our GEMM1's work). No dedup: one pulled row per (token, expert)
    pair, each local expert padded to 128 rows, the schedule built here in TK's order (sources (i + rank) % W, tokens
    ascending); num_comm_sms of its 132 SMs pull, one arm per value of --tk-comm-sms. Timing-only: its rows are a
    different permutation with padding, so they are checked once here against torch on the unpadded rows."""
    H, I, E, topk = model
    E_local, dev = E // W, x.device
    d = os.path.expanduser("~/ThunderKittens/kernels/parallel/moe_dispatch_gemm")
    # the shape is compile-time there: one module per (world, 2I, H, E, top-k); the dsv3 build keeps its short name
    tag = f"_C_nd{W}_i{2 * I}" + ("" if (H, E, topk) == (7168, 256, 8) else f"_h{H}_e{E}_k{topk}")
    cand = glob.glob(f"{d}/{tag}.*.so")
    assert cand, f"no TK module {tag}; run `GPU=H100 make ND={W} I={2 * I} H={H} E={E} K={topk}` in {d}"
    spec = importlib.util.spec_from_file_location("_C", cand[0])
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    pre = m.TKParallelTensor((T_local, H), dtype=torch.bfloat16, local_rank=rank, local_world_size=W, multicast=False)
    pre.data_.copy_(x)
    counts = torch.bincount(topk_idx_g.reshape(-1), minlength=E).to(torch.int32)  # global (token, expert) pairs
    padded = (counts + 127) // 128 * 128
    local = padded[rank * E_local : (rank + 1) * E_local]
    n_pad = int(local.sum())
    pos = torch.cumsum(local, 0) - local  # next free slot per local expert
    pull = torch.full((n_pad, 2), -1, dtype=torch.int32, device=dev)  # (source rank, source token) per padded row
    for q in [(i + rank) % W for i in range(W)]:
        t, k = (topk_idx_g[q] // E_local == rank).nonzero(as_tuple=True)  # ascending token order
        e = topk_idx_g[q][t, k] % E_local
        order = torch.argsort(e, stable=True)
        e, t = e[order], t[order]
        n_e = torch.bincount(e, minlength=E_local)
        slot = pos[e] + torch.arange(e.numel(), device=dev) - (torch.cumsum(n_e, 0) - n_e)[e]
        pull[slot, 0], pull[slot, 1] = q, t.to(torch.int32)
        pos += n_e
    weights = w1.transpose(1, 2).contiguous()  # (E_local, H, 2I): TK's layout
    post = torch.empty(n_pad, H, dtype=torch.bfloat16, device=dev)
    out = torch.empty(n_pad, 2 * I, dtype=torch.bfloat16, device=dev)
    n_max = int(padded.reshape(W, E_local).sum(1).amax())
    barrier = m.TKParallelTensor((2, max(n_max, 1 << 19)), dtype=torch.int, local_rank=rank, local_world_size=W, multicast=True)
    barrier.data_.zero_()  # reset by the kernel afterwards
    dist.barrier()
    torch.cuda.synchronize()

    def launch(n_comm, _keep=(pre, barrier, m)):
        m.moe_dispatch_gemm(pre, post, weights, out, padded, pull, barrier, n_comm, n_pad)

    launch(int(args.tk_comm_sms.split(",")[0]))
    torch.cuda.synchronize()
    valid = pull[:, 0] >= 0
    rows = torch.empty(n_pad, H, dtype=torch.bfloat16, device=dev)
    for q in range(W):  # every rank's x through sonic's peer views
        sel = valid & (pull[:, 0] == q)
        rows[sel] = ws.x_peer_bufs[q][pull[sel, 1].long()]
    starts = (torch.cumsum(local, 0) - local).tolist()
    ok = torch.equal(post[valid], rows[valid])
    err = 0.0
    for e, s in enumerate(starts):
        n_e = int(counts[rank * E_local + e])
        ref = rows[s : s + n_e] @ w1[e].T
        err = max(err, ((out[s : s + n_e].float() - ref.float()).abs().max() / (ref.float().abs().max() + 1e-6)).item())
    label = f"  [rows {'bitwise' if ok else 'MISMATCH'}, gemm max rel err {err:.1e}]"
    return {
        f"tk/c{n}": Kernel(lambda n=n: launch(n), _NOOP, label=f" (comm {n} SMs)" + label)
        for n in (int(v) for v in args.tk_comm_sms.split(","))
    }


BASELINES = {
    "quack_inter": _quack_inter, "quack_intra": _quack_intra, "quack_inter_swap": _quack_inter_swap,
    "quack_intra_swap": _quack_intra_swap, "sonic": _sonic, "legs": _legs, "tk": _tk,
}
ONCE = ("tk", "quack_inter_swap", "quack_intra_swap")  # arms with their own tile: not on the quack tile fallbacks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("baselines", nargs="+", choices=list(BASELINES))
    p.add_argument("--model", choices=list(MODEL_CONFIGS), default="dsv3", help="shape: H, I, E, top-k (MODEL_CONFIGS)")
    p.add_argument("--t-local", type=int, default=8192)
    p.add_argument("--comm-sms", type=int, default=8, help="quack_inter: SMs of the pull's partition; the GEMM gets the rest")
    p.add_argument("--tk-comm-sms", type=str, default="32,40,48", help="tk: its num_comm_sms values, one arm each (TK sweeps 28..53)")
    p.add_argument("--presatisfied", action="store_true", help="also time each baseline with its flags held at target (see the header)")
    p.add_argument("--ref_check", action="store_true", help="each arm's fresh output (zeroed first) vs the reference")
    p.add_argument("--r2r_check", action="store_true", help="run each arm twice; bit-identity of the two outputs")
    args = p.parse_args()

    # Process group, shape, symm-mem workspace, routing and metadata: as bench_dispatch.py.
    rank, W = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    dev = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(dev)
    dist.init_process_group("nccl", device_id=dev)
    model = MODEL_CONFIGS[args.model]
    H, I, E, topk = model
    assert E % W == 0, f"{args.model}: E={E} not divisible by W={W}"
    T_local, E_local = args.t_local, E // W
    ws = SymmMemManager(None, dev)._get_or_alloc(
        T_local, H, topk, E_local, torch.bfloat16, DispatchMode.RANK_DEDUP_DISPATCH_TRITON
    )
    g = torch.Generator(device=dev).manual_seed(123 + rank)
    x = torch.randn(T_local, H, generator=g, device=dev, dtype=torch.bfloat16)
    topk_idx = torch.rand(T_local, E, generator=g, device=dev).topk(topk, dim=-1).indices.to(torch.int32)
    w1 = torch.randn(E_local, 2 * I, H, generator=g, device=dev, dtype=torch.bfloat16) * 0.02  # (l, n, k)
    ws.x_symm.copy_(x)
    topk_idx_g = _ag_routing_decision(ws, topk_idx)
    ws.x_hdl.barrier()
    meta = compute_dispatch_metadata(topk_idx_g, my_rank=rank, E_local=E_local, emit_combine=False)
    consumer = _build_consumer_metadata(
        expert_indices=meta["expert_local_padded"], token_indices=meta["a2a_token_indices"], TK=W * T_local * topk,
        E_local=E_local, s_reverse_idx_symm=ws.s_rev_symm,
    )
    meta["recv_pos"] = consumer["s_reverse_local"]
    meta["s_scatter_idx"] = consumer["s_scatter_idx"]
    meta["expert_frequency_offset"] = consumer["expert_frequency_offset"]
    meta["x_idx_expanded_remap_for_rank_dedup"] = build_rank_dedup_a_idx(
        dst_rank_flat=meta["dst_rank_flat"], s_reverse_local=consumer["s_reverse_local"],
        rank_dedup_recv_pos=meta["rank_dedup_recv_pos"], my_rank=rank, out=ws.x_idx_expanded_remap_for_rank_dedup_buf,
    )

    # The tile config: quack's autotuner over our GEMM on landed rows at this shape, on rank 0 (its shape stands for
    # every rank's), the winner broadcast so all ranks and arms share one config.
    cu = meta["expert_frequency_offset"]
    total_m = int(cu[E_local])
    recv = torch.empty(W * T_local, H, dtype=torch.bfloat16, device=dev)
    _sonic_dispatch(ws, meta, recv, rank, topk)()
    torch.cuda.synchronize()
    config = [None]
    if rank == 0:
        h = torch.empty(total_m, 2 * I, dtype=torch.bfloat16, device=dev)
        gemm1_tuned(recv, w1, h, meta["x_idx_expanded_remap_for_rank_dedup"][:total_m], cu, meta["pair_offset"][(rank + 1) % W, rank : rank + 1])
        config = [gemm1_tuned.best_config.kwargs["config"]]
        del h
    dist.broadcast_object_list(config, src=0)
    config = config[0]
    del recv
    # Every arm runs on the tuner's tile and on the two 2-CTA fallbacks, as `<name>@tile`: the tuner's single-window
    # pick can be contention-poisoned (and is disk-cached), and a kernel sharing its SMs or its A traffic with the
    # transfer can prefer a bigger tile than the bare GEMM. The report marks the best per baseline.
    tile = lambda c: (c.tile_m, c.tile_n, c.cluster_m, c.cluster_n)
    sm100 = get_device_capacity(dev)[0] >= 10  # H100: the tuned config only (no 2-CTA MMA tiles to fall back on)
    fallbacks = [replace(config, tile_m=256, tile_n=n, cluster_m=2, cluster_n=1) for n in (512, 256)] if sm100 else []
    configs = [config] + [c for c in fallbacks if tile(c) != tile(config)]
    tag = lambda c: f"@{c.tile_m}x{c.tile_n}({c.cluster_m},{c.cluster_n})"
    shared = lambda c: (model, x, topk_idx_g, meta, ws, w1, c, rank, W, T_local, args)

    ref = _sonic(*shared(config))["sonic"]
    ref.run()
    torch.cuda.synchronize()
    refs = tuple(t.clone() for t in ref.read_out())  # (h on the landed rows, sonic's receive rows): every arm's reference
    arms = {}
    for name in args.baselines:
        for c in configs if name not in ONCE else configs[:1]:
            arms.update({k.replace(name, name + tag(c), 1): v for k, v in BASELINES[name](*shared(c)).items()})
    checks = {name: {} for name in arms}
    for name, kern in arms.items():
        if kern.ref is None:
            continue
        if args.ref_check:  # the swap arms' postact is a different kernel's: within 2^-7, the rows and flags bitwise
            checks[name]["ref"] = (ref_check_tol if "swap" in name else ref_check)(kern, refs)
        if args.r2r_check:
            checks[name]["r2r"] = r2r_check(kern)

    ms = bench_all({name: kern.run for name, kern in arms.items()}, ws.x_hdl)
    unstable = [None] * W  # rerun set unioned over ranks: every rank must call the same collectives
    dist.all_gather_object(unstable, [k for k, v in ms.items() if arm_stats(v)[3]])
    rerun = [k for k in arms if any(k in u for u in unstable)]
    if rerun:
        again = bench_all({k: arms[k].run for k in rerun}, ws.x_hdl)
        ms.update({k: v for k, v in again.items() if not arm_stats(v)[3]})

    flops = 2 * total_m * H * 2 * I
    nbytes = (n_recv(meta, rank) - int(meta["pair_count"][rank, rank])) * H * 2  # wire rows
    stats = {k: arm_stats(v) for k, v in ms.items()}
    torch.cuda.synchronize()
    halves = {  # per presat arm with two parts: median (comm ms, gemm ms) over every recorded call, all at the same load
        k: (statistics.median(e[0].elapsed_time(e[1]) for e in kern.halves), statistics.median(e[2].elapsed_time(e[3]) for e in kern.halves))
        for k, kern in arms.items() if kern.halves
    }
    gathered = [None] * W
    dist.all_gather_object(gathered, (stats, checks, halves))
    if rank == 0:
        med = {k: max((g[k] for g, _, _ in gathered), key=lambda t: t[0]) for k in ms}  # slowest rank sets the collective
        print(
            f"<bf16 fused dispatch + GEMM1 {args.model} (H={H} I={I} E={E} top-{topk}) | W={W} T_local={T_local} "
            f"total_m={total_m} wire {nbytes / 1e6:.0f} MB | tuned tile {tag(config)[1:]} swizzle {config.max_swizzle_size}, "
            f"every arm also on {', '.join(tag(c)[1:] for c in configs[1:]) or 'nothing else'} | warm {WARM_S * 1e3:.0f} ms, "
            f"{REPS} x {WINDOW_S * 1e3:.0f} ms blocks, rerun={rerun}>"
        )
        sonic_best = min((med[k][0] for k in med if k.startswith("sonic@") and not k.endswith("/presat")), default=None)
        for k, (m, lo, hi, _) in med.items():
            mean = sum(g[k][0] for g, _, _ in gathered) / W
            flag = "  UNSTABLE" if any(g[k][3] for g, _, _ in gathered) else ""
            print(
                f"  {k + arms[k].label + ':':<40} {m * 1e3:7.0f} us  [{lo * 1e3:.0f}-{hi * 1e3:.0f}]  "
                f"{flops / m / 1e9:5.0f} TFLOP/s  rank-mean {mean * 1e3:.0f} us{flag}"
            )
        if args.presatisfied:
            for name in args.baselines:
                cands = [k for k in med if k.startswith(name + "@") and not k.endswith("/presat")]
                best = min(cands, key=lambda k: med[k][0])
                for k in cands:
                    fused, kp = med[k][0], k + "/presat"
                    vs = f"  vs best sonic {fused / sonic_best - 1:+.1%}" if sonic_best and name != "sonic" else ""
                    mark = "  <- best" if k == best and len(cands) > 1 else ""
                    if kp not in med:  # no pre-satisfied twin (tk): fused vs sonic only
                        print(f"  {k}: fused {fused * 1e3:.0f} us{vs}{mark}")
                    elif kp in halves:  # two parts under one power budget (quack_inter concurrent, sonic sequential)
                        comm = max(h[kp][0] for _, _, h in gathered)
                        gemm = max(h[kp][1] for _, _, h in gathered)
                        bound = max(comm, gemm)
                        print(
                            f"  {k}: comm {comm * 1e3:.0f} us ({nbytes / comm / 1e6:.0f} GB/s)  gemm {gemm * 1e3:.0f} us  "
                            f"bound = max = {bound * 1e3:.0f} us  fused {fused * 1e3:.0f} us  vs bound {fused / bound - 1:+.1%}{vs}{mark}"
                        )
                    else:  # one kernel: the pre-satisfied time is the bound
                        bound = med[kp][0]
                        print(f"  {k}: bound = pre-satisfied kernel {bound * 1e3:.0f} us  fused {fused * 1e3:.0f} us  vs bound {fused / bound - 1:+.1%}{vs}{mark}")
        for name in arms:
            for what in checks[name]:
                ok = all(c[name][what][0] for _, c, _ in gathered)
                delta = max(c[name][what][1] for _, c, _ in gathered)
                if what == "ref" and "swap" in name:
                    print(f"  {name} {what}: {'ok' if ok else 'MISMATCH'} max rel err {delta:.3e}")
                else:
                    print(f"  {name} {what}: {'bitwise' if ok else 'MISMATCH'} max|delta|={delta:.3e}")
    for kern in arms.values():
        kern.free()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
