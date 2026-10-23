"""Run fused GEMM2 + combine baselines through one interface (bench_fused_dispatch_gemm.py's structure,
bench_combine.py's protocol and combine arms).

GEMM2 y = a @ w2[expert] over sonic's expert-major rows (a (total_m, I) random, w2 (E_local, H, I)), then the
combine of y into out (T_local, H) on each token's origin rank, K rows per token weighted by the router scores. One
quack GEMM kernel in every arm (branch ep's varlen-M GEMM with the combine scheduler argument: +1 on
flags[expert, M cluster] per stored N tile from each CTA of the cluster, so a cluster's flag reads
cluster size x N tiles once its rows are in HBM; compiled below as bench_fused_dispatch_gemm.py's GEMM1, fake tensors
at compile, torch tensors at launch), on one fixed tile, 256x512 (2,1) static (the M cluster = 256 rows), so the
fusion is the only variable. y is computed once in the prologue by that kernel, so the fp32 reference (bench_combine's)
matches the fused arm's y bitwise. Shape from --model, bf16, uniform routing; metadata built once outside the timed
region.

  quack_inter  inter-SM overlap: the walk build on the GPU (combine_push.make_combine_walk_device: the rows in the
               GEMM's cluster completion order, one launch inside the timed call), then the push
               (combine/combine_push.py, inter mode) on --comm-sms SMs, waiting per M cluster for its flag, pushing
               raw y rows into the origins' slots, alongside the GEMM on the rest, two disjoint green-context
               partitions; then a barrier (every peer's rows landed) and sonic's gather-and-sum (_router_forward's
               kernel: the scores go in here) over the slots
  quack_intra_epi  the GEMM2 whose epilogue stores each subtile from its smem stage straight into the origins' slots
               (fused_gemm_combine/gemm_sm100_intra_combine_push.py, no y, no flags), on all SMs; then the same
               barrier and reduce
  quack_inter_swap  as quack_inter with bench_forward's swap-AB varlen-N GEMM2 (weights on M, tokens on N in
               even-split blocks of the swap tile, D = y.T, the combine flags per block), SM100 and SM90
  quack_intra_swap  the swap-AB GEMM2 whose epilogue stores each token's slice straight into the origins' slots
               (gemm_sm100/sm90_intra_combine_push.py, no y, no flags, no transfer warps), on all SMs; then the
               barrier and the reduce. Nothing in it waits, so the arm is its own bound
  sonic        no fusion: the GEMM on the whole GPU, then sonic's rank-dedup combine (bench_combine's arm, the canary)
  sonic_a2a    reference: sonic's routed combine alone (a2a_combine_triton, bench_combine's arm) on the whole GPU,
               the same wire rows as our push, no GEMM
Through the next layer's norm (fc2 + combine + reduce + RMSNorm): quack_intra_swap+rmsnorm = the intra arm whose
reduce carries the sq-sum epilogue and finalizes rstd per token (unnormalized out + rstd, the deferred form);
deepep_te = TE GroupedLinear fc2 over DeepEP's expanded rows (one setup dispatch for the handle, a permuted into
its layout), the scores onto the rows, DeepEP combine, a torch.compile'd RMSNorm (normalized rows).
--presatisfied also times `rmsnorm pass`, the graphed compiled RMSNorm over the plain reduce's out.

--presatisfied adds `quack_inter/presat`: the same arrangement (same streams, partitions and power budget) with the
flags held at target, so the push never waits and the GEMM still runs; each half is bracketed by its own events on
its own stream, giving t_comm and t_gemm under the shared budget and the bound = max(t_gemm, t_comm) (the barrier
and reduce come after both), and `quack_inter/reduce`: the reduce alone, back-to-back, so bound + reduce is the
floor of the whole arm as built (the epilogue is not overlapped); the epilogue-store arms get `/reduce` only.
sonic's two legs are sequential, so its arm brackets them on every call: its combine
leg and its GEMM leg (= the GEMM alone on all SMs) come from the timed arm itself.

Each factory returns {arm name: Kernel} (bench_dispatch.Kernel); read_out is out as fp32 and ref the fp32 reference,
so --ref_check / --r2r_check work generically; the fused arm's zero() clears its slots too, so an early read would
show. The presat arm's push may read rows before the GEMM stored them, so it is timing-only.

Usage:
    PYTHONPATH=~/quack:~/sonic-moe-pr54 torchrun --nproc_per_node=8 --standalone bench_fused_gemm_combine.py \\
        quack_inter quack_intra_swap sonic --comm-sms 16 --presatisfied --ref_check
"""
import os

os.environ.setdefault("TRITON_CACHE_AUTOTUNING", "1")  # sonic's triton autotune choices persist across launches

import argparse
import statistics
import sys
from types import SimpleNamespace

import cutlass
import cutlass.cute as cute
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # the kernels package root
from kernels.misc import nccl_env  # noqa: E402, F401  (NCCL channel floor for every reference)
from cutlass import Int32, Int64
from quack.compile_utils import make_fake_tensor
from quack.cute_dsl_utils import get_device_capacity, get_max_active_clusters, torch2cute_dtype_map
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
from sonicmoe.functional.distributed import build_rank_dedup_a_idx, compute_dispatch_metadata
from sonicmoe.functional.ep import _ag_routing_decision, _build_consumer_metadata, _build_single_row

from bench_combine import _sonic, _sonic_a2a, ref_check
from bench_dispatch import _NOOP, MODEL_CONFIGS, REPS, WARM_S, WINDOW_S, Kernel, SMStream, arm_stats, bench_all, r2r_check
from combine.combine_push import make_combine_buffers, make_combine_push, make_combine_walk_device
from fused_gemm_combine.gemm_sm100_intra_combine_push import GemmSm100 as GemmSm100IntraCombine
from fused_gemm_combine.gemm_sm90_intra_combine_push import GemmSm90 as GemmSm90IntraCombine
from kernels.misc.utils import capture_graph
from reduce_over_k import ColVecReduce, reduce_epilogue, reduce_over_k

EPS = 1e-6


@reduce_epilogue(reduces={"sqsum": ColVecReduce("sqsum", scaled=True)})
def sqsum_epi(acc):
    return {"D": acc, "sqsum": (acc, acc)}


def rmsnorm(y, w):
    yf = y.float()
    return (yf * torch.rsqrt(yf.pow(2).mean(-1, keepdim=True) + EPS) * w).to(y.dtype)


def _rstd(y):
    return torch.rsqrt(y.float().pow(2).mean(-1, keepdim=True) + EPS)


def _gamma(H, dev):  # the norm weight, the same on every arm
    return 1 + 0.02 * torch.randn(H, generator=torch.Generator(device=dev).manual_seed(7), device=dev)


# The M cluster is 256 rows either way (the walk's BLOCK_ROWS): 2-CTA 256x256 on SM100, 128x256 x cluster 2 on H100.
# 256x256 (two tmem accumulator stages) over 256x512 (one stage, epilogue exposed): GEMM2 alone -5 %, intra -2 %
# (results/2026-09-28/epilogue_store.md, 2026-09-26).
SM100 = get_device_capacity(torch.device("cuda"))[0] >= 10
TILE, CLUSTER = ((256, 256), (2, 1)) if SM100 else ((128, 256), (2, 1))


class GemmDefaultSm100IntraCombine(GemmDefaultEpiMixin, GemmSm100IntraCombine):
    pass


class GemmDefaultSm90IntraCombine(GemmDefaultEpiMixin, GemmSm90IntraCombine):
    pass


def compile_gemm2(dtype, tile=TILE, cluster=CLUSTER, epilogue_store=0):
    """(fn, block_rows, cta_tile_n): the GEMM compiled against symbolic shapes (varlen M, combine scheduler
    argument, static persistence); block_rows = the M cluster (CTA tile M x cluster M), the unit the push waits on.
    epilogue_store = the hidden size: the kernel whose epilogue warps store rows of that width into the slots
    themselves (after the stream: the slot base table and the slots per rank)."""
    if epilogue_store:
        assert SM100, "the epilogue-store kernel is SM100 only here (SM90: the swap arm)"
        gemm = GemmDefaultSm100IntraCombine(
            cutlass.Float32, dtype, tile, (*cluster, 1), use_clc_persistence=False,
            combine_epilogue_store=epilogue_store,
        )
        i64 = lambda: make_fake_tensor(Int64, (cute.sym_int(),), leading_dim=0, divisibility=1)
        walk_fakes = (i64(), Int32(0))
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
    """(out, reduce): reduce_over_k's gather-and-sum over the (T_local, K, H) slots with the scores, into out
    (T_local, H)."""
    out = torch.empty(T_local, H, dtype=torch.bfloat16, device=scores.device)
    return out, lambda: reduce_over_k(buffers.slots, scores, out)


def _gather_and_sum_rms(buffers, scores, T_local, K, H):
    """(out, rstd, reduce): the reduce above carrying the sq-sum epilogue (per-tile partials), rstd finalized per
    token: the unnormalized out + rstd, the deferred form the GEMM+RS+RMSNorm chain uses."""
    out = torch.empty(T_local, H, dtype=torch.bfloat16, device=scores.device)
    partials = torch.empty(1, T_local, -(-H // 1024), dtype=torch.float32, device=out.device)
    rstd = torch.empty(T_local, 1, dtype=torch.float32, device=out.device)

    def reduce():
        sums = reduce_over_k(buffers.slots, scores, out, epilogue=sqsum_epi, epi_args={"sqsum": partials})
        torch.rsqrt(sums["sqsum"].view(T_local, 1) / H + EPS, out=rstd)

    return out, rstd, reduce


def _fork_arm(ws, comm, rest, build, push, gemm, reduce):
    """fork(prologue, halves=None): the walk build and prologue (the flags) on the current stream, then the push on
    comm and the GEMM on rest, joined, then the barrier (every peer's rows have landed in our slots) and the reduce;
    halves: record each half's events."""

    def fork(prologue, halves=None):
        cur = torch.cuda.current_stream()
        build()
        prologue()
        comm.stream.wait_stream(cur)
        rest.stream.wait_stream(cur)
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)] if halves is not None else None
        with comm.ctx():
            if ev:
                ev[0].record()
            push()
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
        ws.o_hdl.barrier(channel=1)
        reduce()
        if ev:
            halves.append(ev)

    return fork


# Every factory: (model, y, scores, scores_g, topk_idx_g, meta, ws, gemm2, rank, W, T_local, args) -> {arm: Kernel},
# as bench_combine's with gemm2 in the SM-partition slot: y = ws.y_symm[:total_m] (the GEMM's D), gemm2 = (fn, a,
# w2, cu, flags, block_rows, flag_target, run) with run = the GEMM on the whole GPU.
def _quack_inter(model, y, scores, scores_g, topk_idx_g, meta, ws, gemm2, rank, W, T_local, args):
    K, H, dev = model.topk, model.H, y.device
    comm = SMStream(dev.index, args.comm_sms)
    assert comm.stream is not None, "--comm-sms must leave SMs for the GEMM"
    rest = comm.complement()
    flags = torch.zeros_like(gemm2.flags)  # this arm's, zeroed per call
    buffers = make_combine_buffers(T_local, K, H, torch.bfloat16, ws.ep_group)
    build, walk = make_combine_walk_device(gemm2.slot_of_row, gemm2.cu, flags, gemm2.first_src_row, buffers, T_local, K, tile_n=gemm2.tile_n)
    push = make_combine_push(y, "inter", comm.sms)(walk, gemm2.flag_target)
    rt = gemm2_args(gemm2.cu, flags, gemm2.slot_of_row, gemm2.first_src_row, sms=rest.sms)
    gemm = lambda: gemm2.fn(gemm2.a, gemm2.w2, y, None, *rt, None, None)
    out, reduce = _gather_and_sum(buffers, scores, T_local, K, H)
    fork = _fork_arm(ws, comm, rest, build, push, gemm, reduce)
    label = f" (push {comm.sms}, gemm {rest.sms} SMs)"
    zero = lambda: (out.zero_(), buffers.slots.zero_())
    # the fused arm zeroes the flags before the fork: the push must not see last call's full flags
    arms = {"quack_inter": Kernel(lambda: fork(flags.zero_), _NOOP, out.float, zero, lambda ref: ref, label=label)}
    if args.presatisfied:  # flags held at target before the fork: the push never waits, the GEMM still runs
        halves = []
        arms["quack_inter/presat"] = Kernel(lambda: fork(lambda: flags.fill_(gemm2.flag_target), halves), _NOOP, label=label, halves=halves)
        arms["quack_inter/reduce"] = Kernel(reduce, _NOOP)  # the epilogue alone: sonic's gather-and-sum over the slots
    return arms


def _quack_intra_epi(model, y, scores, scores_g, topk_idx_g, meta, ws, gemm2, rank, W, T_local, args):
    """Epilogue store: the GEMM's epilogue stores each subtile from its smem stage straight into the origins' slots
    (y is never written; no walk, no flags), on all SMs; then the barrier and the reduce. Nothing in the kernel
    waits, so the arm is its own bound."""
    K, H = model.topk, model.H
    fn, _, _ = compile_gemm2(torch2cute_dtype_map[y.dtype], epilogue_store=H)
    buffers = make_combine_buffers(T_local, K, H, torch.bfloat16, ws.ep_group)
    rt = gemm2_args(gemm2.cu, gemm2.flags, gemm2.slot_of_row, gemm2.first_src_row)
    launch = lambda: fn(gemm2.a, gemm2.w2, y, None, *rt, None, None, buffers.slot_base, T_local * K)
    out, reduce = _gather_and_sum(buffers, scores, T_local, K, H)

    def run():
        launch()
        ws.o_hdl.barrier(channel=1)
        reduce()

    zero = lambda: (out.zero_(), buffers.slots.zero_())
    arms = {"quack_intra_epi": Kernel(run, _NOOP, out.float, zero, lambda ref: ref)}
    if args.presatisfied:
        arms["quack_intra_epi/reduce"] = Kernel(reduce, _NOOP)
    return arms


def _quack_inter_swap(model, y, scores, scores_g, topk_idx_g, meta, ws, gemm2, rank, W, T_local, args):
    """Inter-SM, swap-AB: bench_forward's swap-AB varlen-N GEMM2 (weights on M, tokens on N in even-split blocks of
    the swap tile, D = y.T, its combine flags per block, dynamic persistence) on the --comm-sms complement beside the
    push on the partition, then the barrier and the reduce. The walk is cut on the swap block, not the 256-row M
    cluster. SM100 only for now: on SM90 the push never sees the swap GEMM's flags (hangs, 2026-10-19), skipped."""
    import bench_forward as bf  # here, not at the top: bench_forward imports this module

    if not SM100:
        if rank == 0:
            print("quack_inter_swap: skipped on SM90 (push/flag handshake unresolved)", flush=True)
        return {}
    K, H, E_local, dev = model.topk, model.H, model.E // W, y.device
    fn, block_rows, cluster_tile_m = bf.compile_gemm2_swap(torch2cute_dtype_map[y.dtype], model, W, T_local)
    comm = SMStream(dev.index, args.comm_sms)
    assert comm.stream is not None, "--comm-sms must leave SMs for the GEMM"
    rest = comm.complement()
    flags = torch.zeros(E_local, -(-W * T_local // block_rows), dtype=torch.int32, device=dev)  # this arm's, zeroed per call
    flag_target = CLUSTER[0] * CLUSTER[1] * -(-H // cluster_tile_m)  # CTA tiles along M per block
    buffers = make_combine_buffers(T_local, K, H, torch.bfloat16, ws.ep_group)
    build, walk = make_combine_walk_device(
        gemm2.slot_of_row, gemm2.cu, flags, gemm2.first_src_row, buffers, T_local, K, tile_n=cluster_tile_m, even_split=True,
        block_rows=block_rows,
    )
    push = make_combine_push(y, "inter", comm.sms)(walk, flag_target)
    sem = torch.zeros(1, dtype=torch.int32, device=dev)  # dynamic persistence's tile counter, zeroed per call
    sched = make_scheduler_args(rest.sms // (CLUSTER[0] * CLUSTER[1]), 8, sem, combine_args=(flags, gemm2.slot_of_row, gemm2.first_src_row))
    epi = GemmDefaultEpiMixin.EpilogueArguments(add_to_output=None, rounding_mode=None)
    rt = (epi, sched, make_varlen_args(None, None, None, cu_seqlens_n=gemm2.cu))
    gemm = lambda: fn(gemm2.w2, gemm2.a, y.T, None, *rt, None, None)
    out, reduce = _gather_and_sum(buffers, scores, T_local, K, H)
    fork = _fork_arm(ws, comm, rest, build, push, gemm, reduce)
    label = f" (swap tile {256 if SM100 else 128}x{block_rows}, push {comm.sms}, gemm {rest.sms} SMs)"
    zero = lambda: (out.zero_(), buffers.slots.zero_())
    arms = {"quack_inter_swap": Kernel(lambda: fork(lambda: (flags.zero_(), sem.zero_())), _NOOP, out.float, zero, lambda ref: ref, label=label)}
    if args.presatisfied:
        halves = []
        arms["quack_inter_swap/presat"] = Kernel(
            lambda: fork(lambda: (flags.fill_(flag_target), sem.zero_()), halves), _NOOP, label=label, halves=halves
        )
        arms["quack_inter_swap/reduce"] = Kernel(reduce, _NOOP)
    return arms


def _quack_intra_swap(model, y, scores, scores_g, topk_idx_g, meta, ws, gemm2, rank, W, T_local, args):
    """Swap-AB, epilogue store: the swap-AB GEMM2 whose epilogue stores each token's slice of a subtile straight
    into the origins' slots (per-row slot addresses from the global slot ids; y is never written, no flags), on all
    SMs; then the barrier and the reduce. Nothing in the kernel waits, so the arm is its own bound. SM100 and SM90."""
    import bench_forward as bf  # here, not at the top: bench_forward imports this module

    K, H, E_local, dev = model.topk, model.H, model.E // W, y.device
    fn, block_rows, _ = bf.compile_gemm2_swap(torch2cute_dtype_map[y.dtype], model, W, T_local, intra_push=H)
    buffers = make_combine_buffers(T_local, K, H, torch.bfloat16, ws.ep_group)
    per_rank = T_local * K
    s = gemm2.slot_of_row.long()  # global slot id -> (origin rank, slot) -> the slot's address
    slot_addr = buffers.slot_base[s // per_rank] + (s % per_rank) * (H * y.element_size())
    flags = torch.zeros(E_local, -(-W * T_local // block_rows), dtype=torch.int32, device=dev)  # the scheduler's; never published
    sem = torch.zeros(1, dtype=torch.int32, device=dev)
    clusters = get_max_active_clusters(CLUSTER[0] * CLUSTER[1], get_device_capacity(dev))
    sched = make_scheduler_args(clusters, 8, sem, combine_args=(flags, gemm2.slot_of_row, gemm2.first_src_row))
    epi = GemmDefaultEpiMixin.EpilogueArguments(add_to_output=None, rounding_mode=None)
    rt = (epi, sched, make_varlen_args(None, None, None, cu_seqlens_n=gemm2.cu))
    desc = (torch.empty(torch.cuda.get_device_properties(dev).multi_processor_count * 16, dtype=torch.int64, device=dev),) if SM100 else ()
    launch = lambda: fn(gemm2.w2, gemm2.a, y.T, None, *rt, None, None, buffers.slot_base, per_rank, slot_addr, *desc)
    out, reduce = _gather_and_sum(buffers, scores, T_local, K, H)

    def run():
        sem.zero_()
        launch()
        ws.o_hdl.barrier(channel=1)
        reduce()

    zero = lambda: (out.zero_(), buffers.slots.zero_())
    label = f" (swap tile {256 if SM100 else 128}x{block_rows})"
    arms = {"quack_intra_swap": Kernel(run, _NOOP, out.float, zero, lambda ref: ref, label=label)}
    # through the next layer's norm: the same kernel, the reduce with the sq-sum epilogue -> (out, rstd)
    out_r, rstd, reduce_rms = _gather_and_sum_rms(buffers, scores, T_local, K, H)

    def run_rms():
        sem.zero_()
        launch()
        ws.o_hdl.barrier(channel=1)
        reduce_rms()

    zero_r = lambda: (out_r.zero_(), rstd.zero_(), buffers.slots.zero_())
    arms["quack_intra_swap+rmsnorm"] = Kernel(
        run_rms, _NOOP, lambda: (out_r.float(), rstd), zero_r, lambda ref: (ref, _rstd(ref)), label=label
    )
    if args.presatisfied:
        arms["quack_intra_swap/reduce"] = Kernel(reduce, _NOOP)
        normed = torch.empty_like(out)
        rms_c = torch.compile(lambda o, w, dst: dst.copy_(rmsnorm(o, w)))
        gamma = _gamma(H, dev)
        arms["rmsnorm pass"] = Kernel(capture_graph(lambda: rms_c(out, gamma, normed)), _NOOP)
    return arms


def _deepep_te(model, y, scores, scores_g, topk_idx_g, meta, ws, gemm2, rank, W, T_local, args):
    """Unfused, through the norm: TE GroupedLinear fc2 (cuBLASLt grouped GEMM) over DeepEP's expanded rows (a
    permuted into DeepEP's layout at setup, one dispatch for the handle), the scores onto the rows (DeepEP's combine
    only sums), DeepEP combine to the origins, then a torch.compile'd RMSNorm; the output is the normalized rows."""
    import deep_ep
    import transformer_engine.pytorch as te

    H, I, E, K, E_local, dev = model.H, model.I, model.E, model.topk, model.E // W, y.device
    buffer = deep_ep.ElasticBuffer(ws.ep_group, num_max_tokens_per_rank=T_local, hidden=H, num_topk=K, explicitly_destroy=True)
    num_sms = buffer.get_theoretical_num_sms(E, K)
    _, _, recv_w, handle, _ = buffer.dispatch(
        x=ws.x_symm, topk_idx=topk_idx_g[rank].to(deep_ep.topk_idx_t), topk_weights=scores, do_expand=True,
        do_cpu_sync=True, num_sms=num_sms, num_max_tokens_per_rank=T_local, num_experts=E, expert_alignment=1,
    )
    n_tok, n = int(handle.num_recv_tokens), int(handle.num_expanded_tokens)
    psum = handle.psum_num_recv_tokens_per_expert.to(torch.int64)
    splits = torch.diff(psum, prepend=psum.new_zeros(1))
    # DeepEP's expanded row of (token, k): recv_src_metadata = one row per received token, col 0 its global index
    # origin * T_local + t, cols 2:2+K the expanded row of each top-k slot (-1 when that expert is off-rank);
    # sonic's row m = global slot slot_of_row[m] = (origin, t, k). Pair them on (token, local expert).
    md = handle.recv_src_metadata[:n_tok].to(torch.int64)
    gid, rows = md[:, 0], md[:, 2 : 2 + K]
    valid = rows >= 0
    e_d = topk_idx_g.view(-1, K)[gid][valid] - rank * E_local
    key_d = torch.empty(n, dtype=torch.int64, device=dev)
    key_d[rows[valid]] = gid[:, None].expand_as(rows)[valid] * E_local + e_d
    s = gemm2.slot_of_row.long()
    origin, t, k = s // (T_local * K), s % (T_local * K) // K, s % K
    key_s = (origin * T_local + t) * E_local + topk_idx_g[origin, t, k].long() - rank * E_local
    order = torch.argsort(key_s)
    rows_of = order[torch.searchsorted(key_s[order], key_d)]
    assert n == s.numel() == int(valid.sum()) and torch.equal(key_s[rows_of], key_d), "DeepEP's rows do not pair with sonic's"
    a_te, a_sc = gemm2.a[rows_of], torch.empty(n, I, dtype=torch.bfloat16, device=dev)
    y_te = torch.empty(n, H, dtype=torch.bfloat16, device=dev)
    # TE 2.19's grouped-tensor GroupedLinear (as MoK's bench_deepep_te.py): its grouped GEMM is not ordered with the
    # current stream on either side (tmp repro 2026-10-17: stale rows in ~1 call of 10); the ordered API,
    # general_grouped_gemm, runs 1.3-2x slower. Kept as the baseline's own path; see final_results.md.
    lin2 = te.GroupedLinear(E_local, I, H, bias=False, params_dtype=torch.bfloat16, device=dev, use_grouped_tensor=True)
    for e in range(E_local):
        getattr(lin2, f"weight{e}").data = gemm2.w2[e]
    gamma, normed = _gamma(H, dev), torch.empty(T_local, H, dtype=torch.bfloat16, device=dev)
    rstd, last = torch.empty(T_local, 1, dtype=torch.float32, device=dev), {}

    @torch.compile
    def rms_c(o, w, dst):
        of = o.float()
        rstd.copy_(torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + EPS))
        dst.copy_((of * rstd * w).to(o.dtype))

    def run():
        torch.mul(a_te, recv_w[:, None], out=a_sc)
        lin2(a_sc, splits, out=y_te)
        last["o"] = buffer.combine(y_te, handle=handle, num_sms=num_sms)[0]
        rms_c(last["o"], gamma, normed)

    # checked on the combine output and rstd, the fused arm's quantities: the normalized bf16 rows carry a second
    # bf16 rounding (combine out, then norm out), up to 2 x 2^-8 against the check's 2^-7
    kern = Kernel(
        run, buffer.destroy, lambda: (last["o"].float(), rstd), lambda: (normed.zero_(), rstd.zero_()),
        lambda ref: (ref, _rstd(ref)), label=f" (deepep_sms={num_sms})",
    )
    return {"deepep_te+rmsnorm": kern}


def _sonic_chain(model, y, scores, scores_g, topk_idx_g, meta, ws, gemm2, rank, W, T_local, args):
    """No fusion: the GEMM on the whole GPU, then sonic's combine (bench_combine's arm) on the whole GPU too."""
    whole = SMStream(y.device.index, 1 << 30)
    kern = _sonic(model, y, scores, scores_g, topk_idx_g, meta, ws, whole, rank, W, T_local, args)
    halves = []  # every call: (combine leg, GEMM leg) events, sequential on the one stream

    def run():
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        ev[2].record()
        gemm2.run()
        ev[3].record()
        ev[0].record()
        kern.run()
        ev[1].record()
        halves.append(ev)

    return {"sonic": kern._replace(run=run, halves=halves)}


def _sonic_a2a_ref(model, y, scores, scores_g, topk_idx_g, meta, ws, gemm2, rank, W, T_local, args):
    whole = SMStream(y.device.index, 1 << 30)
    kern = _sonic_a2a(model, y, scores, scores_g, topk_idx_g, meta, ws, whole, rank, W, T_local, args)
    return {"sonic_a2a": kern._replace(label=" (ref: combine alone)")}


BASELINES = {
    "quack_inter": _quack_inter, "quack_intra_epi": _quack_intra_epi,
    "quack_inter_swap": _quack_inter_swap, "quack_intra_swap": _quack_intra_swap, "sonic": _sonic_chain,
    "sonic_a2a": _sonic_a2a_ref, "deepep_te": _deepep_te,
}
QUACK = tuple(k for k in BASELINES if k.startswith("quack"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("baselines", nargs="+", choices=list(BASELINES))
    p.add_argument("--model", choices=list(MODEL_CONFIGS), default="dsv3", help="shape: H, I, E, top-k (MODEL_CONFIGS)")
    p.add_argument("--t-local", type=int, default=8192)
    p.add_argument("--comm-sms", type=int, default=16, help="quack_inter: SMs of the push's partition; the GEMM gets the rest")
    p.add_argument("--presatisfied", action="store_true", help="also time the quack arms with their flags held at target (see the header)")
    p.add_argument("--ref_check", action="store_true", help="each arm's fresh output (zeroed first) vs the fp32 reference")
    p.add_argument("--r2r_check", action="store_true", help="run each arm twice; bit-identity of the two outputs")
    args = p.parse_args()

    # Process group, shape, symm-mem workspace, routing and metadata, then a, w2 and y = GEMM2(a, w2) into y_symm,
    # the router scores and the fp32 reference: as bench_combine.py, y computed instead of random.
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
    ws.x_symm.copy_(x)
    topk_idx_g = _ag_routing_decision(ws, topk_idx)
    ws.x_hdl.barrier()
    meta = compute_dispatch_metadata(topk_idx_g, my_rank=rank, E_local=E_local, emit_combine=True)
    meta["single_row"] = _build_single_row(topk_idx_g, meta["combine_single_k"], rank, E_local)  # rank-dedup gather
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
    cu = meta["expert_frequency_offset"]
    total_m = int(cu[E_local])
    y = ws.y_symm[:total_m]
    a = torch.randn(total_m, I, generator=g, device=dev, dtype=torch.bfloat16)
    w2 = torch.randn(E_local, H, I, generator=g, device=dev, dtype=torch.bfloat16) * 0.02  # (l, n, k)
    fn, block_rows, tile_n = compile_gemm2(torch2cute_dtype_map[torch.bfloat16])
    flags = torch.zeros(E_local, -(-total_m // block_rows), dtype=torch.int32, device=dev)
    flag_target = CLUSTER[0] * CLUSTER[1] * -(-H // tile_n)  # every CTA of the cluster signals each N tile
    # The scheduler's rotation key: sonic's row -> global slot id, ascending inside each expert; the GEMM then stores
    # each expert's clusters starting at the peer after this rank, the order the push walks them in.
    slot_of_row = meta["s_scatter_idx"][:total_m]
    assert torch.equal(ws.s_rev_symm[slot_of_row.long()], torch.arange(total_m, dtype=torch.int32, device=dev))
    first_src_row = torch.tensor([((rank + 1) % W) * T_local * topk], dtype=torch.int32, device=dev)
    rt = gemm2_args(cu, flags, slot_of_row, first_src_row)
    gemm2 = SimpleNamespace(
        fn=fn, a=a, w2=w2, cu=cu, slot_of_row=slot_of_row, first_src_row=first_src_row, flags=flags,
        block_rows=block_rows, tile_n=tile_n, flag_target=flag_target, run=lambda: fn(a, w2, y, None, *rt, None, None),
    )
    gemm2.run()
    torch.cuda.synchronize()
    rows = (cu[1 : E_local + 1] - cu[:E_local]).long()[:, None]
    want = (torch.arange(flags.shape[1], device=dev)[None, :] < -(-rows // block_rows)).int() * flag_target
    assert torch.equal(flags, want), "GEMM2's combine flags are not cluster size x N tiles per M cluster"
    # Router scores: fp32 softmax over the token's K logits, as sonic's TC softmax-topk router emits; every rank's
    # gathered flat, indexed by the global slot id (sonic's _all_gather_topk_scores).
    scores = torch.rand(T_local, topk, generator=g, device=dev).softmax(-1)
    scores_g = torch.empty(W * T_local * topk, dtype=scores.dtype, device=dev)
    dist.all_gather_into_tensor(scores_g, scores.view(-1), group=ws.ep_group)
    ws._ensure_a2a_peer_base()
    ws._ensure_partial_combine_buf()
    ws.o_hdl.barrier()
    # Reference in fp32, following sonic's A2A combine: slot (t, k) is row s_rev[pos] of peer my_dst_rank[t, k].
    dst = meta["my_dst_rank"].view(-1)
    pos = rank * T_local * topk + torch.arange(T_local * topk, device=dev)
    ref = torch.zeros(T_local, H, dtype=torch.float32, device=dev)
    for q in range(W):
        sel = (dst == q).nonzero().flatten()
        peer_rows = ws.s_rev_peer_bufs[q][pos[sel]].long()
        ref.index_add_(0, sel // topk, ws.y_peer_bufs[q][peer_rows].float() * scores.view(-1)[sel, None])
    shared = (model, y, scores, scores_g, topk_idx_g, meta, ws, gemm2, rank, W, T_local, args)

    canary = BASELINES["sonic"](*shared)
    kerns = {}
    for name in args.baselines:
        kerns.update(canary if name == "sonic" else BASELINES[name](*shared))
    if "sonic" not in kerns:
        kerns["sonic"] = canary["sonic"]  # timed next to the baselines: the contention canary

    checks = {name: {} for name in kerns}
    for name, kern in kerns.items():
        if kern.ref is None:
            continue
        if args.ref_check:
            checks[name]["ref"] = ref_check(kern, ref)
        if args.r2r_check:
            checks[name]["r2r"] = r2r_check(kern)

    arms = {name: kern.run for name, kern in kerns.items()}
    ms = bench_all(arms, ws.o_hdl)
    unstable = [None] * W  # rerun set unioned over ranks: every rank must call the same collectives
    dist.all_gather_object(unstable, [k for k, v in ms.items() if arm_stats(v)[3]])
    rerun = [k for k in arms if any(k in u for u in unstable)]
    if rerun:
        again = bench_all({k: arms[k] for k in dict.fromkeys(rerun + ["sonic"])}, ws.o_hdl)
        ms.update({k: v for k, v in again.items() if not arm_stats(v)[3]})

    flops = 2 * total_m * H * I
    # Wire rows for this rank's routing: routed = off-rank (token, k) slots (the push); dedup = present off-rank
    # peers (sonic's combine).
    present = meta["peer_present_mask"]  # (W, T_local)
    nbytes = int((meta["my_dst_rank"] != rank).sum()) * H * 2
    nbytes_dedup = int(present.sum() - present[rank].sum()) * H * 2
    stats = {k: arm_stats(v) for k, v in ms.items()}
    torch.cuda.synchronize()
    halves = {  # per arm with legs: median (comm ms, gemm ms) over every recorded call, all at the same load
        k: (statistics.median(e[0].elapsed_time(e[1]) for e in kern.halves), statistics.median(e[2].elapsed_time(e[3]) for e in kern.halves))
        for k, kern in kerns.items() if kern.halves
    }
    gathered = [None] * W
    dist.all_gather_object(gathered, (stats, checks, halves))
    if rank == 0:
        med = {k: max((s[k] for s, _, _ in gathered), key=lambda t: t[0]) for k in ms}  # slowest rank sets the collective
        print(
            f"<bf16 fused GEMM2 + combine {args.model} (H={H} I={I} E={E} top-{topk}) | W={W} T_local={T_local} "
            f"total_m={total_m} wire {nbytes / 1e6:.0f} MB | tile {TILE[0]}x{TILE[1]}({CLUSTER[0]},{CLUSTER[1]}) static, "
            f"M cluster {block_rows} rows, flag target {flag_target} | warm {WARM_S * 1e3:.0f} ms, "
            f"{REPS} x {WINDOW_S * 1e3:.0f} ms blocks, rerun={rerun}>"
        )
        for k, (m, lo, hi, _) in med.items():
            mean = sum(s[k][0] for s, _, _ in gathered) / W
            flag = "  UNSTABLE" if any(s[k][3] for s, _, _ in gathered) else ""
            vs = f"  vs sonic {m / med['sonic'][0] - 1:+.1%}" if k in QUACK else ""
            rate = "" if k == "rmsnorm pass" else f"{nbytes / m / 1e6:5.0f} GB/s" if k == "sonic_a2a" else f"{flops / m / 1e9:5.0f} TFLOP/s"
            print(
                f"  {k + kerns[k].label + ':':<44} {m * 1e3:7.0f} us  [{lo * 1e3:.0f}-{hi * 1e3:.0f}]  "
                f"{rate}  rank-mean {mean * 1e3:.0f} us{vs}{flag}"
            )
        if "sonic" in halves:  # the chain's legs: combine (dedup rows), GEMM alone on all SMs
            comm = max(h["sonic"][0] for _, _, h in gathered)
            gemm = max(h["sonic"][1] for _, _, h in gathered)
            print(f"  sonic: combine (dedup) {comm * 1e3:.0f} us ({nbytes_dedup / comm / 1e6:.0f} GB/s on dedup rows)  gemm {gemm * 1e3:.0f} us")
        if "quack_inter/presat" in halves:  # two halves under one power budget, concurrent
            comm = max(h["quack_inter/presat"][0] for _, _, h in gathered)
            gemm = max(h["quack_inter/presat"][1] for _, _, h in gathered)
            bound, fused = max(comm, gemm), med["quack_inter"][0]
            red = med["quack_inter/reduce"][0]
            print(
                f"  quack_inter: push {comm * 1e3:.0f} us ({nbytes / comm / 1e6:.0f} GB/s)  gemm {gemm * 1e3:.0f} us  "
                f"bound = max = {bound * 1e3:.0f} us  reduce alone {red * 1e3:.0f} us  bound + reduce {(bound + red) * 1e3:.0f} us  "
                f"fused {fused * 1e3:.0f} us  vs bound {fused / bound - 1:+.1%}  vs bound + reduce {fused / (bound + red) - 1:+.1%}"
            )
        for name in kerns:
            for what in checks[name]:
                ok = all(c[name][what][0] for _, c, _ in gathered)
                err = max(c[name][what][1] for _, c, _ in gathered)
                print(f"  {name} {what}: {'ok' if ok else 'MISMATCH'} max rel err {err:.3e}")
    for kern in kerns.values():
        kern.free()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
