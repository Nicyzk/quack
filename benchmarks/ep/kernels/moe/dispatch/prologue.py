# The quack_intra_swap arm's routing prologue in one launch: the top-k exchange (each rank stores its block into every
# peer's gathered buffer, then an NVLink barrier that also orders the x publish issued before it on the stream), the
# per-forward flag / semaphore resets, the routing metadata (routing_metadata.py's count + emit, plus each row's
# combine slot address for GEMM2's epilogue store) and the dispatch walk (walk.py's count, from the token side, +
# scatter), as phases on 128-thread CTAs over every SM, separated by software grid syncs. Replaces a NCCL all-gather,
# a symm-mem barrier, four memsets and five dependent launches (results/2026-10-05/latency_optimizations.md). Even-split
# blocks only (the swap-AB GEMMs).
import os
import sys
from types import SimpleNamespace

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
import torch.distributed._symmetric_memory as symm_mem
from cutlass.cutlass_dsl import Int32, Int64
from quack.compile_utils import make_fake_tensor
from quack.utils import warp_prefix_sum
from quack.varlen_utils import VARLEN_N_GRANULE, varlen_n_block_rows

from dispatch.walk import (
    GROUP_THREADS,
    NUM_WARPS,
    WalkArguments,
    fake_walk_arguments,
    make_walk_buffers,
    walk_scatter_phase,
    walk_scheduler_args,
)

from quack.ep.ptx import red_add_release_sys_s32 as red_add_release_sys_s32_raw  # same PTX as the megamoe helpers
from quack.ep.ptx import software_grid_sync, stg_b32_raw

SYNC_BARRIER_ID = 12  # the CTA's barrier (10 = software_grid_sync's)


@cute.jit
def group_sync():
    cute.arch.barrier(barrier_id=SYNC_BARRIER_ID, number_of_threads=GROUP_THREADS)


@cute.jit
def publish_topk(topk: cute.Tensor, topk_g_base: cute.Tensor, rank: Int32, cta: Int32, num_ctas: Int32, tid: Int32,
                 W: cutlass.Constexpr[int]):
    """This rank's (T * K,) top-k into every rank's gathered buffer at block `rank` (its own included)."""
    n = Int32(cute.size(topk))
    i = cta * Int32(GROUP_THREADS) + tid
    while i < n:
        v = topk[i]
        for w in cutlass.range_constexpr(W):
            stg_b32_raw(topk_g_base[w] + Int64(rank * n + i) * Int64(4), v)
        i = i + num_ctas * Int32(GROUP_THREADS)


@cute.jit
def zero_flags(flags: cute.Tensor, sem1: cute.Tensor, sem2: cute.Tensor, cta: Int32, num_ctas: Int32, tid: Int32):
    """The per-forward resets: GEMM1's dispatch flags and both GEMMs' dynamic-persistence semaphores."""
    ncols = Int32(cute.size(flags, mode=[1]))
    n = Int32(cute.size(flags))
    i = cta * Int32(GROUP_THREADS) + tid
    while i < n:
        flags[(i // ncols, i % ncols)] = Int32(0)
        i = i + num_ctas * Int32(GROUP_THREADS)
    if (cta == Int32(0)) & (tid == Int32(0)):
        sem1[0] = Int32(0)
        sem2[0] = Int32(0)


@cute.jit
def nvlink_barrier(signal: cute.Tensor, signal_base: cute.Tensor, barrier_counter: cute.Tensor, cta: Int32,
                   lane: Int32, W: cutlass.Constexpr[int]):
    """Warp 0 of every CTA, after a grid sync: CTA 0 arrives on every rank's slot, every CTA spins on its own rank's
    slot (no second grid sync); sense-reversing ping-pong over two slots (MegaMoE's routed barrier) on the
    CTA's own counter word, release / acquire at system scope."""
    status = barrier_counter[cta] & Int32(3)
    phase = status & Int32(1)
    delta, target = Int32(1), Int32(W)
    if (status >> Int32(1)) != Int32(0):
        delta, target = Int32(-1), Int32(0)
    if (cta == Int32(0)) & (lane < Int32(W)):
        red_add_release_sys_s32_raw(signal_base[lane] + Int64(phase) * Int64(4), delta)
    cute.arch.sync_warp()
    if lane == Int32(0):
        barrier_counter[cta] = status + Int32(1)
        while cute.arch.load(signal.iterator + phase, Int32, sem="acquire", scope="sys") != target:
            pass
    cute.arch.sync_warp()


@cute.jit
def tile_slot_masks(topk_g: cute.Tensor, tile: Int32, tid: Int32, lane: Int32, warp: Int32, rank: Int32,
                    masks: cute.Tensor, local: cute.Tensor, K: cutlass.Constexpr[int],
                    E_local: cutlass.Constexpr[int]) -> Int32:
    """Thread tid's token (tile * GROUP_THREADS + tid): local[k] = its slot k's local expert, returned = the bitmask
    of the slots destined to this rank; masks[(e, warp)] (smem, E_local * NUM_WARPS words) = the tile's tokens of
    expert e, one bit per lane, so a slot's rank among the tile's slots of its expert (f order) is a popcount."""
    token = tile * Int32(GROUP_THREADS) + tid
    i = tid
    while i < Int32(E_local * NUM_WARPS):
        masks[i] = Int32(0)
        i = i + Int32(GROUP_THREADS)
    group_sync()
    mask = Int32(0)
    for k in cutlass.range_constexpr(K):
        e = topk_g[token * Int32(K) + Int32(k)] - rank * Int32(E_local)
        local[k] = e
        if (e >= Int32(0)) & (e < Int32(E_local)):
            mask = mask | Int32(1 << k)
            cute.arch.atomic_or(masks.iterator + e * Int32(NUM_WARPS) + warp, Int32(1) << lane, sem="relaxed", scope="cta")
    group_sync()
    return mask


@cute.jit
def slot_rank(masks: cute.Tensor, e: Int32, lane: Int32, warp: Int32) -> Int32:
    """Rank of this lane's token among the tile's tokens of expert e (tile_slot_masks' masks)."""
    r = cute.arch.popc(masks[e * Int32(NUM_WARPS) + warp] & ((Int32(1) << lane) - Int32(1)))
    for w in cutlass.range_constexpr(NUM_WARPS):
        if Int32(w) < warp:
            r = r + cute.arch.popc(masks[e * Int32(NUM_WARPS) + Int32(w)])
    return r


@cute.jit
def count_phase(topk_g: cute.Tensor, counts: cute.Tensor, counts2: cute.Tensor, rank: Int32, cta: Int32,
                num_ctas: Int32, tid: Int32, lane: Int32, warp: Int32, scratch: cute.Tensor, masks: cute.Tensor,
                K: cutlass.Constexpr[int], E_local: cutlass.Constexpr[int]):
    """counts[e, tile] = the tile's slots of local expert e destined to this rank; counts[E_local, tile] = its tokens
    with >= 1 such slot (routing_metadata's count_kernel over GROUP_THREADS-token tiles, transposed)."""
    local = cute.make_rmem_tensor(cute.make_layout(K), Int32)
    n_tiles = Int32(cute.size(counts, mode=[1]))
    n2 = Int32(cute.size(counts2, mode=[1]))  # the per-block owner / reference counts emit accumulates: zeroed here
    i = cta * Int32(GROUP_THREADS) + tid
    while i < n2:
        counts2[0, i] = Int32(0)
        counts2[1, i] = Int32(0)
        i = i + num_ctas * Int32(GROUP_THREADS)
    tile = cta
    while tile < n_tiles:
        mask = tile_slot_masks(topk_g, tile, tid, lane, warp, rank, masks, local, K, E_local)
        present_ballot = cute.arch.vote_ballot_sync(mask != Int32(0))
        if lane == Int32(0):
            scratch[warp] = cute.arch.popc(present_ballot)
        group_sync()
        e = tid
        while e <= Int32(E_local):
            total = Int32(0)
            for w in cutlass.range_constexpr(NUM_WARPS):
                if e < Int32(E_local):
                    total = total + cute.arch.popc(masks[e * Int32(NUM_WARPS) + Int32(w)])
                else:
                    total = total + scratch[w]
            counts[e, tile] = total
            e = e + Int32(GROUP_THREADS)
        group_sync()
        tile = tile + num_ctas


@cute.jit
def expert_tile_sums(counts: cute.Tensor, upto: Int32, lane: Int32, warp: Int32, out_s: cute.Tensor,
                     E_local: cutlass.Constexpr[int]):
    """out_s[e] = sum of counts[e, j] over tiles j < upto, for e <= E_local: warp w takes experts w, w + NUM_WARPS,
    ..., lanes over tiles; per 32-tile pass every expert's load is issued before any is consumed (one round trip)."""
    EPW = (E_local + 1 + NUM_WARPS - 1) // NUM_WARPS
    acc = cute.make_rmem_tensor(cute.make_layout(EPW), Int32)
    for i in cutlass.range_constexpr(EPW):
        acc[i] = Int32(0)
    for p in cutlass.range((upto + Int32(31)) // Int32(32), unroll=1):
        j = p * Int32(32) + lane
        v = cute.make_rmem_tensor(cute.make_layout(EPW), Int32)
        for i in cutlass.range_constexpr(EPW):
            e = Int32(i * NUM_WARPS) + warp
            jc = j
            if (j >= upto) | (e > Int32(E_local)):
                jc = Int32(0)
            ec = e
            if e > Int32(E_local):
                ec = Int32(0)
            v[i] = counts[ec, jc]  # clamped, unconditional: the loads overlap
        for i in cutlass.range_constexpr(EPW):
            e = Int32(i * NUM_WARPS) + warp
            if (j < upto) & (e <= Int32(E_local)):
                acc[i] = acc[i] + v[i]
    for i in cutlass.range_constexpr(EPW):
        e = Int32(i * NUM_WARPS) + warp
        total = acc[i]
        for s in cutlass.range_constexpr(5):
            total = total + cute.arch.shuffle_sync_bfly(total, offset=1 << s)
        if (lane == Int32(0)) & (e <= Int32(E_local)):
            out_s[e] = total


@cute.jit
def emit_phase(topk_g: cute.Tensor, counts: cute.Tensor, cu: cute.Tensor, s_reverse: cute.Tensor,
               s_scatter: cute.Tensor, a_idx: cute.Tensor, slot_addr: cute.Tensor, slot_base: cute.Tensor,
               first_src_row: cute.Tensor, rank: Int32, first_src_rank: Int32, tokens_per_rank: Int32, world: Int32,
               bytes_per_token: Int32, slot_bytes: Int32, cta: Int32, num_ctas: Int32, tid: Int32, lane: Int32, warp: Int32,
               scratch: cute.Tensor, masks: cute.Tensor, cu_s: cute.Tensor, before_s: cute.Tensor, counts2: cute.Tensor,
               ncols: Int32, block_rows: cutlass.Constexpr[int], K: cutlass.Constexpr[int],
               E_local: cutlass.Constexpr[int]):
    """The tables (routing_metadata's emit_kernel): cu, first_src_row (CTA 0), and per tile s_reverse, s_scatter,
    a_idx, slot_addr (row -> its token's combine slot on the origin rank, GEMM2's epilogue store); plus the dispatch
    walk's per-block counts (walk_count_phase, from the token side): a present token's owner row is its lowest local expert's
    row, so counts2[0, e * ncols + cid] += 1 and counts2[1, ...] += its local slots for the owner row's block.
    cu_s / before_s = (E_local + 1,) smem: the expert offsets (cu_s[E_local] = total rows) and the tile's per-expert
    prefixes (before_s[E_local] = present tokens before the tile)."""
    local = cute.make_rmem_tensor(cute.make_layout(K), Int32)
    n_tiles = Int32(cute.size(counts, mode=[1]))
    n2 = Int32(cute.size(counts2, mode=[1]))
    tiles_per_rank = tokens_per_rank // Int32(GROUP_THREADS)
    slots_per_rank = tokens_per_rank * Int32(K)
    lanes_before = (Int32(1) << lane) - Int32(1)
    expert_tile_sums(counts, n_tiles, lane, warp, cu_s, E_local)  # per-expert totals
    if (cta == Int32(0)) & (warp == Int32(1)):  # first_src_row: tokens present from the ranks before first_src_rank
        upto = first_src_rank * tiles_per_rank
        first = Int32(0)
        for p in cutlass.range((upto + Int32(31)) // Int32(32), unroll=1):
            j = p * Int32(32) + lane
            v = Int32(0)
            if j < upto:
                v = counts[E_local, j]
            first = first + v
        for s in cutlass.range_constexpr(5):
            first = first + cute.arch.shuffle_sync_bfly(first, offset=1 << s)
        if lane == Int32(0):
            first_src_row[0] = first
    group_sync()
    if warp == Int32(0):  # exclusive prefix over experts: warp scan, lanes over experts
        carry = Int32(0)
        for p in cutlass.range_constexpr((E_local + 1 + 31) // 32):
            e = Int32(p * 32) + lane
            v = Int32(0)
            if e < Int32(E_local):
                v = cu_s[e]
            inclusive = warp_prefix_sum(v, lane)
            if e <= Int32(E_local):
                cu_s[e] = carry + inclusive - v
            carry = carry + cute.arch.shuffle_sync(inclusive, 31)
    group_sync()
    if cta == Int32(0):
        e = tid
        while e <= Int32(E_local):
            cu[e] = cu_s[e]
            e = e + Int32(GROUP_THREADS)
    tile = cta
    while tile < n_tiles:
        expert_tile_sums(counts, tile, lane, warp, before_s, E_local)  # prefixes over the tiles before this one
        mask = tile_slot_masks(topk_g, tile, tid, lane, warp, rank, masks, local, K, E_local)
        present_ballot = cute.arch.vote_ballot_sync(mask != Int32(0))
        if lane == Int32(0):
            scratch[warp] = cute.arch.popc(present_ballot)
        group_sync()
        token = tile * Int32(GROUP_THREADS) + tid
        pool_row = before_s[E_local] + cute.arch.popc(present_ballot & lanes_before)  # present before tile, warp, lane
        for w in cutlass.range_constexpr(NUM_WARPS):
            if Int32(w) < warp:
                pool_row = pool_row + scratch[w]
        src_rank = token // tokens_per_rank
        slot0 = slot_base[src_rank] + Int64((token - src_rank * tokens_per_rank) * Int32(K)) * Int64(slot_bytes)
        e_min, row_min = Int32(0x7FFFFFFF), Int32(0)
        for k in cutlass.range_constexpr(K):
            if ((mask >> k) & Int32(1)) == Int32(1):
                e_k = local[k]
                f = token * Int32(K) + Int32(k)
                row = cu_s[e_k] + before_s[e_k] + slot_rank(masks, e_k, lane, warp)
                s_reverse[f] = row
                s_scatter[row] = f
                a_idx[row] = pool_row
                slot_addr[row] = slot0 + Int64(k) * Int64(slot_bytes)
                if e_k < e_min:
                    e_min, row_min = e_k, row
        if mask != Int32(0):  # the owner row's block gets this token's owned row and its reader references
            b = varlen_n_block_rows(cu_s[e_min + Int32(1)] - cu_s[e_min], block_rows, VARLEN_N_GRANULE)
            slot = e_min * ncols + (row_min - cu_s[e_min]) // b
            cute.arch.atomic_add(counts2.iterator + slot, Int32(1), sem="relaxed", scope="gpu")
            cute.arch.atomic_add(counts2.iterator + n2 + slot, cute.arch.popc(mask), sem="relaxed", scope="gpu")
        group_sync()
        tile = tile + num_ctas


class Prologue:
    """The launch: num_ctas CTAs of GROUP_THREADS threads, every SM (the grid syncs need them all resident)."""

    def __init__(self, K, E_local, W, block_rows, num_ctas, sf_atom_rows=0):
        self.K, self.E_local, self.W = K, E_local, W
        self.block_rows, self.num_ctas = block_rows, num_ctas
        self.sf_atom_rows = sf_atom_rows  # MXFP8: SF rows per block slot (0: no scales)
        self.smem_bytes = 4 * (NUM_WARPS + E_local * NUM_WARPS + 2 * (E_local + 1)) + 256

    @cute.jit
    def __call__(
        self, topk, topk_g, topk_g_base, signal, signal_base, barrier_counter, grid_counter, counts, cu, s_reverse,
        s_scatter, a_idx, slot_addr, slot_base, first_src_row, flags, counts2, peer_base, walk: WalkArguments, sem1,
        sem2, rank: Int32, first_src_rank: Int32, tokens_per_rank: Int32, bytes_per_token: Int32, slot_bytes: Int32,
        stream, peer_sf_base=None, sf_bytes_per_token: Int32 = Int32(0),
    ):
        sched1 = walk_scheduler_args(a_idx, cu, flags, first_src_row, self.block_rows, True)  # GEMM1's dispatch order
        self.kernel(
            sched1, topk, topk_g, topk_g_base, signal, signal_base, barrier_counter, grid_counter, counts, cu,
            s_reverse, s_scatter, a_idx, slot_addr, slot_base, first_src_row, flags, counts2, peer_base, walk, sem1,
            sem2, rank, first_src_rank, tokens_per_rank, bytes_per_token, slot_bytes, peer_sf_base, sf_bytes_per_token,
        ).launch(grid=[self.num_ctas, 1, 1], block=[GROUP_THREADS, 1, 1], smem=self.smem_bytes, stream=stream)

    @cute.kernel
    def kernel(
        self, sched1, topk, topk_g, topk_g_base, signal, signal_base, barrier_counter, grid_counter, counts, cu,
        s_reverse, s_scatter, a_idx, slot_addr, slot_base, first_src_row, flags, counts2, peer_base,
        walk: WalkArguments, sem1, sem2, rank: Int32, first_src_rank: Int32, tokens_per_rank: Int32,
        bytes_per_token: Int32, slot_bytes: Int32, peer_sf_base, sf_bytes_per_token: Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        warp, lane = tid // 32, tid % 32
        num_ctas, _, _ = cute.arch.grid_dim()  # dynamic: a static count derefines inside the phases
        smem = utils.SmemAllocator()
        scratch = smem.allocate_tensor(Int32, cute.make_layout(NUM_WARPS), byte_alignment=16)
        masks = smem.allocate_tensor(Int32, cute.make_layout(self.E_local * NUM_WARPS), byte_alignment=16)
        cu_s = smem.allocate_tensor(Int32, cute.make_layout(self.E_local + 1), byte_alignment=16)
        before_s = smem.allocate_tensor(Int32, cute.make_layout(self.E_local + 1), byte_alignment=16)
        K, E_local, W = self.K, self.E_local, self.W
        publish_topk(topk, topk_g_base, rank, cta, num_ctas, tid, W)
        zero_flags(flags, sem1, sem2, cta, num_ctas, tid)
        software_grid_sync(grid_counter, cta, num_ctas, tid, num_threads=GROUP_THREADS)  # every CTA's stores issued
        if warp == Int32(0):
            cute.arch.fence_acq_rel_sys()
            nvlink_barrier(signal, signal_base, barrier_counter, cta, lane, W)
        group_sync()
        count_phase(topk_g, counts, counts2, rank, cta, num_ctas, tid, lane, warp, scratch, masks, K, E_local)
        software_grid_sync(grid_counter, cta, num_ctas, tid, num_threads=GROUP_THREADS)
        emit_phase(
            topk_g, counts, cu, s_reverse, s_scatter, a_idx, slot_addr, slot_base, first_src_row, rank, first_src_rank,
            tokens_per_rank, Int32(W), bytes_per_token, slot_bytes, cta, num_ctas, tid, lane, warp, scratch, masks,
            cu_s, before_s, counts2, Int32(cute.size(flags, mode=[1])), self.block_rows, K, E_local,
        )
        software_grid_sync(grid_counter, cta, num_ctas, tid, num_threads=GROUP_THREADS)
        walk_scatter_phase(
            sched1, a_idx, s_scatter, s_reverse, topk_g, cu, flags, counts2, peer_base, walk, rank, tokens_per_rank,
            bytes_per_token, cta, num_ctas, tid, lane, warp, scratch, self.block_rows, K, True, peer_sf_base,
            sf_bytes_per_token, self.sf_atom_rows,
        )


def make_prologue(
    topk_idx, x_peer_bufs, rank, W, T_local, K, E_local, max_rows, block_rows, slot_base, group, x_sf_peer_bufs=None,
    slot_row_bytes=None, first_src_rank=None,
):
    """Host side: (run, meta, walk1). run() = the one launch (the x publish goes before it on the stream; the
    kernel's NVLink barrier orders it for the peers' pulls); run.sem1 / run.sem2 = the GEMMs' dynamic-persistence
    semaphores, zeroed by the kernel each forward. meta = routing_metadata's tables (cu, s_reverse, s_scatter, a_idx,
    first_src_row) plus slot_addr (per row, its combine slot's address: GEMM2's epilogue store); walk1 =
    make_walk_buffers' walk (with x_sf_peer_bufs: the block-scaled scale lane). block_rows = the swap GEMMs' tile N
    (the even-split unit), slot_base = make_combine_buffers'; slot_row_bytes = one combine slot's bytes (default: a
    bf16 row of x's width; pass it when x's rows are not bf16, e.g. fp8). first_src_rank = the source rank each
    expert's walk starts at (default the peer after this rank, the ring; 0 = ascending, bench_ablations.py)."""
    dev = topk_idx.device
    if first_src_rank is None:
        first_src_rank = (rank + 1) % W
    if slot_row_bytes is None:
        slot_row_bytes = x_peer_bufs[rank].shape[1] * 2
    assert topk_idx.dtype == torch.int32 and topk_idx.shape == (T_local, K)
    assert T_local % GROUP_THREADS == 0, "tiles must not straddle source ranks"
    TK = T_local * K
    n_tiles = W * T_local // GROUP_THREADS
    ncols = -(-W * T_local // block_rows)  # blocks per expert <= the pool's
    i32 = lambda *shape: torch.empty(*shape, dtype=torch.int32, device=dev)
    i64 = lambda *shape: torch.empty(*shape, dtype=torch.int64, device=dev)
    num_ctas = torch.cuda.get_device_properties(dev).multi_processor_count
    topk_g = symm_mem.empty(W * TK, dtype=torch.int32, device=dev)
    signal = symm_mem.empty(2, dtype=torch.int32, device=dev)
    hdls = [symm_mem.rendezvous(t, group=group) for t in (topk_g, signal)]
    bases = lambda hdl, t: torch.tensor(
        [hdl.get_buffer(q, t.shape, t.dtype).data_ptr() for q in range(W)], dtype=torch.int64, device=dev
    )
    signal.zero_()
    sync = SimpleNamespace(
        topk_g=topk_g, topk_g_base=bases(hdls[0], topk_g), signal=signal, signal_base=bases(hdls[1], signal),
        barrier_counter=torch.zeros(num_ctas, dtype=torch.int32, device=dev), grid_counter=torch.zeros(1, dtype=torch.int32, device=dev),
        counts=i32(E_local + 1, n_tiles), counts2=i32(2, E_local * ncols), hdls=hdls,  # counts[E_local] = present tokens
    )
    meta = SimpleNamespace(
        cu=i32(E_local + 1), s_reverse=i32(W * TK), s_scatter=i32(max_rows), a_idx=i32(max_rows), slot_addr=i64(max_rows),
        first_src_row=i32(1),
    )
    flags = torch.zeros(E_local, ncols, dtype=torch.int32, device=dev)
    sf = x_sf_peer_bufs is not None
    walk1 = make_walk_buffers(W, T_local, max_rows, flags, meta.first_src_row, block_rows, True, sf)
    peer_base = torch.tensor([b.data_ptr() for b in x_peer_bufs], dtype=torch.int64, device=dev)
    row_bytes = x_peer_bufs[rank].shape[1] * x_peer_bufs[rank].element_size()
    sem1, sem2 = torch.zeros(1, dtype=torch.int32, device=dev), torch.zeros(1, dtype=torch.int32, device=dev)
    kernel = Prologue(K, E_local, W, block_rows, num_ctas, walk1.sf_atom_rows)
    v = lambda: make_fake_tensor(Int32, (cute.sym_int(),), leading_dim=0, divisibility=1)
    v64 = lambda: make_fake_tensor(Int64, (cute.sym_int(),), leading_dim=0, divisibility=1)
    m = lambda: make_fake_tensor(Int32, (cute.sym_int(), cute.sym_int()), leading_dim=1, divisibility=1)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    sf_fake, sf_arg, sf_row_bytes = None, None, 0
    if sf:
        sf_fake = v64()
        sf_arg = torch.tensor([b.data_ptr() for b in x_sf_peer_bufs], dtype=torch.int64, device=dev)
        sf_row_bytes = x_sf_peer_bufs[rank].shape[1] * x_sf_peer_bufs[rank].element_size()
    fn = cute.compile(
        kernel, v(), v(), v64(), v(), v64(), v(), v(), m(), v(), v(), v(), v(), v64(), v64(), v(), m(), m(), v64(),
        fake_walk_arguments(sf), v(), v(), Int32(rank), Int32(first_src_rank), Int32(T_local), Int32(row_bytes), Int32(slot_row_bytes),
        stream, sf_fake, Int32(sf_row_bytes), options="--enable-tvm-ffi",
    )
    run = lambda: fn(
        topk_idx.view(-1), topk_g, sync.topk_g_base, signal, sync.signal_base, sync.barrier_counter, sync.grid_counter,
        sync.counts, meta.cu, meta.s_reverse, meta.s_scatter, meta.a_idx, meta.slot_addr, slot_base, meta.first_src_row,
        flags, sync.counts2, peer_base, walk1.args, sem1, sem2, rank, first_src_rank, T_local, row_bytes, slot_row_bytes, sf_arg,
        sf_row_bytes,
    )
    run.sync = sync
    run.sem1, run.sem2 = sem1, sem2
    return run, meta, walk1
