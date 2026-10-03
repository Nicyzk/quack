# The quack_intra_swap arm's routing prologue in one launch: the top-k exchange (each rank stores its block into every
# peer's gathered buffer, then an NVLink barrier that also orders the x publish issued before it on the stream), the
# per-forward flag / semaphore resets, the routing metadata (routing_metadata.py's count + emit, plus each row's
# combine slot address for GEMM2's epilogue store) and the dispatch walk (dispatch_pull.DispatchWalk's count +
# scatter), as phases on 128-thread CTAs over every SM, separated by software grid syncs. Replaces a NCCL all-gather,
# a symm-mem barrier, four memsets and five dependent launches. Even-split blocks only (the swap-AB GEMMs).
from types import SimpleNamespace

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
import torch.distributed._symmetric_memory as symm_mem
from cutlass.cutlass_dsl import Int32, Int64
from quack.compile_utils import make_fake_tensor
from quack.ep.ptx import red_add_release_sys_s32, software_grid_sync, stg_b32_raw
from quack.gemm_sm100_ep_inter import MClusterWalk, block_rows_of
from quack.tile_scheduler import (
    DispatchSchedulerArguments,
    PersistenceMode,
    RasterOrderOption,
    VarlenMTileSchedulerArguments,
)
from quack.utils import warp_prefix_sum
from quack.varlen_utils import VARLEN_N_GRANULE, varlen_n_block_rows

GROUP_THREADS = 128
NUM_WARPS = GROUP_THREADS // 32
SYNC_BARRIER_ID = 12  # the CTA's barrier (10 = software_grid_sync's, 11 = dispatch -> scheduler)


@cute.jit
def group_sync():
    cute.arch.barrier(barrier_id=SYNC_BARRIER_ID, number_of_threads=GROUP_THREADS)


@cute.jit
def group_sum(val: Int32, scratch: cute.Tensor, warp: Int32, lane: Int32) -> Int32:
    """Sum of val over the CTA, to every thread; scratch = (NUM_WARPS,) smem."""
    for i in cutlass.range_constexpr(5):
        val = val + cute.arch.shuffle_sync_bfly(val, offset=1 << i)
    if lane == Int32(0):
        scratch[warp] = val
    group_sync()
    total = Int32(0)
    for w in cutlass.range_constexpr(NUM_WARPS):
        total = total + scratch[w]
    group_sync()
    return total


@cute.jit
def group_exclusive_scan(val: Int32, scratch: cute.Tensor, warp: Int32, lane: Int32) -> Int32:
    """Exclusive prefix sum of val over the CTA in thread order."""
    inclusive = warp_prefix_sum(val, lane)
    if lane == Int32(31):
        scratch[warp] = inclusive
    group_sync()
    before = Int32(0)
    for w in cutlass.range_constexpr(NUM_WARPS):
        if Int32(w) < warp:
            before = before + scratch[w]
    group_sync()
    return before + inclusive - val


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
    slot (no second grid sync); sense-reversing ping-pong over two slots (routed_dispatch.routed_barrier's) on the
    CTA's own counter word, release / acquire at system scope."""
    status = barrier_counter[cta] & Int32(3)
    phase = status & Int32(1)
    delta, target = Int32(1), Int32(W)
    if (status >> Int32(1)) != Int32(0):
        delta, target = Int32(-1), Int32(0)
    if (cta == Int32(0)) & (lane < Int32(W)):
        red_add_release_sys_s32(signal_base[lane] + Int64(phase) * Int64(4), delta)
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
               first_src_row: cute.Tensor, rank: Int32, tokens_per_rank: Int32, world: Int32, bytes_per_token: Int32,
               cta: Int32, num_ctas: Int32, tid: Int32, lane: Int32, warp: Int32, scratch: cute.Tensor,
               masks: cute.Tensor, cu_s: cute.Tensor, before_s: cute.Tensor, counts2: cute.Tensor, ncols: Int32,
               block_rows: cutlass.Constexpr[int], K: cutlass.Constexpr[int], E_local: cutlass.Constexpr[int]):
    """The tables (routing_metadata's emit_kernel): cu, first_src_row (CTA 0), and per tile s_reverse, s_scatter,
    a_idx, slot_addr (row -> its token's combine slot on the origin rank, GEMM2's epilogue store); plus the dispatch
    walk's per-block counts (DispatchWalk.count_kernel): a present token's owner row is its lowest local expert's
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
    if (cta == Int32(0)) & (warp == Int32(1)):  # first_src_row: tokens present from the ranks before (rank + 1) % world
        upto = ((rank + Int32(1)) % world) * tiles_per_rank
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
        slot0 = slot_base[src_rank] + Int64((token - src_rank * tokens_per_rank) * Int32(K)) * Int64(bytes_per_token)
        e_min, row_min = Int32(0x7FFFFFFF), Int32(0)
        for k in cutlass.range_constexpr(K):
            if ((mask >> k) & Int32(1)) == Int32(1):
                e_k = local[k]
                f = token * Int32(K) + Int32(k)
                row = cu_s[e_k] + before_s[e_k] + slot_rank(masks, e_k, lane, warp)
                s_reverse[f] = row
                s_scatter[row] = f
                a_idx[row] = pool_row
                slot_addr[row] = slot0 + Int64(k) * Int64(bytes_per_token)
                if e_k < e_min:
                    e_min, row_min = e_k, row
        if mask != Int32(0):  # the owner row's block gets this token's owned row and its reader references
            b = varlen_n_block_rows(cu_s[e_min + Int32(1)] - cu_s[e_min], block_rows, VARLEN_N_GRANULE)
            slot = e_min * ncols + (row_min - cu_s[e_min]) // b
            cute.arch.atomic_add(counts2.iterator + slot, Int32(1), sem="relaxed", scope="gpu")
            cute.arch.atomic_add(counts2.iterator + n2 + slot, cute.arch.popc(mask), sem="relaxed", scope="gpu")
        group_sync()
        tile = tile + num_ctas


@cute.jit
def owner_test_flat(expert: Int32, cid: Int32, r: Int32, slot_of_row: cute.Tensor, s_reverse: cute.Tensor,
                    topk_g: cute.Tensor, cu: cute.Tensor, flags: cute.Tensor, rank: Int32, num_experts: Int32,
                    reader_flags: cute.Tensor, reader_rows: cute.Tensor, block_rows: cutlass.Constexpr[int],
                    K: cutlass.Constexpr[int]):
    """dispatch_pull.owner_test (even split) with every load unconditional on clamped indices, so two rows' chains
    issue together: row r of block (expert, cid) -> (owner, readers, token, owned, refs)."""
    e_start, e_end = cu[expert], cu[expert + 1]
    b = varlen_n_block_rows(e_end - e_start, block_rows, VARLEN_N_GRANULE)
    start = e_start + cid * b
    row = start + r
    in_cluster = row < cutlass.min(start + b, e_end)
    row_c = row
    if not in_cluster:
        row_c = start  # a scheduled block has a first row
    token = slot_of_row[row_c] // K
    slot0 = token * K
    first_reader, readers = Int32(0x7FFFFFFF), Int32(0)
    for k in cutlass.range_constexpr(K):
        e = topk_g[slot0 + k] - rank * num_experts
        reader_row = s_reverse[slot0 + k]
        mine = (e >= Int32(0)) & (e < num_experts)
        ec = e
        if not mine:
            ec = Int32(0)
        rb = varlen_n_block_rows(cu[ec + 1] - cu[ec], block_rows, VARLEN_N_GRANULE)
        flag = Int32(cute.crd2idx((ec, (reader_row - cu[ec]) // rb), flags.layout))
        reader_flags[k], reader_rows[k] = Int32(0), Int32(0)
        if mine:
            first_reader = cutlass.min(first_reader, reader_row)
            readers = readers | Int32(1 << k)
            reader_flags[k], reader_rows[k] = flag, reader_row
    owner = in_cluster & (first_reader == row)
    owned, refs = Int32(0), Int32(0)
    if owner:
        owned, refs = Int32(1), cute.arch.popc(readers)
    if not in_cluster:
        readers, token = Int32(0), Int32(0)
    return owner, readers, token, owned, refs


@cute.jit
def walk1_scatter_phase(sched_params, a_idx: cute.Tensor, slot_of_row: cute.Tensor, s_reverse: cute.Tensor,
                        topk_g: cute.Tensor, cu: cute.Tensor, flags: cute.Tensor, counts2: cute.Tensor,
                        peer_base: cute.Tensor, pool_row: cute.Tensor, src_address: cute.Tensor, gate_ptr: cute.Tensor,
                        gates: cute.Tensor, dst_row: cute.Tensor, num_rows: cute.Tensor, rank: Int32,
                        tokens_per_rank: Int32, bytes_per_token: Int32, cta: Int32, num_ctas: Int32, tid: Int32,
                        lane: Int32, warp: Int32, scratch: cute.Tensor, block_rows: cutlass.Constexpr[int],
                        K: cutlass.Constexpr[int]):
    """Each block's owned rows at (walk-earlier blocks' owned rows + rank among its owners), in row order
    (DispatchWalk.scatter_kernel); counts2[(0 | 1), e * ncols + cid] = emit's per-block owned rows / references."""
    PASSES = (block_rows + GROUP_THREADS - 1) // GROUP_THREADS
    reader_flags = [cute.make_rmem_tensor(cute.make_layout(K), Int32) for _ in range(PASSES)]
    reader_rows = [cute.make_rmem_tensor(cute.make_layout(K), Int32) for _ in range(PASSES)]
    num_experts = Int32(cute.size(cu) - 1)
    n2 = Int32(cute.size(counts2, mode=[1]))
    ncols = n2 // num_experts
    if cta == Int32(0):  # totals: num_rows = all owned rows, gate_ptr[num_rows] = all references (= GEMM rows)
        owned = Int32(0)
        i = tid
        while i < n2:
            owned = owned + counts2[0, i]
            i = i + Int32(GROUP_THREADS)
        owned = group_sum(owned, scratch, warp, lane)
        if tid == Int32(0):
            num_rows[0] = owned
            gate_ptr[owned] = cu[num_experts]
    sched = MClusterWalk.create(sched_params)
    q = cta
    tile = sched.m_cluster_at(q)
    while tile.is_valid_tile:
        expert, cid = tile.tile_idx[3], tile.tile_idx[1]
        # walk-earlier blocks: every block of the earlier experts (unused cids hold 0) + this expert's blocks before
        # cid in its rotation from the decode's first block
        nblocks, first = sched._cur_num_clusters_m, sched._cur_first_cid_m
        pos = (cid - first + nblocks) % nblocks
        row_offset, ref_offset = Int32(0), Int32(0)
        i = tid
        while i < expert * ncols:
            row_offset = row_offset + counts2[0, i]
            ref_offset = ref_offset + counts2[1, i]
            i = i + Int32(GROUP_THREADS)
        c = tid
        while c < nblocks:
            if (c - first + nblocks) % nblocks < pos:
                row_offset = row_offset + counts2[0, expert * ncols + c]
                ref_offset = ref_offset + counts2[1, expert * ncols + c]
            c = c + Int32(GROUP_THREADS)
        row_offset = group_sum(row_offset, scratch, warp, lane)
        ref_offset = group_sum(ref_offset, scratch, warp, lane)
        b = block_rows_of(cu, expert, block_rows, True)
        tests = []  # every pass's owner_test first (static passes, load chains overlapping), then the scans
        for p in cutlass.range_constexpr(PASSES):
            tests.append(owner_test_flat(
                expert, cid, Int32(p * GROUP_THREADS) + tid, slot_of_row, s_reverse, topk_g, cu, flags, rank,
                num_experts, reader_flags[p], reader_rows[p], block_rows, K,
            ))
        for p in cutlass.range_constexpr(PASSES):
            owner, readers, token, owned, refs = tests[p]
            entry = row_offset + group_exclusive_scan(owned, scratch, warp, lane)
            ref = ref_offset + group_exclusive_scan(refs, scratch, warp, lane)
            if owner:
                pool_row[entry] = a_idx[cu[expert] + cid * b + Int32(p * GROUP_THREADS) + tid]
                src_address[entry] = peer_base[token // tokens_per_rank] + Int64(token % tokens_per_rank) * Int64(
                    bytes_per_token
                )
                gate_ptr[entry] = ref
                for k in cutlass.range_constexpr(K):
                    if ((readers >> k) & Int32(1)) == Int32(1):
                        gates[ref] = reader_flags[p][k]
                        dst_row[ref] = reader_rows[p][k]
                        ref = ref + Int32(1)
            row_offset = row_offset + group_sum(owned, scratch, warp, lane)
            ref_offset = ref_offset + group_sum(refs, scratch, warp, lane)
        q = q + num_ctas
        tile = sched.m_cluster_at(q)


class Prologue:
    """The launch: num_ctas CTAs of GROUP_THREADS threads, every SM (the grid syncs need them all resident)."""

    def __init__(self, K, E_local, W, block_rows, num_ctas):
        self.K, self.E_local, self.W = K, E_local, W
        self.block_rows, self.num_ctas = block_rows, num_ctas
        self.smem_bytes = 4 * (NUM_WARPS + E_local * NUM_WARPS + 2 * (E_local + 1)) + 256

    @cute.jit
    def __call__(
        self, topk, topk_g, topk_g_base, signal, signal_base, barrier_counter, grid_counter, counts, cu, s_reverse,
        s_scatter, a_idx, slot_addr, slot_base, first_src_row, flags, counts2, peer_base, pool_row, src_address,
        gate_ptr, gates, dst_row, num_rows, sem1, sem2, rank: Int32, tokens_per_rank: Int32, bytes_per_token: Int32,
        stream,
    ):
        num_experts = cute.size(cu) - 1
        sched1 = MClusterWalk.to_underlying_arguments(  # GEMM1's dispatch order (DispatchWalk)
            VarlenMTileSchedulerArguments(
                problem_shape_ntile_mnl=(1, None, num_experts), total_m=cute.size(a_idx), cu_seqlens_m=cu,
                raster_order=RasterOrderOption.AlongN, group_size=1, tile_shape_mn=(1, self.block_rows),
                cluster_shape_mnk=(1, 1, 1), persistence_mode=PersistenceMode.NONE,
                dispatch=DispatchSchedulerArguments(flags, first_src_row), a_idx=a_idx, varlen_dim=1,
            )
        )
        self.kernel(
            sched1, topk, topk_g, topk_g_base, signal, signal_base, barrier_counter, grid_counter, counts, cu,
            s_reverse, s_scatter, a_idx, slot_addr, slot_base, first_src_row, flags, counts2, peer_base, pool_row,
            src_address, gate_ptr, gates, dst_row, num_rows, sem1, sem2, rank, tokens_per_rank, bytes_per_token,
        ).launch(grid=[self.num_ctas, 1, 1], block=[GROUP_THREADS, 1, 1], smem=self.smem_bytes, stream=stream)

    @cute.kernel
    def kernel(
        self, sched1, topk, topk_g, topk_g_base, signal, signal_base, barrier_counter, grid_counter, counts, cu,
        s_reverse, s_scatter, a_idx, slot_addr, slot_base, first_src_row, flags, counts2, peer_base, pool_row,
        src_address, gate_ptr, gates, dst_row, num_rows, sem1, sem2, rank: Int32, tokens_per_rank: Int32,
        bytes_per_token: Int32,
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
            topk_g, counts, cu, s_reverse, s_scatter, a_idx, slot_addr, slot_base, first_src_row, rank,
            tokens_per_rank, Int32(W), bytes_per_token, cta, num_ctas, tid, lane, warp, scratch, masks, cu_s,
            before_s, counts2, Int32(cute.size(flags, mode=[1])), self.block_rows, K, E_local,
        )
        software_grid_sync(grid_counter, cta, num_ctas, tid, num_threads=GROUP_THREADS)
        walk1_scatter_phase(
            sched1, a_idx, s_scatter, s_reverse, topk_g, cu, flags, counts2, peer_base, pool_row, src_address,
            gate_ptr, gates, dst_row, num_rows, rank, tokens_per_rank, bytes_per_token, cta, num_ctas, tid, lane,
            warp, scratch, self.block_rows, K,
        )


def make_prologue(topk_idx, x_peer_bufs, rank, W, T_local, K, E_local, max_rows, block_rows, slot_base, group):
    """Host side: (run, meta, walk1). run() = the one launch (the x publish goes before it on the stream; the
    kernel's NVLink barrier orders it for the peers' pulls); run.sem1 / run.sem2 = the GEMMs' dynamic-persistence
    semaphores, zeroed by the kernel each forward. meta = routing_metadata's tables (cu, s_reverse, s_scatter, a_idx,
    first_src_row) plus slot_addr (per row, its combine slot's address: GEMM2's epilogue store); walk1 =
    make_dispatch_walk's walk (pool_row, src_address, gate_ptr, gates, dst_row, num_rows, flags, first_src_row,
    block_rows). block_rows = the swap GEMMs' tile N (the even-split unit), slot_base = make_combine_buffers'."""
    dev = topk_idx.device
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
    walk1 = SimpleNamespace(
        pool_row=i32(W * T_local), src_address=i64(W * T_local), gate_ptr=i32(W * T_local + 1), gates=i32(max_rows),
        dst_row=i32(max_rows), num_rows=i32(1), flags=flags, counters=flags.view(-1), first_src_row=meta.first_src_row,
        block_rows=block_rows,
    )
    peer_base = torch.tensor([b.data_ptr() for b in x_peer_bufs], dtype=torch.int64, device=dev)
    row_bytes = x_peer_bufs[rank].shape[1] * x_peer_bufs[rank].element_size()
    sem1, sem2 = torch.zeros(1, dtype=torch.int32, device=dev), torch.zeros(1, dtype=torch.int32, device=dev)
    kernel = Prologue(K, E_local, W, block_rows, num_ctas)
    v = lambda: make_fake_tensor(Int32, (cute.sym_int(),), leading_dim=0, divisibility=1)
    v64 = lambda: make_fake_tensor(Int64, (cute.sym_int(),), leading_dim=0, divisibility=1)
    m = lambda: make_fake_tensor(Int32, (cute.sym_int(), cute.sym_int()), leading_dim=1, divisibility=1)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    fn = cute.compile(
        kernel, v(), v(), v64(), v(), v64(), v(), v(), m(), v(), v(), v(), v(), v64(), v64(), v(), m(), m(), v64(),
        v(), v64(), v(), v(), v(), v(), v(), v(), Int32(rank), Int32(T_local), Int32(row_bytes), stream,
        options="--enable-tvm-ffi",
    )
    run = lambda: fn(
        topk_idx.view(-1), topk_g, sync.topk_g_base, signal, sync.signal_base, sync.barrier_counter, sync.grid_counter,
        sync.counts, meta.cu, meta.s_reverse, meta.s_scatter, meta.a_idx, meta.slot_addr, slot_base, meta.first_src_row,
        flags, sync.counts2, peer_base, walk1.pool_row, walk1.src_address, walk1.gate_ptr, walk1.gates, walk1.dst_row,
        walk1.num_rows, sem1, sem2, rank, T_local, row_bytes,
    )
    run.sync = sync
    run.sem1, run.sem2 = sem1, sem2
    return run, meta, walk1
