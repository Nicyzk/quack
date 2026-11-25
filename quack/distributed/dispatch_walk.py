# Copyright (c) 2026, QuACK team.
"""The dispatch walk, built on the device from the top-k in one launch (DispatchWalk): this rank's top-k into every
rank's gathered buffer, the per-call resets (the flags, the GEMM's tile counter), an NVLink barrier that also
orders the token rows published before it, the routing (cu, s_scatter, s_reverse: rows packed per local expert in
(source rank, token, k) order) and the walk, as phases over co-resident CTAs separated by grid syncs. The walk is
the rows the pull moves, in the GEMM's block order. A block is the gated unit of the ragged dim: the swap-AB
GEMM's even-split N block of one expert (quack varlen_n_block_rows; block_rows = its tile N) or the grouped-M
GEMM's fixed tile_M block (even_split False). Blocks come in the GEMM's plain expert-grouped order
(VarlenMTileScheduler over a one-tile fixed dim). A row is owned by its token's local reader with the smallest row
(the token's first use in walk order), found from s_scatter (row -> global slot) and s_reverse (slot -> row) over
the token's K slots. Count: owned rows and reader references per block (counts2[0 | 1, e * ncols + cid]);
scatter: each block's entries at the earlier blocks' counts, as WalkArguments."""

from typing import NamedTuple, Optional

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cutlass_dsl import Int32, Int64
from quack.compile_utils import make_fake_tensor
from quack.cute_dsl_utils import mlir_namedtuple
from quack.distributed.ptx import red_add_release_sys_s32_at, st_weak_global_4xu32_at
from quack.tile_scheduler import (
    PersistenceMode,
    RasterOrderOption,
    VarlenMTileScheduler,
    VarlenMTileSchedulerArguments,
)
from quack.utils import warp_prefix_sum
from quack.varlen_utils import VARLEN_N_GRANULE, varlen_n_block_rows

GROUP_THREADS = 128  # threads per walk CTA = tokens per routing tile
NUM_WARPS = GROUP_THREADS // 32


@mlir_namedtuple
class WalkArguments(NamedTuple):
    """The walk as one kernel argument: what pull_rows reads (the GEMM's transfer warps, or a standalone pull)."""

    src_address: (
        cute.Tensor
    )  # (n_rows,) int64 absolute address of each entry's row in its source rank's x buffer
    gate_ptr: cute.Tensor  # (n_rows + 1,) int32 CSR into gates / dst_row
    gates: (
        cute.Tensor
    )  # (n_refs,) int32 flat flags[e, cid] offsets, bumped once the entry's row has landed
    num_rows: cute.Tensor  # (1,) int32 walk length (the arrays are sized to their bounds)
    dst_row: (
        cute.Tensor
    )  # (n_refs,) int32 each reference's expert-packed row of B (the store target)
    src_sf_address: Optional[cute.Tensor] = (
        None  # (n_rows,) int64 the entry's scale row (block-scaled rows)
    )
    dst_sf_row: Optional[cute.Tensor] = (
        None  # (n_refs,) int32 each reference's SF row (block slot * atom rows + row)
    )


def fake_walk_arguments(sf=False):
    """WalkArguments of fake tensors for cute.compile (the arrays' lengths vary per routing)."""
    i32 = lambda: make_fake_tensor(Int32, (cute.sym_int(),), leading_dim=0, divisibility=1)
    i64 = lambda: make_fake_tensor(Int64, (cute.sym_int(),), leading_dim=0, divisibility=1)
    return WalkArguments(
        i64(), i32(), i32(), i32(), i32(), i64() if sf else None, i32() if sf else None
    )


class MClusterWalk(VarlenMTileScheduler):
    """Work index -> (expert, block), decoded by the calling CTA (persistence NONE: no scheduler warp)."""

    @cute.jit
    def m_cluster_at(self, work_idx: Int32):
        return self._delinearize_work_idx(work_idx)


# --- CTA and grid primitives -----------------------------------------------------------------------------------------


@cute.jit
def block_sum(
    val: Int32, scratch: cute.Tensor, warp: Int32, lane: Int32, num_warps: cutlass.Constexpr[int]
) -> Int32:
    """Sum of val over the CTA, returned to every thread; scratch = (num_warps,) int32 smem."""
    for i in cutlass.range_constexpr(5):
        val = val + cute.arch.shuffle_sync_bfly(val, offset=1 << i)
    if lane == Int32(0):
        scratch[warp] = val
    cute.arch.sync_threads()
    total = Int32(0)
    for w in cutlass.range_constexpr(num_warps):
        total = total + scratch[w]
    cute.arch.sync_threads()
    return total


@cute.jit
def block_exclusive_scan(
    val: Int32, scratch: cute.Tensor, warp: Int32, lane: Int32, num_warps: cutlass.Constexpr[int]
) -> Int32:
    """Exclusive prefix sum of val over the CTA in thread order."""
    inclusive = warp_prefix_sum(val, lane)
    if lane == Int32(31):
        scratch[warp] = inclusive
    cute.arch.sync_threads()
    before = Int32(0)
    for w in cutlass.range_constexpr(num_warps):
        if Int32(w) < warp:
            before = before + scratch[w]
    cute.arch.sync_threads()
    return before + inclusive - val


@cute.jit
def grid_sync(counter: cute.Tensor, tid: Int32, num_ctas: Int32):
    """Every CTA past this point only once every CTA has arrived (the grid is co-resident). The counter is
    monotonic, one arrive per CTA per sync, so the target is its next multiple of num_ctas: no reset."""
    cute.arch.sync_threads()
    if tid == Int32(0):
        cute.arch.fence_acq_rel_gpu()
        old = cute.arch.atomic_add(counter.iterator, Int32(1), sem="acq_rel", scope="gpu")
        target = (old // num_ctas + Int32(1)) * num_ctas
        while cute.arch.load(counter.iterator, Int32, sem="acquire", scope="gpu") < target:
            pass
    cute.arch.sync_threads()


@cute.jit
def nvlink_barrier(
    signal: cute.Tensor,
    signal_base: cute.Tensor,
    grid_counter: cute.Tensor,
    cta: Int32,
    lane: Int32,
    num_ctas: Int32,
    world_size: cutlass.Constexpr[int],
    syncs_per_call: cutlass.Constexpr[int],
):
    """Warp 0 of every CTA, after the call's first grid sync: CTA 0 arrives on every rank's word (release, system
    scope), every CTA waits on its own rank's word (acquire). The word is monotonic, world_size arrivals per call,
    so the target is world_size x (this call's index + 1); the call's index is the grid counter over the syncs per
    call."""
    call = cute.arch.load(grid_counter.iterator, Int32, sem="relaxed", scope="gpu") // (
        Int32(syncs_per_call) * num_ctas
    )
    target = Int32(world_size) * (call + Int32(1))
    if (cta == Int32(0)) & (lane < Int32(world_size)):
        red_add_release_sys_s32_at(signal_base[lane], Int32(1))
    cute.arch.sync_warp()
    if lane == Int32(0):
        while cute.arch.load(signal.iterator, Int32, sem="acquire", scope="sys") < target:
            pass
    cute.arch.sync_warp()


# --- the exchange and the resets -------------------------------------------------------------------------------------


@cute.jit
def publish_topk(
    topk: cute.Tensor,
    topk_g_base: cute.Tensor,
    rank: Int32,
    cta: Int32,
    num_ctas: Int32,
    tid: Int32,
    world_size: cutlass.Constexpr[int],
):
    """This rank's (T * K,) top-k into every rank's gathered buffer at block `rank` (its own included), 4 per store."""
    n = Int32(cute.size(topk))
    i = (cta * Int32(GROUP_THREADS) + tid) * Int32(4)
    while i < n:
        for w in cutlass.range_constexpr(world_size):
            st_weak_global_4xu32_at(
                topk_g_base[w] + Int64(rank * n + i) * Int64(4),
                topk[i],
                topk[i + 1],
                topk[i + 2],
                topk[i + 3],
            )
        i = i + num_ctas * Int32(GROUP_THREADS * 4)


@cute.jit
def zero_flags(flags: cute.Tensor, sem: cute.Tensor, cta: Int32, num_ctas: Int32, tid: Int32):
    """The per-call resets: the dispatch flags and the GEMM's tile counter."""
    ncols = Int32(cute.size(flags, mode=[1]))
    n = Int32(cute.size(flags))
    i = cta * Int32(GROUP_THREADS) + tid
    while i < n:
        flags[(i // ncols, i % ncols)] = Int32(0)
        i = i + num_ctas * Int32(GROUP_THREADS)
    if (cta == Int32(0)) & (tid == Int32(0)):
        sem[0] = Int32(0)


# --- the routing: tiles of GROUP_THREADS tokens, one token (K slots) per thread ---------------------------------------


@cute.jit
def tile_slot_masks(
    topk_g: cute.Tensor,
    tile: Int32,
    tid: Int32,
    lane: Int32,
    warp: Int32,
    rank: Int32,
    n_tokens: Int32,
    masks: cute.Tensor,
    local: cute.Tensor,
    K: cutlass.Constexpr[int],
    E_local: cutlass.Constexpr[int],
) -> Int32:
    """Thread tid's token (tile * GROUP_THREADS + tid): local[k] = its slot k's local expert, returned = the bitmask
    of the slots destined to this rank; masks[(e, warp)] (smem, E_local * NUM_WARPS words) = the tile's tokens of
    expert e, one bit per lane, so a slot's rank among the tile's slots of its expert (f order) is a popcount."""
    token = tile * Int32(GROUP_THREADS) + tid
    valid = token < n_tokens
    tc = token
    if not valid:
        tc = Int32(0)  # clamped, unconditional loads
    i = tid
    while i < Int32(E_local * NUM_WARPS):
        masks[i] = Int32(0)
        i = i + Int32(GROUP_THREADS)
    cute.arch.sync_threads()
    mask = Int32(0)
    for k in cutlass.range_constexpr(K):
        e = topk_g[tc * Int32(K) + Int32(k)] - rank * Int32(E_local)
        local[k] = e
        if valid & (e >= Int32(0)) & (e < Int32(E_local)):
            mask = mask | Int32(1 << k)
            cute.arch.atomic_or(
                masks.iterator + e * Int32(NUM_WARPS) + warp,
                Int32(1) << lane,
                sem="relaxed",
                scope="cta",
            )
    cute.arch.sync_threads()
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
def count_phase(
    topk_g: cute.Tensor,
    counts: cute.Tensor,
    counts2: cute.Tensor,
    rank: Int32,
    n_tokens: Int32,
    cta: Int32,
    num_ctas: Int32,
    tid: Int32,
    lane: Int32,
    warp: Int32,
    masks: cute.Tensor,
    K: cutlass.Constexpr[int],
    E_local: cutlass.Constexpr[int],
):
    """counts[e, tile] = the tile's slots of local expert e destined to this rank; counts2 (the walk's per-block
    owner / reference counts the emit phase accumulates) zeroed."""
    local = cute.make_rmem_tensor(cute.make_layout(K), Int32)
    n_tiles = Int32(cute.size(counts, mode=[1]))
    n2 = Int32(cute.size(counts2, mode=[1]))
    i = cta * Int32(GROUP_THREADS) + tid
    while i < n2:
        counts2[0, i] = Int32(0)
        counts2[1, i] = Int32(0)
        i = i + num_ctas * Int32(GROUP_THREADS)
    tile = cta
    while tile < n_tiles:
        tile_slot_masks(topk_g, tile, tid, lane, warp, rank, n_tokens, masks, local, K, E_local)
        e = tid
        while e < Int32(E_local):
            total = Int32(0)
            for w in cutlass.range_constexpr(NUM_WARPS):
                total = total + cute.arch.popc(masks[e * Int32(NUM_WARPS) + Int32(w)])
            counts[e, tile] = total
            e = e + Int32(GROUP_THREADS)
        cute.arch.sync_threads()
        tile = tile + num_ctas


@cute.jit
def expert_tile_sums(
    counts: cute.Tensor,
    upto: Int32,
    lane: Int32,
    warp: Int32,
    out_s: cute.Tensor,
    E_local: cutlass.Constexpr[int],
):
    """out_s[e] = sum of counts[e, j] over tiles j < upto: warp w takes experts w, w + NUM_WARPS, ..., lanes over
    tiles; per 32-tile pass every expert's load is issued before any is consumed (one round trip)."""
    EPW = (E_local + NUM_WARPS - 1) // NUM_WARPS
    acc = cute.make_rmem_tensor(cute.make_layout(EPW), Int32)
    for i in cutlass.range_constexpr(EPW):
        acc[i] = Int32(0)
    for p in cutlass.range((upto + Int32(31)) // Int32(32), unroll=1):
        j = p * Int32(32) + lane
        v = cute.make_rmem_tensor(cute.make_layout(EPW), Int32)
        for i in cutlass.range_constexpr(EPW):
            e = Int32(i * NUM_WARPS) + warp
            jc, ec = j, e
            if (j >= upto) | (e >= Int32(E_local)):
                jc = Int32(0)
            if e >= Int32(E_local):
                ec = Int32(0)
            v[i] = counts[ec, jc]  # clamped, unconditional: the loads overlap
        for i in cutlass.range_constexpr(EPW):
            e = Int32(i * NUM_WARPS) + warp
            if (j < upto) & (e < Int32(E_local)):
                acc[i] = acc[i] + v[i]
    for i in cutlass.range_constexpr(EPW):
        e = Int32(i * NUM_WARPS) + warp
        total = acc[i]
        for s in cutlass.range_constexpr(5):
            total = total + cute.arch.shuffle_sync_bfly(total, offset=1 << s)
        if (lane == Int32(0)) & (e < Int32(E_local)):
            out_s[e] = total


@cute.jit
def emit_phase(
    topk_g: cute.Tensor,
    counts: cute.Tensor,
    cu: cute.Tensor,
    s_reverse: cute.Tensor,
    s_scatter: cute.Tensor,
    counts2: cute.Tensor,
    cu_tiles: Optional[cute.Tensor],
    rank: Int32,
    n_tokens: Int32,
    cta: Int32,
    num_ctas: Int32,
    tid: Int32,
    lane: Int32,
    warp: Int32,
    masks: cute.Tensor,
    cu_s: cute.Tensor,
    before_s: cute.Tensor,
    block_rows: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    E_local: cutlass.Constexpr[int],
    even_split: cutlass.Constexpr[bool],
):
    """The routing: cu (CTA 0; cu_tiles, the GEMM's per-expert block prefix, with it) and per tile s_reverse /
    s_scatter; plus the walk's per-block counts from the token side: a present token's owner row is its lowest
    local expert's row, so counts2[0, e * ncols + cid] += 1 and counts2[1, ...] += its local slots for the owner
    row's block. cu_s / before_s = (E_local + 1,) smem: the expert offsets (cu_s[E_local] = total rows) and the
    tile's per-expert prefixes."""
    local = cute.make_rmem_tensor(cute.make_layout(K), Int32)
    n_tiles = Int32(cute.size(counts, mode=[1]))
    n2 = Int32(cute.size(counts2, mode=[1]))
    ncols = n2 // Int32(E_local)
    expert_tile_sums(counts, n_tiles, lane, warp, cu_s, E_local)  # per-expert totals
    cute.arch.sync_threads()
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
    cute.arch.sync_threads()
    if cta == Int32(0):
        e = tid
        while e <= Int32(E_local):
            cu[e] = cu_s[e]
            if cutlass.const_expr(cu_tiles is not None):
                prefix = Int32(0)
                for j in cutlass.range(e, unroll=1):
                    prefix = prefix + cute.ceil_div(cu_s[j + 1] - cu_s[j], block_rows)
                cu_tiles[e] = prefix
            e = e + Int32(GROUP_THREADS)
    tile = cta
    while tile < n_tiles:
        expert_tile_sums(
            counts, tile, lane, warp, before_s, E_local
        )  # prefixes over the tiles before this one
        mask = tile_slot_masks(
            topk_g, tile, tid, lane, warp, rank, n_tokens, masks, local, K, E_local
        )
        token = tile * Int32(GROUP_THREADS) + tid
        e_min, row_min = Int32(0x7FFFFFFF), Int32(0)
        for k in cutlass.range_constexpr(K):
            if ((mask >> k) & Int32(1)) == Int32(1):
                e_k = local[k]
                f = token * Int32(K) + Int32(k)
                row = cu_s[e_k] + before_s[e_k] + slot_rank(masks, e_k, lane, warp)
                s_reverse[f] = row
                s_scatter[row] = f
                if e_k < e_min:
                    e_min, row_min = e_k, row
        if mask != Int32(
            0
        ):  # the owner row's block gets this token's owned row and its reader references
            b = Int32(block_rows)
            if cutlass.const_expr(even_split):
                b = varlen_n_block_rows(
                    cu_s[e_min + Int32(1)] - cu_s[e_min], block_rows, VARLEN_N_GRANULE
                )
            slot = e_min * ncols + (row_min - cu_s[e_min]) // b
            cute.arch.atomic_add(counts2.iterator + slot, Int32(1), sem="relaxed", scope="gpu")
            cute.arch.atomic_add(
                counts2.iterator + n2 + slot, cute.arch.popc(mask), sem="relaxed", scope="gpu"
            )
        cute.arch.sync_threads()
        tile = tile + num_ctas


# --- the walk --------------------------------------------------------------------------------------------------------


@cute.jit
def block_rows_of(
    cu_seqlens: cute.Tensor,
    expert: Int32,
    block_rows: cutlass.Constexpr[int],
    even_split: cutlass.Constexpr[bool],
) -> Int32:
    """Rows of one of the expert's gated blocks: the swap-AB GEMM's even-split N block (quack varlen_n_block_rows;
    block_rows = its tile N), or the grouped-M GEMM's fixed tile_M (even_split False)."""
    b = Int32(block_rows)
    if cutlass.const_expr(even_split):
        b = varlen_n_block_rows(
            cu_seqlens[expert + 1] - cu_seqlens[expert], block_rows, VARLEN_N_GRANULE
        )
    return b


@cute.jit
def owner_test(
    expert: Int32,
    cid: Int32,
    r: Int32,
    slot_of_row: cute.Tensor,
    s_reverse: cute.Tensor,
    topk: cute.Tensor,
    cu: cute.Tensor,
    flags: cute.Tensor,
    rank: Int32,
    num_experts: Int32,
    reader_flags: cute.Tensor,
    reader_rows: cute.Tensor,
    block_rows: cutlass.Constexpr[int],
    topk_k: cutlass.Constexpr[int],
    reader_sf_rows: Optional[
        cute.Tensor
    ] = None,  # (K,) int32 out: the reference's SF row (sf_atom_rows > 0)
    sf_atom_rows: cutlass.Constexpr[int] = 0,
    cu_tiles: Optional[
        cute.Tensor
    ] = None,  # (E + 1,) int32 the GEMM's block prefix (cu_tiles_n; sf_atom_rows > 0)
    even_split: cutlass.Constexpr[
        bool
    ] = True,  # the blocks: even-split N (swap-AB) or fixed tile_M (grouped M)
):
    """Row r of block (expert, cid): (owner, readers, token, owned, refs) = whether it owns its token's pull, a bitmask
    over its token's K slots of the local readers, whose flag offsets (flat into flags[e, cid]) go to reader_flags[k]
    and whose expert-major rows to reader_rows[k], the token's global index (source rank * tokens per rank + token),
    and 1 / popcount(readers) if owner else 0 / 0. Every load unconditional on clamped indices, so two rows' chains
    issue together. reader_sf_rows[k] = the reference's SF row: its block slot (cu_tiles[e] + block) * sf_atom_rows +
    its row in the block."""
    e_start, e_end = cu[expert], cu[expert + 1]
    b = block_rows_of(cu, expert, block_rows, even_split)
    start = e_start + cid * b
    row = start + r
    in_cluster = row < cutlass.min(start + b, e_end)
    row_c = row
    if not in_cluster:
        row_c = start  # a scheduled block has a first row
    token = slot_of_row[row_c] // topk_k
    slot0 = token * topk_k
    first_reader, readers = Int32(0x7FFFFFFF), Int32(0)
    for k in cutlass.range_constexpr(topk_k):
        e = topk[slot0 + k] - rank * num_experts
        reader_row = s_reverse[slot0 + k]
        mine = (e >= Int32(0)) & (e < num_experts)
        ec = e
        if not mine:
            ec = Int32(0)
        rb = block_rows_of(cu, ec, block_rows, even_split)
        blk = (reader_row - cu[ec]) // rb
        flag = Int32(cute.crd2idx((ec, blk), flags.layout))
        reader_flags[k], reader_rows[k] = Int32(0), Int32(0)
        if cutlass.const_expr(sf_atom_rows > 0):
            reader_sf_rows[k] = Int32(0)
        if mine:
            first_reader = cutlass.min(first_reader, reader_row)
            readers = readers | Int32(1 << k)
            reader_flags[k], reader_rows[k] = flag, reader_row
            if cutlass.const_expr(sf_atom_rows > 0):
                reader_sf_rows[k] = (cu_tiles[ec] + blk) * Int32(sf_atom_rows) + (
                    reader_row - cu[ec]
                ) % rb
    owner = in_cluster & (first_reader == row)
    owned, refs = Int32(0), Int32(0)
    if owner:
        owned, refs = Int32(1), cute.arch.popc(readers)
    if not in_cluster:
        readers, token = Int32(0), Int32(0)
    return owner, readers, token, owned, refs


@cute.jit
def walk_scatter_phase(
    sched_params,
    slot_of_row: cute.Tensor,
    s_reverse: cute.Tensor,
    topk: cute.Tensor,
    cu: cute.Tensor,
    flags: cute.Tensor,
    counts2: cute.Tensor,
    peer_base: cute.Tensor,
    walk: WalkArguments,
    rank: Int32,
    tokens_per_rank: Int32,
    bytes_per_token: Int32,
    cta: Int32,
    num_ctas: Int32,
    tid: Int32,
    lane: Int32,
    warp: Int32,
    scratch: cute.Tensor,
    block_rows: cutlass.Constexpr[int],
    topk_k: cutlass.Constexpr[int],
    peer_sf_base: Optional[cute.Tensor] = None,
    sf_bytes_per_token: Int32 = Int32(0),
    sf_atom_rows: cutlass.Constexpr[int] = 0,
    cu_tiles: Optional[cute.Tensor] = None,  # the emit phase's block prefix (sf_atom_rows > 0)
    even_split: cutlass.Constexpr[bool] = True,
):
    """Each block's owned rows at (walk-earlier blocks' owned rows + rank among its owners), in row order, from
    counts2; CTA 0 also writes the totals (num_rows, gate_ptr[num_rows] = all references = the GEMM's rows).
    sf_atom_rows > 0: also each entry's scale-row address and each reference's SF row."""
    PASSES = (block_rows + GROUP_THREADS - 1) // GROUP_THREADS
    reader_flags = [cute.make_rmem_tensor(cute.make_layout(topk_k), Int32) for _ in range(PASSES)]
    reader_rows = [cute.make_rmem_tensor(cute.make_layout(topk_k), Int32) for _ in range(PASSES)]
    reader_sf_rows = [None] * PASSES
    if cutlass.const_expr(sf_atom_rows > 0):
        reader_sf_rows = [
            cute.make_rmem_tensor(cute.make_layout(topk_k), Int32) for _ in range(PASSES)
        ]
    num_experts = Int32(cute.size(cu) - 1)
    n2 = Int32(cute.size(counts2, mode=[1]))
    ncols = n2 // num_experts
    if cta == Int32(0):
        owned = Int32(0)
        i = tid
        while i < n2:
            owned = owned + counts2[0, i]
            i = i + Int32(GROUP_THREADS)
        owned = block_sum(owned, scratch, warp, lane, NUM_WARPS)
        if tid == Int32(0):
            walk.num_rows[0] = owned
            walk.gate_ptr[owned] = cu[num_experts]
    sched = MClusterWalk.create(sched_params)
    q = cta
    tile = sched.m_cluster_at(q)
    while tile.is_valid_tile:
        expert, cid = tile.tile_idx[3], tile.tile_idx[1]
        # walk-earlier blocks: every block of the earlier experts (unused slots hold 0) + this expert's blocks before
        # cid (the plain expert-grouped order)
        nblocks, pos = sched._cur_num_clusters_m, cid
        row_offset, ref_offset = Int32(0), Int32(0)
        i = tid
        while i < expert * ncols:
            row_offset = row_offset + counts2[0, i]
            ref_offset = ref_offset + counts2[1, i]
            i = i + Int32(GROUP_THREADS)
        c = tid
        while c < nblocks:
            if c < pos:
                row_offset = row_offset + counts2[0, expert * ncols + c]
                ref_offset = ref_offset + counts2[1, expert * ncols + c]
            c = c + Int32(GROUP_THREADS)
        row_offset = block_sum(row_offset, scratch, warp, lane, NUM_WARPS)
        ref_offset = block_sum(ref_offset, scratch, warp, lane, NUM_WARPS)
        tests = []  # every pass's owner_test first (static passes, load chains overlapping), then the scans
        for p in cutlass.range_constexpr(PASSES):
            tests.append(
                owner_test(
                    expert,
                    cid,
                    Int32(p * GROUP_THREADS) + tid,
                    slot_of_row,
                    s_reverse,
                    topk,
                    cu,
                    flags,
                    rank,
                    num_experts,
                    reader_flags[p],
                    reader_rows[p],
                    block_rows,
                    topk_k,
                    reader_sf_rows[p],
                    sf_atom_rows,
                    cu_tiles,
                    even_split,
                )
            )
        for p in cutlass.range_constexpr(PASSES):
            owner, readers, token, owned, refs = tests[p]
            entry = row_offset + block_exclusive_scan(owned, scratch, warp, lane, NUM_WARPS)
            ref = ref_offset + block_exclusive_scan(refs, scratch, warp, lane, NUM_WARPS)
            if owner:
                walk.src_address[entry] = peer_base[token // tokens_per_rank] + Int64(
                    token % tokens_per_rank
                ) * Int64(bytes_per_token)
                if cutlass.const_expr(sf_atom_rows > 0):
                    walk.src_sf_address[entry] = peer_sf_base[token // tokens_per_rank] + Int64(
                        token % tokens_per_rank
                    ) * Int64(sf_bytes_per_token)
                walk.gate_ptr[entry] = ref
                for k in cutlass.range_constexpr(topk_k):
                    if ((readers >> k) & Int32(1)) == Int32(1):
                        walk.gates[ref] = reader_flags[p][k]
                        walk.dst_row[ref] = reader_rows[p][k]
                        if cutlass.const_expr(sf_atom_rows > 0):
                            walk.dst_sf_row[ref] = reader_sf_rows[p][k]
                        ref = ref + Int32(1)
            row_offset = row_offset + block_sum(owned, scratch, warp, lane, NUM_WARPS)
            ref_offset = ref_offset + block_sum(refs, scratch, warp, lane, NUM_WARPS)
        q = q + num_ctas
        tile = sched.m_cluster_at(q)


def walk_scheduler_args(slot_of_row, cu_seqlens, block_rows):
    """The GEMM's block order over a one-tile fixed dim, decoded by the walk CTAs (MClusterWalk): the plain
    expert-grouped order over the even-split N blocks (varlen_dim 1); slot_of_row's length bounds the rows."""
    return MClusterWalk.to_underlying_arguments(
        VarlenMTileSchedulerArguments(
            problem_shape_ntile_mnl=(1, None, cute.size(cu_seqlens) - 1),
            total_m=cute.size(slot_of_row),
            cu_seqlens_m=cu_seqlens,
            raster_order=RasterOrderOption.AlongN,
            group_size=1,
            tile_shape_mn=(1, block_rows),
            cluster_shape_mnk=(1, 1, 1),
            persistence_mode=PersistenceMode.NONE,
            varlen_dim=1,
        )
    )


def sf_atom_rows_of(block_rows):
    """SF rows one block slot owns: the GEMM's SF window in whole 128-row atoms (GemmSm100.sfb_window_atoms)."""
    return -(-block_rows // 128) * 128


class DispatchWalk:
    """The one launch: num_ctas CTAs of GROUP_THREADS threads, all co-resident (the grid syncs); the phases are
    grid-stride sweeps, so the count is a throughput choice (the SM count: the kernel runs alone on the stream)."""

    num_grid_syncs = (
        3  # per call, in kernel(): the barrier reads the call's index off the grid counter
    )

    def __init__(
        self, topk_k, num_experts, world_size, block_rows, num_ctas, sf_atom_rows=0, even_split=True
    ):
        self.topk_k, self.num_experts, self.world_size = topk_k, num_experts, world_size
        self.block_rows, self.num_ctas = block_rows, num_ctas
        self.sf_atom_rows = sf_atom_rows  # block-scaled: SF rows per block slot (0: no scales)
        self.even_split = (
            even_split  # even-split N blocks (swap-AB) or fixed tile_M blocks (grouped M)
        )
        self.smem_bytes = 4 * (NUM_WARPS + num_experts * NUM_WARPS + 2 * (num_experts + 1)) + 256

    @cute.jit
    def __call__(
        self,
        topk,
        topk_g,
        topk_g_base,
        signal,
        signal_base,
        grid_counter,
        counts,
        counts2,
        cu,
        s_reverse,
        s_scatter,
        flags,
        sem,
        peer_base,
        walk: WalkArguments,
        rank: Int32,
        tokens_per_rank: Int32,
        bytes_per_token: Int32,
        stream,
        peer_sf_base=None,
        sf_bytes_per_token: Int32 = Int32(0),
        cu_tiles=None,
    ):
        sched = walk_scheduler_args(s_scatter, cu, self.block_rows)
        self.kernel(
            sched,
            topk,
            topk_g,
            topk_g_base,
            signal,
            signal_base,
            grid_counter,
            counts,
            counts2,
            cu,
            s_reverse,
            s_scatter,
            flags,
            sem,
            peer_base,
            walk,
            rank,
            tokens_per_rank,
            bytes_per_token,
            peer_sf_base,
            sf_bytes_per_token,
            cu_tiles,
        ).launch(
            grid=[self.num_ctas, 1, 1],
            block=[GROUP_THREADS, 1, 1],
            smem=self.smem_bytes,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        sched_params,
        topk,
        topk_g,
        topk_g_base,
        signal,
        signal_base,
        grid_counter,
        counts,
        counts2,
        cu,
        s_reverse,
        s_scatter,
        flags,
        sem,
        peer_base,
        walk: WalkArguments,
        rank: Int32,
        tokens_per_rank: Int32,
        bytes_per_token: Int32,
        peer_sf_base,
        sf_bytes_per_token: Int32,
        cu_tiles,
    ):
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        warp, lane = tid // 32, tid % 32
        num_ctas, _, _ = cute.arch.grid_dim()  # the count the launch syncs on
        K, E, world_size = self.topk_k, self.num_experts, self.world_size
        smem = utils.SmemAllocator()
        scratch = smem.allocate_tensor(Int32, cute.make_layout(NUM_WARPS), byte_alignment=16)
        masks = smem.allocate_tensor(Int32, cute.make_layout(E * NUM_WARPS), byte_alignment=16)
        cu_s = smem.allocate_tensor(Int32, cute.make_layout(E + 1), byte_alignment=16)
        before_s = smem.allocate_tensor(Int32, cute.make_layout(E + 1), byte_alignment=16)
        n_tokens = tokens_per_rank * Int32(world_size)
        publish_topk(topk, topk_g_base, rank, cta, num_ctas, tid, world_size)
        zero_flags(flags, sem, cta, num_ctas, tid)
        grid_sync(grid_counter, tid, num_ctas)  # every CTA's stores issued
        if warp == Int32(0):
            cute.arch.fence_acq_rel_sys()
            nvlink_barrier(
                signal,
                signal_base,
                grid_counter,
                cta,
                lane,
                num_ctas,
                world_size,
                self.num_grid_syncs,
            )
        cute.arch.sync_threads()
        count_phase(
            topk_g, counts, counts2, rank, n_tokens, cta, num_ctas, tid, lane, warp, masks, K, E
        )
        grid_sync(grid_counter, tid, num_ctas)
        emit_phase(
            topk_g,
            counts,
            cu,
            s_reverse,
            s_scatter,
            counts2,
            cu_tiles,
            rank,
            n_tokens,
            cta,
            num_ctas,
            tid,
            lane,
            warp,
            masks,
            cu_s,
            before_s,
            self.block_rows,
            K,
            E,
            self.even_split,
        )
        grid_sync(grid_counter, tid, num_ctas)
        walk_scatter_phase(
            sched_params,
            s_scatter,
            s_reverse,
            topk_g,
            cu,
            flags,
            counts2,
            peer_base,
            walk,
            rank,
            tokens_per_rank,
            bytes_per_token,
            cta,
            num_ctas,
            tid,
            lane,
            warp,
            scratch,
            self.block_rows,
            K,
            peer_sf_base,
            sf_bytes_per_token,
            self.sf_atom_rows,
            cu_tiles,
            self.even_split,
        )
