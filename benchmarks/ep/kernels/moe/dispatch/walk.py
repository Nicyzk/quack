# The dispatch walk: the rows the pull moves, in the GEMM's block order, built on the device. A block is the gated
# unit of the ragged dim: block_rows consecutive rows of one expert (the GEMM's M cluster, tile M x cluster M) or,
# even_split, the swap-AB GEMM's N block (quack varlen_n_block_rows; block_rows = its tile N). Blocks come in quack's
# varlen scheduler order under the GEMM's DispatchSchedulerArguments over a one-tile fixed dim (expert-major, each
# expert rotated to start at first_src_row). A row is owned by its token's local reader with the smallest row (the
# token's first use in walk order), found from sonic's s_scatter_idx (row -> global slot) and s_reverse_local (slot ->
# row) over the token's K slots. Count: owned rows and reader references per block (counts2[0 | 1, e * ncols + cid]);
# scatter: each block's entries at the earlier blocks' counts, as WalkArguments. Both run on GROUP_THREADS-thread CTAs:
# as phases of prologue.py's one launch (its emit phase counts from the token side, walk_scatter_phase scatters) or
# as DispatchWalk's two launches after routing_metadata. Why the walk is not built inside the pull:
# results/2026-09-21/why_separate_dispatch_ownership_kernel.md.
from types import SimpleNamespace
from typing import NamedTuple, Optional

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass.cutlass_dsl import Int32, Int64
from quack.compile_utils import make_fake_tensor
from quack.cute_dsl_utils import mlir_namedtuple
from quack.tile_scheduler import (
    PersistenceMode,
    RasterOrderOption,
    VarlenMTileScheduler,
    VarlenMTileSchedulerArguments,
)
from quack.utils import warp_prefix_sum

try:  # quack ep: the swap-AB even-split block (varlen_n); absent on main / `all`, which gate M clusters only
    from quack.varlen_utils import VARLEN_N_GRANULE, varlen_n_block_rows
except ImportError:
    VARLEN_N_GRANULE = varlen_n_block_rows = None

GROUP_THREADS = 128  # threads per walk CTA (prologue.py's token tile too)
NUM_WARPS = GROUP_THREADS // 32
BLOCK_ROWS = 256  # the M-cluster GEMMs' block (tile M 256, cluster (2, 1))


@mlir_namedtuple
class WalkArguments(NamedTuple):
    """The walk as one kernel argument: what pull_rows reads (the fused GEMM's transfer warps and DispatchPull)."""

    src_address: cute.Tensor  # (n_rows,) int64 absolute address of each entry's row in its source rank's x buffer
    pool_row: cute.Tensor  # (n_rows,) int32 destination row of each entry, in pull order
    gate_ptr: cute.Tensor  # (n_rows + 1,) int32 CSR into gates
    gates: cute.Tensor  # (n_refs,) int32 flat flags[e, cid] offsets, bumped once the entry's row has landed
    num_rows: cute.Tensor  # (1,) int32 walk length (the arrays are sized to their bounds)
    dst_row: cute.Tensor  # (n_refs,) int32 each reference's expert-major row (the scatter layout's store target)
    src_sf_address: Optional[cute.Tensor] = None  # (n_rows,) int64 the entry's scale row (block-scaled rows)
    dst_sf_row: Optional[cute.Tensor] = None  # (n_refs,) int32 each reference's SF row (block slot * atom rows + row)


def fake_walk_arguments(sf=False):
    """WalkArguments of fake tensors for cute.compile (the arrays' lengths vary per routing)."""
    i32 = lambda: make_fake_tensor(Int32, (cute.sym_int(),), leading_dim=0, divisibility=1)
    i64 = lambda: make_fake_tensor(Int64, (cute.sym_int(),), leading_dim=0, divisibility=1)
    return WalkArguments(i64(), i32(), i32(), i32(), i32(), i32(), i64() if sf else None, i32() if sf else None)


class MClusterWalk(VarlenMTileScheduler):
    """Work index -> (expert, block), decoded by the calling CTA (persistence NONE: no scheduler warp)."""

    @cute.jit
    def m_cluster_at(self, work_idx: Int32):
        return self._delinearize_work_idx(work_idx)


@cute.jit
def block_sum(val: Int32, scratch: cute.Tensor, warp: Int32, lane: Int32, num_warps: cutlass.Constexpr[int]) -> Int32:
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
def block_rows_of(
    cu_seqlens: cute.Tensor, expert: Int32, block_rows: cutlass.Constexpr[int], even_split: cutlass.Constexpr[bool]
) -> Int32:
    """Rows of one of the expert's gated blocks: block_rows (the GEMM's M cluster), or the swap-AB GEMM's
    even-split N block (quack varlen_n_block_rows; block_rows = its tile N)."""
    if cutlass.const_expr(even_split):
        return varlen_n_block_rows(cu_seqlens[expert + 1] - cu_seqlens[expert], block_rows, VARLEN_N_GRANULE)
    return Int32(block_rows)


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
    even_split: cutlass.Constexpr[bool],
    reader_sf_rows: Optional[cute.Tensor] = None,  # (K,) int32 out: the reference's SF row (sf_atom_rows > 0)
    sf_atom_rows: cutlass.Constexpr[int] = 0,
):
    """Row r of block (expert, cid): (owner, readers, token, owned, refs) = whether it owns its A row, a bitmask over
    its token's K slots of the local readers, whose flag offsets (flat into flags[e, cid]) go to reader_flags[k] and
    whose expert-major rows to reader_rows[k], the token's global index (source rank * tokens per rank + token), and
    1 / popcount(readers) if owner else 0 / 0. Every load unconditional on clamped indices, so two rows' chains issue
    together. reader_sf_rows[k] = the reference's SF row: its block slot (the flat flag) * sf_atom_rows + its row in
    the block (the GEMM's cu_tiles_n = e * flags columns)."""
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
        flag = Int32(cute.crd2idx((ec, (reader_row - cu[ec]) // rb), flags.layout))
        reader_flags[k], reader_rows[k] = Int32(0), Int32(0)
        if cutlass.const_expr(sf_atom_rows > 0):
            reader_sf_rows[k] = Int32(0)
        if mine:
            first_reader = cutlass.min(first_reader, reader_row)
            readers = readers | Int32(1 << k)
            reader_flags[k], reader_rows[k] = flag, reader_row
            if cutlass.const_expr(sf_atom_rows > 0):
                reader_sf_rows[k] = flag * Int32(sf_atom_rows) + (reader_row - cu[ec]) % rb
    owner = in_cluster & (first_reader == row)
    owned, refs = Int32(0), Int32(0)
    if owner:
        owned, refs = Int32(1), cute.arch.popc(readers)
    if not in_cluster:
        readers, token = Int32(0), Int32(0)
    return owner, readers, token, owned, refs


@cute.jit
def walk_count_phase(
    sched_params,
    slot_of_row: cute.Tensor,
    s_reverse: cute.Tensor,
    topk: cute.Tensor,
    cu: cute.Tensor,
    flags: cute.Tensor,
    counts2: cute.Tensor,
    rank: Int32,
    cta: Int32,
    num_ctas: Int32,
    tid: Int32,
    lane: Int32,
    warp: Int32,
    scratch: cute.Tensor,
    block_rows: cutlass.Constexpr[int],
    topk_k: cutlass.Constexpr[int],
    even_split: cutlass.Constexpr[bool],
):
    """counts2[0 | 1, e * ncols + cid] = the block's owned rows / reader references, one CTA per scheduled block;
    slots past an expert's blocks are zeroed (the scatter phase sums whole experts)."""
    vd = 1 if even_split else 0
    PASSES = (block_rows + GROUP_THREADS - 1) // GROUP_THREADS
    reader_flags = cute.make_rmem_tensor(cute.make_layout(topk_k), Int32)
    reader_rows = cute.make_rmem_tensor(cute.make_layout(topk_k), Int32)
    num_experts = Int32(cute.size(cu) - 1)
    n2 = Int32(cute.size(counts2, mode=[1]))
    ncols = n2 // num_experts
    i = cta * Int32(GROUP_THREADS) + tid
    while i < n2:
        e, c = i // ncols, i % ncols
        b = block_rows_of(cu, e, block_rows, even_split)
        if c >= (cu[e + 1] - cu[e] + b - Int32(1)) // b:
            counts2[0, i] = Int32(0)
            counts2[1, i] = Int32(0)
        i = i + num_ctas * Int32(GROUP_THREADS)
    sched = MClusterWalk.create(sched_params)
    q = cta
    tile = sched.m_cluster_at(q)
    while tile.is_valid_tile:
        expert, cid = tile.tile_idx[3], tile.tile_idx[vd]
        owned_t, refs_t = Int32(0), Int32(0)
        for p in cutlass.range_constexpr(PASSES):
            _, _, _, owned, refs = owner_test(
                expert, cid, Int32(p * GROUP_THREADS) + tid, slot_of_row, s_reverse, topk, cu, flags, rank,
                num_experts, reader_flags, reader_rows, block_rows, topk_k, even_split,
            )
            owned_t, refs_t = owned_t + owned, refs_t + refs
        owned_t = block_sum(owned_t, scratch, warp, lane, NUM_WARPS)
        refs_t = block_sum(refs_t, scratch, warp, lane, NUM_WARPS)
        if tid == Int32(0):
            counts2[0, expert * ncols + cid] = owned_t
            counts2[1, expert * ncols + cid] = refs_t
        q = q + num_ctas
        tile = sched.m_cluster_at(q)


@cute.jit
def walk_scatter_phase(
    sched_params,
    a_idx: cute.Tensor,
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
    even_split: cutlass.Constexpr[bool],
    peer_sf_base: Optional[cute.Tensor] = None,
    sf_bytes_per_token: Int32 = Int32(0),
    sf_atom_rows: cutlass.Constexpr[int] = 0,
):
    """Each block's owned rows at (walk-earlier blocks' owned rows + rank among its owners), in row order, from
    counts2; CTA 0 also writes the totals (num_rows, gate_ptr[num_rows] = all references = the GEMM's rows).
    sf_atom_rows > 0: also each entry's scale-row address and each reference's SF row."""
    vd = 1 if even_split else 0
    PASSES = (block_rows + GROUP_THREADS - 1) // GROUP_THREADS
    reader_flags = [cute.make_rmem_tensor(cute.make_layout(topk_k), Int32) for _ in range(PASSES)]
    reader_rows = [cute.make_rmem_tensor(cute.make_layout(topk_k), Int32) for _ in range(PASSES)]
    reader_sf_rows = [None] * PASSES
    if cutlass.const_expr(sf_atom_rows > 0):
        reader_sf_rows = [cute.make_rmem_tensor(cute.make_layout(topk_k), Int32) for _ in range(PASSES)]
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
        expert, cid = tile.tile_idx[3], tile.tile_idx[vd]
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
        b = block_rows_of(cu, expert, block_rows, even_split)
        tests = []  # every pass's owner_test first (static passes, load chains overlapping), then the scans
        for p in cutlass.range_constexpr(PASSES):
            tests.append(owner_test(
                expert, cid, Int32(p * GROUP_THREADS) + tid, slot_of_row, s_reverse, topk, cu, flags, rank,
                num_experts, reader_flags[p], reader_rows[p], block_rows, topk_k, even_split, reader_sf_rows[p],
                sf_atom_rows,
            ))
        for p in cutlass.range_constexpr(PASSES):
            owner, readers, token, owned, refs = tests[p]
            entry = row_offset + block_exclusive_scan(owned, scratch, warp, lane, NUM_WARPS)
            ref = ref_offset + block_exclusive_scan(refs, scratch, warp, lane, NUM_WARPS)
            if owner:
                walk.pool_row[entry] = a_idx[cu[expert] + cid * b + Int32(p * GROUP_THREADS) + tid]
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


def walk_scheduler_args(a_idx, cu_seqlens, flags, first_src_row, block_rows, even_split):
    """The GEMM's block order over a one-tile fixed dim, decoded by the walk CTAs (MClusterWalk): the plain
    expert-grouped order (the landing-order rotation by first_src_row measured as no gain; first_src_row and
    flags are unused here). varlen_dim rides only the even-split path (quack ep's scheduler)."""
    return MClusterWalk.to_underlying_arguments(
        VarlenMTileSchedulerArguments(
            problem_shape_ntile_mnl=(
                (None, 1, cute.size(cu_seqlens) - 1) if not even_split else (1, None, cute.size(cu_seqlens) - 1)
            ),
            total_m=cute.size(a_idx),
            cu_seqlens_m=cu_seqlens,
            raster_order=RasterOrderOption.AlongN,
            group_size=1,
            tile_shape_mn=(block_rows, 1) if not even_split else (1, block_rows),
            cluster_shape_mnk=(1, 1, 1),
            persistence_mode=PersistenceMode.NONE,
            **({"varlen_dim": 1} if even_split else {}),
        )
    )


class DispatchWalk:
    """The standalone builder (after routing_metadata): count, then scatter, each one launch of num_ctas CTAs."""

    def __init__(self, topk_k, block_rows, num_ctas, even_split=False, sf_atom_rows=0):
        self.topk_k, self.block_rows, self.num_ctas = topk_k, block_rows, num_ctas
        self.even_split = even_split
        self.sf_atom_rows = sf_atom_rows  # block-scaled: SF rows per block slot (0: no scales)

    @cute.jit
    def __call__(
        self, a_idx, slot_of_row, s_reverse, topk, cu_seqlens, flags, first_src_row, counts2, peer_base,
        walk: WalkArguments, rank: Int32, tokens_per_rank: Int32, bytes_per_token: Int32, stream, peer_sf_base=None,
        sf_bytes_per_token: Int32 = Int32(0),
    ):
        sched = lambda: walk_scheduler_args(a_idx, cu_seqlens, flags, first_src_row, self.block_rows, self.even_split)
        launch = dict(grid=[self.num_ctas, 1, 1], block=[GROUP_THREADS, 1, 1], smem=256, stream=stream)
        self.count_kernel(sched(), slot_of_row, s_reverse, topk, cu_seqlens, flags, counts2, rank).launch(**launch)
        self.scatter_kernel(  # one scheduler params per launch: a launch rebinds the params' tensors to its arguments
            sched(), a_idx, slot_of_row, s_reverse, topk, cu_seqlens, flags, counts2, peer_base, walk, rank,
            tokens_per_rank, bytes_per_token, peer_sf_base, sf_bytes_per_token,
        ).launch(**launch)

    @cute.kernel
    def count_kernel(self, sched_params, slot_of_row, s_reverse, topk, cu_seqlens, flags, counts2, rank: Int32):
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        scratch = utils.SmemAllocator().allocate_tensor(Int32, cute.make_layout(NUM_WARPS), byte_alignment=16)
        walk_count_phase(
            sched_params, slot_of_row, s_reverse, topk, cu_seqlens, flags, counts2, rank, Int32(cta),
            Int32(self.num_ctas), tid, tid % 32, tid // 32, scratch, self.block_rows, self.topk_k, self.even_split,
        )

    @cute.kernel
    def scatter_kernel(
        self, sched_params, a_idx, slot_of_row, s_reverse, topk, cu_seqlens, flags, counts2, peer_base,
        walk: WalkArguments, rank: Int32, tokens_per_rank: Int32, bytes_per_token: Int32, peer_sf_base,
        sf_bytes_per_token: Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        scratch = utils.SmemAllocator().allocate_tensor(Int32, cute.make_layout(NUM_WARPS), byte_alignment=16)
        walk_scatter_phase(
            sched_params, a_idx, slot_of_row, s_reverse, topk, cu_seqlens, flags, counts2, peer_base, walk, rank,
            tokens_per_rank, bytes_per_token, Int32(cta), Int32(self.num_ctas), tid, tid % 32, tid // 32, scratch,
            self.block_rows, self.topk_k, self.even_split, peer_sf_base, sf_bytes_per_token, self.sf_atom_rows,
        )


# --- host ------------------------------------------------------------------------------------------------------------


def even_split_block_rows(rows, tile_n, granule=VARLEN_N_GRANULE):
    """Host twin of quack's varlen_n_block_rows over a tensor of per-expert row counts (0 rows -> tile_n)."""
    num_blocks = ((rows + tile_n - 1) // tile_n).clamp(min=1)
    return ((rows + num_blocks - 1) // num_blocks + granule - 1) // granule * granule


def flag_targets(cu_seqlens_m, flags, block_rows=BLOCK_ROWS, even_split=False):
    """What the flags read once every row has landed: rows of each (expert, block) in flags' shape; the blocks are
    block_rows rows (the GEMM's M cluster) or, even_split, the swap-AB GEMM's N blocks (block_rows = its tile N)."""
    rows = (cu_seqlens_m[1:] - cu_seqlens_m[:-1]).to(torch.int64)[:, None]
    b = even_split_block_rows(rows, block_rows) if even_split else torch.full_like(rows, block_rows)
    first = torch.arange(flags.shape[1], device=flags.device)[None, :] * b
    return torch.minimum((rows - first).clamp_(min=0), b).to(torch.int32)


def sf_atom_rows_of(block_rows):
    """SF rows one block slot owns: the GEMM's SF window in whole 128-row atoms (GemmSm100.sfb_window_atoms)."""
    return -(-block_rows // 128) * 128


def make_walk_buffers(W, T_local, max_rows, flags, first_src_row, block_rows, even_split=False, sf=False):
    """The walk's tensors, sized to their bounds (owned rows <= W * T_local, references <= max_rows): args = the
    WalkArguments the kernels take, plus the fields the host and the GEMM's scheduler use (flags (E_local, blocks)
    int32 with counters = its flat view, first_src_row (1,) int32, block_rows; sf: sf_atom_rows and cu_tiles = the
    GEMM's per-expert block prefix over the flag-indexed SF slots)."""
    dev = flags.device
    i32 = lambda n: torch.empty(n, dtype=torch.int32, device=dev)
    i64 = lambda n: torch.empty(n, dtype=torch.int64, device=dev)
    args = WalkArguments(
        src_address=i64(W * T_local), pool_row=i32(W * T_local), gate_ptr=i32(W * T_local + 1), gates=i32(max_rows),
        num_rows=i32(1), dst_row=i32(max_rows),
        src_sf_address=i64(W * T_local) if sf else None, dst_sf_row=i32(max_rows) if sf else None,
    )
    ncols = flags.shape[1]
    return SimpleNamespace(
        **args._asdict(), args=args, flags=flags, counters=flags.view(-1), first_src_row=first_src_row,
        block_rows=block_rows, even_split=even_split, sf_atom_rows=sf_atom_rows_of(block_rows) if sf else 0,
        cu_tiles=torch.arange(flags.shape[0] + 1, dtype=torch.int32, device=dev) * ncols if sf else None,
    )


def make_dispatch_walk(
    topk_idx_g, s_scatter, s_reverse, first_src_row, rank, a_idx, cu_seqlens, x_peer_bufs, block_rows=BLOCK_ROWS,
    even_split=False, x_sf_peer_bufs=None,
):
    """GPU-built walk in the GEMM's order, for make_dispatch_pull's run(walk) and the GEMM's dispatch scheduler:
    (build, walk). build() launches count + scatter in one host call, no sync; walk = make_walk_buffers' namespace
    plus targets = flag_targets(...) (the flags once every row has landed). s_scatter / s_reverse = sonic's
    s_scatter_idx / s_reverse_local (meta["recv_pos"]); a_idx / cu_seqlens = the GEMM's; block_rows = its block (the
    GEMM asserts it at compile); x_peer_bufs = every rank's x buffer; x_sf_peer_bufs (block-scaled formats: every
    rank's (T, K / sf_vec_size) 1-byte scale rows) adds the scale lane."""
    W, T_local, K = topk_idx_g.shape
    dev = a_idx.device
    topk = topk_idx_g.reshape(-1)
    peer_base = torch.tensor([b.data_ptr() for b in x_peer_bufs], dtype=torch.int64, device=dev)
    row_bytes = x_peer_bufs[rank].shape[1] * x_peer_bufs[rank].element_size()
    assert topk.dtype == s_scatter.dtype == s_reverse.dtype == a_idx.dtype == cu_seqlens.dtype == torch.int32
    # an expert reads each pool row at most once, so its blocks <= the pool's
    flags = torch.zeros(cu_seqlens.numel() - 1, -(-W * T_local // block_rows), dtype=torch.int32, device=dev)
    counts2 = torch.empty(2, flags.numel(), dtype=torch.int32, device=dev)
    sf = x_sf_peer_bufs is not None
    walk = make_walk_buffers(W, T_local, a_idx.numel(), flags, first_src_row, block_rows, even_split, sf)
    walk.targets = flag_targets(cu_seqlens, flags, block_rows, even_split)

    kernel = DispatchWalk(
        K, block_rows, torch.cuda.get_device_properties(dev).multi_processor_count, even_split, walk.sf_atom_rows
    )
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
        kernel, v(), v(), v(), v(), v(), m(), v(), m(), v64(), fake_walk_arguments(sf), Int32(rank), Int32(T_local),
        Int32(row_bytes), stream, sf_fake, Int32(sf_row_bytes), options="--enable-tvm-ffi",
    )
    build = lambda: fn(
        a_idx, s_scatter, s_reverse, topk, cu_seqlens, flags, walk.first_src_row, counts2, peer_base, walk.args,
        rank, T_local, row_bytes, sf_arg, sf_row_bytes,
    )
    return build, walk
