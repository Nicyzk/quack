"""Inter-SM EP transfer kernels: the dispatch pull and the combine push on their own SMs (a green-context partition
next to quack's GEMM), the device-built walks they run over, and the pull_rows / push_rows loops the intra-SM fused
GEMMs (quack.gemm_sm100_ep_intra_dispatch / _combine) run on their transfer warps."""
# Dispatch transport leg: MegaMoE's deterministic `token_in` transfer-warp loop as a standalone kernel, one CTA
# per SM, one row in flight per warp, bf16 rows (no scale factors). Pulls rows from peers' x buffers into the
# receiver's pool with cp.async.bulk via a per-warp smem stage; once a row's bytes have landed, its gate
# counters get a release-add of 1 each. Two modes: "intra" = 4 warps / 56 KB stage, what the transfer warps
# look like inside the GEMM; "inter" = as many warps as the SM's smem holds rows (16 on B300), the inter-SM leg
# with the most bytes in flight per SM.
#
# The walk is a list of pool rows in pull order plus a CSR of gate ids per row, built on the GPU by
# make_dispatch_walk() (end of file) in the GEMM's M cluster order (expert first, shifted source rank inside; one
# gate = one flag[expert, cid_m] per M cluster, so the gated GEMM trails the pull by one cluster). Each walk entry
# carries its source row's absolute address (source rank's x buffer + token row), so the pull reads no routing
# metadata. Deviations from MegaMoE: source addresses from the walk (torch symm-mem has no single heap offset), walk
# from gmem, SF pull dropped for bf16, and a full wait_group (no .read) before each publication batch so the
# counters follow the landed bytes.
#
# Signals are batched, not per row: a red.release carries a gpu-scope fence that waits on the previous add, so
# one per gate serialized on the warp. Measured at W=2, 16 SMs, full mode, expert-src_rank order (~4 gates per row):
# 157 GB/s per gate vs 371 with no signals; parking gates in lanes and firing 8 rows' worth in one instruction
# gives 351, a publish lag of 8 rows.
from types import SimpleNamespace

import torch
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.experimental import iket
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import AddressSpace
from cutlass.cutlass_dsl import Boolean, Int32, Int64
from torch.distributed import _symmetric_memory as symm_mem
from quack.compile_utils import make_fake_tensor
from quack.ep.ptx import cp_async_bulk_s2g, red_add_release_gpu_s32, red_add_release_sys_s32, tma_load_1d
from quack.utils import warp_prefix_sum
from quack.varlen_utils import VARLEN_N_GRANULE, varlen_n_block_rows
from quack.tile_scheduler import (
    CombineSchedulerArguments,
    DispatchSchedulerArguments,
    PersistenceMode,
    RasterOrderOption,
    VarlenMTileScheduler,
    VarlenMTileSchedulerArguments,
)


def pull_chunk_bytes(hidden, dtype, max_bytes=4096):
    """One transfer warp's stage: the row halved until <= max_bytes (megamoe's num_bytes_per_pull; 3584 B at H 7168)."""
    chunk = hidden * dtype.width // 8
    while chunk > max_bytes:
        assert chunk % 2 == 0
        chunk //= 2
    return chunk


def stage_bytes(hidden, dtype, warps, chunk_bytes=0):
    per_warp = chunk_bytes or hidden * dtype.width // 8  # a chunk of the row, or the row
    return warps * per_warp + 8 * warps + 256  # stages + mbarriers + alignment slack


@cute.jit
def pull_rows(
    pool_base: Int64,  # address of pool row 0; rows are bytes_per_token apart
    bytes_per_token: Int32,
    walk_src_address: cute.Tensor,  # (n_rows,) int64 absolute address of each entry's row in its source's x buffer
    walk_pool_row: cute.Tensor,  # (n_rows,) int32 pool row of each walk entry, in pull order
    walk_gate_ptr: cute.Tensor,  # (n_rows + 1,) int32 CSR into walk_gates
    walk_gates: cute.Tensor,  # (n_refs,) int32 gate ids to bump once the entry's row has landed
    gate_counters: cute.Tensor,  # (n_gates,) int32, zeroed before launch
    stage: cute.Tensor,  # this warp's smem row (hidden,)
    mbarrier: cute.Pointer,  # this warp's smem mbarrier, initialized with arrive count 1
    global_warp_idx: Int32,
    global_warp_count: Int32,
    lane_idx: Int32,
    flag_batch: cutlass.Constexpr[int],
    num_rows: cute.Tensor,  # (1,) int32 walk length (the walk arrays are sized to their bounds)
    walk_dst_row: cute.Tensor,  # (n_refs,) int32 expert-major row of each reference (parallel to walk_gates)
    scatter_all: cutlass.Constexpr[bool],  # no-gather layout: the row lands in every reader's row, not the pool row
):
    """One transfer warp's share of the walk (entries global_warp_idx, += global_warp_count): MegaMoE's token_in
    loop (SF path dropped) with lane-distributed publication. Shared by the standalone kernel and the fused GEMM."""
    activation_bytes = Int32(cute.cosize(stage) * int(stage.element_type.width) // 8)
    n_rows = num_rows[0]
    pull_count = Int32(0)
    if global_warp_idx < n_rows:
        pull_count = (n_rows - global_warp_idx + global_warp_count - Int32(1)) // global_warp_count
    pull_phase = Int32(0)
    # Lane-distributed publication (MegaMoE flag_batch): lane k parks the k-th pending gate; every
    # flag_batch rows (or when the lanes are full) the warp waits for its stores to land and fires one
    # red.release per parked lane in a single instruction, so the fences overlap into one round trip.
    pending = Int32(0)  # parked gates, warp-uniform
    rows_parked = Int32(0)
    lane_gate = Int32(0)  # this lane's parked gate (valid while lane_idx < pending)
    for pull_round in cutlass.range(pull_count, unroll=1):
        entry = global_warp_idx + Int32(pull_round) * global_warp_count
        pool_token_idx = walk_pool_row[entry]
        remote_activation_address = walk_src_address[entry]

        gate_begin = walk_gate_ptr[entry]
        gate_count = walk_gate_ptr[entry + Int32(1)] - gate_begin
        # the row moves through the stage in chunks of the stage's size (one chunk = the whole row)
        for chunk in cutlass.range(bytes_per_token // activation_bytes, unroll=1):
            chunk_offset = Int64(chunk) * Int64(activation_bytes)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(mbarrier, activation_bytes)
                tma_load_1d(
                    stage.iterator, Int64(remote_activation_address) + chunk_offset, mbarrier, activation_bytes
                )
            cute.arch.sync_warp()
            cute.arch.mbarrier_wait(mbarrier, pull_phase)
            pull_phase = pull_phase ^ Int32(1)
            if cutlass.const_expr(scatter_all):
                for j in cutlass.range(gate_count, unroll=1):  # one store per reader row, all in one bulk group
                    destination_activation = cute.make_ptr(
                        stage.element_type,
                        pool_base + Int64(walk_dst_row[gate_begin + j]) * Int64(bytes_per_token) + chunk_offset,
                        AddressSpace.gmem,
                        assumed_align=16,
                    )
                    with cute.arch.elect_one():
                        cp_async_bulk_s2g(destination_activation, stage.iterator, activation_bytes)
                    cute.arch.sync_warp()
            else:
                destination_activation = cute.make_ptr(
                    stage.element_type,
                    pool_base + Int64(pool_token_idx) * Int64(bytes_per_token) + chunk_offset,
                    AddressSpace.gmem,
                    assumed_align=16,
                )
                with cute.arch.elect_one():
                    cp_async_bulk_s2g(destination_activation, stage.iterator, activation_bytes)
                cute.arch.sync_warp()
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(0, read=True)  # stage free for the next chunk; landing is waited at flush
            cute.arch.sync_warp()
        if pending + gate_count > Int32(32):  # no lanes left: flush what is parked first
            cute.arch.cp_async_bulk_wait_group(0, read=False)
            if lane_idx < pending:
                red_add_release_gpu_s32(gate_counters.iterator + lane_gate, Int32(1))
            pending = Int32(0)
            rows_parked = Int32(0)
        if (lane_idx >= pending) & (lane_idx < pending + gate_count):
            lane_gate = walk_gates[gate_begin + (lane_idx - pending)]
        pending = pending + gate_count
        rows_parked = rows_parked + Int32(1)
        if rows_parked == Int32(flag_batch):
            cute.arch.cp_async_bulk_wait_group(0, read=False)  # the parked rows' bytes have landed
            if lane_idx < pending:
                red_add_release_gpu_s32(gate_counters.iterator + lane_gate, Int32(1))
            pending = Int32(0)
            rows_parked = Int32(0)
        cute.arch.sync_warp()
    if pending > Int32(0):
        cute.arch.cp_async_bulk_wait_group(0, read=False)
        if lane_idx < pending:
            red_add_release_gpu_s32(gate_counters.iterator + lane_gate, Int32(1))
    cute.arch.sync_warp()


class DispatchPull:
    flag_batch = 8  # rows per publication batch (MegaMoE token_in_flag_batch)

    def __init__(self, hidden, dtype, sm_count, transfer_warps):
        self.hidden = hidden
        self.dtype = dtype
        self.sm_count = sm_count
        self.transfer_warp_count = transfer_warps
        self.bytes_per_token = hidden * dtype.width // 8
        # Request more than half the SM's smem so only one CTA fits: the fused kernel's occupancy.
        self.smem_bytes = max(stage_bytes(hidden, dtype, transfer_warps), 116 * 1024)

    @cute.jit
    def __call__(
        self, pool, walk_src_address, walk_pool_row, walk_gate_ptr, walk_gates, gate_counters, walk_num_rows, stream
    ):
        self.kernel(
            pool, walk_src_address, walk_pool_row, walk_gate_ptr, walk_gates, gate_counters, walk_num_rows
        ).launch(
            grid=[self.sm_count, 1, 1],
            block=[self.transfer_warp_count * 32, 1, 1],
            smem=self.smem_bytes,
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        pool: cute.Tensor,  # (pool rows, hidden) destination
        walk_src_address: cute.Tensor,  # (n_rows,) int64 absolute address of each entry's source row
        walk_pool_row: cute.Tensor,  # (n_rows,) int32 pool row of each walk entry, in pull order
        walk_gate_ptr: cute.Tensor,  # (n_rows + 1,) int32 CSR into walk_gates
        walk_gates: cute.Tensor,  # (n_refs,) int32 gate ids to bump once the entry's row has landed
        gate_counters: cute.Tensor,  # (n_gates,) int32, zeroed before launch
        walk_num_rows: cute.Tensor,  # (1,) int32 entries to pull (the walk arrays may be longer)
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        transfer_warp_idx = tidx // 32
        lane_idx = tidx % 32
        global_warp_idx = bidx * Int32(self.transfer_warp_count) + transfer_warp_idx
        global_warp_count = Int32(self.sm_count * self.transfer_warp_count)

        smem = utils.SmemAllocator()
        token_in_activation = smem.allocate_tensor(
            self.dtype,
            cute.make_layout((self.transfer_warp_count, self.hidden), stride=(self.hidden, 1)),  # one row per warp
            byte_alignment=128,
        )
        token_in_mbarriers = smem.allocate_array(cutlass.Int64, self.transfer_warp_count)
        warp_mbarrier = token_in_mbarriers + transfer_warp_idx
        if lane_idx == Int32(0):
            cute.arch.mbarrier_init(warp_mbarrier, 1)
        cute.arch.sync_warp()
        pull_rows(
            pool.iterator.toint(),
            Int32(self.bytes_per_token),
            walk_src_address,
            walk_pool_row,
            walk_gate_ptr,
            walk_gates,
            gate_counters,
            token_in_activation[transfer_warp_idx, None],
            warp_mbarrier,
            global_warp_idx,
            global_warp_count,
            lane_idx,
            self.flag_batch,
            walk_num_rows,
            walk_pool_row,  # unused: the inter-SM kernel keeps the dedup pool layout
            False,
        )

def make_dispatch_pull(x_peer_bufs, rank, mode="intra", sm_count=None, pool=None):
    """Host side: the pool (every rank's tokens fit: sonic's dedup receive layout) and a compiled pull in `mode`
    ("intra" / "inter") on sm_count CTAs (default: every SM). Returns (run, pool); run(walk) is the callable for a
    make_dispatch_walk walk: zero its counters (unless zero_counters=False: a caller holding them at target), pull."""
    W, T_local = len(x_peer_bufs), x_peer_bufs[rank].shape[0]
    H = x_peer_bufs[rank].shape[1]
    dev = x_peer_bufs[rank].device
    dtype = x_peer_bufs[rank].dtype
    cute_dtype = {torch.bfloat16: cutlass.BFloat16, torch.float16: cutlass.Float16}[dtype]
    props = torch.cuda.get_device_properties(dev)
    if mode == "intra":
        warps = 4  # what the transfer warps look like inside the GEMM
    else:  # "inter": as many rows in flight as the SM's smem holds
        assert mode == "inter", mode
        warps = max(w for w in range(1, 33) if stage_bytes(H, cute_dtype, w) <= props.shared_memory_per_block_optin)
    sm_count = props.multi_processor_count if sm_count is None else sm_count
    pool = torch.empty(W * T_local, H, dtype=dtype, device=dev) if pool is None else pool

    kernel = DispatchPull(H, cute_dtype, sm_count, warps)
    i32 = lambda: make_fake_tensor(Int32, (cute.sym_int(),), leading_dim=0, divisibility=1)  # walk arrays vary
    i64 = lambda: make_fake_tensor(Int64, (cute.sym_int(),), leading_dim=0, divisibility=1)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    fn = cute.compile(
        kernel, from_dlpack(pool, enable_tvm_ffi=True), i64(), i32(), i32(), i32(), i32(), i32(), stream,
        options="--enable-tvm-ffi",
    )

    def run(walk, zero_counters=True):
        def go():
            if zero_counters:
                walk.counters.zero_()
            fn(pool, walk.src_address, walk.pool_row, walk.gate_ptr, walk.gates, walk.counters, walk.num_rows)

        return go

    return run, pool


# --- The pull's walk built on the device, in the GEMM's M cluster order (no host walk) --------------------------
# M cluster = 256 consecutive M rows of one expert: the GEMM tile's M extent and the unit one flag gates. One CTA per
# cluster (256 threads, one M row each), clusters in the GEMM's walk order: quack's varlen-M scheduler with the GEMM's
# DispatchSchedulerArguments over a one-N-cluster problem (expert-major, each expert rotated to start at
# first_src_row). An A row is owned by the token's local reader with the smallest M row (its first use in walk order),
# found from sonic's s_scatter_idx (M row -> global slot) and s_reverse_local (slot -> M row) over the token's K slots.
# Launch 1 counts owned rows and reader references per cluster; launch 2 offsets each cluster by the earlier
# clusters' counts and writes the list (pool_row, src_address, gate_ptr CSR, gates as flat flags[e, cid_m] offsets,
# num_rows) for pull_rows.


class MClusterWalk(VarlenMTileScheduler):
    """Work index -> (expert, cid_m), decoded by the calling warp (persistence NONE: no scheduler warp)."""

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
    cid_m: Int32,
    tidx: Int32,
    slot_of_row: cute.Tensor,
    s_reverse: cute.Tensor,
    topk: cute.Tensor,
    cu_seqlens: cute.Tensor,
    flags: cute.Tensor,
    rank: Int32,
    num_experts: Int32,
    reader_flags: cute.Tensor,
    reader_rows: cute.Tensor,
    block_rows: cutlass.Constexpr[int],
    topk_k: cutlass.Constexpr[int],
    even_split: cutlass.Constexpr[bool] = False,
):
    """M row tidx of cluster (expert, cid_m): (owner, readers, token, owned, refs) = whether it owns its A row, a
    bitmask over its token's K slots of the local readers, whose flag offsets (flat into flags[e, cid_m]) go to
    reader_flags[k] and whose expert-major rows to reader_rows[k], the token's global index (source rank * tokens
    per rank + token), and 1 / popcount(readers) if owner else 0 / 0. even_split: blocks of block_rows_of(e)."""
    b = block_rows_of(cu_seqlens, expert, block_rows, even_split)
    cluster_start = cu_seqlens[expert] + cid_m * b
    row = cluster_start + tidx
    in_cluster = row < cutlass.min(cluster_start + b, cu_seqlens[expert + 1])
    owner = Boolean(False)
    readers = Int32(0)
    token = Int32(0)
    if in_cluster:
        token = slot_of_row[row] // topk_k
        token_slot = token * topk_k
        first_reader = Int32(0x7FFFFFFF)
        for k in cutlass.range_constexpr(topk_k):
            e = topk[token_slot + k] - rank * num_experts
            reader_row = s_reverse[token_slot + k]
            reader_flags[k] = Int32(0)
            reader_rows[k] = Int32(0)
            if (e >= 0) & (e < num_experts):
                first_reader = cutlass.min(first_reader, reader_row)
                readers = readers | Int32(1 << k)
                reader_flags[k] = Int32(
                    cute.crd2idx(
                        (e, (reader_row - cu_seqlens[e]) // block_rows_of(cu_seqlens, e, block_rows, even_split)),
                        flags.layout,
                    )
                )
                reader_rows[k] = reader_row
        owner = first_reader == row
    owned, refs = Int32(0), Int32(0)
    if owner:
        owned, refs = Int32(1), cute.arch.popc(readers)
    return owner, readers, token, owned, refs


class DispatchWalk:
    """Both launches: one CTA per M cluster (block_rows threads, the GEMM's M cluster), persistent over num_ctas
    CTAs in walk order. even_split: the swap-AB GEMM's N blocks (block_rows = its tile N, the block ids ride the
    scheduler's N coordinate)."""

    def __init__(self, topk_k, block_rows, num_ctas, even_split=False):
        self.topk_k, self.block_rows, self.num_ctas = topk_k, block_rows, num_ctas
        self.even_split = even_split
        self.vd = 1 if even_split else 0  # the ragged dim in the scheduler mirror
        self.num_warps = block_rows // 32

    @cute.jit
    def __call__(
        self, a_idx, slot_of_row, s_reverse, topk, cu_seqlens, flags, first_src_row, counts, peer_base,
        pool_row, src_address, gate_ptr, gates, dst_row, num_rows, rank: Int32, tokens_per_rank: Int32,
        bytes_per_token: Int32, stream,
    ):
        def sched_params():  # one per launch: a launch rebinds the params' tensors to its own kernel's arguments
            return MClusterWalk.to_underlying_arguments(
                VarlenMTileSchedulerArguments(
                    problem_shape_ntile_mnl=(
                        (None, 1, cute.size(cu_seqlens) - 1)
                        if not self.even_split
                        else (1, None, cute.size(cu_seqlens) - 1)
                    ),
                    total_m=cute.size(a_idx),
                    cu_seqlens_m=cu_seqlens,
                    raster_order=RasterOrderOption.AlongN,
                    group_size=1,
                    tile_shape_mn=(self.block_rows, 1) if not self.even_split else (1, self.block_rows),
                    cluster_shape_mnk=(1, 1, 1),
                    persistence_mode=PersistenceMode.NONE,
                    dispatch=DispatchSchedulerArguments(flags, first_src_row),
                    a_idx=a_idx,
                    varlen_dim=self.vd,
                )
            )

        common = (a_idx, slot_of_row, s_reverse, topk, cu_seqlens, flags, counts, rank)
        launch = dict(grid=[self.num_ctas, 1, 1], block=[self.block_rows, 1, 1], smem=256, stream=stream)
        self.count_kernel(sched_params(), *common).launch(**launch)
        self.scatter_kernel(
            sched_params(), *common, peer_base, pool_row, src_address, gate_ptr, gates, dst_row, num_rows,
            tokens_per_rank, bytes_per_token,
        ).launch(**launch)

    @cute.kernel
    def count_kernel(
        self, sched_params, a_idx, slot_of_row, s_reverse, topk, cu_seqlens, flags, counts, rank: Int32
    ):
        """counts[0, q] = owned A rows, counts[1, q] = their reader references, of the cluster at walk position q."""
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp, lane = tidx // 32, tidx % 32
        scratch = utils.SmemAllocator().allocate_tensor(Int32, cute.make_layout(self.num_warps), byte_alignment=16)
        reader_flags = cute.make_rmem_tensor(cute.make_layout(self.topk_k), Int32)
        reader_rows = cute.make_rmem_tensor(cute.make_layout(self.topk_k), Int32)
        num_experts = Int32(cute.size(cu_seqlens) - 1)
        sched = MClusterWalk.create(sched_params)
        q = Int32(bidx)
        tile = sched.m_cluster_at(q)
        while tile.is_valid_tile:
            _, _, _, owned, refs = owner_test(
                tile.tile_idx[3], tile.tile_idx[self.vd], tidx, slot_of_row, s_reverse, topk, cu_seqlens, flags,
                rank, num_experts, reader_flags, reader_rows, self.block_rows, self.topk_k, self.even_split,
            )
            owned = block_sum(owned, scratch, warp, lane, self.num_warps)
            refs = block_sum(refs, scratch, warp, lane, self.num_warps)
            if tidx == Int32(0):
                counts[0, q] = owned
                counts[1, q] = refs
            q = q + Int32(self.num_ctas)
            tile = sched.m_cluster_at(q)

    @cute.kernel
    def scatter_kernel(
        self, sched_params, a_idx, slot_of_row, s_reverse, topk, cu_seqlens, flags, counts, rank: Int32,
        peer_base, pool_row, src_address, gate_ptr, gates, dst_row, num_rows, tokens_per_rank: Int32,
        bytes_per_token: Int32,
    ):
        """Each cluster's owned A rows at (earlier clusters' owned rows + rank among its owners), in M row order."""
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp, lane = tidx // 32, tidx % 32
        scratch = utils.SmemAllocator().allocate_tensor(Int32, cute.make_layout(self.num_warps), byte_alignment=16)
        reader_flags = cute.make_rmem_tensor(cute.make_layout(self.topk_k), Int32)
        reader_rows = cute.make_rmem_tensor(cute.make_layout(self.topk_k), Int32)
        num_experts = Int32(cute.size(cu_seqlens) - 1)
        if bidx == Int32(0):  # totals: num_rows = all owned rows, gate_ptr[num_rows] = all references (= M rows)
            clusters = Int32(0)
            for i in cutlass.range((num_experts - tidx + self.block_rows - 1) // self.block_rows, unroll=1):
                e = tidx + i * self.block_rows
                clusters = clusters + (cu_seqlens[e + 1] - cu_seqlens[e] + self.block_rows - 1) // self.block_rows
            clusters = block_sum(clusters, scratch, warp, lane, self.num_warps)
            owned = Int32(0)
            for i in cutlass.range((clusters - tidx + self.block_rows - 1) // self.block_rows, unroll=1):
                owned = owned + counts[0, tidx + i * self.block_rows]
            owned = block_sum(owned, scratch, warp, lane, self.num_warps)
            if tidx == Int32(0):
                num_rows[0] = owned
                gate_ptr[owned] = cu_seqlens[num_experts]
        sched = MClusterWalk.create(sched_params)
        q = Int32(bidx)
        tile = sched.m_cluster_at(q)
        while tile.is_valid_tile:
            row_offset, ref_offset = Int32(0), Int32(0)
            for i in cutlass.range((q - tidx + self.block_rows - 1) // self.block_rows, unroll=1):
                row_offset = row_offset + counts[0, tidx + i * self.block_rows]
                ref_offset = ref_offset + counts[1, tidx + i * self.block_rows]
            row_offset = block_sum(row_offset, scratch, warp, lane, self.num_warps)
            ref_offset = block_sum(ref_offset, scratch, warp, lane, self.num_warps)
            expert, cid_m = tile.tile_idx[3], tile.tile_idx[self.vd]
            owner, readers, token, owned, refs = owner_test(
                expert, cid_m, tidx, slot_of_row, s_reverse, topk, cu_seqlens, flags, rank, num_experts,
                reader_flags, reader_rows, self.block_rows, self.topk_k, self.even_split,
            )
            entry = row_offset + block_exclusive_scan(owned, scratch, warp, lane, self.num_warps)
            ref = ref_offset + block_exclusive_scan(refs, scratch, warp, lane, self.num_warps)
            if owner:
                b = block_rows_of(cu_seqlens, expert, self.block_rows, self.even_split)
                pool_row[entry] = a_idx[cu_seqlens[expert] + cid_m * b + tidx]
                src_address[entry] = peer_base[token // tokens_per_rank] + Int64(token % tokens_per_rank) * Int64(
                    bytes_per_token
                )
                gate_ptr[entry] = ref
                for k in cutlass.range_constexpr(self.topk_k):
                    if ((readers >> k) & Int32(1)) == Int32(1):
                        gates[ref] = reader_flags[k]
                        dst_row[ref] = reader_rows[k]
                        ref = ref + Int32(1)
            q = q + Int32(self.num_ctas)
            tile = sched.m_cluster_at(q)


BLOCK_ROWS = 256  # the GEMM's M cluster (tile M 256, cluster (2, 1)): the unit one flag gates and the walk is built on


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


def make_dispatch_walk(
    topk_idx_g, s_scatter, s_reverse, first_src_row, rank, a_idx, cu_seqlens, x_peer_bufs, block_rows=BLOCK_ROWS,
    even_split=False,
):
    """GPU-built walk in the GEMM's order, for make_dispatch_pull's run(walk) and the GEMM's dispatch scheduler:
    (build, walk). build() launches both kernels in one host call, no sync; walk = (pool_row, src_address, gate_ptr,
    gates, num_rows) sized to their bounds, flags (E_local, max M clusters) int32 with counters = its flat view and
    targets = flag_targets(...) (the flags once every row has landed), first_src_row (1,) int32 (the scheduler's
    DispatchSchedulerArguments) and block_rows. s_scatter / s_reverse = sonic's s_scatter_idx / s_reverse_local
    (meta["recv_pos"]); a_idx / cu_seqlens = the GEMM's; block_rows = its M cluster (the GEMM asserts it at
    compile); x_peer_bufs = every rank's x buffer."""
    W, T_local, K = topk_idx_g.shape
    dev = a_idx.device
    topk = topk_idx_g.reshape(-1)
    peer_base = torch.tensor([b.data_ptr() for b in x_peer_bufs], dtype=torch.int64, device=dev)
    row_bytes = x_peer_bufs[rank].shape[1] * x_peer_bufs[rank].element_size()
    assert topk.dtype == s_scatter.dtype == s_reverse.dtype == a_idx.dtype == cu_seqlens.dtype == torch.int32
    i32 = lambda n: torch.empty(n, dtype=torch.int32, device=dev)
    # an expert reads each pool row at most once, so its M clusters <= the pool's
    flags = torch.zeros(cu_seqlens.numel() - 1, -(-W * T_local // block_rows), dtype=torch.int32, device=dev)
    counts = torch.empty(2, flags.numel(), dtype=torch.int32, device=dev)  # clusters <= flags' size
    walk = SimpleNamespace(
        pool_row=i32(W * T_local),  # owned A rows <= A rows
        src_address=torch.empty(W * T_local, dtype=torch.int64, device=dev),
        gate_ptr=i32(W * T_local + 1),
        gates=i32(a_idx.numel()),  # one reference per M row
        dst_row=i32(a_idx.numel()),  # the reference's expert-major row (the no-gather layout's store target)
        num_rows=i32(1),
        flags=flags,
        counters=flags.view(-1),
        targets=flag_targets(cu_seqlens, flags, block_rows, even_split),
        first_src_row=first_src_row,  # (1,) int32: the pool row where source rank (rank + 1) % W's tokens start
        block_rows=block_rows,
    )

    kernel = DispatchWalk(K, block_rows, torch.cuda.get_device_properties(dev).multi_processor_count, even_split)
    v = lambda: make_fake_tensor(Int32, (cute.sym_int(),), leading_dim=0, divisibility=1)
    v64 = lambda: make_fake_tensor(Int64, (cute.sym_int(),), leading_dim=0, divisibility=1)
    m = lambda: make_fake_tensor(Int32, (cute.sym_int(), cute.sym_int()), leading_dim=1, divisibility=1)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    fn = cute.compile(
        kernel, v(), v(), v(), v(), v(), m(), v(), m(), v64(), v(), v64(), v(), v(), v(), v(),
        Int32(rank), Int32(T_local), Int32(row_bytes), stream,
        options="--enable-tvm-ffi",
    )
    build = lambda: fn(
        a_idx, s_scatter, s_reverse, topk, cu_seqlens, flags, walk.first_src_row, counts, peer_base,
        walk.pool_row, walk.src_address, walk.gate_ptr, walk.gates, walk.dst_row, walk.num_rows, rank, T_local,
        row_bytes,
    )
    return build, walk


# =====================================================================================================================
# Combine transport leg: the dispatch pull above run the other way. One CTA per SM, one row
# in flight per warp, bf16 rows (scores folded in beforehand): each transfer warp walks its share of this rank's
# expert-major y rows, stages a row in smem with cp.async.bulk and pushes it to slot (src_token, src_topk) of the
# origin rank's (T_local, K, H) slot buffer; once a batch of rows has landed, each row's origin gets one
# release-add on its per-token counter (sys scope: the consumer is a peer), lane-distributed as in pull_rows, so
# the origin can reduce a token once its counter reads K. Two modes as the pull: "intra" = 4 warps / 56 KB stage
# (the transfer warps inside a GEMM), "inter" = as many warps as the SM's smem holds rows (16 on B300). Inside the
# GEMM the row goes out in chunk_bytes pieces, each waited to completion before the next: a bulk store to a peer
# holds the SM's TMA path for its NVLink round trip in front of the GEMM's operand loads, so what is outstanding
# per warp must stay small (2 KB recovers 2/3 of the loss).
#
# The walk (make_combine_walk, end of file) is built with torch: y rows in GEMM2's order (expert-major) with the
# destination rank rotated inside each expert (rank+1 first, as the pull's source order: on a small SM partition
# the warps in flight cover one destination's run of rows, and every rank pushing to the same peer at once is the
# packed-by-destination case measured 2.6x slower); each entry carries its y row, its peer slot address and its
# peer counter address. Deviations from MegaMoE's token_back: walk from gmem with absolute peer addresses, no
# fc2_done wait (standalone rows are ready; the fused GEMM adds the flag wait), no pacing, no scale factors.


@cute.jit
def push_rows(
    y_base: Int64,  # address of y row 0; rows are bytes_per_token apart
    bytes_per_token: Int32,
    walk_row: cute.Tensor,  # (n_rows,) int32 y row of each walk entry, in push order
    walk_dst_address: cute.Tensor,  # (n_rows,) int64 absolute address of the entry's slot in its origin's buffer
    walk_gate_address: cute.Tensor,  # (n_rows,) int64 absolute address of the origin's counter for the row's token
    walk_flag_address: cute.Tensor,  # (n_rows,) int64 absolute address of the row's M cluster flag (GEMM2 stored)
    flag_target: Int32,  # what a cluster's flag reads once GEMM2 stored all its N tiles
    stage: cute.Tensor,  # this warp's smem stage: the row (hidden,), or one chunk of it (chunk_bytes)
    mbarrier: cute.Pointer,  # this warp's smem mbarrier, initialized with arrive count 1
    global_warp_idx: Int32,
    global_warp_count: Int32,
    lane_idx: Int32,
    flag_batch: cutlass.Constexpr[int],
    num_rows: cute.Tensor,  # (1,) int32 walk length (the walk arrays are sized to their bounds)
    chunk_bytes: cutlass.Constexpr[int] = 0,  # > 0: the row goes out in chunks of this size through a chunk-sized stage
):
    """One transfer warp's share of the walk (entries global_warp_idx, += global_warp_count): pull_rows mirrored,
    local row in (once its M cluster's flag says GEMM2 stored it), peer slot out, one remote counter per row.
    Shared by the standalone kernel and the fused GEMM."""
    stage_bytes_ = cute.cosize(stage) * int(stage.element_type.width) // 8
    assert chunk_bytes == 0 or (stage_bytes_ == chunk_bytes and chunk_bytes % 16 == 0), (chunk_bytes, stage_bytes_)
    activation_bytes = Int32(stage_bytes_)
    n_rows = num_rows[0]
    push_count = Int32(0)
    if global_warp_idx < n_rows:
        push_count = (n_rows - global_warp_idx + global_warp_count - Int32(1)) // global_warp_count
    push_phase = Int32(0)
    # Lane-distributed publication (MegaMoE flag_batch): lane k parks the k-th pending row's counter; every
    # flag_batch rows the warp waits for its stores to land and fires one red.release per parked lane in a single
    # instruction, so the sys-scope fences overlap into one round trip.
    pending = Int32(0)  # parked rows, warp-uniform
    lane_gate = Int64(0)  # this lane's parked counter address (valid while lane_idx < pending)
    last_flag = Int64(-1)  # the M cluster flag last seen satisfied (warp-uniform)
    for push_round in cutlass.range(push_count, unroll=1):
        entry = global_warp_idx + Int32(push_round) * global_warp_count
        row = walk_row[entry]
        flag_address = walk_flag_address[entry]
        if flag_address != last_flag:  # a new M cluster: wait until GEMM2 stored every N tile of it
            # relaxed, as quack's dispatch_wait_m_cluster: the gated rows are read by TMA at L2, never through L1
            iket.range_push("push_wait")
            flag = cute.make_ptr(Int32, flag_address, AddressSpace.gmem, assumed_align=4)
            val = cute.arch.load(flag.llvm_ptr, Int32, sem="relaxed", scope="gpu")
            while val < flag_target:
                val = cute.arch.load(flag.llvm_ptr, Int32, sem="relaxed", scope="gpu")
            last_flag = flag_address
            iket.range_pop()
        iket.range_push("push_row")
        source = y_base + Int64(row) * Int64(bytes_per_token)
        destination = cute.make_ptr(
            stage.element_type, walk_dst_address[entry], AddressSpace.gmem, assumed_align=16
        )
        if cutlass.const_expr(chunk_bytes):
            chunk_elems = chunk_bytes * 8 // int(stage.element_type.width)
            for chunk in cutlass.range(bytes_per_token // Int32(chunk_bytes), unroll=1):  # one chunk-sized stage
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(mbarrier, Int32(chunk_bytes))
                    tma_load_1d(stage.iterator, source + Int64(chunk) * Int64(chunk_bytes), mbarrier, Int32(chunk_bytes))
                cute.arch.sync_warp()
                cute.arch.mbarrier_wait(mbarrier, push_phase)
                push_phase = push_phase ^ Int32(1)
                with cute.arch.elect_one():
                    cp_async_bulk_s2g(destination + chunk * Int32(chunk_elems), stage.iterator, Int32(chunk_bytes))
                cute.arch.sync_warp()
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=False)  # landed: the stage is free for the next chunk
                cute.arch.sync_warp()
        else:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(mbarrier, activation_bytes)
                tma_load_1d(stage.iterator, source, mbarrier, activation_bytes)
            cute.arch.sync_warp()
            cute.arch.mbarrier_wait(mbarrier, push_phase)
            push_phase = push_phase ^ Int32(1)
            with cute.arch.elect_one():
                cp_async_bulk_s2g(destination, stage.iterator, activation_bytes)
            cute.arch.sync_warp()
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(0, read=True)  # stage free for the next load; landing is waited at flush
            cute.arch.sync_warp()
        iket.range_pop()
        if lane_idx == pending:
            lane_gate = walk_gate_address[entry]
        pending = pending + Int32(1)
        if pending == Int32(flag_batch):
            cute.arch.cp_async_bulk_wait_group(0, read=False)  # the parked rows' bytes have landed
            if lane_idx < pending:
                red_add_release_sys_s32(lane_gate, Int32(1))
            pending = Int32(0)
        cute.arch.sync_warp()
    if pending > Int32(0):
        cute.arch.cp_async_bulk_wait_group(0, read=False)
        if lane_idx < pending:
            red_add_release_sys_s32(lane_gate, Int32(1))
    cute.arch.sync_warp()


class CombinePush:
    flag_batch = 8  # rows per publication batch (MegaMoE flag_batch)

    def __init__(self, hidden, dtype, sm_count, transfer_warps):
        self.hidden = hidden
        self.dtype = dtype
        self.sm_count = sm_count
        self.transfer_warp_count = transfer_warps
        self.bytes_per_token = hidden * dtype.width // 8
        # Request more than half the SM's smem so only one CTA fits: the fused kernel's occupancy.
        self.smem_bytes = max(stage_bytes(hidden, dtype, transfer_warps), 116 * 1024)

    @cute.jit
    def __call__(
        self, y, walk_row, walk_dst_address, walk_gate_address, walk_flag_address, walk_num_rows,
        flag_target: Int32, stream,
    ):
        self.kernel(
            y, walk_row, walk_dst_address, walk_gate_address, walk_flag_address, walk_num_rows, flag_target
        ).launch(
            grid=[self.sm_count, 1, 1],
            block=[self.transfer_warp_count * 32, 1, 1],
            smem=self.smem_bytes,
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        y: cute.Tensor,  # (rows, hidden) this rank's send rows, expert-major, scores folded in
        walk_row: cute.Tensor,  # (n_rows,) int32 y row of each walk entry, in push order
        walk_dst_address: cute.Tensor,  # (n_rows,) int64 absolute address of the entry's peer slot
        walk_gate_address: cute.Tensor,  # (n_rows,) int64 absolute address of the entry's peer counter
        walk_flag_address: cute.Tensor,  # (n_rows,) int64 absolute address of the entry's M cluster flag
        walk_num_rows: cute.Tensor,  # (1,) int32 entries to push (the walk arrays may be longer)
        flag_target: Int32,  # a cluster's flag once GEMM2 stored all its N tiles
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        transfer_warp_idx = tidx // 32
        lane_idx = tidx % 32
        global_warp_idx = bidx * Int32(self.transfer_warp_count) + transfer_warp_idx
        global_warp_count = Int32(self.sm_count * self.transfer_warp_count)

        smem = utils.SmemAllocator()
        token_back_activation = smem.allocate_tensor(
            self.dtype,
            cute.make_layout((self.transfer_warp_count, self.hidden), stride=(self.hidden, 1)),  # one row per warp
            byte_alignment=128,
        )
        token_back_mbarriers = smem.allocate_array(cutlass.Int64, self.transfer_warp_count)
        warp_mbarrier = token_back_mbarriers + transfer_warp_idx
        if lane_idx == Int32(0):
            cute.arch.mbarrier_init(warp_mbarrier, 1)
        cute.arch.sync_warp()
        push_rows(
            y.iterator.toint(),
            Int32(self.bytes_per_token),
            walk_row,
            walk_dst_address,
            walk_gate_address,
            walk_flag_address,
            flag_target,
            token_back_activation[transfer_warp_idx, None],
            warp_mbarrier,
            global_warp_idx,
            global_warp_count,
            lane_idx,
            self.flag_batch,
            walk_num_rows,
        )


def make_combine_push(y, mode="intra", sm_count=None):
    """Host side: a compiled push of y's rows in `mode` ("intra" / "inter") on sm_count CTAs (default: every SM).
    Returns run; run(walk, flag_target) is the callable for a make_combine_walk walk, flag_target = what an M
    cluster's flag reads once GEMM2 stored all its N tiles (the walk's flags pre-filled to it for a standalone push)."""
    H, dev, dtype = y.shape[1], y.device, y.dtype
    cute_dtype = {torch.bfloat16: cutlass.BFloat16, torch.float16: cutlass.Float16}[dtype]
    props = torch.cuda.get_device_properties(dev)
    if mode == "intra":
        warps = 4  # what the transfer warps look like inside the GEMM
    else:  # "inter": as many rows in flight as the SM's smem holds
        assert mode == "inter", mode
        warps = max(w for w in range(1, 33) if stage_bytes(H, cute_dtype, w) <= props.shared_memory_per_block_optin)
    sm_count = props.multi_processor_count if sm_count is None else sm_count

    kernel = CombinePush(H, cute_dtype, sm_count, warps)
    i32 = lambda: make_fake_tensor(Int32, (cute.sym_int(),), leading_dim=0, divisibility=1)  # walk arrays vary
    i64 = lambda: make_fake_tensor(Int64, (cute.sym_int(),), leading_dim=0, divisibility=1)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    fn = cute.compile(
        kernel, from_dlpack(y, enable_tvm_ffi=True), i32(), i64(), i64(), i64(), i32(), Int32(0), stream,
        options="--enable-tvm-ffi",
    )

    def run(walk, flag_target):
        return lambda: fn(
            y, walk.row, walk.dst_address, walk.gate_address, walk.flag_address, walk.num_rows, flag_target
        )

    return run


# --- Symmetric buffers and the walk, built with torch ------------------------------------------------------------


def make_combine_buffers(T_local, K, H, dtype, group):
    """Per rank: the (T_local, K, H) slot buffer every peer pushes into and the (T_local,) int32 per-token counters
    (K once a token's slots all landed), with every rank's base addresses. Returns a namespace (slots, counters,
    slot_base, counter_base, targets, hdls)."""
    dev = torch.cuda.current_device()
    W = group.size()
    slots = symm_mem.empty(T_local, K, H, dtype=dtype, device=dev)
    counters = symm_mem.empty(T_local, dtype=torch.int32, device=dev)
    slots_hdl, counters_hdl = symm_mem.rendezvous(slots, group=group), symm_mem.rendezvous(counters, group=group)
    slot_base = torch.tensor(
        [slots_hdl.get_buffer(q, (T_local, K, H), dtype).data_ptr() for q in range(W)], dtype=torch.int64, device=dev
    )
    counter_base = torch.tensor(
        [counters_hdl.get_buffer(q, (T_local,), torch.int32).data_ptr() for q in range(W)], dtype=torch.int64, device=dev
    )
    counters.zero_()
    return SimpleNamespace(
        slots=slots, counters=counters, slot_base=slot_base, counter_base=counter_base,
        targets=torch.full((T_local,), K, dtype=torch.int32, device=dev), hdls=(slots_hdl, counters_hdl),
    )


def make_combine_walk(topk_idx_g, s_rev_symm, expert_offset, rank, E_local, buffers, flags, block_rows):
    """The push's walk over this rank's y rows (expert-major, GEMM2's order), the destination rank rotated inside
    each expert (rank+1 first). Returns (walk, pos): walk = (row, dst_address, gate_address, flag_address,
    num_rows) for make_combine_push's run(walk, ...); pos (rows,) int64 = each y row's global slot id (source rank
    * T_local * K + token * K + k), the index of its score in the gathered scores. s_rev_symm = sonic's slot -> y
    row map; expert_offset = GEMM2's cu_seqlens; flags (E_local, max M clusters) int32 = GEMM2's combine flags
    (+1 per stored N tile of cluster (expert, cid_m), cid_m over block_rows-row clusters)."""
    W, T_local, K = topk_idx_g.shape
    dev = s_rev_symm.device
    total_m = int(expert_offset[E_local])
    # Global slot ids whose expert is mine, in slot order; sonic's s_rev maps each to its row of my y.
    slot = (topk_idx_g // E_local == rank).flatten().nonzero().flatten()
    rows = s_rev_symm[slot].long()
    assert rows.numel() == total_m and torch.equal(rows.sort().values, torch.arange(total_m, device=dev))
    src_rank, src_tok, k = slot // (T_local * K), (slot // K) % T_local, slot % K
    row_bytes = buffers.slots.shape[2] * buffers.slots.element_size()
    pos = torch.empty(total_m, dtype=torch.int64, device=dev)
    pos[rows] = slot
    dst_address, gate_address = torch.empty_like(pos), torch.empty_like(pos)
    dst_address[rows] = buffers.slot_base[src_rank] + (src_tok * K + k) * row_bytes
    gate_address[rows] = buffers.counter_base[src_rank] + src_tok * 4
    # Push order: expert-major, destination rank rotated from rank+1 inside each expert, token inside a rank.
    expert = torch.bucketize(rows, expert_offset[1 : E_local + 1].long(), right=True)
    order = torch.argsort(expert * (W * T_local) + ((src_rank - rank - 1) % W) * T_local + src_tok)
    cid_m = (rows - expert_offset[expert].long()) // block_rows
    flag_address = flags.data_ptr() + (expert * flags.shape[1] + cid_m) * 4
    walk = SimpleNamespace(
        row=rows[order].to(torch.int32).contiguous(),
        dst_address=dst_address[rows[order]].contiguous(),
        gate_address=gate_address[rows[order]].contiguous(),
        flag_address=flag_address[order].contiguous(),
        num_rows=torch.tensor([total_m], dtype=torch.int32, device=dev),
        flags=flags,  # the walk holds the tensor its flag addresses point into (a freed one gets reused -> spin)
    )
    return walk, pos


# --- The same walk built on the GPU, in the GEMM's completion order (make_dispatch_walk's counterpart) -------------


class CombineWalk:
    """One launch: a CTA per expert (block_rows threads), persistent over num_ctas CTAs. The expert's M clusters are
    ranked by the work index of their last N tile in the GEMM's own scheduler (its completion order, rotation and
    swizzle included), and their rows are listed in that order, ascending inside a cluster."""

    max_clusters = 512  # M clusters per expert the smem rank keys hold

    def __init__(self, block_rows, ntile_n, num_ctas, even_split=False):
        """ntile_n = the GEMM's cluster tiles along its fixed dim; even_split = the swap-AB GEMM (ragged N blocks
        of block_rows_of(e) rows over tile N = block_rows, the fixed dim M)."""
        self.block_rows, self.ntile_n, self.num_ctas = block_rows, ntile_n, num_ctas
        self.even_split = even_split
        self.vd = 1 if even_split else 0  # the ragged dim in the scheduler mirror
        self.num_warps = block_rows // 32

    @cute.jit
    def __call__(
        self, row_key, cu_seqlens, flags, first_src_row, slot_base, counter_base, walk_row, walk_dst, walk_gate,
        walk_flag, num_rows, ncols: Int32, tokens_per_rank: Int32, topk: Int32, bytes_per_token: Int32, stream,
    ):
        num_experts = cute.size(cu_seqlens) - 1
        sched_params = MClusterWalk.to_underlying_arguments(
            VarlenMTileSchedulerArguments(
                problem_shape_ntile_mnl=(
                    (None, self.ntile_n, num_experts) if not self.even_split else (self.ntile_n, None, num_experts)
                ),
                total_m=cute.size(row_key),
                cu_seqlens_m=cu_seqlens,
                raster_order=RasterOrderOption.AlongN,
                group_size=8,
                tile_shape_mn=(self.block_rows, 512) if not self.even_split else (1, self.block_rows),
                cluster_shape_mnk=(1, 1, 1),
                persistence_mode=PersistenceMode.NONE,
                combine=CombineSchedulerArguments(flags, row_key, first_src_row),
                varlen_dim=self.vd,
            )
        )
        self.kernel(
            sched_params, row_key, cu_seqlens, flags, slot_base, counter_base, walk_row, walk_dst, walk_gate,
            walk_flag, num_rows, ncols, tokens_per_rank, topk, bytes_per_token,
        ).launch(grid=[self.num_ctas, 1, 1], block=[self.block_rows, 1, 1], smem=8192, stream=stream)

    @cute.kernel
    def kernel(
        self, sched_params, row_key, cu_seqlens, flags, slot_base, counter_base, walk_row, walk_dst, walk_gate,
        walk_flag, num_rows, ncols: Int32, tokens_per_rank: Int32, topk: Int32, bytes_per_token: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp, lane = tidx // 32, tidx % 32
        smem = utils.SmemAllocator()
        scratch = smem.allocate_tensor(Int32, cute.make_layout(self.num_warps), byte_alignment=16)
        key = smem.allocate_tensor(Int32, cute.make_layout(self.max_clusters), byte_alignment=16)
        base = smem.allocate_tensor(Int32, cute.make_layout(self.max_clusters), byte_alignment=16)
        num_experts = Int32(cute.size(cu_seqlens) - 1)
        flags_base = flags.iterator.toint()
        sched = MClusterWalk.create(sched_params)
        if bidx == Int32(0) and tidx == Int32(0):
            num_rows[0] = cu_seqlens[num_experts]
        for i in cutlass.range((num_experts - bidx + self.num_ctas - 1) // self.num_ctas, unroll=1):
            e = bidx + i * Int32(self.num_ctas)
            start = cu_seqlens[e]
            n = cu_seqlens[e + 1] - start
            clusters = (n + Int32(self.block_rows - 1)) // Int32(self.block_rows)
            # Work indices before this expert: its predecessors' clusters x N tiles.
            before = Int32(0)
            for j in cutlass.range((e - tidx + self.block_rows - 1) // self.block_rows, unroll=1):
                ep = tidx + j * Int32(self.block_rows)
                before = before + (cu_seqlens[ep + 1] - cu_seqlens[ep] + Int32(self.block_rows - 1)) // Int32(self.block_rows)
            before = block_sum(before, scratch, warp, lane, self.num_warps) * Int32(self.ntile_n)
            # Each cluster's completion key: the work index of its last N tile. The decode's rotation ballots the
            # warp, so every lane decodes on every trip (a clamped index past the end) and only in-range ones write.
            total = clusters * Int32(self.ntile_n)
            b = block_rows_of(cu_seqlens, e, self.block_rows, self.even_split)
            for j in cutlass.range((total + self.block_rows - 1) // self.block_rows, unroll=1):
                w = tidx + j * Int32(self.block_rows)
                w_in = w
                if w_in >= total:
                    w_in = total - Int32(1)
                tile = sched.m_cluster_at(before + w_in)
                if w < total and tile.tile_idx[1 - self.vd] == Int32(self.ntile_n - 1):
                    key[tile.tile_idx[self.vd]] = w
            cute.arch.sync_threads()
            # Entry base of a cluster: the rows of the clusters completing before it.
            rows_before = Int32(0)
            if tidx < clusters:
                for c in cutlass.range(clusters, unroll=1):
                    if key[c] < key[tidx]:
                        size = n - c * b
                        if size > b:
                            size = b
                        rows_before = rows_before + size
                base[tidx] = rows_before
            cute.arch.sync_threads()
            for c in cutlass.range(clusters, unroll=1):
                row_in = c * b + tidx
                if tidx < b and row_in < n:  # b < block_rows threads under even_split: the rest are block c+1's
                    row = start + row_in
                    slot = row_key[row]
                    src_rank = slot // (tokens_per_rank * topk)
                    in_rank = slot - src_rank * (tokens_per_rank * topk)  # token * topk + k
                    entry = start + base[c] + tidx
                    walk_row[entry] = row
                    walk_dst[entry] = slot_base[src_rank] + Int64(in_rank) * Int64(bytes_per_token)
                    walk_gate[entry] = counter_base[src_rank] + Int64(in_rank // topk) * Int64(4)
                    walk_flag[entry] = flags_base + Int64(e * ncols + c) * Int64(4)
            cute.arch.sync_threads()


def make_combine_walk_device(
    row_key, cu_seqlens, flags, first_src_row, buffers, T_local, K, tile_n=512, even_split=False, block_rows=BLOCK_ROWS
):
    """GPU-built walk for make_combine_push's run(walk, ...) and the intra kernel: (build, walk). build() is one
    launch, no sync; walk = (row, dst_address, gate_address, flag_address, num_rows) sized to total_m, holding flags.
    row_key = the GEMM's combine scheduler key (sonic's s_scatter_idx over the D rows), cu_seqlens / flags /
    first_src_row = the GEMM's, buffers = make_combine_buffers'; the clusters follow the GEMM's completion order.
    tile_n = the GEMM's cluster tile along its fixed dim (N; under even_split, the swap-AB GEMM's M tile); block_rows =
    the gated block (the GEMM's M cluster; under even_split its tile N, the even-split unit)."""
    dev = row_key.device
    total_m, H = row_key.numel(), buffers.slots.shape[2]
    row_bytes = H * buffers.slots.element_size()
    assert flags.shape[1] <= CombineWalk.max_clusters and flags.shape[0] == cu_seqlens.numel() - 1
    i64 = lambda: torch.empty(total_m, dtype=torch.int64, device=dev)
    walk = SimpleNamespace(
        row=torch.empty(total_m, dtype=torch.int32, device=dev), dst_address=i64(), gate_address=i64(),
        flag_address=i64(), num_rows=torch.empty(1, dtype=torch.int32, device=dev), flags=flags,
    )
    kernel = CombineWalk(
        block_rows, -(-H // tile_n), min(flags.shape[0], torch.cuda.get_device_properties(dev).multi_processor_count),
        even_split,
    )
    v = lambda: make_fake_tensor(Int32, (cute.sym_int(),), leading_dim=0, divisibility=1)
    v64 = lambda: make_fake_tensor(Int64, (cute.sym_int(),), leading_dim=0, divisibility=1)
    m = lambda: make_fake_tensor(Int32, (cute.sym_int(), cute.sym_int()), leading_dim=1, divisibility=1)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    fn = cute.compile(
        kernel, v(), v(), m(), v(), v64(), v64(), v(), v64(), v64(), v64(), v(),
        Int32(0), Int32(0), Int32(0), Int32(0), stream, options="--enable-tvm-ffi",
    )
    build = lambda: fn(
        row_key, cu_seqlens, flags, first_src_row, buffers.slot_base, buffers.counter_base, walk.row,
        walk.dst_address, walk.gate_address, walk.flag_address, walk.num_rows,
        flags.shape[1], T_local, K, row_bytes,
    )
    return build, walk
