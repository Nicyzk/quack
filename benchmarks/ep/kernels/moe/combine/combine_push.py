# Combine transport leg: the dispatch pull (dispatch/pull.py) run the other way. One CTA per SM, one row
# in flight per warp, bf16 rows (scores folded in beforehand): each transfer warp walks its share of this rank's
# expert-major y rows, stages a row in smem with cp.async.bulk and pushes it to slot (src_token, src_topk) of the
# origin rank's (T_local, K, H) slot buffer; once a batch of rows has landed, each row's origin gets one
# release-add on its per-token counter (sys scope: the consumer is a peer), lane-distributed as in pull_rows, so
# the origin can reduce a token once its counter reads K. Two modes as the pull: "intra" = 4 warps / 56 KB stage
# (the transfer warps inside a GEMM), "inter" = as many warps as the SM's smem holds rows (16 on B300). Inside the
# GEMM the row goes out in chunk_bytes pieces, each waited to completion before the next: a bulk store to a peer
# holds the SM's TMA path for its NVLink round trip in front of the GEMM's operand loads, so what is outstanding
# per warp must stay small (results/2026-09-28/intra_tma_link_serialization.md: 2 KB recovers 2/3 of the loss).
#
# The walk (make_combine_walk, end of file) is built with torch: y rows in GEMM2's order (expert-major) with the
# destination rank rotated inside each expert (rank+1 first, as the pull's source order: on a small SM partition
# the warps in flight cover one destination's run of rows, and every rank pushing to the same peer at once is the
# packed-by-destination case measured 2.6x slower); each entry carries its y row, its peer slot address and its
# peer counter address. Deviations from MegaMoE's token_back: walk from gmem with absolute peer addresses, no
# fc2_done wait (standalone rows are ready; the fused GEMM adds the flag wait), no pacing, no scale factors.
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.experimental import iket
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import AddressSpace
from cutlass.cutlass_dsl import Int32, Int64
from quack.compile_utils import make_fake_tensor
try:  # quack ep's combine rotation bundle (the push walk builder); absent on main / `all`
    from quack.tile_scheduler import CombineSchedulerArguments
except ImportError:
    CombineSchedulerArguments = None
from quack.tile_scheduler import (
    PersistenceMode,
    RasterOrderOption,
    VarlenMTileSchedulerArguments,
)
from torch.distributed import _symmetric_memory as symm_mem

from quack.ep.ptx import cp_async_bulk_s2g, red_add_release_sys_s32, tma_load_1d  # same PTX as the megamoe helpers

from dispatch.pull import stage_bytes  # noqa: E402
from dispatch.walk import BLOCK_ROWS, MClusterWalk, block_rows_of, block_sum  # noqa: E402


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

    max_clusters = 1024  # M clusters per expert the smem rank keys hold (W=8 x 16384 tokens / 192 = 683)

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
        ).launch(grid=[self.num_ctas, 1, 1], block=[self.block_rows, 1, 1], smem=2 * 4 * self.max_clusters + 256, stream=stream)

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
