# Dispatch transport leg: MegaMoE's deterministic `token_in` transfer-warp loop, rows moved as bytes (any dtype;
# block-scaled formats also scatter each row's scales, see make_dispatch_pull). Pulls rows from peers' x buffers into
# the receiver's pool with cp.async.bulk via a per-warp smem stage; once a row's bytes have landed, its gate counters
# get a release-add of 1 each. pull_rows is shared by the standalone kernel (DispatchPull: one CTA per SM, one row in
# flight per warp) and the fused GEMM (fused_gemm_dispatch/gemm_sm100_intra_dispatch_pull.py). Two standalone modes:
# "intra" = 4 warps / 56 KB stage, what the transfer warps look like inside the GEMM; "inter" = as many warps as the
# SM's smem holds rows (16 on B300), the inter-SM leg with the most bytes in flight per SM.
#
# The walk (walk.py) is a list of rows in pull order plus a CSR of gate ids per row, in the GEMM's block order; each
# entry carries its source row's absolute address, so the pull reads no routing metadata. Deviations from MegaMoE:
# source addresses from the walk (torch symm-mem has no single heap offset), walk from gmem, and a full wait_group
# (no .read) before each publication batch so the counters follow the landed bytes.
#
# Signals are batched, not per row: a red.release carries a gpu-scope fence that waits on the previous add, so
# one per gate serialized on the warp. Measured at W=2, 16 SMs, full mode, expert-src_rank order (~4 gates per row):
# 157 GB/s per gate vs 371 with no signals; parking gates in lanes and firing 8 rows' worth in one instruction
# gives 351, a publish lag of 8 rows (results/2026-09-21/dispatch_pull_walks.md).
import sys
from pathlib import Path

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import AddressSpace
from cutlass.cutlass_dsl import Int32, Int64
from quack.blockscaled.operand import BlockScaledFormat
from quack.compile_utils import make_fake_tensor

from dispatch.walk import WalkArguments, fake_walk_arguments

from quack.ep.ptx import cp_async_bulk_s2g, red_add_release_gpu_s32, tma_load_1d  # same PTX as the megamoe helpers


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
    walk: WalkArguments,
    gate_counters: cute.Tensor,  # (n_gates,) int32, zeroed before launch
    stage: cute.Tensor,  # this warp's smem stage: the row, or a chunk of it
    mbarrier: cute.Pointer,  # this warp's smem mbarrier, initialized with arrive count 1
    global_warp_idx: Int32,
    global_warp_count: Int32,
    lane_idx: Int32,
    flag_batch: cutlass.Constexpr[int],
    scatter_all: cutlass.Constexpr[bool] = False,  # no-gather layout: the row lands in every reader's row (dst_row)
    # block-scaled rows: the token's 1-byte scales (sf_words x 4, K-major) land in each reader block's SF atoms
    # (the GEMM's per-block SFB: VarlenManager.offset_block_SFB) at row walk.dst_sf_row, i.e. byte
    # ((row // 128) * sf_words + word) * 512 + (row % 32) * 16 + ((row // 32) % 4) * 4; sf_words = 0: no scales
    sf_words: cutlass.Constexpr[int] = 0,
    sf_pool_base: Int64 = Int64(0),  # the SF atoms' address
):
    """One transfer warp's share of the walk (entries global_warp_idx, += global_warp_count): MegaMoE's token_in
    loop with lane-distributed publication; the scale row (sf_words > 0) goes out as plain 4-byte stores by the lanes
    before the row's gates are parked."""
    activation_bytes = Int32(cute.cosize(stage) * int(stage.element_type.width) // 8)
    n_rows = walk.num_rows[0]
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
        pool_token_idx = walk.pool_row[entry]
        remote_activation_address = walk.src_address[entry]

        gate_begin = walk.gate_ptr[entry]
        gate_count = walk.gate_ptr[entry + Int32(1)] - gate_begin
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
            # one bulk store per destination row (the pool row, or every reader's row), all in one bulk group
            store_count = Int32(1)
            if cutlass.const_expr(scatter_all):
                store_count = gate_count
            for j in cutlass.range(store_count, unroll=1):
                destination_row = pool_token_idx
                if cutlass.const_expr(scatter_all):
                    destination_row = walk.dst_row[gate_begin + j]
                destination_activation = cute.make_ptr(
                    stage.element_type,
                    pool_base + Int64(destination_row) * Int64(bytes_per_token) + chunk_offset,
                    AddressSpace.gmem,
                    assumed_align=16,
                )
                with cute.arch.elect_one():
                    cp_async_bulk_s2g(destination_activation, stage.iterator, activation_bytes)
                cute.arch.sync_warp()
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(0, read=True)  # stage free for the next chunk; landing is waited at flush
            cute.arch.sync_warp()
        if cutlass.const_expr(sf_words > 0):  # the scale row: one 4-byte word per lane per round, into each reader's atoms
            src_sf = cute.make_tensor(
                cute.make_ptr(Int32, walk.src_sf_address[entry], AddressSpace.gmem, assumed_align=4),
                cute.make_layout(sf_words),
            )
            for w in cutlass.range_constexpr((sf_words + 31) // 32):
                word = Int32(w * 32) + lane_idx
                if word < Int32(sf_words):
                    val = src_sf[word]
                    for j in cutlass.range(gate_count, unroll=1):
                        sf_row = walk.dst_sf_row[gate_begin + j]
                        offset = ((sf_row // 128) * Int32(sf_words) + word) * Int32(512) + (sf_row % 32) * Int32(16) + (
                            (sf_row // 32) % 4
                        ) * Int32(4)
                        dst_sf = cute.make_tensor(
                            cute.make_ptr(Int32, sf_pool_base + Int64(offset), AddressSpace.gmem, assumed_align=4),
                            cute.make_layout(1),
                        )
                        dst_sf[0] = val
            cute.arch.sync_warp()  # the lanes' scale stores precede the parked lanes' release adds
        if pending + gate_count > Int32(32):  # no lanes left: flush what is parked first
            cute.arch.cp_async_bulk_wait_group(0, read=False)
            if lane_idx < pending:
                red_add_release_gpu_s32(gate_counters.iterator + lane_gate, Int32(1))
            pending = Int32(0)
            rows_parked = Int32(0)
        if (lane_idx >= pending) & (lane_idx < pending + gate_count):
            lane_gate = walk.gates[gate_begin + (lane_idx - pending)]
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
    """The standalone pull over rows of row_bytes each: bytes, so one kernel serves every dtype and format."""

    flag_batch = 8  # rows per publication batch (MegaMoE token_in_flag_batch)

    def __init__(self, row_bytes, sm_count, transfer_warps, scatter=False, sf_words=0):
        self.row_bytes = row_bytes
        self.sm_count = sm_count
        self.transfer_warp_count = transfer_warps
        self.scatter = scatter  # rows land in every reader's expert-major row (walk.dst_row), not the pool row
        self.sf_words = sf_words  # block-scaled rows: 4-byte scale words per token (scales / 4); 0 = no scales
        # Request more than half the SM's smem so only one CTA fits: the fused kernel's occupancy.
        self.smem_bytes = max(stage_bytes(row_bytes, cutlass.Uint8, transfer_warps), 116 * 1024)

    @cute.jit
    def __call__(self, pool, walk: WalkArguments, gate_counters, stream, sf_pool=None):
        self.kernel(pool, walk, gate_counters, sf_pool).launch(
            grid=[self.sm_count, 1, 1],
            block=[self.transfer_warp_count * 32, 1, 1],
            smem=self.smem_bytes,
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        pool: cute.Tensor,  # (pool rows, row_bytes) uint8 destination
        walk: WalkArguments,
        gate_counters: cute.Tensor,  # (n_gates,) int32, zeroed before launch
        sf_pool: cute.Tensor,  # the SF atoms (sf_words > 0), uint8
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        transfer_warp_idx = tidx // 32
        lane_idx = tidx % 32
        global_warp_idx = bidx * Int32(self.transfer_warp_count) + transfer_warp_idx
        global_warp_count = Int32(self.sm_count * self.transfer_warp_count)

        smem = utils.SmemAllocator()
        token_in_activation = smem.allocate_tensor(
            cutlass.Uint8,
            cute.make_layout((self.transfer_warp_count, self.row_bytes), stride=(self.row_bytes, 1)),  # one row per warp
            byte_alignment=128,
        )
        token_in_mbarriers = smem.allocate_array(cutlass.Int64, self.transfer_warp_count)
        warp_mbarrier = token_in_mbarriers + transfer_warp_idx
        if lane_idx == Int32(0):
            cute.arch.mbarrier_init(warp_mbarrier, 1)
        cute.arch.sync_warp()
        sf_pool_base = Int64(0)
        if cutlass.const_expr(self.sf_words > 0):
            sf_pool_base = sf_pool.iterator.toint()
        pull_rows(
            pool.iterator.toint(),
            Int32(self.row_bytes),
            walk,
            gate_counters,
            token_in_activation[transfer_warp_idx, None],
            warp_mbarrier,
            global_warp_idx,
            global_warp_count,
            lane_idx,
            self.flag_batch,
            self.scatter,
            self.sf_words,
            sf_pool_base,
        )


def make_dispatch_pull(
    x_peer_bufs, rank, mode="intra", sm_count=None, pool=None, scatter=False, a_format=None, sf_pool=None
):
    """Host side: the pool (every rank's tokens fit: sonic's dedup receive layout) and a compiled pull in `mode`
    ("intra" / "inter") on sm_count CTAs (default: every SM). Returns (run, pool); run(walk) is the callable for a
    make_dispatch_walk walk: zero its counters (unless zero_counters=False: a caller holding them at target), pull.
    Rows move as bytes, so x may be any dtype. scatter: rows land in every reader's expert-major row (pool = the
    expert-packed rows). a_format (a quack BlockScaledFormat or its name, as BlockScaledAllGatherRunner): x holds the
    format's qdata rows (storage K wide) and the walk was built with x_sf_peer_bufs; the pull also scatters each
    token's 1-byte scales into sf_pool, the GEMM's per-block SFB atoms (block slots * sf_atom_rows, K / (4 vec),
    32, 4, 4), one row per reader."""
    W, T_local = len(x_peer_bufs), x_peer_bufs[rank].shape[0]
    x = x_peer_bufs[rank]
    row_bytes = x.shape[1] * x.element_size()
    dev = x.device
    props = torch.cuda.get_device_properties(dev)
    if mode == "intra":
        warps = 4  # what the transfer warps look like inside the GEMM
    else:  # "inter": as many rows in flight as the SM's smem holds
        assert mode == "inter", mode
        warps = max(
            w for w in range(1, 33) if stage_bytes(row_bytes, cutlass.Uint8, w) <= props.shared_memory_per_block_optin
        )
    sm_count = props.multi_processor_count if sm_count is None else sm_count
    pool = torch.empty(W * T_local, x.shape[1], dtype=x.dtype, device=dev) if pool is None else pool
    sf_words = 0
    if a_format is not None:
        fmt = a_format if isinstance(a_format, BlockScaledFormat) else BlockScaledFormat.from_name(a_format)
        assert x.dtype == fmt.qdata_dtype, f"x rows must be {fmt.name} qdata ({fmt.qdata_dtype}), got {x.dtype}"
        assert fmt.scale_dtype.itemsize == 1, f"{fmt.name}: the scale scatter moves 1-byte scales"
        k = fmt.logical_k(x.shape[1])
        assert k % (4 * fmt.sf_vec_size) == 0, f"{fmt.name}: K={k} must fill whole 4-scale SF words"
        sf_words = k // (4 * fmt.sf_vec_size)  # scale words per row = SF atom columns
        assert sf_pool is not None and tuple(sf_pool.shape[-4:]) == (sf_words, 32, 4, 4), (
            f"sf_pool must be blocked (slots * atoms, {sf_words}, 32, 4, 4), got {None if sf_pool is None else tuple(sf_pool.shape)}"
        )
    else:
        assert sf_pool is None, "sf_pool needs a_format"
    # the kernel addresses bytes: the pool and the atoms go in as uint8 views (fp4x2 containers included)
    pool_u8 = pool.view(torch.uint8)
    sf_u8 = sf_pool.view(torch.uint8) if sf_pool is not None else None

    kernel = DispatchPull(row_bytes, sm_count, warps, scatter, sf_words)
    i32 = lambda: make_fake_tensor(Int32, (cute.sym_int(),), leading_dim=0, divisibility=1)  # the flags vary
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    sf_fake = from_dlpack(sf_u8, enable_tvm_ffi=True) if sf_pool is not None else None
    fn = cute.compile(
        kernel, from_dlpack(pool_u8, enable_tvm_ffi=True), fake_walk_arguments(sf_words > 0), i32(), stream, sf_fake,
        options="--enable-tvm-ffi",
    )

    def run(walk, zero_counters=True):
        def go():
            if zero_counters:
                walk.counters.zero_()
            fn(pool_u8, walk.args, walk.counters, sf_u8)

        return go

    return run, pool
