# MoE dispatch as the GEMM's comm object (gemm_base.Comm). DispatchGate = the gate alone: wait_tile spins until
# flags[expert, block] reaches the block's row count before the tile's gathered loads (the rows land by another
# kernel). DispatchPull adds transfer warps per CTA running pull_rows over the device walk (dispatch_walk.py),
# landing each row once in the pool the GEMM's gather-A reads and bumping flags[expert, block]. The scheduler is the
# GEMM's plain expert-grouped order: nothing rides the scheduler args. The runner (DispatchRunner) owns the pool,
# the flags and the walk; the GEMM's D is the caller's output (wire = None).
#
# pull_rows is MegaMoE's deterministic `token_in` transfer-warp loop, rows moved as bytes (any dtype; block-scaled
# formats also scatter each row's scales): a row is pulled from its peer's x buffer into a per-warp smem stage with
# cp.async.bulk and stored to the pool row; once its bytes have landed, each of its gate counters gets a release add
# of 1. The walk is a list of rows in pull order plus a CSR of gate ids per row, in the GEMM's block order; each
# entry carries its source row's absolute address, so the pull reads no routing metadata. Signals are batched: a
# red.release carries a gpu-scope fence, so lanes park the pending gates and fire them in one instruction.
# Forked from duckpond kernels/moe/dispatch/pull.py and the tmp operator (2026-10-22) so quack stands alone.
from typing import NamedTuple, Optional

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, const_expr
from cutlass.cute.typing import AddressSpace
from quack.blockscaled.operand import BlockScaledFormat
from quack.compile_utils import make_fake_tensor
from quack.cute_dsl_utils import mlir_namedtuple
from quack.distributed.dispatch_walk import WalkArguments, fake_walk_arguments, make_dispatch_walk
from quack.distributed.ptx import cp_async_bulk_s2g, red_add_release_gpu_s32, tma_load_1d
from quack.gemm_base import Comm, CommRunner

TRANSFER_WARPS = 4


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


# --- the operators ---------------------------------------------------------------------------------------------------


@mlir_namedtuple
class GateArguments(NamedTuple):
    """DispatchGate's bundle (the GEMM's comm_args)."""

    flags: cute.Tensor  # (E_local, max M blocks) int32, +1 per landed row: the gate's counters


@mlir_namedtuple
class DispatchArguments(NamedTuple):
    """The transfer warps' bundle (the GEMM's comm_args)."""

    walk: WalkArguments  # make_dispatch_walk's walk.args
    pool: cute.Tensor  # the rows the pull writes: the gather-A pool (= mA) or the expert-packed rows (scatter)
    flags: cute.Tensor  # (E_local, max M blocks) int32, +1 per landed row: published here, gated on by wait_tile
    sf_pool: Optional[cute.Tensor] = None  # block-scaled: the SF atoms the pulled scales land in (the GEMM's SFB)


def _fake_flags():
    return make_fake_tensor(Int32, (cute.sym_int(), cute.sym_int()), leading_dim=1, divisibility=1)


def fake_dispatch_arguments(pool_dtype, sf=False):
    """DispatchArguments of fake tensors for cute.compile."""
    pool = make_fake_tensor(pool_dtype, (cute.sym_int(), cute.sym_int()), leading_dim=1, divisibility=8)
    sf_pool = make_fake_tensor(cutlass.Uint8, (cute.sym_int(),), leading_dim=0, divisibility=16) if sf else None
    return DispatchArguments(fake_walk_arguments(sf), pool, _fake_flags(), sf_pool)


class DispatchGate(Comm):
    """The dispatch gate alone (no comm warps): before a tile's gathered loads, spin until the tile's M block has
    all its rows (flags[expert, block] == rows in the block). The rows land by any transport."""

    def fake_args(self, d_dtype):
        return GateArguments(_fake_flags())

    @cute.jit
    def wait_tile(self, gemm, ctx, tile_coord_mnkl, gate: Int32) -> Int32:
        pid_m, batch = tile_coord_mnkl[0], tile_coord_mnkl[3]
        cluster_m = gemm.cluster_shape_mnk[0]
        block_rows = Int32(gemm.cta_tile_shape_mnk[0] * cluster_m)
        cid_m = pid_m // cluster_m
        cu = ctx.sched_params.cu_seqlens_m
        num_rows = cutlass.min(cu[batch + 1] - cu[batch] - cid_m * block_rows, block_rows)
        flags = ctx.args.flags
        ptr = flags.iterator + cute.crd2idx((batch, cid_m), flags.layout)
        # relaxed sys loads, as the AG gate: the gated rows are read by TMA at L2, never through L1
        val = cute.arch.load(ptr.llvm_ptr, Int32, sem="relaxed", scope="sys")
        while val < num_rows:
            val = cute.arch.load(ptr.llvm_ptr, Int32, sem="relaxed", scope="sys")
        return gate


class DispatchPull(DispatchGate):
    """The dispatch pull as the GEMM's comm warps, gated as DispatchGate. pull_bytes = one row's bytes; scatter =
    the row lands in every reader's expert-major row (walk.dst_row) instead of the pool row; sf_words = 4-byte
    scale words per row for block-scaled rows (K / (4 sf_vec_size)), 0 = none."""

    num_warps = TRANSFER_WARPS
    regs = (232, 120, 40)  # 4 warp groups share 512: epilogue 232, the GEMM's others 120 each, the transfer loop 40
    flag_batch = 1  # publish each row as it lands: ~6 rows per warp, batching only delays every block's readiness

    def __init__(self, pull_bytes: int, scatter: bool = False, sf_words: int = 0):
        self.pull_bytes = pull_bytes
        self.pull_chunk = pull_chunk_bytes(pull_bytes, cutlass.Uint8)  # a chunk of the row per stage
        self.scatter = scatter
        self.sf_words = sf_words

    def fake_args(self, d_dtype):
        return fake_dispatch_arguments(d_dtype, sf=self.sf_words > 0)

    def smem_bytes(self, gemm):
        return stage_bytes(self.pull_chunk, cutlass.Uint8, TRANSFER_WARPS)

    def smem_struct(self, gemm):
        class PullStorage:
            stage: cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, TRANSFER_WARPS * self.pull_chunk], 128]
            mbar: cute.struct.Align[cute.struct.MemRange[cutlass.Int64, TRANSFER_WARPS], 8]

        return cute.struct(PullStorage)

    @cute.jit
    def comm_warps(self, gemm, ctx, warp_idx: Int32, tidx: Int32):
        """The transfer warps: this CTA's share of the walk, rows into the pool, flags bumped as rows land."""
        chunk = self.pull_chunk
        comm_args, smem = ctx.args, ctx.smem
        stage = smem.stage.get_tensor(cute.make_layout((TRANSFER_WARPS, chunk), stride=(chunk, 1)))
        transfer_warp = warp_idx - gemm.comm_warp_ids[0]
        mbar = smem.mbar.data_ptr() + transfer_warp
        if tidx % 32 == Int32(0):
            cute.arch.mbarrier_init(mbar, 1)
        cute.arch.sync_warp()
        bid, gd = cute.arch.block_idx(), cute.arch.grid_dim()
        cta = bid[0] + gd[0] * (bid[1] + gd[1] * bid[2])
        num_ctas = gd[0] * gd[1] * gd[2]
        sf_pool_base = Int64(0)
        if const_expr(self.sf_words > 0):
            sf_pool_base = comm_args.sf_pool.iterator.toint()
        pull_rows(
            comm_args.pool.iterator.toint(),
            Int32(self.pull_bytes),
            comm_args.walk,
            comm_args.flags,
            stage[transfer_warp, None],
            mbar,
            cta * Int32(TRANSFER_WARPS) + transfer_warp,
            num_ctas * Int32(TRANSFER_WARPS),
            tidx % 32,
            self.flag_batch,
            self.scatter,
            self.sf_words,
            sf_pool_base,
        )


# --- host ------------------------------------------------------------------------------------------------------------


def sf_words_of(a_format, x):
    """4-byte scale words per row of a block-scaled x (K / (4 sf_vec_size)); 0 for a plain dtype."""
    if a_format is None:
        return 0
    fmt = a_format if isinstance(a_format, BlockScaledFormat) else BlockScaledFormat.from_name(a_format)
    assert x.dtype == fmt.qdata_dtype, f"x rows must be {fmt.name} qdata ({fmt.qdata_dtype}), got {x.dtype}"
    assert fmt.scale_dtype.itemsize == 1, f"{fmt.name}: the scale scatter moves 1-byte scales"
    k = fmt.logical_k(x.shape[1])
    assert k % (4 * fmt.sf_vec_size) == 0, f"{fmt.name}: K={k} must fill whole 4-scale SF words"
    return k // (4 * fmt.sf_vec_size)


class DispatchRunner(CommRunner):
    """GEMM1 + dispatch pull: owns the pool the GEMM's gather-A reads (pass runner.pool as A), the flags and the
    device walk (rebuilt every launch from the routing); the GEMM's D is the caller's output (wire = None).
    topk_idx_g (W, T_local, K) int32, s_scatter / s_reverse / a_idx / cu_seqlens int32 as make_dispatch_walk;
    x_peer_bufs = every rank's x buffer (rows of one dtype or a block-scaled format's qdata); block_rows = the
    GEMM's M block (tile M x cluster M). scatter: rows land in every reader's expert-major row. a_format +
    x_sf_peer_bufs + sf_pool: the block-scaled scale lane (sf_pool = the GEMM's per-block SF atoms)."""

    wire = None  # D stays the caller's

    def __init__(
        self, topk_idx_g, s_scatter, s_reverse, rank, a_idx, cu_seqlens, x_peer_bufs, block_rows, scatter=False,
        a_format=None, x_sf_peer_bufs=None, sf_pool=None,
    ):
        x = x_peer_bufs[rank]
        W, T_local, _ = topk_idx_g.shape
        self.comm = DispatchPull(x.shape[1] * x.element_size(), scatter, sf_words_of(a_format, x))
        self.build, self.walk = make_dispatch_walk(
            topk_idx_g, s_scatter, s_reverse, rank, a_idx, cu_seqlens, x_peer_bufs, block_rows,
            x_sf_peer_bufs=x_sf_peer_bufs,
        )
        self.pool = torch.empty(W * T_local, x.shape[1], dtype=x.dtype, device=x.device)
        self.sf_pool = sf_pool
        self.out = None

    def args(self):
        return DispatchArguments(self.walk.args, self.pool, self.walk.flags, self.sf_pool)

    def before_launch(self):
        self.build()
        self.walk.flags.zero_()
