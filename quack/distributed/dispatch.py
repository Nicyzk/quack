# MoE dispatch as the GEMM's distributed operator (gemm_base.DistOp) on a grouped GEMM whose token operand is packed
# per expert along the ragged dim: swap-AB (varlen_n: tokens are the B rows in even-split N blocks) or varlen_m
# (tokens are the A rows in fixed tile_M blocks). DispatchPull: transfer warps per CTA run pull_rows
# over the device walk (dispatch_walk.py), landing each pulled row in every reader expert's row of the token operand
# and bumping flags[expert, block]; wait_tile spins until flags[expert, block] reaches the block's row count before
# the tile's loads. The GEMM reads the operand as a plain contiguous tensor: no index, every GEMM feature composes.
# The scheduler is the
# GEMM's expert-grouped order on the operator's tile counter (make_scheduler_args: the dynamic scheduler over the
# resident grid, CLC off). DispatchRunner owns the symmetric token rows and the routing's buffers; its prepare is
# the one launch that exchanges the top-k, routes and builds the walk (flags and counter reset with it); the GEMM's
# D is the caller's output.
#
# pull_rows is MegaMoE's deterministic `token_in` transfer-warp loop, rows moved as bytes (any dtype; block-scaled
# formats also scatter each row's scales): a row is pulled from its peer's x buffer into a per-warp smem stage with
# cp.async.bulk and stored to each reader's row; once its bytes have landed, each of its gate counters gets a release
# add of 1. The walk is a list of rows in pull order plus a CSR of (gate, destination row) per row, in the GEMM's
# block order; each entry carries its source row's absolute address, so the pull reads no routing metadata. Signals
# are batched: a red.release carries a gpu-scope fence, so lanes park the pending gates and fire them in one
# instruction.
# Forked from duckpond kernels/moe/dispatch/pull.py and the tmp operator (2026-10-22) so quack stands alone.
from typing import NamedTuple, Optional

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, const_expr
from cutlass.cute.typing import AddressSpace
from quack.blockscaled.operand import BlockScaledFormat
from quack.compile_utils import make_fake_tensor
from quack.cute_dsl_utils import get_device_capacity, mlir_namedtuple
from quack.distributed.dispatch_walk import (
    GROUP_THREADS,
    DispatchWalk,
    WalkArguments,
    fake_walk_arguments,
    sf_atom_rows_of,
)
from quack.distributed.ptx import cp_async_bulk_s2g, red_add_release_gpu_s32, tma_load_1d
from quack.gemm_base import DistOp, grouped_ragged_dim
from quack.gemm_tvm_ffi_utils import varlen_n_block_atoms
from quack.varlen_utils import varlen_n_block_rows


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
    rows_base: Int64,  # address of B row 0; rows are bytes_per_token apart
    bytes_per_token: Int32,
    walk: WalkArguments,
    gate_counters: cute.Tensor,  # (n_gates,) int32, zeroed before launch
    stage: cute.Tensor,  # this warp's smem stage: the row, or a chunk of it
    mbarrier: cute.Pointer,  # this warp's smem mbarrier, initialized with arrive count 1
    global_warp_idx: Int32,
    global_warp_count: Int32,
    lane_idx: Int32,
    flag_batch: cutlass.Constexpr[int],
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
            # one bulk store per reader's row, all in one bulk group
            for j in cutlass.range(gate_count, unroll=1):
                destination_row = walk.dst_row[gate_begin + j]
                destination_activation = cute.make_ptr(
                    stage.element_type,
                    rows_base + Int64(destination_row) * Int64(bytes_per_token) + chunk_offset,
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
class DispatchArguments(NamedTuple):
    """The transfer warps' bundle (the GEMM's dist_op_args)."""

    walk: WalkArguments  # DispatchWalk's output
    rows: cute.Tensor  # the token operand as bytes: every reader expert's row of each pulled token
    flags: cute.Tensor  # (E_local, max blocks) int32, +1 per landed row: published here, gated on by wait_tile
    sem: cute.Tensor  # (1,) int32 the GEMM's tile counter (make_scheduler_args), zeroed per call
    sf_pool: Optional[cute.Tensor] = None  # block-scaled: the SF atoms the pulled scales land in (the GEMM's SFB)


def _fake_flags():
    return make_fake_tensor(Int32, (cute.sym_int(), cute.sym_int()), leading_dim=1, divisibility=1)


def fake_dispatch_arguments(sf=False):
    """DispatchArguments of fake tensors for cute.compile."""
    rows = make_fake_tensor(cutlass.Uint8, (cute.sym_int(), cute.sym_int()), leading_dim=1, divisibility=16)
    sem = make_fake_tensor(Int32, (1,), leading_dim=0, divisibility=1)
    sf_pool = make_fake_tensor(cutlass.Uint8, (cute.sym_int(),), leading_dim=0, divisibility=16) if sf else None
    return DispatchArguments(fake_walk_arguments(sf), rows, _fake_flags(), sem, sf_pool)


class DispatchPull(DistOp):
    """The dispatch pull as the GEMM's comm warps, gated per tile on the landed row counts. pull_bytes = one row's
    bytes; sf_words = 4-byte scale words per row for block-scaled rows (K / (4 sf_vec_size)), 0 = none; ragged_dim =
    the ragged dim the walk is built for, which the call must match: "n" = swap-AB, tokens are the B rows in
    even-split N blocks (cu_seqlens_n); "m" = varlen_m, tokens are the A rows in fixed tile_M blocks (cu_seqlens_m)."""

    num_warps = 4
    regs = (232, 120, 40)  # 4 warp groups share 512: epilogue 232, the GEMM's others 120 each, the transfer loop 40
    flag_batch = 1  # publish each row as it lands: ~6 rows per warp, batching only delays every block's readiness

    def __init__(self, pull_bytes: int, sf_words: int = 0, ragged_dim: str = "n"):
        assert ragged_dim in ("m", "n")
        assert sf_words == 0 or ragged_dim == "n", "block-scaled dispatch scatters scales per even-split block"
        self.ragged_dim = ragged_dim
        self.pull_bytes = pull_bytes
        self.pull_chunk = pull_chunk_bytes(pull_bytes, cutlass.Uint8)  # a chunk of the row per stage
        self.sf_words = sf_words

    def fake_args(self, d_dtype):
        return fake_dispatch_arguments(sf=self.sf_words > 0)

    def make_scheduler_args(self, dist_op_args):
        """The tile counter: CLC's arrival order over the resident grid, which CLC's oversized grid would cancel
        the transfer warps of (static loses the arrival order's L2 reuse: results/2026-10-05/ablations.md)."""
        return {"tile_count_semaphore": dist_op_args.sem}

    @cute.jit
    def wait_tile(self, gemm, ctx, tile_coord_mnkl, gate: Int32) -> Int32:
        """Before a tile's gathered loads: spin until its block has all its rows (flags[expert, block])."""
        batch = tile_coord_mnkl[3]
        cu = ctx.sched_params.cu_seqlens_m  # the ragged dim's cu_seqlens
        if const_expr(gemm.varlen_n):  # even-split N blocks
            cid = tile_coord_mnkl[1]
            block_rows = varlen_n_block_rows(
                cu[batch + 1] - cu[batch], gemm.cta_tile_shape_mnk[1], gemm.varlen_n_granule
            )
        else:  # fixed tile_M blocks
            cid = tile_coord_mnkl[0]
            block_rows = Int32(gemm.cta_tile_shape_mnk[0])
        num_rows = cutlass.min(cu[batch + 1] - cu[batch] - cid * block_rows, block_rows)
        flags = ctx.args.flags
        ptr = flags.iterator + cute.crd2idx((batch, cid), flags.layout)
        # relaxed sys loads, as the AG gate: the gated rows are read by TMA at L2, never through L1
        val = cute.arch.load(ptr.llvm_ptr, Int32, sem="relaxed", scope="sys")
        while val < num_rows:
            val = cute.arch.load(ptr.llvm_ptr, Int32, sem="relaxed", scope="sys")
        return gate

    def smem_bytes(self, gemm):
        return stage_bytes(self.pull_chunk, cutlass.Uint8, self.num_warps)

    def smem_struct(self, gemm):
        class PullStorage:
            stage: cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, self.num_warps * self.pull_chunk], 128]
            mbar: cute.struct.Align[cute.struct.MemRange[cutlass.Int64, self.num_warps], 8]

        return cute.struct(PullStorage)

    @cute.jit
    def comm_warps(self, gemm, ctx, warp_idx: Int32, tidx: Int32):
        """The transfer warps: this CTA's share of the walk, rows into B, flags bumped as rows land."""
        chunk = self.pull_chunk
        dist_op_args, smem = ctx.args, ctx.smem
        stage = smem.stage.get_tensor(cute.make_layout((self.num_warps, chunk), stride=(chunk, 1)))
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
            sf_pool_base = dist_op_args.sf_pool.iterator.toint()
        pull_rows(
            dist_op_args.rows.iterator.toint(),
            Int32(self.pull_bytes),
            dist_op_args.walk,
            dist_op_args.flags,
            stage[transfer_warp, None],
            mbar,
            cta * Int32(self.num_warps) + transfer_warp,
            num_ctas * Int32(self.num_warps),
            tidx % 32,
            self.flag_batch,
            self.sf_words,
            sf_pool_base,
        )


# --- host ------------------------------------------------------------------------------------------------------------


class DispatchRunner:
    """GEMM1 + dispatch pull of tokens_local tokens x k experts x hidden. `x` = this rank's (tokens_local, hidden)
    token rows in symmetric memory, the buffer every rank's pulls read; `rows` = the token operand the pull fills,
    (max_rows, hidden) packed per local expert: the GEMM's B with cu_seqlens_n=cu under "n" (swap-AB), its A with
    cu_seqlens_m=cu under "m" (varlen_m); `cu` (num_local_experts + 1) and `s_scatter` (max_rows; each row's
    global slot = (source rank, token, k) flattened, combine's row_key) = the routing; `op` = DispatchPull.
    block_rows = the GEMM's tile along the ragged dim, the gated unit (tile N under "n", the per-CTA tile_M under
    "m"); max_rows bounds the rows this rank receives (world x tokens_local x k is sure); ragged_dim = the GEMM
    call's ragged dim, None = this arch's default (gemm_base.grouped_ragged_dim). dtype = the rows' torch dtype, or
    a block-scaled format name: x holds the format's qdata, `x_sf` (tokens_local, hidden / sf_vec_size) its 1-byte
    scale rows and `sf_pool` the GEMM's per-block SF atoms (its SFB). prepare(x, topk_idx[, x_sf]) publishes this
    rank's rows and routes the call in one launch (the top-k exchange, an NVLink barrier that also orders the
    rows for the peers' pulls, the routing, the walk, the flag and counter resets): every rank calls it; it
    returns the GEMM's dist_op_args. The next prepare may start only once every rank's pulls of this call are done
    (a forward's combine barrier)."""

    def __init__(
        self, tokens_local, k, hidden, dtype, group, max_rows, block_rows, num_local_experts, ragged_dim=None
    ):
        import torch.distributed._symmetric_memory as symm_mem

        dev = torch.device(torch.cuda.current_device())
        rank, world = group.rank(), group.size()
        if ragged_dim is None:
            ragged_dim = grouped_ragged_dim(get_device_capacity(dev)[0])
        fmt = BlockScaledFormat.from_name(dtype) if isinstance(dtype, str) else None
        cols = hidden
        if fmt is not None:
            assert fmt.scale_dtype.itemsize == 1 and hidden % (4 * fmt.sf_vec_size) == 0, (
                f"{fmt.name}: the scale lane moves 1-byte scales in whole 4-scale words"
            )
            dtype, cols = fmt.qdata_dtype, fmt.storage_k(hidden)
        assert tokens_local * k % 4 == 0, "the top-k is published 4 slots per store"
        self._handles = []  # the symmetric mappings live as long as the runner

        def symm(shape, dtype):
            buf = symm_mem.empty(*shape, dtype=dtype, device=dev)
            hdl = symm_mem.rendezvous(buf, group=group)
            self._handles.append(hdl)
            bases = [hdl.get_buffer(q, shape, dtype).data_ptr() for q in range(world)]
            return buf, torch.tensor(bases, dtype=torch.int64, device=dev)

        i32 = lambda *shape: torch.empty(*shape, dtype=torch.int32, device=dev)
        i64 = lambda *shape: torch.empty(*shape, dtype=torch.int64, device=dev)
        n_tokens = world * tokens_local
        self.x, peer_base = symm((tokens_local, cols), dtype)
        topk_g, topk_g_base = symm((n_tokens * k,), torch.int32)
        signal, signal_base = symm((1,), torch.int32)
        signal.zero_()
        self._handles[-1].barrier()  # every rank's word is zero before any arrive
        ncols = -(-n_tokens // block_rows)  # an expert reads each token at most once: its blocks <= the tokens'
        sf = fmt is not None
        walk = WalkArguments(
            src_address=i64(n_tokens),
            gate_ptr=i32(n_tokens + 1),
            gates=i32(max_rows),
            num_rows=i32(1),
            dst_row=i32(max_rows),
            src_sf_address=i64(n_tokens) if sf else None,
            dst_sf_row=i32(max_rows) if sf else None,
        )
        flags = i32(num_local_experts, ncols)
        self.sem = torch.zeros(1, dtype=torch.int32, device=dev)  # the GEMM's tile counter
        self.rows = torch.empty(max_rows, cols, dtype=dtype, device=dev)
        self.cu, self.s_scatter = i32(num_local_experts + 1), i32(max_rows)
        s_reverse = i32(n_tokens * k)
        counts = i32(num_local_experts, -(-n_tokens // GROUP_THREADS))
        counts2 = i32(2, num_local_experts * ncols)
        grid_counter = torch.zeros(1, dtype=torch.int32, device=dev)
        sf_words, sf_pool_bytes, sf_base, sf_row_bytes, cu_tiles = 0, None, None, 0, None
        if sf:
            vec = fmt.sf_vec_size
            self.x_sf, sf_base = symm((tokens_local, hidden // vec), fmt.scale_dtype)
            self.sf_pool = torch.zeros(
                1,
                num_local_experts * ncols * varlen_n_block_atoms(block_rows),
                hidden // (4 * vec),
                32,
                4,
                4,
                dtype=fmt.scale_dtype,
                device=dev,
            )
            sf_words, sf_row_bytes = hidden // (4 * vec), hidden // vec
            sf_pool_bytes = self.sf_pool.view(torch.uint8).view(-1)  # the pull addresses bytes
            cu_tiles = i32(num_local_experts + 1)
        row_bytes = cols * self.x.element_size()
        self.op = DispatchPull(row_bytes, sf_words, ragged_dim)
        self.args = DispatchArguments(walk, self.rows.view(torch.uint8), flags, self.sem, sf_pool_bytes)
        self.tokens_local, self.k = tokens_local, k

        kernel = DispatchWalk(
            k,
            num_local_experts,
            world,
            block_rows,
            torch.cuda.get_device_properties(dev).multi_processor_count,
            sf_atom_rows_of(block_rows) if sf else 0,
            ragged_dim == "n",
        )
        v = lambda: make_fake_tensor(Int32, (cute.sym_int(),), leading_dim=0, divisibility=1)
        v64 = lambda: make_fake_tensor(Int64, (cute.sym_int(),), leading_dim=0, divisibility=1)
        m = lambda: make_fake_tensor(Int32, (cute.sym_int(), cute.sym_int()), leading_dim=1, divisibility=1)
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        fn = cute.compile(
            kernel, v(), v(), v64(), v(), v64(), v(), m(), m(), v(), v(), v(), m(), v(), v64(),
            fake_walk_arguments(sf), Int32(rank), Int32(tokens_local), Int32(row_bytes), stream,
            v64() if sf else None, Int32(sf_row_bytes), v() if sf else None, options="--enable-tvm-ffi",
        )
        self._walk = lambda topk: fn(
            topk, topk_g, topk_g_base, signal, signal_base, grid_counter, counts, counts2, self.cu, s_reverse,
            self.s_scatter, flags, self.sem, peer_base, walk, rank, tokens_local, row_bytes, sf_base, sf_row_bytes,
            cu_tiles,
        )

    def prepare(self, x, topk_idx, x_sf=None):
        assert topk_idx.shape == (self.tokens_local, self.k) and topk_idx.dtype == torch.int32
        self.x.copy_(x)  # the publish: the walk's barrier orders it for the peers' pulls
        if x_sf is not None:
            self.x_sf.copy_(x_sf)
        self._walk(topk_idx.contiguous().view(-1))
        return self.args
