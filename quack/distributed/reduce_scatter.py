# GEMM + reduce-scatter as the GEMM's distributed operator (runs_evt: the GEMM's epilogue warps store raw
# partials, this operator's comm warps reduce them and run the EVT). Every rank's epilogue warps store
# their raw K-partial of each tile into their symmetric partials buffer (partial_dtype, the wire;
# epilogue(store_partial)) and release-add the flag of the rank whose M slab owns the tile. The comm warps
# step this CTA's static schedule: at
# tile row pid_m = (stripe, pos) they reduce stripe `stripe` of owned tile `pos` (the cutlass RS remap:
# the world's CTAs at one position share one owned tile, equal work, no slab walk), wait for the tile's
# world partials, multimem ld_reduce the stripe from the multicast partials view, run the GEMM's epilogue
# on the sum (the epi ops, framed on the stripe) and commit it into D, this rank's (m // world, n) slab
# (the user's output, any dtype quack stores D in: a quantized D runs its SF op here, after the sum).
# Usage: d, args = make_reduce_scatter_buffers(m, n, dtype, tile_m, tile_n, ctas); mod(a, b, out={"D": d},
# config=cfg, dist_op=ReduceScatter(rank, world), dist_op_args=args). Direct:
# GemmSm100(..., dist_op=ReduceScatter(rank, world)), launch with mD=d, dist_op_args=args.
# AllReduce = the same reduce with a broadcast commit: the reduced stripe goes by multimem st into every
# rank's symmetric (m, n) D (two-shot AR); make_all_reduce_buffers.
# A C operand (residual) is applied after the reduce: the epi-load warp stages it per comm subtile at
# the result's coordinates, so C is slab-shaped (m // world, n) for RS and the full (m, n) for AR.
import math
from functools import partial
from typing import NamedTuple

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import Int32, Int64
from cutlass.cute.typing import AddressSpace
from quack.compile_utils import make_fake_tensor
from quack.cute_dsl_utils import mlir_namedtuple
from quack.gemm_base import DistOp
from quack.tile_scheduler import RasterOrderOption
from quack.varlen_utils import VarlenManager

def _frame(dtype, rows, cols, threads, subtile_atoms=8):
    """Thread layout over a (rows, cols) frame, 16-byte atoms n fastest, ~subtile_atoms atoms in flight per
    thread per subtile (the measured knee): (thr_shape, val_shape, chunk)."""
    atom = 128 // dtype.width
    thr_n = math.gcd(cols // atom, threads)
    thr_m = threads // thr_n
    assert rows % thr_m == 0 and cols % (atom * thr_n) == 0, "the frame tiles over the threads"
    loop_m, loop_n = rows // thr_m, cols // (atom * thr_n)
    chunk = math.gcd(loop_m, max(1, subtile_atoms // loop_n))
    return (thr_m, thr_n), (1, atom), chunk


@mlir_namedtuple
class ReduceScatterArguments(NamedTuple):
    """The comm warps' bundle (the GEMM's dist_op_args). flags (ntile_m, ntile_n) int32 is this rank's
    copy of the tile flags (every rank's producers add to the owner's copy, the owner's consumers
    consume); peer_flags (world,) int64 holds every rank's flags address."""

    partials: cute.Tensor  # the symmetric (m, n) partials buffer (partial_dtype): the GEMM stores into it
    partials_mc: cute.Tensor  # its multicast view: the reducers' source
    flags: cute.Tensor
    peer_flags: cute.Tensor
    exit_flags: cute.Tensor  # (ctas,) int32 exit-barrier slots, one per resident CTA
    exit_flags_mc: cute.Tensor


@mlir_namedtuple
class AllReduceArguments(NamedTuple):
    """ReduceScatterArguments plus the symmetric (m, n) D's multicast view (the broadcast commit's
    target), with the flags' multicast view in place of the peer table: every rank reduces a stripe of
    every tile, so a tile's arrival is published to every rank's flag copy at once."""

    partials: cute.Tensor
    partials_mc: cute.Tensor
    d_mc: cute.Tensor
    flags: cute.Tensor
    flags_mc: cute.Tensor
    exit_flags: cute.Tensor
    exit_flags_mc: cute.Tensor


def _fake_arguments(cls, d_dtype):
    def mat(dt, div):
        return make_fake_tensor(
            dt, (cute.sym_int(), cute.sym_int()), leading_dim=1, divisibility=div
        )

    def vec(dt):
        return make_fake_tensor(dt, (cute.sym_int(),), leading_dim=0, divisibility=1)

    partials = (mat(ReduceScatter.partial_dtype, 8), mat(ReduceScatter.partial_dtype, 8))
    if cls is AllReduceArguments:
        return cls(*partials, mat(d_dtype, 8), mat(Int32, 1), mat(Int32, 1), vec(Int32), vec(Int32))
    return cls(*partials, mat(Int32, 1), vec(Int64), vec(Int32), vec(Int32))


@cute.jit
def stripe_rows(
    pid_m: Int32,
    tiles_per_rank: Int32,
    rank: cutlass.Constexpr[int],
    world: cutlass.Constexpr[int],
    broadcast: cutlass.Constexpr[bool],
):
    """(in_row, out_row) in stripe units (cta_m // world rows) for the schedule tile at row pid_m. AR:
    stripe `rank` of the tile this CTA computed (no slab, any tile count). RS: the result rows belong to
    one rank, so the CTA reduces stripe pid_m // tiles_per_rank of its own rank's slab tile at position
    pid_m % tiles_per_rank, and out_row is slab-local."""
    if cutlass.const_expr(broadcast):
        in_row = pid_m * world + rank
        return in_row, in_row
    stripe, pos = pid_m // tiles_per_rank, pid_m % tiles_per_rank
    in_row = (pos + rank * tiles_per_rank) * world + stripe
    return in_row, pos * world + stripe


@cute.jit
def multimem_reduce_subtile(
    frgD_mc: cute.Tensor,
    tRS_rD: cute.Tensor,
    epi_coord: cute.Coord,
    no_release: cutlass.Constexpr[bool] = False,
):
    """gemm.epilogue's load_acc_subtile: the subtile's world partials summed by multimem ld_reduce
    into tRS_rD. All loads issue before any store, so a visit pays one switch round trip.
    Numerics: the switch's bf16x2 ld_reduce truncates (no RN), so a sum is within ~1 ulp toward
    zero of the fp32 sum; the .f32 variant rounds to nearest."""
    _atom, chunk, loop_n = tRS_rD.shape
    atom = cute.size(tRS_rD, mode=[0])  # elements per 16-byte vector
    words = cute.make_rmem_tensor((4, chunk, loop_n), Int32)
    for ii in cutlass.range_constexpr(chunk):
        i = epi_coord[0] * chunk + ii
        for jj in cutlass.range_constexpr(loop_n):
            j = epi_coord[1] * loop_n + jj
            x, y, z, w = utils.distributed.multimem_ld_reduce(
                frgD_mc[None, i, j].iterator, dtype=frgD_mc.element_type, num_elements=atom
            )
            words[0, ii, jj], words[1, ii, jj], words[2, ii, jj], words[3, ii, jj] = x, y, z, w
    tRS_rD.store(cute.recast_tensor(words, frgD_mc.element_type).load().to(tRS_rD.element_type))


@cute.jit
def load_subtile(frgIn: cute.Tensor, tRS_rC: cute.Tensor, epi_coord: cute.Coord):
    """gemm.epilogue's load_C: the subtile of C as 16-byte loads straight into the C fragment."""
    _atom, chunk, loop_n = tRS_rC.shape
    for ii in cutlass.range_constexpr(chunk):
        i = epi_coord[0] * chunk + ii
        for jj in cutlass.range_constexpr(loop_n):
            cute.autovec_copy(frgIn[None, i, epi_coord[1] * loop_n + jj], tRS_rC[None, ii, jj])


@cute.jit
def commit_subtile_local(frgD: cute.Tensor, vals: cute.Tensor, epi_coord: cute.Coord):
    """gemm.epilogue's commit_D: the reduced subtile, after the epi ops and the convert to D's dtype, as
    vector stores into the slab."""
    _atom, chunk, loop_n = vals.shape
    for ii in cutlass.range_constexpr(chunk):
        i = epi_coord[0] * chunk + ii
        for jj in cutlass.range_constexpr(loop_n):
            cute.autovec_copy(vals[None, ii, jj], frgD[None, i, epi_coord[1] * loop_n + jj])


@cute.jit
def commit_subtile_broadcast(frgD_mc: cute.Tensor, vals: cute.Tensor, epi_coord: cute.Coord):
    """gemm.epilogue's commit_D for all-reduce: the converted reduced subtile as 16-byte multimem
    stores into every rank's D."""
    _atom, chunk, loop_n = vals.shape
    words = cute.recast_tensor(vals, Int32)
    for ii in cutlass.range_constexpr(chunk):
        i = epi_coord[0] * chunk + ii
        for jj in cutlass.range_constexpr(loop_n):
            j = epi_coord[1] * loop_n + jj
            utils.distributed.multimem_st_4xb32(
                frgD_mc[None, i, j].iterator,
                words[0, ii, jj].ir_value(),
                words[1, ii, jj].ir_value(),
                words[2, ii, jj].ir_value(),
                words[3, ii, jj].ir_value(),
            )


class ReduceScatter(DistOp):
    """GEMM + reduce-scatter over `world` ranks along M: D = this rank's (m // world, n) slab."""

    num_warps = 4
    regs = None  # 12 warps at 168 each, as intra-sm ran it; a split must fit the released pool
    runs_evt = True
    raster_order = RasterOrderOption.AlongN  # the reducers' measured raster (S2 schedule)
    broadcast = False  # AllReduce: commit by multimem st into every rank's (m, n) D
    partial_dtype = cutlass.BFloat16  # the wire (bf16 beat an e4m3 wire at TP=2)

    def __init__(self, rank: int, world: int):
        self.rank, self.world = rank, world

    def fake_args(self, d_dtype):
        return _fake_arguments(
            AllReduceArguments if self.broadcast else ReduceScatterArguments, d_dtype
        )

    def d_shape(self, shape):
        m, n = shape
        return shape if self.broadcast else (m // self.world, n)

    def partial_tensor(self, dist_op_args):
        return dist_op_args.partials

    def _stripe(self, gemm):
        """The comm threads' _frame over this rank's (cta_m // world, cta_n) stripe of the tile:
        (slice_m, thr_shape, val_shape, chunk)."""
        cta_m, cta_n = gemm.cta_tile_shape_mnk[0], gemm.cta_tile_shape_mnk[1]
        assert cta_m % self.world == 0, "the tile's M splits into one stripe per rank"
        slice_m = cta_m // self.world
        threads = gemm.num_comm_warps * cute.arch.WARP_SIZE
        return slice_m, *_frame(self.partial_dtype, slice_m, cta_n, threads)

    def epi_warp_shape_mnk(self, gemm):
        _, (_, atom_thr_n), _, _ = self._stripe(gemm)
        warps_n = max(1, atom_thr_n // cute.arch.WARP_SIZE)
        return (gemm.num_comm_warps // warps_n, warps_n, 1)

    def epi_tile(self, gemm):
        _, (thr_m, _), _, chunk = self._stripe(gemm)
        return (chunk * thr_m, gemm.cta_tile_shape_mnk[1])

    @cute.jit
    def signal_tile(self, gemm, ctx, tile_coord_mnkl, epi_store_pipeline, is_tma_warp, tidx: Int32):
        """Drain the partial's TMA stores, then one release add on the tile's flag, in every rank's copy
        (AR: every rank reduces a stripe of it) or the owner's (RS)."""
        dist_op_args = ctx.args
        if is_tma_warp:
            epi_store_pipeline.producer_tail()
            flags = dist_op_args.flags
            idx = cute.crd2idx((tile_coord_mnkl[0], tile_coord_mnkl[1]), flags.layout)
            with cute.arch.elect_one():
                if cutlass.const_expr(self.broadcast):
                    utils.distributed.multimem_red_add1(
                        lock_ptr=dist_op_args.flags_mc.iterator + idx, order="release", scope="gpu"
                    )
                else:
                    owner = tile_coord_mnkl[0] // (flags.shape[0] // self.world)
                    flag = cute.make_ptr(
                        Int32,
                        dist_op_args.peer_flags[owner] + Int64(idx) * Int64(4),
                        AddressSpace.gmem,
                        assumed_align=4,
                    )
                    utils.distributed.red_add1(lock_ptr=flag, order="release", scope="gpu")

    @cute.jit
    def comm_warps(self, gemm, ctx, warp_idx: Int32, tidx: Int32):
        assert gemm.split_k == 1, "reduce-scatter v1: no split-K"
        rank, world, broadcast = self.rank, self.world, self.broadcast
        # bound methods and ctx fields cross into the loop as locals; the gemm / comm / ctx objects
        # cannot (TYPE_DYNAMIC_EXPR_UNSUPPORTED)
        epilogue, comm_barrier = gemm.epilogue, gemm.comm_barrier
        dist_op_args, TileSchedulerCls, varlen_manager = ctx.args, ctx.TileSchedulerCls, ctx.varlen
        epilogue_params, epi_smem_tensors = ctx.epi_params, ctx.epi_smem_tensors
        mC, mD = ctx.mC, ctx.mD
        cta_m, cta_n = gemm.cta_tile_shape_mnk[0], gemm.cta_tile_shape_mnk[1]
        slice_m, thr_shape, val_shape, chunk = self._stripe(gemm)
        partials_mc, flags = dist_op_args.partials_mc, dist_op_args.flags
        if cutlass.const_expr(broadcast):
            assert mD.element_type.width == self.partial_dtype.width, (
                "the broadcast commit stores the frame's 16-byte vectors: D's dtype must be the wire's width"
            )
            out, commit_subtile = dist_op_args.d_mc, commit_subtile_broadcast
        else:
            out, commit_subtile = mD, commit_subtile_local
        tiles_per_rank = cute.ceil_div(varlen_manager.len_m(Int32(0)), cta_m) // world
        tiled_copy = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), partials_mc.element_type),
            cute.make_layout(thr_shape, stride=(thr_shape[1], 1)),
            cute.make_layout(val_shape, stride=(val_shape[1], 1)),
        )
        comm_tidx = tidx - gemm.comm_warp_ids[0] * cute.arch.WARP_SIZE
        thr_copy = tiled_copy.get_slice(comm_tidx)
        gIn = cute.local_tile(partials_mc, (slice_m, cta_n), (None, None))
        gOut = cute.local_tile(out, (slice_m, cta_n), (None, None))
        _atom, _, loop_n = thr_copy.partition_S(cute.make_identity_tensor((slice_m, cta_n))).shape
        epi_tile = self.epi_tile(gemm)  # the epilogue's subtile on the stripe
        tRS_rD = cute.make_rmem_tensor((_atom, chunk, loop_n), gemm.acc_dtype)
        # C: read per subtile straight from gmem, the same thread copy as the partials
        tRS_rC, gC = None, None
        if cutlass.const_expr(mC is not None):
            tRS_rC = cute.make_rmem_tensor((_atom, chunk, loop_n), mC.element_type)
            gC = cute.local_tile(mC, (slice_m, cta_n), (None, None))
        is_tma_warp = warp_idx == gemm.comm_warp_ids[0]
        epi_store_pipeline = gemm.make_epi_store_pipeline(gemm.num_comm_warps)  # aux outputs' TMA stores
        # the epilogue's frame: slab-local row limits (m // world) for RS, the full (m, n) out for AR
        out_varlen = varlen_manager
        if cutlass.const_expr(not self.broadcast):
            out_varlen = VarlenManager.create(
                varlen_manager.params,
                len_m_static=varlen_manager.len_m(Int32(0)) // world,
                len_k_static=varlen_manager.len_k(Int32(0)),
                len_n_static=varlen_manager.len_n(),
            )

        # this CTA's static schedule, stepped locally (never the scheduler pipeline)
        tile_sched = TileSchedulerCls()
        work_tile = tile_sched.initial_work_tile_info()
        while work_tile.is_valid_tile:
            pid_m, pid_n = work_tile.tile_idx[0], work_tile.tile_idx[1]
            in_row, out_row = stripe_rows(pid_m, tiles_per_rank, rank, world, broadcast)
            flag = flags.iterator + cute.crd2idx((in_row // world, pid_n), flags.layout)
            # world partials landed; the flag's consumers (AR: this CTA alone; RS: the slab tile's world
            # stripe reducers) each add 1 and the last resets it
            consumers = 1 if cutlass.const_expr(broadcast) else world
            if comm_tidx == 0:
                utils.distributed.spin_lock_ld_lt_relaxed_wait(
                    lock_ptr=flag, expected_val=Int32(world), scope="gpu"
                )
                seen = cute.arch.atomic_add(flag.llvm_ptr, Int32(1), sem="relaxed", scope="gpu")
                if seen == world + consumers - 1:
                    cute.arch.store(flag.llvm_ptr, Int32(0), sem="relaxed", scope="gpu")
            comm_barrier.arrive_and_wait()
            load_C = None
            if cutlass.const_expr(mC is not None):
                load_C = partial(load_subtile, thr_copy.partition_S(gC[None, None, out_row, pid_n]))
            epilogue(
                epilogue_params,
                epi_smem_tensors,
                None,  # epi_pipeline: C is read directly
                epi_store_pipeline,
                None,  # epi_read_state
                None,  # epi_producer_state
                epi_tile,
                partial(
                    multimem_reduce_subtile, thr_copy.partition_S(gIn[None, None, in_row, pid_n])
                ),
                tRS_rD=tRS_rD,
                tRS_rC=tRS_rC,
                tiled_copy_t2r=None,
                tiled_copy_r2s=tiled_copy,
                tRS_sD=None,
                tiled_copy_s2r=None,
                tSR_rC=None,
                tSR_sC=None,
                copy_D=None,
                copy_C=None,
                tile_coord_mnkl=(out_row, pid_n, Int32(0), Int32(0)),
                varlen_manager=out_varlen,
                epilogue_barrier=comm_barrier,
                tile_scheduler=tile_sched,
                tidx=comm_tidx,
                is_tma_warp=is_tma_warp,
                commit_D=partial(
                    commit_subtile, thr_copy.partition_S(gOut[None, None, out_row, pid_n])
                ),
                load_C=load_C,
                tile_shape_mn=(slice_m, cta_n),
            )
            tile_sched._current_work_idx = tile_sched._fetch_next_work_idx()
            work_tile = tile_sched._delinearize_work_idx(tile_sched._current_work_idx)

        if is_tma_warp:
            epi_store_pipeline.producer_tail()
        # exit barrier: no rank leaves (and lets the next launch overwrite its D) while a peer still
        # reduces from it; for AR it also orders every peer's broadcast into this rank's out
        comm_barrier.arrive_and_wait()
        if is_tma_warp:
            with cute.arch.elect_one():
                bidx, bidy, bidz = cute.arch.block_idx()
                gdx, gdy, _ = cute.arch.grid_dim()
                slot = bidx + gdx * (bidy + gdy * bidz)
                utils.distributed.multimem_red_add1(
                    lock_ptr=dist_op_args.exit_flags_mc.iterator + slot, order="release", scope="gpu"
                )
                exit_flag = dist_op_args.exit_flags.iterator + slot
                utils.distributed.spin_lock_ld_lt_relaxed_wait(
                    lock_ptr=exit_flag, expected_val=Int32(world), scope="gpu"
                )
                cute.arch.atomic_add(exit_flag.llvm_ptr, Int32(-world), sem="relaxed", scope="gpu")


class AllReduce(ReduceScatter):
    """GEMM + all-reduce over `world` ranks: the reduce-scatter whose reduced stripes are broadcast into
    every rank's symmetric (m, n) D. Reduce outputs are broadcast too (store_sink): pass the multicast
    view of a symmetric buffer and finalize the local copy."""

    broadcast = True

    @cute.jit
    def store_sink(self, dst: cute.Tensor, idx, value):
        assert dst.element_type.width == 32, "AR reduce outputs: 32-bit slots (one multimem st.b32 each)"
        word = cute.make_rmem_tensor((1,), dst.element_type)
        word[0] = value
        utils.distributed.multimem_st_1xb32(
            dst.iterator + cute.crd2idx(idx, dst.layout),
            cute.recast_tensor(word, Int32)[0].ir_value(),
        )


# ---- host ----


def _wrap_device_ptr(ptr, shape, torch_dtype):
    """Non-owning CUDA tensor over a raw device pointer (a symm_mem multicast / peer VA)."""
    import torch

    dev = torch.device("cuda", torch.cuda.current_device())
    storage = torch._C._construct_storage_from_data_pointer(
        ptr, dev, math.prod(shape) * torch_dtype.itemsize
    )
    return torch.empty(0, dtype=torch_dtype, device=dev).set_(storage).view(shape)


def _symm(shape, torch_dtype, zero=False):
    """(local, multicast view, handle) of a symmetric allocation over the world group."""
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem

    base = symm_mem.empty(tuple(shape), dtype=torch_dtype, device="cuda")
    if zero:
        base.zero_()
    hdl = symm_mem.rendezvous(base, group=dist.group.WORLD)
    return base, _wrap_device_ptr(hdl.multicast_ptr, tuple(shape), torch_dtype), hdl


def _make_partials(m, n, tile_m, tile_n, ctas):
    """(partials, partials_mc, flags, flags_mc, peer_flags, exit_flags, exit_flags_mc): the symmetric
    partials buffer the GEMM stores into (partial_dtype) with its multicast view, the tile flags with their
    multicast view and every rank's flags address, and the exit slots (ctas = the persistent grid's CTA
    count)."""
    import torch

    from quack.cute_dsl_utils import torch2cute_dtype_map

    assert m % tile_m == 0 and n % tile_n == 0, "full tiles"
    wire = next(t for t, c in torch2cute_dtype_map.items() if c is ReduceScatter.partial_dtype)
    partials, partials_mc, _ = _symm((m, n), wire)
    flags, flags_mc, flags_hdl = _symm((m // tile_m, n // tile_n), torch.int32, zero=True)
    peer_flags = torch.tensor(flags_hdl.buffer_ptrs, dtype=torch.int64, device="cuda")
    exit_flags, exit_flags_mc, _ = _symm((ctas,), torch.int32, zero=True)
    return partials, partials_mc, flags, flags_mc, peer_flags, exit_flags, exit_flags_mc


def make_reduce_scatter_buffers(m, n, torch_dtype, tile_m, tile_n, ctas):
    """(d, args) for GEMM + RS of an (m, n) output: D = this rank's (m // world, n) slab in torch_dtype, and
    the ReduceScatterArguments. The operator is ReduceScatter(rank, world); the flags self-reset, so
    nothing runs between launches."""
    import torch
    import torch.distributed as dist

    world = dist.get_world_size()
    assert m % (world * tile_m) == 0, "RS: a whole number of tiles per slab"
    partials, partials_mc, flags, _, *rest = _make_partials(m, n, tile_m, tile_n, ctas)
    cols = n // 2 if torch_dtype == torch.float4_e2m1fn_x2 else n  # fp4 D packs two values a byte
    d = torch.empty(m // world, cols, dtype=torch_dtype, device="cuda")
    torch.cuda.synchronize()
    return d, ReduceScatterArguments(partials, partials_mc, flags, *rest)


def make_all_reduce_buffers(m, n, torch_dtype, tile_m, tile_n, ctas):
    """(d, args) for GEMM + two-shot AR: D = this rank's symmetric copy of the full (m, n) result, and the
    AllReduceArguments. The operator is AllReduce(rank, world)."""
    import torch

    partials, partials_mc, flags, flags_mc, _, *rest = _make_partials(m, n, tile_m, tile_n, ctas)
    d, d_mc, _ = _symm((m, n), torch_dtype)
    torch.cuda.synchronize()
    return d, AllReduceArguments(partials, partials_mc, d_mc, flags, flags_mc, *rest)
