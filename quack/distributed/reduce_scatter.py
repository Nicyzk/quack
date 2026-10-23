# GEMM + reduce-scatter as the GEMM's comm object, through gemm_base's split-rank protocol: ranks are
# one more split of K with spatial roles. Every rank's epilogue warps store their raw K-partial of each
# tile into their symmetric D (the d_dtype wire, gemm_base.split_rank_partial_commit) and release-add
# the flag of the rank whose M slab owns the tile. The comm warps step this CTA's static schedule: at
# tile row pid_m = (stripe, pos) they reduce stripe `stripe` of owned tile `pos` (the cutlass RS remap:
# the world's CTAs at one position share one owned tile, equal work, no slab walk), wait for the tile's
# world partials, multimem ld_reduce the stripe from the multicast D view, run the GEMM's epilogue on
# the sum (the epi ops, framed on the stripe) and store it into this rank's slab.
# Usage: rs = ReduceScatterRunner(m, n, dtype, tile_m, tile_n, ctas); mod(a, b, config=cfg, comm=rs)["D"] is
# this rank's slab (the runner supplies the wire as D and the comm_args). Direct: GemmSm100(..., comm=rs.comm),
# launch with mD=rs.wire, comm_args=rs.args(); rs.out is the result.
# AllReduce = the same reduce with a broadcast commit: the reduced stripe goes by multimem st into every
# rank's symmetric out (two-shot AR); AllReduceRunner, out is the full result on every rank.
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
from quack.gemm_base import Comm, CommRunner, EpiFrame
from quack.varlen_utils import VarlenManager

COMM_WARPS = 4
SUBTILE_ATOMS = 8  # 16-byte atoms in flight per thread per reduce visit (the measured knee)


@mlir_namedtuple
class ReduceScatterArguments(NamedTuple):
    """The comm warps' bundle (the GEMM's comm_args). flags (ntile_m, ntile_n) int32 is this rank's
    copy of the tile flags (every rank's producers add to the owner's copy, the owner's consumers
    consume); peer_flags (world,) int64 holds every rank's flags address."""

    d_mc: cute.Tensor  # multicast view of the symmetric D (m, n): the partials wire
    out: cute.Tensor  # this rank's slab (m // world, n), the result
    flags: cute.Tensor
    peer_flags: cute.Tensor
    exit_flags: cute.Tensor  # (ctas,) int32 exit-barrier slots, one per resident CTA
    exit_flags_mc: cute.Tensor


@mlir_namedtuple
class AllReduceArguments(NamedTuple):
    """ReduceScatterArguments with the symmetric (m, n) out's multicast view in place of the slab, and
    the flags' multicast view in place of the peer table: every rank reduces a stripe of every tile, so a
    tile's arrival is published to every rank's flag copy at once."""

    d_mc: cute.Tensor
    out_mc: cute.Tensor
    flags: cute.Tensor
    flags_mc: cute.Tensor
    exit_flags: cute.Tensor
    exit_flags_mc: cute.Tensor


def _fake_arguments(cls, dtype):
    def mat(dt, div):
        return make_fake_tensor(
            dt, (cute.sym_int(), cute.sym_int()), leading_dim=1, divisibility=div
        )

    def vec(dt):
        return make_fake_tensor(dt, (cute.sym_int(),), leading_dim=0, divisibility=1)

    peers = mat(Int32, 1) if cls is AllReduceArguments else vec(Int64)  # flags_mc / peer_flags
    return cls(mat(dtype, 8), mat(dtype, 8), mat(Int32, 1), peers, vec(Int32), vec(Int32))


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
def commit_subtile_local(frgOut: cute.Tensor, tRS_rD: cute.Tensor, epi_coord: cute.Coord):
    """gemm.epilogue's commit_D: the reduced subtile, after the epi ops, as 16-byte stores into the
    slab."""
    atom, chunk, loop_n = tRS_rD.shape
    vals = cute.make_rmem_tensor(tRS_rD.shape, frgOut.element_type)
    vals.store(tRS_rD.load().to(frgOut.element_type))
    for ii in cutlass.range_constexpr(chunk):
        i = epi_coord[0] * chunk + ii
        for jj in cutlass.range_constexpr(loop_n):
            cute.autovec_copy(vals[None, ii, jj], frgOut[None, i, epi_coord[1] * loop_n + jj])


@cute.jit
def commit_subtile_broadcast(frgOut_mc: cute.Tensor, tRS_rD: cute.Tensor, epi_coord: cute.Coord):
    """gemm.epilogue's commit_D for all-reduce: the reduced subtile as 16-byte multimem stores into
    every rank's out."""
    _atom, chunk, loop_n = tRS_rD.shape
    vals = cute.make_rmem_tensor(tRS_rD.shape, frgOut_mc.element_type)
    vals.store(tRS_rD.load().to(frgOut_mc.element_type))
    words = cute.recast_tensor(vals, Int32)
    for ii in cutlass.range_constexpr(chunk):
        i = epi_coord[0] * chunk + ii
        for jj in cutlass.range_constexpr(loop_n):
            j = epi_coord[1] * loop_n + jj
            utils.distributed.multimem_st_4xb32(
                frgOut_mc[None, i, j].iterator,
                words[0, ii, jj].ir_value(),
                words[1, ii, jj].ir_value(),
                words[2, ii, jj].ir_value(),
                words[3, ii, jj].ir_value(),
            )


class ReduceScatter(Comm):
    """GEMM + reduce-scatter over `world` ranks along M (slab = m // world rows per rank), d_dtype
    wire."""

    num_warps = COMM_WARPS
    regs = None  # 12 warps at 168 each, as intra-sm ran it; a split must fit the released pool
    split_rank = True
    broadcast = False  # AllReduce: commit by multimem st into every rank's (m, n) out

    def __init__(self, rank: int, world: int):
        self.rank, self.world = rank, world

    def fake_args(self, d_dtype):
        return _fake_arguments(
            AllReduceArguments if self.broadcast else ReduceScatterArguments, d_dtype
        )

    def _stripe(self, gemm):
        """The comm threads over a (cta_m // world, cta_n) stripe, 16-byte atoms n fastest, ~SUBTILE_ATOMS
        atoms in flight per thread per epilogue subtile: (slice_m, thr_shape, val_shape, chunk)."""
        cta_m, cta_n = gemm.cta_tile_shape_mnk[0], gemm.cta_tile_shape_mnk[1]
        assert cta_m % self.world == 0, "the tile's M splits into one stripe per rank"
        slice_m = cta_m // self.world
        atom = 128 // gemm.d_dtype.width
        threads = gemm.num_comm_warps * cute.arch.WARP_SIZE
        atom_thr_n = math.gcd(cta_n // atom, threads)
        atom_thr_m = threads // atom_thr_n
        loop_m, loop_n = slice_m // atom_thr_m, cta_n // (atom * atom_thr_n)
        chunk = math.gcd(loop_m, max(1, SUBTILE_ATOMS // loop_n))
        return slice_m, (atom_thr_m, atom_thr_n), (1, atom), chunk

    def epi_frame(self, gemm):
        slice_m, (atom_thr_m, atom_thr_n), _, chunk = self._stripe(gemm)
        cta_n = gemm.cta_tile_shape_mnk[1]
        warps_n = max(1, atom_thr_n // cute.arch.WARP_SIZE)
        return EpiFrame(
            (slice_m, cta_n),
            (chunk * atom_thr_m, cta_n),
            (gemm.num_comm_warps // warps_n, warps_n, 1),
        )

    @cute.jit
    def signal_tile(self, gemm, ctx, tile_coord_mnkl, epi_store_pipeline, is_tma_warp, tidx: Int32):
        """The partial's TMA stores drained in split_rank_partial_commit: one release add on the tile's
        flag, in every rank's copy (AR: every rank reduces a stripe of it) or the owner's (RS)."""
        comm_args = ctx.args
        if is_tma_warp:
            flags = comm_args.flags
            idx = cute.crd2idx((tile_coord_mnkl[0], tile_coord_mnkl[1]), flags.layout)
            with cute.arch.elect_one():
                if cutlass.const_expr(self.broadcast):
                    utils.distributed.multimem_red_add1(
                        lock_ptr=comm_args.flags_mc.iterator + idx, order="release", scope="gpu"
                    )
                else:
                    owner = tile_coord_mnkl[0] // (flags.shape[0] // self.world)
                    flag = cute.make_ptr(
                        Int32,
                        comm_args.peer_flags[owner] + Int64(idx) * Int64(4),
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
        comm_args, TileSchedulerCls, varlen_manager = ctx.args, ctx.TileSchedulerCls, ctx.varlen
        epilogue_params, epi_smem_tensors, mC = ctx.epi_params, ctx.epi_smem_tensors, ctx.mC
        cta_m, cta_n = gemm.cta_tile_shape_mnk[0], gemm.cta_tile_shape_mnk[1]
        slice_m, thr_shape, val_shape, chunk = self._stripe(gemm)
        d_mc, flags = comm_args.d_mc, comm_args.flags
        if cutlass.const_expr(broadcast):
            out, commit_subtile = comm_args.out_mc, commit_subtile_broadcast
        else:
            out, commit_subtile = comm_args.out, commit_subtile_local
        tiles_per_rank = cute.ceil_div(varlen_manager.len_m(Int32(0)), cta_m) // world
        tiled_copy = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), d_mc.element_type),
            cute.make_layout(thr_shape, stride=(thr_shape[1], 1)),
            cute.make_layout(val_shape, stride=(val_shape[1], 1)),
        )
        comm_tidx = tidx - gemm.comm_warp_ids[0] * cute.arch.WARP_SIZE
        thr_copy = tiled_copy.get_slice(comm_tidx)
        gIn = cute.local_tile(d_mc, (slice_m, cta_n), (None, None))
        gOut = cute.local_tile(out, (slice_m, cta_n), (None, None))
        _atom, _, loop_n = thr_copy.partition_S(cute.make_identity_tensor((slice_m, cta_n))).shape
        epi_tile = (chunk * thr_shape[0], cta_n)  # the epilogue's subtile on the stripe
        tRS_rD = cute.make_rmem_tensor((_atom, chunk, loop_n), gemm.acc_dtype)
        # C: read per subtile straight from gmem, the same thread copy as the partials
        tRS_rC, gC = None, None
        if cutlass.const_expr(mC is not None):
            tRS_rC = cute.make_rmem_tensor((_atom, chunk, loop_n), mC.element_type)
            gC = cute.local_tile(mC, (slice_m, cta_n), (None, None))
        is_tma_warp = warp_idx == gemm.comm_warp_ids[0]
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
                None,  # epi_store_pipeline
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

        # exit barrier: no rank leaves (and lets the next launch overwrite its D) while a peer still
        # reduces from it; for AR it also orders every peer's broadcast into this rank's out
        comm_barrier.arrive_and_wait()
        if is_tma_warp:
            with cute.arch.elect_one():
                bidx, bidy, bidz = cute.arch.block_idx()
                gdx, gdy, _ = cute.arch.grid_dim()
                slot = bidx + gdx * (bidy + gdy * bidz)
                utils.distributed.multimem_red_add1(
                    lock_ptr=comm_args.exit_flags_mc.iterator + slot, order="release", scope="gpu"
                )
                exit_flag = comm_args.exit_flags.iterator + slot
                utils.distributed.spin_lock_ld_lt_relaxed_wait(
                    lock_ptr=exit_flag, expected_val=Int32(world), scope="gpu"
                )
                cute.arch.atomic_add(exit_flag.llvm_ptr, Int32(-world), sem="relaxed", scope="gpu")


class AllReduce(ReduceScatter):
    """GEMM + all-reduce over `world` ranks: the reduce-scatter whose reduced stripes are broadcast into
    every rank's symmetric (m, n) out."""

    broadcast = True


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


def _make_wire(m, n, torch_dtype, tile_m, tile_n, ctas):
    """(d, d_mc, flags, flags_mc, peer_flags, exit_flags, exit_flags_mc): the symmetric D the GEMM stores
    its partials into, the tile flags with their multicast view and every rank's flags address, and the
    exit slots (ctas = the persistent grid's CTA count)."""
    import torch

    assert m % tile_m == 0 and n % tile_n == 0, "full tiles"
    d, d_mc, _ = _symm((m, n), torch_dtype)
    flags, flags_mc, flags_hdl = _symm((m // tile_m, n // tile_n), torch.int32, zero=True)
    peer_flags = torch.tensor(flags_hdl.buffer_ptrs, dtype=torch.int64, device="cuda")
    exit_flags, exit_flags_mc, _ = _symm((ctas,), torch.int32, zero=True)
    return d, d_mc, flags, flags_mc, peer_flags, exit_flags, exit_flags_mc


class ReduceScatterRunner(CommRunner):
    """GEMM + RS for an (m, n) output: wire = the symmetric D, out = this rank's (m // world, n) slab.
    mod(a, b, config=cfg, comm=runner)["D"] is the slab; direct: mD=runner.wire, comm_args=runner.args()."""

    def __init__(self, m, n, torch_dtype, tile_m, tile_n, ctas):
        import torch
        import torch.distributed as dist

        rank, world = dist.get_rank(), dist.get_world_size()
        assert m % (world * tile_m) == 0, "RS: a whole number of tiles per slab"
        self.comm = ReduceScatter(rank, world)
        self.wire, d_mc, flags, _, *rest = _make_wire(m, n, torch_dtype, tile_m, tile_n, ctas)
        self.out = torch.empty(m // world, n, dtype=torch_dtype, device="cuda")
        self._args = ReduceScatterArguments(d_mc, self.out, flags, *rest)
        torch.cuda.synchronize()


class AllReduceRunner(CommRunner):
    """GEMM + two-shot AR: wire = the symmetric D, out = this rank's copy of the full (m, n) result."""

    def __init__(self, m, n, torch_dtype, tile_m, tile_n, ctas):
        import torch
        import torch.distributed as dist

        rank, world = dist.get_rank(), dist.get_world_size()
        self.comm = AllReduce(rank, world)
        self.wire, d_mc, flags, flags_mc, _, *rest = _make_wire(
            m, n, torch_dtype, tile_m, tile_n, ctas
        )
        self.out, out_mc, _ = _symm((m, n), torch_dtype)
        self._args = AllReduceArguments(d_mc, out_mc, flags, flags_mc, *rest)
        torch.cuda.synchronize()
