# GEMM + all-reduce, one-shot (cutlass Lamport), as the GEMM's comm object: no flags, the data is the flag.
# D is the multicast view of 3 slots x world mirrors of (m, n), stacked along M; slot k = launch counter % 3.
# Every rank's epilogue warps store their -0.0-scrubbed raw partial of each tile into slot k's mirror `rank`
# (one TMA multicast store lands on every rank, gemm_base.split_rank_partial_commit through this operator's
# store_coord / store_fixup), then re-arm last call's slot (k - 1) over that tile in every local mirror:
# peers write that slot again only in call i + 2, after this rank's call i + 1 data, so nothing orders the
# re-arm. The comm warps poll their local slot-k mirrors per 16 B atom until no word is the sentinel, sum in
# f32, run the GEMM's epilogue on the whole tile and store this rank's plain out.
# Usage: lp = LamportRunner(m, n, dtype, tile_m); mod(a, b, config=cfg, comm=lp)["D"] is the result (the
# runner advances the slot, supplies the wire as D and the comm_args). Direct: GemmSm100(..., comm=lp.comm);
# lp.before_launch(); launch with mD=lp.wire, comm_args=lp.args(); lp.out is the result.
import math
from functools import partial
from typing import NamedTuple

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass import Int32, const_expr
from cutlass.cutlass_dsl import any_, while_generate, yield_out
from quack.compile_utils import make_fake_tensor
from quack.cute_dsl_utils import mlir_namedtuple
from quack.distributed.ptx import ld_relaxed_gpu_global_4xu32, st_weak_global_4xu32
from quack.distributed.reduce_scatter import COMM_WARPS, SUBTILE_ATOMS, _symm, commit_subtile_local
from quack.gemm_base import Comm, CommRunner, EpiFrame

SLOTS = 3
LAMPORT_BATCH = 8  # 16-byte atoms in flight per sentinel wait (1 = serial spin)


@mlir_namedtuple
class LamportArguments(NamedTuple):
    """The comm warps' bundle (the GEMM's comm_args)."""

    slots: cute.Tensor  # this rank's mirrors (SLOTS * world * m, n), slot-major then mirror
    out: cute.Tensor  # this rank's (m, n) result
    slot: cute.Tensor  # (1,) int32 launch counter; the host adds 1 before every launch


def _fake_arguments(dtype):
    def mat(dt):
        return make_fake_tensor(dt, (cute.sym_int(), cute.sym_int()), leading_dim=1, divisibility=8)

    return LamportArguments(
        mat(dtype),
        mat(dtype),
        make_fake_tensor(Int32, (cute.sym_int(),), leading_dim=0, divisibility=1),
    )


def sentinel_word(dtype) -> int:
    """-0.0 in every lane of one 32-bit word, as a signed int32 (16-bit dtypes pack two lanes)."""
    word = 0x80008000 if dtype.width == 16 else 0x80000000
    return word - (1 << 32)


@cute.jit
def _slot(counter: cute.Tensor) -> Int32:
    """This launch's slot from the device-resident counter (graph-capturable: nothing host-baked)."""
    return cute.arch.load(counter.iterator.llvm_ptr, Int32, sem="relaxed", scope="gpu") % SLOTS


@cute.jit
def lamport_rearm_tile(frg_clear: tuple) -> None:
    """Store the sentinel over this tile in each fragment (one per local mirror)."""
    neg_zero = sentinel_word(frg_clear[0].element_type)
    _atom, loop_m, loop_n = frg_clear[0].shape
    for r in cutlass.range_constexpr(len(frg_clear)):
        for i in cutlass.range(loop_m, unroll=1):
            for j in cutlass.range(loop_n, unroll=1):
                st_weak_global_4xu32(
                    frg_clear[r][None, i, j].iterator, neg_zero, neg_zero, neg_zero, neg_zero
                )


@cute.jit
def lamport_reduce_subtile(
    frg_in: tuple,
    tRS_rD: cute.Tensor,
    epi_coord: cute.Coord,
    no_release: cutlass.Constexpr[bool] = False,
    batch: cutlass.Constexpr[int] = LAMPORT_BATCH,
) -> None:
    """gemm.epilogue's load_acc_subtile: batched poll over the (atom, source mirror) lattice, atom-major:
    issue ``batch`` 16 B loads back to back, spin reloading them until no last word is the sentinel (the
    multicast TMA store lands atomically per 16 B, so the probe that sees data carries the whole atom),
    then sum the landed atoms in tRS_rD's dtype."""
    _atom, chunk, sub_loop_n = tRS_rD.shape
    ws_dtype = frg_in[0].element_type
    neg_zero = sentinel_word(ws_dtype)
    num_peers = len(frg_in)
    tmp = cute.make_rmem_tensor((4, chunk, sub_loop_n), cutlass.Int32)
    tmp_ws = cute.recast_tensor(tmp, ws_dtype)
    # flat accumulator in tmp_ws's profile: only the final store crosses into tRS_rD's atom profile
    acc = cute.make_rmem_tensor(tmp_ws.shape, tRS_rD.element_type)
    L = chunk * sub_loop_n * num_peers
    W = batch
    for g in cutlass.range_constexpr((L + W - 1) // W):
        q0, q1 = g * W, min(L, (g + 1) * W)
        n = q1 - q0
        ptrs = []
        state = []
        for q in cutlass.range_constexpr(q0, q1):
            c, r = q // num_peers, q % num_peers
            ii, jj = c // sub_loop_n, c % sub_loop_n
            i, j = epi_coord[0] * chunk + ii, epi_coord[1] * sub_loop_n + jj
            ptr = frg_in[r][None, i, j].iterator
            ptrs.append(ptr)
            state.extend(ld_relaxed_gpu_global_4xu32(ptr))
        spin = while_generate(state, lambda *rg: any_(rg[4 * k + 3] == neg_zero for k in range(n)))
        with spin:
            nxt = []
            for k in cutlass.range_constexpr(n):
                nxt.extend(ld_relaxed_gpu_global_4xu32(ptrs[k]))
            yield_out(nxt)
        res = spin.results
        for k in cutlass.range_constexpr(n):
            c, r = (q0 + k) // num_peers, (q0 + k) % num_peers
            ii, jj = c // sub_loop_n, c % sub_loop_n
            tmp[0, ii, jj] = res[4 * k]
            tmp[1, ii, jj] = res[4 * k + 1]
            tmp[2, ii, jj] = res[4 * k + 2]
            tmp[3, ii, jj] = res[4 * k + 3]
            vals = tmp_ws[None, ii, jj].load().to(tRS_rD.element_type)
            if const_expr(r == 0):
                acc[None, ii, jj].store(vals)
            else:
                acc[None, ii, jj].store(acc[None, ii, jj].load() + vals)
    tRS_rD.store(acc.load())


def _frame(dtype, rows, cols, threads):
    """Thread layout over a (rows, cols) frame, 16-byte atoms n fastest, ~SUBTILE_ATOMS atoms in flight per
    thread per subtile: (thr_shape, val_shape, chunk)."""
    atom = 128 // dtype.width
    thr_n = math.gcd(cols // atom, threads)
    thr_m = threads // thr_n
    assert rows % thr_m == 0 and cols % (atom * thr_n) == 0, "the frame tiles over the threads"
    loop_m, loop_n = rows // thr_m, cols // (atom * thr_n)
    chunk = math.gcd(loop_m, max(1, SUBTILE_ATOMS // loop_n))
    return (thr_m, thr_n), (1, atom), chunk


def _tiled_copy(dtype, thr_shape, val_shape):
    return cute.make_tiled_copy_tv(
        cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), dtype),
        cute.make_layout(thr_shape, stride=(thr_shape[1], 1)),
        cute.make_layout(val_shape, stride=(val_shape[1], 1)),
    )


class AllReduceLamport(Comm):
    """GEMM + one-shot all-reduce over `world` ranks, d_dtype wire; the whole tile is this CTA's."""

    num_warps = COMM_WARPS
    regs = None
    split_rank = True

    def __init__(self, rank: int, world: int):
        self.rank, self.world = rank, world

    def fake_args(self, d_dtype):
        return _fake_arguments(d_dtype)

    def wire_shape(self, shape):
        m, n = shape
        return (SLOTS * self.world * m, n)

    def epi_frame(self, gemm):
        cta_m, cta_n = gemm.cta_tile_shape_mnk[0], gemm.cta_tile_shape_mnk[1]
        threads = gemm.num_comm_warps * cute.arch.WARP_SIZE
        (thr_m, thr_n), _, chunk = _frame(gemm.d_dtype, cta_m, cta_n, threads)
        warps_n = max(1, thr_n // cute.arch.WARP_SIZE)
        return EpiFrame(
            (cta_m, cta_n), (chunk * thr_m, cta_n), (gemm.num_comm_warps // warps_n, warps_n, 1)
        )

    @cute.jit
    def store_coord(self, gemm, ctx, tile_coord_mnkl):
        """Slot k, mirror `rank`: the tile's row in the stacked mirrors."""
        comm_args = ctx.args
        tiles_m = comm_args.out.shape[0] // gemm.cta_tile_shape_mnk[0]
        row = (_slot(comm_args.slot) * self.world + self.rank) * tiles_m + tile_coord_mnkl[0]
        return (row, tile_coord_mnkl[1], tile_coord_mnkl[2], tile_coord_mnkl[3])

    @cute.jit
    def store_fixup(self, frg):
        """x + 0.0 is bit-exact except -0.0 -> +0.0: scrub the sentinel from the partial."""
        vec = frg.load()
        frg.store(vec + cute.full_like(vec, 0.0))

    @cute.jit
    def signal_tile(self, gemm, ctx, tile_coord_mnkl, epi_store_pipeline, is_tma_warp, tidx: Int32):
        """Re-arm last call's slot over this tile in every local mirror, spread over the epilogue warps."""
        comm_args = ctx.args
        cta_m, cta_n = gemm.cta_tile_shape_mnk[0], gemm.cta_tile_shape_mnk[1]
        dtype = comm_args.slots.element_type
        threads = gemm.num_epi_warps * cute.arch.WARP_SIZE
        thr_shape, val_shape, _ = _frame(dtype, cta_m, cta_n, threads)
        thr_copy = _tiled_copy(dtype, thr_shape, val_shape).get_slice(tidx)
        gSlots = cute.local_tile(comm_args.slots, (cta_m, cta_n), (None, None))
        tiles_m = comm_args.out.shape[0] // cta_m
        k = (_slot(comm_args.slot) + (SLOTS - 1)) % SLOTS
        pid_m, pid_n = tile_coord_mnkl[0], tile_coord_mnkl[1]
        rows = [(k * self.world + r) * tiles_m + pid_m for r in range(self.world)]
        lamport_rearm_tile(
            tuple(thr_copy.partition_S(gSlots[None, None, row, pid_n]) for row in rows)
        )

    @cute.jit
    def comm_warps(self, gemm, ctx, warp_idx: Int32, tidx: Int32):
        assert gemm.split_k == 1, "lamport all-reduce: no split-K"
        world = self.world
        # bound methods and ctx fields cross into the loop as locals; the objects cannot
        epilogue, comm_barrier = gemm.epilogue, gemm.comm_barrier
        comm_args, TileSchedulerCls, varlen_manager = ctx.args, ctx.TileSchedulerCls, ctx.varlen
        epilogue_params, epi_smem_tensors = ctx.epi_params, ctx.epi_smem_tensors
        epi_pipeline, sC = ctx.epi_pipeline, ctx.sC
        cta_m, cta_n = gemm.cta_tile_shape_mnk[0], gemm.cta_tile_shape_mnk[1]
        slots, out = comm_args.slots, comm_args.out
        thr_shape, val_shape, chunk = _frame(
            slots.element_type, cta_m, cta_n, gemm.num_comm_warps * cute.arch.WARP_SIZE
        )
        tiled_copy = _tiled_copy(slots.element_type, thr_shape, val_shape)
        comm_tidx = tidx - gemm.comm_warp_ids[0] * cute.arch.WARP_SIZE
        thr_copy = tiled_copy.get_slice(comm_tidx)
        gSlots = cute.local_tile(slots, (cta_m, cta_n), (None, None))
        gOut = cute.local_tile(out, (cta_m, cta_n), (None, None))
        _atom, _, loop_n = thr_copy.partition_S(cute.make_identity_tensor((cta_m, cta_n))).shape
        epi_tile = (chunk * thr_shape[0], cta_n)  # the epilogue's subtile on the tile
        tRS_rD = cute.make_rmem_tensor((_atom, chunk, loop_n), gemm.acc_dtype)
        # C: staged by the epi-load warp per subtile through the GEMM's C pipeline, same thread copy
        tRS_rC, tSR_rC, tSR_sC = None, None, None
        if const_expr(sC is not None):
            tRS_rC = cute.make_rmem_tensor((_atom, chunk, loop_n), gemm.c_dtype)
            tSR_rC = tiled_copy.retile(tRS_rC)
            tSR_sC = thr_copy.partition_S(sC)
        epi_read_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, gemm.epi_c_stage
        )
        is_tma_warp = warp_idx == gemm.comm_warp_ids[0]
        tiles_m = out.shape[0] // cta_m
        k = _slot(comm_args.slot)

        # this CTA's static schedule, stepped locally (never the scheduler pipeline)
        tile_sched = TileSchedulerCls()
        work_tile = tile_sched.initial_work_tile_info()
        while work_tile.is_valid_tile:
            pid_m, pid_n = work_tile.tile_idx[0], work_tile.tile_idx[1]
            mirrors = tuple(
                thr_copy.partition_S(gSlots[None, None, (k * world + r) * tiles_m + pid_m, pid_n])
                for r in range(world)
            )
            epi_read_state, _ = epilogue(
                epilogue_params,
                epi_smem_tensors,
                epi_pipeline,
                None,  # epi_store_pipeline
                epi_read_state,
                None,  # epi_producer_state
                epi_tile,
                partial(lamport_reduce_subtile, mirrors),
                tRS_rD=tRS_rD,
                tRS_rC=tRS_rC,
                tiled_copy_t2r=None,
                tiled_copy_r2s=tiled_copy,
                tRS_sD=None,
                tiled_copy_s2r=tiled_copy,
                tSR_rC=tSR_rC,
                tSR_sC=tSR_sC,
                copy_D=None,
                copy_C=None,
                tile_coord_mnkl=(pid_m, pid_n, Int32(0), Int32(0)),
                varlen_manager=varlen_manager,
                epilogue_barrier=comm_barrier,
                tile_scheduler=tile_sched,
                tidx=comm_tidx,
                is_tma_warp=is_tma_warp,
                commit_D=partial(
                    commit_subtile_local, thr_copy.partition_S(gOut[None, None, pid_m, pid_n])
                ),
                tile_shape_mn=(cta_m, cta_n),
            )
            tile_sched._current_work_idx = tile_sched._fetch_next_work_idx()
            work_tile = tile_sched._delinearize_work_idx(tile_sched._current_work_idx)
        # no exit barrier: a peer never reads this rank's memory after its own kernel ends


# ---- host ----


class LamportRunner(CommRunner):
    """GEMM + one-shot AR for an (m, n) output: wire = the mirrors' multicast view, out = this rank's full
    result; every launch advances the slot. mod(a, b, config=cfg, comm=runner)["D"] is out; direct:
    runner.before_launch(); mD=runner.wire, comm_args=runner.args()."""

    def __init__(self, m, n, torch_dtype, tile_m):
        import torch
        import torch.distributed as dist

        rank, world = dist.get_rank(), dist.get_world_size()
        assert m % tile_m == 0, "whole tiles along M"
        self.comm = AllReduceLamport(rank, world)
        slots, self.wire, _ = _symm(self.comm.wire_shape((m, n)), torch_dtype)
        slots.fill_(-0.0)
        self.out = torch.empty(m, n, dtype=torch_dtype, device="cuda")
        self.slot = torch.zeros(1, dtype=torch.int32, device="cuda")
        self._args = LamportArguments(slots, self.out, self.slot)
        torch.cuda.synchronize()
        dist.barrier()  # every rank's mirrors are armed before any peer stores into them

    def before_launch(self):
        self.slot.add_(1)
