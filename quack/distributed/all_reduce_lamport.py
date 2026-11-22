# GEMM + all-reduce, one-shot (cutlass Lamport), as the GEMM's distributed operator: no flags, the data is the flag.
# The partials buffer is 3 slots x world mirrors of (m, n), stacked along M, stored through its multicast
# view; slot k = launch counter % 3.
# Every rank's epilogue warps store their -0.0-scrubbed raw partial of each tile into slot k's mirror `rank`
# (one TMA multicast store lands on every rank, the GEMM's partial-only epilogue through this operator's
# store_tile / scrub_partial), then re-arm last call's slot (k - 1) over that tile in every local mirror:
# peers write that slot again only in call i + 2, after this rank's call i + 1 data, so nothing orders the
# re-arm. The comm warps poll their local slot-k mirrors per 16 B atom until no word is the sentinel, sum in
# f32, run the GEMM's epilogue on the whole tile and commit it into D, this rank's plain (m, n) output.
# Usage: d, args = make_lamport_buffers(m, n, dtype, tile_m); args.slot.add_(1) before every launch;
# mod(a, b, out={"D": d}, config=cfg, dist_op=AllReduceLamport(rank, world), dist_op_args=args).
# Direct: GemmSm100(..., dist_op=AllReduceLamport(rank, world)), launch with mD=d, dist_op_args=args.
from functools import partial
from typing import NamedTuple

import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr
from cutlass.cutlass_dsl import any_, while_generate, yield_out
from quack.compile_utils import make_fake_tensor
from quack.cute_dsl_utils import mlir_namedtuple
from quack.distributed.ptx import ld_relaxed_gpu_global_4xu32, st_weak_global_4xu32
from quack.distributed.reduce_scatter import _frame, _symm, commit_subtile_local, load_subtile
from quack.gemm_base import DistOp

SLOTS = 3


@mlir_namedtuple
class LamportArguments(NamedTuple):
    """The comm warps' bundle (the GEMM's dist_op_args)."""

    slots: cute.Tensor  # this rank's mirrors (SLOTS * world * m, n), slot-major then mirror
    slots_mc: cute.Tensor  # their multicast view: the GEMM's partial store lands in every rank's mirror
    slot: cute.Tensor  # (1,) int32 launch counter; the host adds 1 before every launch


def _fake_arguments():
    def mat(dt):
        return make_fake_tensor(dt, (cute.sym_int(), cute.sym_int()), leading_dim=1, divisibility=8)

    return LamportArguments(
        mat(AllReduceLamport.partial_dtype),
        mat(AllReduceLamport.partial_dtype),
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
    batch: cutlass.Constexpr[int] = 8,  # 16-byte atoms in flight per sentinel wait (1 = serial spin)
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


def _tiled_copy(dtype, thr_shape, val_shape):
    return cute.make_tiled_copy_tv(
        cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), dtype),
        cute.make_layout(thr_shape, stride=(thr_shape[1], 1)),
        cute.make_layout(val_shape, stride=(val_shape[1], 1)),
    )


class AllReduceLamport(DistOp):
    """GEMM + one-shot all-reduce over `world` ranks: D = the full (m, n) result; the whole tile is this
    CTA's."""

    num_warps = 4
    regs = None
    runs_evt = True
    partial_dtype = cutlass.BFloat16  # the wire

    def __init__(self, rank: int, world: int):
        self.rank, self.world = rank, world

    def fake_args(self, d_dtype):
        return _fake_arguments()

    def partial_tensor(self, dist_op_args):
        return dist_op_args.slots_mc

    def epi_warp_shape_mnk(self, gemm):
        cta_m, cta_n = gemm.cta_tile_shape_mnk[0], gemm.cta_tile_shape_mnk[1]
        threads = gemm.num_comm_warps * cute.arch.WARP_SIZE
        (_, thr_n), _, _ = _frame(self.partial_dtype, cta_m, cta_n, threads)
        warps_n = max(1, thr_n // cute.arch.WARP_SIZE)
        return (gemm.num_comm_warps // warps_n, warps_n, 1)

    def epi_tile(self, gemm):
        cta_m, cta_n = gemm.cta_tile_shape_mnk[0], gemm.cta_tile_shape_mnk[1]
        threads = gemm.num_comm_warps * cute.arch.WARP_SIZE
        (thr_m, _), _, chunk = _frame(self.partial_dtype, cta_m, cta_n, threads)
        return (chunk * thr_m, cta_n)

    def store_tile(self, gemm, ctx, sD, epi_tile, tidx, tile_coord_mnkl, batch_idx, varlen_manager):
        """The GEMM's TMA store into slot k, mirror `rank`: the tile's row in the stacked mirrors."""
        dist_op_args = ctx.args
        tiles_m = ctx.mD.shape[0] // gemm.cta_tile_shape_mnk[0]
        row = (_slot(dist_op_args.slot) * self.world + self.rank) * tiles_m + tile_coord_mnkl[0]
        return (row, tile_coord_mnkl[1], tile_coord_mnkl[2], tile_coord_mnkl[3]), None

    @cute.jit
    def scrub_partial(self, frg):
        """x + 0.0 is bit-exact except -0.0 -> +0.0: scrub the sentinel from the partial."""
        vec = frg.load()
        frg.store(vec + cute.full_like(vec, 0.0))

    @cute.jit
    def signal_tile(self, gemm, ctx, tile_coord_mnkl, epi_store_pipeline, is_tma_warp, tidx: Int32):
        """Re-arm last call's slot over this tile in every local mirror, spread over the epilogue warps."""
        dist_op_args = ctx.args
        cta_m, cta_n = gemm.cta_tile_shape_mnk[0], gemm.cta_tile_shape_mnk[1]
        dtype = dist_op_args.slots.element_type
        threads = gemm.num_epi_warps * cute.arch.WARP_SIZE
        thr_shape, val_shape, _ = _frame(dtype, cta_m, cta_n, threads)
        thr_copy = _tiled_copy(dtype, thr_shape, val_shape).get_slice(tidx)
        gSlots = cute.local_tile(dist_op_args.slots, (cta_m, cta_n), (None, None))
        tiles_m = ctx.mD.shape[0] // cta_m
        k = (_slot(dist_op_args.slot) + (SLOTS - 1)) % SLOTS
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
        dist_op_args, TileSchedulerCls, varlen_manager = ctx.args, ctx.TileSchedulerCls, ctx.varlen
        epilogue_params, epi_smem_tensors, mC = ctx.epi_params, ctx.epi_smem_tensors, ctx.mC
        cta_m, cta_n = gemm.cta_tile_shape_mnk[0], gemm.cta_tile_shape_mnk[1]
        slots, out = dist_op_args.slots, ctx.mD
        thr_shape, val_shape, chunk = _frame(
            slots.element_type, cta_m, cta_n, gemm.num_comm_warps * cute.arch.WARP_SIZE
        )
        tiled_copy = _tiled_copy(slots.element_type, thr_shape, val_shape)
        comm_tidx = tidx - gemm.comm_warp_ids[0] * cute.arch.WARP_SIZE
        thr_copy = tiled_copy.get_slice(comm_tidx)
        gSlots = cute.local_tile(slots, (cta_m, cta_n), (None, None))
        gOut = cute.local_tile(out, (cta_m, cta_n), (None, None))
        _atom, _, loop_n = thr_copy.partition_S(cute.make_identity_tensor((cta_m, cta_n))).shape
        epi_tile = self.epi_tile(gemm)  # the epilogue's subtile on the tile
        tRS_rD = cute.make_rmem_tensor((_atom, chunk, loop_n), gemm.acc_dtype)
        # C: read per subtile straight from gmem, the same thread copy as the mirrors
        tRS_rC, gC = None, None
        if const_expr(mC is not None):
            tRS_rC = cute.make_rmem_tensor((_atom, chunk, loop_n), mC.element_type)
            gC = cute.local_tile(mC, (cta_m, cta_n), (None, None))
        is_tma_warp = warp_idx == gemm.comm_warp_ids[0]
        epi_store_pipeline = gemm.make_epi_store_pipeline(gemm.num_comm_warps)  # aux outputs' TMA stores
        tiles_m = out.shape[0] // cta_m
        k = _slot(dist_op_args.slot)

        # this CTA's static schedule, stepped locally (never the scheduler pipeline)
        tile_sched = TileSchedulerCls()
        work_tile = tile_sched.initial_work_tile_info()
        while work_tile.is_valid_tile:
            pid_m, pid_n = work_tile.tile_idx[0], work_tile.tile_idx[1]
            mirrors = tuple(
                thr_copy.partition_S(gSlots[None, None, (k * world + r) * tiles_m + pid_m, pid_n])
                for r in range(world)
            )
            load_C = None
            if const_expr(mC is not None):
                load_C = partial(load_subtile, thr_copy.partition_S(gC[None, None, pid_m, pid_n]))
            epilogue(
                epilogue_params,
                epi_smem_tensors,
                None,  # epi_pipeline: C is read directly
                epi_store_pipeline,
                None,  # epi_read_state
                None,  # epi_producer_state
                epi_tile,
                partial(lamport_reduce_subtile, mirrors),
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
                tile_coord_mnkl=(pid_m, pid_n, Int32(0), Int32(0)),
                varlen_manager=varlen_manager,
                epilogue_barrier=comm_barrier,
                tile_scheduler=tile_sched,
                tidx=comm_tidx,
                is_tma_warp=is_tma_warp,
                commit_D=partial(
                    commit_subtile_local, thr_copy.partition_S(gOut[None, None, pid_m, pid_n])
                ),
                load_C=load_C,
                tile_shape_mn=(cta_m, cta_n),
            )
            tile_sched._current_work_idx = tile_sched._fetch_next_work_idx()
            work_tile = tile_sched._delinearize_work_idx(tile_sched._current_work_idx)
        if is_tma_warp:
            epi_store_pipeline.producer_tail()
        # no exit barrier: a peer never reads this rank's memory after its own kernel ends


# ---- host ----


def make_lamport_buffers(m, n, torch_dtype, tile_m):
    """(d, args) for GEMM + one-shot AR of an (m, n) output: D = this rank's full (m, n) result, args =
    the slot mirrors (partial_dtype) with the launch counter: args.slot.add_(1) before every launch. The
    operator is AllReduceLamport(rank, world)."""
    import torch
    import torch.distributed as dist

    from quack.cute_dsl_utils import torch2cute_dtype_map

    world = dist.get_world_size()
    assert m % tile_m == 0, "whole tiles along M"
    wire = next(t for t, c in torch2cute_dtype_map.items() if c is AllReduceLamport.partial_dtype)
    slots, slots_mc, _ = _symm((SLOTS * world * m, n), wire)
    slots.fill_(-0.0)
    cols = n // 2 if torch_dtype == torch.float4_e2m1fn_x2 else n  # fp4 D packs two values a byte
    d = torch.empty(m, cols, dtype=torch_dtype, device="cuda")
    slot = torch.zeros(1, dtype=torch.int32, device="cuda")
    torch.cuda.synchronize()
    dist.barrier()  # every rank's mirrors are armed before any peer stores into them
    return d, LamportArguments(slots, slots_mc, slot)
