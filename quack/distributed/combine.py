# MoE GEMM2 + combine as the GEMM's distributed operator: no gate, no publish, no comm warps. The grouped GEMM's
# tokens are D's columns (swap-AB, varlen_n: D = y.T (hidden, total_n) m-major) or D's rows (grouped M, varlen_m:
# D = y (total_m, hidden) n-major), packed per expert either way; the operator drains each staged D subtile straight
# into the origin rank's slot for that (token, k) pair, 16 B per lane along hidden (store_tile's own copy: the GEMM
# stages accumulator -> registers -> epi ops -> smem as always, the TMA store is replaced). D is never written,
# nothing is waited on; completion is the kernel's end, then the caller barriers and reduces every token's K slots
# (quack.distributed.reduce_over_k). Slot id of token t = row_key[t] = rank * slots_per_rank + local. Usage: runner =
# CombineRunner(tokens_local, k, hidden, dtype, group, max_rows); per call gemm(w2, act, runner.D, ...,
# cu_seqlens_n=cu, dist_op=runner.op, dist_op_args=runner.prepare(row_key)) on SM100, gemm(act, w2, runner.D, ...,
# cu_seqlens_m=cu, ...) on SM90; then barrier; the landed rows are runner.slots (tokens_local, k, hidden).
from functools import partial
from typing import NamedTuple

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64
from quack.compile_utils import make_fake_tensor
from quack.cute_dsl_utils import get_device_capacity, mlir_namedtuple
from quack.distributed.ptx import st_weak_global_4xu32_at
from quack.gemm_base import DistOp, grouped_ragged_dim


@mlir_namedtuple
class CombineArguments(NamedTuple):
    """The drain's bundle (the GEMM's dist_op_args)."""

    slot_base: cute.Tensor  # (world,) int64: every rank's slot buffer base address
    slots_per_rank: Int32  # tokens_local * k
    row_key: cute.Tensor  # (total,) int32: each token's global slot id, in the GEMM's row order
    sem: cute.Tensor  # (1,) int32 the GEMM's tile counter (make_scheduler_args), zeroed per call


@cute.jit
def store_subtile(
    tiled_copy,
    tSR_s: cute.Tensor,
    tSR_r: cute.Tensor,
    epi_tile,
    cta_tile_mn,
    dist_op_args,
    row_bytes: Int32,
    lanes: cutlass.Constexpr[int],
    tokens_per_step: cutlass.Constexpr[int],
    granule: cutlass.Constexpr[int],
    tidx: Int32,
    varlen_manager,
    tile_coord_mnkl,
    batch_idx: Int32,
    src_idx,
    dst_idx,
):
    """The drain (the epilogue's copy_D, every epilogue thread): the subtile in smem stage src_idx at subtile
    coordinate dst_idx goes into the origins' slots, each token column's slice of epi_m hidden values at the same
    hidden offset in its slot row; columns past the block's rows are skipped. The slot addresses are fetched before
    the smem copy so their latency overlaps it (an address chain per step was ~45 % of the kernel's stalls)."""
    epi_m, epi_n = cute.size(epi_tile[0]), cute.size(epi_tile[1])
    elem_bytes = tSR_r.element_type.width // 8
    elems = 16 // elem_bytes
    STEPS = epi_n // tokens_per_step
    pid_n, tile_n = tile_coord_mnkl[1], cta_tile_mn[1]
    b = varlen_manager.block_n_rows(batch_idx, tile_n, granule)
    base_col = varlen_manager.params.cu_seqlens_n[batch_idx] + pid_n * b
    n_valid = varlen_manager.block_n_valid(batch_idx, pid_n, tile_n, granule)
    col0 = dst_idx[1] * epi_n + tidx // lanes
    slots_per_rank = dist_op_args.slots_per_rank
    addr = cute.make_rmem_tensor(cute.make_layout(STEPS), Int64)
    for step in cutlass.range_constexpr(STEPS):
        c = cutlass.min(col0 + step * tokens_per_step, n_valid - Int32(1))  # clamped: the load is unconditional
        slot = dist_op_args.row_key[base_col + c]
        addr[step] = dist_op_args.slot_base[slot // slots_per_rank] + Int64(slot % slots_per_rank) * Int64(row_bytes)
    h_off = Int64((tile_coord_mnkl[0] * cta_tile_mn[0] + dst_idx[0] * epi_m + (tidx % lanes) * elems) * elem_bytes)
    cute.copy(tiled_copy, tSR_s[None, None, None, src_idx], tSR_r)
    words = cute.recast_tensor(tSR_r, Int32)  # ((4, 1), 1, steps): the lane's 16 B per step
    for step in cutlass.range_constexpr(STEPS):
        if col0 + step * tokens_per_step < n_valid:
            w0, w1 = words[(0, 0), 0, step], words[(1, 0), 0, step]
            w2, w3 = words[(2, 0), 0, step], words[(3, 0), 0, step]
            st_weak_global_4xu32_at(addr[step] + h_off, w0, w1, w2, w3)


@cute.jit
def store_subtile_m(
    tiled_copy,
    tSR_s: cute.Tensor,
    tSR_r: cute.Tensor,
    epi_tile,
    cta_tile_mn,
    dist_op_args,
    row_bytes: Int32,
    lanes: cutlass.Constexpr[int],
    tokens_per_step: cutlass.Constexpr[int],
    tidx: Int32,
    varlen_manager,
    tile_coord_mnkl,
    batch_idx: Int32,
    src_idx,
    dst_idx,
):
    """store_subtile for the grouped-M GEMM: tokens are D's rows (n-major stage), each row's slice of epi_n
    hidden values goes to the same hidden offset in its slot row; rows past the expert's are skipped."""
    epi_m, epi_n = cute.size(epi_tile[0]), cute.size(epi_tile[1])
    elem_bytes = tSR_r.element_type.width // 8
    elems = 16 // elem_bytes
    STEPS = epi_m // tokens_per_step
    pid_m, tile_m = tile_coord_mnkl[0], cta_tile_mn[0]
    cu = varlen_manager.params.cu_seqlens_m
    base_row = cu[batch_idx] + pid_m * tile_m
    m_valid = cutlass.min(cu[batch_idx + 1] - cu[batch_idx] - pid_m * tile_m, tile_m)
    row0 = dst_idx[0] * epi_m + tidx // lanes
    slots_per_rank = dist_op_args.slots_per_rank
    addr = cute.make_rmem_tensor(cute.make_layout(STEPS), Int64)
    for step in cutlass.range_constexpr(STEPS):
        r = cutlass.min(row0 + step * tokens_per_step, m_valid - Int32(1))  # clamped: the load is unconditional
        slot = dist_op_args.row_key[base_row + r]
        addr[step] = dist_op_args.slot_base[slot // slots_per_rank] + Int64(slot % slots_per_rank) * Int64(row_bytes)
    h_off = Int64((tile_coord_mnkl[1] * cta_tile_mn[1] + dst_idx[1] * epi_n + (tidx % lanes) * elems) * elem_bytes)
    cute.copy(tiled_copy, tSR_s[None, None, None, src_idx], tSR_r)
    words = cute.recast_tensor(tSR_r, Int32)  # ((4, 1), steps, 1): the lane's 16 B per step
    for step in cutlass.range_constexpr(STEPS):
        if row0 + step * tokens_per_step < m_valid:
            w0, w1 = words[(0, 0), step, 0], words[(1, 0), step, 0]
            w2, w3 = words[(2, 0), step, 0], words[(3, 0), step, 0]
            st_weak_global_4xu32_at(addr[step] + h_off, w0, w1, w2, w3)


class CombineStore(DistOp):
    """GEMM2 + combine: the epilogue's tokens land in their origin ranks' slots. ragged_dim = the GEMM call's
    ragged dim: tokens are D's columns under "n" (cu_seqlens_n, hidden = D's M extent) or D's rows under "m"
    (cu_seqlens_m, hidden = D's N extent)."""

    def __init__(self, hidden: int, ragged_dim: str = "n"):
        assert ragged_dim in ("m", "n")
        self.ragged_dim = ragged_dim
        self.hidden = hidden

    def make_scheduler_args(self, dist_op_args):
        """The tile counter (as DispatchPull): the benched schedule for the grouped GEMM2."""
        return {"tile_count_semaphore": dist_op_args.sem}

    def fake_args(self, d_dtype):
        return CombineArguments(
            make_fake_tensor(Int64, (cute.sym_int(),), leading_dim=0, divisibility=1),
            Int32(0),
            make_fake_tensor(Int32, (cute.sym_int(),), leading_dim=0, divisibility=1),
            make_fake_tensor(Int32, (1,), leading_dim=0, divisibility=1),
        )

    def store_tile(self, gemm, ctx, sD, epi_tile, tidx, tile_coord_mnkl, batch_idx, varlen_manager):
        """The operator's own copy, 16 B per lane along hidden over the D stage: a token's subtile slice of
        hidden_sub elements is hidden_sub * width / 128 lanes, num_epi_warps * 32 / lanes tokens per step.
        Thread -> (lane along M, token column along N) under cu_seqlens_n, (token row along M, lane along N)
        under cu_seqlens_m."""
        width = gemm.d_dtype.width
        tokens_mode = 1 if gemm.varlen_n else 0  # the subtile mode that is tokens
        hidden_sub = cute.size(epi_tile[1 - tokens_mode])
        lanes = hidden_sub * width // 128
        assert lanes * 128 == hidden_sub * width, "a token's subtile slice is whole 16 B lanes"
        elems = 128 // width
        tokens_per_step = gemm.num_epi_warps * cute.arch.WARP_SIZE // lanes
        assert cute.size(epi_tile[tokens_mode]) % tokens_per_step == 0, "the subtile's tokens are whole steps"
        op = cute.nvgpu.CopyUniversalOp()
        atom = cute.make_copy_atom(op, gemm.d_dtype, num_bits_per_copy=128)
        if gemm.varlen_n:
            thr = cute.make_layout((lanes, tokens_per_step), stride=(1, lanes))
            val = cute.make_layout((elems, 1), stride=(1, elems))
        else:
            thr = cute.make_layout((tokens_per_step, lanes), stride=(lanes, 1))
            val = cute.make_layout((1, elems), stride=(elems, 1))
        tiled_copy = cute.make_tiled_copy_tv(atom, thr, val)
        tSR_s = tiled_copy.get_slice(tidx).partition_S(sD)
        tSR_r = cute.make_rmem_tensor(tSR_s[None, None, None, 0].shape, gemm.d_dtype)
        granule = (gemm.varlen_n_granule,) if gemm.varlen_n else ()
        return None, partial(
            store_subtile if gemm.varlen_n else store_subtile_m,
            tiled_copy,
            tSR_s,
            tSR_r,
            epi_tile,
            gemm.cta_tile_shape_mnk[:2],
            ctx.args,
            Int32(self.hidden * width // 8),
            lanes,
            tokens_per_step,
            *granule,
            tidx,
            varlen_manager,
            tile_coord_mnkl,
            batch_idx,
        )


class CombineRunner:
    """GEMM2 + combine of tokens_local tokens x k experts x hidden: `D` = the GEMM's D operand, a (max_rows,
    hidden) buffer the drain never writes, as (hidden, max_rows) m-major under "n" (swap-AB) or as is under "m"
    (varlen_m); `slots` = the symmetric slots every peer's GEMM2 lands rows in (reduce over k after a cross-rank
    barrier: reduce_over_k), `op` = CombineStore(hidden, ragged_dim); ragged_dim = the GEMM call's ragged dim,
    None = this arch's default (gemm_base.grouped_ragged_dim); prepare(row_key) = the GEMM's
    dist_op_args for a call's routing (each token's global slot id, in the GEMM's row order)."""

    def __init__(self, tokens_local, k, hidden, torch_dtype, group, max_rows, ragged_dim=None):
        import torch.distributed._symmetric_memory as symm_mem

        dev = torch.cuda.current_device()
        if ragged_dim is None:
            ragged_dim = grouped_ragged_dim(get_device_capacity(torch.device(dev))[0])
        self.slots = symm_mem.empty(tokens_local, k, hidden, dtype=torch_dtype, device=dev)
        hdl = symm_mem.rendezvous(self.slots, group=group)
        shape = (tokens_local, k, hidden)
        bases = [hdl.get_buffer(q, shape, torch_dtype).data_ptr() for q in range(group.size())]
        self._args = partial(
            CombineArguments, torch.tensor(bases, dtype=torch.int64, device=dev), tokens_local * k
        )
        self.op = CombineStore(hidden, ragged_dim)
        self.sem = torch.zeros(1, dtype=torch.int32, device=dev)  # the GEMM's tile counter
        d = torch.empty(max_rows, hidden, dtype=torch_dtype, device=dev)
        self.D = d.T if ragged_dim == "n" else d

    def prepare(self, row_key):
        self.sem.zero_()
        return self._args(row_key, self.sem)
