# MoE GEMM2 + combine as the GEMM's distributed operator: no gate, no publish, no comm warps. The swap-AB grouped
# GEMM's D columns are tokens (varlen_n, D = y.T (hidden, total_n) m-major, columns packed per expert); the operator
# drains each staged D subtile straight into the origin rank's slot for that (token, k) pair, 16 B per lane along
# hidden (store_tile's own copy: the GEMM stages accumulator -> registers -> epi ops -> smem as always, the TMA store
# is replaced). D is never written, nothing is waited on; completion is the kernel's end, then the caller barriers
# and reduces every token's K slots (quack.distributed.reduce_over_k). Slot id of token column c = row_key[c] = rank *
# slots_per_rank + local. Usage: d, slots, args = make_combine_buffers(tokens_local, k, hidden, dtype, group,
# max_rows); per call gemm(w2, act, d.T, ..., cu_seqlens_n=cu, dist_op=CombineStore(hidden),
# dist_op_args=args(row_key)); then barrier; the landed rows are slots (tokens_local, k, hidden).
from functools import partial
from typing import NamedTuple

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64
from quack.compile_utils import make_fake_tensor
from quack.cute_dsl_utils import mlir_namedtuple
from quack.distributed.ptx import st_weak_global_4xu32_at
from quack.gemm_base import DistOp


@mlir_namedtuple
class CombineArguments(NamedTuple):
    """The drain's bundle (the GEMM's dist_op_args)."""

    slot_base: cute.Tensor  # (world,) int64: every rank's slot buffer base address
    slots_per_rank: Int32  # tokens_local * k
    row_key: cute.Tensor  # (total_n,) int32: each token column's global slot id


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


class CombineStore(DistOp):
    """GEMM2 + combine: the epilogue's token columns land in their origin ranks' slots (hidden = D's M extent)."""

    ragged_dim = "n"  # D columns are the expert-grouped tokens (the swap-AB GEMM, cu_seqlens_n)

    def __init__(self, hidden: int):
        self.hidden = hidden

    def fake_args(self, d_dtype):
        return CombineArguments(
            make_fake_tensor(Int64, (cute.sym_int(),), leading_dim=0, divisibility=1),
            Int32(0),
            make_fake_tensor(Int32, (cute.sym_int(),), leading_dim=0, divisibility=1),
        )

    def store_tile(self, gemm, ctx, sD, epi_tile, tidx, tile_coord_mnkl, batch_idx, varlen_manager):
        """The operator's own copy. Thread -> (lane along M, token column along N), 16 B per lane over the
        m-major D stage: a token's subtile slice of epi_m elements is epi_m * width / 128 lanes,
        num_epi_warps * 32 / lanes tokens per step."""
        width = gemm.d_dtype.width
        epi_m = cute.size(epi_tile[0])
        lanes = epi_m * width // 128
        assert lanes * 128 == epi_m * width, "a token's subtile slice is whole 16 B lanes"
        elems = 128 // width
        tokens_per_step = gemm.num_epi_warps * cute.arch.WARP_SIZE // lanes
        assert cute.size(epi_tile[1]) % tokens_per_step == 0, "the subtile's columns are whole steps"
        op = cute.nvgpu.CopyUniversalOp()
        atom = cute.make_copy_atom(op, gemm.d_dtype, num_bits_per_copy=128)
        thr = cute.make_layout((lanes, tokens_per_step), stride=(1, lanes))
        val = cute.make_layout((elems, 1), stride=(1, elems))
        tiled_copy = cute.make_tiled_copy_tv(atom, thr, val)
        tSR_s = tiled_copy.get_slice(tidx).partition_S(sD)
        tSR_r = cute.make_rmem_tensor(tSR_s[None, None, None, 0].shape, gemm.d_dtype)
        return None, partial(
            store_subtile,
            tiled_copy,
            tSR_s,
            tSR_r,
            epi_tile,
            gemm.cta_tile_shape_mnk[:2],
            ctx.args,
            Int32(self.hidden * width // 8),
            lanes,
            tokens_per_step,
            gemm.varlen_n_granule,
            tidx,
            varlen_manager,
            tile_coord_mnkl,
            batch_idx,
        )


class CombineRunner:
    """GEMM2 + combine of tokens_local tokens x k experts x hidden: `d` = a (max_rows, hidden) D the drain never
    writes (the kernel's D shape), `slots` = the symmetric slots every peer's GEMM2 lands rows in (reduce over
    k after a cross-rank barrier: reduce_over_k), `op` = CombineStore(hidden); prepare(row_key) = the GEMM's
    dist_op_args for a call's routing (each D row's global slot id)."""

    def __init__(self, tokens_local, k, hidden, torch_dtype, group, max_rows):
        import torch
        import torch.distributed._symmetric_memory as symm_mem

        dev = torch.cuda.current_device()
        self.slots = symm_mem.empty(tokens_local, k, hidden, dtype=torch_dtype, device=dev)
        hdl = symm_mem.rendezvous(self.slots, group=group)
        shape = (tokens_local, k, hidden)
        bases = [hdl.get_buffer(q, shape, torch_dtype).data_ptr() for q in range(group.size())]
        self._args = partial(
            CombineArguments, torch.tensor(bases, dtype=torch.int64, device=dev), tokens_local * k
        )
        self.d = torch.empty(max_rows, hidden, dtype=torch_dtype, device=dev)
        self.op = CombineStore(hidden)

    def prepare(self, row_key):
        return self._args(row_key)
