# MoE GEMM2 + combine as the GEMM's distributed operator: no gate, no publish, no comm warps. The grouped GEMM's
# D rows are tokens (varlen-M, rows packed per expert); the operator drains each staged D subtile straight into
# the origin rank's slot for that (token, k) pair, 16 B per lane, at the row's hidden offset (store_drain: the
# GEMM stages accumulator -> registers -> epi ops -> smem as always, the TMA store is replaced). D itself is
# never written, nothing is waited on; completion is the kernel's end, then the caller barriers and reduces
# every token's K slots (duckpond reduce_over_k). Slot id of row r = row_key[r] = rank * slots_per_rank + local.
# Usage: d, slots, args = make_combine_buffers(tokens_local, k, hidden, dtype, group, max_rows); per call
# GemmSm100(..., dist_op=CombineStore(hidden)), mD=d, dist_op_args=args(row_key); then barrier; the landed rows
# are slots (tokens_local, k, hidden).
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
    row_key: cute.Tensor  # (rows,) int32: each D row's global slot id


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
    tidx: Int32,
    varlen_manager,
    tile_coord_mnkl,
    batch_idx: Int32,
    src_idx,
    dst_idx,
):
    """The drain (the epilogue's copy_D, every epilogue thread): the subtile in smem stage src_idx at subtile
    coordinate dst_idx goes into the origins' slots, each token's slice at the same hidden offset in its slot
    row; rows past the batch are skipped."""
    epi_m, epi_n = cute.size(epi_tile[0]), cute.size(epi_tile[1])
    elem_bytes = tSR_r.element_type.width // 8
    cute.copy(tiled_copy, tSR_s[None, None, None, src_idx], tSR_r)
    words = cute.recast_tensor(tSR_r, Int32)  # ((4, 1), steps, 1): 16 B per step's row
    row0 = tile_coord_mnkl[0] * cta_tile_mn[0] + dst_idx[0] * epi_m
    col_off = Int64((tile_coord_mnkl[1] * cta_tile_mn[1] + dst_idx[1] * epi_n) * elem_bytes)
    len_m = varlen_manager.len_m(batch_idx)
    base_row = varlen_manager.params.cu_seqlens_m[batch_idx]
    lane_off = Int64((tidx % lanes) * 16)
    slots_per_rank = dist_op_args.slots_per_rank
    for step in cutlass.range_constexpr(epi_m // tokens_per_step):
        r = row0 + step * tokens_per_step + tidx // lanes
        if r < len_m:
            slot = dist_op_args.row_key[base_row + r]
            dst = (
                dist_op_args.slot_base[slot // slots_per_rank]
                + Int64(slot % slots_per_rank) * Int64(row_bytes)
                + col_off
                + lane_off
            )
            w0, w1 = words[(0, 0), step, 0], words[(1, 0), step, 0]
            w2, w3 = words[(2, 0), step, 0], words[(3, 0), step, 0]
            st_weak_global_4xu32_at(dst, w0, w1, w2, w3)


class CombineStore(DistOp):
    """GEMM2 + combine: the epilogue's rows land in their origin ranks' slots (hidden = the D row width)."""

    def __init__(self, hidden: int):
        self.hidden = hidden

    def fake_args(self, d_dtype):
        return CombineArguments(
            make_fake_tensor(Int64, (cute.sym_int(),), leading_dim=0, divisibility=1),
            Int32(0),
            make_fake_tensor(Int32, (cute.sym_int(),), leading_dim=0, divisibility=1),
        )

    def store_drain(self, gemm, ctx, sD, epi_tile, tidx):
        """Thread -> (token row along M, lane along N), 16 B per lane over the D stage: a subtile row of
        epi_n elements is epi_n * width / 128 lanes, num_epi_warps * 32 / lanes tokens per step."""
        dist_op_args = ctx.args
        width = gemm.d_dtype.width
        epi_n = cute.size(epi_tile[1])
        lanes = epi_n * width // 128
        assert lanes * 128 == epi_n * width, "a token's subtile slice is whole 16 B lanes"
        elems = 128 // width
        tokens_per_step = gemm.num_epi_warps * cute.arch.WARP_SIZE // lanes
        assert cute.size(epi_tile[0]) % tokens_per_step == 0, "the subtile's rows are whole steps"
        op = cute.nvgpu.CopyUniversalOp()
        atom = cute.make_copy_atom(op, gemm.d_dtype, num_bits_per_copy=128)
        thr = cute.make_layout((tokens_per_step, lanes), stride=(lanes, 1))
        val = cute.make_layout((1, elems), stride=(elems, 1))
        tiled_copy = cute.make_tiled_copy_tv(atom, thr, val)
        tSR_s = tiled_copy.get_slice(tidx).partition_S(sD)
        tSR_r = cute.make_rmem_tensor(tSR_s[None, None, None, 0].shape, gemm.d_dtype)
        return partial(
            store_subtile,
            tiled_copy,
            tSR_s,
            tSR_r,
            epi_tile,
            gemm.cta_tile_shape_mnk[:2],
            dist_op_args,
            Int32(self.hidden * width // 8),
            lanes,
            tokens_per_step,
            tidx,
        )


def make_combine_buffers(tokens_local, k, hidden, torch_dtype, group, max_rows):
    """(d, slots, args) for GEMM2 + combine of tokens_local tokens x k experts x hidden: a (max_rows, hidden) D
    the drain never writes (the kernel's D shape), the symmetric slots every peer's GEMM2 lands rows in (reduce
    over k after a cross-rank barrier), and args(row_key) -> CombineArguments for a call's routing (each D row's
    global slot id). The operator is CombineStore(hidden)."""
    import torch
    import torch.distributed._symmetric_memory as symm_mem

    dev = torch.cuda.current_device()
    world = group.size()
    slots = symm_mem.empty(tokens_local, k, hidden, dtype=torch_dtype, device=dev)
    hdl = symm_mem.rendezvous(slots, group=group)
    shape = (tokens_local, k, hidden)
    bases = [hdl.get_buffer(q, shape, torch_dtype).data_ptr() for q in range(world)]
    slot_base = torch.tensor(bases, dtype=torch.int64, device=dev)
    d = torch.empty(max_rows, hidden, dtype=torch_dtype, device=dev)
    return d, slots, partial(CombineArguments, slot_base, tokens_local * k)
