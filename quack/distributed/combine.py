# MoE GEMM2 + combine as the GEMM's distributed operator: no gate, no publish, no comm warps. The grouped GEMM's
# D rows are tokens (varlen-M, rows packed per expert); the operator drains each staged D subtile straight into
# the origin rank's slot for that (token, k) pair, 16 B per lane, at the row's hidden offset (store_drain: the
# GEMM stages accumulator -> registers -> epi ops -> smem as always, the TMA store is replaced). D itself is
# never written, nothing is waited on; completion is the kernel's end, then the caller barriers and reduces
# every token's K slots (duckpond reduce_over_k). Slot id of row r = row_key[r] = rank * slots_per_rank + local.
# Usage: cb = CombineRunner(tokens_local, k, hidden, dtype, group, max_rows); cb.route(row_key);
# mod(a, w2, cu_seqlens_m=cu, config=cfg, comm=cb) ... or direct: GemmSm100(..., comm=cb.comm), mD=cb.wire,
# comm_args=cb.args(); then barrier; the landed slots are cb.out (tokens_local, k, hidden).
from functools import partial
from typing import NamedTuple

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64
from quack.compile_utils import make_fake_tensor
from quack.cute_dsl_utils import mlir_namedtuple
from quack.distributed.ptx import st_weak_global_4xu32_at
from quack.gemm_base import Comm, CommRunner


@mlir_namedtuple
class CombineArguments(NamedTuple):
    """The drain's bundle (the GEMM's comm_args)."""

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
    comm_args,
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
    slots_per_rank = comm_args.slots_per_rank
    for step in cutlass.range_constexpr(epi_m // tokens_per_step):
        r = row0 + step * tokens_per_step + tidx // lanes
        if r < len_m:
            slot = comm_args.row_key[base_row + r]
            dst = (
                comm_args.slot_base[slot // slots_per_rank]
                + Int64(slot % slots_per_rank) * Int64(row_bytes)
                + col_off
                + lane_off
            )
            w0, w1 = words[(0, 0), step, 0], words[(1, 0), step, 0]
            w2, w3 = words[(2, 0), step, 0], words[(3, 0), step, 0]
            st_weak_global_4xu32_at(dst, w0, w1, w2, w3)


class CombineStore(Comm):
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
        comm_args = ctx.args
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
            comm_args,
            Int32(self.hidden * width // 8),
            lanes,
            tokens_per_step,
            tidx,
        )


class CombineRunner(CommRunner):
    """GEMM2 + combine for tokens_local tokens x k experts x hidden: owns the symmetric slots every peer's GEMM2
    lands rows in, the slot base table and a D-shaped wire the kernel never writes. route(row_key) per call
    (each D row's global slot id, from the routing); out = the slots, to reduce over k after a cross-rank
    barrier."""

    def __init__(self, tokens_local, k, hidden, torch_dtype, group, max_rows):
        import torch
        import torch.distributed._symmetric_memory as symm_mem

        dev = torch.cuda.current_device()
        world = group.size()
        self.comm = CombineStore(hidden)
        self.out = symm_mem.empty(tokens_local, k, hidden, dtype=torch_dtype, device=dev)
        hdl = symm_mem.rendezvous(self.out, group=group)
        shape = (tokens_local, k, hidden)
        bases = [hdl.get_buffer(q, shape, torch_dtype).data_ptr() for q in range(world)]
        self.slot_base = torch.tensor(bases, dtype=torch.int64, device=dev)
        self.slots_per_rank = tokens_local * k
        # D's shape for the kernel (row count, dtype); the drain never writes it
        self.wire = torch.empty(max_rows, hidden, dtype=torch_dtype, device=dev)
        self.row_key = None

    def route(self, row_key):
        self.row_key = row_key

    def args(self):
        assert self.row_key is not None, "route(row_key) before the launch"
        return CombineArguments(self.slot_base, self.slots_per_rank, self.row_key)
