import math
from typing import NamedTuple, Optional
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32, const_expr

from quack.cute_dsl_utils import mlir_namedtuple
from quack.sm90_utils import partition_for_epilogue
from quack import layout_utils

class GemmLseRsMixin():
    """LSE (log-sum-exp) computation in the RS warp block.

    Mirrors the SM90 LSEOp.begin pattern: derive a per-thread m-row accumulator
    layout from thr_copy_fake (non-hardcoded), allocate tDr_max/tDr_sum, return
    them as state for visit and end hooks.
    """
    @mlir_namedtuple
    class EpiRSArguments(NamedTuple):
        mLSE: cute.Tensor                       # (m, ceildiv(n, tile_n), batch)
        # If mTarget and mTargetLogit are provided, we write the logit corresponding to @mTarget
        # to @mTargetLogit.
        mTarget: Optional[cute.Tensor]          # (m, batch)
        mTargetLogit: Optional[cute.Tensor]     # (m, batch)
        D_limit_n: Optional[Int32]              # If None, we don't do bound checking
        multiplier: Optional[Float32] = None

    @dataclass
    class EpiRSParams:
        mLSE: cute.Tensor
        mTarget: Optional[cute.Tensor]
        mTargetLogit: Optional[cute.Tensor]
        D_limit_n: Optional[Int32]
        multiplier: Optional[Float32] = None

    def epi_rs_to_underlying_arguments(
        self, args: EpiRSArguments, *, loc=None, ip=None
    ) -> EpiRSParams:
        if const_expr(args.mTarget is not None):
            assert (
                args.mTargetLogit is not None
            ), "If mTarget is provided, mTargetLogit must also be provided"
        return self.EpiRSParams(args.mLSE, args.mTarget, args.mTargetLogit, args.D_limit_n, args.multiplier)

    @cute.jit
    def epi_rs_begin(self, epi_rs_params, epi_tile_rs, tiled_copy, tCpD_local_rank, tidx_rs):
        # m-only layout: stride=(1,0) makes m unique, n broadcast.
        tile_layout = cute.make_layout(tCpD_local_rank.shape, stride=(1, 0))
        tDrLSE_layout = partition_for_epilogue(
            cute.make_rmem_tensor(tile_layout, Float32),
            epi_tile=epi_tile_rs,
            tiled_copy=tiled_copy,
            tidx=tidx_rs,
            reference_src=True,
        ).layout
        tDr_max = cute.make_rmem_tensor(tDrLSE_layout, Float32)
        tDr_sum = cute.make_rmem_tensor(tDrLSE_layout, Float32)
        cute.filter_zeros(tDr_max).fill(-Float32.inf)
        cute.filter_zeros(tDr_sum).fill(0.0)
        return tDr_max, tDr_sum

    @cute.jit
    def epi_rs_begin_loop(self, epi_rs_state, epi_coord):
        tDr_max, tDr_sum = epi_rs_state
        # Shape from partition_for_epilogue: (CPY, CPY_M, CPY_N, EPI_M, EPI_N)
        # Group (EPI_M, EPI_N) into mode 3, index with epi_coord
        tDr_max_loop = cute.group_modes(tDr_max, 3, cute.rank(tDr_max))[(None, None), None, None, epi_coord]
        tDr_sum_loop = cute.group_modes(tDr_sum, 3, cute.rank(tDr_sum))[(None, None), None, None, epi_coord]
        return tDr_max_loop, tDr_sum_loop

    @cute.jit
    def epi_rs_visit_subtile_slice(self, loop_state, epi_rs_params, tRS_rD, frpD):
        tDr_max_loop, tDr_sum_loop = loop_state
        do_bound_check = const_expr(epi_rs_params.D_limit_n is not None)
        mult = epi_rs_params.multiplier if const_expr(epi_rs_params.multiplier is not None) else 1.0
        atom_thr_n = const_expr(self.mma_tiler[1] // (128 // self.d_dtype.width))

        tDr_max_flt = cute.filter_zeros(tDr_max_loop)
        tDr_sum_flt = cute.filter_zeros(tDr_sum_loop)

        # Snapshot prev max before update
        tDr_prev_max = cute.make_rmem_tensor_like(tDr_max_flt, Float32)
        tDr_prev_max.store(tDr_max_flt.load())

        # Per-element max update; tDr_max_loop[k] maps k to the right m-row via stride=(1,0)
        for k in cutlass.range(cute.size(tDr_max_loop), unroll_full=True):
            m_k, n_k, l_k = frpD[k]
            if not do_bound_check or n_k < epi_rs_params.D_limit_n:
                tDr_max_loop[k] = cute.arch.fmax(tDr_max_loop[k], tRS_rD[k])

        # Warp-reduce max and rescale sum — one pass per unique m-row
        for r in cutlass.range(cute.size(tDr_max_flt), unroll_full=True):
            prev_max = tDr_prev_max[r] if tDr_prev_max[r] > -Float32.inf else 0.0
            tDr_max_flt[r] = cute.arch.warp_reduction_max(tDr_max_flt[r], threads_in_group=atom_thr_n)
            cur_max = tDr_max_flt[r] if tDr_max_flt[r] > -Float32.inf else 0.0
            tDr_sum_flt[r] *= cute.math.exp2(math.log2(math.e) * mult * (prev_max - cur_max), fastmath=True)

        # Per-element sum update
        for k in cutlass.range(cute.size(tDr_max_loop), unroll_full=True):
            m_k, n_k, l_k = frpD[k]
            if not do_bound_check or n_k < epi_rs_params.D_limit_n:
                tDr_sum_loop[k] += cute.math.exp2(math.log2(math.e) * mult * (tRS_rD[k] - tDr_max_loop[k]), fastmath=True)

    @cute.jit
    def epi_rs_end(self, epi_rs_state, epi_rs_params, frpD, mma_tile_coord_mnl, tidx_rs):
        tDr_max, tDr_sum = epi_rs_state
        mult = epi_rs_params.multiplier if const_expr(epi_rs_params.multiplier is not None) else 1.0
        atom_thr_n = const_expr(self.mma_tiler[1] // (128 // self.d_dtype.width))
        n_tile_idx = mma_tile_coord_mnl[1]
        batch_idx = mma_tile_coord_mnl[2]

        # Final warp-reduce sum across atom_thr_n n-threads (once per unique m-row).
        # Max was already warp-reduced inside epi_rs_visit_subtile_slice.
        tDr_sum_flt = cute.filter_zeros(tDr_sum)
        for r in cutlass.range(cute.size(tDr_sum_flt), unroll_full=True):
            tDr_sum_flt[r] = cute.arch.warp_reduction_sum(tDr_sum_flt[r], threads_in_group=atom_thr_n)

        # Per-row views: collapse every mode that has stride 0 in tDr_max's layout
        # (i.e. the N modes broadcast by the stride=(1,0) source layout). frpD shares
        # the same partition structure, so the remaining modes line up element-for-element.
        tDr_max_m = layout_utils.convert_layout_zero_stride(tDr_max, tDr_max.layout)[None, 0]
        tDr_sum_m = layout_utils.convert_layout_zero_stride(tDr_sum, tDr_max.layout)[None, 0]
        frpD_m = layout_utils.convert_layout_zero_stride(frpD, tDr_max.layout)[None, 0]

        # One writer per m-row: thread 0 in each n-group.
        if tidx_rs % atom_thr_n == 0:
            for m in cutlass.range(cute.size(frpD_m, mode=[0]), unroll_full=True):
                row_idx, _, _ = frpD_m[m]
                if cute.elem_less((row_idx, n_tile_idx), epi_rs_params.mLSE.shape[:2]):
                    row_max = 0.0 if tDr_max_m[m] == -Float32.inf else tDr_max_m[m]
                    lse_val = mult * row_max + cute.math.log(tDr_sum_m[m], fastmath=True)
                    epi_rs_params.mLSE[row_idx, n_tile_idx, batch_idx] = lse_val
