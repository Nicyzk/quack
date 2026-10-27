# Referenced from: quack/rmsnorm.py (main, fwd kernel) and
# cutlass/examples/.../distributed/all_reduce_two_shot_multimem.py (multimem two-shot + exit flag).
"""Two-shot multimem all-reduce fused into quack's RMSNorm fwd kernel.

Each rank owns a row slab (M/world rows; M % world == 0 required — the norm's
row_reduce needs whole rows on the computing rank): it multimem.ld_reduces its
slab's x across all ranks at the NVLink switch (acc::f32, one rounding back to
dtype), adds the residual, runs the quack RMSNorm epilogue on its rows, and
multimem.st-broadcasts res_out and the normed out so every rank ends with the
full (M, N) results. x, out and res_out must be symmetric: the kernel takes
this rank's (slab, N) view of each multicast tensor; res and w are plain local
tensors. Completion is the example's per-SM exit flag (release multimem.red +1,
relaxed CAS world->0 spin): kernel exit on any rank implies every rank's stores
are done. The caller must ensure all ranks' x are written before launch (a
dist.barrier before the first call; steady-state, the previous call's exit
barrier bounds skew).
"""

from functools import partial
from typing import Optional, Type

import cuda.bindings.driver as cuda

import cutlass
import cutlass.utils as utils
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.base_dsl.arch import Arch

import quack.copy_utils as copy_utils
import quack.layout_utils as layout_utils
from quack.reduce import row_reduce
from quack.reduction_base import ReductionBase
from quack.rmsnorm_config import RmsNormFwdConfig

from kernels.rs.reduce_scatter_primitives import _multimem_ld_reduce_128b


class ARRmsNormTwoShotMultimem(ReductionBase):
    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        N: int,
        config: Optional[RmsNormFwdConfig] = None,
    ):
        super().__init__(dtype, N, stage=1)
        if config is None:
            # Same default as the one-shot: one 128b atom per thread (capped by the
            # 1024-thread CTA limit).
            vec = 128 // dtype.width
            tpr = min(1024, max(32, N // vec))
            config = RmsNormFwdConfig(
                num_threads=tpr, threads_per_row=tpr, cluster_n=1, reload_from=None
            )
        self.config = config
        self.delay_w_load = config.delay_w_load
        self._num_threads_val = config.num_threads
        self._threads_per_row_val = config.threads_per_row
        self._cluster_n_val = config.cluster_n

    def _num_threads(self):
        return self._num_threads_val

    def _threads_per_row(self):
        return self._threads_per_row_val

    def _set_cluster_n(self):
        arch = cutlass.base_dsl.BaseDSL._get_dsl().get_arch_enum()
        if arch < Arch.sm_90:
            self.cluster_n = 1
            return
        max_cluster = 8 if arch.major == 12 else 16
        self.cluster_n = min(self._cluster_n_val, max_cluster)

    @cute.jit
    def __call__(
        self,
        world_size: cutlass.Constexpr,
        mX: cute.Tensor,  # (slab, N) this rank's slab view of the input MC tensor
        mW: Optional[cute.Tensor],  # (N,)
        mRes: Optional[cute.Tensor],  # (slab, N) local residual slab view
        mO: cute.Tensor,  # (slab, N) slab view of the output MC tensor
        mResO: Optional[cute.Tensor],  # (slab, N) slab view of the res_out MC tensor
        flag: cute.Tensor,  # (num CTAs,) Int32 local flags
        flag_mc: cute.Tensor,  # multicast view of flag
        eps: Float32,
        stream: cuda.CUstream,
    ):
        assert mX.element_type == self.dtype
        # One tiled copy at the multimem atom width: 128b atoms, all operands same width.
        for t in [mW, mRes, mO, mResO]:
            if t is not None:
                assert t.element_type.width == self.dtype.width
        vecsize = 128 // self.dtype.width
        assert self.N % vecsize == 0, "multimem atoms are 128b"
        self._set_cluster_n()
        self._cap_cluster_n(vecsize)
        tiled_copy, tiler_mn, threads_per_row = self._get_tiled_copy(vecsize=vecsize)
        num_threads = tiled_copy.size
        if const_expr(mW is not None):
            mW = layout_utils.expand(mW, dim=0, size=tiler_mn[0])
        self.kernel(
            world_size, mX, mW, mRes, mO, mResO, flag, flag_mc, eps,
            tiler_mn, tiled_copy, threads_per_row,
        ).launch(
            grid=[cute.ceil_div(mX.shape[0], tiler_mn[0]), self.cluster_n, 1],
            block=[num_threads, 1, 1],
            cluster=[1, self.cluster_n, 1] if const_expr(self.cluster_n > 1) else None,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        world_size: cutlass.Constexpr,
        mX: cute.Tensor,
        mW: Optional[cute.Tensor],
        mRes: Optional[cute.Tensor],
        mO: cute.Tensor,
        mResO: Optional[cute.Tensor],
        flag: cute.Tensor,
        flag_mc: cute.Tensor,
        eps: Float32,
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
        threads_per_row: cutlass.Constexpr[int],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        cluster_y = const_expr(0) if const_expr(self.cluster_n == 1) else cute.arch.block_idx()[1]
        tv_layout = tiled_copy.layout_tv_tiled

        smem = cutlass.utils.SmemAllocator()
        reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(smem, tv_layout)

        shape = (cute.size(mX, mode=[0]), cute.size(mX, mode=[1]))
        idX = cute.make_identity_tensor(shape)
        gX, gRes, gO, gResO, cX = [
            cute.local_tile(mT, tiler_mn, (bidx, cluster_y)) if mT is not None else None
            for mT in (mX, mRes, mO, mResO, idX)
        ]
        gW = cute.local_tile(mW, tiler_mn, (0, cluster_y)) if const_expr(mW is not None) else None

        thr_copy_X = tiled_copy.get_slice(tidx)
        tXgX = thr_copy_X.partition_S(gX)
        tXgO = thr_copy_X.partition_D(gO)
        tXgW = thr_copy_X.partition_S(gW) if const_expr(mW is not None) else None
        tXgRes = thr_copy_X.partition_S(gRes) if const_expr(mRes is not None) else None
        tXgResO = thr_copy_X.partition_D(gResO) if const_expr(mResO is not None) else None
        tXcX = thr_copy_X.partition_S(cX)[(0, None), None, None]

        # dtype fragments with Int32 word views: multimem ld_reduce/st move 4x b32 per atom.
        tXrX, tXrO = [cute.make_rmem_tensor_like(t) for t in (tXgX, tXgO)]
        tXrXw, tXrOw = [cute.recast_tensor(t, Int32) for t in (tXrX, tXrO)]
        tXrW = cute.make_rmem_tensor_like(tXgW) if const_expr(mW is not None) else None
        tXrRes = cute.make_rmem_tensor_like(tXgRes) if const_expr(mRes is not None) else None

        num_warps = cute.size(tiled_copy) // cute.arch.WARP_SIZE
        self._initialize_cluster(tidx, mbar_ptr, num_warps)

        is_even_N = const_expr(shape[1] == tiler_mn[1] * self.cluster_n)
        tXpX = (
            copy_utils.predicate_k(thr_copy_X.partition_S(cX), limit=shape[1])
            if not is_even_N
            else None
        )
        copy = partial(copy_utils.copy, pred=tXpX)

        ld_reduce = _multimem_ld_reduce_128b(mX.element_type)
        n_rv = const_expr(cute.size(tXrX.shape[0][1]))
        n_rk = const_expr(cute.size(tXrX.shape[2]))

        # Switch-reduce this rank's slab rows from the input multicast address.
        tXrX.fill(0.0)
        row = tXcX[0][0]
        if row < shape[0]:
            for rv in cutlass.range_constexpr(n_rv):
                for rk in cutlass.range_constexpr(n_rk):
                    if const_expr(tXpX is None) or tXpX[rv, 0, rk]:
                        w0, w1, w2, w3 = ld_reduce(tXgX[(None, rv), None, rk].iterator)
                        tXrXw[(0, rv), 0, rk] = w0
                        tXrXw[(1, rv), 0, rk] = w1
                        tXrXw[(2, rv), 0, rk] = w2
                        tXrXw[(3, rv), 0, rk] = w3

        if const_expr(not self.delay_w_load):
            if const_expr(mW is not None):
                copy(tXgW, tXrW)
        x = tXrX.load().to(cute.Float32)
        if const_expr(mRes is not None):
            tXrRes.fill(0.0)
            if row < shape[0]:
                copy(tXgRes, tXrRes)
            x += tXrRes.load().to(cute.Float32)
        if const_expr(mResO is not None):
            tXrResO = cute.make_rmem_tensor_like(tXgResO)
            tXrResOw = cute.recast_tensor(tXrResO, Int32)
            tXrResO.store(x.to(tXrResO.element_type))
            if row < shape[0]:
                for rv in cutlass.range_constexpr(n_rv):
                    for rk in cutlass.range_constexpr(n_rk):
                        if const_expr(tXpX is None) or tXpX[rv, 0, rk]:
                            # .ir_value(): the util feeds these raw into llvm.inline_asm
                            utils.distributed.multimem_st_4xb32(
                                tXgResO[(None, rv), None, rk].iterator,
                                tXrResOw[(0, rv), 0, rk].ir_value(),
                                tXrResOw[(1, rv), 0, rk].ir_value(),
                                tXrResOw[(2, rv), 0, rk].ir_value(),
                                tXrResOw[(3, rv), 0, rk].ir_value(),
                            )

        sum_sq_x = row_reduce(
            x * x,
            cute.ReductionOp.ADD,
            threads_per_row,
            reduction_buffer[None, None, 0],
            mbar_ptr,
            init_val=0.0,
            hook_fn=cute.arch.cluster_wait if const_expr(self.cluster_n > 1) else None,
        )
        rstd = cute.math.rsqrt(sum_sq_x / shape[1] + eps, fastmath=True)
        if const_expr(self.delay_w_load):
            if const_expr(mW is not None):
                copy(tXgW, tXrW)
        y = x * rstd
        if const_expr(mW is not None):
            y *= tXrW.load().to(cute.Float32)
        tXrO.store(y.to(tXrO.element_type))
        if row < shape[0]:
            for rv in cutlass.range_constexpr(n_rv):
                for rk in cutlass.range_constexpr(n_rk):
                    if const_expr(tXpX is None) or tXpX[rv, 0, rk]:
                        utils.distributed.multimem_st_4xb32(
                            tXgO[(None, rv), None, rk].iterator,
                            tXrOw[(0, rv), 0, rk].ir_value(),
                            tXrOw[(1, rv), 0, rk].ir_value(),
                            tXrOw[(2, rv), 0, rk].ir_value(),
                            tXrOw[(3, rv), 0, rk].ir_value(),
                        )

        # Ensure all threads in cta have finish issue multimem.ld_reduce and multimem.st instructions
        cute.arch.sync_threads()

        if tidx == 0:
            # Linear id of current SM.
            sm_id_linear = (
                cute.arch.block_idx()[0]
                + cute.arch.block_idx()[1] * cute.arch.grid_dim()[0]
                + cute.arch.block_idx()[2]
                * cute.arch.grid_dim()[0]
                * cute.arch.grid_dim()[1]
            )
            # Release flag with sys scope
            utils.distributed.multimem_red_add1(
                flag_mc.iterator + sm_id_linear,
                scope="sys",
                order="release",
            )
            # Relaxed spin-lock wait flag with sys scope
            utils.distributed.spin_lock_atom_cas_relaxed_wait(
                flag.iterator + sm_id_linear,
                expected_val=world_size,
                reset_val=0,
                scope="sys",
            )
