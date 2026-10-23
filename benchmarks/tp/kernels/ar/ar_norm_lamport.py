# Referenced from: quack/rmsnorm.py (main, fwd kernel) and
# cutlass/examples/.../distributed/all_reduce_one_shot_lamport.py (trtllm lamport protocol).
"""One-shot lamport all-reduce fused into quack's RMSNorm fwd kernel.

res_out = sum_r x_r + res; out = rmsnorm(res_out) * w. The sum arrives via the
trtllm lamport protocol: each CTA volatile-sys-stores its rows of the local
input into slot [rank, ping] of every rank's symmetric buffer, then polls its
LOCAL buffer's [r, ping] slots and accumulates in fp32; quack's row_reduce
epilogue runs unchanged on the reduced rows. A slot element equal to -0.0 means
"not written yet" (writes flush real -0.0 to +0.0; per-element checks make torn
128b stores benign). 3 slots rotate per call: call k uses slot k%3 and clears
slot (k+1)%3 for the next call. Buffers must start all -0.0 (make_lamport_buffers)
and the caller passes signal = call index (host counter, as in bench_ar's
cutlass_1shot_lamport arm: captured windows stay in phase if their length % 3 == 0).
"""

import math
from functools import partial
from typing import Optional, Type

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Float32, Int32, const_expr
from cutlass.cutlass_dsl import T
from cutlass._mlir.dialects import vector
from cutlass.base_dsl.arch import Arch

import quack.copy_utils as copy_utils
import quack.layout_utils as layout_utils
from quack.reduce import row_reduce
from quack.reduction_base import ReductionBase
from quack.rmsnorm_config import RmsNormFwdConfig

PING_PONG_SIZE = 3


def _is_not_neg0(vec: cute.TensorSSA, neg0_i32_vec: cute.TensorSSA) -> Boolean:
    """The example's isNotNeg0 check, at i32-word granularity (bf16 pairs pack to 0x80008000)."""
    n_words = cute.size(neg0_i32_vec.shape)
    frg_vector_i32 = cute.TensorSSA(
        vector.bitcast(T.vector(n_words, T.i32()), vec), (n_words,), Int32
    )
    return cute.all_(
        cute.TensorSSA(frg_vector_i32 != neg0_i32_vec, frg_vector_i32.shape, Boolean)
    )


class ARRmsNormOneShot(ReductionBase):
    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        N: int,
        config: Optional[RmsNormFwdConfig] = None,
    ):
        super().__init__(dtype, N, stage=1)
        assert dtype.width in (16, 32), "lamport sentinel needs 16- or 32-bit elements"
        if config is None:
            # The example's decomposition: one 128b atom per thread (capped by the
            # 1024-thread CTA limit; quack's local-HBM norm heuristic gives each
            # thread 8 atoms and loses 4-8x on the sys-scope AR phase).
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
        rank: cutlass.Constexpr,
        world_size: cutlass.Constexpr,
        signal: Int32,
        mX: cute.Tensor,  # (M, N) local input, plain gmem
        mW: Optional[cute.Tensor],  # (N,)
        mRes: Optional[cute.Tensor],  # (M, N)
        mO: cute.Tensor,  # (M, N)
        mResO: Optional[cute.Tensor],  # (M, N)
        buffers: list[cute.Tensor],  # per-rank symmetric (M, N, world, PING_PONG)
        eps: Float32,
        stream: cuda.CUstream,
    ):
        assert mX.element_type == self.dtype
        assert buffers[0].element_type == self.dtype
        self._set_cluster_n()
        largest_dtype_width = const_expr(
            max(*(t.element_type.width for t in [mX, mRes, mW, mO, mResO] if t is not None))
        )
        vecsize = math.gcd(self.N, 128 // largest_dtype_width)
        assert vecsize * self.dtype.width % 32 == 0
        self._cap_cluster_n(vecsize)
        tiled_copy, tiler_mn, threads_per_row = self._get_tiled_copy(vecsize=vecsize)
        num_threads = tiled_copy.size
        if const_expr(mW is not None):
            mW = layout_utils.expand(mW, dim=0, size=tiler_mn[0])
        self.kernel(
            rank, signal, mX, mW, mRes, mO, mResO, buffers, eps,
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
        rank: cutlass.Constexpr,
        signal: Int32,
        mX: cute.Tensor,
        mW: Optional[cute.Tensor],
        mRes: Optional[cute.Tensor],
        mO: cute.Tensor,
        mResO: Optional[cute.Tensor],
        buffers: list[cute.Tensor],
        eps: Float32,
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
        threads_per_row: cutlass.Constexpr[int],
    ):
        world_size = const_expr(len(buffers))
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        cluster_y = const_expr(0) if const_expr(self.cluster_n == 1) else cute.arch.block_idx()[1]
        tv_layout = tiled_copy.layout_tv_tiled
        ping = signal % PING_PONG_SIZE
        pong = (signal + 1) % PING_PONG_SIZE

        smem = cutlass.utils.SmemAllocator()
        reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(smem, tv_layout)

        shape = (cute.size(mX, mode=[0]), cute.size(mX, mode=[1]))
        idX = cute.make_identity_tensor(shape)
        gX, gRes, gO, gResO, cX = [
            cute.local_tile(mT, tiler_mn, (bidx, cluster_y)) if mT is not None else None
            for mT in (mX, mRes, mO, mResO, idX)
        ]
        gW = cute.local_tile(mW, tiler_mn, (0, cluster_y)) if const_expr(mW is not None) else None
        # Lamport slots, tiled identically to gX so fragments correspond 1:1.
        gRead = [
            cute.local_tile(buffers[rank][None, None, r, ping], tiler_mn, (bidx, cluster_y))
            for r in range(world_size)
        ]
        gClear = [
            cute.local_tile(buffers[rank][None, None, r, pong], tiler_mn, (bidx, cluster_y))
            for r in range(world_size)
        ]
        gWrite = [
            cute.local_tile(buffers[p][None, None, rank, ping], tiler_mn, (bidx, cluster_y))
            for p in range(world_size)
        ]

        thr_copy_X = tiled_copy.get_slice(tidx)
        tXgX = thr_copy_X.partition_S(gX)
        tXgO = thr_copy_X.partition_D(gO)
        tXgW = thr_copy_X.partition_S(gW) if const_expr(mW is not None) else None
        tXgRes = thr_copy_X.partition_S(gRes) if const_expr(mRes is not None) else None
        tXgResO = thr_copy_X.partition_D(gResO) if const_expr(mResO is not None) else None
        tXbR = [thr_copy_X.partition_S(g) for g in gRead]
        tXbC = [thr_copy_X.partition_D(g) for g in gClear]
        tXbW = [thr_copy_X.partition_D(g) for g in gWrite]
        tXcX = thr_copy_X.partition_S(cX)[(0, None), None, None]

        tXrX, tXrO = [cute.make_rmem_tensor_like(t) for t in (tXgX, tXgO)]
        tXrW = cute.make_rmem_tensor_like(tXgW) if const_expr(mW is not None) else None
        tXrRes = cute.make_rmem_tensor_like(tXgRes) if const_expr(mRes is not None) else None
        tXrAcc = cute.make_rmem_tensor_like(tXrX, Float32)
        frg_r = cute.make_rmem_tensor_like(tXrX)

        num_warps = cute.size(tiled_copy) // cute.arch.WARP_SIZE
        self._initialize_cluster(tidx, mbar_ptr, num_warps)

        is_even_N = const_expr(shape[1] == tiler_mn[1] * self.cluster_n)
        tXpX = (
            copy_utils.predicate_k(thr_copy_X.partition_S(cX), limit=shape[1])
            if not is_even_N
            else None
        )
        copy = partial(copy_utils.copy, pred=tXpX)

        dtype = mX.element_type
        atom_elems = const_expr(cute.size(tXrX.shape[0][0]))
        atom_bits = const_expr(atom_elems * dtype.width)
        sentinel_word = const_expr(0x80008000 if dtype.width == 16 else 0x80000000)
        vec_t = T.f32 if dtype.width == 32 else (T.bf16 if dtype == cutlass.BFloat16 else T.f16)
        # assume all buffers have the same element type with input
        copy_atom_load = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            buffers[0].element_type,
            num_bits_per_copy=atom_bits,
            memory_scope=cute.nvgpu.common.MemoryScope.SYS,
            memory_order=cute.nvgpu.common.MemoryOrder.VOLATILE,
        )
        copy_atom_store = cute.make_copy_atom(
            cute.nvgpu.CopyR2GOp(),
            buffers[0].element_type,
            num_bits_per_copy=atom_bits,
            memory_scope=cute.nvgpu.common.MemoryScope.SYS,
            memory_order=cute.nvgpu.common.MemoryOrder.VOLATILE,
        )
        n_rv = const_expr(cute.size(tXrX.shape[0][1]))
        n_rk = const_expr(cute.size(tXrX.shape[2]))

        # clear a next buffer to be all negtive 0 (example verbatim, one copy atom wide)
        frg_clear = cute.make_rmem_tensor_like(tXrX[(None, 0), None, 0])
        clear_tensor = frg_clear.load()
        frg_size = cute.size(clear_tensor.shape)
        n_words = const_expr(frg_size * dtype.width // 32)
        clear_words = cute.TensorSSA(
            vector.bitcast(T.vector(n_words, T.i32()), clear_tensor), (n_words,), Int32
        )
        neg0_i32_vec = cute.full_like(clear_words, sentinel_word, Int32)
        neg0_vec = vector.bitcast(T.vector(frg_size, vec_t()), neg0_i32_vec)
        frg_clear.store(cute.TensorSSA(neg0_vec, clear_tensor.shape, dtype))

        row = tXcX[0][0]
        if row < shape[0]:
            copy(tXgX, tXrX)
            for r in cutlass.range_constexpr(world_size):
                for rv in cutlass.range_constexpr(n_rv):
                    for rk in cutlass.range_constexpr(n_rk):
                        if const_expr(tXpX is None) or tXpX[rv, 0, rk]:
                            cute.copy(copy_atom_store, frg_clear, tXbC[r][(None, rv), None, rk])
            # trtllm's remove_neg_zero: real -0.0 would read as "empty" (the example
            # omits this; its fp32 randn never hits -0.0)
            vx = tXrX.load()
            zeros = cute.full_like(vx, 0.0)
            tXrX.store(cute.where(cute.TensorSSA(vx == zeros, vx.shape, Boolean), zeros, vx))
            # write local data to every buffer at different ranks
            for i in cutlass.range_constexpr(world_size):
                p = (rank + i) % world_size  # stagger so peers don't hammer rank 0 first
                for rv in cutlass.range_constexpr(n_rv):
                    for rk in cutlass.range_constexpr(n_rk):
                        if const_expr(tXpX is None) or tXpX[rv, 0, rk]:
                            cute.copy(copy_atom_store, tXrX[(None, rv), None, rk], tXbW[p][(None, rv), None, rk])

        if const_expr(not self.delay_w_load):
            if const_expr(mW is not None):
                copy(tXgW, tXrW)
        if const_expr(mRes is not None):
            tXrRes.fill(0.0)
            if row < shape[0]:
                copy(tXgRes, tXrRes)

        # loop over each buffer and accumulate the data (example's poll, per copy atom;
        # OOB rows/atoms are never written and stay 0 in the acc)
        tXrAcc.fill(0.0)
        frg_r.fill(0.0)
        if row < shape[0]:
            for i in cutlass.range_constexpr(world_size):
                r = (rank + i) % world_size  # own slot first: the local write lands earliest
                for rv in cutlass.range_constexpr(n_rv):
                    for rk in cutlass.range_constexpr(n_rk):
                        if const_expr(tXpX is None) or tXpX[rv, 0, rk]:
                            src = tXbR[r][(None, rv), None, rk]
                            dst = frg_r[(None, rv), None, rk]
                            cute.copy(copy_atom_load, src, dst)
                            isNotNeg0 = _is_not_neg0(dst.load(), neg0_i32_vec)
                            # if the data is negtive 0, it means data are not ready or not
                            # visible yet, so we need to read the data again
                            while not isNotNeg0:
                                cute.copy(copy_atom_load, src, dst)
                                isNotNeg0 = _is_not_neg0(dst.load(), neg0_i32_vec)
                tXrAcc.store(tXrAcc.load() + frg_r.load().to(Float32))

        x = tXrAcc.load()
        if const_expr(mRes is not None):
            x += tXrRes.load().to(Float32)
        if const_expr(mResO is not None):
            tXrResO = cute.make_rmem_tensor_like(tXgResO)
            tXrResO.store(x.to(tXrResO.element_type))
            if row < shape[0]:
                copy(tXrResO, tXgResO)

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
            y *= tXrW.load().to(Float32)
        tXrO.store(y.to(tXrO.element_type))
        if row < shape[0]:
            copy(tXrO, tXgO)


def make_lamport_buffers(M, N, world, dtype):
    """Symmetric (PING_PONG, world, M, N) buffer, every element -0.0, plus per-rank
    peer views permuted to the kernel's (M, N, world, PING_PONG) layout."""
    import torch
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    from kernels.misc.utils import make_peer_tensor

    assert dtype in (torch.bfloat16, torch.float16, torch.float32)
    buf, peers = make_peer_tensor((PING_PONG_SIZE, world, M, N), dtype)
    int_view = torch.int16 if dtype.itemsize == 2 else torch.int32
    neg0 = -32768 if dtype.itemsize == 2 else -(2**31)
    buf.view(int_view).fill_(neg0)
    return buf, [p.permute(2, 3, 1, 0) for p in peers]
