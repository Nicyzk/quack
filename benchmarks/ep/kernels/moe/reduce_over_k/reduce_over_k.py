"""The combine's gather-and-sum over K in CuTe-DSL: out[t, :] = sum_k w[t, k] * x[t, k, :] (w optional), the
replacement for sonic's token_gather_and_sum_varlen_K_triton as the forward calls it (identity perm, K rows per
token). Memory-bound: each thread loads vecs 16 B vectors of each of the token's K rows, accumulates in fp32 in fixed
k order (bitwise run-to-run) and stores the vectors in the output dtype; a CTA covers (num_threads / threads_per_row)
tokens by threads_per_row * vec * vecs columns, the grid tiles H then T.

Fused epilogues come through quack's EpiOp machinery (epilogue.py): the fp32 sum is the accumulator fragment, the
CTA tile is the one epilogue subtile, and the kernel runs quack's hooks in gemm_base.epilogue's order (epi_begin,
epi_begin_loop, epi_visit_subtile, epi_end_loop, the stores, epi_end). The attributes the ops read off the GEMM
(cta_tile_shape_mnk, num_epi_warps, acc_dtype, epi_m_major, epilogue_barrier, arch) are provided here."""
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import torch
from cutlass import Float32, Int32, const_expr
from quack.compile_utils import make_fake_tensor
from quack.cute_dsl_utils import ParamsBase, get_device_capacity, torch2cute_dtype_map
from quack.epilogue.mixin import ComposableEpiMixin
from quack.gemm_runtime.host import FakeArgCtx
from quack.varlen_utils import VarlenManager
import quack.copy_utils as copy_utils


class ReduceOverK(ComposableEpiMixin):
    """The plain kernel (no epilogue ops); a @reduce_epilogue mod mints a subclass carrying its _epi_ops and the
    fn visit (epilogue.py)."""

    _epi_ops = ()
    EpilogueArguments = None
    EpilogueParams = ParamsBase
    acc_dtype = Float32
    epi_m_major = False

    def __init__(self, dtype, topk, has_w, arch, num_threads=128, threads_per_row=64, vecs=2):
        """vecs: 16 B vectors per thread per row (K * vecs loads in flight); defaults = the B300 sweep's best at
        dsv3 8192 (vecs=1 is 10 % slower, 512-thread CTAs 5-50 % slower)."""
        assert num_threads % 32 == 0 and num_threads % threads_per_row == 0, "whole warps, whole rows per CTA"
        self.dtype, self.topk, self.has_w, self.arch = dtype, topk, has_w, arch
        self.num_threads, self.threads_per_row = num_threads, threads_per_row
        self.vec = 128 // dtype.width
        self.tiler_mn = (num_threads // threads_per_row, threads_per_row * self.vec * vecs)
        self.cta_tile_shape_mnk = (*self.tiler_mn, 1)
        self.num_epi_warps = num_threads // 32
        self.epilogue_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=num_threads)

    def epi_smem_warp_shape_mnk(self):
        """warps along (tokens, columns, 1): a row's threads span threads_per_row / 32 warps."""
        warps_per_row = max(self.threads_per_row // 32, 1)
        return (self.num_epi_warps // warps_per_row, warps_per_row, 1)

    def epi_to_underlying_arguments(self, args):
        return self.EpilogueParams(**self._epi_ops_to_params_dict(args))

    def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None):
        return ()

    @cute.jit
    def __call__(self, mX, mW, mOut, epilogue_args, stream):
        """mX (T, K, H), mW (T, K) fp32 or None, mOut (T, H), epilogue_args = the minted class's EpilogueArguments
        (None for the plain kernel)."""
        params = self.epi_to_underlying_arguments(epilogue_args)
        self.shared_storage = None
        smem = 0
        if const_expr(len(self._epi_ops) > 0):

            @cute.struct
            class SharedStorage:
                epi: self.epi_get_smem_struct(params)

            smem = SharedStorage.size_in_bytes()
            if const_expr(smem > 0):  # ops without smem (scalars, per-token values) leave it empty; smem_alloca rejects 0 bytes
                self.shared_storage = SharedStorage
        self.kernel(mX, mW, mOut, params).launch(
            grid=[cute.ceil_div(mOut.shape[1], self.tiler_mn[1]), cute.ceil_div(mOut.shape[0], self.tiler_mn[0]), 1],
            block=[self.num_threads, 1, 1],
            smem=smem,
            stream=stream,
        )

    @cute.jit
    def tile_partition(self, mT, tile_coord_mnkl, tidx):
        """This thread's vectors of mT's (tile_t, tile_h) tile at the tile coordinate (mT (T, H) or a view of it)."""
        gT = cute.local_tile(mT, self.tiler_mn, (tile_coord_mnkl[0], tile_coord_mnkl[1]))
        return self.tiled_copy.get_slice(tidx).partition_S(gT)

    @cute.jit
    def tile_predicate(self, shape, tile_coord_mnkl, tidx):
        """(per-vector column predicate, row in range, the row) of this thread's vectors of a (T, H) tile."""
        tXcX = self.tile_partition(cute.make_identity_tensor(shape), tile_coord_mnkl, tidx)
        row = tXcX[(0, None), None, None][0][0]
        tXpX = copy_utils.predicate_k(tXcX, limit=shape[1])  # whole vectors: H % vec == 0 (host asserts); not named `pred`: the DSL's dynamic-if rewrite turns it scalar
        return tXpX, row < shape[0], row

    @cute.kernel
    def kernel(self, mX, mW, mOut, params):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        tile_coord_mnkl = (bidy, bidx, 0, 0)
        # Built here, not in __call__: a value made in the host jit's region cannot be used inside the kernel's.
        self.tiled_copy = copy_utils.tiled_copy_2d(self.dtype, self.threads_per_row, self.num_threads, num_copy_elems=self.vec)
        epi_smem_tensors = {}
        if const_expr(self.shared_storage is not None):
            storage = cutlass.utils.SmemAllocator().allocate(self.shared_storage)
            epi_smem_tensors = self.epi_get_smem_tensors(params, storage)
        tXpX, row_ok, row = self.tile_predicate(mOut.shape, tile_coord_mnkl, tidx)
        tXgOut = self.tile_partition(mOut, tile_coord_mnkl, tidx)
        tXgX = [self.tile_partition(mX[None, k, None], tile_coord_mnkl, tidx) for k in range(self.topk)]
        frags = [cute.make_rmem_tensor_like(t) for t in tXgX]
        for k in cutlass.range_constexpr(self.topk):
            frags[k].fill(0.0)  # out-of-range lanes stay zero: the fold identity for the reduce sinks
        if row_ok:
            for k in cutlass.range_constexpr(self.topk):
                copy_utils.copy(tXgX[k], frags[k], pred=tXpX)
        acc = frags[0].load().to(Float32)
        if const_expr(self.has_w):
            acc = acc * self._weight(mW, row, row_ok, 0)
        for k in cutlass.range_constexpr(1, self.topk):
            x = frags[k].load().to(Float32)
            if const_expr(self.has_w):
                x = x * self._weight(mW, row, row_ok, k)
            acc = acc + x
        tXrAcc = cute.make_rmem_tensor(frags[0].layout, Float32)
        tXrAcc.store(acc)
        # The epilogue, in gemm_base.epilogue's hook order over the one subtile (0, 0).
        varlen_manager = VarlenManager(VarlenManager.Params(), mOut.shape[0], Int32(0), mOut.shape[1])
        epi_tensors = self.epi_begin(
            params, epi_smem_tensors, self.tiler_mn, None, self.tiled_copy, tile_coord_mnkl, varlen_manager,
            self.epilogue_barrier, tidx, tXrAcc.layout,
        )
        epi_coord = (0, 0)
        if const_expr(getattr(self, "_epi_mod_prepass_fn", None) is not None):  # statistics over the raw sum first
            self.epi_prepass_subtile(params, epi_tensors, tXrAcc, epi_coord, 0)
            self.epi_prepass_end(params, epi_tensors)
        epi_loop_tensors = self.epi_begin_loop(params, epi_tensors, epi_coord)
        aux_outs = self.epi_visit_subtile(params, epi_loop_tensors, tXrAcc)
        self.epi_end_loop(
            params, epi_tensors, epi_coord, tXrAcc, self.tiler_mn, None, self.tiled_copy, tile_coord_mnkl,
            varlen_manager, tidx,
        )
        tXrOut = cute.make_rmem_tensor_like(tXgOut)
        tXrOut.store(tXrAcc.load().to(self.dtype))
        if row_ok:
            copy_utils.copy(tXrOut, tXgOut, pred=tXpX)
        store_ops = self._epi_store_ops()
        for i in cutlass.range_constexpr(len(store_ops)):
            store_ops[i].store(self, getattr(params, store_ops[i].name), aux_outs[i], tile_coord_mnkl, tidx)
        self.epi_end(params, epi_tensors, self.tiler_mn, None, self.tiled_copy, tile_coord_mnkl, varlen_manager, tidx)

    @cute.jit
    def _weight(self, mW, row, row_ok, k):
        w = Float32(0.0)
        if row_ok:
            w = Float32(mW[row, k])
        return w


_cache = {}


def compile_reduce_over_k(dtype, topk, has_w, arch, cls=ReduceOverK, epi_keys=(), **config):
    """The tvm-ffi compiled kernel: fn(x (T, K, H), w (T, K) | None, out (T, H), epilogue_args). cls = ReduceOverK
    or a @reduce_epilogue mint; epi_keys = ((op name, op.host_arg_key(value)), ...) over cls._epi_ops, from which
    each op rebuilds its fake argument."""
    key = (cls, dtype, topk, has_w, arch, epi_keys, tuple(sorted(config.items())))
    if key not in _cache:
        kernel = cls(dtype, topk, has_w, arch, **config)
        t, h = cute.sym_int(), cute.sym_int()
        x = make_fake_tensor(dtype, (t, topk, h), divisibility=kernel.vec, leading_dim=2)
        w = make_fake_tensor(Float32, (t, topk), divisibility=1, leading_dim=1) if has_w else None
        out = make_fake_tensor(dtype, (t, h), divisibility=kernel.vec, leading_dim=1)
        args = None
        if cls.EpilogueArguments is not None:
            fctx = FakeArgCtx(m=t, n=h, k=None, l=cute.sym_int(), batched=True, varlen_m=False)
            ops = {op.name: op for op in cls._epi_ops}
            args = cls.EpilogueArguments(**{name: ops[name].host_fake_arg(k, fctx) for name, k in epi_keys if k is not None})
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        _cache[key] = cute.compile(kernel, x, w, out, args, stream, options="--enable-tvm-ffi")
    return _cache[key]


def reduce_over_k(x, w, out, epilogue=None, epi_args=None, **config):
    """out (T, H) = sum over k of w[t, k] * x[t, k, :] (plain sum when w is None); x (T, K, H) contiguous rows,
    w (T, K) or (T * K,) fp32, out same dtype as x. Launches on the current stream. epilogue: a @reduce_epilogue
    mod applied to the sum, its operands / outputs / reduce buffers in epi_args by name (returns the finalized
    reduces, as EpiMod.gemm does)."""
    T, K, H = x.shape
    dtype = torch2cute_dtype_map[x.dtype]
    assert out.shape == (T, H) and out.dtype == x.dtype and x.stride(2) == 1 and out.stride(1) == 1
    assert H % (128 // dtype.width) == 0, "16 B vectors must not straddle a row"
    if w is not None:
        assert w.dtype == torch.float32
        w = w.view(T, K)
    arch = get_device_capacity(x.device)[0] * 10
    if epilogue is not None:
        return epilogue(x, w, out, epi_args or {}, arch, **config)
    compile_reduce_over_k(dtype, K, w is not None, arch, **config)(x, w, out, None)
