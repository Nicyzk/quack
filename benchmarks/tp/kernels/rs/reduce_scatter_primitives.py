"""Reduce-scatter device primitives (nvshmem-free): one-warpgroup multimem reduce + store helpers.
Shared by the standalone reduce_scatter_multimem kernel and the fused GEMM RS warp group."""
import cutlass
import cutlass.utils as utils
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op


# One warpgroup (4 warps); thr is fixed (4,32) so a 256-wide tile maps to exactly these 128 threads.
RS_WARPGROUP_THREADS = 128


@dsl_user_op
def st_global_4xb32(ptr, x, y, z, w, *, loc=None, ip=None):
    ptr_int = ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    llvm.inline_asm(
        None,
        [ptr_int, x, y, z, w],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_global_sys_relaxed_4xb32(ptr, x, y, z, w, *, loc=None, ip=None):
    """System-scope relaxed 128b store — cross-GPU visible; for peer-memory writes in the fused RS."""
    ptr_int = ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    llvm.inline_asm(
        None,
        [ptr_int, x, y, z, w],
        "st.global.sys.relaxed.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


def multimem_rs_tv(dtype, tile_m=128, *, copy_bits=128):
    """One-warpgroup TV layout (thr fixed 4x32, tile_n pinned to 32*vec=256); returns (thr, val, tiler_mn)."""
    vec = copy_bits // dtype.width
    assert tile_m % 4 == 0, "tile_m must be a multiple of thr_M=4"
    thr_layout = cute.make_ordered_layout((4, 32), order=(1, 0))   # 128 threads, only layout valid for N=256
    val_layout = cute.make_ordered_layout((tile_m // 4, vec), order=(1, 0))
    tiler_mn, _ = cute.make_layout_tv(thr_layout, val_layout)
    return thr_layout, val_layout, tiler_mn


def _multimem_ld_reduce_128b(dtype):
    """Pick the 128-bit multimem.ld_reduce(add) variant for dtype; all return 4x b32 (x, y, z, w)."""
    if dtype == cutlass.Float16:
        return utils.distributed.multimem_ld_reduce_8xf16
    elif dtype == cutlass.Float32:
        return utils.distributed.multimem_ld_reduce_4xf32
    elif dtype == cutlass.BFloat16:
        return utils.distributed.multimem_ld_reduce_8xbf16
    elif dtype == cutlass.Float8E4M3FN:
        return utils.distributed.multimem_ld_reduce_16xe4m3
    elif dtype == cutlass.Float8E5M2:
        return utils.distributed.multimem_ld_reduce_16xe5m2
    raise NotImplementedError(f"multimem_rs_tile: unsupported dtype {dtype}")


@cute.jit
def multimem_rs_tile(mc_tile, out_tile, lane, frags_in_flight: cutlass.Constexpr = 8):
    """One warpgroup (lane 0..127) multimem-reduces mc_tile across peers, stores to out_tile (tile_m/N from
    mc_tile; N = 32*vec, vec = 128b/elem). frags_in_flight batches that many ld_reduce loads before draining
    to stores so their NVLink latencies overlap (MLP); 1 = one at a time. Must divide rest_m (= tile_m // 4)."""
    dtype = mc_tile.element_type
    ld_reduce = _multimem_ld_reduce_128b(dtype)
    assert cute.size(mc_tile, mode=[1]) == 32 * (128 // dtype.width), "tile_n must be 32*vec (128b per N-fragment)"
    thr_layout, val_layout, _ = multimem_rs_tv(dtype, tile_m=cute.size(mc_tile, mode=[0]))
    thr_copy = cute.make_tiled_copy_tv(
        cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), dtype, num_bits_per_copy=128),
        thr_layout,
        val_layout,
    ).get_slice(lane)
    tIn = thr_copy.partition_S(mc_tile)
    tOut = thr_copy.partition_D(out_tile)
    (_, rest_m), _, _ = tIn.shape
    assert rest_m % frags_in_flight == 0, "frags_in_flight must divide rest_m (= tile_m // 4)"
    for lo in cutlass.range_constexpr(0, rest_m, frags_in_flight):
        frags = []
        for k in cutlass.range_constexpr(frags_in_flight):  # issue all loads before any store -> in flight
            frags.append(ld_reduce(tIn[(None, lo + k), 0, 0].iterator))
        for k in cutlass.range_constexpr(frags_in_flight):
            st_global_4xb32(tOut[(None, lo + k), 0, 0].iterator, *frags[k])
