# Referenced from: https://github.com/NVIDIA/cutlass/examples/python/CuTeDSL/cute/blackwell/kernel/distributed/all_reduce_two_shot_multimem.py
"""Reduce-scatter (CuTe-DSL, multimem): one warpgroup per tile hardware-reduces across peers,
stores to local output. reduce_scatter(dtype, out, inp, rank, world) returns a nullary callable
(a CUDA-graph replay here)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # package-qualified: hosts own bare `utils`
from kernels.misc.sync import make_barrier
from kernels.misc.utils import capture_graph, import_nvshmem, on_finalize

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from kernels.rs.reduce_scatter_primitives import RS_WARPGROUP_THREADS, multimem_rs_tv, multimem_rs_tile

nvshmem = import_nvshmem()


@cute.kernel
def reduce_scatter_multimem_kernel(gIn: cute.Tensor, gOut: cute.Tensor, frags_in_flight: cutlass.Constexpr = 8):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    blk_coord = ((None, None), bidx)
    multimem_rs_tile(gIn[blk_coord], gOut[blk_coord], tidx, frags_in_flight=frags_in_flight)


@cute.jit
def reduce_scatter_multimem(
    mIn: cute.Tensor,
    mOut: cute.Tensor,
    stream,
    tile_m: cutlass.Constexpr = 128,
    copy_bits: cutlass.Constexpr = 128,
    frags_in_flight: cutlass.Constexpr = 8,
):
    _, _, tiler_mn = multimem_rs_tv(mIn.element_type, tile_m=tile_m, copy_bits=copy_bits)
    gIn = cute.zipped_divide(mIn, tiler_mn)
    gOut = cute.zipped_divide(mOut, tiler_mn)
    reduce_scatter_multimem_kernel(gIn, gOut, frags_in_flight=frags_in_flight).launch(
        grid=[cute.size(gOut, mode=[1]), 1, 1],
        block=[RS_WARPGROUP_THREADS, 1, 1],
        stream=stream,
    )


def reduce_scatter(dtype, out, inp, rank, world, frags_in_flight=8):
    """Multimem reduce over peers. Contract: returns a nullary CUDA-graph replay of
    [bar_a; kernel; bar_b] baked to the symmetric buffers out (M//world, N) and inp (M, N)."""
    slab = inp.shape[0] // world
    bar_a, free_a = make_barrier(world)
    bar_b, free_b = make_barrier(world)
    on_finalize(free_a)
    on_finalize(free_b)
    inp_mc = nvshmem.core.get_multicast_tensor(nvshmem.core.Teams.TEAM_NODE, inp)
    on_finalize(lambda: nvshmem.core.free_tensor(inp_mc))
    in_slab = inp_mc[rank * slab:(rank + 1) * slab]
    in_dlp = from_dlpack(in_slab, enable_tvm_ffi=True)
    out_dlp = from_dlpack(out, enable_tvm_ffi=True)
    fn = cute.compile(
        reduce_scatter_multimem, in_dlp, out_dlp,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        frags_in_flight=frags_in_flight,
        options="--enable-tvm-ffi",
    )

    def run():
        bar_a()
        fn(in_dlp, out)  # live out binds the env (capturing) stream
        bar_b()

    return capture_graph(run)
