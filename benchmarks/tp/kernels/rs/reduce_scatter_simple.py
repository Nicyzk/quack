# Referenced from: https://github.com/NVIDIA/cutlass/examples/python/CuTeDSL/cute/blackwell/kernel/distributed/all_reduce_simple.py
"""Reduce-scatter (CuTe-DSL, SIMT pull): each rank sums all peers' slabs in f32 into its local output.
reduce_scatter(dtype, out, inp, rank, world) returns a nullary callable (a CUDA-graph replay here)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # package-qualified: hosts own bare `utils`
from kernels.misc.sync import make_barrier
from kernels.misc.utils import capture_graph, import_nvshmem, on_finalize

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

nvshmem = import_nvshmem()


@cute.kernel
def reduce_scatter_simple_kernel(
    inputs: list[cute.Tensor],
    gOut: cute.Tensor,
    thr_layout: cute.Layout,
    val_layout: cute.Layout,
    local_rank: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    blk_coord = ((None, None), bidx)
    local_tile_out = gOut[blk_coord]
    local_tile_list = [t[blk_coord] for t in inputs]

    assert all(t.element_type == inputs[0].element_type for t in inputs)

    copy_atom_load = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        inputs[0].element_type,
    )
    copy_atom_store = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        inputs[0].element_type,
    )
    tiled_copy = cute.make_tiled_copy_tv(copy_atom_load, thr_layout, val_layout)
    thr_copy = tiled_copy.get_slice(tidx)

    thr_tensor_list = [thr_copy.partition_S(tensor) for tensor in local_tile_list]
    thr_out = thr_copy.partition_D(local_tile_out)
    frg_tensor_list = [cute.make_fragment_like(tensor) for tensor in thr_tensor_list]
    frg_acc = cute.make_rmem_tensor(thr_out.layout, cutlass.Float32)
    frg_acc.fill(0.0)

    world_size = len(thr_tensor_list)
    for i in cutlass.range_constexpr(world_size):
        # ring order: at each step ranks read a permutation of sources (no incast)
        peer = (local_rank + i) % world_size
        for rank_i in cutlass.range_constexpr(world_size):
            if peer == rank_i:
                cute.copy(copy_atom_load, thr_tensor_list[rank_i], frg_tensor_list[rank_i])
                tmp = frg_tensor_list[rank_i].load().to(cutlass.Float32) + frg_acc.load()
                frg_acc.store(tmp)

    frg_out = cute.make_fragment_like(thr_out)
    frg_out.store(frg_acc.load().to(thr_out.element_type))
    cute.copy(copy_atom_store, frg_out, thr_out)


@cute.jit
def reduce_scatter_simple(
    inputs: list[cute.Tensor],
    output: cute.Tensor,
    local_rank: cutlass.Constexpr,
    stream,
    copy_bits: cutlass.Constexpr = 128,
):
    dtype = inputs[0].element_type
    vector_size = copy_bits // dtype.width

    thr_layout = cute.make_ordered_layout((4, 32), order=(1, 0))
    val_layout = cute.make_ordered_layout((4, vector_size), order=(1, 0))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    divided_inputs = [cute.zipped_divide(tensor, tiler_mn) for tensor in inputs]
    gOut = cute.zipped_divide(output, tiler_mn)
    reduce_scatter_simple_kernel(
        divided_inputs,
        gOut,
        thr_layout,
        val_layout,
        local_rank,
    ).launch(
        grid=[cute.size(gOut, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
        stream=stream,
    )


def reduce_scatter(dtype, out, inp, rank, world):
    """SIMT pull RS. Contract: returns a nullary CUDA-graph replay of [bar_a; kernel; bar_b]
    baked to the symmetric buffers out (M//world, N) and inp (M, N)."""
    slab = inp.shape[0] // world
    bar_a, free_a = make_barrier(world)
    bar_b, free_b = make_barrier(world)
    on_finalize(free_a)
    on_finalize(free_b)
    peers = [nvshmem.core.get_peer_tensor(inp, r) for r in range(world)]
    in_dlp = [from_dlpack(p[rank * slab:(rank + 1) * slab, :], enable_tvm_ffi=True) for p in peers]
    out_dlp = from_dlpack(out, enable_tvm_ffi=True)
    fn = cute.compile(
        reduce_scatter_simple, in_dlp, out_dlp, rank,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )

    def run():
        bar_a()
        fn(in_dlp, out)  # live out binds the env (capturing) stream
        bar_b()

    return capture_graph(run)
