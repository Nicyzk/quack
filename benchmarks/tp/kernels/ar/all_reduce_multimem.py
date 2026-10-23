# Referenced from: duckpond/kernels/rs/reduce_scatter_multimem.py
"""All-reduce (CuTe-DSL, multimem two-shot): each rank switch-reduces its slab from the input
multicast address (multimem.ld_reduce) and multimem-stores the sum to the output multicast
address, so every rank ends with the full result. NVLS-style traffic (~S per direction per
rank) — the model that lets NCCL beat any pull-based two-shot at large shapes."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # package-qualified: hosts own bare `utils`
from kernels.misc.utils import capture_graph, import_nvshmem, on_finalize

import cutlass
import cutlass.utils as utils
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from kernels.rs.reduce_scatter_primitives import (
    RS_WARPGROUP_THREADS,
    multimem_rs_tv,
    _multimem_ld_reduce_128b,
)

nvshmem = import_nvshmem()


@cute.jit
def multimem_ar_tile(in_mc_tile, out_mc_tile, lane, frags_in_flight: cutlass.Constexpr = 8):
    """multimem_rs_tile with a multicast store: ld_reduce across peers, multimem_st the sum."""
    dtype = in_mc_tile.element_type
    ld_reduce = _multimem_ld_reduce_128b(dtype)
    assert cute.size(in_mc_tile, mode=[1]) == 32 * (128 // dtype.width), "tile_n must be 32*vec"
    thr_layout, val_layout, _ = multimem_rs_tv(dtype, tile_m=cute.size(in_mc_tile, mode=[0]))
    thr_copy = cute.make_tiled_copy_tv(
        cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), dtype, num_bits_per_copy=128),
        thr_layout,
        val_layout,
    ).get_slice(lane)
    tIn = thr_copy.partition_S(in_mc_tile)
    tOut = thr_copy.partition_D(out_mc_tile)
    (_, rest_m), _, _ = tIn.shape
    assert rest_m % frags_in_flight == 0, "frags_in_flight must divide rest_m (= tile_m // 4)"
    for lo in cutlass.range_constexpr(0, rest_m, frags_in_flight):
        frags = []
        for k in cutlass.range_constexpr(frags_in_flight):  # issue all loads before any store
            frags.append(ld_reduce(tIn[(None, lo + k), 0, 0].iterator))
        for k in cutlass.range_constexpr(frags_in_flight):
            utils.distributed.multimem_st_4xb32(tOut[(None, lo + k), 0, 0].iterator, *frags[k])


@cute.kernel
def all_reduce_multimem_kernel(
    gIn: cute.Tensor,
    gOut: cute.Tensor,
    flags: cute.Tensor,
    flags_mc: cute.Tensor,
    world_size: cutlass.Constexpr,
    frags_in_flight: cutlass.Constexpr = 8,
):
    """Rank barriers live in the kernel, per-CTA slot, the cutlass fused-example method
    (elected multimem_red_add1 release + CAS(world->0) acquire spin): entry slot bidx before
    the loads, exit slot grid+bidx after the stores. Entry releases at gpu scope (it fences
    only this rank's local partial, which peers read from this L2; the sys variant costs
    ~4 us more for nothing — results/2026-08-31/gemm_ar.md); exit at sys scope (it fences
    this rank's remote multimem stores). The entry/exit alternation keeps the CAS reset
    replay-safe (a peer cannot bank a next-phase token before this rank consumed the phase)."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    gdim, _, _ = cute.arch.grid_dim()
    if tidx < 32:  # entry: peers' inputs committed at their producer's grid boundary
        with cute.arch.elect_one():
            utils.distributed.multimem_red_add1(
                lock_ptr=flags_mc.iterator + bidx, scope="gpu", order="release"
            )
            utils.distributed.spin_lock_atom_cas_acquire_wait(
                lock_ptr=flags.iterator + bidx,
                expected_val=world_size,
                reset_val=0,
                scope="sys",
            )
    cute.arch.barrier()
    blk_coord = ((None, None), bidx)
    multimem_ar_tile(gIn[blk_coord], gOut[blk_coord], tidx, frags_in_flight=frags_in_flight)
    cute.arch.barrier()  # CTA's multimem stores precede the elected release
    if tidx < 32:  # exit: no rank leaves while another still ld_reduces its slab
        with cute.arch.elect_one():
            utils.distributed.multimem_red_add1(
                lock_ptr=flags_mc.iterator + gdim + bidx, scope="sys", order="release"
            )
            utils.distributed.spin_lock_atom_cas_acquire_wait(
                lock_ptr=flags.iterator + gdim + bidx,
                expected_val=world_size,
                reset_val=0,
                scope="sys",
            )


@cute.jit
def all_reduce_multimem(
    mIn: cute.Tensor,
    mOut: cute.Tensor,
    flags: cute.Tensor,
    flags_mc: cute.Tensor,
    stream,
    world_size: cutlass.Constexpr,
    tile_m: cutlass.Constexpr = 128,
    copy_bits: cutlass.Constexpr = 128,
    frags_in_flight: cutlass.Constexpr = 8,
):
    _, _, tiler_mn = multimem_rs_tv(mIn.element_type, tile_m=tile_m, copy_bits=copy_bits)
    gIn = cute.zipped_divide(mIn, tiler_mn)
    gOut = cute.zipped_divide(mOut, tiler_mn)
    all_reduce_multimem_kernel(
        gIn, gOut, flags, flags_mc, world_size=world_size, frags_in_flight=frags_in_flight
    ).launch(
        grid=[cute.size(gOut, mode=[1]), 1, 1],
        block=[RS_WARPGROUP_THREADS, 1, 1],
        stream=stream,
    )


def make_ar_flags(slab, n, tile_m, dtype_width=16):
    """Zeroed symmetric flag array (2 slots per CTA tile: entry [0, G), exit [G, 2G))
    and its multicast view; caller owns freeing via on_finalize."""
    import torch

    tile_n = 32 * (128 // dtype_width)
    num_tiles = (slab // tile_m) * (n // tile_n)
    flags = nvshmem.core.tensor((2 * num_tiles,), dtype=torch.int32)
    flags.fill_(0)
    flags_mc = nvshmem.core.get_multicast_tensor(nvshmem.core.Teams.TEAM_NODE, flags)
    on_finalize(lambda: (nvshmem.core.free_tensor(flags_mc), nvshmem.core.free_tensor(flags)))
    return flags, flags_mc


def all_reduce(dtype, out, inp, rank, world, frags_in_flight=8):
    """Two-shot multimem AR over each rank's slab. Contract: returns a nullary CUDA-graph
    replay of the single kernel baked to the full-size (M, N) symmetric buffers — both
    rank barriers run inside the kernel (per-CTA slot, cutlass exit_bar method)."""
    slab = inp.shape[0] // world
    # Clamp tile to the slab: a 128-tall tile over a shorter slab redundantly re-reduces
    # peers' rows (correct values, 128/slab x traffic) and runs off the buffer end.
    # e.g. M=64 TP=4: slab=16 rows under a 128-tall tile -> 8x traffic + OOB.
    tile_m = min(128, slab)
    assert slab % tile_m == 0 and tile_m % 4 == 0, f"slab {slab} not tileable"
    inp_mc = nvshmem.core.get_multicast_tensor(nvshmem.core.Teams.TEAM_NODE, inp)
    out_mc = nvshmem.core.get_multicast_tensor(nvshmem.core.Teams.TEAM_NODE, out)
    on_finalize(lambda: (nvshmem.core.free_tensor(inp_mc), nvshmem.core.free_tensor(out_mc)))
    in_slab = inp_mc[rank * slab:(rank + 1) * slab]
    out_slab = out_mc[rank * slab:(rank + 1) * slab]
    flags, flags_mc = make_ar_flags(slab, inp.shape[1], tile_m, dtype_width=inp.element_size() * 8)
    fn = cute.compile(
        all_reduce_multimem,
        from_dlpack(in_slab, enable_tvm_ffi=True),
        from_dlpack(out_slab, enable_tvm_ffi=True),
        from_dlpack(flags, enable_tvm_ffi=True),
        from_dlpack(flags_mc, enable_tvm_ffi=True),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        world_size=world,
        tile_m=tile_m,
        frags_in_flight=min(frags_in_flight, tile_m // 4),
        options="--enable-tvm-ffi",
    )

    def run():
        fn(in_slab, out_slab, flags, flags_mc)  # live mc slabs bind the env (capturing) stream

    return capture_graph(run)
