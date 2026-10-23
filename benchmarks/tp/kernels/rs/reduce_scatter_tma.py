# Referenced from: https://github.com/NVIDIA/cutlass/examples/python/CuTeDSL/cute/blackwell/kernel/distributed/all_reduce_tma.py
"""Reduce-scatter (CuTe-DSL, TMA two-shot pull, persistent tile scheduler)."""

import cutlass
import cutlass.utils as utils
import cutlass.cute as cute
import cutlass.cute.testing as testing
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync

from quack.tile_scheduler import (
    TileScheduler,
    TileSchedulerArguments,
    TileSchedulerOptions,
    PersistenceMode,
    RasterOrderOption,
)
from quack.gemm_tvm_ffi_utils import make_scheduler_args, make_fake_scheduler_args
from quack.cute_dsl_utils import get_max_active_clusters


class ReduceScatterTmaKernel:

    _tile_m: int = 128
    _tile_n: int = 128
    _elems_per_cta: int = _tile_m * _tile_n  # one 2D (tile_m, tile_n) output tile per CTA
    _tma_threads: int = 32
    _consumer_threads: int = 128
    _threads_per_cta: int = _tma_threads + _consumer_threads
    _num_stages: int = 7

    def __init__(self, dtype):
        self.dtype = dtype

        # 2D output tile (tile_m, tile_n); SMEM holds num_stages of these for the peer pipeline
        self.smem_layout_shape = (self._tile_m, self._tile_n)
        self.tiler = self.smem_layout_shape

        # TMA transaction bytes (computed from dtype size)
        # dtype.width is in bits, divide by 8 to get bytes
        self.tma_bytes = (dtype.width // 8) * self._elems_per_cta

        # Dynamically create SharedStorage type based on dtype
        elems = self._elems_per_cta
        stages = self._num_stages

        @cute.struct
        class SharedStorage:
            mbar_array: cute.struct.MemRange[cutlass.Int64, stages * 2]
            smem_buffer: cute.struct.Align[
                cute.struct.MemRange[dtype, elems * stages],
                128,
            ]

        self._SharedStorage = SharedStorage

    @cute.jit
    def __call__(
        self,
        input_tensors: list[cute.Tensor],
        output_tensor_mc: cute.Tensor,
        local_rank: cutlass.Constexpr,
        world_size: cutlass.Constexpr,
        scheduler_args: TileSchedulerOptions,
        stream,
        d_stored_flags: cute.Tensor = None,
        mma_tile_shape: cutlass.Constexpr = None,
    ):
        # Layout validation
        ref_layout = input_tensors[0].layout
        ref_size = cute.size(ref_layout)
        ref_cosize = cute.cosize(ref_layout)

        # Check compact: size == cosize (no holes in memory)
        assert ref_size == ref_cosize, (
            f"Input tensor must be compact: size={ref_size}, cosize={ref_cosize}"
        )
        assert self.tma_bytes % 16 == 0, f"Not aligned to 16B, TMA should not be used."

        # Check all input tensors have the same layout
        for i in cutlass.range_constexpr(world_size):
            assert input_tensors[i].layout == ref_layout, (
                f"All input tensors must have the same layout. "
                f"input_tensors[0].layout={ref_layout}, "
                f"input_tensors[{i}].layout={input_tensors[i].layout}"
            )

        # Check output tensor has the same layout
        assert output_tensor_mc.layout == ref_layout, (
            f"Output tensor must have the same layout as input tensors. "
            f"input layout={ref_layout}, output layout={output_tensor_mc.layout}"
        )

        # ======================================================================
        # Extract tensor info
        # ======================================================================
        # Verify dtype matches
        assert input_tensors[0].element_type == self.dtype, (
            f"Input tensor dtype mismatch: expected {self.dtype}, "
            f"got {input_tensors[0].element_type}"
        )

        # input/output are 2D (rows, cols); tile each into (tile_m, tile_n) blocks
        rows = cute.size(input_tensors[0], mode=[0])
        cols = cute.size(input_tensors[0], mode=[1])

        # SMEM layout: row-major (tile_n contiguous) to match the row-major (rows, cols) gmem
        smem_layout = cute.make_layout(self.smem_layout_shape, stride=(self._tile_n, 1))

        # Create TMA load descriptors (one per rank), over the 2D tensors directly
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp()
        tma_load_atoms = []
        tma_load_tensors = []

        for i in cutlass.range_constexpr(world_size):
            tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(
                tma_load_op,
                input_tensors[i],
                smem_layout,
                self.tiler,
            )
            tma_load_atoms.append(tma_atom)
            tma_load_tensors.append(tma_tensor)

        # Create TMA store descriptor
        tma_store_op = cpasync.CopyBulkTensorTileS2GOp()
        tma_store_atom, tma_store_tensor = cpasync.make_tiled_tma_atom(
            tma_store_op,
            output_tensor_mc,
            smem_layout,
            self.tiler,
        )

        num_tiles_m = cute.ceil_div(rows, self._tile_m)
        num_tiles_n = cute.ceil_div(cols, self._tile_n)
        num_tiles = num_tiles_m * num_tiles_n

        # Same as gemm_base.get_scheduler_arguments: combine the make_scheduler_args knobs with
        # our tile grid. NONE persistence for now: one CTA per tile, swizzle only (persistent
        # grid is the follow-up).
        # When gated by the GEMM (overlap), force AlongN to match the GEMM's tile-production order
        # so each rank's M-slice becomes ready together (the heuristic would pick AlongM here, a
        # mismatch). Standalone (no gate) keeps the heuristic -- RS order is perf-neutral there.
        rs_raster_order = (
            RasterOrderOption.AlongN
            if cutlass.const_expr(d_stored_flags is not None)
            else scheduler_args.raster_order
        )
        tile_sched_args = TileSchedulerArguments(
            problem_shape_ntile_mnl=(num_tiles_m, num_tiles_n, cutlass.Int32(1)),
            raster_order=rs_raster_order,
            group_size=scheduler_args.max_swizzle_size,
            cluster_shape_mnk=(1, 1, 1),
            tile_count_semaphore=scheduler_args.tile_count_semaphore,
            batch_idx_permute=scheduler_args.batch_idx_permute,
            persistence_mode=PersistenceMode.NONE,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        # persistent grid: one wave of CTAs, each striding over the tile space (STATIC schedule)
        num_persistent = cutlass.min(num_tiles, scheduler_args.max_active_clusters)

        # SMEM size from SharedStorage
        smem_bytes = self._SharedStorage.size_in_bytes()

        # S2.1 gate (None standalone): the RS tile must divide the GEMM CTA(flag)-tile so each
        # RS tile maps to exactly one flag; everything else is derived in-kernel.
        if cutlass.const_expr(d_stored_flags is not None):
            assert mma_tile_shape[0] % self._tile_m == 0 and mma_tile_shape[1] % self._tile_n == 0

        # Launch kernel
        self.kernel(
            tma_load_atoms,
            tma_load_tensors,
            tma_store_atom,
            tma_store_tensor,
            local_rank,
            world_size,
            num_tiles,
            tile_sched_params,
            d_stored_flags,
            mma_tile_shape,
        ).launch(
            grid=[num_persistent, 1, 1],
            block=[self._threads_per_cta, 1, 1],
            smem=smem_bytes,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        # TMA atoms and tensors for loading from each rank
        tma_load_atoms: list[cute.CopyAtom],
        tma_load_tensors: list[cute.Tensor],
        # TMA atom and tensor for storing to multicast address
        tma_store_atom: cute.CopyAtom,
        tma_store_tensor: cute.Tensor,
        # Rank info
        local_rank: cutlass.Constexpr,
        world_size: cutlass.Constexpr,
        num_tiles: cutlass.Int32,
        tile_sched_params: TileScheduler.Params,
        d_stored_flags: cute.Tensor,
        mma_tile_shape: cutlass.Constexpr,
    ):
        # ======================================================================
        # Thread/Block indexing
        # ======================================================================
        tidx = cute.arch.thread_idx()[0]
        bidx = cute.arch.block_idx()[0]
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        # ======================================================================
        # SMEM allocation
        # ======================================================================
        staged_smem_layout = cute.make_layout(
            (*self.smem_layout_shape, self._num_stages),
            stride=(self._tile_n, 1, self._tile_m * self._tile_n),
        )

        smem = utils.SmemAllocator()
        storage = smem.allocate(self._SharedStorage)
        mbar_ptr = storage.mbar_array.data_ptr()
        staged_smem_tensor = storage.smem_buffer.get_tensor(staged_smem_layout)

        # ======================================================================
        # TMA Pipeline setup
        # ======================================================================
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self._consumer_threads
        )

        tma_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=mbar_ptr,
            num_stages=self._num_stages,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=self.tma_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )

        # Persistent: each CTA walks tiles work_idx = bidx, bidx + grid, ... with quack's
        # swizzle. Pipeline states are created once and carried across tiles (the peer pipeline
        # cycles continuously). Producer/store warp and consumer warps each run their own tile
        # loop and rendezvous twice per tile (after the loads/accumulate, and after the store).
        num_persistent = cute.arch.grid_dim()[0]
        sched = TileScheduler.create(tile_sched_params)

        if warp_idx == 0:
            producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self._num_stages
            )
            # S2.1 gate constants (cluster=1 -> ncluster == ntile). q_* = RS tiles per flag tile.
            # d_stored_flags is THIS rank's slab of the flag grid, so the index is slab-local.
            if cutlass.const_expr(d_stored_flags is not None):
                q_m = mma_tile_shape[0] // self._tile_m
                q_n = mma_tile_shape[1] // self._tile_n
                n_flags_n = sched.params.problem_shape_ncluster_mnl[1] // q_n  # N flag count
            work_idx = bidx
            while work_idx < num_tiles:
                tile_m, tile_n = sched._swizzle_cta(work_idx)  # cluster=1 -> cta id == cid
                # Gate: wait until every rank has stored the GEMM flag-tile covering this RS tile.
                if cutlass.const_expr(d_stored_flags is not None):
                    if cute.arch.lane_idx() == 0:
                        flag_idx = (tile_m // q_m) * n_flags_n + tile_n // q_n
                        utils.distributed.spin_lock_ld_lt_relaxed_wait(
                            d_stored_flags.iterator + flag_idx,
                            expected_val=world_size,
                            scope="sys",
                        )
                    cute.arch.sync_warp()
                for i in cutlass.range_constexpr(world_size):
                    # ring order: at each step ranks read a permutation of sources (no incast)
                    peer = (local_rank + i) % world_size
                    tma_pipeline.producer_acquire(producer_state)
                    stage_idx = producer_state.index
                    smem_tile = cute.slice_(staged_smem_tensor, (None, None, stage_idx))
                    s_tile_flat = cute.group_modes(smem_tile, 0, cute.rank(smem_tile))
                    for rank_i in cutlass.range_constexpr(world_size):
                        if peer == rank_i:
                            g_tensor_tiled = cute.zipped_divide(tma_load_tensors[rank_i], self.tiler)
                            g_tile = g_tensor_tiled[(None, None), (tile_m, tile_n)]
                            g_tile_flat = cute.group_modes(g_tile, 0, cute.rank(g_tile))
                            s_part, g_part = cute.nvgpu.cpasync.tma_partition(
                                tma_load_atoms[rank_i], 0, cute.make_layout(1),
                                s_tile_flat, g_tile_flat,
                            )
                            cute.copy(
                                tma_load_atoms[rank_i], g_part, s_part,
                                tma_bar_ptr=tma_pipeline.producer_get_barrier(producer_state),
                            )
                    tma_pipeline.producer_commit(producer_state)
                    producer_state.advance()
                cute.arch.sync_threads()  # consumers have written the reduced tile to stage 0
                cute.arch.fence_proxy("async.shared", space="cta")
                smem_tile_out = cute.slice_(staged_smem_tensor, (None, None, 0))
                g_output_tiled = cute.zipped_divide(tma_store_tensor, self.tiler)
                g_output_tile = g_output_tiled[(None, None), (tile_m, tile_n)]
                g_out_flat = cute.group_modes(g_output_tile, 0, cute.rank(g_output_tile))
                s_out_flat = cute.group_modes(smem_tile_out, 0, cute.rank(smem_tile_out))
                s_part, g_part = cute.nvgpu.cpasync.tma_partition(
                    tma_store_atom, 0, cute.make_layout(1), s_out_flat, g_out_flat,
                )
                cute.copy(tma_store_atom, s_part, g_part)
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0)
                cute.arch.sync_threads()  # store done -> stage 0 free for the next tile
                work_idx += num_persistent
        else:
            consumer_tid = tidx - self._tma_threads
            vec_size = 4
            chunk_size = vec_size * self._consumer_threads
            # row-major tile is physically contiguous -> force a flat (elems,) view (coalesce
            # leaves it 2D since it's mode-1-major, which breaks the per-thread tiling).
            smem_tensor_wo_stage = cute.make_tensor(
                cute.slice_(staged_smem_tensor, (None, None, 0)).iterator,
                cute.make_layout(self._elems_per_cta),
            )
            smem_tensor_tiled_by_thr_vec = cute.zipped_divide(smem_tensor_wo_stage, (chunk_size,))
            smem_tensor_tiled_by_thr_vec_tiled_by_vec = cute.logical_divide(
                smem_tensor_tiled_by_thr_vec, (vec_size,)
            )
            per_thread_smem_tensor = cute.slice_(
                smem_tensor_tiled_by_thr_vec_tiled_by_vec, ((None, consumer_tid), None)
            )
            accum = cute.make_rmem_tensor(per_thread_smem_tensor.layout, cutlass.Float32)
            consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self._num_stages
            )
            work_idx = bidx
            while work_idx < num_tiles:
                accum.fill(0.0)
                for rank_i in cutlass.range_constexpr(world_size):
                    tma_pipeline.consumer_wait(consumer_state)
                    stage_idx = consumer_state.index
                    smem_tile = cute.make_tensor(
                        cute.slice_(staged_smem_tensor, (None, None, stage_idx)).iterator,
                        cute.make_layout(self._elems_per_cta),
                    )
                    smem_tiled_by_thr_vec = cute.zipped_divide(smem_tile, (chunk_size,))
                    smem_tiled_by_thr_vec_tiled_by_vec = cute.logical_divide(
                        smem_tiled_by_thr_vec, (vec_size,)
                    )
                    per_thread_smem_view = cute.slice_(
                        smem_tiled_by_thr_vec_tiled_by_vec, ((None, consumer_tid), None)
                    )
                    fragment = per_thread_smem_view.load()
                    accum.store(accum.load() + fragment.to(cutlass.Float32))
                    tma_pipeline.sync_object_empty.arrive(
                        consumer_state.index, tma_pipeline.consumer_mask
                    )
                    consumer_state.advance()
                per_thread_smem_tensor.store(accum.load().to(self.dtype))
                cute.arch.sync_threads()  # hand stage 0 to the producer/store warp
                cute.arch.sync_threads()  # wait for the store before reusing smem
                work_idx += num_persistent


# nvshmem runtime imports for the reduce_scatter factory below.
import os
import sys

from cutlass.cute.runtime import from_dlpack

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # package-qualified: hosts own bare `utils`
from kernels.misc.utils import capture_graph, import_nvshmem, on_finalize
from kernels.misc.sync import make_barrier

nvshmem = import_nvshmem()


def reduce_scatter(dtype, out, inp, rank, world):
    """TMA two-shot pull RS (persistent). Contract: returns a nullary CUDA-graph replay of
    [bar_a; kernel; bar_b] baked to the symmetric buffers out (M//world, N) and inp (M, N) —
    bench, gate, and PKB solutions all call the same fn()."""
    slab = inp.shape[0] // world
    bar_a, free_a = make_barrier(world)
    bar_b, free_b = make_barrier(world)
    on_finalize(free_a)
    on_finalize(free_b)
    peers = [nvshmem.core.get_peer_tensor(inp, r) for r in range(world)]
    in_dlp = [from_dlpack(p[rank * slab:(rank + 1) * slab], enable_tvm_ffi=True) for p in peers]
    out_dlp = from_dlpack(out, enable_tvm_ffi=True)
    fn = cute.compile(
        ReduceScatterTmaKernel(dtype), in_dlp, out_dlp, rank, world,
        make_fake_scheduler_args(False, False, 1),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )
    sched = make_scheduler_args(get_max_active_clusters(1), cutlass.Int32(8), None)

    def run():
        bar_a()
        fn(in_dlp, out, sched)  # live out binds the env (capturing) stream
        bar_b()

    return capture_graph(run)
