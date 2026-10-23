"""Decode-payload standalone AR sweep -- the AR halves of gemm_ar/bench_baselines'
unfused chains: the cutlass all_reduce_one_shot_lamport example (vendored below,
bf16-generalized), duckpond's two-shot multimem AR, flashinfer trtllm kAllReduce
(its own oneshot/twoshot heuristic) vs NCCL. The vendored all_reduce_two_shot_multimem / all_reduce_tma arms remain
importable. Every arm bf16 with the same data: one rank-seeded fp32 randn cast
to bf16 (make_base_input).

ONE method for every arm: the quack-autotuner methodology
(bench_rotate.bench_arm) over N_SETS symm operand sets as arm-owned arg_sets
-- the whole timed window is one captured graph, replayed once. The per-set
graph round-robin this replaced paid a ~3 us replay boundary per iteration
and let flag-polling kernels absorb the peer's boundary into their poll
phase (nsys 2026-08-24: trtllm oneshot kernel 8.1 us back-to-back vs 4.9-6.1
GEMM-spaced -- near-2x on the reported cell). The lamport arms advance their
slot phase per call (cutlass: host counter, N_SETS a multiple of 3 keeps
replays in phase; trtllm: device-side flag, any count). Full L2-cold turnover
would need ~3xL2/payload (~hundreds of symm sets); N_SETS is capped --
identical for every arm, so the comparison is exact even though inputs stay
partially L2-resident. Every arm is ref-checked against NCCL before timing.

Usage: torchrun --nproc_per_node=<world> bench_ar.py --m 64 --n 4096
"""
import argparse
import os
import sys

import torch
import torch.distributed as dist
import cuda.bindings.driver as cuda_drv
import cutlass
import cutlass.utils as utils
import cutlass.cute as cute
from cutlass import Boolean, Int32, const_expr
from cutlass.cutlass_dsl import T
from cutlass._mlir.dialects import vector
from cutlass.cute.runtime import from_dlpack

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # package-qualified: hosts own bare `utils`
from kernels.misc.utils import (
    torchrun_init_nvshmem as init_nvshmem,
    clean_print,
    on_finalize,
)
from kernels.misc.bench_rotate import Arm, bench_arm, run0
import nvshmem.core

from kernels.ar.ar_norm_lamport import PING_PONG_SIZE, make_lamport_buffers, _is_not_neg0
from kernels.rs.reduce_scatter_primitives import _multimem_ld_reduce_128b

CUTLASS_AR_DIR = "~/cutlass/examples/python/CuTeDSL/cute/blackwell/kernel/distributed"
N_SETS = 24  # rotated input sets (multiple of the lamport slot count 3)
# Arm builders return (Arm, bf16 NCCL ref, per-set local inputs); the decomposed
# GEMM+AR bench writes its GEMM output into those inputs and sets n_sets itself.


def make_base_input(M, N):
    """One data distribution for every arm: rank-seeded fp32 randn cast to bf16."""
    torch.manual_seed(dist.get_rank())
    return torch.randn(M, N, dtype=torch.float32, device="cuda").to(torch.bfloat16)


# Referenced-from: all_reduce_one_shot_lamport.AllReduceOneShotLamportKernel — verbatim
# structure, dtype-generalized like ar_norm_lamport.py: i32-word sentinel (0x80008000
# packs two bf16 -0.0), clear pattern built by bitcast, trtllm's -0.0 write flush.
class AllReduceOneShotLamportKernelBf16:
    @cute.jit
    def __call__(self, rank: cutlass.Constexpr, world_size: cutlass.Constexpr,
                 signal: Int32, local_input: cute.Tensor, local_output: cute.Tensor,
                 buffers: list, stream: cuda_drv.CUstream):
        copy_bits = 128
        dtype = local_input.element_type
        vector_size = copy_bits // dtype.width
        thr_layout = cute.make_ordered_layout((4, 32), order=(1, 0))
        val_layout = cute.make_ordered_layout((1, vector_size), order=(1, 0))
        tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
        grouped_buffers = [cute.group_modes(buffer, 0, 2) for buffer in buffers]
        tiled_buffers = [
            cute.zipped_divide(buffer, (tiler_mn, world_size, PING_PONG_SIZE))
            for buffer in grouped_buffers
        ]
        tiled_input = cute.zipped_divide(local_input, tiler_mn)
        tiled_output = cute.zipped_divide(local_output, tiler_mn)
        self.kernel(tiled_buffers, tiled_input, tiled_output, thr_layout, val_layout,
                    signal, rank).launch(
            grid=[cute.size(tiled_input, mode=[1]), 1, 1],
            block=[cute.size(tv_layout, mode=[0]), 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, buffers: list, local_input: cute.Tensor, local_output: cute.Tensor,
               thr_layout: cute.Layout, val_layout: cute.Layout, signal: Int32,
               rank: cutlass.Constexpr):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        ping = signal % 3
        pong = (signal + 1) % 3
        buffer_local = buffers[rank]
        cta_coord = ((None, None), bidx)
        local_tile_in = local_input[cta_coord]
        local_tile_out = local_output[cta_coord]
        ping_coord = (((None, None), None, ping), bidx)
        pong_coord = (((None, None), None, pong), bidx)
        read_buffer = buffer_local[ping_coord]
        clear_buffer = buffer_local[pong_coord]
        write_coord = (((None, None), rank, ping), bidx)
        write_buffers = [buffer[write_coord] for buffer in buffers]
        dtype = buffers[0].element_type
        sentinel_word = const_expr(0x80008000 if dtype.width == 16 else 0x80000000)
        vec_t = T.f32 if dtype.width == 32 else (T.bf16 if dtype == cutlass.BFloat16 else T.f16)
        copy_atom_load = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(), dtype, num_bits_per_copy=128,
            memory_scope=cute.nvgpu.common.MemoryScope.SYS,
            memory_order=cute.nvgpu.common.MemoryOrder.VOLATILE,
        )
        copy_atom_store = cute.make_copy_atom(
            cute.nvgpu.CopyR2GOp(), dtype, num_bits_per_copy=128,
            memory_scope=cute.nvgpu.common.MemoryScope.SYS,
            memory_order=cute.nvgpu.common.MemoryOrder.VOLATILE,
        )
        tiled_copy = cute.make_tiled_copy_tv(copy_atom_load, thr_layout, val_layout)
        thr_copy = tiled_copy.get_slice(tidx)
        thr_write_buffer_list = [thr_copy.partition_D(t) for t in write_buffers]
        thr_read_buffer = thr_copy.partition_S(read_buffer)
        thr_clear_buffer = thr_copy.partition_D(clear_buffer)
        thr_in = thr_copy.partition_S(local_tile_in)
        thr_out = thr_copy.partition_D(local_tile_out)
        frg_in = cute.make_fragment_like(thr_in)
        frg_clear = cute.make_fragment_like(thr_clear_buffer)
        frg_acc = cute.make_fragment_like(thr_out)
        frg_acc.fill(0.0)

        # clear a next buffer to be all negtive 0
        clear_tensor = frg_clear.load()
        frg_size = cute.size(clear_tensor.shape)
        n_words = const_expr(frg_size * dtype.width // 32)
        clear_words = cute.TensorSSA(
            vector.bitcast(T.vector(n_words, T.i32()), clear_tensor), (n_words,), Int32
        )
        neg0_i32_vec = cute.full_like(clear_words, sentinel_word, Int32)
        neg0_vec = vector.bitcast(T.vector(frg_size, vec_t()), neg0_i32_vec)
        frg_clear.store(cute.TensorSSA(neg0_vec, clear_tensor.shape, dtype))
        cute.copy(copy_atom_store, frg_clear, thr_clear_buffer)

        # read local data to the register
        cute.copy(copy_atom_load, thr_in, frg_in)
        # trtllm's remove_neg_zero: real -0.0 would read as "empty"
        vx = frg_in.load()
        zeros = cute.full_like(vx, 0.0)
        frg_in.store(cute.where(cute.TensorSSA(vx == zeros, vx.shape, Boolean), zeros, vx))
        # write local data to every buffer at different ranks
        for thr_write_buffer in thr_write_buffer_list:
            cute.copy(copy_atom_store, frg_in, thr_write_buffer)

        # per-atom sentinel words for the poll (frg_in is one 128b atom per thread)
        in_words = cute.size(frg_in.shape) * dtype.width // 32
        read_tensor = frg_in.load()
        read_words = cute.TensorSSA(
            vector.bitcast(T.vector(in_words, T.i32()), read_tensor), (in_words,), Int32
        )
        neg0_read = cute.full_like(read_words, sentinel_word, Int32)

        # loop over each buffer and accumulate the data
        for i in cutlass.range_constexpr(len(buffers)):
            read_coord = (None, 0, 0, i)
            cute.copy(copy_atom_load, thr_read_buffer[read_coord], frg_in[None, 0, 0])
            isNotNeg0 = _is_not_neg0(frg_in.load(), neg0_read)
            # if the data is negtive 0, data are not ready or not visible yet: read again
            while not isNotNeg0:
                cute.copy(copy_atom_load, thr_read_buffer[read_coord], frg_in[None, 0, 0])
                isNotNeg0 = _is_not_neg0(frg_in.load(), neg0_read)
            frg_acc.store(frg_in.load() + frg_acc.load())

        cute.copy(copy_atom_store, frg_acc, thr_out)


def trtllm_arm(M, N, rank, world, oneshot, n_sets=N_SETS):
    """flashinfer kAllReduce; oneshot True/False forces the strategy, None leaves it
    to flashinfer's size heuristic (one-shot at every decode site). N_SETS input
    clones, shared workspace and output (workspaces stay fixed under rotation, as
    in bench_baselines)."""
    import flashinfer.comm as fcomm

    ws = fcomm.create_allreduce_fusion_workspace(
        "trtllm", world_size=world, rank=rank, max_token_num=M, hidden_dim=N,
        dtype=torch.bfloat16, group=dist.group.WORLD)
    base = make_base_input(M, N)
    ins = [base.clone() for _ in range(n_sets)]
    outp = torch.empty_like(base)

    def fn(inp):
        fcomm.allreduce_fusion(inp, ws, fcomm.AllReduceFusionPattern.kAllReduce,
                               output=outp, use_oneshot=oneshot)

    ref = base.clone()
    dist.all_reduce(ref)
    return Arm(fn, arg_sets=[(ins[i],) for i in range(n_sets)],
               read_out=lambda: outp.float(), keep=(ws, ins, outp)), ref, ins


# Referenced-from: all_reduce_two_shot_multimem.all_reduce_multimem_kernel — verbatim
# with the 128b ld_reduce variant picked by dtype (8xbf16) instead of hardcoded 4xf32.
@cute.kernel
def cutlass_2shot_multimem_kernel(gIn: cute.Tensor, gOut: cute.Tensor, flag: cute.Tensor,
                                  flag_mc: cute.Tensor, thr_layout: cute.Layout,
                                  val_layout: cute.Layout, local_rank: cutlass.Constexpr,
                                  world_size: cutlass.Constexpr):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    num_ctas = cute.size(gIn, mode=[1])
    chunk_size = num_ctas // world_size
    blk_idx = local_rank * chunk_size + bidx
    blk_coord = ((None, None), blk_idx)
    local_tile_out = gOut[blk_coord]
    local_tile_in = gIn[blk_coord]
    copy_atom_load = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), gIn.element_type, num_bits_per_copy=128)
    tiled_copy = cute.make_tiled_copy_tv(copy_atom_load, thr_layout, val_layout)
    thr_copy = tiled_copy.get_slice(tidx)
    thr_in = thr_copy.partition_S(local_tile_in)
    thr_out = thr_copy.partition_D(local_tile_out)
    (_, rest_m), _, _ = thr_in.shape
    ld_reduce = _multimem_ld_reduce_128b(gIn.element_type)
    for i in cutlass.range_constexpr(rest_m):
        x, y, z, w = ld_reduce(thr_in[(None, i), 0, 0].iterator)
        utils.distributed.multimem_st_4xb32(thr_out[(None, i), 0, 0].iterator, x, y, z, w)
    # Ensure all threads in cta have finish issue multimem.ld_reduce and multimem.st instructions
    cute.arch.sync_threads()
    if tidx == 0:
        # Linear id of current SM.
        sm_id_linear = (
            cute.arch.block_idx()[0]
            + cute.arch.block_idx()[1] * cute.arch.grid_dim()[0]
            + cute.arch.block_idx()[2] * cute.arch.grid_dim()[0] * cute.arch.grid_dim()[1]
        )
        # Release flag with sys scope
        utils.distributed.multimem_red_add1(
            flag_mc.iterator + sm_id_linear, scope="sys", order="release")
        # Relaxed spin-lock wait flag with sys scope
        utils.distributed.spin_lock_atom_cas_relaxed_wait(
            flag.iterator + sm_id_linear, expected_val=world_size, reset_val=0, scope="sys")


def cutlass_2shot_multimem_arm(M, N, rank, world, n_sets=N_SETS):
    """cutlass all_reduce_two_shot_multimem (bf16 via cutlass_2shot_multimem_kernel): multimem
    ld_reduce + st per rank-chunk plus its per-SM sys-scope exit barrier
    (multimem_red_add1 release + CAS-reset spin). Payload reshaped to 128 rows
    for the fixed 128-row tile."""
    # Referenced-from: all_reduce_two_shot_multimem.all_reduce_multimem —
    # verbatim jit body plus the stream arg its launch lacks (the example
    # launches on the default stream, which escapes graph capture).
    @cute.jit
    def ar2s_jit(mIn: cute.Tensor, mOut: cute.Tensor, flag: cute.Tensor,
                 flag_mc: cute.Tensor, local_rank: cutlass.Constexpr,
                 world_size: cutlass.Constexpr, stream: cuda_drv.CUstream):
        dtype = mIn.element_type
        vector_size = 128 // dtype.width
        thr_layout = cute.make_ordered_layout((4, 32), order=(1, 0))
        val_layout = cute.make_ordered_layout((32, vector_size), order=(1, 0))
        tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
        gIn = cute.zipped_divide(mIn, tiler_mn)
        gOut = cute.zipped_divide(mOut, tiler_mn)
        cutlass_2shot_multimem_kernel(
            gIn, gOut, flag, flag_mc, thr_layout, val_layout,
            local_rank, world_size,
        ).launch(grid=[cute.size(gOut, mode=[1]) // world_size, 1, 1],
                 block=[cute.size(tv_layout, mode=[0]), 1, 1],
                 stream=stream)

    elems = M * N
    tile_elems = 128 * 32 * (128 // 16)  # thr (4,32) x val (32, 8) bf16 tile
    assert elems % (tile_elems * world) == 0, "payload must tile evenly across ranks"
    shape = (128, elems // 128)
    ins, outs, ins_mc, outs_mc = [], [], [], []
    for _ in range(n_sets):
        inp = nvshmem.core.tensor(shape, dtype=torch.bfloat16)
        out = nvshmem.core.tensor(shape, dtype=torch.bfloat16)
        inp_mc = nvshmem.core.get_multicast_tensor(nvshmem.core.Teams.TEAM_NODE, inp)
        out_mc = nvshmem.core.get_multicast_tensor(nvshmem.core.Teams.TEAM_NODE, out)
        on_finalize(lambda a=inp, b=out, c=inp_mc, d=out_mc:
                    [nvshmem.core.free_tensor(x) for x in (c, d, a, b)])
        ins.append(inp)
        outs.append(out)
        ins_mc.append(inp_mc)
        outs_mc.append(out_mc)
    base = make_base_input(M, N)
    for t in ins:
        t.copy_(base.reshape(shape))
    flag = nvshmem.core.tensor((elems // tile_elems,), dtype=torch.int32)
    flag.fill_(0)
    flag_mc = nvshmem.core.get_multicast_tensor(nvshmem.core.Teams.TEAM_NODE, flag)
    on_finalize(lambda: [nvshmem.core.free_tensor(x) for x in (flag_mc, flag)])
    stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(
        ar2s_jit, from_dlpack(ins_mc[0]), from_dlpack(outs_mc[0]),
        from_dlpack(flag), from_dlpack(flag_mc), rank, world, stream)

    def fn(inp_mc, out_mc):
        s = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)
        compiled(from_dlpack(inp_mc), from_dlpack(out_mc),
                 from_dlpack(flag), from_dlpack(flag_mc), s)

    ref = ins[0].clone()
    dist.all_reduce(ref)
    return Arm(fn, arg_sets=[(ins_mc[i], outs_mc[i]) for i in range(n_sets)],
               read_out=lambda: outs[0].float(),
               keep=(ins, outs, flag, compiled)), ref, ins


def cutlass_lamport_arm(M, N, rank, world, n_sets=N_SETS):
    """cutlass all_reduce_one_shot_lamport (trtllm mirror), bf16 via the vendored
    AllReduceOneShotLamportKernelBf16. The 3-slot phase advances per call (host
    counter baked per recorded call); N_SETS is a multiple of 3 so captured
    rounds replay in phase."""
    buf, bufs = make_lamport_buffers(M, N, world, torch.bfloat16)
    base = make_base_input(M, N)
    ins = [base.clone() for _ in range(n_sets)]
    outp = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(
        AllReduceOneShotLamportKernelBf16(), rank, world, cutlass.Int32(0),
        from_dlpack(ins[0], assumed_align=16), from_dlpack(outp, assumed_align=16),
        [from_dlpack(b, assumed_align=16) for b in bufs], stream=stream)

    assert n_sets % 3 == 0
    it = {"i": 0}

    def fn(inp):
        s = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)
        compiled(cutlass.Int32(it["i"] % 3), from_dlpack(inp, assumed_align=16),
                 from_dlpack(outp, assumed_align=16),
                 [from_dlpack(b, assumed_align=16) for b in bufs], s)
        it["i"] += 1

    ref = base.clone()
    dist.all_reduce(ref)
    return Arm(fn, arg_sets=[(ins[i],) for i in range(n_sets)],
               read_out=lambda: outp.float(),
               keep=(buf, bufs, ins, outp, compiled)), ref, ins


def duckpond_2shot_arm(M, N, rank, world, n_sets=N_SETS):
    """duckpond two-shot multimem AR (kernels/ar/all_reduce_multimem.py: each rank
    switch-reduces its row slab from the multicast input and multimem-stores the
    sum to every rank). Both rank barriers run inside the kernel (per-CTA slot,
    cutlass exit_bar method). Per-set symmetric inputs, one symmetric output."""
    from kernels.ar.all_reduce_multimem import all_reduce_multimem, make_ar_flags

    assert M % world == 0, "two-shot slabs are row-aligned"
    slab = M // world
    tile_m = min(128, slab)
    assert slab % tile_m == 0 and tile_m % 4 == 0, f"slab {slab} not tileable"

    def symm(shape, dtype):
        t = nvshmem.core.tensor(shape, dtype=dtype)
        mc = nvshmem.core.get_multicast_tensor(nvshmem.core.Teams.TEAM_NODE, t)
        on_finalize(lambda a=t, b=mc: [nvshmem.core.free_tensor(x) for x in (b, a)])
        return t, mc

    base = make_base_input(M, N)
    ins, ins_mc = zip(*[symm((M, N), torch.bfloat16) for _ in range(n_sets)])
    for t in ins:
        t.copy_(base)
    out, out_mc = symm((M, N), torch.bfloat16)
    rows = slice(rank * slab, (rank + 1) * slab)
    flags, flags_mc = make_ar_flags(slab, N, tile_m)
    kernel = cute.compile(
        all_reduce_multimem,
        from_dlpack(ins_mc[0][rows], enable_tvm_ffi=True),
        from_dlpack(out_mc[rows], enable_tvm_ffi=True),
        from_dlpack(flags, enable_tvm_ffi=True),
        from_dlpack(flags_mc, enable_tvm_ffi=True),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        world_size=world,
        tile_m=tile_m,
        frags_in_flight=min(8, tile_m // 4),
        options="--enable-tvm-ffi",
    )

    def fn(in_mc_slab):
        kernel(in_mc_slab, out_mc[rows], flags, flags_mc)  # live mc slabs bind the env stream

    ref = base.clone()
    dist.all_reduce(ref)
    return Arm(fn, arg_sets=[(ins_mc[i][rows],) for i in range(n_sets)],
               read_out=lambda: out.float(),
               keep=(ins, ins_mc, out, out_mc, flags, flags_mc, kernel)), ref, list(ins)


def cutlass_tma_arm(M, N, rank, world, n_sets=N_SETS):
    """cutlass all_reduce_tma (bf16): N_SETS symm inputs with per-set peer
    views; shared output and flags."""
    import cuda.bindings.driver as cuda_drv
    sys.path.insert(0, os.path.expanduser(CUTLASS_AR_DIR))
    from all_reduce_tma import AllReduceTmaKernel

    ins, peers = [], []
    for _ in range(n_sets):
        inp = nvshmem.core.tensor((M, N), dtype=torch.bfloat16)
        ins.append(inp)
        peers.append([nvshmem.core.get_peer_tensor(inp, r) for r in range(world)])
        on_finalize(lambda a=inp: nvshmem.core.free_tensor(a))
    base = make_base_input(M, N)
    for t in ins:
        t.copy_(base)
    outp = nvshmem.core.tensor((M, N), dtype=torch.bfloat16)
    outp.fill_(0)
    out_mc = nvshmem.core.get_multicast_tensor(nvshmem.core.Teams.TEAM_NODE, outp)
    elems_per_cta = AllReduceTmaKernel._elems_per_cta
    num_tiles = (M * N + elems_per_cta - 1) // elems_per_cta
    ctas_per_rank = (num_tiles + world - 1) // world
    flag = nvshmem.core.tensor((ctas_per_rank,), dtype=torch.int32)
    flag.fill_(0)
    flag_mc = nvshmem.core.get_multicast_tensor(nvshmem.core.Teams.TEAM_NODE, flag)
    on_finalize(lambda: [nvshmem.core.free_tensor(x) for x in (out_mc, flag_mc, outp, flag)])
    stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(
        AllReduceTmaKernel(cutlass.BFloat16),
        [from_dlpack(x) for x in peers[0]],
        from_dlpack(out_mc), from_dlpack(flag), from_dlpack(flag_mc),
        rank, world, stream=stream)

    def fn(peer_list):
        s = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)
        compiled([from_dlpack(x) for x in peer_list],
                 from_dlpack(out_mc), from_dlpack(flag), from_dlpack(flag_mc), s)

    ref = ins[0].clone()
    dist.all_reduce(ref)
    return Arm(fn, arg_sets=[(peers[i],) for i in range(n_sets)],
               read_out=lambda: outp.float(),
               keep=(ins, peers, outp, flag, compiled)), ref, ins


def nccl_arm(M, N, rank, world, n_sets=N_SETS):
    base = make_base_input(M, N)
    ins = [base.clone() for _ in range(n_sets)]
    outp = torch.empty_like(base)

    def fn(inp):
        outp.copy_(inp)
        dist.all_reduce(outp)

    ref = base.clone()
    dist.all_reduce(ref)
    return Arm(fn, arg_sets=[(ins[i],) for i in range(n_sets)],
               read_out=lambda: outp.float(), keep=(ins, outp)), ref, ins


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, default=64)
    p.add_argument("--n", type=int, default=4096)
    args = p.parse_args()

    init_nvshmem()
    rank, world = dist.get_rank(), dist.get_world_size()
    M, N = args.m, args.n
    assert M % world == 0

    arms = [  # the AR halves of gemm_ar/bench_baselines' unfused chains
        ("cutlass_1shot_lamport", *cutlass_lamport_arm(M, N, rank, world)),
        ("duckpond_2shot", *duckpond_2shot_arm(M, N, rank, world)),
        ("trtllm", *trtllm_arm(M, N, rank, world, None)),  # its own 1shot/2shot heuristic
        ("nccl_graphed", *nccl_arm(M, N, rank, world)),
    ]

    for name, arm, aref, _ in arms:
        run0(arm)
        torch.cuda.synchronize()
        dist.barrier()
        err = (arm.read_out() - aref.float()).abs().max().item()
        clean_print(f"  {name}: ref check max|err| = {err:.4e}", print_once=True)
        assert err < 0.5, f"{name} ref check failed"

    clean_print("=" * 70, print_once=True)
    clean_print(f"<BF16 AR sweep | world={world} | {M}x{N} | {N_SETS} rotated sets, "
                f"bench_arm single-captured-graph mode>", print_once=True)
    for name, arm, *_ in arms:
        dist.barrier()
        (_, ms), = bench_arm(arm)
        clean_print(f"  {name:<22} {ms * 1e3:6.1f} us", print_once=True)

    # Teardown wedges (captured-NCCL graphs + nvshmem/symm_mem/trtllm ipc in
    # one process); this is a bench -- sync, then hard-exit past the finalizers.
    torch.cuda.synchronize()
    dist.barrier()
    os._exit(0)


if __name__ == "__main__":
    main()
