"""Run every GEMM+AR arm through one interface (decode: M=64, row-parallel sites).

Fused (one kernel):
  cutlass_1shot     cutlass distributed_gemm_all_reduce_lamport_blackwell verbatim (one-shot
                    lamport, peer-serial poll)
  cutlass_2shot     cutlass distributed_gemm_all_reduce_blackwell (LDMCxSTMC: multimem
                    ld_reduce + multimem st in the comm warps)
  cutlass_1shot_batch  the lamport example with OUR batched poll and nothing else changed
                    (kernels/gemm_ar/distributed_gemm_all_reduce_lamport_blackwell_batch.py:
                    BATCH_WIDTH=8 128b atoms in flight per sentinel wait, crossing peers);
                    kept on cutlass's GEMM so fused and unfused share one GEMM leg -- the
                    poll is the change to port into quack's AR; configs = swizzle {1, 8}
  quack_2shot       quack gemm epi_reduce_mode="all_reduce" (branch intra-sm, ~/quack-rs)
  quack_1shot       quack gemm epi_reduce_mode="all_reduce_lamport" (one-shot, batched 8-atom poll;
                    branch intra-sm, ~/quack-rs, since 2026-10-13)
  tk                ParallelKittens matmul_all_reduce (H100 only)
Unfused (a GEMM kernel, then the AR kernel as a second launch in the same captured graph;
AR arms from kernels/ar/bench_ar.py):
  cutlass+1shot            cutlass dense_gemm_persistent + the one-shot lamport AR verbatim
  cutlass+2shot            same GEMM + duckpond's two-shot multimem AR (ar/all_reduce_multimem.py:
                           switch ld_reduce of the rank's slab + multimem store, entry/exit
                           rank barriers in-kernel). Not the cutlass all_reduce_two_shot_multimem
                           example: it is pull-based with only an exit barrier, so nothing
                           orders a peer's GEMM before this rank's ld_reduce, and standalone at
                           M=64 it reads 72-75 us, 15x the one-shot lamport and 3.5x NCCL (its
                           fixed 128x256 tile puts the payload on 4-16 CTAs with 32 sequential
                           switch round trips per thread; results/2026-08-24/ar_sweep.md).
  cutlass+flashinfer       same GEMM + flashinfer trtllm kAllReduce (its own oneshot/twoshot
                           heuristic: one-shot at every decode site)
Reference: cuBLAS GEMM + NCCL all_reduce, timed last in every run (contention canary).

Both cutlass examples live in ~/cutlass/examples/python/CuTeDSL/cute/blackwell/kernel/
distributed (torch symmetric memory; the example's own NCCL init). Every arm runs one
per process under torchrun. Protocol: bench_rotate.bench_arm -- cloned (A, B) k-shards
rotate (L2-cold), symm workspaces and the unfused chains' intermediate D stay fixed (as
production reuses them), the whole timed window one captured graph after a warm-up of the
same graph; every config timed, the best re-timed for --rounds, the reference last.
Inputs: each rank draws A,B ~ N(0, 1/sqrt(K)) bf16 once (seed 1111+rank) and every arm
consumes those exact values. Shapes come from --model / --site (utils.MODEL_CONFIGS,
SITES): K is the global contraction dim, sharded across TP ranks (k_local = K/world);
every rank ends with the full (M, N) reduced D. Per-rank FLOPs = 2*M*N*k_local.

Usage:
    torchrun --nproc_per_node=4 bench_baselines.py cutlass_1shot --model llama3_8b --site out --m 64
    torchrun --nproc_per_node=4 bench_baselines.py quack_2shot --model llama3_8b --site out --m 64 --ref_check
"""
import argparse
import math
import os
import subprocess
import sys
from functools import partial

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # package-qualified: hosts own bare `utils`
from kernels.misc.bench_rotate import Arm, bench_arm, l2_rotate_count, run0  # noqa: E402
from kernels.misc.utils import (  # noqa: E402
    MODEL_CONFIGS,
    SITES,
    clean_print,
    torchrun_init_nvshmem,
)

CUTLASS_AR_DIR = "~/cutlass/examples/python/CuTeDSL/cute/blackwell/kernel/distributed"
CUTLASS_GEMM_DIR = "~/cutlass/examples/python/CuTeDSL/cute/blackwell/kernel/dense_gemm"


def _cutlass_tile(M):
    """The cutlass examples reject partial tiles: (mma_tiler_mn, cluster_shape_mn, use_2cta)
    by M. 64x64 1-CTA at M=64: max CTAs in one wave, no padded rows."""
    if M % 256 == 0:
        return (256, 256), (2, 1), True
    if M % 128 == 0:
        return (128, 256), (1, 1), False
    return (64, 64), (1, 1), False


def _quack(A, B, M, N, K, rank, world, mode="all_reduce"):
    """quack gemm with epi_reduce_mode=mode: "all_reduce" (two-shot: stripe reduce + multicast broadcast) or
    "all_reduce_lamport" (one-shot: every rank's tile into 3 rotating sentinel-armed mirrors on every peer, the
    receiver reduces whole tiles; quack branch intra-sm, ~/quack-rs; the slot phase is a per-call
    argument cycled over three pre-filled counters, the operand sets rounded to the slot multiple as _lamport does,
    so no add_ kernel rides in the captured graph)."""
    from quack.distributed.gemm_epi_reduce import make_epi_reduce_args
    from quack.gemm import gemm as quack_gemm

    # Large M: SM90 cooperative 128x256 (256x256 runs ~15-25x slower there, see
    # results/2026-08-24), SM100 256x256. Tiny M (decode): 64x64 on both (matches
    # cutlass: no padded rows, no idle comm slices; one MMA warp group on SM90).
    # QUACK_TILE=tm,tn,cm,cn overrides (the other harnesses' tile-sweep knob).
    sm90 = torch.cuda.get_device_capability()[0] == 9
    if os.environ.get("QUACK_TILE"):
        tile_m, tile_n, cluster_m, cluster_n = (int(x) for x in os.environ["QUACK_TILE"].split(","))
    elif M % 256 == 0:
        tile_m, tile_n, cluster_m, cluster_n = (128, 256, 2, 1) if sm90 else (256, 256, 2, 1)
    else:
        tile_m, tile_n, cluster_m, cluster_n = 64, 64, 1, 1
    d_arg, era = make_epi_reduce_args(mode, torch.bfloat16, M, N, 1, tile_m, tile_n, cluster_m, world)
    eras = [era]
    if era.slot is not None:  # lamport: one args record per slot phase
        eras = [era._replace(slot=torch.full((1,), k, dtype=torch.int32, device="cuda"))
                for k in range(N_SLOTS)]
    it = {"i": 0}

    def fn(a, b, swizzle=8):  # caller-order (l, m/n, k_local)
        e = eras[it["i"] % len(eras)]
        it["i"] += 1
        quack_gemm(a.unsqueeze(0), b.unsqueeze(0), d_arg, None, None, tile_m, tile_n,
                   cluster_m, cluster_n, max_swizzle_size=swizzle,
                   epi_reduce_mode=mode, epi_reduce_args=e)

    configs = [{"swizzle": 1}, {"swizzle": 8}]
    if era.slot is None:
        return Arm(fn, args=(A, B), configs=configs, read_out=lambda: d_arg[0].float(),
                   zero=d_arg.zero_)
    n = _lamport_sets(A, B)
    return Arm(fn, arg_sets=[(A.clone(), B.clone()) for _ in range(n)], configs=configs,
               read_out=lambda: d_arg[0].float(), zero=d_arg.zero_)


def _cutlass_2shot(A, B, M, N, K, rank, world):
    """cutlass fused GEMM+AR (LDMCxSTMC), compiled once per raster swizzle; one
    operand set per rotation slot from repeated allocate_tensors calls (the
    example owns the cute views), each seeded with A/B, comm buffers + flags
    shared from set 0; fn(args) launches the shared compiled kernel on it."""
    sys.path.insert(0, os.path.expanduser(CUTLASS_AR_DIR))
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as cutlass_utils
    from distributed_gemm_all_reduce_blackwell import (
        Sm100PersistentDenseGemmAllReduceLDMCxSTMCKernel as KernelCls,
        allocate_tensors,
    )

    ab_dtype = c_dtype = cutlass.BFloat16
    acc_dtype = cutlass.Float32
    mma_tiler_mn, cluster_shape_mn, use_2cta = _cutlass_tile(M)
    mnkl = (M, N, K // world, 1)

    gemms = {}
    for s in (1, 8):
        gemm = KernelCls(
            acc_dtype, c_dtype, use_2cta, mma_tiler_mn, cluster_shape_mn, True,
            rank_id=rank, num_ranks=world, all_reduce="LDMCxSTMC", swizzle_size=s)
        gemm.__call__.set_name_prefix(f"bench_ar_ldmcxstmc_r{rank}_w{world}_sw{s}")
        assert gemm.can_implement(mnkl, ab_dtype, c_dtype, "k", "k", "n")
        gemms[s] = gemm

    sets = []
    for _ in range(l2_rotate_count(A, B)):
        t = allocate_tensors(
            mnkl=mnkl, ab_dtype=ab_dtype, c_dtype=c_dtype, a_major="k", b_major="k",
            c_major="n", world_size=world, num_workspace=1,
            device=torch.device("cuda", torch.cuda.current_device()),
            slot_init_mode="test", global_rank=rank)
        # Standardized inputs replace allocate_tensors' random init ((M,K,L)/(N,K,L), k-major).
        t["_anchors"][0][0].copy_(A.unsqueeze(-1))
        t["_anchors"][1][0].copy_(B.unsqueeze(-1))
        sets.append(t)
    t0 = sets[0]
    shared = dict(
        c=t0["cute_tensor_c"],
        comm_in_multicast_tensor=t0["cute_tensor_comm_in_mc"],
        comm_out_multicast_tensor=t0["cute_tensor_comm_out_mc"],
        barrier_flag_unicast=t0["cute_tensor_flag_unicast"],
        barrier_flag_multicast=t0["cute_tensor_flag_multicast"])
    arg_list = [dict(shared, a=t["cute_tensor_a_list"][0], b=t["cute_tensor_b_list"][0])
                for t in sets]

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    max_active_clusters = cutlass_utils.HardwareInfo().get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1])
    compiled = {s: cute.compile(g, max_active_clusters=max_active_clusters, stream=stream,
                                **arg_list[0]) for s, g in gemms.items()}

    def launch(args, swizzle=8):
        # stream at call time, not bake: capture swaps the current stream
        compiled[swizzle](**args, stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream))

    out = t0["torch_tensor_comm_out_gpu"]  # (M, N, L); multimem_st broadcast target
    return Arm(launch, arg_sets=[(args,) for args in arg_list],
               configs=[{"swizzle": 1}, {"swizzle": 8}],
               read_out=lambda: out.permute(2, 0, 1)[0].float(), zero=out.zero_,
               keep=(sets,))


N_SLOTS = 3  # lamport ping/pong/cooling: the example's fixed 3-slot rotation


def _lamport_sets(A, B):
    """Rotation count rounded up to the slot multiple so every captured round
    (rounds x n_sets calls) leaves the slot phase invariant."""
    n = max(l2_rotate_count(A, B), 2 * N_SLOTS)
    return -(-n // N_SLOTS) * N_SLOTS


def _lamport(batch_widths, A, B, M, N, K, rank, world):
    """cutlass one-shot lamport GEMM+AR. batch_widths None: the example verbatim,
    configs = swizzle {1, 8}. Else our batched-poll copy: BATCH_WIDTH is read at
    trace time, so each width compiles right after it is set (distinct name
    prefixes keep the compile-cache keys apart); configs = batch x swizzle.
    n_sets operand sets from repeated allocate_tensors calls, each seeded with
    A/B, the slot set shared from set 0; launch advances the slot phase per call."""
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as cutlass_utils
    if batch_widths is None:
        sys.path.insert(0, os.path.expanduser(CUTLASS_AR_DIR))
        import distributed_gemm_all_reduce_lamport_blackwell as mod
        variants = [(None, s) for s in (1, 8)]
    else:
        import distributed_gemm_all_reduce_lamport_blackwell_batch as mod
        variants = [(w, s) for w in batch_widths for s in (1, 8)]

    ab_dtype = c_dtype = cutlass.BFloat16
    acc_dtype = cutlass.Float32
    mma_tiler_mn, cluster_shape_mn, use_2cta = _cutlass_tile(M)
    mnkl = (M, N, K // world, 1)

    sets = []
    for _ in range(_lamport_sets(A, B)):
        t = mod.allocate_tensors(
            mnkl=mnkl, ab_dtype=ab_dtype, c_dtype=c_dtype, a_major="k", b_major="k",
            c_major="n", num_workspace=1, world_size=world, global_rank=rank,
            device=torch.device("cuda", torch.cuda.current_device()),
            slot_init_mode="benchmark")
        t["_anchors"][0][0].copy_(A.unsqueeze(-1))
        t["_anchors"][1][0].copy_(B.unsqueeze(-1))
        sets.append(t)
    t0 = sets[0]
    assert len(t0["cute_tensors_c_uc_per_peer_grouped"]) == N_SLOTS

    def slot_kwargs(i):
        ping, pong = i % N_SLOTS, (i + 1) % N_SLOTS
        return dict(
            c_multicast_tensor=t0["cute_tensors_c_mc_per_peer_grouped"][ping][rank],
            comm_in_unicast_tensor_per_peer=t0["cute_tensors_c_uc_per_peer_grouped"][ping],
            comm_clear_unicast_tensor_per_peer=t0["cute_tensors_c_uc_per_peer_grouped"][pong],
            comm_out_unicast_tensor=t0["cute_tensor_comm_out_uc"])

    slots = [slot_kwargs(i) for i in range(N_SLOTS)]
    arg_list = [dict(a=t["cute_tensor_a_list"][0], b=t["cute_tensor_b_list"][0]) for t in sets]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    max_active_clusters = cutlass_utils.HardwareInfo().get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1])
    compiled = {}
    for width, s in variants:
        if width is not None:
            mod.BATCH_WIDTH = width
        g = mod.Sm100PersistentDenseGemmAllReduceLamportKernel(
            acc_dtype, c_dtype, use_2cta, mma_tiler_mn, cluster_shape_mn, True,
            rank_id=rank, num_ranks=world, all_reduce="Lamport", swizzle_size=s)
        tag = "" if width is None else f"_b{width}"
        g.__call__.set_name_prefix(f"bench_ar_lamport{tag}_r{rank}_w{world}_sw{s}")
        compiled[(width, s)] = cute.compile(g, **arg_list[0], **slots[0], stream=stream,
                                            max_active_clusters=max_active_clusters)
    it = {"i": 0}

    def launch(args, batch=variants[0][0], swizzle=8):  # config-less calls (checks) take the first
        # slot phase advances per call; stream at call time (capture swaps it)
        compiled[(batch, swizzle)](**args, **slots[it["i"] % N_SLOTS],
                                   stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream))
        it["i"] += 1

    out = t0["torch_tensor_comm_out_gpu"]  # (M, N, L) final AR output
    configs = ([{"swizzle": s} for s in (1, 8)] if batch_widths is None else
               [{"batch": w, "swizzle": s} for w in batch_widths for s in (1, 8)])
    return Arm(launch, arg_sets=[(args,) for args in arg_list], configs=configs,
               read_out=lambda: out.permute(2, 0, 1)[0].float(), zero=out.zero_,
               keep=(sets,))


def _tk(A, B, M, N, K, rank, world):
    """ParallelKittens (ThunderKittens) matmul_all_reduce, H100 only
    (~/ThunderKittens/kernels/parallel/gemm_ar; NCCL init, needs GPU env).
    Comp/comm SMs are split in-kernel; the configs sweep the comm carve-out.
    TK-shaped (A, B_tk=(k_local,N)) k-shards are the L2-turnover payload;
    the multicast C_tk / barrier stay fixed in fn's closure."""
    # TK bakes NUM_DEVICES at compile time, so _C is per-world-size. Cross-built
    # _C_nd{2,4,8} live alongside the default _C (nd8); load the one matching world.
    import glob
    import importlib.util

    d = os.path.expanduser("~/ThunderKittens/kernels/parallel/gemm_ar")
    cand = glob.glob(os.path.join(d, f"_C_nd{world}.*.so")) or glob.glob(os.path.join(d, "_C.*.so"))
    assert cand, f"no TK _C for world={world}; build _C_nd{world} (NUM_DEVICES={world})"
    spec = importlib.util.spec_from_file_location("_C", cand[0])
    _tkC = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_tkC)
    TKParallelTensor, matmul_all_reduce = _tkC.TKParallelTensor, _tkC.matmul_all_reduce

    assert torch.cuda.get_device_capability()[0] == 9, "TK gemm_ar is H100-only"
    # TK H100 wants B as (k_local, N); the standardized B is (N, k_local).
    B_tk = B.T.contiguous()
    C_tk = TKParallelTensor((M, N), dtype=torch.bfloat16, local_rank=rank,
                            local_world_size=world, multicast=True)
    C_tk.data_.zero_()
    # 2 MB is the multicast-object minimum anyway (matches TK's own benchmark).
    barrier = TKParallelTensor((2, 1024, 1024), dtype=torch.int, local_rank=rank,
                               local_world_size=world, multicast=True)
    barrier.data_.zero_()

    def fn(a, b, num_comm_sms=16):  # a=(M,k_local), b=(k_local,N)
        matmul_all_reduce(a, b, C_tk, barrier, num_comm_sms)

    return Arm(fn, args=(A, B_tk),
               configs=[{"num_comm_sms": v} for v in (8, 16, 24, 32, 48)],
               read_out=lambda: C_tk.data_.float(), zero=C_tk.data_.zero_)


def _cutlass_gemm_leg(A_sets, B_sets, D, M, N, K, world):
    """cutlass dense_gemm_persistent K-shard GEMM (the fused cutlass arms' tile)
    writing D. create_cute_tensor_for_fp8 aliases each clone's storage, so the
    sets stay distinct (L2-cold; a recycled temp buffer read L2-hot, 2026-09-08).
    Returns (launch(a, b), per-set (a, b) cute views, keep)."""
    sys.path.insert(0, os.path.expanduser(CUTLASS_GEMM_DIR))
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.utils as cutlass_utils
    from cutlass.utils import create_cute_tensor_for_fp8
    from dense_gemm_persistent import compile_bmm

    tiler, cluster, use2 = _cutlass_tile(M)
    Kl = K // world
    ct = lambda t, ld: create_cute_tensor_for_fp8(  # noqa: E731
        t, cutlass.BFloat16, ld, source_f32_tensor=t.float())
    a_c = [ct(a.reshape(1, M, Kl), 2) for a in A_sets]
    # b_major="k": B as (l, K, N) with K contiguous = the (N, Kl) clone viewed transposed
    b_c = [ct(b.reshape(1, N, Kl).transpose(1, 2), 1) for b in B_sets]
    d_c = ct(D.reshape(1, M, N), 2)
    max_ac = cutlass_utils.HardwareInfo().get_max_active_clusters(cluster[0] * cluster[1])
    cgemm = compile_bmm((M, N, Kl, 1), a_c[0], b_c[0], d_c, cutlass.Float32,
                        "k", "k", "n", tiler, cluster, max_ac, use2, use_tma_store=True)

    def launch(a, b):
        cgemm(a, b, d_c, cuda.CUstream(torch.cuda.current_stream().cuda_stream))

    return launch, list(zip(a_c, b_c)), (a_c, b_c, d_c, cgemm)


def _chain(gemm_leg, ar_builder, A, B, M, N, K, rank, world):
    """Unfused chain: the GEMM leg writes the AR arm's set-0 input (one shared D,
    as production reuses it), then the AR runs on it. Only the GEMM operands
    rotate; n_sets is the fused arms' count rounded to the lamport slot multiple.
    AR arms are kernels/ar/bench_ar's: (Arm, ref, inputs)."""
    n = _lamport_sets(A, B)
    arm, _ref, ins = ar_builder(M, N, rank, world, n_sets=n)
    A_sets, B_sets = [A.clone() for _ in range(n)], [B.clone() for _ in range(n)]
    launch, sets, keep = gemm_leg(A_sets, B_sets, ins[0].view(M, N), M, N, K, world)

    def fn(a, b, ar_args):
        launch(a, b)
        arm.fn(*ar_args)

    return Arm(fn, arg_sets=[(*sets[i], arm.arg_sets[0]) for i in range(n)],
               read_out=arm.read_out, keep=(arm, A_sets, B_sets, keep))


def _cublas_gemm_leg(A_sets, B_sets, D, M, N, K, world):
    """cuBLAS K-shard GEMM writing D (torch.matmul, its own kernel choice)."""
    return (lambda a, b: torch.matmul(a, b.T, out=D)), list(zip(A_sets, B_sets)), ()


def _unfused(ar, A, B, M, N, K, rank, world, gemm="cutlass"):
    """cutlass dense (or cuBLAS) GEMM leg + one of kernels/ar/bench_ar's AR arms."""
    from kernels.ar import bench_ar as ars

    ars_ = {"1shot": ars.cutlass_lamport_arm,
            "2shot": ars.duckpond_2shot_arm,
            "flashinfer": partial(ars.trtllm_arm, oneshot=None)}
    legs = {"cutlass": _cutlass_gemm_leg, "cublas": _cublas_gemm_leg}
    return _chain(legs[gemm], ars_[ar], A, B, M, N, K, rank, world)


def _nccl(A, B, M, N, K, rank, world):
    """cuBLAS GEMM + NCCL all-reduce reference."""
    D_full = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")

    def fn(a, b):
        torch.matmul(a, b.T, out=D_full)
        dist.all_reduce(D_full)

    return Arm(fn, args=(A, B), read_out=lambda: D_full.float(), zero=D_full.zero_)


# init: quack = plain NCCL (symm_mem inside quack); tk = ParallelKittens' own; symm_mem = the
# cutlass example's; nvshmem = torchrun_init_nvshmem (the AR arms' symmetric buffers).
BASELINES = {
    "cutlass_1shot":          {"factory": partial(_lamport, None),   "init": "symm_mem"},
    "cutlass_2shot":          {"factory": _cutlass_2shot,            "init": "symm_mem"},
    "cutlass_1shot_batch":    {"factory": partial(_lamport, (8,)),  "init": "symm_mem"},
    "quack_2shot":            {"factory": _quack,                    "init": "quack"},
    "quack_1shot":            {"factory": partial(_quack, mode="all_reduce_lamport"), "init": "quack"},
    "tk":                     {"factory": _tk,                       "init": "tk"},
    "triton_dist":            {"h100_lib": True},
    "cutlass+1shot":      {"factory": partial(_unfused, "1shot"),      "init": "nvshmem"},
    "cutlass+2shot":      {"factory": partial(_unfused, "2shot"),      "init": "nvshmem"},
    "cutlass+flashinfer": {"factory": partial(_unfused, "flashinfer"), "init": "nvshmem"},
    "cublas+flashinfer":  {"factory": partial(_unfused, "flashinfer", gemm="cublas"),
                           "init": "nvshmem"},
}


def tflops(ms, M, N, K, world):
    return 2 * M * N * (K // world) / (ms * 1e9)


def _run_synced(kern):
    """Zero, barrier, run set 0, sync, barrier -- one fresh cross-rank-consistent output."""
    if kern.zero:
        kern.zero()
    torch.cuda.synchronize()
    dist.barrier()
    run0(kern)
    torch.cuda.synchronize()
    dist.barrier()


def ref_check(kern, A, B, world):
    """Max rel err of kern's full (M, N) output vs an fp32 full-K reference."""
    _run_synced(kern)
    got = kern.read_out()
    A_shards = [torch.empty_like(A) for _ in range(world)]
    B_shards = [torch.empty_like(B) for _ in range(world)]
    dist.all_gather(A_shards, A)
    dist.all_gather(B_shards, B)
    A_full = torch.cat(A_shards, dim=1).float()  # (M, K)
    B_full = torch.cat(B_shards, dim=1).float()  # (N, K)
    ref = A_full @ B_full.T                      # (M, N)
    denom = ref.abs()
    mask = denom > 1e-3
    return ((got - ref).abs()[mask] / denom[mask]).max().item() if mask.any() else 0.0


def r2r_check(kern):
    """Run twice; return (bit-identical, max |delta|) of the full output."""
    _run_synced(kern)
    o1 = kern.read_out().clone()
    _run_synced(kern)
    o2 = kern.read_out().clone()
    return torch.equal(o1, o2), (o1 - o2).abs().max().item()


H100_LIB_ENVS = {"triton_dist": "/scratch/gpfs/TRIDAO/zy4586/envs/triton-dist-main"}  # upstream main 7908e4e, 2026-09-18


def run_h100_lib(lib, M, N, K):
    """Triton-distributed only builds for H100 (sm90) and lives in its own conda env, so
    shell out to h100_lib_runner.py under that env's torchrun (plain-python invocation,
    like the ag_gemm / gemm_rs benches; world = visible GPU count)."""
    cap = torch.cuda.get_device_capability()
    if cap != (9, 0):
        sys.exit(f"{lib} baseline is H100 (sm90)-only; this GPU is sm{cap[0]}{cap[1]}")
    torchrun = os.path.join(H100_LIB_ENVS[lib], "bin", "torchrun")
    runner = os.path.join(os.path.dirname(os.path.abspath(__file__)), "h100_lib_runner.py")
    env = os.environ.copy()
    # nvshmem's host lib is not on the loader path; env/lib = the libstdc++ (>= GLIBCXX_3.4.30) libtriton links
    env["LD_LIBRARY_PATH"] = ":".join([
        os.path.join(H100_LIB_ENVS[lib], "lib/python3.12/site-packages/nvidia/nvshmem/lib"),
        os.path.join(H100_LIB_ENVS[lib], "lib"), env.get("LD_LIBRARY_PATH", "")])
    # "--": torchrun abbrev-matches --m/--n/--k to its own flags without it
    subprocess.run([torchrun, f"--nproc_per_node={torch.cuda.device_count()}", "--", runner,
                    lib, f"--m={M}", f"--n={N}", f"--k={K}"], check=True, env=env)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("baseline", choices=list(BASELINES))
    p.add_argument("--model", choices=list(MODEL_CONFIGS), required=True, help="shape source (MODEL_CONFIGS)")
    p.add_argument("--site", choices=("out", "down"), required=True, help="row-parallel GEMM (SITES)")
    p.add_argument("--m", type=int, required=True, help="rows (64 = decode)")
    p.add_argument("--ref_check", action="store_true",
                   help="Max rel err of the kernel & reference vs an fp32 full-K GEMM.")
    p.add_argument("--r2r_check", action="store_true",
                   help="Run kernel & reference twice; report bit-identity and max |delta|.")
    p.add_argument("--eager", action="store_true",
                   help="Time kernel & reference eager (capture=False), as the triton_dist runner does.")
    p.add_argument("--rounds", type=int, default=1,
                   help="re-time the winning config N-1 more times back-to-back; "
                        "the reference is timed once, last, as the contention canary")
    args = p.parse_args()
    M, (N, K) = args.m, SITES[args.site](MODEL_CONFIGS[args.model])

    spec = BASELINES[args.baseline]
    if spec.get("h100_lib"):
        run_h100_lib(args.baseline, M, N, K)
        return
    if spec["init"] == "quack":
        # plain NCCL init; symmetric memory is torch symm_mem inside quack
        # (fallback: older quack branches predate these helpers)
        try:
            from quack.dist_utils import init_distributed, clean_distributed
        except ImportError:
            from kernels.misc.utils import torchrun_init as init_distributed
            from kernels.misc.utils import torchrun_finalize as clean_distributed
        init_distributed()
        finalize = clean_distributed
    elif spec["init"] == "tk":
        # ParallelKittens needs its own init: TKParallelTensor's PGL is built here,
        # and plain NCCL init leaves it mismatched ("Number of devices mismatch").
        sys.path.insert(0, os.path.expanduser("~/ThunderKittens/kernels/parallel"))
        from common import init_distributed_environment, destroy_distributed_environment
        init_distributed_environment()
        finalize = destroy_distributed_environment
    elif spec["init"] == "nvshmem":
        torchrun_init_nvshmem()
        # Teardown wedges (captured-NCCL graphs + nvshmem/trtllm ipc in one process):
        # sync, then hard-exit past the finalizers, as kernels/ar/bench_ar.
        finalize = partial(os._exit, 0)
    else:
        sys.path.insert(0, os.path.expanduser(CUTLASS_AR_DIR))
        from distributed_gemm_all_reduce_ldxstmc_blackwell import (
            init_distributed, clean_distributed)
        init_distributed()
        finalize = clean_distributed
    rank, world = dist.get_rank(), dist.get_world_size()
    assert M % world == 0 and K % world == 0
    assert N % 8 == 0, "n must be divisible by 8 (16B multimem vectors)"

    # Standardized inputs: every implementation consumes these exact values.
    torch.manual_seed(1111 + rank)
    inv_sqrt_k = 1.0 / math.sqrt(K)
    A = (torch.empty(M, K // world, dtype=torch.float32, device="cuda")
         .normal_().mul_(inv_sqrt_k).to(torch.bfloat16))
    B = (torch.empty(N, K // world, dtype=torch.float32, device="cuda")
         .normal_().mul_(inv_sqrt_k).to(torch.bfloat16))

    # One protocol: the quack-autotuner L2-cold rotate methodology
    # (bench_rotate.bench_arm) — the whole timed window is one captured
    # graph, replayed once. Every factory builds its rotation sets up front;
    # checks run eagerly on set 0 of the same build. An arm that truly
    # CANNOT capture (host-blocking op, iteration is itself a captured
    # graph) sets "capture": False in BASELINES with a comment saying why,
    # and reports as eager-timed (per-call replay boundary included — not
    # comparable to captured cells at decode sizes).
    kern = spec["factory"](A, B, M, N, K, rank, world)
    ref = _nccl(A, B, M, N, K, rank, world)

    checks = []
    if args.ref_check:
        checks.append(("rel err", f"{ref_check(kern, A, B, world):.4e} (|ref|>1e-3)",
                       f"{ref_check(ref, A, B, world):.4e}"))
    if args.r2r_check:
        kb, kd = r2r_check(kern)
        rb, rd = r2r_check(ref)
        checks.append(("r2r", f"bit-identical={kb} max|Δ|={kd:.4e}",
                       f"bit-identical={rb} max|Δ|={rd:.4e}"))

    capture = spec.get("capture", True) and not args.eager
    proto = "L2-rotate" if capture else "L2-rotate (eager-timed)"
    sweep_ms = bench_arm(kern, capture=capture)
    best_label, ms = min(sweep_ms, key=lambda r: r[1])
    # Rounds re-time the winning config back-to-back; the reference (cuBLAS+NCCL)
    # times ONCE at the end under the same regime, as ag_gemm / gemm_rs.
    cfgs = kern.configs or [{}]
    best_cfg = cfgs[[" ".join(f"{k}={v}" for k, v in c.items()) for c in cfgs].index(best_label)]
    rounds = [bench_arm(kern._replace(configs=[best_cfg]), capture=capture)[0][1]
              for _ in range(args.rounds - 1)]
    ref_ms = bench_arm(ref, capture=capture)[0][1]

    clean_print("=" * 80, print_once=True)
    clean_print(f"<BF16 GEMM+AR {args.baseline} | world={world} | {M}x{K}x{N} | {proto}>",
                print_once=True)
    kern_name = f"{args.baseline} best ({best_label})" if len(sweep_ms) > 1 else args.baseline
    rows = [(f"{args.baseline} {label}:", v) for label, v in sweep_ms] if len(sweep_ms) > 1 else []
    rows += [(kern_name + ":", ms), ("cuBLAS+NCCL:", ref_ms)]
    rows += [(f"{args.baseline} round{i + 1}:", v) for i, v in enumerate(rounds)]
    for label, v in rows:
        clean_print(f"  {label:<14} {v:.4f} ms | {tflops(v, M, N, K, world):.2f} TFLOp/s",
                    print_once=True)
    for label, kv, rv in checks:
        clean_print(f"  {args.baseline + ' ' + label + ':':<22} {kv}", print_once=True)
        clean_print(f"  {'cuBLAS+NCCL ' + label + ':':<22} {rv}", print_once=True)

    if kern.free:
        kern.free()
    if ref.free:
        ref.free()
    torch.cuda.synchronize()
    dist.barrier()
    finalize()


if __name__ == "__main__":
    main()
