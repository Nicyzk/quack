"""GEMM+AR + residual + RMSNorm: the llama TP decode layer site, four pipelines.

Per layer there are two of these (o_proj and down_proj): the rank's K-shard GEMM,
the TP all-reduce, a residual add, and RMSNorm feeding the next op. Each arm runs
its own convention's complete kernel chain to reach "ready for the next op":

  quack        fused GEMM+AR with the sq-reduce epilogue (CODA format): one kernel
               emits D = (AR + residual)*gamma plus per-N-tile squared-sum
               partials, broadcast full-shape through the colvec mc view (like
               D), so rms_final_reduce finalizes full-M rstd locally — no
               exchange. The 1/rms scale rides the next op (per-row scalar);
               the raw residual stream is not materialized (fewer bytes written
               than the other arms).
  cutlass      cutlass LDMCxSTMC fused GEMM+AR, then flashinfer's standalone
               fused_add_rmsnorm (residual add + full norm; extra read+write of D).
  cublas_nccl  cuda-graphed cuBLAS GEMM + NCCL AR + torch.compile'd add+rmsnorm.
  flashinfer   cuda-graphed cuBLAS GEMM + flashinfer allreduce_fusion
               (trtllm kernels: AR + residual + RMSNorm in one kernel, oneshot
               heuristics; norm_out and the residual stream both written).
  quack_ar_norm_1shot / _2shot
               cuBLAS GEMM + quack's fused AR + residual + RMSNorm kernel
               (kernels/ar/ar_norm_lamport.py one-shot lamport; ar_norm_two_shot.py
               row-slab multimem two-shot). The two-shot pulls the peers' GEMM
               output with only an exit barrier, so after a GEMM it is unsynced
               (timed as-is; its ref check runs the chain twice).

Inputs standardized as in bench_baselines.py (A,B ~ N(0, 1/sqrt(K)) bf16, seed
1111+rank); residual and gamma are rank-invariant (seeds 3333/2222). Decode
shapes: --m 64, o_proj N=K=hidden, down_proj N=hidden K=intermediate.
Timing: the quack-autotuner L2-cold rotate methodology (bench_rotate.bench_arm:
each arm's chain captured over cloned (A, B) k-shards, one graph replayed once;
sw1/sw8 sweep, best reported); cutlass_lamport's arg_sets are a multiple of the
3-slot rotation so captured rounds replay in phase.

Usage:
    torchrun --nproc_per_node=4 bench_llama_ar.py quack --m 64 --n 4096 --k 4096
    torchrun --nproc_per_node=4 bench_llama_ar.py flashinfer --m 64 --n 8192 --k 28672 --ref_check
"""
import argparse
import math
import os
import sys
from functools import partial

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "misc"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # the kernels package root
from bench_rotate import Arm, bench_arm, l2_rotate_count, run0  # noqa: E402
from utils import clean_print  # noqa: E402

from bench_baselines import CUTLASS_AR_DIR, _lamport  # noqa: E402  (the cutlass builders: imported by their arms)
from bench_baselines import _quack as _quack_plain  # noqa: E402

EPS = 1e-6


def _quack(A, B, res_full, gamma, M, N, K, rank, world):
    from quack.dist_utils import make_symm_mem_tensor
    from quack.distributed.gemm_epi_reduce import make_epi_reduce_args
    from quack.epilogue.library import sq_reduce_mod
    from quack.rms_final_reduce import rms_final_reduce

    if M % 256 == 0:
        tile_m, tile_n, cluster_m, cluster_n = 256, 256, 2, 1
    else:
        # 64x64 splits the B stream best, but the sq epilogue's per-tile tail
        # is exposed per wave: widen tile_n so the grid stays one wave.
        tile_m, cluster_m, cluster_n = 64, 1, 1
        tile_n = 64 if (N + 63) // 64 <= 148 else 128
    n_tiles = (N + tile_n - 1) // tile_n
    c_gpu = res_full.unsqueeze(0).contiguous()  # AR C: the full replicated residual
    w_gpu = gamma.float().reshape(1, N).contiguous()
    # AR broadcasts the sq partials full-shape through the mc view (like D):
    # every rank finalizes full-M rstd locally, no exchange.
    colvec, colvec_mc = make_symm_mem_tensor((1, M, n_tiles), torch.float32, (0, 1, 2))
    colvec.zero_()
    rstd_full = torch.empty(M, dtype=torch.float32, device="cuda")
    d_arg, era = make_epi_reduce_args(
        "all_reduce", torch.bfloat16, M, N, 1, tile_m, tile_n, cluster_m, world)
    mod = sq_reduce_mod(has_c=True, has_rowvec=True, has_aux=False)
    epi_args = {"mRowVecBroadcast": w_gpu, "mColVecReduce": colvec_mc}

    def fn(a, b, swizzle=8):
        mod.gemm(a.unsqueeze(0), b.unsqueeze(0), d_arg, c_gpu, epi_args=epi_args,
                 tile_M=tile_m, tile_N=tile_n, cluster_M=cluster_m,
                 cluster_N=cluster_n, is_dynamic_persistent=False,
                 max_swizzle_size=swizzle, epi_reduce_mode="all_reduce",
                 epi_reduce_args=era)
        rstd_full.copy_(rms_final_reduce(colvec[0], scale=1.0 / N, eps=EPS))

    # D = (AR + res) * gamma; the deferred 1/rms is the per-row scalar.
    read = lambda: d_arg[0].float() * rstd_full[:, None]  # noqa: E731
    return Arm(fn, args=(A, B), configs=[{"swizzle": 1}, {"swizzle": 8}],
               read_out=read)


def _cutlass_norm(A, B, res_full, gamma, M, N, K, rank, world):
    """cutlass LDMCxSTMC AR (bench_baselines._cutlass_build) + flashinfer's
    fused_add_rmsnorm: l2_rotate_count operand sets and the sw1/sw8 kernels,
    exposed as arg_sets for fn(args, swizzle)."""
    from flashinfer.norm import fused_add_rmsnorm

    from bench_baselines import _cutlass_build  # retired with the 2026-10-13 bench_baselines rework

    launch, arg_list, t = _cutlass_build("LDMCxSTMC", A, B, M, N, K, rank, world,
                                         n_sets=l2_rotate_count(A, B), swizzles=(1, 8))
    # comm_out is the (M, N, L) bf16 broadcast target, re-written fully by
    # every run, so in-place norm is safe.
    out2d = t["torch_tensor_comm_out_gpu"].permute(2, 0, 1)[0]
    gamma_bf = gamma.to(torch.bfloat16).contiguous()
    res = res_full.clone()

    def fn(args, swizzle=8):
        launch(args, swizzle)
        # In place: out2d -> normed, res += out2d. res drifts across timed reps
        # (timing-equivalent bytes); the ref-check run is the fresh first run.
        fused_add_rmsnorm(out2d, res, gamma_bf, EPS)

    return Arm(fn, arg_sets=[(args,) for args in arg_list],
               configs=[{"swizzle": 1}, {"swizzle": 8}],
               read_out=lambda: out2d.float(), keep=(t,))


def _cutlass_sentinel(A, B, res_full, gamma, M, N, K, rank, world):
    """Sentinel-exit LDMCxSTMC (no exit barrier; D self-certifies per-16B) +
    the polling add+RMSNorm consumer, which re-arms D after reading. The
    norm's polls double as the workspace-reuse gate: broadcasts landing
    certify every rank's ld_reduce of the C workspace completed."""
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as cutlass_utils
    import cutlass_gemm_ar_sentinel as armod

    n_sets, swizzles = l2_rotate_count(A, B), (1, 8)
    ab_dtype = c_dtype = cutlass.BFloat16
    acc_dtype = cutlass.Float32
    if M % 256 == 0:
        mma_tiler_mn, cluster_shape_mn, use_2cta = (256, 256), (2, 1), True
    elif M % 128 == 0:
        mma_tiler_mn, cluster_shape_mn, use_2cta = (128, 256), (1, 1), False
    else:
        mma_tiler_mn, cluster_shape_mn, use_2cta = (64, 64), (1, 1), False
    mnkl = (M, N, K // world, 1)
    # Compiled once per swizzle; n_sets operand sets from repeated
    # allocate_tensors calls, comm buffers + flags shared from set 0 (as
    # bench_baselines._cutlass_build).
    gemms = {}
    for s in swizzles:
        gemm = armod.Sm100PersistentDenseGemmAllReduceLDMCxSTMCKernel(
            acc_dtype, c_dtype, use_2cta, mma_tiler_mn, cluster_shape_mn, True,
            rank_id=rank, num_ranks=world, all_reduce="LDMCxSTMC", sentinel_exit=True,
            swizzle_size=s)
        gemm.__call__.set_name_prefix(f"sentinel_ar_r{rank}_w{world}_sw{s}")
        assert gemm.can_implement(mnkl, ab_dtype, c_dtype, "k", "k", "n")
        gemms[s] = gemm
    sets = []
    for _ in range(n_sets):
        t = armod.allocate_tensors(
            mnkl=mnkl, ab_dtype=ab_dtype, c_dtype=c_dtype, a_major="k", b_major="k",
            c_major="n", world_size=world, num_workspace=1,
            device=torch.device("cuda", torch.cuda.current_device()),
            slot_init_mode="test", global_rank=rank)
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

    comm_out = t0["torch_tensor_comm_out_gpu"]  # (M, N, L) bf16 broadcast target
    out2d = comm_out.permute(2, 0, 1)[0]
    assert out2d.is_contiguous()
    comm_out.view(torch.int16).fill_(-32768)  # arm D with the bf16 -0.0 sentinel
    gamma_bf = gamma.to(torch.bfloat16).contiguous()
    out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    res_out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    # quack rmsnorm with sentinel_poll (PDL, polls gate on D, re-arms D);
    # compiled fn called directly -- the torch-op/FFI host path costs extra
    # us eagerly (llama_ar accounting note)
    import cutlass as _cutlass_mod
    from quack.rmsnorm import _compile_rmsnorm_fwd
    bf16 = _cutlass_mod.BFloat16
    norm_fn = _compile_rmsnorm_fwd(
        bf16, bf16, bf16, bf16, None, bf16, N,
        False, False, False, False, 0.0, True)
    # pin: the compiled calls bake device pointers
    _keep = (sets, out2d, gamma_bf, out, res_out)

    def fn(args, swizzle=8):
        # stream at call time, not bake: capture swaps the current stream
        compiled[swizzle](**args, stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream))
        norm_fn(out2d, gamma_bf, None, res_full, out, res_out, None, None, EPS)

    return Arm(fn, arg_sets=[(args,) for args in arg_list],
               configs=[{"swizzle": 1}, {"swizzle": 8}],
               read_out=lambda: out.float(), keep=_keep)


def _cutlass_lamport_norm(A, B, res_full, gamma, M, N, K, rank, world):
    """cutlass one-shot lamport GEMM+AR (bench_baselines._cutlass_lamport_build:
    slot phase advances per call, stream resolved at call time, so the rotate
    driver captures the whole timed window) + flashinfer fused_add_rmsnorm.
    n_sets rounded up to a multiple of the 3-slot rotation so every captured
    round replays in phase."""
    from flashinfer.norm import fused_add_rmsnorm

    from bench_baselines import _cutlass_lamport_build  # retired with the 2026-10-13 bench_baselines rework

    N_SLOTS = 3  # lamport ping/pong/cooling — the example's fixed 3-slot rotation
    n = max(l2_rotate_count(A, B), 2 * N_SLOTS)
    n = -(-n // N_SLOTS) * N_SLOTS  # round up to a multiple of the slot count
    launch, arg_list, t, n_slots = _cutlass_lamport_build(
        A, B, M, N, K, rank, world, n_sets=n, swizzles=(1, 8))
    assert n_slots == N_SLOTS, f"lamport n_slots {n_slots} != assumed {N_SLOTS}"
    out2d = t["torch_tensor_comm_out_gpu"].permute(2, 0, 1)[0]
    gamma_bf = gamma.to(torch.bfloat16).contiguous()
    res = res_full.clone()

    def fn(args, swizzle=8):
        launch(args, swizzle)
        # In place: out2d -> normed, res += out2d. res drifts across timed reps
        # (timing-equivalent bytes); the ref-check run is the fresh first run.
        fused_add_rmsnorm(out2d, res, gamma_bf, EPS)

    return Arm(fn, arg_sets=[(args,) for args in arg_list],
               configs=[{"swizzle": 1}, {"swizzle": 8}],
               read_out=lambda: out2d.float(), keep=(t,))


def _batch_lamport_norm(A, B, res_full, gamma, M, N, K, rank, world):
    """bench_baselines' batched-poll lamport GEMM+AR (cutlass_1shot_batch, width 8)
    + flashinfer fused_add_rmsnorm; configs = batch x swizzle as there."""
    from flashinfer.norm import fused_add_rmsnorm

    arm = _lamport((8,), A, B, M, N, K, rank, world)
    out2d = arm.keep[0][0]["torch_tensor_comm_out_gpu"].permute(2, 0, 1)[0]
    gamma_bf = gamma.to(torch.bfloat16).contiguous()
    res = res_full.clone()

    def fn(args, batch=8, swizzle=8):
        arm.fn(args, batch, swizzle)
        fused_add_rmsnorm(out2d, res, gamma_bf, EPS)

    return Arm(fn, arg_sets=arm.arg_sets, configs=arm.configs,
               read_out=lambda: out2d.float(), keep=(arm,))


def _cublas_nccl(A, B, res_full, gamma, M, N, K, rank, world):
    gamma_bf = gamma.to(torch.bfloat16)
    D = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    normed = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")

    def add_norm(d, r, w):
        x = (d + r).float()
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)).to(
            torch.bfloat16) * w

    add_norm_c = torch.compile(add_norm)

    def fn(a, b):
        torch.matmul(a, b.T, out=D)
        dist.all_reduce(D)
        normed.copy_(add_norm_c(D, res_full, gamma_bf))

    return Arm(fn, args=(A, B), read_out=lambda: normed.float())


def _flashinfer(A, B, res_full, gamma, M, N, K, rank, world):
    import flashinfer.comm as fcomm

    gamma_bf = gamma.to(torch.bfloat16).contiguous()
    ws = fcomm.create_allreduce_fusion_workspace(
        "trtllm", world_size=world, rank=rank, max_token_num=M, hidden_dim=N,
        dtype=torch.bfloat16, group=dist.group.WORLD)
    D = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    norm_out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    res_out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")

    def fn(a, b, _keep=ws):
        torch.matmul(a, b.T, out=D)
        fcomm.allreduce_fusion(
            D, ws, fcomm.AllReduceFusionPattern.kARResidualRMSNorm,
            residual_in=res_full, rms_gamma=gamma_bf,
            norm_out=norm_out, residual_out=res_out, rms_eps=EPS)

    return Arm(fn, args=(A, B), read_out=lambda: norm_out.float())


def _quack_ar_norm(A, B, res_full, gamma, M, N, K, rank, world, two_shot=False, gemm="cublas"):
    """cuBLAS GEMM into the AR input, then quack's fused AR+residual+RMSNorm
    kernel: one-shot lamport (slot phase advances per call, stream at call
    time) or the row-slab multimem two-shot (symmetric D/out/res_out, this
    rank's slab views). arg_sets: l2_rotate_count (A, B) clones rounded up to
    the lamport 3-slot multiple."""
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    import nvshmem.core
    from kernels.ar.ar_norm_lamport import ARRmsNormOneShot, PING_PONG_SIZE, make_lamport_buffers
    from kernels.ar.ar_norm_two_shot import ARRmsNormTwoShotMultimem

    fd = lambda t: from_dlpack(t, assumed_align=16)  # noqa: E731
    n = -(-max(l2_rotate_count(A, B), 2 * PING_PONG_SIZE) // PING_PONG_SIZE) * PING_PONG_SIZE
    A_sets = [A.clone() for _ in range(n)]
    B_sets = [B.clone() for _ in range(n)]
    gamma_bf = gamma.to(torch.bfloat16).contiguous()
    eps = cutlass.Float32(EPS)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    if not two_shot:
        buf, bufs = make_lamport_buffers(M, N, world, torch.bfloat16)
        D = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
        out = torch.empty_like(D)
        res_out = torch.empty_like(D)
        compiled = cute.compile(
            ARRmsNormOneShot(cutlass.BFloat16, N), rank, world, cutlass.Int32(0),
            fd(D), fd(gamma_bf), fd(res_full), fd(out), fd(res_out), [fd(b) for b in bufs],
            eps, stream)
        it = {"i": 0}

        def ar(s):
            compiled(cutlass.Int32(it["i"] % PING_PONG_SIZE), fd(D), fd(gamma_bf), fd(res_full),
                     fd(out), fd(res_out), [fd(b) for b in bufs], eps, s)
            it["i"] += 1

        keep = (buf, bufs, compiled)
    else:
        assert M % world == 0, "two-shot slabs are row-aligned"
        slab = M // world
        rows = slice(rank * slab, (rank + 1) * slab)
        kern = ARRmsNormTwoShotMultimem(cutlass.BFloat16, N)
        tiler_m = kern.config.num_threads // kern.config.threads_per_row
        num_ctas = -(-slab // tiler_m) * kern.config.cluster_n

        def symm(shape, dtype):
            t = nvshmem.core.tensor(shape, dtype=dtype)
            return t, nvshmem.core.get_multicast_tensor(nvshmem.core.Teams.TEAM_NODE, t)

        D, D_mc = symm((M, N), torch.bfloat16)
        out, out_mc = symm((M, N), torch.bfloat16)
        res_out, res_out_mc = symm((M, N), torch.bfloat16)
        flag, flag_mc = symm((num_ctas,), torch.int32)
        flag.fill_(0)
        compiled = cute.compile(
            kern, world, fd(D_mc[rows]), fd(gamma_bf), fd(res_full[rows]), fd(out_mc[rows]),
            fd(res_out_mc[rows]), fd(flag), fd(flag_mc), eps, stream)

        def ar(s):
            compiled(fd(D_mc[rows]), fd(gamma_bf), fd(res_full[rows]), fd(out_mc[rows]),
                     fd(res_out_mc[rows]), fd(flag), fd(flag_mc), eps, s)

        keep = (D_mc, out_mc, res_out_mc, flag, flag_mc, compiled)

    if gemm == "cutlass":
        # Same GEMM the fused lamport kernel uses (cutlass dense_gemm_persistent, the
        # fused arm's tile ladder), so fusion is the only variable. Operands are the
        # arm's own A/B clones (distinct storage; prepare_tensors' temp aliases sets).
        sys.path.insert(0, os.path.expanduser(
            "~/cutlass/examples/python/CuTeDSL/cute/blackwell/kernel/dense_gemm"))
        import cutlass.utils as cutlass_utils
        from cutlass.utils import create_cute_tensor_for_fp8
        from dense_gemm_persistent import compile_bmm
        assert not two_shot, "cutlass GEMM wired for the one-shot (plain D) path only"
        if M % 256 == 0:
            tiler, cluster, use2 = (256, 256), (2, 1), True
        elif M % 128 == 0:
            tiler, cluster, use2 = (128, 256), (1, 1), False
        else:
            tiler, cluster, use2 = (64, 64), (1, 1), False
        Kl = K // world
        mnkl = (M, N, Kl, 1)
        ct = lambda t, ld: create_cute_tensor_for_fp8(  # noqa: E731
            t, cutlass.BFloat16, ld, source_f32_tensor=t.float())
        a_sets_c = [ct(a.reshape(1, M, Kl), 2) for a in A_sets]
        # b_major="k" wants B as (l, K, N) with K contiguous: the (N, Kl) row-major
        # clone viewed transposed, ld=1. Presenting it as (1, N, Kl) ld=2 swaps N/K
        # inside the kernel (half of D left unwritten, rel_fro ~1.2).
        b_sets_c = [ct(b.reshape(1, N, Kl).transpose(1, 2), 1) for b in B_sets]
        D3 = D.reshape(1, M, N)
        D_cute = ct(D3, 2)
        max_ac = cutlass_utils.HardwareInfo().get_max_active_clusters(cluster[0] * cluster[1])
        cgemm = compile_bmm(mnkl, a_sets_c[0], b_sets_c[0], D_cute, cutlass.Float32,
                            "k", "k", "n", tiler, cluster, max_ac, use2, use_tma_store=True)

        def fn(a_, b_):
            s = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
            cgemm(a_, b_, D_cute, s)
            ar(s)

        return Arm(fn, arg_sets=[(a_sets_c[i], b_sets_c[i]) for i in range(n)],
                   read_out=lambda: out.float(),
                   keep=(A_sets, B_sets, a_sets_c, b_sets_c, D, D_cute, cgemm, out, res_out,
                         gamma_bf, keep))

    def fn(a, b):
        torch.matmul(a, b.T, out=D)
        ar(cuda.CUstream(torch.cuda.current_stream().cuda_stream))  # stream at call time

    return Arm(fn, arg_sets=[(A_sets[i], B_sets[i]) for i in range(n)],
               read_out=lambda: out.float(),
               keep=(A_sets, B_sets, D, out, res_out, gamma_bf, keep))


def _quack_norm(A, B, res_full, gamma, M, N, K, rank, world):
    """Plain quack fused GEMM+AR + the same separate norm kernel as the
    cutlass arm — the accounting cross-check."""
    from flashinfer.norm import fused_add_rmsnorm

    kern = _quack_plain(A, B, M, N, K, rank, world)
    out2d = kern.zero.__self__[0]  # (M, N) bf16 view of d_arg
    gamma_bf = gamma.to(torch.bfloat16).contiguous()
    res = res_full.clone()

    def fn(a, b, swizzle=8):
        kern.fn(a, b, swizzle)
        # In place on res (drifts across timed reps); the ref-check run is the fresh first run.
        fused_add_rmsnorm(out2d, res, gamma_bf, EPS)

    return Arm(fn, args=(A, B), configs=[{"swizzle": 1}, {"swizzle": 8}],
               read_out=lambda: out2d.float())


def _quack_sentinel(A, B, res_full, gamma, M, N, K, rank, world):
    """quack gemm+ar sentinel-exit producer (no exit barrier, scrubs -0.0 in the
    broadcast; QUACK_SENTINEL_EXIT=1) + the certifier-shaped fused norm
    (rmsnorm sentinel_fused: per-16B-chunk poll, gmem->register load, re-arm
    after load). The quack side (worktree ~/quack-sentinel-exit) was removed
    2026-08-30; derivation in results/2026-08-24/bench.md, so this arm no
    longer runs."""
    import cutlass as _cutlass_mod
    from quack.distributed.gemm_epi_reduce import make_epi_reduce_args
    from quack.gemm import gemm as quack_gemm
    from quack.rmsnorm import _compile_rmsnorm_fwd

    assert M % 256 != 0, "decode path: 64x64 1-CTA tiles"
    tile_m, tile_n, cluster_m, cluster_n = 64, 64, 1, 1
    d_arg, era = make_epi_reduce_args(
        "all_reduce", torch.bfloat16, M, N, 1, tile_m, tile_n, cluster_m, world)
    out2d = d_arg[0]  # (M, N) bf16 view of symmetric D (the AR output)
    out2d.view(torch.int16).fill_(-32768)  # arm D with the bf16 -0.0 sentinel
    gamma_bf = gamma.to(torch.bfloat16).contiguous()
    res = res_full.to(torch.bfloat16).contiguous()
    out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    res_out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    bf16 = _cutlass_mod.BFloat16
    norm_fn = _compile_rmsnorm_fwd(
        bf16, bf16, bf16, bf16, None, bf16, N,
        False, False, False, False, 0.0, sentinel_fused=True)
    _keep = (d_arg, era, out2d, gamma_bf, res, out, res_out)  # noqa: F841

    def fn(a, b, swizzle=8):
        quack_gemm(a.unsqueeze(0), b.unsqueeze(0), d_arg, None, None, tile_m, tile_n,
                   cluster_m, cluster_n, max_swizzle_size=swizzle,
                   epi_reduce_mode="all_reduce", epi_reduce_args=era)
        norm_fn(out2d, gamma_bf, None, res, out, res_out, None, None, EPS)

    return Arm(fn, args=(A, B), configs=[{"swizzle": 1}, {"swizzle": 8}],
               read_out=lambda: out.float())


ARMS = {
    "quack":       {"factory": _quack,        "init": "nvshmem"},
    "quack_norm":  {"factory": _quack_norm,   "init": "nvshmem"},
    "quack_sentinel": {"factory": _quack_sentinel, "init": "nvshmem"},
    "cutlass":     {"factory": _cutlass_norm, "init": "symm_mem"},
    "cutlass_sentinel": {"factory": _cutlass_sentinel, "init": "symm_mem"},
    "cutlass_lamport":  {"factory": _cutlass_lamport_norm, "init": "symm_mem"},
    "batch_lamport":    {"factory": _batch_lamport_norm,   "init": "symm_mem"},
    "cublas_nccl": {"factory": _cublas_nccl,  "init": "plain"},
    "flashinfer":  {"factory": _flashinfer,   "init": "plain"},
    "quack_ar_norm_1shot": {"factory": _quack_ar_norm, "init": "nvshmem_core"},
    "cutlass_ar_norm_1shot": {"factory": partial(_quack_ar_norm, gemm="cutlass"),
                              "init": "nvshmem_core"},
    "quack_ar_norm_2shot": {"factory": partial(_quack_ar_norm, two_shot=True),
                            "init": "nvshmem_core", "unsynced": True},
}

def ref_norm(A, B, res_full, gamma, world):
    """fp32 reference: rmsnorm(full-K GEMM sum + residual) * gamma."""
    A_sh = [torch.empty_like(A) for _ in range(world)]
    B_sh = [torch.empty_like(B) for _ in range(world)]
    dist.all_gather(A_sh, A)
    dist.all_gather(B_sh, B)
    x = torch.cat(A_sh, 1).float() @ torch.cat(B_sh, 1).float().T + res_full.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * gamma.float()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("baseline", choices=list(ARMS))
    p.add_argument("--m", type=int, required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--ref_check", action="store_true")
    p.add_argument("--rounds", type=int, default=1,
                   help="re-time the winning config N-1 more times back-to-back; "
                        "cublas_nccl is timed once, last, as the contention canary")
    args = p.parse_args()
    M, N, K = args.m, args.n, args.k

    spec = ARMS[args.baseline]
    if spec["init"] == "nvshmem":
        from quack.dist_utils import init_distributed, clean_distributed
        init_distributed()
        finalize = clean_distributed
    elif spec["init"] == "symm_mem":
        sys.path.insert(0, os.path.expanduser(CUTLASS_AR_DIR))
        from distributed_gemm_all_reduce_ldxstmc_blackwell import (
            init_distributed, clean_distributed)
        init_distributed()
        finalize = clean_distributed
    elif spec["init"] == "nvshmem_core":
        from kernels.misc.utils import torchrun_init_nvshmem
        torchrun_init_nvshmem()
        # teardown wedges (captured-NCCL graph + nvshmem in one process): hard-exit past it
        finalize = lambda: os._exit(0)  # noqa: E731
    else:
        from utils import torchrun_init, torchrun_finalize
        torchrun_init()
        finalize = torchrun_finalize
    rank, world = dist.get_rank(), dist.get_world_size()
    assert M % world == 0 and K % world == 0

    torch.manual_seed(1111 + rank)
    inv_sqrt_k = 1.0 / math.sqrt(K)
    A = (torch.empty(M, K // world, dtype=torch.float32, device="cuda")
         .normal_().mul_(inv_sqrt_k).to(torch.bfloat16))
    B = (torch.empty(N, K // world, dtype=torch.float32, device="cuda")
         .normal_().mul_(inv_sqrt_k).to(torch.bfloat16))
    torch.manual_seed(3333)
    res_full = torch.randn(M, N, dtype=torch.float32).to(torch.bfloat16).cuda()
    torch.manual_seed(2222)
    gamma = (torch.randn(N, dtype=torch.float32) * 0.02 + 1.0).cuda()

    # One protocol: bench_rotate.bench_arm (the quack-autotuner methodology);
    # every factory builds its rotation sets up front, checks run eagerly on
    # set 0 of the same build. A non-capturable arm would set "capture": False
    # in its ARMS entry with a comment saying why (none currently).
    arm = spec["factory"](A, B, res_full, gamma, M, N, K, rank, world)

    check = None
    if args.ref_check:  # the eager first run: fresh residual
        for _ in range(2 if spec.get("unsynced") else 1):  # 2nd pass: peers' GEMM output complete
            run0(arm)
            torch.cuda.synchronize()
            dist.barrier()
        got = arm.read_out()
        ref = ref_norm(A, B, res_full, gamma, world)
        mask = ref.abs() > 1e-3
        check = ((got - ref).abs()[mask] / ref.abs()[mask]).max().item()

    capture = spec.get("capture", True)
    proto = "L2-rotate" if capture else "L2-rotate (eager-timed)"
    sweep_ms = bench_arm(arm, capture=capture)
    best_label, ms = min(sweep_ms, key=lambda r: r[1])
    # Rounds re-time the winning config back-to-back; the canary times once, last
    # (as bench_baselines; skipped when cublas_nccl is itself the arm).
    cfgs = arm.configs or [{}]
    best_cfg = cfgs[[" ".join(f"{k}={v}" for k, v in c.items()) for c in cfgs].index(best_label)]
    rounds = [bench_arm(arm._replace(configs=[best_cfg]), capture=capture)[0][1]
              for _ in range(args.rounds - 1)]
    ref = None if args.baseline == "cublas_nccl" else _cublas_nccl(
        A, B, res_full, gamma, M, N, K, rank, world)
    ref_ms = bench_arm(ref, capture=capture)[0][1] if ref else None

    clean_print("=" * 80, print_once=True)
    clean_print(f"<llama GEMM+AR+resid+RMSNorm {args.baseline} | world={world} | "
                f"{M}x{N}x{K} | {proto}>", print_once=True)
    if len(sweep_ms) > 1:
        for label, v in sweep_ms:
            clean_print(f"  {args.baseline + ' ' + label + ':':<22} {v * 1e3:.1f} us",
                        print_once=True)
    name = f"{args.baseline} best ({best_label})" if len(sweep_ms) > 1 else args.baseline
    clean_print(f"  {name + ':':<12} {ms * 1e3:.1f} us", print_once=True)
    for i, v in enumerate(rounds):
        clean_print(f"  {args.baseline + f' round{i + 1}:':<22} {v * 1e3:.1f} us", print_once=True)
    if ref_ms is not None:
        clean_print(f"  {'cuBLAS+NCCL:':<22} {ref_ms * 1e3:.1f} us", print_once=True)
    if check is not None:
        clean_print(f"  rel err vs fp32 ref: {check:.4e} (|ref|>1e-3)", print_once=True)

    # A live captured-NCCL CUDA graph wedges destroy_process_group; drop every
    # holder first (the rotate driver's timed graph died at its return).
    arm = ref = None  # noqa: F841
    dist.barrier()
    finalize()


if __name__ == "__main__":
    main()
