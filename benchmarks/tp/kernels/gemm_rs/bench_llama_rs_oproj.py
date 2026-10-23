"""GEMM+RS + residual + RMSNorm: the llama o_proj sequence-parallel RS chain
(bench_llama_ag_swiglu.py is the AG side it feeds; down_proj is the same
chain at (N=hidden, K=intermediate)). Each rank computes its K-shard GEMM;
reduce-scatter leaves the rank its M-slab; residual add + RMSNorm feed the
next op's AG. Arms:

  quack     fused GEMM+RS with the sq-reduce epilogue: one kernel emits the
            slab D = (RS + residual)*gamma plus per-N-tile squared-sum
            partials (slab-local -- sequence parallel needs no rstd
            exchange); rms_final_reduce folds them into slab rstd. The 1/rms
            scale rides the next op (ag_gemm.md SwiGLU section). Whole chain
            in one CUDA graph.
  quack_unfused  plain quack GEMM+RS + the shared compiled add+rmsnorm --
            quack minus this = the sq-reduce fusion win with transport and
            GEMM engine held constant.
  cutlass   cutlass fused GEMM+RS (LDMC) + torch.compile'd residual+RMSNorm
            over the slab output (the slab traffic quack fuses away).
  tk        ThunderKittens matmul_reduce_scatter + the shared norm kernel
            (GPU=B300 env; main's A/B copied into the factory's tensors).
  asynctp   torch fused_matmul_reduce_scatter + the shared norm kernel.
            EAGER -- the symm_mem op host-blocks and is not graph-safe.

82 RS excluded: fixed binary, the norm cannot run inside its process.
Timing: the L2-cold rotate protocol of bench_baselines (do_bench_l2_rotate_all
captures each arm's chain over cloned (A, B) k-shards; sw1/sw8 sweep, best
reported); tk falls back to counts and says so. Reference leg (canary, every
invocation): cuBLAS GEMM + NCCL reduce_scatter + the same compiled add+rmsnorm.

Inputs standardized as bench_baselines.py (A,B ~ N(0,1/sqrt(K)) bf16, seed
1111+rank); residual and gamma rank-invariant (seeds 3333/2222); residual is
slab-shaped per rank (sequence parallel). o_proj shapes, M=8192: N=K=hidden
4096/8192/16384 (8B/70B/405B).

Usage:
    torchrun --nproc_per_node=4 bench_llama_rs_oproj.py quack --m 8192 --n 4096 --k 4096
    torchrun --nproc_per_node=4 bench_llama_rs_oproj.py cutlass --m 8192 --n 4096 --k 4096 --ref_check
"""
import argparse
import gc
import math
import os
import sys
import time
from collections import namedtuple
from functools import partial

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "misc"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_rotate import do_bench_l2_rotate_all, l2_rotate_arg_sets, l2_rotate_count  # noqa: E402
from utils import do_bench_counts, clean_print, parse_quack_configs, quack_config_label  # noqa: E402

from bench_baselines import _asynctp, _blackwell_build, _tk  # noqa: E402

_NOOP = lambda: None  # noqa: E731
EPS = 1e-6
# quack config candidates (tile x swizzle; static only — epi_reduce_mode requires static
# persistence); main overrides from --quack_configs
quack_configs = parse_quack_configs(clc_ok=False)

# run: the chain, eager. read_norm: the effective normed slab (M/world, N) fp32.
# fn: the chain on caller-supplied operands (the rotate protocol's per-set launch),
# rotating `operands` (default the (A, B) k-shards) or, for arms that own their operand
# buffers, the prebuilt `arg_sets`.
Arm = namedtuple("Arm", ["run", "read_norm", "fn", "operands", "arg_sets", "sweep"],
                 defaults=[None, None, None, None])


def _quack_cfgs(M):
    return quack_configs if M % 256 == 0 else [(64, 64, 1, 1, 8, False)]


def _quack(A, B, res_slab, gamma, M, N, K, rank, world):
    from quack.distributed.gemm_epi_reduce import make_epi_reduce_args
    from quack.epilogue.library import sq_reduce_mod
    from quack.rms_final_reduce import rms_final_reduce

    cfgs = _quack_cfgs(M)
    m_per = M // world
    c_gpu = res_slab.unsqueeze(0).contiguous()
    w_gpu = gamma.float().reshape(1, N).contiguous()
    rstd = torch.empty(m_per, dtype=torch.float32, device="cuda")
    mod = sq_reduce_mod(has_c=True, has_rowvec=True, has_aux=False)
    # per candidate tile: RS output/workspace/flags + the per-N-tile sq-sum partials
    per_tile = {}
    for t in sorted({c[:4] for c in cfgs}):
        tile_m, tile_n, cl_m, _ = t
        d_arg, era = make_epi_reduce_args(
            "reduce_scatter", torch.bfloat16, M, N, 1, tile_m, tile_n, cl_m, world)
        n_tiles = (N + tile_n - 1) // tile_n
        colvec = torch.zeros(1, m_per, n_tiles, dtype=torch.float32, device="cuda")
        per_tile[t] = (d_arg, era, colvec)
    last = {}

    def fn(a, b, cfg=None):
        tile_m, tile_n, cl_m, cl_n, swizzle, clc = cfg or cfgs[0]
        d_arg, era, colvec = per_tile[(tile_m, tile_n, cl_m, cl_n)]
        mod.gemm(a.unsqueeze(0), b.unsqueeze(0), d_arg, c_gpu,
                 epi_args={"mRowVecBroadcast": w_gpu, "mColVecReduce": colvec},
                 tile_M=tile_m, tile_N=tile_n, cluster_M=cl_m, cluster_N=cl_n,
                 is_dynamic_persistent=clc, max_swizzle_size=swizzle,
                 epi_reduce_mode="reduce_scatter", epi_reduce_args=era)
        rstd.copy_(rms_final_reduce(colvec[0], scale=1.0 / N, eps=EPS))
        last["d"] = d_arg

    read = lambda: last["d"][0].float() * rstd[:, None]  # noqa: E731
    sweep = [(quack_config_label(c), partial(fn, cfg=c)) for c in cfgs]
    return Arm(lambda: fn(A, B), read, fn, sweep=sweep)


def _norm_fn(gamma_bf):
    def add_norm(d, r, w):
        x = (d + r).float()
        return ((x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS))
                .to(torch.bfloat16) * w)
    return torch.compile(add_norm)


def _cutlass_norm(A, B, res_slab, gamma, M, N, K, rank, world, rotate=False):
    """cutlass RS (bench_baselines._blackwell_build) + the compiled residual+RMSNorm
    over the slab output. rotate: l2_rotate_count operand sets and the sw1/sw8
    kernels, exposed as arg_sets for fn(args, swizzle)."""
    n_sets, swizzles = (l2_rotate_count(A, B), (1, 8)) if rotate else (1, (8,))
    launch, arg_list, t, _kept = _blackwell_build(torch.bfloat16, M, N, K, rank, world, A, B,
                                                  n_sets=n_sets,
                                                  configs=[(256, 256, 2, 1, s) for s in swizzles])
    # comm_out is the (M/world, N, L) slab the RS commits into.
    out2d = t["torch_tensor_comm_out_gpu"].permute(2, 0, 1)[0]
    gamma_bf = gamma.to(torch.bfloat16).contiguous()
    normed = torch.empty_like(out2d)
    norm = _norm_fn(gamma_bf)

    def fn(args, swizzle=8):
        launch(args, (256, 256, 2, 1, swizzle))
        normed.copy_(norm(out2d, res_slab, gamma_bf))

    return Arm(lambda: fn(arg_list[0]), lambda: normed.float(), fn,
               arg_sets=[(args,) for args in arg_list])


def _quack_unfused(A, B, res_slab, gamma, M, N, K, rank, world):
    from quack.distributed.gemm_epi_reduce import make_epi_reduce_args
    from quack.gemm import gemm as quack_gemm

    cfgs = _quack_cfgs(M)
    per_tile = {  # RS output/workspace/flags depend on the tile
        t: make_epi_reduce_args("reduce_scatter", torch.bfloat16, M, N, 1, t[0], t[1], t[2], world)
        for t in sorted({c[:4] for c in cfgs})
    }
    gamma_bf = gamma.to(torch.bfloat16).contiguous()
    normed = torch.empty(M // world, N, dtype=torch.bfloat16, device="cuda")
    norm = _norm_fn(gamma_bf)

    def fn(a, b, cfg=None):
        tile_m, tile_n, cl_m, cl_n, swizzle, clc = cfg or cfgs[0]
        d_arg, era = per_tile[(tile_m, tile_n, cl_m, cl_n)]
        quack_gemm(a.unsqueeze(0), b.unsqueeze(0), d_arg, None, None, tile_m, tile_n, cl_m, cl_n,
                   is_dynamic_persistent=clc, max_swizzle_size=swizzle,
                   epi_reduce_mode="reduce_scatter", epi_reduce_args=era)
        normed.copy_(norm(d_arg[0], res_slab, gamma_bf))

    sweep = [(quack_config_label(c), partial(fn, cfg=c)) for c in cfgs]
    return Arm(lambda: fn(A, B), lambda: normed.float(), fn, sweep=sweep)


def _tk_norm(A, B, res_slab, gamma, M, N, K, rank, world):
    kern = _tk(torch.bfloat16, M, N, K, rank, world)
    kern.A.copy_(A)  # exact standardized inputs (run() reads these tensors)
    kern.B.copy_(B)
    slab = kern.raw  # (M/world, N) slab the RS commits into
    gamma_bf = gamma.to(torch.bfloat16).contiguous()
    normed = torch.empty_like(slab)
    norm = _norm_fn(gamma_bf)

    def chain():
        kern.run()
        normed.copy_(norm(slab, res_slab, gamma_bf))

    return Arm(chain, lambda: normed.float())


def _asynctp_norm(A, B, res_slab, gamma, M, N, K, rank, world):
    kern = _asynctp(torch.bfloat16, M, N, K, rank, world)
    kern.A.copy_(A)  # exact standardized inputs
    kern.B.copy_(B)
    b_kn = kern.raw["B_kn"]
    b_kn.copy_(B.T)  # the op reads this CONTIGUOUS COPY, not B
    gamma_bf = gamma.to(torch.bfloat16).contiguous()
    normed = torch.empty(M // world, N, dtype=torch.bfloat16, device="cuda")
    norm = _norm_fn(gamma_bf)

    def fn(a, b_kn):
        kern.fn(a, b_kn)
        normed.copy_(norm(kern.raw["out"], res_slab, gamma_bf))

    return Arm(lambda: fn(kern.A, b_kn), lambda: normed.float(), fn, operands=(A, b_kn))


def _cublas_nccl(A, B, res_slab, gamma, M, N, K, rank, world):
    gamma_bf = gamma.to(torch.bfloat16)
    D_full = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    D_slab = torch.empty(M // world, N, dtype=torch.bfloat16, device="cuda")
    normed = torch.empty_like(D_slab)
    norm = _norm_fn(gamma_bf)

    def fn(a, b):
        torch.matmul(a, b.T, out=D_full)
        dist.reduce_scatter_tensor(D_slab, D_full)
        normed.copy_(norm(D_slab, res_slab, gamma_bf))

    return Arm(lambda: fn(A, B), lambda: normed.float(), fn)


ARMS = {
    "quack":         _quack,
    "quack_unfused": _quack_unfused,
    "tk":            _tk_norm,
    "cutlass":       _cutlass_norm,
    "asynctp":       _asynctp_norm,
}

# arm -> (factory for the rotate build, capture, swizzle sweep). capture=False for
# asynctp: the symm_mem op host-blocks and is not graph-safe. tk: not rotate-capable.
ROTATE_INPUTS = {
    "quack":         (_quack, True, None),  # sweep = the arm's quack configs
    "quack_unfused": (_quack_unfused, True, None),
    "cutlass":       (partial(_cutlass_norm, rotate=True), True, (1, 8)),
    "asynctp":       (_asynctp_norm, False, None),
}


def _arm_l2_rotate_inputs(factory, swizzles, A, B, res_slab, gamma, M, N, K, rank, world):
    """(fn, arg_sets, keep, sweep) for do_bench_l2_rotate_all: the arm's prebuilt
    arg_sets, else cloned copies of its operands (the L2-turnover payload; the
    residual and every symm workspace stay fixed in fn's closure). sweep = the
    arm's own (quack configs) else the swizzle candidates."""
    arm = factory(A, B, res_slab, gamma, M, N, K, rank, world)
    arg_sets = arm.arg_sets or l2_rotate_arg_sets(*(arm.operands or (A, B)))
    sweep = arm.sweep or (
        [(f"sw{s}", partial(arm.fn, swizzle=s)) for s in swizzles] if swizzles else None)
    return arm.fn, arg_sets, (arm,), sweep


def ref_norm(A, B, res_slab, gamma, M, rank, world):
    """fp32 reference: rmsnorm(full-K GEMM RS slab + residual) * gamma."""
    A_sh = [torch.empty_like(A) for _ in range(world)]
    B_sh = [torch.empty_like(B) for _ in range(world)]
    dist.all_gather(A_sh, A)
    dist.all_gather(B_sh, B)
    x_full = torch.cat(A_sh, 1).float() @ torch.cat(B_sh, 1).float().T
    m_per = M // world
    x = x_full[rank * m_per : (rank + 1) * m_per] + res_slab.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * gamma.float()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("baseline", choices=list(ARMS))
    p.add_argument("--m", type=int, required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--ref_check", action="store_true")
    p.add_argument("--quack_configs", default=None,
                   help="quack sweep: 'tm,tn,cm,cn[,sw,static];...' (default: QUACK_TILE + "
                        "256x256/256x512 c(2,1) x swizzle {1,8}; static only under epi_reduce)")
    args = p.parse_args()
    M, N, K = args.m, args.n, args.k
    global quack_configs
    quack_configs = parse_quack_configs(args.quack_configs, clc_ok=False)

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
    m_per = M // world
    torch.manual_seed(3333)
    res_slab = (torch.randn(M, N, dtype=torch.float32).to(torch.bfloat16)
                [rank * m_per : (rank + 1) * m_per].contiguous().cuda())
    torch.manual_seed(2222)
    gamma = (torch.randn(N, dtype=torch.float32) * 0.02 + 1.0).cuda()

    rot = ROTATE_INPUTS.get(args.baseline)
    need_plain = args.ref_check or rot is None
    arm = ARMS[args.baseline](A, B, res_slab, gamma, M, N, K, rank, world) if need_plain else None
    ref = _cublas_nccl(A, B, res_slab, gamma, M, N, K, rank, world) if need_plain else None

    checks = []
    if args.ref_check:
        for name, a in ((args.baseline, arm), ("ref", ref)):
            a.run()
            torch.cuda.synchronize()
            dist.barrier()
            got = a.read_norm()
            r = ref_norm(A, B, res_slab, gamma, M, rank, world)
            mask = r.abs() > 1e-3
            checks.append((name, ((got - r).abs()[mask] / r.abs()[mask]).max().item()))
        del a

    sweep_ms = []
    if rot is not None:
        from quack.autotuner import _gpu_warmup

        factory, capture, swizzles = rot
        proto = "L2-rotate" if capture else "L2-rotate (eager-timed)"
        fn, arg_sets, _keep, sweep = _arm_l2_rotate_inputs(
            factory, swizzles, A, B, res_slab, gamma, M, N, K, rank, world)
        ref_fn, ref_arg_sets, _ref_keep, _ = _arm_l2_rotate_inputs(
            _cublas_nccl, None, A, B, res_slab, gamma, M, N, K, rank, world)
        _gpu_warmup()
        kwarg_sets = [{}] * len(arg_sets)
        if sweep:  # swizzle 1 vs 8: report each, rank the best
            for label, fn_v in sweep:
                dist.barrier()
                sweep_ms.append((label, do_bench_l2_rotate_all(
                    fn_v, arg_sets, kwarg_sets, {}, capture=capture)))
            best_label, ms = min(sweep_ms, key=lambda r: r[1])
        else:
            dist.barrier()
            ms = do_bench_l2_rotate_all(fn, arg_sets, kwarg_sets, {}, capture=capture)
        dist.barrier()
        ref_ms = do_bench_l2_rotate_all(ref_fn, ref_arg_sets, [{}] * len(ref_arg_sets), {})
    else:
        proto = "counts — not rotate-capable yet"
        dist.barrier()
        time.sleep(1.0)
        ms = do_bench_counts(arm.run)
        dist.barrier()
        time.sleep(1.0)
        ref_ms = do_bench_counts(ref.run)

    tf = lambda v: 2 * M * N * (K // world) / (v * 1e9)  # noqa: E731
    clean_print("=" * 80, print_once=True)
    clean_print(f"<llama RS o_proj chain {args.baseline} | world={world} | "
                f"{M}x{N}x{K} (K global) | {proto}>", print_once=True)
    for label, v in sweep_ms:
        clean_print(f"  {args.baseline + ' ' + label + ':':<22} {v:.3f} ms | {tf(v):.2f} TFLOp/s",
                    print_once=True)
    name = f"{args.baseline} best ({best_label})" if sweep_ms else args.baseline
    clean_print(f"  {name + ':':<18} {ms:.3f} ms | {tf(ms):.2f} TFLOp/s", print_once=True)
    clean_print(f"  {'cuBLAS+NCCL+norm:':<18} {ref_ms:.3f} ms | {tf(ref_ms):.2f} TFLOp/s",
                print_once=True)
    for name, v in checks:
        clean_print(f"  {name + ' rel err:':<18} {v:.4e} (|ref|>1e-3)", print_once=True)

    # captured-NCCL graphs must die before finalize (ncclCommFinalize polls
    # forever with a live graph): drop every holder and collect NOW.
    arm = ref = _keep = _ref_keep = None  # noqa: F841
    gc.collect()
    torch.cuda.synchronize()
    dist.barrier()
    finalize()


if __name__ == "__main__":
    main()
