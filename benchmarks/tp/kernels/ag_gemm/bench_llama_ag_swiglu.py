"""AG + column-parallel GEMM + SwiGLU: the llama gate_up_proj
sequence-parallel AG site (bench_llama_rs.py is the RS side that feeds it;
the qkv/RoPE site -- CODA's linear_qknorm_rope analog -- is a future
sibling). Per the layer decomposition the RS side emits the UNSCALED slab
(RS+residual)*gamma plus per-slab-row rstd, deferring the 1/rms row scale to
this op. Every arm consumes the SAME seeded activations A; only the
quack-fused arm additionally consumes rstd, an (M,) RESIDENT LOCAL INPUT
(the RS side distributes it: sq partials to a multicast object + replicated
tiny final reduce -- full-M assembly being designed there, assumed solved
here, so it costs this op nothing but the in-kernel colvec reads). The
other stacks' RS side applied the norm before this op, so their chains here
are norm-free: quack computes swiglu(rstd * (A_full @ W^T)), everyone else
swiglu(A_full @ W^T) -- identical cost, different constants; each arm is
ref-checked against its own formula.

SwiGLU pairing is INTERLEAVED (silu(pre[..., ::2]) * pre[..., 1::2]; W rows
(gate_0, up_0, gate_1, up_1, ...), a benign permutation of the fused
weight). PostAct (M, n_local/2); training semantics: the pre-act D
(M, n_local) is materialized by every arm.

Fusion audit (why the arms look like this): only quack's epilogue-mod path
expresses colvec-scale + gated pairing inside the GEMM -- cutlass efc and
cublasLt epilogues and inductor mm-template fusion are elementwise-only, tk
ships no activation variants, 82 is a fixed binary (excluded). Baselines get
their best honest form: ONE shared torch.compile'd swiglu kernel (out.copy_
fused in, single launch).

  quack           AllGatherRunner zc + self-composed norm_act_mod
                  (colvec=rstd, gated swiglu, ag_args): ONE kernel writing
                  D + PostAct; 4-call CUDA graph
  quack_unfused   plain gemm writing D + shared swiglu kernel, same AG and
                  graph -- quack minus this = the fusion win with transport
                  and GEMM engine held constant
  tk / asynctp    bench_baselines arm + shared swiglu kernel (eager)
  cutlass         bench_baselines graph + per-source-band swiglu views,
                  chained eagerly (an outer capture of the inner replay is
                  invalid; launches amortize in the burst)

NOTE on numerics: quack applies scale+act on the fp32 accumulator, the
unfused arms on the bf16-rounded D.

Timed against a NCCL all-gather + cuBLAS + shared-swiglu reference leg
(canary) under bench_baselines' L2-cold rotate protocol (cloned (B, D[,
PostAct]) sets rotate; the cutlass chain is eager-timed over one arm instance
per set; tk/asynctp fall back to counts and say so); TFLOP/s counts GEMM
flops only (2*M*n_local*K). N/K are GLOBAL weight dims (n_local = N/world).
Llama gate_up shapes, M=8192: N 28672/57344/106496 x K 4096/8192/16384
(8B/70B/405B).

Usage:
    torchrun --nproc_per_node=4 bench_llama_ag_swiglu.py quack --m 8192 --n 28672 --k 4096
    GPU=B300 torchrun --nproc_per_node=4 bench_llama_ag_swiglu.py tk --m 8192 --n 28672 --k 4096
"""
import argparse
import math
import os
import sys
import time
from functools import partial

import torch
import torch.distributed as dist
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "misc"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_rotate import do_bench_l2_rotate_all, l2_rotate_arg_sets, l2_rotate_count  # noqa: E402
from utils import (  # noqa: E402
    clean_print,
    parse_quack_configs,
    quack_config_label,
    torchrun_init,
    torchrun_finalize,
)

import bench_baselines as bb  # noqa: E402
from bench_baselines import Kernel, do_bench_counts, _NOOP  # noqa: E402

# quack config candidates (tile x swizzle x scheduler); main overrides from --quack_configs
quack_configs = parse_quack_configs()


def _operands(M, N, K, rank, world):
    """Standardized per-rank inputs: slab A and weight shard B (seed
    1111+rank, the bench_baselines factory generation order, so a seeded
    factory call regenerates them identically). rstd is the FULL (M,)
    resident vector (assumed pre-distributed by the RS side, see module
    docstring), built bit-identically on every rank from per-rank seeds --
    no collective. Only the quack-fused arm consumes it."""
    n_local = N // world
    shard_m = M // world
    torch.manual_seed(1111 + rank)
    A = torch.randn(shard_m, K, dtype=torch.bfloat16, device="cuda") / math.sqrt(K)
    B = torch.randn(n_local, K, dtype=torch.bfloat16, device="cuda") / math.sqrt(K)
    blocks = []
    for r in range(world):
        torch.manual_seed(7777 + r)
        blocks.append(torch.rand(shard_m, dtype=torch.float32, device="cuda") * 0.5 + 0.75)
    return A, B, torch.cat(blocks)


def make_epi(out):
    """The shared unfused swiglu epilogue, ONE compiled kernel (interleaved
    pairing); out.copy_ fuses in."""

    def f(x):
        pre = x.float()
        out.copy_((F.silu(pre[..., ::2]) * pre[..., 1::2]).to(torch.bfloat16))

    return torch.compile(f)


def _quack(M, N, K, rank, world):
    """Fused arm: flag-gated AG + GEMM + colvec(rstd) + swiglu in one kernel
    (norm_act_mod composed here -- mod.gemm takes ag_args like gemm_act
    does); rstd is a resident input, nothing else to distribute."""
    from quack.distributed import AllGatherRunner
    from quack.epilogue.library import norm_act_mod

    n_local = N // world
    shard_m = M // world
    A, B, rstd = _operands(M, N, K, rank, world)
    D = torch.empty(M, n_local, dtype=torch.bfloat16, device="cuda")
    PostAct = torch.empty(M, n_local // 2, dtype=torch.bfloat16, device="cuda")
    mod = norm_act_mod("swiglu", gated=True, has_c=False, has_rowvec=False,
                       has_colvec=True, sr=False)
    runner = AllGatherRunner(shard_m, K, torch.bfloat16)

    def fn(b, d, post, cfg=None):
        tile_m, tile_n, cluster_m, cluster_n, swizzle, clc = cfg or quack_configs[0]
        with runner.gather(runner.next_local_slot()) as (a_full, ag_args):
            mod.gemm(a_full, b, d, None,
                     epi_args=dict(mAuxOut=post, mColVecBroadcast=rstd),
                     ag_args=ag_args, tile_M=tile_m, tile_N=tile_n,
                     cluster_M=cluster_m, cluster_N=cluster_n, is_dynamic_persistent=clc,
                     max_swizzle_size=swizzle, tile_count_semaphore=None)

    for _ in range(len(runner.bufs)):
        runner.next_local_slot().copy_(A)
        runner.last_parity ^= 1
    torch.cuda.synchronize()
    return Kernel(lambda: fn(B, D, PostAct), _NOOP, A, B, lambda: PostAct.float(),
                  PostAct.zero_, raw=PostAct, fn=fn, operands=(B, D, PostAct))


def _quack_unfused(M, N, K, rank, world):
    """The unfused decomposition on the quack engine: plain GEMM writing D,
    shared swiglu kernel. quack - this = the fusion win with transport and
    GEMM engine held constant."""
    from quack.distributed import AllGatherRunner
    from quack.gemm import gemm as quack_gemm

    n_local = N // world
    shard_m = M // world
    A, B, _ = _operands(M, N, K, rank, world)
    D = torch.empty(M, n_local, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(M, n_local // 2, dtype=torch.bfloat16, device="cuda")
    epi = make_epi(out)
    runner = AllGatherRunner(shard_m, K, torch.bfloat16)

    def fn(b, d, cfg=None):
        tile_m, tile_n, cluster_m, cluster_n, swizzle, clc = cfg or quack_configs[0]
        with runner.gather(runner.next_local_slot()) as (a_full, ag_args):
            quack_gemm(a_full, b, d, None, None, tile_m, tile_n, cluster_m, cluster_n,
                       is_dynamic_persistent=clc, max_swizzle_size=swizzle, ag_args=ag_args)
        epi(d)

    for _ in range(len(runner.bufs)):
        runner.next_local_slot().copy_(A)
        runner.last_parity ^= 1
    torch.cuda.synchronize()
    return Kernel(lambda: fn(B, D), _NOOP, A, B, lambda: out.float(), out.zero_,
                  raw=out, fn=fn, operands=(B, D))


def _with_epi(kern, get_x, M, n_local):
    """Wrap a bench_baselines Kernel with the shared swiglu kernel over its
    raw output."""
    out = torch.empty(M, n_local // 2, dtype=torch.bfloat16, device="cuda")
    epi = make_epi(out)

    def chain(r):
        def timed():
            r()
            epi(get_x())
        return timed

    sweep = [(label, chain(r)) for label, r in kern.sweep] if kern.sweep else None

    def zero():
        kern.zero()
        out.zero_()

    return Kernel(chain(kern.run), kern.free, kern.A, kern.B, lambda: out.float(),
                  zero, sweep, kern.calls, raw=out)


def _tk(M, N, K, rank, world):
    torch.manual_seed(1111 + rank)  # factory regenerates the same A, B
    kern = bb._tk(torch.bfloat16, M, N // world, K, rank, world)
    return _with_epi(kern, lambda: kern.raw, M, N // world)


def _asynctp(M, N, K, rank, world):
    torch.manual_seed(1111 + rank)  # factory regenerates the same A, B
    kern = bb._asynctp(torch.bfloat16, M, N // world, K, rank, world)
    return _with_epi(kern, lambda: kern.raw["out"], M, N // world)


def _blackwell(M, N, K, rank, world, n_sets=1):
    """bench_baselines' graph-captured arm, one instance per rotation set
    (sharing A, cloned B, as its _blackwell_l2_rotate_inputs does); the swiglu
    runs over each instance's per-source M-band outputs (views into one out).
    The chain stays EAGER: kern.run is itself a graph replay, and capturing a
    cudaGraphLaunch into an outer capture is invalid
    (cudaErrorStreamCaptureInvalidated); the rotate driver's eager-timed mode
    hides the enqueue under the backlog."""
    n_local = N // world
    A, B, _ = _operands(M, N, K, rank, world)
    arms = [bb._blackwell(torch.bfloat16, M, n_local, K, rank, world, A=A, B=B.clone())
            for _ in range(n_sets)]
    m_per = M // world
    out = torch.empty(M, n_local // 2, dtype=torch.bfloat16, device="cuda")

    def f(x, o):
        pre = x.float()
        o.copy_((F.silu(pre[..., ::2]) * pre[..., 1::2]).to(torch.bfloat16))

    fepi = torch.compile(f)

    def fn(arm):
        arm.run()
        bands = arm.raw  # c_local_torch_list, source-rank order, (m_per, n, 1)
        for r in range(world):
            rows = slice(r * m_per, (r + 1) * m_per)
            fepi(bands[r][:, :, 0], out[rows])

    def zero():
        arms[0].zero()
        out.zero_()

    return Kernel(lambda: fn(arms[0]), arms[0].free, A, B, lambda: out.float(), zero,
                  None, arms[0].calls, raw=out, fn=fn, arg_sets=[(arm,) for arm in arms])


def _ref(M, N, K, rank, world):
    """NCCL all-gather + cuBLAS + shared swiglu: the reference leg and
    contention canary, run in every invocation."""
    n_local = N // world
    A, B, _ = _operands(M, N, K, rank, world)
    A_full = torch.empty(M, K, dtype=torch.bfloat16, device="cuda")
    D = torch.empty(M, n_local, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(M, n_local // 2, dtype=torch.bfloat16, device="cuda")
    epi = make_epi(out)

    def fn(b, d):
        dist.all_gather_into_tensor(A_full, A)
        torch.matmul(A_full, b.T, out=d)
        epi(d)

    return Kernel(lambda: fn(B, D), _NOOP, A, B, lambda: out.float(), out.zero_, raw=out,
                  fn=fn, operands=(B, D))


ARMS = {
    "quack": _quack,
    "quack_unfused": _quack_unfused,
    "tk": _tk,
    "cutlass":   _blackwell,
    "asynctp": _asynctp,
}


def _blackwell_rotate(M, N, K, rank, world):
    _, B, _ = _operands(M, N, K, rank, world)
    D = torch.empty(M, N // world, dtype=torch.bfloat16, device="cuda")  # sizing only
    return _blackwell(M, N, K, rank, world, n_sets=l2_rotate_count(B, D))


# arm -> (factory for the rotate build, capture, sweep the quack configs). cutlass:
# eager-timed (its iteration is a graph replay). tk/asynctp: not rotate-capable, counts.
ROTATE_INPUTS = {
    "quack":         (_quack, True, True),
    "quack_unfused": (_quack_unfused, True, True),
    "cutlass":       (_blackwell_rotate, False, False),
}


def _kernel_l2_rotate_inputs(factory, M, N, K, rank, world, configs=None):
    """(fn, arg_sets, keep, sweep) for do_bench_l2_rotate_all: the arm's prebuilt
    arg_sets, else cloned copies of its operands (the L2-turnover payload; A
    rides the runner and is CE-rewritten every call). configs: quack config
    candidates, each bound to fn as a sweep entry (best reported)."""
    kern = factory(M, N, K, rank, world)
    sweep = [(quack_config_label(c), partial(kern.fn, cfg=c)) for c in configs] if configs else None
    return kern.fn, kern.arg_sets or l2_rotate_arg_sets(*kern.operands), (kern,), sweep


def ref_check(kern, M, N, K, rank, world, scaled):
    """max|out - ref| / max|ref| of the arm's final output vs an fp32 reference of
    ITS OWN formula (regenerated seeded operands): swiglu(A_full @ B^T), with the
    rstd colvec folded in when `scaled` (the quack-fused arm only). Scale-relative
    (bench_mxfp8_ag's metric): the 1/sqrt(K)-scaled operands put the SwiGLU
    outputs near 1/K, so a per-element mask leaves nothing at large K."""
    bb._run_synced(kern)
    got = kern.read_out()
    A, B, rstd = _operands(M, N, K, rank, world)
    A_shards = [torch.empty_like(A) for _ in range(world)]
    dist.all_gather(A_shards, A)
    pre = torch.cat(A_shards).float() @ B.float().T
    if scaled:
        pre = rstd[:, None] * pre
    ref = F.silu(pre[..., ::2]) * pre[..., 1::2]
    return ((got - ref).abs().max() / ref.abs().max()).item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("arm", choices=list(ARMS))
    p.add_argument("--m", type=int, required=True)
    p.add_argument("--n", type=int, required=True, help="GLOBAL N (n_local = N/world)")
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--ref_check", action="store_true")
    p.add_argument("--quack_configs", default=None,
                   help="quack sweep: 'tm,tn,cm,cn[,sw,static|clc];...' (default: QUACK_TILE + "
                        "256x256/256x512 c(2,1) x swizzle {1,8} x {static,clc})")
    args = p.parse_args()
    M, N, K = args.m, args.n, args.k
    global quack_configs
    quack_configs = parse_quack_configs(args.quack_configs)

    torchrun_init()
    rank, world = dist.get_rank(), dist.get_world_size()
    n_local = N // world
    assert N % world == 0 and M % world == 0 and n_local % 2 == 0
    for c in quack_configs:
        assert (M // world) % (c[0] * c[2]) == 0, f"shard_m must be a multiple of tile_M*cluster_M ({c})"

    rot = ROTATE_INPUTS.get(args.arm)
    need_plain = args.ref_check or rot is None
    kern = ARMS[args.arm](M, N, K, rank, world) if need_plain else None
    ref = _ref(M, N, K, rank, world) if need_plain else None

    checks = []
    if args.ref_check:
        scaled = args.arm == "quack"  # only the fused arm folds the rstd colvec
        checks.append(("max|err|/max|ref|", f"{ref_check(kern, M, N, K, rank, world, scaled):.4e}",
                       f"{ref_check(ref, M, N, K, rank, world, False):.4e}"))

    tf = lambda ms: 2 * M * n_local * K / (ms * 1e9)  # noqa: E731
    sweep_ms = []
    if rot is not None:
        from quack.autotuner import _gpu_warmup

        factory, capture, cfg_sweep = rot
        proto = "L2-rotate" if capture else "L2-rotate (eager-timed)"
        fn, arg_sets, _keep, sweep = _kernel_l2_rotate_inputs(
            factory, M, N, K, rank, world, configs=quack_configs if cfg_sweep else None)
        ref_fn, ref_arg_sets, _ref_keep, _ = _kernel_l2_rotate_inputs(_ref, M, N, K, rank, world)
        _gpu_warmup()
        kwarg_sets = [{}] * len(arg_sets)
        if sweep:  # quack configs: report each, rank the best
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
        if kern.sweep:
            for label, run in kern.sweep:
                dist.barrier()
                time.sleep(1.0)
                sweep_ms.append((label, do_bench_counts(run, calls_per_inv=kern.calls)))
            best_label, ms = min(sweep_ms, key=lambda r: r[1])
        else:
            dist.barrier()
            time.sleep(1.0)
            ms = do_bench_counts(kern.run, calls_per_inv=kern.calls)
        dist.barrier()
        time.sleep(1.0)
        ref_ms = do_bench_counts(ref.run)

    clean_print("=" * 80, print_once=True)
    clean_print(f"<llama AG swiglu {args.arm} | world={world} | {M}x{K}x{N} (N global) | {proto}>",
                print_once=True)
    for label, v_ms in sweep_ms:
        clean_print(f"  {args.arm + ' ' + label + ':':<24} {v_ms:.3f} ms | {tf(v_ms):.2f} TFLOp/s",
                    print_once=True)
    name = f"{args.arm} best ({best_label})" if sweep_ms else args.arm
    clean_print(f"  {name + ':':<24} {ms:.3f} ms | {tf(ms):.2f} TFLOp/s", print_once=True)
    clean_print(f"  {'NCCL+cuBLAS+epi:':<24} {ref_ms:.3f} ms | {tf(ref_ms):.2f} TFLOp/s",
                print_once=True)
    for label, kv, rv in checks:
        clean_print(f"  {args.arm + ' ' + label + ':':<24} {kv}", print_once=True)
        clean_print(f"  {'ref ' + label + ':':<24} {rv}", print_once=True)

    if kern is not None:
        kern.free()
    dist.barrier()
    torchrun_finalize()


if __name__ == "__main__":
    main()
