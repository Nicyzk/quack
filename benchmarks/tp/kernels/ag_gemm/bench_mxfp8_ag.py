"""MXFP8 AG+GEMM: quack (BlockScaledAllGatherRunner + blockscaled GEMM, the fp8 bytes
and the blocked e8m0 scales ride the same CE push / arrival flags) vs NCCL+cuBLAS
(NCCL all_gather of the fp8 shard and its blocked scales, then torch._scaled_mm) under
bench_baselines' L2-cold rotate protocol, plus the bound legs (solo _scaled_mm on a
pre-gathered A; CE push of the fp8 shard + SFA with no GEMM).

A/B: bf16 ~N(0, 1/sqrt(K)) quantized to mxfp8_e4m3 (sf_vec 32) per rank; D bf16. M is
the gathered dim (shard_m = M/world), --n is the rank-LOCAL N, K full. TFLOP/s counts
2*M*n_local*K. quack config sweep = --quack_configs over the blockscaled tiles
(tile_n <= 256, 64-multiples) x swizzle {1,8} x {static, clc}; the best is reported.

Usage:
    torchrun --nproc_per_node=2 bench_mxfp8_ag.py quack --m 8192 --n 3072 --k 4096 --ref_check
    torchrun --nproc_per_node=2 bench_mxfp8_ag.py legs  --m 8192 --n 3072 --k 4096
"""
import argparse
import os
import sys
from functools import partial

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "misc"))
from bench_rotate import do_bench_l2_rotate_all, l2_rotate_arg_sets  # noqa: E402
from utils import clean_print, parse_quack_configs, quack_config_label, torchrun_init, torchrun_finalize  # noqa: E402

SF_VEC = 32
FMT = "mxfp8_e4m3"
MX = dict(bs_format_a=FMT, bs_format_b=FMT)
MX_TILES = "256,256,2,1;256,128,2,1;256,192,2,1"  # blockscaled: tile_n <= 256, 64-multiple
quack_configs = parse_quack_configs(MX_TILES)


def quantize(x_bf16):
    """(fp8 qdata, 2-D e8m0 scale (rows, K/32), blocked (1, rm, rk, 32, 4, 4) scale)."""
    from quack.blockscaled.quantize import pack_scale_2d_to_blocked_contig, to_mx

    q, sc = to_mx(x_bf16, SF_VEC)
    sc = sc.contiguous()
    return q.contiguous(), sc, pack_scale_2d_to_blocked_contig(sc.view(1, *sc.shape))


def operands(M, N, K, rank, world):
    """Per-rank quantized A shard, B shard, plus the NCCL-gathered A for the legs/ref."""
    shard_m = M // world
    torch.manual_seed(1111 + rank)
    a_hp = torch.randn(shard_m, K, dtype=torch.bfloat16, device="cuda") / K**0.5
    b_hp = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / K**0.5
    a_q, a_sc, a_blk = quantize(a_hp)
    b_q, b_sc, b_blk = quantize(b_hp)
    a_q_full = torch.empty(M, K, dtype=torch.uint8, device="cuda")
    dist.all_gather_into_tensor(a_q_full, a_q.view(torch.uint8))
    a_sc_full = torch.empty(M, K // SF_VEC, dtype=a_sc.dtype, device="cuda")
    dist.all_gather_into_tensor(a_sc_full.view(torch.uint8), a_sc.view(torch.uint8))
    a_blk_full = torch.empty((1, M // 128, a_blk.shape[2], 32, 4, 4), dtype=a_blk.dtype, device="cuda")
    dist.all_gather_into_tensor(a_blk_full.view(torch.uint8).view(-1), a_blk.view(torch.uint8).view(-1))
    return dict(a_q=a_q, a_sc=a_sc, a_blk=a_blk, b_q=b_q, b_sc=b_sc, b_blk=b_blk,
                a_q_full=a_q_full.view(torch.float8_e4m3fn), a_sc_full=a_sc_full, a_blk_full=a_blk_full)


def scaled_mm(a_q, a_blk, b_q, b_blk, d):
    torch._scaled_mm(a_q, b_q.t(), scale_a=a_blk.view(torch.float8_e8m0fnu).view(-1, 4 * SF_VEC),
                     scale_b=b_blk.view(torch.float8_e8m0fnu).view(-1, 4 * SF_VEC),
                     out_dtype=torch.bfloat16, out=d)


def _runner(M, K, world, a_q):
    from quack.distributed import BlockScaledAllGatherRunner

    runner = BlockScaledAllGatherRunner(M // world, K, FMT)
    for _ in range(len(runner.bufs)):  # prefill both slots: staging out of the timed loop
        runner.next_local_slot().copy_(a_q.view(torch.uint8))
        runner.last_parity ^= 1
    torch.cuda.synchronize()
    return runner


def arm_quack(ops, M, N, K, rank, world):
    from quack.blockscaled.operand import BlockScaledOperand
    from quack.gemm import gemm as quack_gemm

    runner = _runner(M, K, world, ops["a_q"])
    D = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")

    def fn(b_q, b_blk, d, cfg=None):
        tm, tn, cm, cn, sw, clc = cfg or quack_configs[0]
        # zero-copy: the shard already sits in the slot the next gather() uses
        op = BlockScaledOperand(runner.next_local_slot().view(torch.float8_e4m3fn), ops["a_blk"], FMT)
        with runner.gather(op) as (a_op, ag_args):
            quack_gemm(a_op.qdata, b_q, d, None, None, tm, tn, cm, cn, is_dynamic_persistent=clc,
                       max_swizzle_size=sw, SFA=a_op.scale, SFB=b_blk, ag_args=ag_args, **MX)

    sweep = [(quack_config_label(c), partial(fn, cfg=c)) for c in quack_configs]
    return fn, l2_rotate_arg_sets(ops["b_q"], ops["b_blk"], D), sweep, (runner,)


def arm_nccl(ops, M, N, K, rank, world):
    a_full = torch.empty(M, K, dtype=torch.uint8, device="cuda")
    sfa = torch.empty_like(ops["a_blk_full"])
    D = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")

    def fn(b_q, b_blk, d):
        dist.all_gather_into_tensor(a_full, ops["a_q"].view(torch.uint8))
        dist.all_gather_into_tensor(sfa.view(torch.uint8).view(-1), ops["a_blk"].view(torch.uint8).view(-1))
        scaled_mm(a_full.view(torch.float8_e4m3fn), sfa, b_q, b_blk, d)

    return fn, l2_rotate_arg_sets(ops["b_q"], ops["b_blk"], D), None, ()


def arm_legs(ops, M, N, K, rank, world):
    from quack.blockscaled.operand import BlockScaledOperand

    runner = _runner(M, K, world, ops["a_q"])
    D = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")

    def gemm_leg(b_q, b_blk, d):
        scaled_mm(ops["a_q_full"], ops["a_blk_full"], b_q, b_blk, d)

    def push_leg(b_q, b_blk, d):
        op = BlockScaledOperand(runner.next_local_slot().view(torch.float8_e4m3fn), ops["a_blk"], FMT)
        with runner.gather(op):
            pass

    sweep = [("cuBLAS GEMM", gemm_leg), ("CE AG", push_leg)]
    return gemm_leg, l2_rotate_arg_sets(ops["b_q"], ops["b_blk"], D), sweep, (runner,)


ARMS = {"quack": arm_quack, "nccl": arm_nccl, "legs": arm_legs}


def ref_err(ops, d):
    """max|D - fp32 dequant reference| / max|ref| (the blockscaled tests' criterion)."""
    a = ops["a_q_full"].float() * ops["a_sc_full"].float().repeat_interleave(SF_VEC, dim=-1)
    b = ops["b_q"].float() * ops["b_sc"].float().repeat_interleave(SF_VEC, dim=-1)
    ref = a @ b.T
    return ((d.float() - ref).abs().max() / ref.abs().max()).item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("arm", choices=list(ARMS))
    p.add_argument("--m", type=int, required=True)
    p.add_argument("--n", type=int, required=True, help="rank-LOCAL N")
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--ref_check", action="store_true")
    p.add_argument("--quack_configs", default=None,
                   help="'tm,tn,cm,cn[,sw,static|clc];...' (default: blockscaled tiles x sw {1,8} x {static,clc})")
    args = p.parse_args()
    M, N, K = args.m, args.n, args.k
    global quack_configs
    quack_configs = parse_quack_configs(args.quack_configs or MX_TILES)

    torchrun_init()
    rank, world = dist.get_rank(), dist.get_world_size()
    assert M % (128 * world) == 0 and K % SF_VEC == 0
    from quack.autotuner import _gpu_warmup

    ops = operands(M, N, K, rank, world)
    fn, arg_sets, sweep, _keep = ARMS[args.arm](ops, M, N, K, rank, world)
    ref_fn, ref_sets, _, _ref_keep = arm_nccl(ops, M, N, K, rank, world)
    checks = []
    if args.ref_check:  # every arm writes set 0's cloned D (arg_sets[0][2])
        fn(*arg_sets[0])
        ref_fn(*ref_sets[0])
        torch.cuda.synchronize()
        dist.barrier()
        checks = [(args.arm, ref_err(ops, arg_sets[0][2])), ("NCCL+cuBLAS", ref_err(ops, ref_sets[0][2]))]
    _gpu_warmup()
    kw = [{}] * len(arg_sets)
    sweep_ms = []
    if sweep:
        for label, fn_v in sweep:
            dist.barrier()
            sweep_ms.append((label, do_bench_l2_rotate_all(fn_v, arg_sets, kw, {})))
        pick = max if args.arm == "legs" else min  # legs: the bound is the slower leg
        best_label, ms = pick(sweep_ms, key=lambda r: r[1])
    else:
        dist.barrier()
        ms = do_bench_l2_rotate_all(fn, arg_sets, kw, {})
    dist.barrier()
    ref_ms = do_bench_l2_rotate_all(ref_fn, ref_sets, [{}] * len(ref_sets), {})

    tf = lambda v: 2 * M * N * K / (v * 1e9)  # noqa: E731
    clean_print("=" * 80, print_once=True)
    clean_print(f"<MXFP8 AG+GEMM {args.arm} | world={world} | {M}x{K}x{N} (N local) | L2-rotate>",
                print_once=True)
    for label, v in sweep_ms:
        clean_print(f"  {args.arm + ' ' + label + ':':<22} {v:.3f} ms | {tf(v):.2f} TFLOp/s", print_once=True)
    if sweep_ms:
        name = f"bound ({best_label})" if args.arm == "legs" else f"{args.arm} best ({best_label})"
    else:
        name = args.arm
    clean_print(f"  {name + ':':<14} {ms:.3f} ms | {tf(ms):.2f} TFLOp/s", print_once=True)
    clean_print(f"  {'NCCL+cuBLAS:':<14} {ref_ms:.3f} ms | {tf(ref_ms):.2f} TFLOp/s", print_once=True)
    for name, v in checks:
        clean_print(f"  {name + ' rel err:':<22} {v:.4e} (max|err|/max|ref|, fp32 dequant ref)", print_once=True)
    dist.barrier()
    torchrun_finalize()


if __name__ == "__main__":
    main()
