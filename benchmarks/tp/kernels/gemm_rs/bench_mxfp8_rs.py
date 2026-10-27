"""MXFP8 GEMM+RS: quack (blockscaled GEMM with the fused reduce_scatter epilogue,
epi_reduce_mode) vs cuBLAS+NCCL (torch._scaled_mm partial in bf16, NCCL reduce_scatter)
under bench_baselines' L2-cold rotate protocol, plus the bound legs (solo _scaled_mm;
NCCL reduce_scatter of the bf16 (M, N) partial alone).

A/B: each rank's k-shard, bf16 ~N(0, 1/sqrt(K)) quantized to mxfp8_e4m3 (sf_vec 32);
D bf16 slab (M/world, N). K is global (k_local = K/world). TFLOP/s counts
2*M*N*k_local. quack sweep = --quack_configs over the blockscaled tiles x swizzle
{1,8}, static only (epi_reduce_mode requires static persistence); best reported.

Usage:
    torchrun --nproc_per_node=2 bench_mxfp8_rs.py quack --m 8192 --n 4096 --k 4096 --ref_check
    torchrun --nproc_per_node=2 bench_mxfp8_rs.py legs  --m 8192 --n 4096 --k 4096
"""
import argparse
import os
import sys
from functools import partial

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "misc"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_rotate import do_bench_l2_rotate_all, l2_rotate_arg_sets, l2_rotate_count  # noqa: E402
from utils import clean_print, parse_quack_configs, quack_config_label, torchrun_init, torchrun_finalize  # noqa: E402

SF_VEC = 32
FMT = "mxfp8_e4m3"
MX = dict(bs_format_a=FMT, bs_format_b=FMT)
MX_TILES = "256,256,2,1;256,128,2,1"  # blockscaled tiles that divide the RS sites' N (no partial N tiles under epi_reduce)
quack_configs = parse_quack_configs(MX_TILES, clc_ok=False)


def quantize(x_bf16):
    """(fp8 qdata, 2-D e8m0 scale, blocked (1, rm, rk, 32, 4, 4) scale)."""
    from quack.blockscaled.quantize import pack_scale_2d_to_blocked_contig, to_mx

    q, sc = to_mx(x_bf16, SF_VEC)
    sc = sc.contiguous()
    return q.contiguous(), sc, pack_scale_2d_to_blocked_contig(sc.view(1, *sc.shape))


def operands(M, N, K, rank, world):
    k_local = K // world
    torch.manual_seed(1111 + rank)
    a_hp = torch.randn(M, k_local, dtype=torch.bfloat16, device="cuda") / K**0.5
    b_hp = torch.randn(N, k_local, dtype=torch.bfloat16, device="cuda") / K**0.5
    a_q, a_sc, a_blk = quantize(a_hp)
    b_q, b_sc, b_blk = quantize(b_hp)
    return dict(a_q=a_q, a_sc=a_sc, a_blk=a_blk, b_q=b_q, b_sc=b_sc, b_blk=b_blk, a_hp=a_hp, b_hp=b_hp)


def scaled_mm(a_q, a_blk, b_q, b_blk, d):
    torch._scaled_mm(a_q, b_q.t(), scale_a=a_blk.view(torch.float8_e8m0fnu).view(-1, 4 * SF_VEC),
                     scale_b=b_blk.view(torch.float8_e8m0fnu).view(-1, 4 * SF_VEC),
                     out_dtype=torch.bfloat16, out=d)


def arm_quack(ops, M, N, K, rank, world):
    """Every rotate set owns cloned (a_q, a_blk, b_q, b_blk) and its own RS workspace
    per candidate tile (the reducer's flags/workspace depend on the tile)."""
    from quack.distributed.gemm_epi_reduce import make_epi_reduce_args
    from quack.gemm import gemm as quack_gemm

    tiles = sorted({c[:4] for c in quack_configs})
    n_sets = l2_rotate_count(ops["a_q"], ops["b_q"])
    sets = []
    for _ in range(n_sets):
        eras = {t: make_epi_reduce_args("reduce_scatter", torch.bfloat16, M, N, 1, t[0], t[1], t[2], world)
                for t in tiles}
        sets.append((ops["a_q"].clone(), ops["a_blk"].clone(), ops["b_q"].clone(), ops["b_blk"].clone(), eras))
    last = {}

    def fn(a_q, a_blk, b_q, b_blk, eras, cfg=None):
        tm, tn, cm, cn, sw, clc = cfg or quack_configs[0]
        d_arg, era = eras[(tm, tn, cm, cn)]
        quack_gemm(a_q.unsqueeze(0), b_q.unsqueeze(0), d_arg, None, None, tm, tn, cm, cn,
                   is_dynamic_persistent=clc, max_swizzle_size=sw, SFA=a_blk, SFB=b_blk,
                   epi_reduce_mode="reduce_scatter", epi_reduce_args=era, **MX)
        last["d"] = d_arg

    sweep = [(quack_config_label(c), partial(fn, cfg=c)) for c in quack_configs]
    return fn, [tuple(s) for s in sets], sweep, (sets,), lambda: last["d"][0]


def arm_nccl(ops, M, N, K, rank, world):
    D_full = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    D_rs = torch.empty(M // world, N, dtype=torch.bfloat16, device="cuda")

    def fn(a_q, a_blk, b_q, b_blk):
        scaled_mm(a_q, a_blk, b_q, b_blk, D_full)
        dist.reduce_scatter_tensor(D_rs, D_full)

    sets = l2_rotate_arg_sets(ops["a_q"], ops["a_blk"], ops["b_q"], ops["b_blk"])
    return fn, sets, None, (D_full, D_rs), lambda: D_rs


def arm_legs(ops, M, N, K, rank, world):
    D_full = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    D_rs = torch.empty(M // world, N, dtype=torch.bfloat16, device="cuda")

    def gemm_leg(a_q, a_blk, b_q, b_blk):
        scaled_mm(a_q, a_blk, b_q, b_blk, D_full)

    def rs_leg(a_q, a_blk, b_q, b_blk):
        dist.reduce_scatter_tensor(D_rs, D_full)

    sets = l2_rotate_arg_sets(ops["a_q"], ops["a_blk"], ops["b_q"], ops["b_blk"])
    return gemm_leg, sets, [("cuBLAS GEMM", gemm_leg), ("NCCL RS", rs_leg)], (D_full, D_rs), lambda: D_rs


def arm_te(ops, M, N, K, rank, world):
    """Transformer Engine Userbuffers GEMM+RS on MXFP8 operands (misc/te_ub.py): te.Linear row-parallel with
    ub_overlap_rs under the MXFP8 recipe; A pre-quantized by TE's MXFP8 quantizer per rotate set (as the other arms
    take quantized A), the weight quantized once by the module (is_first_microbatch) and cached."""
    import transformer_engine.pytorch as te
    import transformer_engine_torch as tex
    from transformer_engine.common.recipe import MXFP8BlockScaling
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

    from te_ub import init_ub, rs_linear

    init_ub(M, N, world)
    lin = rs_linear(K, N, world, ops["b_hp"])
    recipe = MXFP8BlockScaling()
    quant = MXFP8Quantizer(tex.DType.kFloat8E4M3, rowwise=True, columnwise=False)
    sets = [(quant(ops["a_hp"].clone()),) for _ in range(l2_rotate_count(ops["a_q"], ops["b_q"]))]
    last = {}

    def fn(a_te, first=False):
        with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
            last["d"] = lin(a_te, is_first_microbatch=first)

    with torch.no_grad():
        fn(sets[0][0], first=True)  # the weight's one-time quantization
    return fn, sets, None, (lin, sets), lambda: last["d"]


ARMS = {"quack": arm_quack, "nccl": arm_nccl, "te": arm_te, "legs": arm_legs}


def ref_err_hp(ops, d_slab, M, rank, world):
    """max|slab - fp32 full-K reference from the bf16 operands| / max|ref| (for arms that quantize with their own
    quantizer: the error is then the fp8 level for every arm, not the quantizer difference)."""
    a_sh = [torch.empty_like(ops["a_hp"]) for _ in range(world)]
    b_sh = [torch.empty_like(ops["b_hp"]) for _ in range(world)]
    dist.all_gather(a_sh, ops["a_hp"])
    dist.all_gather(b_sh, ops["b_hp"])
    ref = torch.cat(a_sh, 1).float() @ torch.cat(b_sh, 1).float().T
    m_per = M // world
    ref = ref[rank * m_per:(rank + 1) * m_per]
    return ((d_slab.float() - ref).abs().max() / ref.abs().max()).item()


def ref_err(ops, d_slab, M, rank, world):
    """max|slab - fp32 full-K dequant reference slab| / max|ref|."""
    a = ops["a_q"].float() * ops["a_sc"].float().repeat_interleave(SF_VEC, dim=-1)
    b = ops["b_q"].float() * ops["b_sc"].float().repeat_interleave(SF_VEC, dim=-1)
    a_sh = [torch.empty_like(a) for _ in range(world)]
    b_sh = [torch.empty_like(b) for _ in range(world)]
    dist.all_gather(a_sh, a)
    dist.all_gather(b_sh, b)
    ref = torch.cat(a_sh, 1) @ torch.cat(b_sh, 1).T
    m_per = M // world
    ref = ref[rank * m_per:(rank + 1) * m_per]
    return ((d_slab.float() - ref).abs().max() / ref.abs().max()).item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("arm", choices=list(ARMS))
    p.add_argument("--m", type=int, required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--k", type=int, required=True, help="global K (sharded K/world per rank)")
    p.add_argument("--ref_check", action="store_true")
    p.add_argument("--quack_configs", default=None,
                   help="'tm,tn,cm,cn[,sw,static];...' (default: blockscaled tiles x sw {1,8}, static)")
    p.add_argument("--te-rs-sms", type=int, default=16, help="te: the RS pipeline's comm SMs (TE's default 16; one value per process)")
    args = p.parse_args()
    import te_ub

    te_ub.RS_SMS = args.te_rs_sms
    M, N, K = args.m, args.n, args.k
    global quack_configs
    quack_configs = parse_quack_configs(args.quack_configs or MX_TILES, clc_ok=False)

    torchrun_init()
    rank, world = dist.get_rank(), dist.get_world_size()
    assert M % (128 * world) == 0 and K % (world * SF_VEC) == 0
    from quack.autotuner import _gpu_warmup

    ops = operands(M, N, K, rank, world)
    fn, arg_sets, sweep, _keep, read = ARMS[args.arm](ops, M, N, K, rank, world)
    ref_fn, ref_sets, _, _ref_keep, ref_read = arm_nccl(ops, M, N, K, rank, world)
    checks = []
    if args.ref_check:
        fn(*arg_sets[0])
        ref_fn(*ref_sets[0])
        torch.cuda.synchronize()
        dist.barrier()
        err = ref_err_hp if args.arm == "te" else ref_err  # te quantizes with TE's quantizer: check against the bf16 operands
        checks = [(args.arm, err(ops, read(), M, rank, world)),
                  ("cuBLAS+NCCL", ref_err(ops, ref_read(), M, rank, world))]
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

    tf = lambda v: 2 * M * N * (K // world) / (v * 1e9)  # noqa: E731
    clean_print("=" * 80, print_once=True)
    clean_print(f"<MXFP8 GEMM+RS {args.arm} | world={world} | {M}x{K}x{N} | L2-rotate>", print_once=True)
    for label, v in sweep_ms:
        clean_print(f"  {args.arm + ' ' + label + ':':<22} {v:.3f} ms | {tf(v):.2f} TFLOp/s", print_once=True)
    if sweep_ms:
        name = f"bound ({best_label})" if args.arm == "legs" else f"{args.arm} best ({best_label})"
    else:
        name = args.arm
    clean_print(f"  {name + ':':<14} {ms:.3f} ms | {tf(ms):.2f} TFLOp/s", print_once=True)
    clean_print(f"  {'cuBLAS+NCCL:':<14} {ref_ms:.3f} ms | {tf(ref_ms):.2f} TFLOp/s", print_once=True)
    for name, v in checks:
        ref_kind = "fp32 ref from the bf16 operands (TE quantizes them itself)" if name == "te" else "fp32 dequant ref"
        clean_print(f"  {name + ' rel err:':<22} {v:.4e} (max|err|/max|ref|, {ref_kind})", print_once=True)
    dist.barrier()
    torchrun_finalize()


if __name__ == "__main__":
    main()
