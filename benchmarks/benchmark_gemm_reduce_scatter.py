# Copyright (c) 2026, QuACK team.
"""GEMM + reduce-scatter benchmark: the ReduceScatter operator (quack/distributed/reduce_scatter.py) vs the
cuBLAS + NCCL chain, on the paper's row-parallel sites. K is sharded K/world_size per rank; every rank ends
with its M/world_size slab of the full-K product.

Methods:
  quack_rs     gemm() with dist_op=ReduceScatter: the GEMM's epilogue warps store raw partials, the comm warps
               reduce them with multimem loads and commit the slab; sweep of the paper's tiles (256x256 and
               256x512, cluster (2,1), static schedule) x swizzle 1 / 8, the best reported with its config
  quack_rs_rms the same with the residual and the RMSNorm statistic fused (gemm_rms's epilogue,
               sq_reduce_mod, in the comm warps after the reduce): D = (slab + residual) * weight and the
               per-tile squared-sum partials, rstd finalized by rms_final_reduce; the same sweep
  gemm_only    the cuBLAS GEMM alone   } the slower one is the perfect-overlap bound
  nccl_rs      the NCCL reduce-scatter alone }
  nccl_seq_rms nccl_seq, then the residual add and RMSNorm in torch (rmsnorm_ref)
  nccl_seq     torch matmul of the partial, then dist.reduce_scatter_single (the reference, timed last as
               the contention canary)

--dtype mxfp8_e4m3 | nvfp4 (SM100): the quack methods take block-scaled A and B (bf16 partials on the wire,
nvfp4's per-tensor scales as one alpha after the reduce); the NCCL methods stay bfloat16 on the dequantized
values, which is the reference the quack methods are checked against.

Protocol = quack's autotuner methodology made collective (quack.bench.bench_utils.bench_l2_rotate_dist):
cloned operand sets that turn L2 over, one captured 200-call window warmed by replaying itself, cross-rank
max, --rounds rounds. TFLOPS on the per-rank FLOPs 2 M N K / world_size.

  torchrun --nproc_per_node=8 benchmarks/benchmark_gemm_reduce_scatter.py --m 16384
"""

import argparse
import math
import os

import torch
import torch.distributed as dist

from quack.bench.bench_utils import bench_l2_rotate_dist, l2_rotate_arg_sets_dist
from quack.blockscaled.operand import BlockScaledOperand
from quack.blockscaled.quantize import nvfp4_per_tensor_scale
from quack.distributed.reduce_scatter import ReduceScatterRunner
from quack.cute_dsl_utils import get_device_capacity
from quack.epilogue.library import sq_reduce_mod
from quack.gemm import gemm as quack_gemm
from quack.gemm_config import GemmConfig, blockscaled_config_ok
from quack.rms_final_reduce import rms_final_reduce
from quack.rmsnorm import rmsnorm_ref

DEFAULT_SHAPES = [
    (4096, 4096),  # Llama 8B out
    (4096, 14336),  # Llama 8B down
    (8192, 8192),  # Llama 70B out
    (8192, 28672),  # Llama 70B down
    (16384, 16384),  # Llama 405B out
    (16384, 53248),  # Llama 405B down
]  # (N, K) global
TILES = ((256, 256, 2), (256, 512, 2))  # (tile_M, tile_N, cluster_M): the paper's two configs
SWIZZLES = (1, 8)


def _print0(rank, *args):
    if rank == 0:
        print(*args, flush=True)


def _check(rank, name, d, ref, tol) -> bool:
    err = (d.float() - ref).abs().max().item()
    flag = torch.tensor([1.0 if err > tol else 0.0], device=d.device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    if flag.item() > 0:
        _print0(
            rank,
            f"  [{name}] SKIP: correctness check failed (err={err:.3e} "
            f"tol={tol:.3e}) — not timing a wrong result",
        )
        return False
    _print0(rank, f"  [{name}] correctness OK (err={err:.2e})")
    return True


def bench_shape(m, n, k, bs_format, rank, world_size, device, args):
    dtype = torch.bfloat16
    k_local = k // world_size
    torch.manual_seed(1000 + rank)
    a = torch.randn(m, k_local, dtype=dtype, device=device) / math.sqrt(k)
    b = torch.randn(n, k_local, dtype=dtype, device=device) / math.sqrt(k)
    slab = m // world_size
    flops = 2 * m * n * k_local

    # block-scaled operands for the quack methods; the references run on the same values, dequantized
    quack_in, bs_kw, alpha = (a, b), {}, 1.0
    blockscaled = bs_format is not None
    capacity = get_device_capacity(device)[0]
    if blockscaled:
        pts_a = pts_b = None
        if bs_format == "nvfp4":  # per-tensor scales, shared by every rank: one alpha after the reduce
            amax = torch.stack([a.abs().amax(), b.abs().amax()])
            dist.all_reduce(amax, op=dist.ReduceOp.MAX)
            pts_a, pts_b = nvfp4_per_tensor_scale(amax[0]), nvfp4_per_tensor_scale(amax[1])
            alpha = float(pts_a * pts_b)
        a_op = BlockScaledOperand.quantize(a, bs_format, per_tensor_scale=pts_a)
        b_op = BlockScaledOperand.quantize(b, bs_format, per_tensor_scale=pts_b)
        quack_in = (a_op.qdata, b_op.qdata, a_op.scale, b_op.scale)
        bs_kw = dict(bs_format_a=bs_format, bs_format_b=bs_format)
        a, b = (op.dequantize(torch.float32).to(dtype) for op in (a_op, b_op))

    def rounds_of(fn, sets):
        return [bench_l2_rotate_dist(fn, sets) for _ in range(args.rounds)]  # ms per call

    # fp32 reference for loose correctness checks: this rank's slab of the full-K product
    ref = torch.empty(slab, n, dtype=torch.float32, device=device)
    dist.reduce_scatter_single(ref, a.float() @ b.float().T)
    tol = ref.abs().max().item() * 5e-2 + 1e-2
    _print0(rank, f"M={m} N={n} K={k}")

    # the fused chain's operands: the slab's residual and the norm weight, rank-invariant
    torch.manual_seed(1234)
    c = torch.randn(slab, n, dtype=dtype, device=device)
    weight = torch.randn(n, dtype=torch.float32, device=device)
    rstd = torch.empty(slab, dtype=torch.float32, device=device)
    rms_mod = sq_reduce_mod(has_c=True, has_rowvec=True, has_aux=False, has_alpha=alpha != 1.0)
    rms_args = {"mRowVecBroadcast": weight} | ({"alpha": alpha} if alpha != 1.0 else {})
    ref_rms = rmsnorm_ref(ref, weight, residual=c)[0]
    tol_rms = ref_rms.abs().max().item() * 5e-2 + 1e-2

    results = {}
    best = {}  # per quack method: every (tile, swizzle) config, the best kept with its label
    for tile_m, tile_n, cluster_m in TILES:
        if n % tile_n or m % (world_size * tile_m):
            continue
        config = GemmConfig(
            tile_m=tile_m, tile_n=tile_n, cluster_m=cluster_m, cluster_n=1, device_capacity=capacity
        )
        if blockscaled and not blockscaled_config_ok(config):
            continue
        rs = ReduceScatterRunner(m, n, dtype, tile_m // cluster_m, tile_n)  # flags are per CTA tile
        partials = torch.zeros(slab, n // tile_n, dtype=torch.float32, device=device)
        sets = l2_rotate_arg_sets_dist(*quack_in)
        for sw in SWIZZLES:

            def run_quack_rs(x, y, sfa=None, sfb=None, sw=sw):
                quack_gemm(
                    x,
                    y,
                    rs.d,
                    None,
                    None,
                    tile_m,
                    tile_n,
                    cluster_m,
                    1,
                    max_swizzle_size=sw,
                    alpha=alpha,
                    dist_op=rs.op,
                    dist_op_args=rs.prepare(),
                    SFA=sfa,
                    SFB=sfb,
                    **bs_kw,
                )

            def run_quack_rs_rms(x, y, sfa=None, sfb=None, sw=sw):
                rms_mod.gemm(
                    x,
                    y,
                    rs.d,
                    c,
                    epi_args={**rms_args, "mColVecReduce": partials},
                    tile_M=tile_m,
                    tile_N=tile_n,
                    cluster_M=cluster_m,
                    cluster_N=1,
                    max_swizzle_size=sw,
                    dist_op=rs.op,
                    dist_op_args=rs.prepare(),
                    SFA=sfa,
                    SFB=sfb,
                    **bs_kw,
                )
                rstd.copy_(rms_final_reduce(partials, scale=1.0 / n))

            cfg = f"{tile_m}x{tile_n} c({cluster_m},1) sw{sw}"
            for name, fn, read, method_ref, method_tol in (
                ("quack_rs", run_quack_rs, lambda: rs.d, ref, tol),
                ("quack_rs_rms", run_quack_rs_rms, lambda: rs.d.float() * rstd[:, None], ref_rms, tol_rms),
            ):
                fn(*quack_in)
                torch.cuda.synchronize(device)
                if not _check(rank, f"{name} {cfg}", read(), method_ref, method_tol):
                    continue
                ms = min(rounds_of(fn, sets))
                if name not in best or ms < best[name][0]:
                    best[name] = (ms, cfg)
    results.update(best)
    partial = a @ b.T
    out = torch.empty(slab, n, dtype=dtype, device=device)
    sets = l2_rotate_arg_sets_dist(a, b)

    def run_gemm_only(x, y):
        torch.matmul(x, y.T, out=partial)

    def run_nccl_rs(p):
        dist.reduce_scatter_single(out, p)

    def run_nccl_seq(x, y):
        dist.reduce_scatter_single(out, x @ y.T)

    def run_nccl_seq_rms(x, y):
        dist.reduce_scatter_single(out, x @ y.T)
        rmsnorm_ref(out, weight, residual=c)

    results["gemm_only"] = (min(rounds_of(run_gemm_only, sets)), "")
    results["nccl_rs"] = (min(rounds_of(run_nccl_rs, l2_rotate_arg_sets_dist(partial))), "")
    results["nccl_seq_rms"] = (min(rounds_of(run_nccl_seq_rms, sets)), "")
    results["nccl_seq"] = (min(rounds_of(run_nccl_seq, sets)), "")  # the reference, last
    bound = max(results["gemm_only"][0], results["nccl_rs"][0])
    for name, (ms, cfg) in results.items():
        _print0(rank, f"  {name:<12} {ms:8.3f} ms  {flops / (ms * 1e-3) / 1e12:6.0f} TFLOPS  {cfg}")
    vs = f"  quack_rs vs bound {100 * (best['quack_rs'][0] / bound - 1):+.0f}%" if "quack_rs" in best else ""
    _print0(rank, f"  bound        {bound:8.3f} ms{vs}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shapes",
        type=str,
        default=";".join(",".join(map(str, s)) for s in DEFAULT_SHAPES),
        help="N,K;... (global)",
    )
    parser.add_argument("--m", type=int, default=16384)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "mxfp8_e4m3", "nvfp4"])
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank % torch.cuda.device_count())
    device = torch.device("cuda", rank % torch.cuda.device_count())
    dist.init_process_group("nccl", device_id=device)
    bs_format = None if args.dtype == "bfloat16" else args.dtype
    if rank == 0:
        print(f"GEMM+RS world_size={world_size} {args.dtype}, ms per call (best round)")
    for shape in args.shapes.split(";"):
        n, k = map(int, shape.split(","))
        bench_shape(args.m, n, k, bs_format, rank, world_size, device, args)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
