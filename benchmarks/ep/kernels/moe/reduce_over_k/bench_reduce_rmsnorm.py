"""The combine's gather-and-sum over K followed by the next layer's RMSNorm, on one GPU (the per-rank shape of the
W=8 forward is the same reduce): sonic's triton gather-and-sum + a torch.compile'd RMSNorm vs reduce_over_k's fused
forms. bench_reduce_over_k's shapes, data, checks and timing protocol.

  triton+torch      token_gather_and_sum_varlen_K_triton, then torch.compile(rmsnorm) over its bf16 output
  cute_stats        reduce_over_k on its fast 2 x 1024 tile with the sq-sum ColVecReduce epilogue (per-tile partials,
                    the rstd finalized by a 7-column torch sum): the UNNORMALIZED bf16 output + rstd per token, quack's
                    rms_partial -> next-GEMM-epilogue link (rstd commutes through the GEMM, the weight folds into it)

    PYTHONPATH=~/quack:~/sonic-moe-pr54 python bench_reduce_rmsnorm.py --model dsv3 --t-local 8192
"""
import os

os.environ.setdefault("TRITON_CACHE_AUTOTUNING", "1")

import argparse
import statistics
import sys

import torch
from sonicmoe.functional.reduction_over_k_gather import token_gather_and_sum_varlen_K_triton

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from bench_dispatch import MODEL_CONFIGS
from bench_reduce_over_k import bench
from reduce_over_k import ColVecReduce, reduce_epilogue, reduce_over_k

EPS = 1e-6


@reduce_epilogue(reduces={"sqsum": ColVecReduce("sqsum", scaled=True)})
def sqsum_epi(acc):
    return {"D": acc, "sqsum": (acc, acc)}


def rmsnorm(y, w):
    yf = y.float()
    return (yf * torch.rsqrt(yf.pow(2).mean(-1, keepdim=True) + EPS) * w).to(y.dtype)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=list(MODEL_CONFIGS), default="dsv3")
    p.add_argument("--t-local", type=int, default=8192)
    args = p.parse_args()
    H, K, T = MODEL_CONFIGS[args.model].H, MODEL_CONFIGS[args.model].topk, args.t_local
    dev = torch.device("cuda")
    torch.manual_seed(0)
    slots = torch.randn(T, K, H, dtype=torch.bfloat16, device=dev)
    scores = torch.rand(T, K, dtype=torch.float32, device=dev)
    w = torch.rand(H, dtype=torch.float32, device=dev) + 0.5
    perm = torch.arange(T * K, dtype=torch.int32, device=dev)
    offset = torch.arange(0, T * K + 1, K, dtype=torch.int32, device=dev)
    y = {k: torch.empty(T, H, dtype=torch.bfloat16, device=dev) for k in ("triton", "cute_stats")}
    out = {"triton+torch": None, "cute_stats": None}
    rmsnorm_c = torch.compile(rmsnorm)
    partials = torch.empty(1, T, -(-H // 1024), dtype=torch.float32, device=dev)

    def triton_torch():
        token_gather_and_sum_varlen_K_triton(slots.view(T * K, H), scores.view(-1), y["triton"], perm, offset, T, K, H, False)
        out["triton+torch"] = rmsnorm_c(y["triton"], w)

    def cute_stats():
        sums = reduce_over_k(slots, scores, y["cute_stats"], epilogue=sqsum_epi, epi_args={"sqsum": partials})
        out["cute_stats"] = torch.rsqrt(sums["sqsum"].view(T, 1) / H + EPS)

    fns = {"triton+torch": triton_torch, "cute_stats": cute_stats}
    for fn in fns.values():
        fn()
    torch.cuda.synchronize()
    yref = (slots.float() * scores[..., None]).sum(1)
    rstd = torch.rsqrt(yref.pow(2).mean(-1, keepdim=True) + EPS)
    ref = yref * rstd * w
    print(f"{'triton+torch':13s} normalized out max |err| vs fp32 ref {(out['triton+torch'].float() - ref).abs().max().item():.3e}")
    print(f"{'cute_stats':13s} y max |err| {(y['cute_stats'].float() - yref).abs().max().item():.3e}, "
          f"rstd max rel err {((out['cute_stats'] - rstd).abs() / rstd).max().item():.3e}")
    ms = bench(fns)
    print(f"{args.model} T_local={T} K={K} H={H}")
    for k, v in ms.items():
        print(f"  {k:13s} {statistics.median(v) * 1e3:7.1f} us [{min(v) * 1e3:.1f}-{max(v) * 1e3:.1f}]")


if __name__ == "__main__":
    main()
