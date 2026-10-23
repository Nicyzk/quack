"""The combine's gather-and-sum over K on one GPU: sonic's triton kernel as bench_forward calls it (identity perm,
K rows per token, fp32 scores) vs reduce_over_k.py. Shape from --model (bench_dispatch.MODEL_CONFIGS: H, top-k) and
--t-local; bf16 slots, random fp32 scores. Both arms checked against a fp32 torch reference, then timed round-major:
per block each arm warms ~200 ms, then a fixed call count is event-timed; median of REPS block medians with min-max.

    PYTHONPATH=~/quack:~/sonic-moe-pr54 python bench_reduce_over_k.py --model dsv3 --t-local 8192
"""
import os

os.environ.setdefault("TRITON_CACHE_AUTOTUNING", "1")  # sonic's triton autotune choices persist across launches

import argparse
import statistics
import sys
import time

import torch
from sonicmoe.functional.reduction_over_k_gather import token_gather_and_sum_varlen_K_triton

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from bench_dispatch import MODEL_CONFIGS, WARM_S, WINDOW_S
from quack.epilogue.library import residual_epi
from reduce_over_k import ColVecReduce, reduce_epilogue, reduce_over_k


@reduce_epilogue(reduces={"sqsum": ColVecReduce("sqsum", scaled=True)})
def residual_sqsum(acc, res):
    """The next layer's RMSNorm statistics fused onto the residual add: per-token sum of squares partials."""
    y = acc + res
    return {"D": y, "sqsum": (y, y)}

REPS = 5


def bench(fns):
    """{name: [block median ms, ...]}: each block runs every arm (order rotated per block), ~WARM_S of warm-up then
    ~WINDOW_S of calls timed as 10 event-bracketed chunks whose median is the block's."""
    counts = {}
    for k, fn in fns.items():
        fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            fn()
        torch.cuda.synchronize()
        s = (time.perf_counter() - t0) / 20
        counts[k] = (max(10, int(WARM_S / s)), max(20, int(WINDOW_S / s)))
    ms, names = {k: [] for k in fns}, list(fns)
    for rep in range(REPS):
        shift = rep % len(names)
        for k in names[shift:] + names[:shift]:
            fn, (warm, iters) = fns[k], counts[k]
            for _ in range(warm):
                fn()
            chunk = max(1, iters // 10)
            events = []
            for i in range(iters):
                if i % chunk == 0:
                    events.append(torch.cuda.Event(enable_timing=True))
                    events[-1].record()
                fn()
            events.append(torch.cuda.Event(enable_timing=True))
            events[-1].record()
            torch.cuda.synchronize()
            bounds = list(range(0, iters, chunk)) + [iters]
            ms[k].append(statistics.median(events[j].elapsed_time(events[j + 1]) / (bounds[j + 1] - bounds[j]) for j in range(len(bounds) - 1)))
    return ms


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
    perm = torch.arange(T * K, dtype=torch.int32, device=dev)
    offset = torch.arange(0, T * K + 1, K, dtype=torch.int32, device=dev)
    out = {k: torch.empty(T, H, dtype=torch.bfloat16, device=dev) for k in ("triton", "cute")}
    fns = {
        "triton": lambda: token_gather_and_sum_varlen_K_triton(slots.view(T * K, H), scores.view(-1), out["triton"], perm, offset, T, K, H, False),
        "cute": lambda: reduce_over_k(slots, scores, out["cute"]),
    }
    for fn in fns.values():
        fn()
    torch.cuda.synchronize()
    ref = (slots.float() * scores[..., None]).sum(1)
    for k, o in out.items():
        print(f"{k:7s} max |err| vs fp32 ref {(o.float() - ref).abs().max().item():.3e}")
    print(f"triton vs cute max |diff| {(out['triton'].float() - out['cute'].float()).abs().max().item():.3e}")
    # Epilogue checks (quack's residual_epi fn unchanged; a residual + sq-sum reduce): correctness only, not timed.
    res = torch.randn(T, H, dtype=torch.bfloat16, device=dev)
    fused = {k: torch.empty(T, H, dtype=torch.bfloat16, device=dev) for k in ("residual", "sqsum")}
    reduce_over_k(slots, scores, fused["residual"], epilogue=reduce_epilogue()(residual_epi.fn), epi_args={"res": res})
    partials = torch.empty(1, T, -(-H // 1024), dtype=torch.float32, device=dev)
    sums = reduce_over_k(slots, scores, fused["sqsum"], epilogue=residual_sqsum, epi_args={"res": res, "sqsum": partials})
    torch.cuda.synchronize()
    y = ref + res.float()
    for k, o in fused.items():
        print(f"cute+{k:8s} max |err| vs fp32 ref {(o.float() - y).abs().max().item():.3e}")
    print(f"sqsum rel err {((sums['sqsum'].view(T) - (y * y).sum(1)).abs() / (y * y).sum(1)).max().item():.3e}")
    ms = bench(fns)
    gb = (T * K * H * 2 + T * H * 2 + T * K * 4) / 1e9
    print(f"{args.model} T_local={T} K={K} H={H}  ({gb:.3f} GB moved)")
    for k, v in ms.items():
        med = statistics.median(v)
        print(f"  {k:7s} {med * 1e3:7.1f} us [{min(v) * 1e3:.1f}-{max(v) * 1e3:.1f}]  {gb / med * 1e3:7.0f} GB/s")


if __name__ == "__main__":
    main()
