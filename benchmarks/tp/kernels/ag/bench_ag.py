import argparse
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "misc"))
from utils import do_bench_all, torchrun_init, torchrun_finalize, clean_print


def run(
    M: int,
    N: int,
    local_rank: int,
    local_world_size: int,
):
    device = "cuda"

    input_ = torch.randn(M // local_world_size, N, dtype=torch.bfloat16, device=device)
    output = torch.zeros(M, N, dtype=torch.bfloat16, device=device)
    dist.barrier()

    ag_ms = do_bench_all(
        lambda: dist.all_gather_into_tensor(output, input_),
        warmup=5, rep=20,
    )

    n_bytes = M * N * 2  # bf16, output size
    alg_bw = n_bytes / (ag_ms * 1e-3) / 1e9
    bus_bw = alg_bw * (local_world_size - 1) / local_world_size
    def fmt(b): return f"{b/1e9:.2f} GB" if b >= 1e9 else f"{b/1e6:.1f} MB"
    xfer = n_bytes * (local_world_size - 1) / local_world_size
    clean_print(f"===============================================================================", print_once=True)
    clean_print(f"<BF16 AG | world_size={local_world_size} | {M}x{N} | data={fmt(n_bytes)} | xfer={fmt(xfer)}>", print_once=True)
    clean_print(f"NCCL: {ag_ms:.3f} ms | algBW={alg_bw:.1f} GB/s | busBW={bus_bw:.1f} GB/s ({100*bus_bw/900:.1f}% NVLink)")

    torch.cuda.synchronize()
    dist.barrier()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, default=8192)
    p.add_argument("--n", type=int, default=8192)
    args = p.parse_args()
    local_rank, local_world_size = torchrun_init()
    run(args.m, args.n, local_rank, local_world_size)
    torchrun_finalize()
