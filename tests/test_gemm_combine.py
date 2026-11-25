# Copyright (C) 2026, Tri Dao.
"""Multi-rank correctness tests for CombineRunner (quack/distributed/combine.py): MoE combine fused into the
grouped FC2 GEMM. Each expert-grouped row's result is drained from the epilogue straight into its origin rank's
slot for that (token, k) pair; after a barrier `reduce_over_k` sums every token's K slots with the score weights,
the residual and the RMSNorm statistic (the gemm_rms shape). Tokens on either ragged dim: "n" = swap-AB (tokens
are D columns; SM100+), "m" = varlen_m (tokens are D rows). The host refusals (an operator built for the other
ragged dim, a dense call, the GEMM's own dynamic-scheduler knob) are in-process tests beside the multi-rank one.

Runs under torchrun (one rank per GPU); the pytest entry point spawns torchrun as a subprocess on the free GPUs.

    torchrun --nproc_per_node=2 tests/test_gemm_combine.py
    pytest tests/test_gemm_combine.py -x
"""

import os
import subprocess
import sys

import pytest
import torch


def routing(topk_g, rank, e_local):
    """Rank `rank`'s tables from the gathered top-k (W, T_local, K) of global expert ids: cu (E_local + 1),
    s_scatter (row -> global slot), s_reverse (slot -> row). A row's token is its slot // K."""
    f = topk_g.reshape(-1)
    mine = (f // e_local == rank).nonzero().squeeze(1)
    e = f[mine] - rank * e_local
    s_scatter = mine[torch.argsort(e, stable=True)]
    counts = torch.bincount(e, minlength=e_local)
    cu = torch.cat([torch.zeros(1, dtype=torch.int64, device=f.device), counts.cumsum(0)])
    s_reverse = torch.zeros(f.numel(), dtype=torch.int64, device=f.device)
    s_reverse[s_scatter] = torch.arange(s_scatter.numel(), device=f.device)
    i32 = lambda t: t.to(torch.int32)
    return i32(cu), i32(s_scatter), i32(s_reverse)


def _run_rank():
    import torch.distributed as dist

    from quack.cute_dsl_utils import get_device_capacity
    from quack.distributed.combine import CombineRunner
    from quack.distributed.reduce_over_k import reduce_over_k
    from quack.distributed.reduce_over_k_epilogue import rms_partial
    from quack.gemm import gemm

    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    device = torch.device("cuda", rank % torch.cuda.device_count())
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")
    group = dist.group.WORLD
    sm100 = get_device_capacity(device)[0] >= 10
    failures = []
    dtype = torch.bfloat16

    H, I = 1024, 512
    e_local, topk, t_local = 4, 2, 512
    # (ragged dim, tile_M, tile_N, cluster_M): swap-AB puts the weights on M (2-CTA) and the tokens on N (256 and
    # the 16-granule odd even-split, 192); varlen_m puts the tokens on M at the per-CTA tile
    tiles = ([("n", 256, 256, 2), ("n", 256, 192, 2)] if sm100 else []) + [("m", 128, 256, 1)]

    torch.manual_seed(0)  # the same routing and weights on every rank
    topk_g = (
        torch.stack(
            [torch.randperm(world * e_local, device=device)[:topk] for _ in range(world * t_local)]
        )
        .view(world, t_local, topk)
        .to(torch.int32)
    )
    w2 = torch.randn(e_local, H, I, device=device, dtype=dtype) * 0.05
    cu, s_scatter, s_reverse = routing(topk_g, rank, e_local)
    total = int(cu[-1])
    torch.manual_seed(100 + rank)
    act = torch.randn(total, I, device=device, dtype=dtype)  # this rank's expert-grouped FC1 rows
    scores = torch.rand(t_local, topk, device=device, dtype=torch.float32)
    res = torch.randn(t_local, H, device=device, dtype=dtype)

    # reference: every rank's y rows land in a global slot table; this rank's tokens read their K slots
    y_rows = torch.cat(
        [act[int(cu[e]) : int(cu[e + 1])].float() @ w2[e].float().T for e in range(e_local)]
    )
    n_rows = torch.tensor([total], device=device)
    counts = [torch.zeros_like(n_rows) for _ in range(world)]
    dist.all_gather(counts, n_rows)
    cap = max(int(c) for c in counts)
    pad = lambda t: torch.cat([t, t.new_zeros(cap - t.shape[0], *t.shape[1:])])
    rows_all = [torch.empty(cap, H, device=device, dtype=torch.float32) for _ in range(world)]
    slots_all = [torch.empty(cap, device=device, dtype=torch.int32) for _ in range(world)]
    dist.all_gather(rows_all, pad(y_rows))
    dist.all_gather(slots_all, pad(s_scatter))
    table = torch.zeros(world * t_local * topk, H, device=device, dtype=torch.float32)
    for r, c in zip(range(world), counts):
        table[slots_all[r][: int(c)].long()] = rows_all[r][: int(c)]
    mine = table[rank * t_local * topk : (rank + 1) * t_local * topk].view(t_local, topk, H)
    y_ref = (mine * scores[:, :, None]).sum(1) + res.float()
    y_base = (mine.to(dtype).float() * scores[:, :, None]).sum(
        1
    ) + res.float()  # the slots' bf16 rounding

    def check(label, out, ref, base):
        """out vs the fp32 ref, within twice the error of the same-dtype baseline."""
        err = (out.float() - ref).abs().max().item()
        tol = 2 * (base - ref).abs().max().item() + 1e-3
        if err > tol:
            failures.append(f"rank{rank} {label}: err={err} tol={tol}")

    for ragged_dim, tile_m, tile_n, cluster_m in tiles:
        cb = CombineRunner(t_local, topk, H, dtype, total, ragged_dim=ragged_dim)
        for it in range(
            2
        ):  # back-to-back: the slots are overwritten, the counter re-zeroed by prepare
            args = cb.prepare(s_scatter)
            if ragged_dim == "n":
                gemm(
                    w2,
                    act,
                    cb.d,
                    None,
                    None,
                    tile_m,
                    tile_n,
                    cluster_m,
                    1,
                    cu_seqlens_n=cu,
                    dist_op=cb.op,
                    dist_op_args=args,
                )
            else:
                gemm(
                    act,
                    w2,
                    cb.d,
                    None,
                    None,
                    tile_m,
                    tile_n,
                    cluster_m,
                    1,
                    cu_seqlens_m=cu,
                    dist_op=cb.op,
                    dist_op_args=args,
                )
            torch.cuda.synchronize(device)
            dist.barrier()  # every rank's drains have landed
            out = torch.empty(t_local, H, device=device, dtype=dtype)
            partials = torch.zeros(1, t_local, -(-H // 1024), device=device, dtype=torch.float32)
            sums = reduce_over_k(
                cb.slots,
                scores,
                out,
                epilogue=rms_partial,
                epi_args={"res": res, "sqsum": partials},
            )
            torch.cuda.synchronize(device)
            label = f"combine {ragged_dim} {tile_m}x{tile_n} c{cluster_m} iter={it}"
            check(label, out, y_ref, y_base.to(dtype).float())
            check(
                f"{label} sqsum",
                sums["sqsum"].view(-1),
                (y_ref * y_ref).sum(-1),
                (y_base * y_base).sum(-1),
            )
            dist.barrier()  # peers finish reading their slots before the next launch overwrites them

    fail_count = torch.tensor([len(failures)], device=device)
    dist.all_reduce(fail_count)
    if failures:
        print("\n".join(failures), file=sys.stderr, flush=True)
    dist.destroy_process_group()
    if int(fail_count.item()) > 0:
        sys.exit(1)
    if rank == 0:
        print("ALL FC2+COMBINE CHECKS PASSED", flush=True)


def _visible_gpus() -> str:
    """GPUs with no running compute processes (the node is shared), else all of them."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        free = [
            line.split(",")[0].strip()
            for line in out.strip().splitlines()
            if int(line.split(",")[1]) < 5
        ]
        return ",".join(free)
    except Exception:
        return ",".join(str(i) for i in range(torch.cuda.device_count()))


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs >= 2 GPUs")
def test_gemm_combine_multirank():
    gpus = _visible_gpus()
    nproc = min(len(gpus.split(",")), 8)
    assert nproc >= 2, f"not enough free GPUs: {gpus}"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpus
    env.setdefault("PYTHONPATH", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={nproc}",
            "--standalone",
            os.path.abspath(__file__),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=1200,
    )
    if res.returncode != 0:
        print(res.stdout[-4000:])
        print(res.stderr[-4000:], file=sys.stderr)
    assert res.returncode == 0
    assert "ALL FC2+COMBINE CHECKS PASSED" in res.stdout


# --- host refusals, one GPU: local slots stand in for the symmetric buffer (the drain stores to an address) ---

SM = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
needs_gpu = pytest.mark.skipif(SM < 9, reason="the distributed operators run on SM90 / SM100")


def _problem(device):
    from quack.distributed.combine import CombineArguments

    H, I, topk = 1024, 512, 2
    counts = torch.tensor([300, 520, 64, 700, 5])
    total, E = int(counts.sum()), len(counts)
    cu = torch.cat([torch.zeros(1, dtype=torch.int32), counts.cumsum(0).to(torch.int32)]).to(device)
    act = torch.randn(total, I, device=device, dtype=torch.bfloat16)
    w2 = torch.randn(E, H, I, device=device, dtype=torch.bfloat16) * 0.05
    tokens = -(-total // topk)
    slots = torch.zeros(tokens, topk, H, device=device, dtype=torch.bfloat16)
    row_key = torch.randperm(tokens * topk, device=device)[:total].to(torch.int32)
    slot_base = torch.tensor([slots.data_ptr()], dtype=torch.int64, device=device)
    d = torch.empty(total, H, device=device, dtype=torch.bfloat16)
    sem = torch.zeros(1, dtype=torch.int32, device=device)
    return H, cu, act, w2, d, CombineArguments(slot_base, tokens * topk, row_key, sem)


def _combine_gemm(ragged_dim, cu, act, w2, d, **kw):
    from quack.gemm import gemm

    if ragged_dim == "n":
        return gemm(w2, act, d.T, None, None, 256, 256, 2, 1, cu_seqlens_n=cu, **kw)
    return gemm(act, w2, d, None, None, 128, 256, 1, 1, cu_seqlens_m=cu, **kw)


@needs_gpu
@pytest.mark.parametrize(
    "case", ["dense op under varlen_m", "n-operator under varlen_m", "m-operator on a dense call"]
)
def test_gemm_combine_ragged_dim_mismatch(case):
    """An operator built for another ragged dim than the call's is refused on the host."""
    from quack.distributed.combine import CombineStore
    from quack.gemm import gemm
    from quack.gemm_base import DistOp

    device = torch.device("cuda")
    H, cu, act, w2, d, args = _problem(device)
    with pytest.raises(AssertionError, match="ragged dim"):
        if case == "dense op under varlen_m":
            _combine_gemm("m", cu, act, w2, d, dist_op=DistOp(), dist_op_args=args)
        elif case == "n-operator under varlen_m":
            _combine_gemm("m", cu, act, w2, d, dist_op=CombineStore(H, "n"), dist_op_args=args)
        else:
            gemm(
                act,
                w2[0],
                d,
                None,
                None,
                128,
                256,
                1,
                1,
                dist_op=CombineStore(H, "m"),
                dist_op_args=args,
            )


@needs_gpu
def test_gemm_combine_scheduler_is_the_operators():
    """The operator runs the GEMM on its tile counter; the GEMM's dynamic-scheduler knob is refused with it."""
    from quack.distributed.combine import CombineStore

    device = torch.device("cuda")
    H, cu, act, w2, d, args = _problem(device)
    with pytest.raises(AssertionError, match="tile counter"):
        _combine_gemm(
            "m",
            cu,
            act,
            w2,
            d,
            dist_op=CombineStore(H, "m"),
            dist_op_args=args,
            is_dynamic_persistent=True,
        )


@needs_gpu
@pytest.mark.skipif(SM >= 10, reason="cu_seqlens_n is admitted on SM100+")
def test_gemm_combine_swap_ab_refused_below_sm100():
    """The declared hardware limit: the swap-AB grouped GEMM needs the runtime-N MMA."""
    from quack.distributed.combine import CombineStore

    device = torch.device("cuda")
    H, cu, act, w2, d, args = _problem(device)
    with pytest.raises(AssertionError, match="SM100"):
        _combine_gemm("n", cu, act, w2, d, dist_op=CombineStore(H, "n"), dist_op_args=args)


if __name__ == "__main__":
    _run_rank()
