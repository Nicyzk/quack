# Copyright (C) 2026, Tri Dao.
"""GEMM2 + combine through quack.gemm.gemm on one GPU: the operator declares the ragged dim it runs on
(CombineStore.ragged_dim: "n" = the swap-AB GEMM with tokens as D columns, "m" = varlen_m with tokens as D
rows), the host admits only the matching call, and the drain lands every expert-grouped token in its slot. SM90
has no cu_seqlens_n: the "n" case checks the host refusal there. The slots are a local buffer standing in for
the symmetric one: the drain stores to an address, the rank only selects the base."""

import pytest
import torch

from quack.cute_dsl_utils import get_device_capacity
from quack.distributed.combine import CombineArguments, CombineStore
from quack.gemm import gemm
from quack.gemm_base import DistOp

SM = get_device_capacity(torch.device("cuda"))[0] if torch.cuda.is_available() else 0
needs_gpu = pytest.mark.skipif(SM < 9, reason="the distributed operators run on SM90 / SM100")
H, I = 1024, 512
COUNTS = [300, 520, 64, 700, 5]
TOPK = 2


def make_problem(device):
    counts = torch.tensor(COUNTS)
    total, E = int(counts.sum()), len(COUNTS)
    cu = torch.cat([torch.zeros(1, dtype=torch.int32), counts.cumsum(0).to(torch.int32)]).to(device)
    act = torch.randn(total, I, device=device, dtype=torch.bfloat16)
    w2 = torch.randn(E, H, I, device=device, dtype=torch.bfloat16) * 0.05
    tokens = -(-total // TOPK)  # every row a distinct (token, k) slot
    slots = torch.zeros(tokens, TOPK, H, device=device, dtype=torch.bfloat16)
    row_key = torch.randperm(tokens * TOPK, device=device)[:total].to(torch.int32)
    slot_base = torch.tensor([slots.data_ptr()], dtype=torch.int64, device=device)
    d = torch.empty(total, H, device=device, dtype=torch.bfloat16)  # the GEMM's D, never written
    sem = torch.zeros(1, dtype=torch.int32, device=device)  # the GEMM's tile counter, zero per call
    return cu, act, w2, slots, d, CombineArguments(slot_base, tokens * TOPK, row_key, sem)


def combine_gemm(ragged_dim, cu, act, w2, d, **kw):
    """GEMM2 with tokens on the ragged dim: swap-AB (weights on M, D = d.T, cu_seqlens_n) or varlen_m."""
    if ragged_dim == "n":
        return gemm(w2, act, d.T, None, None, 256, 256, 2, 1, cu_seqlens_n=cu, **kw)
    return gemm(act, w2, d, None, None, 128, 256, 1, 1, cu_seqlens_m=cu, **kw)


@needs_gpu
@pytest.mark.parametrize("ragged_dim", ["m", "n"])
def test_gemm_combine(ragged_dim):
    device = torch.device("cuda")
    torch.random.manual_seed(0)
    cu, act, w2, slots, d, args = make_problem(device)
    op = CombineStore(H, ragged_dim)
    if ragged_dim == "n" and SM < 10:  # the declared limit: cu_seqlens_n is SM100+
        with pytest.raises(AssertionError, match="SM100"):
            combine_gemm(ragged_dim, cu, act, w2, d, dist_op=op, dist_op_args=args)
        return
    combine_gemm(ragged_dim, cu, act, w2, d, dist_op=op, dist_op_args=args)
    torch.cuda.synchronize()
    landed = slots.view(-1, H)[args.row_key.long()]
    for e in range(len(COUNTS)):
        rows = slice(int(cu[e]), int(cu[e + 1]))
        ref = act[rows].float() @ w2[e].float().T
        err = (landed[rows].float() - ref).abs().max()
        assert err < 2 * (ref.bfloat16().float() - ref).abs().max() + 1e-3, f"expert {e}"

    # the combine's sum over k with the block epilogue: score-weighted sum + residual into D, and the
    # per-token sum of squares (the RMSNorm statistic, finalized downstream as gemm_rms does)
    from quack.distributed.reduce_over_k import reduce_over_k
    from quack.distributed.reduce_over_k_epilogue import rms_partial

    tokens = slots.shape[0]
    scores = torch.rand(tokens, TOPK, device=device, dtype=torch.float32)
    res = torch.randn(tokens, H, device=device, dtype=torch.bfloat16)
    out = torch.empty(tokens, H, device=device, dtype=torch.bfloat16)
    partials = torch.zeros(1, tokens, -(-H // 1024), device=device, dtype=torch.float32)
    sums = reduce_over_k(
        slots, scores, out, epilogue=rms_partial, epi_args={"res": res, "sqsum": partials}
    )
    torch.cuda.synchronize()
    y = (slots.float() * scores[:, :, None]).sum(1) + res.float()
    assert (out.float() - y).abs().max() < 2 * (y.bfloat16().float() - y).abs().max() + 1e-3
    assert torch.allclose(sums["sqsum"].view(-1), (y * y).sum(-1), rtol=2e-2, atol=1e-2)


@needs_gpu
@pytest.mark.parametrize(
    "dist_op, call",
    [(DistOp(), "m"), (CombineStore(H, "n"), "m"), (CombineStore(H, "m"), None)],
    ids=["dense op under varlen_m", "n-operator under varlen_m", "m-operator on a dense call"],
)
def test_gemm_combine_ragged_dim_mismatch(dist_op, call):
    """An operator built for another ragged dim than the call's is refused on the host."""
    device = torch.device("cuda")
    cu, act, w2, slots, d, args = make_problem(device)
    with pytest.raises(AssertionError, match="ragged dim"):
        if call == "m":
            combine_gemm("m", cu, act, w2, d, dist_op=dist_op, dist_op_args=args)
        else:
            gemm(act, w2[0], d, None, None, 128, 256, 1, 1, dist_op=dist_op, dist_op_args=args)


@needs_gpu
def test_gemm_combine_scheduler_is_the_operators():
    """The operator runs the GEMM on its tile counter; the GEMM's dynamic-scheduler knob is refused with it."""
    device = torch.device("cuda")
    cu, act, w2, slots, d, args = make_problem(device)
    with pytest.raises(AssertionError, match="tile counter"):
        combine_gemm(
            "m",
            cu,
            act,
            w2,
            d,
            dist_op=CombineStore(H, "m"),
            dist_op_args=args,
            is_dynamic_persistent=True,
        )
