# Copyright (C) 2026, Tri Dao.
"""GEMM2 + combine through quack.gemm.gemm on one GPU: the operator declares the ragged dim it runs on
(CombineStore.ragged_dim = "n", the swap-AB GEMM with tokens as D columns), the host admits cu_seqlens_n with
it, and the drain lands every expert-grouped token in its slot. The slots are a local buffer standing in for
the symmetric one: the drain stores to an address, the rank only selects the base."""

import pytest
import torch

from quack.cute_dsl_utils import get_device_capacity
from quack.distributed.combine import CombineArguments, CombineStore
from quack.gemm import gemm
from quack.gemm_base import DistOp

sm100_only = pytest.mark.skipif(
    not torch.cuda.is_available() or get_device_capacity(torch.device("cuda"))[0] not in (10, 11),
    reason="the distributed operators run on SM100",
)
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
    d = torch.empty(
        total, H, device=device, dtype=torch.bfloat16
    )  # the GEMM's D = d.T, never written
    return cu, act, w2, slots, d, CombineArguments(slot_base, tokens * TOPK, row_key)


@sm100_only
def test_gemm_combine():
    device = torch.device("cuda")
    torch.random.manual_seed(0)
    cu, act, w2, slots, d, args = make_problem(device)
    gemm(
        w2,
        act,
        d.T,
        None,
        None,
        256,
        256,
        2,
        1,
        cu_seqlens_n=cu,
        dist_op=CombineStore(H),
        dist_op_args=args,
    )
    torch.cuda.synchronize()
    landed = slots.view(-1, H)[args.row_key.long()]
    for e in range(len(COUNTS)):
        rows = slice(int(cu[e]), int(cu[e + 1]))
        ref = act[rows].float() @ w2[e].float().T
        err = (landed[rows].float() - ref).abs().max()
        assert err < 2 * (ref.bfloat16().float() - ref).abs().max() + 1e-3, f"expert {e}"


@sm100_only
@pytest.mark.parametrize("dist_op, varlen", [(DistOp(), True), (CombineStore(H), False)])
def test_gemm_combine_ragged_dim_mismatch(dist_op, varlen):
    """A dense operator under varlen_n, or a ragged-N operator on a dense call, is refused on the host."""
    device = torch.device("cuda")
    cu, act, w2, slots, d, args = make_problem(device)
    kw = dict(cu_seqlens_n=cu) if varlen else {}
    with pytest.raises(AssertionError, match="ragged dim"):
        gemm(
            w2 if varlen else w2[0],
            act,
            d.T,
            None,
            None,
            256,
            256,
            2,
            1,
            dist_op=dist_op,
            dist_op_args=args,
            **kw,
        )
