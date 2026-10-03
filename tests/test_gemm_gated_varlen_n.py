# Copyright (C) 2025, Tri Dao.
"""Swap-AB gated GEMM on ragged N (SM100): A = an expert's gate/up weights interleaved per 8 rows
(E, 2I, H), B = the tokens (total_n, H) expert-major with cu_seqlens_n, aux (I, total_n) m-major =
SwiGLU(gate) * up, i.e. the postact transposed. Pairs over M rows (TileStore pair_dim="m")."""

import pytest
import torch
import torch.nn.functional as F

import cutlass
import cutlass.cute as cute

from quack.compile_utils import make_fake_stream
from quack.cute_dsl_utils import get_device_capacity, get_max_active_clusters, torch2cute_dtype_map
from quack.epilogue.library import linear_act_mod
from quack.gemm_runtime.host import FakeArgCtx
from quack.gemm_tvm_ffi_utils import (
    make_fake_gemm_tensors,
    make_fake_scheduler_args,
    make_fake_varlen_args,
    make_scheduler_args,
    make_varlen_args,
)
from quack.rounding import RoundingMode

sm100_only = pytest.mark.skipif(
    not torch.cuda.is_available() or get_device_capacity(torch.device("cuda"))[0] not in (10, 11),
    reason="M-paired gated epilogue is SM100 only",
)


def interleave_gate_up(w1_cat):
    """(E, 2I, H) [gate; up] -> gate/up rows interleaved per 8: row 16q + i is gate[8q + i] for
    i < 8, up[8q + i - 8] otherwise."""
    E, I2, H = w1_cat.shape
    g, u = w1_cat.chunk(2, dim=1)
    return torch.stack([g.view(E, I2 // 16, 8, H), u.view(E, I2 // 16, 8, H)], dim=2).reshape(
        E, I2, H
    )


def compile_gated_swap_ab(dtype, use_clc, gather_B=False):
    """gather_B: B rows by index (mBIdx), TMA gather4 — the dedup token pool of the MoE dispatch."""
    mod = linear_act_mod("swiglu", gated=True, has_c=False, has_rowvec=False, has_colvec=False)
    Cls = mod._mint((), 10, True, False, (), RoundingMode.RN, (), False, pair_dim="m")
    gemm = Cls(
        cutlass.Float32,
        dtype,
        (256, 256),
        (2, 1, 1),
        use_clc_persistence=use_clc,
        gather_B=gather_B,
    )
    gemm.b_transposed = gemm.a_transposed = gemm.cd_transposed = False
    gemm.cd_packed = None
    mA, mB, mD, mC, m, n, k, l = make_fake_gemm_tensors(
        dtype, dtype, None, None, "k", "k", "n", "n", varlen_n=True, gather_B=gather_B
    )
    aux = next(op for op in Cls._epi_ops if op.name == "mAuxOut")
    epi = Cls.EpilogueArguments(
        mAuxOut=aux.host_fake_arg((dtype, "m"), FakeArgCtx(m, n, k, l, False, False))
    )
    sched = make_fake_scheduler_args(False, False, l)
    varlen = make_fake_varlen_args(False, False, False, None, varlen_n=True, gather_B=gather_B)
    fn = cute.compile(
        gemm, mA, mB, mD, mC, epi, sched, varlen, make_fake_stream(), options="--enable-tvm-ffi"
    )
    return fn, Cls


@sm100_only
@pytest.mark.parametrize(
    "use_clc, gather_B, pool_frac",
    # pool_frac 0.5: fewer pool rows than GEMM rows (duplicate indices) — the grid bound must come
    # from the index tensor, not the pool (tail tiles were unscheduled, 2026-10-05)
    [(True, False, 1.0), (False, False, 1.0), (True, True, 2.0), (True, True, 0.5)],
)
def test_gemm_gated_swap_ab_varlen_n(use_clc, gather_B, pool_frac):
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.random.manual_seed(0)
    H, I = 1024, 512
    counts = torch.tensor([300, 520, 64, 700, 5])  # 1 / 3 / 1 / 3 blocks, a thin expert
    total = int(counts.sum())
    E = len(counts)
    cu = torch.cat([torch.zeros(1, dtype=torch.int32), counts.cumsum(0).to(torch.int32)]).to(device)
    w1_cat = torch.randn(E, 2 * I, H, device=device, dtype=dtype) * 0.05
    w1_il = interleave_gate_up(w1_cat)
    tokens = torch.randn(total, H, device=device, dtype=dtype)
    postact = torch.empty(total, I, device=device, dtype=dtype)
    B, B_idx = tokens, None
    if gather_B:  # GEMM row i reads pool row B_idx[i]
        pool_rows = int(total * pool_frac)
        if pool_frac >= 1:
            B_idx = torch.randperm(pool_rows, device=device)[:total]
        else:
            B_idx = torch.randint(0, pool_rows, (total,), device=device)
        B = torch.randn(pool_rows, H, device=device, dtype=dtype)
        B[B_idx] = tokens  # with duplicates the last write wins: re-read the rows actually used
        tokens = B[B_idx]
        B_idx = B_idx.to(torch.int32)

    fn, Cls = compile_gated_swap_ab(torch2cute_dtype_map[dtype], use_clc, gather_B)
    clusters = get_max_active_clusters(2, get_device_capacity(device))
    fn(
        w1_il,
        B,
        None,
        None,
        Cls.EpilogueArguments(mAuxOut=postact.T),
        make_scheduler_args(clusters, 8, None),
        make_varlen_args(None, None, None, cu_seqlens_n=cu, B_idx=B_idx),
    )
    torch.cuda.synchronize()

    for e in range(E):
        x = tokens[cu[e] : cu[e + 1]]
        g, u = w1_cat[e].chunk(2, dim=0)
        ref = F.silu(x.float() @ g.float().T) * (x.float() @ u.float().T)
        ref_bf16 = F.silu((x @ g.T).float()) * (x @ u.T).float()
        got = postact[cu[e] : cu[e + 1]].float()
        assert (got - ref).abs().max() < 2 * (ref_bf16 - ref).abs().max() + 1e-3, f"expert {e}"


def even_split_targets(counts, flags_cols, tile_n=256, granule=64):
    """flags once every row has landed: rows of each expert's even-split N block (quack varlen_n_block_rows)."""
    rows = counts.to(torch.int64)[:, None]
    num_blocks = ((rows + tile_n - 1) // tile_n).clamp(min=1)
    b = ((rows + num_blocks - 1) // num_blocks + granule - 1) // granule * granule
    first = torch.arange(flags_cols)[None, :] * b
    return torch.minimum((rows - first).clamp(min=0), b).to(torch.int32)


@sm100_only
@pytest.mark.parametrize("use_clc", [True, False])
def test_varlen_n_dispatch_and_combine_gates(use_clc):
    """The MoE gates on the ragged N dim: GEMM1 (gather_B) with the dispatch flags held at their even-split targets
    and a rotation start, GEMM2-shaped plain D with the combine flags, which must count the fixed dim's CTA tiles."""
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.random.manual_seed(1)
    H, I = 1024, 512
    counts = torch.tensor([300, 520, 64, 700, 5])
    total, E = int(counts.sum()), len(counts)
    cu = torch.cat([torch.zeros(1, dtype=torch.int32), counts.cumsum(0).to(torch.int32)]).to(device)
    flags_cols = -(-total // 256)
    w1_cat = torch.randn(E, 2 * I, H, device=device, dtype=dtype) * 0.05
    tokens = torch.randn(total, H, device=device, dtype=dtype)
    B_idx = (
        torch.randperm(2 * total, device=device)[:total].sort().values
    )  # ascending, as the pool's rows land
    B = torch.randn(2 * total, H, device=device, dtype=dtype)
    B[B_idx] = tokens
    B_idx = B_idx.to(torch.int32)
    postact = torch.empty(total, I, device=device, dtype=dtype)
    clusters = get_max_active_clusters(2, get_device_capacity(device))

    # GEMM1: dispatch-gated, flags at target, the walk starting mid-pool (the rotation)
    mod = linear_act_mod("swiglu", gated=True, has_c=False, has_rowvec=False, has_colvec=False)
    Cls = mod._mint((), 10, True, False, (), RoundingMode.RN, (), False, pair_dim="m")
    gemm = Cls(
        cutlass.Float32,
        torch2cute_dtype_map[dtype],
        (256, 256),
        (2, 1, 1),
        use_clc_persistence=use_clc,
        gather_B=True,
    )
    gemm.b_transposed = gemm.a_transposed = gemm.cd_transposed = False
    gemm.cd_packed = None
    cdt = torch2cute_dtype_map[dtype]
    mA, mB, mD, mC, m, n, k, l = make_fake_gemm_tensors(
        cdt, cdt, None, None, "k", "k", "n", "n", varlen_n=True, gather_B=True
    )
    aux = next(op for op in Cls._epi_ops if op.name == "mAuxOut")
    epi = Cls.EpilogueArguments(
        mAuxOut=aux.host_fake_arg((cdt, "m"), FakeArgCtx(m, n, k, l, False, False))
    )
    sched = make_fake_scheduler_args(False, False, l, has_dispatch=True)
    varlen = make_fake_varlen_args(False, False, False, None, varlen_n=True, gather_B=True)
    fn1 = cute.compile(
        gemm, mA, mB, mD, mC, epi, sched, varlen, make_fake_stream(), options="--enable-tvm-ffi"
    )
    flags1 = even_split_targets(counts, flags_cols).to(device)
    first_src_row = torch.tensor([total], dtype=torch.int32, device=device)
    fn1(
        interleave_gate_up(w1_cat),
        B,
        None,
        None,
        Cls.EpilogueArguments(mAuxOut=postact.T),
        make_scheduler_args(clusters, 8, None, dispatch_args=(flags1, first_src_row)),
        make_varlen_args(None, None, None, cu_seqlens_n=cu, B_idx=B_idx),
    )
    torch.cuda.synchronize()
    for e in range(E):
        x = tokens[cu[e] : cu[e + 1]]
        g, u = w1_cat[e].chunk(2, dim=0)
        ref = F.silu(x.float() @ g.float().T) * (x.float() @ u.float().T)
        ref_bf16 = F.silu((x @ g.T).float()) * (x @ u.T).float()
        got = postact[cu[e] : cu[e + 1]].float()
        assert (got - ref).abs().max() < 2 * (ref_bf16 - ref).abs().max() + 1e-3, f"expert {e}"

    # GEMM2: plain D (weights (H, I) x tokens (total, I)^T), combine flags +1 per stored CTA tile
    from quack.gemm_default_epi import GemmDefaultSm100

    gemm2 = GemmDefaultSm100(
        cutlass.Float32, cdt, (256, 256), (2, 1, 1), use_clc_persistence=use_clc
    )
    gemm2.b_transposed = gemm2.a_transposed = gemm2.cd_transposed = False
    gemm2.cd_packed = None
    mA2, mB2, mD2, mC2, m2, n2, k2, l2 = make_fake_gemm_tensors(
        cdt, cdt, cdt, None, "k", "k", "m", "n", varlen_n=True
    )
    epi2 = GemmDefaultSm100.EpilogueArguments(add_to_output=None, rounding_mode=None)
    sched2 = make_fake_scheduler_args(False, False, l2, has_combine=True)
    varlen2 = make_fake_varlen_args(False, False, False, None, varlen_n=True)
    fn2 = cute.compile(
        gemm2,
        mA2,
        mB2,
        mD2,
        mC2,
        epi2,
        sched2,
        varlen2,
        make_fake_stream(),
        options="--enable-tvm-ffi",
    )
    w2 = torch.randn(E, H, I, device=device, dtype=dtype) * 0.05
    act = torch.randn(total, I, device=device, dtype=dtype)
    y = torch.empty(total, H, device=device, dtype=dtype)
    flags2 = torch.zeros(E, flags_cols, dtype=torch.int32, device=device)
    row_key = torch.arange(total, dtype=torch.int32, device=device)
    fn2(
        w2,
        act,
        y.T,
        None,
        epi2,
        make_scheduler_args(clusters, 8, None, combine_args=(flags2, row_key, first_src_row)),
        make_varlen_args(None, None, None, cu_seqlens_n=cu),
    )
    torch.cuda.synchronize()
    for e in range(E):
        ref = act[cu[e] : cu[e + 1]].float() @ w2[e].float().T
        assert (y[cu[e] : cu[e + 1]].float() - ref).abs().max() < 2 * (
            act[cu[e] : cu[e + 1]] @ w2[e].T - ref
        ).abs().max() + 1e-3
    num_blocks = ((counts + 255) // 256).to(device)
    expected = (torch.arange(flags_cols, device=device)[None, :] < num_blocks[:, None]).to(
        torch.int32
    ) * (2 * (H // 256))
    assert torch.equal(flags2, expected), (flags2, expected)
