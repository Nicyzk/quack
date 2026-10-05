# Copyright (C) 2026, Tri Dao.
"""Swap-AB gated GEMM on ragged N with MXFP8 operands (SM100): A = an expert's gate/up weights interleaved per 8
rows (E, 2I, H) e4m3 with ue8m0 scales per 32 along H (the standard batched (E, rm, rk, 32, 4, 4) layout), B = the
tokens (total_n, H) e4m3 expert-major with cu_seqlens_n, their scales in per-block atoms: even-split block j of
expert e owns the MMA's SF window (2 x 128-row atoms) at block slot cu_tiles_n[e] + j, so every tile's window starts
at an atom boundary and the MMAs run at the block's rows (16-granular, runtime instruction descriptor). aux
(I, total_n) m-major = SwiGLU(gate) * up in bf16."""

import pytest
import torch
import torch.nn.functional as F

import cutlass
import cutlass.cute as cute

from quack.blockscaled.quantize import to_mx
from quack.compile_utils import make_fake_stream
from quack.cute_dsl_utils import get_device_capacity, get_max_active_clusters
from quack.epilogue.library import linear_act_mod
from quack.gemm_runtime.host import FakeArgCtx
from quack.gemm_tvm_ffi_utils import (
    make_fake_gemm_tensors,
    make_fake_scheduler_args,
    make_fake_sf_tensor,
    make_fake_varlen_args,
    make_scheduler_args,
    make_varlen_args,
)
from quack.rounding import RoundingMode
from quack.varlen_utils import VARLEN_N_GRANULE

sm100_only = pytest.mark.skipif(
    not torch.cuda.is_available() or get_device_capacity(torch.device("cuda"))[0] not in (10, 11),
    reason="M-paired gated epilogue is SM100 only",
)
SF_VEC = 32


def interleave_gate_up(w1_cat):
    """(E, 2I, X) [gate; up] -> gate/up rows interleaved per 8 (the weights and their scale rows alike)."""
    E, I2, X = w1_cat.shape
    g, u = w1_cat.chunk(2, dim=1)
    return torch.stack([g.view(E, I2 // 16, 8, X), u.view(E, I2 // 16, 8, X)], dim=2).reshape(E, I2, X)


def blocked_sf(scale):
    """(rows, sf_k) e8m0, rows % 128 == 0 -> the kernel's (rm, rk, 32, 4, 4) blocked atoms (sf_k padded to 4)."""
    rows, sf_k = scale.shape
    rm, rk = rows // 128, -(-sf_k // 4)
    out = torch.zeros(rm, rk, 32, 4, 4, dtype=torch.uint8, device=scale.device)
    r = torch.arange(rows, device=scale.device)[:, None]
    k = torch.arange(sf_k, device=scale.device)[None, :]
    out[r // 128, k // 4, r % 32, (r // 32) % 4, k % 4] = scale.view(torch.uint8)
    return out.view(torch.float8_e8m0fnu)


def even_split(counts, tile_n, granule=VARLEN_N_GRANULE):
    """Host twin of varlen_n_block_rows: (blocks per expert, rows per block) over a counts tensor."""
    num_blocks = ((counts + tile_n - 1) // tile_n).clamp(min=1)
    b = ((counts + num_blocks - 1) // num_blocks + granule - 1) // granule * granule
    return num_blocks, b


def compile_gated_swap_ab_mxfp8(tile_n):
    mod = linear_act_mod("swiglu", gated=True, has_c=False, has_rowvec=False, has_colvec=False)
    Cls = mod._mint((), 10, True, False, (), RoundingMode.RN, (), False, pair_dim="m")
    gemm = Cls(
        cutlass.Float32,
        cutlass.Float8E4M3FN,
        (256, tile_n),
        (2, 1, 1),
        sf_vec_size=SF_VEC,
        use_clc_persistence=False,
        varlen_n_dynamic_n=True,
    )
    gemm.b_transposed = gemm.a_transposed = gemm.cd_transposed = False
    gemm.cd_packed = None
    fp8 = cutlass.Float8E4M3FN
    mA, mB, mD, mC, m, n, k, l = make_fake_gemm_tensors(fp8, fp8, None, None, "k", "k", "n", "n", varlen_n=True)
    aux = next(op for op in Cls._epi_ops if op.name == "mAuxOut")
    epi = Cls.EpilogueArguments(
        mAuxOut=aux.host_fake_arg((cutlass.BFloat16, "m"), FakeArgCtx(m, n, k, l, False, False))
    )
    sched = make_fake_scheduler_args(False, False, l)
    varlen = make_fake_varlen_args(False, False, False, None, varlen_n=True, has_cu_tiles_n=True)
    sfa = make_fake_sf_tensor(cutlass.Float8E8M0FNU, l)
    sfb = make_fake_sf_tensor(cutlass.Float8E8M0FNU, cute.sym_int())
    fn = cute.compile(
        gemm, mA, mB, mD, mC, epi, sched, varlen, make_fake_stream(), sfa, sfb, options="--enable-tvm-ffi"
    )
    return fn, Cls, gemm.sfb_window_atoms


@sm100_only
@pytest.mark.parametrize("tile_n", [192, 256])
def test_gemm_gated_swap_ab_varlen_n_mxfp8(tile_n):
    device = torch.device("cuda")
    torch.random.manual_seed(0)
    H, I = 1024, 512
    counts = torch.tensor([300, 520, 64, 700, 5])  # thin, one-block, multi-block experts; ragged last blocks
    total, E = int(counts.sum()), len(counts)
    cu = torch.cat([torch.zeros(1, dtype=torch.int32), counts.cumsum(0).to(torch.int32)]).to(device)
    num_blocks, b = even_split(counts, tile_n)
    cu_tiles = torch.cat([torch.zeros(1, dtype=torch.int32), num_blocks.cumsum(0).to(torch.int32)]).to(device)

    # Weights: quantize [gate; up] per row along H, then interleave values and scale rows alike.
    w1_cat = torch.randn(E, 2 * I, H, device=device, dtype=torch.bfloat16) * 0.05
    w1_q, w1_s = to_mx(w1_cat, SF_VEC)  # (E, 2I, H) e4m3, (E, 2I, H/32) e8m0
    w1_dq = w1_q.float() * w1_s.float().repeat_interleave(SF_VEC, dim=-1)
    w1_q_il = interleave_gate_up(w1_q)
    w1_s_il = interleave_gate_up(w1_s.view(torch.uint8)).view(torch.float8_e8m0fnu)  # no e8m0 cat kernel
    sfa = torch.stack([blocked_sf(w1_s_il[e]).view(torch.uint8) for e in range(E)]).view(torch.float8_e8m0fnu)

    # Tokens: quantize per token along H; scales into per-block atoms (slot = cu_tiles[e] + j).
    tokens = torch.randn(total, H, device=device, dtype=torch.bfloat16)
    tok_q, tok_s = to_mx(tokens, SF_VEC)
    tok_dq = tok_q.float() * tok_s.float().repeat_interleave(SF_VEC, dim=-1)
    fn, Cls, atoms = compile_gated_swap_ab_mxfp8(tile_n)
    slot_rows = atoms * 128
    sf_pad = torch.zeros(int(cu_tiles[E]) * slot_rows, H // SF_VEC, dtype=torch.uint8, device=device)
    for e in range(E):
        for j in range(int(num_blocks[e])):
            lo = int(cu[e]) + j * int(b[e])
            hi = min(lo + int(b[e]), int(cu[e + 1]))
            slot = int(cu_tiles[e]) + j
            sf_pad[slot * slot_rows : slot * slot_rows + hi - lo] = tok_s[lo:hi].view(torch.uint8)
    sfb = blocked_sf(sf_pad.view(torch.float8_e8m0fnu))[None]  # (1, slots * atoms, H/128, 32, 4, 4)

    postact = torch.empty(total, I, device=device, dtype=torch.bfloat16)
    clusters = get_max_active_clusters(2, get_device_capacity(device))
    fn(
        w1_q_il,
        tok_q,
        None,
        None,
        Cls.EpilogueArguments(mAuxOut=postact.T),
        make_scheduler_args(clusters, 8, None),
        make_varlen_args(None, None, None, cu_seqlens_n=cu, cu_tiles_n=cu_tiles),
        sfa,
        sfb,
    )
    torch.cuda.synchronize()

    for e in range(E):
        x = tok_dq[cu[e] : cu[e + 1]]
        g, u = w1_dq[e].chunk(2, dim=0)
        ref = F.silu(x @ g.T) * (x @ u.T)
        got = postact[cu[e] : cu[e + 1]].float()
        assert (got - ref).abs().max() < 2 * (ref.bfloat16().float() - ref).abs().max() + 1e-3, f"expert {e}"


def unblock_sf(sf6):
    """Inverse of blocked_sf: (rm, rk, 32, 4, 4) -> (rm * 128, rk * 4) e8m0."""
    rm, rk = sf6.shape[:2]
    out = torch.empty(rm * 128, rk * 4, dtype=torch.uint8, device=sf6.device)
    r = torch.arange(rm * 128, device=sf6.device)[:, None]
    k = torch.arange(rk * 4, device=sf6.device)[None, :]
    out[r, k] = sf6.view(torch.uint8)[r // 128, k // 4, r % 32, (r // 32) % 4, k % 4]
    return out.view(torch.float8_e8m0fnu)


def compile_gated_quant_swap_ab_mxfp8(tile_n):
    """fc1 with the postact quantized in the epilogue: fp8 (I, total_n) m-major values and ue8m0 scales per 32
    along I into the per-block atoms fc2's B reads (gated_quant_mod pair_dim "m", column-direction SFD)."""
    from quack.epilogue.library import gated_quant_mod

    Cls = gated_quant_mod("swiglu", "m")._mint((), 10, True, False, (), RoundingMode.RN, (), False, pair_dim="m")
    gemm = Cls(
        cutlass.Float32,
        cutlass.Float8E4M3FN,
        (256, tile_n),
        (2, 1, 1),
        sf_vec_size=SF_VEC,
        use_clc_persistence=False,
        varlen_n_dynamic_n=True,
    )
    gemm.b_transposed = gemm.a_transposed = gemm.cd_transposed = False
    gemm.cd_packed = None
    fp8 = cutlass.Float8E4M3FN
    mA, mB, mD, mC, m, n, k, l = make_fake_gemm_tensors(fp8, fp8, None, None, "k", "k", "n", "n", varlen_n=True)
    fctx = FakeArgCtx(m, n, k, l, False, False)
    ops = {op.name: op for op in Cls._epi_ops}
    epi = Cls.EpilogueArguments(
        postact=ops["postact"].host_fake_arg((fp8, "m"), fctx),
        # per-block SF atoms: batch extent 1, not the experts (host_fake_arg's batched default)
        postact_sf=make_fake_sf_tensor(cutlass.Float8E8M0FNU, cute.sym_int()),
    )
    sched = make_fake_scheduler_args(False, False, l)
    varlen = make_fake_varlen_args(False, False, False, None, varlen_n=True, has_cu_tiles_n=True)
    sfa = make_fake_sf_tensor(cutlass.Float8E8M0FNU, l)
    sfb = make_fake_sf_tensor(cutlass.Float8E8M0FNU, cute.sym_int())
    fn = cute.compile(
        gemm, mA, mB, mD, mC, epi, sched, varlen, make_fake_stream(), sfa, sfb, options="--enable-tvm-ffi"
    )
    return fn, Cls, gemm.sfb_window_atoms


def compile_fc2_swap_ab_mxfp8(tile_n):
    """fc2: plain swap-AB blockscaled GEMM, A = w2 (E, H, I), B = the fp8 postact (total_n, I) with its per-block
    scales, D = y.T (m-major) bf16."""
    from quack.gemm_default_epi import GemmDefaultSm100

    gemm = GemmDefaultSm100(
        cutlass.Float32, cutlass.Float8E4M3FN, (256, tile_n), (2, 1, 1), sf_vec_size=SF_VEC,
        use_clc_persistence=False, varlen_n_dynamic_n=True,
    )
    gemm.b_transposed = gemm.a_transposed = gemm.cd_transposed = False
    gemm.cd_packed = None
    fp8 = cutlass.Float8E4M3FN
    mA, mB, mD, mC, m, n, k, l = make_fake_gemm_tensors(fp8, fp8, cutlass.BFloat16, None, "k", "k", "m", "n", varlen_n=True)
    epi = GemmDefaultSm100.EpilogueArguments(add_to_output=False, rounding_mode=RoundingMode.RN)
    sched = make_fake_scheduler_args(False, False, l)
    varlen = make_fake_varlen_args(False, False, False, None, varlen_n=True, has_cu_tiles_n=True)
    sfa = make_fake_sf_tensor(cutlass.Float8E8M0FNU, l)
    sfb = make_fake_sf_tensor(cutlass.Float8E8M0FNU, cute.sym_int())
    fn = cute.compile(
        gemm, mA, mB, mD, mC, epi, sched, varlen, make_fake_stream(), sfa, sfb, options="--enable-tvm-ffi"
    )
    return fn


@sm100_only
@pytest.mark.parametrize("tile_n", [192, 256])
def test_gemm_gated_swap_ab_varlen_n_mxfp8_quant_postact(tile_n):
    """fc1 quantizes its postact (values + per-block scales along I) and fc2 consumes them as its B."""
    device = torch.device("cuda")
    torch.random.manual_seed(0)
    H, I = 1024, 512
    counts = torch.tensor([300, 520, 64, 700, 5])
    total, E = int(counts.sum()), len(counts)
    cu = torch.cat([torch.zeros(1, dtype=torch.int32), counts.cumsum(0).to(torch.int32)]).to(device)
    num_blocks, b = even_split(counts, tile_n)
    cu_tiles = torch.cat([torch.zeros(1, dtype=torch.int32), num_blocks.cumsum(0).to(torch.int32)]).to(device)
    w1_cat = torch.randn(E, 2 * I, H, device=device, dtype=torch.bfloat16) * 0.05
    w1_q, w1_s = to_mx(w1_cat, SF_VEC)
    w1_dq = w1_q.float() * w1_s.float().repeat_interleave(SF_VEC, dim=-1)
    w1_q_il = interleave_gate_up(w1_q)
    w1_s_il = interleave_gate_up(w1_s.view(torch.uint8)).view(torch.float8_e8m0fnu)
    sfa = torch.stack([blocked_sf(w1_s_il[e]).view(torch.uint8) for e in range(E)]).view(torch.float8_e8m0fnu)
    tokens = torch.randn(total, H, device=device, dtype=torch.bfloat16)
    tok_q, tok_s = to_mx(tokens, SF_VEC)
    tok_dq = tok_q.float() * tok_s.float().repeat_interleave(SF_VEC, dim=-1)
    fn1, Cls, atoms = compile_gated_quant_swap_ab_mxfp8(tile_n)
    slot_rows = atoms * 128
    slots = int(cu_tiles[E])
    sf_pad = torch.zeros(slots * slot_rows, H // SF_VEC, dtype=torch.uint8, device=device)
    for e in range(E):
        for j in range(int(num_blocks[e])):
            lo = int(cu[e]) + j * int(b[e])
            hi = min(lo + int(b[e]), int(cu[e + 1]))
            slot = int(cu_tiles[e]) + j
            sf_pad[slot * slot_rows : slot * slot_rows + hi - lo] = tok_s[lo:hi].view(torch.uint8)
    sfb = blocked_sf(sf_pad.view(torch.float8_e8m0fnu))[None]

    postact = torch.zeros(total, I, device=device, dtype=torch.float8_e4m3fn)
    postact_sf = torch.zeros(1, slots * atoms, I // 128, 32, 4, 4, dtype=torch.float8_e8m0fnu, device=device)
    clusters = get_max_active_clusters(2, get_device_capacity(device))
    varlen_rt = make_varlen_args(None, None, None, cu_seqlens_n=cu, cu_tiles_n=cu_tiles)
    fn1(
        w1_q_il, tok_q, None, None, Cls.EpilogueArguments(postact=postact.T, postact_sf=postact_sf),
        make_scheduler_args(clusters, 8, None), varlen_rt, sfa, sfb,
    )
    torch.cuda.synchronize()

    # The dequantized postact against the fp32 reference: per 32-wide block along I, within e4m3's half ulp
    # of the block's amax (2^-4) plus the block-scale rounding (one more bit)
    act_sf = unblock_sf(postact_sf[0])  # (slots * atom rows, I / 32)
    act_dq = torch.empty(total, I, device=device, dtype=torch.float32)
    for e in range(E):
        for j in range(int(num_blocks[e])):
            lo = int(cu[e]) + j * int(b[e])
            hi = min(lo + int(b[e]), int(cu[e + 1]))
            s0 = (int(cu_tiles[e]) + j) * slot_rows
            act_dq[lo:hi] = postact[lo:hi].float() * act_sf[s0 : s0 + hi - lo].float().repeat_interleave(SF_VEC, dim=1)
    for e in range(E):
        x = tok_dq[cu[e] : cu[e + 1]]
        g, u = w1_dq[e].chunk(2, dim=0)
        ref = F.silu(x @ g.T) * (x @ u.T)
        err = (act_dq[cu[e] : cu[e + 1]] - ref).abs().view(-1, I // SF_VEC, SF_VEC).amax(dim=-1)
        amax = ref.abs().view(-1, I // SF_VEC, SF_VEC).amax(dim=-1)
        assert (err <= amax * (2.0**-4 + 2.0**-5) + 1e-3).all(), f"expert {e}: postact quantization off"

    # fc2 on the quantized postact: y = postact @ w2[e].T per expert
    fn2 = compile_fc2_swap_ab_mxfp8(tile_n)
    w2 = torch.randn(E, H, I, device=device, dtype=torch.bfloat16) * 0.05
    w2_q, w2_s = to_mx(w2, SF_VEC)
    w2_dq = w2_q.float() * w2_s.float().repeat_interleave(SF_VEC, dim=-1)
    sfa2 = torch.stack([blocked_sf(w2_s[e]).view(torch.uint8) for e in range(E)]).view(torch.float8_e8m0fnu)
    y = torch.zeros(total, H, device=device, dtype=torch.bfloat16)
    from quack.gemm_default_epi import GemmDefaultSm100

    fn2(
        w2_q, postact, y.T, None, GemmDefaultSm100.EpilogueArguments(add_to_output=None, rounding_mode=None),
        make_scheduler_args(clusters, 8, None), varlen_rt, sfa2, postact_sf,
    )
    torch.cuda.synchronize()
    for e in range(E):
        ref = act_dq[cu[e] : cu[e + 1]] @ w2_dq[e].T
        got = y[cu[e] : cu[e + 1]].float()
        assert (got - ref).abs().max() < 2 * (ref.bfloat16().float() - ref).abs().max() + 1e-3, f"expert {e}: fc2"
