# Copyright (C) 2026, Tri Dao.
"""Swap-AB grouped GEMM on ragged N (SM100): A = per-expert weights (E, M, K) k-major, B = the tokens
(total_n, K) expert-major with cu_seqlens_n, D / aux (M, total_n) m-major. fc1 is the SwiGLU gated
GEMM with gate/up rows interleaved per 8 (TileStore pair_dim="m"); fc2 is the plain GEMM. Formats:
bf16, mxfp8 (e4m3, ue8m0 per 32 along K), nvfp4 (e2m1, e4m3 per 16, one per-tensor scale each applied
as alpha). Blockscaled B scales live in per-block atoms: even-split block j of expert e owns the MMA's
SF window at slot cu_tiles_n[e] + j, so every tile's window starts at an atom boundary."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import cutlass
import cutlass.cute as cute
from cutlass import Float32

from quack.blockscaled.operand import BlockScaledOperand
from quack.blockscaled.quantize import nvfp4_per_tensor_scale, to_mx, unpack_scale_blocked_to_2d
from quack.compile_utils import make_fake_stream
from quack.cute_dsl_utils import get_device_capacity, get_max_active_clusters
from quack.epilogue.library import gated_quant_mod, linear_act_mod
from quack.gemm_default_epi import GemmDefaultSm100
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
    reason="varlen_n (runtime-N MMA, M-paired gated epilogue) is SM100 only",
)
FP4X2 = torch.float4_e2m1fn_x2
# ab dtype, scale dtype (cute, torch), scale vector
FORMATS = {
    "bf16": (cutlass.BFloat16, None, None, None),
    "mxfp8": (cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU, torch.float8_e8m0fnu, 32),
    "nvfp4": (cutlass.Float4E2M1FN, cutlass.Float8E4M3FN, torch.float8_e4m3fn, 16),
}
H, I = 1024, 512
COUNTS = [300, 520, 64, 700, 5]  # thin, one-block, multi-block experts; ragged last blocks


def interleave_gate_up(w):
    """(E, 2I, X) [gate; up] -> gate/up rows interleaved per 8: row 16q + i is gate[8q + i] for
    i < 8, up[8q + i - 8] otherwise. Values and scale rows alike (byte view: no e8m0/fp4 cat)."""
    E, I2, X = w.shape
    g, u = w.view(torch.uint8).chunk(2, dim=1)
    il = torch.stack([g.view(E, I2 // 16, 8, -1), u.view(E, I2 // 16, 8, -1)], dim=2)
    return il.reshape(E, I2, -1).view(w.dtype)


def blocked_sf(scale):
    """(rows, sf_k), rows % 128 == 0 -> the kernel's (rm, rk, 32, 4, 4) blocked atoms (sf_k padded to 4)."""
    rows, sf_k = scale.shape
    rm, rk = rows // 128, -(-sf_k // 4)
    out = torch.zeros(rm, rk, 32, 4, 4, dtype=torch.uint8, device=scale.device)
    r = torch.arange(rows, device=scale.device)[:, None]
    k = torch.arange(sf_k, device=scale.device)[None, :]
    out[r // 128, k // 4, r % 32, (r // 32) % 4, k % 4] = scale.view(torch.uint8)
    return out.view(scale.dtype)


def unblock_sf(sf6):
    """Inverse of blocked_sf: (rm, rk, 32, 4, 4) -> (rm * 128, rk * 4)."""
    rm, rk = sf6.shape[:2]
    out = torch.empty(rm * 128, rk * 4, dtype=torch.uint8, device=sf6.device)
    r = torch.arange(rm * 128, device=sf6.device)[:, None]
    k = torch.arange(rk * 4, device=sf6.device)[None, :]
    out[r, k] = sf6.view(torch.uint8)[r // 128, k // 4, r % 32, (r // 32) % 4, k % 4]
    return out.view(sf6.dtype)


def even_split(counts, tile_n, granule=VARLEN_N_GRANULE):
    """Host twin of varlen_n_block_rows: (blocks per expert, rows per block)."""
    num_blocks = ((counts + tile_n - 1) // tile_n).clamp(min=1)
    b = ((counts + num_blocks - 1) // num_blocks + granule - 1) // granule * granule
    return num_blocks, b


def cumsum0(x):
    return torch.cat([torch.zeros(1, dtype=torch.int32), x.cumsum(0).to(torch.int32)])


def block_spans(p):
    """(expert, lo, hi, slot) of every even-split block."""
    for e in range(p.E):
        for j in range(int(p.num_blocks[e])):
            lo = int(p.cu[e]) + j * int(p.b[e])
            yield e, lo, min(lo + int(p.b[e]), int(p.cu[e + 1])), int(p.cu_tiles[e]) + j


def quantize(x, fmt):
    """(..., K) bf16 -> (values, 2-D scales or None, per-tensor scale or None, fp32 dequantized)."""
    if fmt == "bf16":
        return x, None, None, x.float()
    vec = FORMATS[fmt][3]
    if fmt == "mxfp8":
        q, s = to_mx(x, vec)
        return q, s, None, q.float() * s.float().repeat_interleave(vec, dim=-1)
    pts = nvfp4_per_tensor_scale(x.abs().amax())
    x2 = x.reshape(-1, x.shape[-1])
    op = BlockScaledOperand.quantize(x2, "nvfp4", per_tensor_scale=pts)
    sf2d = unpack_scale_blocked_to_2d(op.scale.unsqueeze(0), x2.shape[0], x2.shape[1] // vec)[0]
    q = op.qdata.view(torch.uint8).reshape(*x.shape[:-1], x.shape[-1] // 2).view(FP4X2)
    return q, sf2d.reshape(*x.shape[:-1], -1), pts, op.dequantize(torch.float32).reshape(x.shape)


def block_sf_atoms(p, tok_s, atoms):
    """Token scales into per-block atoms, (1, slots * atoms, K / 128, 32, 4, 4)."""
    slot_rows = atoms * 128
    pad = torch.zeros(
        int(p.cu_tiles[-1]) * slot_rows, tok_s.shape[1], dtype=torch.uint8, device=tok_s.device
    )
    for _, lo, hi, slot in block_spans(p):
        pad[slot * slot_rows : slot * slot_rows + hi - lo] = tok_s[lo:hi].view(torch.uint8)
    return blocked_sf(pad)[None].view(tok_s.dtype)


def stack_sf(scales):
    """Per-expert blocked atoms, stacked through a byte view (no e8m0 cat kernel)."""
    return torch.stack([blocked_sf(s).view(torch.uint8) for s in scales]).view(scales.dtype)


def mint_fc1(fmt, quant=False):
    """The fc1 class: SwiGLU gated, pairs over M rows; nvfp4 takes the per-tensor scales as alpha."""
    if quant:
        mod = gated_quant_mod("swiglu", "m")
    else:
        mod = linear_act_mod(
            "swiglu",
            gated=True,
            has_c=False,
            has_rowvec=False,
            has_colvec=False,
            has_alpha=fmt == "nvfp4",
        )
    scalars = (("alpha", "scalar"),) if fmt == "nvfp4" else ()
    return mod._mint(scalars, 10, True, False, (), RoundingMode.RN, (), False, pair_dim="m")


def compile_varlen_n(
    Cls, fmt, tile_n, epi_fake, d_dtype=None, d_major="n", use_clc=False, gather_B=False
):
    """Compile Cls on the varlen_n fakes; epi_fake(Cls, ctx) builds the fake EpilogueArguments."""
    ab, sf, _, vec = FORMATS[fmt]
    kw = {} if sf is None else dict(sf_vec_size=vec)
    gemm = Cls(
        Float32, ab, (256, tile_n), (2, 1, 1), use_clc_persistence=use_clc, gather_B=gather_B, **kw
    )
    gemm.b_transposed = gemm.a_transposed = gemm.cd_transposed = False
    gemm.cd_packed = None
    mA, mB, mD, mC, m, n, k, l = make_fake_gemm_tensors(
        ab, ab, d_dtype, None, "k", "k", d_major, "n", varlen_n=True, gather_B=gather_B
    )
    epi = epi_fake(Cls, FakeArgCtx(m, n, k, l, False, False))
    sched = make_fake_scheduler_args(False, False, l)
    varlen = make_fake_varlen_args(
        False, False, False, None, varlen_n=True, gather_B=gather_B, has_cu_tiles_n=sf is not None
    )
    sfs = (
        () if sf is None else (make_fake_sf_tensor(sf, l), make_fake_sf_tensor(sf, cute.sym_int()))
    )
    fn = cute.compile(
        gemm,
        mA,
        mB,
        mD,
        mC,
        epi,
        sched,
        varlen,
        make_fake_stream(),
        *sfs,
        options="--enable-tvm-ffi",
    )
    return fn, gemm


def aux_fake(dtype, fmt):
    def fake(Cls, ctx):
        aux = next(op for op in Cls._epi_ops if op.name == "mAuxOut")
        kw = {"alpha": Float32(0)} if fmt == "nvfp4" else {}
        return Cls.EpilogueArguments(mAuxOut=aux.host_fake_arg((dtype, "m"), ctx), **kw)

    return fake


def make_problem(fmt, tile_n, device, gather_B=False, pool_frac=1.0):
    """Experts' counts, gate/up weights and tokens in fmt; the token pool when gather_B."""
    p = SimpleNamespace(fmt=fmt, E=len(COUNTS), counts=torch.tensor(COUNTS))
    p.total = int(p.counts.sum())
    p.cu = cumsum0(p.counts).to(device)
    p.num_blocks, p.b = even_split(p.counts, tile_n)
    p.cu_tiles = cumsum0(p.num_blocks).to(device) if FORMATS[fmt][1] is not None else None
    w1_cat = torch.randn(p.E, 2 * I, H, device=device, dtype=torch.bfloat16) * 0.05
    w1_q, w1_s, p.pts_w, p.w1_dq = quantize(w1_cat, fmt)
    p.w1_q_il = interleave_gate_up(w1_q)
    p.sfa = None if w1_s is None else stack_sf(interleave_gate_up(w1_s))
    tokens = torch.randn(p.total, H, device=device, dtype=torch.bfloat16)
    p.B_idx = None
    if gather_B:  # GEMM row i reads pool row B_idx[i]; with duplicates the last write wins
        assert fmt == "bf16"
        pool_rows = int(p.total * pool_frac)
        if pool_frac >= 1:
            B_idx = torch.randperm(pool_rows, device=device)[: p.total]
        else:
            B_idx = torch.randint(0, pool_rows, (p.total,), device=device)
        pool = torch.randn(pool_rows, H, device=device, dtype=torch.bfloat16)
        pool[B_idx] = tokens
        tokens, p.B_idx = pool[B_idx], B_idx.to(torch.int32)
        p.B = pool
    p.tok_q, p.tok_s, p.pts_tok, p.tok_dq = quantize(tokens, fmt)
    if not gather_B:
        p.B = p.tok_q
    return p


def launch_args(p, device):
    clusters = get_max_active_clusters(2, get_device_capacity(device))
    return (
        make_scheduler_args(clusters, 8, None),
        make_varlen_args(None, None, None, cu_seqlens_n=p.cu, B_idx=p.B_idx, cu_tiles_n=p.cu_tiles),
    )


def assert_bf16_close(got, ref, msg):
    assert (got.float() - ref).abs().max() < 2 * (
        ref.bfloat16().float() - ref
    ).abs().max() + 1e-3, msg


def ref_fc1(p, e):
    x = p.tok_dq[p.cu[e] : p.cu[e + 1]]
    g, u = p.w1_dq[e].chunk(2, dim=0)
    return F.silu(x @ g.T) * (x @ u.T)


@sm100_only
@pytest.mark.parametrize(
    "fmt, tile_n, use_clc, gather_B, pool_frac",
    # pool_frac 0.5: fewer pool rows than GEMM rows (duplicate indices) — the grid bound must come
    # from the index tensor, not the pool (tail tiles were unscheduled, 2026-10-05)
    [
        ("bf16", 256, True, False, 1.0),
        ("bf16", 256, False, False, 1.0),
        ("bf16", 256, True, True, 2.0),
        ("bf16", 256, True, True, 0.5),
        ("mxfp8", 192, False, False, 1.0),
        ("mxfp8", 256, False, False, 1.0),
        ("nvfp4", 192, False, False, 1.0),
        ("nvfp4", 256, False, False, 1.0),
    ],
)
def test_fc1_gated(fmt, tile_n, use_clc, gather_B, pool_frac):
    """fc1: aux (I, total_n) m-major = SwiGLU(gate) * up in bf16."""
    device = torch.device("cuda")
    torch.random.manual_seed(0)
    fn, gemm = compile_varlen_n(
        mint_fc1(fmt),
        fmt,
        tile_n,
        aux_fake(cutlass.BFloat16, fmt),
        use_clc=use_clc,
        gather_B=gather_B,
    )
    p = make_problem(fmt, tile_n, device, gather_B, pool_frac)
    postact = torch.empty(p.total, I, device=device, dtype=torch.bfloat16)
    epi_kw = {"alpha": Float32(float(p.pts_w * p.pts_tok))} if fmt == "nvfp4" else {}
    sfs = () if p.sfa is None else (p.sfa, block_sf_atoms(p, p.tok_s, gemm.sfb_window_atoms))
    Cls = type(gemm)
    fn(
        p.w1_q_il,
        p.B,
        None,
        None,
        Cls.EpilogueArguments(mAuxOut=postact.T, **epi_kw),
        *launch_args(p, device),
        *sfs,
    )
    torch.cuda.synchronize()
    for e in range(p.E):
        assert_bf16_close(postact[p.cu[e] : p.cu[e + 1]], ref_fc1(p, e), f"expert {e}")


@sm100_only
@pytest.mark.parametrize("use_clc", [True, False])
def test_fc2_plain_d(use_clc):
    """fc2: plain D, weights (H, I) x tokens (total, I)^T, y.T m-major."""
    device = torch.device("cuda")
    torch.random.manual_seed(1)
    fmt, dtype = "bf16", torch.bfloat16
    plain = lambda Cls, ctx: Cls.EpilogueArguments(add_to_output=None, rounding_mode=None)
    fn, _ = compile_varlen_n(
        GemmDefaultSm100, fmt, 256, plain, cutlass.BFloat16, "m", use_clc=use_clc
    )
    p = make_problem(fmt, 256, device)
    w2 = torch.randn(p.E, H, I, device=device, dtype=dtype) * 0.05
    act = torch.randn(p.total, I, device=device, dtype=dtype)
    y = torch.empty(p.total, H, device=device, dtype=dtype)
    fn(w2, act, y.T, None, plain(GemmDefaultSm100, None), *launch_args(p, device))
    torch.cuda.synchronize()
    for e in range(p.E):
        ref = act[p.cu[e] : p.cu[e + 1]].float() @ w2[e].float().T
        assert_bf16_close(y[p.cu[e] : p.cu[e + 1]], ref, f"expert {e}")


@sm100_only
@pytest.mark.parametrize("fmt, tile_n", [("mxfp8", 192), ("mxfp8", 256)])
def test_fc1_quant_postact_fc2(fmt, tile_n):
    """fc1 quantizes its postact in the epilogue (values (I, total_n) m-major, scales per 32 along I into
    the per-block atoms, column-direction SFD) and fc2 consumes them as its B."""
    device = torch.device("cuda")
    torch.random.manual_seed(0)
    ab, sf, sf_torch, vec = FORMATS[fmt]

    def quant_fake(Cls, ctx):
        ops = {op.name: op for op in Cls._epi_ops}
        # per-block SF atoms: batch extent 1, not the experts (host_fake_arg's batched default)
        return Cls.EpilogueArguments(
            postact=ops["postact"].host_fake_arg((ab, "m"), ctx),
            postact_sf=make_fake_sf_tensor(sf, cute.sym_int()),
        )

    fn1, gemm1 = compile_varlen_n(mint_fc1(fmt, quant=True), fmt, tile_n, quant_fake)
    p = make_problem(fmt, tile_n, device)
    atoms = gemm1.sfb_window_atoms
    slots = int(p.cu_tiles[-1])
    postact = torch.zeros(p.total, I, device=device, dtype=torch.float8_e4m3fn)
    postact_sf = torch.zeros(1, slots * atoms, I // 128, 32, 4, 4, dtype=sf_torch, device=device)
    sched, varlen = launch_args(p, device)
    Cls = type(gemm1)
    fn1(
        p.w1_q_il,
        p.B,
        None,
        None,
        Cls.EpilogueArguments(postact=postact.T, postact_sf=postact_sf),
        sched,
        varlen,
        p.sfa,
        block_sf_atoms(p, p.tok_s, atoms),
    )
    torch.cuda.synchronize()

    # The dequantized postact against the fp32 reference: per block along I, within e4m3's half ulp of
    # the block's amax (2^-4) plus the block-scale rounding (one more bit)
    act_sf = unblock_sf(postact_sf[0])
    act_dq = torch.empty(p.total, I, device=device, dtype=torch.float32)
    for _, lo, hi, slot in block_spans(p):
        s0 = slot * atoms * 128
        act_dq[lo:hi] = postact[lo:hi].float() * act_sf[
            s0 : s0 + hi - lo
        ].float().repeat_interleave(vec, dim=1)
    for e in range(p.E):
        ref = ref_fc1(p, e)
        err = (act_dq[p.cu[e] : p.cu[e + 1]] - ref).abs().view(-1, I // vec, vec).amax(dim=-1)
        amax = ref.abs().view(-1, I // vec, vec).amax(dim=-1)
        assert (err <= amax * (2.0**-4 + 2.0**-5) + 1e-3).all(), (
            f"expert {e}: postact quantization off"
        )

    # fc2 on the quantized postact: y = postact @ w2[e].T per expert
    plain = lambda Cls, ctx: Cls.EpilogueArguments(add_to_output=None, rounding_mode=None)
    fn2, _ = compile_varlen_n(GemmDefaultSm100, fmt, tile_n, plain, cutlass.BFloat16, "m")
    w2 = torch.randn(p.E, H, I, device=device, dtype=torch.bfloat16) * 0.05
    w2_q, w2_s, _, w2_dq = quantize(w2, fmt)
    sfa2 = stack_sf(w2_s)
    y = torch.zeros(p.total, H, device=device, dtype=torch.bfloat16)
    fn2(w2_q, postact, y.T, None, plain(GemmDefaultSm100, None), sched, varlen, sfa2, postact_sf)
    torch.cuda.synchronize()
    for e in range(p.E):
        ref = act_dq[p.cu[e] : p.cu[e + 1]] @ w2_dq[e].T
        assert_bf16_close(y[p.cu[e] : p.cu[e + 1]], ref, f"expert {e}: fc2")
