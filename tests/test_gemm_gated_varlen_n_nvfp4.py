# Copyright (C) 2026, Tri Dao.
"""Swap-AB gated GEMM on ragged N with NVFP4 operands (SM100): the MXFP8 test's shapes and per-block B scale atoms
with e2m1 values and e4m3 scales per 16 along H, both operands quantized with one per-tensor scale each (quack's
NVFP4 recipe: amax / 2688), their product applied as the epilogue's alpha before the SwiGLU. aux (I, total_n)
m-major = SwiGLU(gate) * up in bf16."""

import pytest
import torch
import torch.nn.functional as F

import cutlass
import cutlass.cute as cute
from cutlass import Float32

from quack.blockscaled.operand import BlockScaledOperand
from quack.blockscaled.quantize import nvfp4_per_tensor_scale, unpack_scale_blocked_to_2d
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

from test_gemm_gated_varlen_n_mxfp8 import blocked_sf, even_split, interleave_gate_up

sm100_only = pytest.mark.skipif(
    not torch.cuda.is_available() or get_device_capacity(torch.device("cuda"))[0] not in (10, 11),
    reason="M-paired gated epilogue is SM100 only",
)
SF_VEC = 16
FP4X2 = torch.float4_e2m1fn_x2


def quantize_nvfp4(x):
    """(..., K) bf16 -> (packed e2m1 (..., K/2) as float4_e2m1fn_x2, e4m3 scales (..., K/16), per-tensor scale fp32,
    dequantized fp32) with one per-tensor scale over the whole tensor."""
    pts = nvfp4_per_tensor_scale(x.abs().amax())
    x2 = x.reshape(-1, x.shape[-1])
    op = BlockScaledOperand.quantize(x2, "nvfp4", per_tensor_scale=pts)
    sf2d = unpack_scale_blocked_to_2d(op.scale.unsqueeze(0), x2.shape[0], x2.shape[1] // SF_VEC)[0]
    q = op.qdata.view(torch.uint8).reshape(*x.shape[:-1], x.shape[-1] // 2).view(FP4X2)
    # dequantize to fp32: the default returns the operand's bf16 and rounds codes * scale * pts (2^-9 relative)
    return (
        q,
        sf2d.reshape(*x.shape[:-1], x.shape[-1] // SF_VEC),
        pts,
        op.dequantize(torch.float32).reshape(x.shape),
    )


def compile_gated_swap_ab_nvfp4(tile_n):
    mod = linear_act_mod(
        "swiglu", gated=True, has_c=False, has_rowvec=False, has_colvec=False, has_alpha=True
    )
    Cls = mod._mint(
        (("alpha", "scalar"),), 10, True, False, (), RoundingMode.RN, (), False, pair_dim="m"
    )
    gemm = Cls(
        cutlass.Float32,
        cutlass.Float4E2M1FN,
        (256, tile_n),
        (2, 1, 1),
        sf_vec_size=SF_VEC,
        use_clc_persistence=False,
        varlen_n_dynamic_n=True,
    )
    gemm.b_transposed = gemm.a_transposed = gemm.cd_transposed = False
    gemm.cd_packed = None
    fp4 = cutlass.Float4E2M1FN
    mA, mB, mD, mC, m, n, k, l = make_fake_gemm_tensors(
        fp4, fp4, None, None, "k", "k", "n", "n", varlen_n=True
    )
    aux = next(op for op in Cls._epi_ops if op.name == "mAuxOut")
    epi = Cls.EpilogueArguments(
        mAuxOut=aux.host_fake_arg((cutlass.BFloat16, "m"), FakeArgCtx(m, n, k, l, False, False)),
        alpha=Float32(0),
    )
    sched = make_fake_scheduler_args(False, False, l)
    varlen = make_fake_varlen_args(False, False, False, None, varlen_n=True, has_cu_tiles_n=True)
    sfa = make_fake_sf_tensor(cutlass.Float8E4M3FN, l)
    sfb = make_fake_sf_tensor(cutlass.Float8E4M3FN, cute.sym_int())
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
        sfa,
        sfb,
        options="--enable-tvm-ffi",
    )
    return fn, Cls, gemm.sfb_window_atoms


@sm100_only
@pytest.mark.parametrize("tile_n", [192, 256])
def test_gemm_gated_swap_ab_varlen_n_nvfp4(tile_n):
    device = torch.device("cuda")
    torch.random.manual_seed(0)
    H, I = 1024, 512
    counts = torch.tensor([300, 520, 64, 700, 5])
    total, E = int(counts.sum()), len(counts)
    cu = torch.cat([torch.zeros(1, dtype=torch.int32), counts.cumsum(0).to(torch.int32)]).to(device)
    num_blocks, b = even_split(counts, tile_n)
    cu_tiles = torch.cat(
        [torch.zeros(1, dtype=torch.int32), num_blocks.cumsum(0).to(torch.int32)]
    ).to(device)

    # Weights: one per-tensor scale over all experts (alpha is one scalar per launch), then interleave gate/up rows
    # of the packed values and of the scale rows alike.
    w1_cat = torch.randn(E, 2 * I, H, device=device, dtype=torch.bfloat16) * 0.05
    w1_q, w1_s, pts_w, w1_dq = quantize_nvfp4(w1_cat)
    w1_q_il = interleave_gate_up(w1_q.view(torch.uint8)).view(FP4X2)
    w1_s_il = interleave_gate_up(w1_s.view(torch.uint8))
    sfa = torch.stack(
        [blocked_sf(w1_s_il[e].view(torch.float8_e8m0fnu)).view(torch.uint8) for e in range(E)]
    )
    sfa = sfa.view(torch.float8_e4m3fn)

    tokens = torch.randn(total, H, device=device, dtype=torch.bfloat16)
    tok_q, tok_s, pts_tok, tok_dq = quantize_nvfp4(tokens)
    fn, Cls, atoms = compile_gated_swap_ab_nvfp4(tile_n)
    slot_rows = atoms * 128
    sf_pad = torch.zeros(
        int(cu_tiles[E]) * slot_rows, H // SF_VEC, dtype=torch.uint8, device=device
    )
    for e in range(E):
        for j in range(int(num_blocks[e])):
            lo = int(cu[e]) + j * int(b[e])
            hi = min(lo + int(b[e]), int(cu[e + 1]))
            slot = int(cu_tiles[e]) + j
            sf_pad[slot * slot_rows : slot * slot_rows + hi - lo] = tok_s[lo:hi].view(torch.uint8)
    sfb = (
        blocked_sf(sf_pad.view(torch.float8_e8m0fnu))
        .view(torch.uint8)
        .view(torch.float8_e4m3fn)[None]
    )

    postact = torch.empty(total, I, device=device, dtype=torch.bfloat16)
    clusters = get_max_active_clusters(2, get_device_capacity(device))
    fn(
        w1_q_il,
        tok_q,
        None,
        None,
        Cls.EpilogueArguments(mAuxOut=postact.T, alpha=Float32(float(pts_w * pts_tok))),
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
        assert (got - ref).abs().max() < 2 * (ref.bfloat16().float() - ref).abs().max() + 1e-3, (
            f"expert {e}"
        )
