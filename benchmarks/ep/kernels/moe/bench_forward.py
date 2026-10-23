"""Run full MoE EP forward baselines through one interface (bench_dispatch.py's protocol and Kernel record).

Timed region = everything after the router: (x (T_local, H), top-k expert ids (T_local, K), router scores (T_local, K)
fp32) -> out (T_local, H) on the token's rank, every stage included (x publish, routing exchange, metadata, dispatch,
GEMM1 + SwiGLU, GEMM2, combine with the scores in the reduce). Shape from --model, bf16, uniform routing. Expert e's
weights come from generator seed e on every rank (expert_weights), so the reference regenerates any expert's weights
without holding all of them. w1 rows are [gate; up] (w1_cat) for DeepGEMM / TE / MegaMoE and gate/up interleaved
(w1_il, row 2j = gate j) for quack's gated epilogue; both are the same experts.

  sonic            sonic-moe PR #54's forward minus the router: sonicmoe.functional.ep._moe_ep_forward_inner after
                   the x publish, top-k all-gather and barrier of moe_ep_TC_softmax_topk_forward (rank-dedup dispatch
                   + combine, quack gemm_gated / gemm autotuned on first use). The canary: always timed.
  deepep_deepgemm  DeepEP V2 (~/DeepEP) expanded dispatch, experts aligned to DeepGEMM's 256 rows -> DeepGEMM
                   m-grouped bf16 GEMM (psum layout, no host sync) -> Transformer Engine's fused swiglu -> DeepGEMM
                   GEMM -> DeepEP combine (scores applied in its reduce). DeepGEMM's own baseline, with TE's swiglu
                   in place of its tilelang op (not installed).
  deepep_te        the same with Transformer Engine's GroupedLinear (cuBLASLt grouped GEMM on device split sizes, no
                   host sync; needs cuBLASLt 13.3+) and expert alignment 1.
  megamoe          DeepGEMM's bf16 Mega MoE (DeepSeek): x, top-k ids and scores copied into its symmetric buffer,
                   one persistent kernel for dispatch, both GEMMs, SwiGLU and combine.
  megamoe_mxfp8    the same in its fp8xfp8 mode (MXFP8: e4m3 + ue8m0 per 32 along K): x and the weights prequantized
                   at setup, x copied in the timed call, fc1's epilogue requantizes the postact, combine bf16; against
                   reference_megamoe_mxfp8 (its own rounding rules).
  megamoe_cutedsl  NVIDIA's CuTe-DSL bf16 Mega MoE (cutedsl_megamoe-main/moe_bf16_glu, built through its own tester):
                   x, top-k ids and scores copied into its NVSHMEM symmetric heap, one persistent kernel for
                   dispatch, fc1 + SwiGLU (scores applied there), fc2 and the write-back of every (token, k) cell
                   into a (T_local, K, H) buffer (its deterministic form A; the in-kernel atomic reduce is not
                   used) on dedicated write-back warps, then our gather-and-sum over K (reduce_over_k).
  quack_inter      the top-k all-gather, our routing metadata kernels (dispatch/routing_metadata.py: the same tables
                   as sonic's, 2 launches), then our inter-SM pipeline: the device walks,
                   fused dispatch + GEMM1 (dispatch/pull.py pull on --comm-sms SMs, quack's gated GEMM on
                   the rest, dispatch scheduler; bench_fused_dispatch_gemm.py's arrangement with the SwiGLU
                   epilogue), fused GEMM2 + combine (combine/combine_push.py push on the same partition, the GEMM
                   on the rest; bench_fused_gemm_combine.py's arrangement), barrier, our gather-and-sum with the
                   scores (reduce_over_k, CuTe-DSL; replaced sonic's triton kernel 2026-10-04, same speed or better). GEMM1 on 256x512 (2,1) static tiles, GEMM2 on 256x256 (2,1): its K is short, so the
                   256x512 tile's single tmem accumulator stage left the epilogue exposed (-5 % GEMM alone at 256x256).
  quack_intra_swap the intra-SM kernels on all SMs, no partitions: the swap-AB gated GEMM1 whose CTAs carry 4
                   transfer warps pulling the walk (fused_gemm_dispatch/gemm_sm100_intra_dispatch_pull.py), then the
                   swap-AB GEMM2 whose epilogue stores each token's slice straight into the origins' slots
                   (fused_gemm_combine/gemm_sm100_intra_combine_push.py), barrier, the same gather-and-sum.

--ref_check compares each arm's fresh output with a bf16-matmul reference (max rel err < 2^-6: three bf16 rounding
points and a K-term weighted sum), --r2r_check runs each arm twice for bit-identity.

Usage:
    PYTHONPATH=~/quack:~/sonic-moe-pr54:~/DeepEP LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH \\
        torchrun --nproc_per_node=4 --standalone bench_forward.py \\
        sonic deepep_deepgemm deepep_te megamoe quack_inter quack_intra_swap --comm-sms 16 --ref_check
"""
import os

os.environ.setdefault("TRITON_CACHE_AUTOTUNING", "1")  # sonic's triton autotune choices persist across launches
os.environ.setdefault("QUACK_COMPILE_WORKERS", "1")  # >1 workers hit a broken pipe (per the PR's test)

import argparse
import math
from types import SimpleNamespace
import sys

import cutlass
import cutlass.cute as cute
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.nn.functional as F
from cutlass import Float32, Int32, Int64
from quack.blockscaled.operand import BlockScaledOperand
from quack.blockscaled.quantize import nvfp4_per_tensor_scale, to_mx, unpack_scale_blocked_to_2d
from quack.compile_utils import make_fake_tensor
from quack.cute_dsl_utils import get_device_capacity, get_max_active_clusters, torch2cute_dtype_map
from quack.epilogue.library import gated_quant_mod, linear_act_mod
from quack.gemm_default_epi import GemmDefaultEpiMixin, GemmDefaultSm100
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
from sonicmoe.distributed_utils import CombineMode, DispatchMode, RuntimeEPConfig, SymmMemManager
from sonicmoe.enums import ActivationType
from sonicmoe.functional.distributed import build_rank_dedup_a_idx, compute_dispatch_metadata
from sonicmoe.functional.ep import _ag_routing_decision, _build_consumer_metadata, _moe_ep_forward_inner

from bench_dispatch import _NOOP, _run_synced, MODEL_CONFIGS, REPS, WARM_S, WINDOW_S, Kernel, SMStream, arm_stats, bench_all, r2r_check
from bench_fused_gemm_combine import (
    CLUSTER, SM100, TILE, GemmDefaultSm90IntraCombine, GemmDefaultSm100IntraCombine, _gather_and_sum, compile_gemm2,
    gemm2_args,
)
from combine.combine_push import make_combine_buffers, make_combine_push, make_combine_walk_device
from dispatch.pull import make_dispatch_pull
from dispatch.walk import fake_walk_arguments, make_dispatch_walk
from dispatch.prologue import make_prologue
from dispatch.routing_metadata import make_routing_metadata
from fused_gemm_dispatch.gemm_sm100_intra_dispatch_pull import GemmSm100 as GemmSm100IntraDispatch
from fused_gemm_dispatch.gemm_sm90_intra_dispatch_pull import GemmSm90 as GemmSm90IntraDispatch
from reduce_over_k import reduce_over_k

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../misc"))
from utils import capture_graph  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../all"))
from gemm_dispatch_pull import DispatchPull, fake_dispatch_arguments  # noqa: E402
from gemm_sm100 import GemmSm100 as GemmSm100Comm  # noqa: E402  # quack main's kernel + the comm object surface

sys.path.insert(0, os.path.expanduser("~/DeepEP"))


def expert_weights(e, H, I, dev):
    """Expert e's bf16 (w1 (2I, H) rows [gate; up], w2 (H, I)), the same on every rank."""
    g = torch.Generator(device=dev).manual_seed(1000 + e)
    w1 = torch.randn(2 * I, H, generator=g, device=dev, dtype=torch.bfloat16) * 0.02
    w2 = torch.randn(H, I, generator=g, device=dev, dtype=torch.bfloat16) * 0.02
    return w1, w2


def swiglu_cat(h):
    g, u = h.float().chunk(2, dim=-1)
    return (F.silu(g) * u).to(h.dtype)


def reference(x, topk_idx, scores, E, I):
    """out (T_local, H) fp32: every expert's bf16 matmul chain on its tokens, weighted by the scores."""
    T, H = x.shape
    out = torch.zeros(T, H, dtype=torch.float32, device=x.device)
    for e in range(E):
        t, k = (topk_idx == e).nonzero(as_tuple=True)
        if t.numel() == 0:
            continue
        w1, w2 = expert_weights(e, H, I, x.device)
        y = swiglu_cat(x[t] @ w1.T) @ w2.T
        out.index_add_(0, t, scores[t, k, None] * y.float())
    return out


def ref_check(kern, ref):
    """(ok, max rel err) of kern's fresh output vs the reference."""
    _run_synced(kern)
    got, want = kern.read_out(), kern.ref(ref)
    err = ((got - want).abs().max() / want.abs().max()).item()
    return err < 2 ** -6, err


def blocked_sf(scale):
    """(rows, sf_k) ue8m0 (rows % 128 == 0) -> quack's (rm, rk, 32, 4, 4) blocked scale atoms."""
    rows, sf_k = scale.shape
    rm, rk = rows // 128, -(-sf_k // 4)
    out = torch.zeros(rm, rk, 32, 4, 4, dtype=torch.uint8, device=scale.device)
    r = torch.arange(rows, device=scale.device)[:, None]
    k = torch.arange(sf_k, device=scale.device)[None, :]
    out[r // 128, k // 4, r % 32, (r // 32) % 4, k % 4] = scale.view(torch.uint8)
    return out.view(torch.float8_e8m0fnu)


def quantize_mx(w):
    """(..., K) -> MXFP8: e4m3 values, ue8m0 scales per 32 along K, and the fp32 dequantized values."""
    q, s = to_mx(w, 32)
    return q, s, q.float() * s.float().repeat_interleave(32, dim=-1)


def reference_mxfp8(x, topk_idx, scores, E, I):
    """reference() with MXFP8 operands: x, the weights and the SwiGLU postact quantized per 32 along their K (the
    postact along I), the GEMMs in fp32."""
    T, H = x.shape
    _, _, x_dq = quantize_mx(x)
    out = torch.zeros(T, H, dtype=torch.float32, device=x.device)
    for e in range(E):
        t, k = (topk_idx == e).nonzero(as_tuple=True)
        if t.numel() == 0:
            continue
        w1, w2 = expert_weights(e, H, I, x.device)
        _, _, w1_dq = quantize_mx(w1)
        _, _, w2_dq = quantize_mx(w2)
        g, u = (x_dq[t] @ w1_dq.T).chunk(2, dim=-1)
        _, _, p_dq = quantize_mx(F.silu(g) * u)
        out.index_add_(0, t, scores[t, k, None] * (p_dq @ w2_dq.T))
    return out


def dequant_megamoe(w, gran=32):
    """w quantized as DeepGEMM does (e4m3, ue8m0 = ceil(amax / 448) to a power of two, per `gran` along K) -> fp32."""
    from deep_gemm.utils import per_token_cast_to_fp8

    q, sf = per_token_cast_to_fp8(w, use_ue8m0=True, gran_k=gran)
    return q.float() * sf.repeat_interleave(gran, dim=-1)


def reference_megamoe_mxfp8(x, topk_idx, scores, E, I):
    """reference_mxfp8 with megamoe's fp8xfp8 rounding: DeepGEMM's ue8m0 rule, gate and up rounded to bf16 before the
    SwiGLU, the score applied before the postact quantization (its fc1 epilogue). Against to_mx's rounding megamoe
    sits at 4e-2; against this at 4e-3."""
    T, H = x.shape
    x_dq = dequant_megamoe(x)
    out = torch.zeros(T, H, dtype=torch.float32, device=x.device)
    for e in range(E):
        t, k = (topk_idx == e).nonzero(as_tuple=True)
        if t.numel() == 0:
            continue
        w1, w2 = expert_weights(e, H, I, x.device)
        g, u = (x_dq[t] @ dequant_megamoe(w1).T).to(torch.bfloat16).float().chunk(2, dim=-1)
        p = dequant_megamoe(F.silu(g) * u * scores[t, k, None])
        out.index_add_(0, t, p @ dequant_megamoe(w2).T)
    return out


def quantize_nvfp4(x, amax=None):
    """(..., K) bf16 -> NVFP4 with one per-tensor scale (quack's recipe: amax / 2688; amax = the tensor's unless given,
    e.g. x's over all ranks so every rank's rows share the scale the receiving GEMM applies): packed e2m1 (..., K/2) as
    float4_e2m1fn_x2, e4m3 scale rows (..., K/16) as uint8, the fp32 per-tensor scale, the fp32 dequantized values."""
    pts = nvfp4_per_tensor_scale(x.abs().amax() if amax is None else amax)
    x2 = x.reshape(-1, x.shape[-1])
    op = BlockScaledOperand.quantize(x2, "nvfp4", per_tensor_scale=pts)
    s = unpack_scale_blocked_to_2d(op.scale.unsqueeze(0), x2.shape[0], x2.shape[1] // 16)[0].view(torch.uint8)
    q = op.qdata.view(torch.uint8).reshape(*x.shape[:-1], x.shape[-1] // 2).view(torch.float4_e2m1fn_x2)
    return q, s.reshape(*x.shape[:-1], x.shape[-1] // 16), pts, op.dequantize(torch.float32).reshape(x.shape)


_compiled = {}
GEMM1_TILE = (256, 512) if SM100 else TILE  # GEMM1's K is 3.5x GEMM2's: the exposed epilogue is small, the wide tile wins


def compile_gemm1_gated(dtype, tile=GEMM1_TILE, cluster=CLUSTER, intra_pull=0, gather_A=True):
    """(fn, Cls, block_rows): quack's gated (SwiGLU) epilogue kernel class, minted by the same frontend gemm_gated uses
    (acc pairs over adjacent N, the postact tile half of N, no D store), compiled as bench_fused_dispatch_gemm's
    GEMM1: varlen M, gather A, the dispatch scheduler argument, static persistence. Launch:
    fn(A, B, None, None, Cls.EpilogueArguments(mAuxOut=postact), scheduler, varlen, None, None). intra_pull = the
    hidden size: the same epilogue on the dispatch-pull comm subclass (transfer warps pulling A's rows; the
    DispatchArguments bundle after the SF tensors, as bench_fused_dispatch_gemm's compile_gemm1). gather_A=False:
    plain TMA A loads over the same varlen rows."""
    key = (dtype, tile, cluster, intra_pull, gather_A)
    if key not in _compiled:
        sm = get_device_capacity(torch.device("cuda"))[0]
        assert sm >= 9, "the dispatch-scheduler gated GEMM needs quack's SM90 or SM100 kernel"
        mod = linear_act_mod("swiglu", gated=True, has_c=False, has_rowvec=False, has_colvec=False)
        Cls = mod._mint((), sm, True, False, (), RoundingMode.RN, (), False)
        kw = {}
        if intra_pull:  # the minted class rebased from quack's GemmSm100 onto the comm-surface copy, comm = the pull
            assert sm >= 10, "the intra-SM dispatch kernel is SM100 only"
            attrs = {k: v for k, v in vars(Cls).items() if k not in ("__dict__", "__weakref__")}
            Cls = type(Cls.__name__ + "_intra", (Cls.__bases__[0], GemmSm100Comm), attrs)
            kw = {"comm": DispatchPull(pull_bytes=intra_pull * dtype.width // 8)}
        gemm = Cls(cutlass.Float32, dtype, tile, (*cluster, 1), gather_A=gather_A, use_clc_persistence=False, **kw)
        gemm.b_transposed = gemm.a_transposed = gemm.cd_transposed = False  # as quack's compile_gemm_kernel sets
        gemm.cd_packed = None
        mA, mB, mD, mC, m, n, k, l = make_fake_gemm_tensors(dtype, dtype, None, None, "k", "k", "n", "n", varlen_m=True, gather_A=gather_A)
        aux = next(op for op in Cls._epi_ops if op.name == "mAuxOut")
        epi = Cls.EpilogueArguments(mAuxOut=aux.host_fake_arg((dtype, "n"), FakeArgCtx(m, n, k, l, True, True)))
        sched = make_fake_scheduler_args(False, False, l, has_dispatch=True)
        varlen = make_fake_varlen_args(True, False, True, m)  # mAIdx in both modes: the dispatch scheduler reads it
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        walk_fakes = (fake_dispatch_arguments(dtype),) if intra_pull else ()
        fn = cute.compile(gemm, mA, mB, mD, mC, epi, sched, varlen, stream, None, None, *walk_fakes, options="--enable-tvm-ffi")
        _compiled[key] = (fn, Cls, gemm.cta_tile_shape_mnk[0] * cluster[0])
    return _compiled[key]


def interleave_gate_up(w1_cat):
    """(E, 2I, H) [gate; up] -> gate/up rows interleaved per 8 (row 16q + i = gate[8q + i] for i < 8, else
    up[8q + i - 8]): the swap-AB gated epilogue pairs the rows r / r+8 of its 16-row accumulator groups."""
    E, I2, H = w1_cat.shape
    g, u = w1_cat.chunk(2, dim=1)
    return torch.stack([g.view(E, I2 // 16, 8, H), u.view(E, I2 // 16, 8, H)], dim=2).reshape(E, I2, H)


# the swap-AB kernels: 256 weights (2-CTA) x tile_n tokens (<= 256, varlen_n's one-instruction limit). tile_n = the
# token box (tile_n / 2 rows per CTA per stage) and the even-split block (the gate / flag / scale-slot unit), picked
# per launch by swap_tile_n (megamoe's block_m rule); the MMA-N sizes a tile may run at (quack varlen_n_variants;
# None = every 64) matter only without SWAP_DYNAMIC_N. The kernels recompile per tile_n.
SWAP_N_VARIANTS = (64, 112, 128, 144, 160)
SWAP_DYNAMIC_N = True  # quack varlen_n_dynamic_n: runtime UMMA N (16-granular) instead of the compiled variants


def swap_sm90_variants(tile_n):
    """SM90 (no runtime N): the compiled wgmma MMA-N set below the tile, or None = every tile at the full tile N
    (SWAP_FULL_TILE=1 in the environment: the A/B arm)."""
    if SWAP_N_VARIANTS is None or os.environ.get("SWAP_FULL_TILE"):
        return None
    return tuple(n for n in SWAP_N_VARIANTS if n < tile_n) or None


def swap_tile_n(model, W, T_local):
    """The swap kernels' token block: the fewest 256-row blocks covering the expected rows per expert (T_local * W *
    top-k / E), each sized to the expected rows plus their square root, rounded up to 32 (full blocks stay in the
    efficient MMA-N class; 16-odd N runs 5-8 % slower per row), in [64, 256]. W=4: 160 at 1024 tokens, 256 from 2048."""
    expected = T_local * W * model.topk / model.E
    blocks = math.ceil(expected / 256)
    return int(min(256, max(64, math.ceil((expected + expected**0.5) / blocks / 32) * 32)))


def compile_gemm1_gated_swap(
    dtype, model, W, T_local, intra_pull=0, gather_B=True, mx=False, activation="swiglu", gated=True, quant_postact=None
):
    """(fn, Cls, block_rows): the gated epilogue on the swap-AB kernel — weights on M (interleave_gate_up's rows,
    pairs over M), tokens on N (varlen_n even split), the dispatch scheduler, dynamic persistence (resident grid +
    atomic tile counter: CLC's arrival order without CLC's oversized grid, which would cancel transfer warps; static
    loses ~300 µs of L2 reuse, results/2026-10-05/ablations.md). gather_B: B rows by index over the dedup pool (TMA
    gather4; costs ~90 µs at W=4 1024, results/2026-10-05/hbm_slowdown.md); False: one plain TMA box over rows packed
    per expert, which on the intra kernel the transfer warps land per reader (intra_dispatch_scatter). Launch:
    fn(w1_il8, rows, None, None, Cls.EpilogueArguments(mAuxOut=postact.T), scheduler, varlen) with a zeroed int32
    semaphore in the scheduler arguments. block_rows = tile N, the unit the even-split blocks and their flags are cut
    from. intra_pull = the hidden size: the same on the intra-SM dispatch kernel (transfer warps pulling the walk;
    six walk tensors after the varlen argument, as compile_gemm1_gated). mx: MXFP8 operands (dtype e4m3, ue8m0 scales
    per 32: A's in the batched blocked layout, B's in per-block atoms the transfer warps fill, cu_tiles_n in the varlen
    argument) and the postact requantized along I in the epilogue (quack gated_quant_mod pair_dim "m": EpilogueArguments
    postact (fp8, m-major) + postact_sf (the per-block atoms GEMM2 reads as its B scales)); two scale-walk tensors follow
    the six walk tensors. activation / gated: quack's gate_fn_map name (gated, pairs over M) or act_fn_map name
    (gated=False: weights (I, H), the postact is act(acc) row for row; bench_new_kernels.py)."""
    tile_n = swap_tile_n(model, W, T_local)  # the token block is derived from the shape, never caller-chosen
    key = ("swap", dtype, intra_pull, gather_B, mx, activation, gated, tile_n, SWAP_N_VARIANTS, SWAP_DYNAMIC_N)
    if key not in _compiled:
        # SM90: 2 warp groups x 64 weight rows per CTA, the token box multicast over the cluster, no runtime N
        tile = (256, tile_n) if SM100 else (128, tile_n)
        nvfp4 = mx == "nvfp4"  # mx: False, True / "mxfp8" (e4m3, ue8m0 per 32), "nvfp4" (e2m1, ue4m3 per 16 + alpha)
        if quant_postact is None:  # the MXFP8 pipeline requantizes the postact for fc2; a bare fc1 stage keeps bf16
            quant_postact = bool(mx) and not nvfp4
        if mx:
            assert SM100 and SWAP_DYNAMIC_N and not gather_B, "mx: SM100, runtime N, packed rows"
            assert dtype == (cutlass.Float4E2M1FN if nvfp4 else cutlass.Float8E4M3FN), (dtype, mx)
        assert SM100 or not gather_B, "gather_B is SM100 only"
        if quant_postact:
            assert gated, "the quantized postact is the gated epilogue's"
            mod = gated_quant_mod(activation, "m", has_alpha=nvfp4)  # nvfp4: alpha = the operands' per-tensor scales
        else:
            mod = linear_act_mod(activation, gated=gated, has_c=False, has_rowvec=False, has_colvec=False, has_alpha=nvfp4)
        kind_sig = (("alpha", "scalar"),) if nvfp4 else ()
        Cls = mod._mint(kind_sig, 10 if SM100 else 9, gated, False, (), RoundingMode.RN, (), False, pair_dim="m" if gated else "n")
        kw = {"varlen_n_dynamic_n": True} if SWAP_DYNAMIC_N and SM100 else {}
        if mx:
            kw["sf_vec_size"] = 16 if nvfp4 else 32
        if intra_pull or not SM100:  # the minted class rebased from quack's GemmSm100 / GemmSm90 onto the intra dispatch kernel
            attrs = {k: v for k, v in vars(Cls).items() if k not in ("__dict__", "__weakref__")}
            Cls = type(Cls.__name__ + "_intra", (Cls.__bases__[0], GemmSm100IntraDispatch if SM100 else GemmSm90IntraDispatch), attrs)
            kw.update({"intra_dispatch_pull": intra_pull, "intra_dispatch_scatter": not gather_B})
        if SM100:
            kw["gather_B"] = gather_B
        variants = SWAP_N_VARIANTS if SM100 else swap_sm90_variants(tile_n)
        gemm = Cls(cutlass.Float32, dtype, tile, (*CLUSTER, 1), use_clc_persistence=False, varlen_n_variants=variants, **kw)
        gemm.b_transposed = gemm.a_transposed = gemm.cd_transposed = False
        gemm.cd_packed = None
        mA, mB, mD, mC, m, n, k, l = make_fake_gemm_tensors(dtype, dtype, None, None, "k", "k", "n", "n", varlen_n=True, gather_B=gather_B)
        ops = {op.name: op for op in Cls._epi_ops}
        fctx = FakeArgCtx(m, n, k, l, False, False)
        epi_kw = {"alpha": Float32(0)} if nvfp4 else {}
        if quant_postact:
            epi = Cls.EpilogueArguments(  # the postact stays mxfp8 (fp8 values, ue8m0 per 32 along I) under nvfp4 too
                postact=ops["postact"].host_fake_arg((cutlass.Float8E4M3FN, "m"), fctx),
                postact_sf=make_fake_sf_tensor(cutlass.Float8E8M0FNU, cute.sym_int()),  # per-block atoms: batch 1
                **epi_kw,
            )
        else:  # bf16 postact behind quantized operands
            aux_dtype = cutlass.BFloat16 if mx else dtype
            epi = Cls.EpilogueArguments(mAuxOut=ops["mAuxOut"].host_fake_arg((aux_dtype, "m"), fctx), **epi_kw)
        if mx:
            sf_dtype = cutlass.Float8E4M3FN if nvfp4 else cutlass.Float8E8M0FNU
            sf_fakes = (make_fake_sf_tensor(sf_dtype, l), make_fake_sf_tensor(sf_dtype, cute.sym_int()))
        else:
            sf_fakes = (None, None)
        sched = make_fake_scheduler_args(True, False, l, has_dispatch=True, has_dispatch_row_key=not gather_B)
        varlen = make_fake_varlen_args(False, False, False, None, varlen_n=True, gather_B=gather_B, has_cu_tiles_n=mx)
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        i64 = lambda: make_fake_tensor(Int64, (cute.sym_int(),), leading_dim=0, divisibility=1)
        # the walk (with the scale lane under mx), then the per-CTA tensormap slots (SM100)
        walk_fakes = (fake_walk_arguments(mx),) + ((i64(),) if SM100 else ()) if intra_pull else ()
        fn = cute.compile(gemm, mA, mB, mD, mC, epi, sched, varlen, stream, *sf_fakes, *walk_fakes, options="--enable-tvm-ffi")
        _compiled[key] = (fn, Cls, tile_n)
    return _compiled[key]


def compile_gemm2_swap(dtype, model, W, T_local, intra_push=0, mx=False):
    """(fn, block_rows, cluster_tile_m): the plain swap-AB kernel for GEMM2 — weights (H, I) on M, tokens on N
    (varlen_n even split), D = y.T (m-major), the combine scheduler (+1 per stored CTA tile, so a block's flag target
    is CLUSTER CTAs x ceil(H / cluster_tile_m)), dynamic persistence (zeroed semaphore in sched, as
    compile_gemm1_gated_swap). Launch: fn(w2, a, y.T, None, epi, sched, varlen).
    intra_push = the hidden size: the intra-SM combine kernel whose epilogue stores each token's slice of a subtile
    straight into the origins' slots (no y, no flags, no transfer warps); after the varlen argument five None
    placeholders, the slot base table (world,) int64, the slots per rank and the per-row slot addresses. mx: MXFP8
    operands (A's scales batched, B's = GEMM1's postact atoms, cu_tiles_n in the varlen argument)."""
    tile_n = swap_tile_n(model, W, T_local)  # the same block as GEMM1's (asserted at the call sites)
    key = ("swap2", dtype, intra_push, mx, tile_n, SWAP_N_VARIANTS, SWAP_DYNAMIC_N)
    if key not in _compiled:
        tile = (256, tile_n) if SM100 else (128, tile_n)  # as compile_gemm1_gated_swap
        dyn = {"varlen_n_dynamic_n": True} if SWAP_DYNAMIC_N and SM100 else {}
        if mx:
            assert SM100 and dtype == cutlass.Float8E4M3FN and SWAP_DYNAMIC_N, "mx: SM100, e4m3, runtime N"
            dyn["sf_vec_size"] = 32
        variants = SWAP_N_VARIANTS if SM100 else swap_sm90_variants(tile_n)
        i64 = lambda: make_fake_tensor(Int64, (cute.sym_int(),), leading_dim=0, divisibility=1)
        if intra_push:
            IntraCls = GemmDefaultSm100IntraCombine if SM100 else GemmDefaultSm90IntraCombine
            gemm = IntraCls(
                cutlass.Float32, dtype, tile, (*CLUSTER, 1), use_clc_persistence=False,
                varlen_n_variants=variants, combine_epilogue_store=intra_push, **dyn,
            )
            # slot base table, slots per rank, slot addresses, then the tensormap slots (SM100)
            walk_fakes = (i64(), Int32(0), i64()) + ((i64(),) if SM100 else ())
        elif SM100:
            gemm = GemmDefaultSm100(cutlass.Float32, dtype, tile, (*CLUSTER, 1), use_clc_persistence=False,
                                    varlen_n_variants=variants, **dyn)
            walk_fakes = ()
        else:  # quack's GemmSm90 has no varlen_n: the intra kernel without transfer warps or slot store
            gemm = GemmDefaultSm90IntraCombine(cutlass.Float32, dtype, tile, (*CLUSTER, 1), varlen_n_variants=variants)
            walk_fakes = ()
        gemm.b_transposed = gemm.a_transposed = gemm.cd_transposed = False
        gemm.cd_packed = None
        d_dtype = cutlass.BFloat16 if mx else dtype
        mA, mB, mD, mC, m, n, k, l = make_fake_gemm_tensors(dtype, dtype, d_dtype, None, "k", "k", "m", "n", varlen_n=True)
        epi = GemmDefaultEpiMixin.EpilogueArguments(add_to_output=False, rounding_mode=RoundingMode.RN)
        sched = make_fake_scheduler_args(True, False, l, has_combine=True)
        varlen = make_fake_varlen_args(False, False, False, None, varlen_n=True, has_cu_tiles_n=mx)
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        sf_fakes = (None, None)
        if mx:
            sf_fakes = (make_fake_sf_tensor(cutlass.Float8E8M0FNU, l), make_fake_sf_tensor(cutlass.Float8E8M0FNU, cute.sym_int()))
        fn = cute.compile(gemm, mA, mB, mD, mC, epi, sched, varlen, stream, *sf_fakes, *walk_fakes, options="--enable-tvm-ffi")
        _compiled[key] = (fn, tile_n, gemm.cta_tile_shape_mnk[0] * CLUSTER[0])
    return _compiled[key]


# Every factory: (model, x, topk_idx, scores, w1_cat, w1_il, w2, ws, rank, W, T_local, args) -> {arm: Kernel};
# w1_cat / w1_il (E_local, 2I, H), w2 (E_local, H, I); ws: sonic's symm-mem workspace (rank-dedup mode).
def _sonic(model, x, topk_idx, scores, w1_cat, w1_il, w2, ws, rank, W, T_local, args):
    K, E_local = model.topk, model.E // W
    node_size = ws.node_size if ws.node_size else 1
    cfg = RuntimeEPConfig(
        dispatch_mode=DispatchMode.RANK_DEDUP_DISPATCH_TRITON, W=W, K=K, combine_mode=CombineMode.RANK_DEDUP_COMBINE_TRITON,
        I=model.I, is_glu_act=True, E_local=E_local, MAX_ROWS_PER_RANK_STATIC=T_local * W * min(K, E_local),
        num_nodes=ws.num_nodes, node_size=node_size, node_id=rank // node_size, local_id=rank % node_size,
    )
    w1s, w2s = w1_il.permute(1, 2, 0), w2.permute(0, 2, 1)  # the (2I, H, E_local) / (E_local, I, H) views its forward takes
    out = {}

    def run():
        ws.x_symm.copy_(x)
        topk_idx_g = _ag_routing_decision(ws, topk_idx)
        ws.x_hdl.barrier()
        out["o"] = _moe_ep_forward_inner(x, topk_idx_g, scores, w1s, None, w2s, None, ws, cfg, ActivationType.SWIGLU, True, False, False)

    return {"sonic": Kernel(run, _NOOP, lambda: out["o"].float(), _NOOP, lambda ref: ref)}


def _deepep(model, x, topk_idx, scores, w1_cat, w1_il, w2, ws, rank, W, T_local, args, gemms):
    """DeepEP expanded dispatch -> gemms (setup returning (alignment, gemm1(recv_x, handle) -> h, gemm2(a, handle) -> y))
    -> TE swiglu -> the expanded scores onto the SwiGLU rows (torch mul, as DeepGEMM's baseline folds them into its
    tilelang swiglu) -> DeepEP combine (a plain sum at the origin: its topk_weights argument is only reduced, not applied)."""
    import deep_ep
    import transformer_engine.pytorch  # noqa: F401  (loads the torch extension)
    import transformer_engine_torch as tex

    H, I, E, K, dev = model.H, model.I, model.E, model.topk, x.device
    buffer = deep_ep.ElasticBuffer(ws.ep_group, num_max_tokens_per_rank=T_local, hidden=H, num_topk=K, explicitly_destroy=True)
    num_sms = buffer.get_theoretical_num_sms(E, K)
    idx64 = topk_idx.to(deep_ep.topk_idx_t)
    alignment, gemm1, gemm2 = gemms(dev)
    common = dict(num_sms=num_sms, num_max_tokens_per_rank=T_local, num_experts=E, expert_alignment=alignment)
    out = {}

    def run():
        recv_x, _, recv_w, handle, _ = buffer.dispatch(
            x=x, topk_idx=idx64, topk_weights=scores, do_expand=True, do_cpu_sync=False, do_handle_copy=False, **common
        )
        a = tex.swiglu(gemm1(recv_x, handle), None)
        a.mul_(recv_w[:, None])  # DeepEP's combine only sums: the scores go on the rows first (torch op)
        out["o"] = buffer.combine(gemm2(a, handle), handle=handle, num_sms=num_sms)[0]

    return Kernel(run, buffer.destroy, lambda: out["o"].float(), _NOOP, lambda ref: ref, label=f" (deepep_sms={num_sms})")


def _deepep_deepgemm(model, x, topk_idx, scores, w1_cat, w1_il, w2, ws, rank, W, T_local, args):
    import deep_gemm

    H, I = model.H, model.I

    def gemms(dev):
        alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()
        deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
        kw = dict(compiled_dims="", use_psum_layout=True)

        def gemm1(recv_x, handle):
            h = torch.empty(recv_x.shape[0], 2 * I, dtype=torch.bfloat16, device=dev)
            deep_gemm.m_grouped_bf16_gemm_nt_contiguous(recv_x, w1_cat, h, handle.psum_num_recv_tokens_per_expert, **kw)
            return h

        def gemm2(a, handle):
            y = torch.empty(a.shape[0], H, dtype=torch.bfloat16, device=dev)
            deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a, w2, y, handle.psum_num_recv_tokens_per_expert, **kw)
            return y

        return alignment, gemm1, gemm2

    return {"deepep_deepgemm": _deepep(model, x, topk_idx, scores, w1_cat, w1_il, w2, ws, rank, W, T_local, args, gemms)}


def _deepep_te(model, x, topk_idx, scores, w1_cat, w1_il, w2, ws, rank, W, T_local, args):
    import transformer_engine.pytorch as te

    H, I, E_local = model.H, model.I, model.E // W

    def gemms(dev):
        def grouped(w):  # GroupedLinear over the local experts, its parameters = our weight rows (no copy)
            # TE 2.19's grouped-tensor GroupedLinear (as MoK's bench_deepep_te.py): its grouped GEMM is not ordered
            # with the current stream on either side (tmp repro 2026-10-17); kept as the baseline's own path
            lin = te.GroupedLinear(E_local, w.shape[2], w.shape[1], bias=False, params_dtype=torch.bfloat16, device=dev, use_grouped_tensor=True)
            for e in range(E_local):
                getattr(lin, f"weight{e}").data = w[e]
            return lin

        lin1, lin2 = grouped(w1_cat), grouped(w2)

        def splits(handle):  # alignment 1: the psum is a plain inclusive prefix sum; device int64, no host sync
            psum = handle.psum_num_recv_tokens_per_expert
            return torch.diff(psum, prepend=psum.new_zeros(1)).to(torch.int64)

        def gemm1(recv_x, handle):
            h = torch.empty(recv_x.shape[0], 2 * I, dtype=torch.bfloat16, device=dev)
            lin1(recv_x, splits(handle), out=h)
            return h

        def gemm2(a, handle):
            y = torch.empty(a.shape[0], H, dtype=torch.bfloat16, device=dev)
            lin2(a, splits(handle), out=y)
            return y

        return 1, gemm1, gemm2

    return {"deepep_te": _deepep(model, x, topk_idx, scores, w1_cat, w1_il, w2, ws, rank, W, T_local, args, gemms)}


def _megamoe(model, x, topk_idx, scores, w1_cat, w1_il, w2, ws, rank, W, T_local, args):
    import deep_gemm

    H, I, E, K, dev = model.H, model.I, model.E, model.topk, x.device
    buf = deep_gemm.get_symm_buffer_for_mega_moe(ws.ep_group, E, T_local, K, H, I, mma_type="bf16xbf16")
    l1, l2 = deep_gemm.transform_weights_for_mega_moe(w1_cat, w2)
    idx64 = topk_idx.to(torch.int64)
    y = torch.empty(T_local, H, dtype=torch.bfloat16, device=dev)

    def run():
        buf.x[:T_local].copy_(x)
        buf.topk_idx[:T_local].copy_(idx64)
        buf.topk_weights[:T_local].copy_(scores)
        deep_gemm.bf16_mega_moe(y=y, l1_weights=l1, l2_weights=l2, sym_buffer=buf)

    return {"megamoe": Kernel(run, buf.destroy, y.float, y.zero_, lambda ref: ref)}


def megamoe_fp8_weights(w):
    """(E_local, N, K) bf16 -> DeepGEMM's grouped MXFP8 weights: (e4m3, scales in its UTCCP layout), per 32 along K."""
    import deep_gemm
    from deep_gemm.utils import per_token_cast_to_fp8

    E_local, n, k = w.shape
    q = torch.empty_like(w, dtype=torch.float8_e4m3fn)
    sf = torch.empty(E_local, n, k // 32, dtype=torch.float32, device=w.device)
    for e in range(E_local):
        q[e], sf[e] = per_token_cast_to_fp8(w[e], use_ue8m0=True, gran_k=32)
    return q, deep_gemm.transform_sf_into_required_layout(sf, n, k, (1, 32), E_local)


def megamoe_fp8_x(x):
    """x -> (e4m3, packed ue8m0 scales (T, H/128) int32) as megamoe's symmetric x / x_sf buffers take them."""
    from deep_gemm.utils import per_token_cast_to_fp8

    return per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)


def _megamoe_mxfp8(model, x, topk_idx, scores, w1_cat, w1_il, w2, ws, rank, W, T_local, args):
    import deep_gemm

    H, I, E, K, dev = model.H, model.I, model.E, model.topk, x.device
    buf = deep_gemm.get_symm_buffer_for_mega_moe(ws.ep_group, E, T_local, K, H, I, mma_type="fp8xfp8")
    l1, l2 = deep_gemm.transform_weights_for_mega_moe(megamoe_fp8_weights(w1_cat), megamoe_fp8_weights(w2))
    idx64 = topk_idx.to(torch.int64)
    xq, xs = megamoe_fp8_x(x)  # x arrives quantized; only the copy is timed
    y = torch.empty(T_local, H, dtype=torch.bfloat16, device=dev)

    def run():
        buf.x[:T_local].copy_(xq)
        buf.x_sf[:T_local].copy_(xs)
        buf.topk_idx[:T_local].copy_(idx64)
        buf.topk_weights[:T_local].copy_(scores)
        deep_gemm.fp8_fp4_mega_moe(y=y, l1_weights=l1, l2_weights=l2, sym_buffer=buf)

    ref = lambda ref_bf16: reference_megamoe_mxfp8(x, topk_idx, scores, E, I)
    return {"megamoe_mxfp8": Kernel(run, buf.destroy, y.float, y.zero_, ref)}


def _megamoe_cutedsl(model, x, topk_idx, scores, w1_cat, w1_il, w2, ws, rank, W, T_local, args):
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cutedsl_megamoe-main")
    if root not in sys.path:
        sys.path.insert(0, root)
    import numpy as np
    import nvshmem.core
    from moe_bf16_glu.epilogue_bf16 import Fc1GateUpInterleave as g
    from moe_bf16_glu.mega_runner import MegaMoEBf16Tester, _sym_zeros
    from moe_bf16_glu.runner_common import TrainingImplDesc
    from moe_nvfp4_swapab.mega_runner import TokenCommProblemDesc
    from moe_nvfp4_swapab.runner_fc12_common import MiscDesc

    try:
        from cuda.core.experimental import Device
    except ImportError:
        from cuda.core import Device
    H, I, E, K, dev = model.H, model.I, model.E, model.topk, x.device
    E_local = E // W
    # NVSHMEM once per process: src/bootstrap.py's UID broadcast, over our process group; the heap holds x, the
    # (T_local, K, H) cells and the kernel's shared workspace (NVSHMEM's default is 1 GiB)
    os.environ.setdefault("NVSHMEM_SYMMETRIC_SIZE", f"{T_local * H * 2 * (K + 2) // 2**20 + 1024}M")
    device = Device(dev.index)
    device.set_current()
    uid = nvshmem.core.get_unique_id(empty=(rank != 0))
    uid_t = torch.from_numpy(uid._data.view(np.uint8).copy()).to(dev)
    dist.broadcast(uid_t, src=0, group=ws.ep_group)
    dist.barrier()
    uid._data[:] = uid_t.cpu().numpy().view(uid._data.dtype)
    nvshmem.core.init(device=device, uid=uid, rank=rank, nranks=W, initializer_method="uid")
    # The runner's descriptors with its bf16 CLI defaults; form A (no in-kernel reduce), no pre-SwiGLU c output, the
    # combine write-back on dedicated warps (5.5 % faster than the default epilogue warps at W=4 8192).
    problem = TokenCommProblemDesc(world_size=W, num_tokens_per_rank=T_local, num_topk=K, num_total_experts=E, hidden=H, intermediate=2 * I)
    impl = TrainingImplDesc(in_kernel_fc2_reduce=False, generate_c=False, epi_flag_batch=(2, 4), flag_batch=1, token_back_mode="standalone_warps")
    tester = MegaMoEBf16Tester(problem, impl, MiscDesc(skip_ref_check=True), rank=rank)
    # Its inputs in place of generate_inputs': fc1 rows in the kernel's gate/up interleave (g gate rows, then g up
    # rows, per pair) and both weights as (experts, K dim, N) views with the K dim stride 1.
    tester.my_activation = _sym_zeros((T_local, H), torch.bfloat16)
    tester.my_topk_idx = _sym_zeros((T_local, K), torch.int64)
    tester.my_topk_weights = _sym_zeros((T_local, K), torch.float32)
    tester.combine_output = _sym_zeros((T_local, K, H), torch.bfloat16)
    w1_nv = w1_cat.view(E_local, 2, I // g, g, H).transpose(1, 2).reshape(E_local, 2 * I, H).contiguous()
    tester.my_fc1_weight = w1_nv.permute(0, 2, 1)
    tester.my_fc2_weight = w2.permute(0, 2, 1)
    launch = {}
    tester._launch_target_kernels_with_optional_torch_profiler = launch.update  # run_kernel compiles, we keep the launch args
    tester.run_kernel()
    idx64 = topk_idx.to(torch.int64)
    slots = tester.combine_output
    out = torch.empty(T_local, H, dtype=torch.bfloat16, device=dev)

    def run():
        tester.my_activation.copy_(x)
        tester.my_topk_idx.copy_(idx64)
        tester.my_topk_weights.copy_(scores)
        tester._compiled_kernel(**launch)
        reduce_over_k(slots, None, out)

    def free():
        for t in (tester.my_activation, tester.my_topk_idx, tester.my_topk_weights, slots, tester.shared_workspace):
            nvshmem.core.free_tensor(t)
        nvshmem.core.finalize()

    run()
    torch.cuda.synchronize()
    zero = lambda: (out.zero_(), slots.zero_())
    return {"megamoe_cutedsl": Kernel(run, free, out.float, zero, lambda ref: ref)}


def routed_rows(topk_idx, rank, W, E_local, group):
    """Rows this rank receives: every rank's (token, expert) pairs routed to its experts, rounded up to the even-split
    block. The routing bound W * T_local * min(K, E_local) is 8x this under uniform routing (55 GB at T_local 32768)."""
    topk_g = torch.empty(W, *topk_idx.shape, dtype=topk_idx.dtype, device=topk_idx.device)
    dist.all_gather_into_tensor(topk_g, topk_idx, group=group)
    n = int((topk_g // E_local == rank).sum())
    return -(-n // 512) * 512


def _quack_metadata(x, topk_idx, ws, rank, W, K, E_local, max_rows, sonic_metadata=False):
    """The metadata the quack walks and GEMMs read, in fixed buffers: (topk_g, meta, prologue) with meta =
    routing_metadata's tables (cu, s_reverse, s_scatter, a_idx, first_src_row); prologue() = x publish, top-k
    all-gather, barrier, then the two metadata kernels (dispatch/routing_metadata.py). sonic_metadata: the same
    tables from sonic's compute_dispatch_metadata + _build_consumer_metadata + build_rank_dedup_a_idx (12 launches)
    instead, for the A/B."""
    T_local, dev = x.shape[0], x.device
    TK = W * T_local * K
    topk_g = torch.empty(W, T_local, K, dtype=torch.int32, device=dev)
    if sonic_metadata:
        i32 = lambda *shape: torch.empty(*shape, dtype=torch.int32, device=dev)
        cu, s_scatter, pair_offset = i32(E_local + 1), i32(TK), i32(W, W)
        meta = SimpleNamespace(
            cu=cu, s_reverse=ws.s_rev_symm[:TK], s_scatter=s_scatter[:max_rows],
            a_idx=ws.x_idx_expanded_remap_for_rank_dedup_buf[:max_rows], first_src_row=pair_offset[(rank + 1) % W, rank : rank + 1],
        )

        def build():
            m = compute_dispatch_metadata(topk_g, my_rank=rank, E_local=E_local, emit_combine=False)
            consumer = _build_consumer_metadata(
                expert_indices=m["expert_local_padded"], token_indices=m["a2a_token_indices"], TK=TK, E_local=E_local,
                s_reverse_idx_symm=ws.s_rev_symm,
            )
            build_rank_dedup_a_idx(
                dst_rank_flat=m["dst_rank_flat"], s_reverse_local=consumer["s_reverse_local"],
                rank_dedup_recv_pos=m["rank_dedup_recv_pos"], my_rank=rank, out=ws.x_idx_expanded_remap_for_rank_dedup_buf,
            )
            cu.copy_(consumer["expert_frequency_offset"])
            s_scatter.copy_(consumer["s_scatter_idx"])
            pair_offset.copy_(m["pair_offset"])
    else:
        build, meta = make_routing_metadata(topk_g.view(-1), rank, W, T_local, K, E_local, max_rows)

    def prologue():
        ws.x_symm.copy_(x)
        dist.all_gather_into_tensor(topk_g.view(-1, K), topk_idx, group=ws.ep_group)
        ws.x_hdl.barrier()
        build()

    return topk_g, meta, prologue


def _quack_inter(model, x, topk_idx, scores, w1_cat, w1_il, w2, ws, rank, W, T_local, args):
    H, I, K, E_local, dev = model.H, model.I, model.topk, model.E // W, x.device
    dtype = torch2cute_dtype_map[x.dtype]
    TK, max_rows = W * T_local * K, routed_rows(topk_idx, rank, W, E_local, ws.ep_group)
    comm = SMStream(dev.index, args.comm_sms)
    assert comm.stream is not None, "--comm-sms must leave SMs for the GEMMs"
    rest = comm.complement()
    topk_g, meta, prologue = _quack_metadata(x, topk_idx, ws, rank, W, K, E_local, max_rows, args.sonic_metadata)
    cu, s_scatter, recv_pos, a_idx = meta.cu, meta.s_scatter, meta.s_reverse, meta.a_idx
    # fused dispatch + gated GEMM1: pool (W * T_local, H) <- peers' x rows, a (rows, I) = SwiGLU(pool[a_idx] @ w1)
    fn1, Cls1, block_rows = compile_gemm1_gated(dtype)
    build1, walk1 = make_dispatch_walk(topk_g, s_scatter, recv_pos, meta.first_src_row, rank, a_idx, cu, ws.x_peer_bufs, block_rows)
    run_pull, pool = make_dispatch_pull(ws.x_peer_bufs, rank, "inter", comm.sms)
    pull = run_pull(walk1, zero_counters=False)
    a = torch.empty(max_rows, I, dtype=torch.bfloat16, device=dev)
    sched1 = make_scheduler_args(rest.sms // (CLUSTER[0] * CLUSTER[1]), 8, None, dispatch_args=(walk1.flags, walk1.first_src_row))
    rt1 = (Cls1.EpilogueArguments(mAuxOut=a), sched1, make_varlen_args(cu, None, a_idx))
    gemm1 = lambda: fn1(pool, w1_il, None, None, *rt1, None, None)
    # fused GEMM2 + combine: y (rows, H) = a @ w2, pushed per M cluster into the origins' slots
    fn2, block_rows2, tile_n = compile_gemm2(dtype)
    assert block_rows2 == block_rows
    y = torch.empty(max_rows, H, dtype=torch.bfloat16, device=dev)
    flags2 = torch.zeros(E_local, -(-W * T_local // block_rows), dtype=torch.int32, device=dev)  # an expert's rows <= W * T_local
    flag_target = CLUSTER[0] * CLUSTER[1] * -(-H // tile_n)
    slot_of_row = s_scatter
    first_src_row2 = torch.tensor([((rank + 1) % W) * TK // W], dtype=torch.int32, device=dev)
    buffers = make_combine_buffers(T_local, K, H, torch.bfloat16, ws.ep_group)
    build2, walk2 = make_combine_walk_device(slot_of_row, cu, flags2, first_src_row2, buffers, T_local, K, tile_n=tile_n)
    push = make_combine_push(y, "inter", comm.sms)(walk2, flag_target)
    rt2 = gemm2_args(cu, flags2, slot_of_row, first_src_row2, sms=rest.sms)
    gemm2 = lambda: fn2(a, w2, y, None, *rt2, None, None)
    out, reduce = _gather_and_sum(buffers, scores, T_local, K, H)

    def fork(build, prologue, left, right):
        cur = torch.cuda.current_stream()
        build()
        prologue()
        comm.stream.wait_stream(cur)
        rest.stream.wait_stream(cur)
        with comm.ctx():
            left()
        with rest.ctx():
            right()
        cur.wait_stream(comm.stream)
        cur.wait_stream(rest.stream)

    steps = (
        prologue, lambda: fork(build1, walk1.flags.zero_, pull, gemm1), lambda: fork(build2, flags2.zero_, push, gemm2),
        lambda: ws.o_hdl.barrier(channel=1), reduce,
    )

    def run():
        for step in steps:
            step()

    run.steps, run.step_names = steps, ("prologue", "dispatch+gemm1", "gemm2+combine", "barrier", "reduce")  # for capture bisection
    # every kernel once on one stream before any fork: a first (module-loading) launch inside the fork stalls
    prologue()
    for step in (build1, walk1.flags.zero_, pull, gemm1, build2, flags2.zero_, gemm2, push):
        step()
    ws.o_hdl.barrier(channel=1)
    reduce()

    zero = lambda: (out.zero_(), buffers.slots.zero_())
    label = f" (comm {comm.sms}, gemm {rest.sms} SMs)"
    return {"quack_inter": Kernel(run, _NOOP, out.float, zero, lambda ref: ref, label=label)}


def _quack_inter_swap(model, x, topk_idx, scores, w1_cat, w1_il, w2, ws, rank, W, T_local, args):
    """quack_inter's stages on the swap-AB kernels (compile_gemm1_gated_swap / compile_gemm2_swap): weights on M,
    tokens on N in varlen_n's even-split blocks, GEMM1 gathering the dedup pool through B_idx (= a_idx); the walks
    and flags follow the same blocks (even_split)."""
    H, I, K, E_local, dev = model.H, model.I, model.topk, model.E // W, x.device
    dtype = torch2cute_dtype_map[x.dtype]
    TK, max_rows = W * T_local * K, routed_rows(topk_idx, rank, W, E_local, ws.ep_group)
    comm = SMStream(dev.index, args.comm_sms)
    assert comm.stream is not None, "--comm-sms must leave SMs for the GEMMs"
    rest = comm.complement()
    topk_g, meta, prologue = _quack_metadata(x, topk_idx, ws, rank, W, K, E_local, max_rows, args.sonic_metadata)
    cu, s_scatter, recv_pos, a_idx = meta.cu, meta.s_scatter, meta.s_reverse, meta.a_idx
    w1_il8 = interleave_gate_up(w1_cat)
    fn1, Cls1, block_rows = compile_gemm1_gated_swap(dtype, model, W, T_local)
    build1, walk1 = make_dispatch_walk(
        topk_g, s_scatter, recv_pos, meta.first_src_row, rank, a_idx, cu, ws.x_peer_bufs, block_rows, even_split=True
    )
    run_pull, pool = make_dispatch_pull(ws.x_peer_bufs, rank, "inter", comm.sms)
    pull = run_pull(walk1, zero_counters=False)
    a = torch.empty(max_rows, I, dtype=torch.bfloat16, device=dev)
    clusters = rest.sms // (CLUSTER[0] * CLUSTER[1])
    sem1 = torch.zeros(1, dtype=torch.int32, device=dev)  # dynamic persistence's tile counter, zeroed per launch
    sched1 = make_scheduler_args(clusters, 8, sem1, dispatch_args=(walk1.flags, walk1.first_src_row))
    rt1 = (Cls1.EpilogueArguments(mAuxOut=a.T), sched1, make_varlen_args(None, None, None, cu_seqlens_n=cu, B_idx=a_idx))
    gemm1 = lambda: fn1(w1_il8, pool, None, None, *rt1, None, None)
    # GEMM2 + combine: y (rows, H) = a @ w2 as D = y.T, pushed per N block into the origins' slots
    fn2, block_rows2, cluster_tile_m = compile_gemm2_swap(dtype, model, W, T_local)
    assert block_rows2 == block_rows
    y = torch.empty(max_rows, H, dtype=torch.bfloat16, device=dev)
    flags2 = torch.zeros(E_local, -(-W * T_local // block_rows), dtype=torch.int32, device=dev)
    flag_target = CLUSTER[0] * CLUSTER[1] * -(-H // cluster_tile_m)  # CTA tiles along M per block
    slot_of_row = s_scatter
    first_src_row2 = torch.tensor([((rank + 1) % W) * TK // W], dtype=torch.int32, device=dev)
    buffers = make_combine_buffers(T_local, K, H, torch.bfloat16, ws.ep_group)
    build2, walk2 = make_combine_walk_device(
        slot_of_row, cu, flags2, first_src_row2, buffers, T_local, K, tile_n=cluster_tile_m, even_split=True, block_rows=block_rows2
    )
    push = make_combine_push(y, "inter", comm.sms)(walk2, flag_target)
    epi2 = GemmDefaultEpiMixin.EpilogueArguments(add_to_output=None, rounding_mode=None)
    sem2 = torch.zeros(1, dtype=torch.int32, device=dev)
    sched2 = make_scheduler_args(clusters, 8, sem2, combine_args=(flags2, slot_of_row, first_src_row2))
    rt2 = (epi2, sched2, make_varlen_args(None, None, None, cu_seqlens_n=cu))
    gemm2 = lambda: fn2(w2, a, y.T, None, *rt2, None, None)
    out, reduce = _gather_and_sum(buffers, scores, T_local, K, H)

    def fork(build, prologue, left, right):
        cur = torch.cuda.current_stream()
        build()
        prologue()
        comm.stream.wait_stream(cur)
        rest.stream.wait_stream(cur)
        with comm.ctx():
            left()
        with rest.ctx():
            right()
        cur.wait_stream(comm.stream)
        cur.wait_stream(rest.stream)

    reset1 = lambda: (walk1.flags.zero_(), sem1.zero_())
    reset2 = lambda: (flags2.zero_(), sem2.zero_())
    steps = (
        prologue, lambda: fork(build1, reset1, pull, gemm1), lambda: fork(build2, reset2, push, gemm2),
        lambda: ws.o_hdl.barrier(channel=1), reduce,
    )

    def run():
        for step in steps:
            step()

    run.steps, run.step_names = steps, ("prologue", "dispatch+gemm1", "gemm2+combine", "barrier", "reduce")
    run.parts = SimpleNamespace(  # bench_stages.py's bounds: the walks, operands and flags the standalone transports take
        walk1=walk1, pool=pool, walk2=walk2, y=y, flags2=flags2, flag_target=flag_target, a=a, cu=cu, slot_of_row=slot_of_row,
        first_src_row2=first_src_row2, build1=build1, build2=build2,
    )
    prologue()  # every kernel once on one stream before any fork (module loading)
    for step in (build1, reset1, pull, gemm1, build2, reset2, gemm2, push):
        step()
    ws.o_hdl.barrier(channel=1)
    reduce()

    zero = lambda: (out.zero_(), buffers.slots.zero_())
    label = f" (comm {comm.sms}, gemm {rest.sms} SMs)"
    return {"quack_inter_swap": Kernel(run, _NOOP, out.float, zero, lambda ref: ref, label=label)}


def _quack_intra_swap(model, x, topk_idx, scores, w1_cat, w1_il, w2, ws, rank, W, T_local, args):
    """quack_inter_swap's stages with the intra-SM kernels on all SMs (quack_intra's structure): one prologue launch
    (dispatch/prologue.py: top-k exchange + barrier, resets, routing tables, the dispatch walk); the swap-AB gated
    GEMM1 whose CTAs carry 4 transfer warps pulling the walk (each row once over NVLink) and landing it in every local
    reader's row of the expert-packed B operand, read with one plain TMA box (no gather); then the swap-AB GEMM2 whose
    epilogue stores each token's slice straight into the origins' combine slots (megamoe's combine: no y, no flags,
    no push warps); even-split blocks, one stream, no partitions."""
    H, I, K, E_local, dev = model.H, model.I, model.topk, model.E // W, x.device
    dtype = torch2cute_dtype_map[x.dtype]
    TK, max_rows = W * T_local * K, routed_rows(topk_idx, rank, W, E_local, ws.ep_group)
    clusters = get_max_active_clusters(CLUSTER[0] * CLUSTER[1], get_device_capacity(dev))
    w1_il8 = interleave_gate_up(w1_cat)
    fn1, Cls1, block_rows = compile_gemm1_gated_swap(dtype, model, W, T_local, intra_pull=H, gather_B=False)
    fn2, block_rows2, cluster_tile_m = compile_gemm2_swap(dtype, model, W, T_local, intra_push=H)
    assert block_rows2 == block_rows
    buffers = make_combine_buffers(T_local, K, H, torch.bfloat16, ws.ep_group)
    first_src_rank = (rank + 1) % W if getattr(args, "src_order", "ring") == "ring" else 0  # --src-order
    prologue, meta, walk1 = make_prologue(
        topk_idx, ws.x_peer_bufs, rank, W, T_local, K, E_local, max_rows, block_rows, buffers.slot_base, ws.ep_group,
        first_src_rank=first_src_rank,
    )
    cu, s_scatter, a_idx = meta.cu, meta.s_scatter, meta.a_idx
    rows = torch.empty(max_rows, H, dtype=torch.bfloat16, device=dev)  # B rows packed per expert (cu order)
    a = torch.empty(max_rows, I, dtype=torch.bfloat16, device=dev)
    sched1 = make_scheduler_args(clusters, 8, prologue.sem1, dispatch_args=(walk1.flags, walk1.first_src_row, a_idx))
    rt1 = (Cls1.EpilogueArguments(mAuxOut=a.T), sched1, make_varlen_args(None, None, None, cu_seqlens_n=cu))
    # per-CTA copies of the weight tensormaps (the kernels patch L2 promotion to 256B): 128 B slots
    desc = lambda: torch.empty(torch.cuda.get_device_properties(dev).multi_processor_count * 16, dtype=torch.int64, device=dev)
    desc1, desc2 = ((desc(),), (desc(),)) if SM100 else ((), ())  # SM90 patches no tensormaps
    gemm1 = lambda: fn1(w1_il8, rows, None, None, *rt1, None, None, walk1.args, *desc1)
    y = torch.empty(max_rows, H, dtype=torch.bfloat16, device=dev)  # D's shape for the kernel; never written
    flags2 = torch.zeros(E_local, -(-W * T_local // block_rows), dtype=torch.int32, device=dev)  # the scheduler's; never published
    first_src_row2 = torch.tensor([first_src_rank * TK // W], dtype=torch.int32, device=dev)
    epi2 = GemmDefaultEpiMixin.EpilogueArguments(add_to_output=None, rounding_mode=None)
    sched2 = make_scheduler_args(clusters, 8, prologue.sem2, combine_args=(flags2, s_scatter, first_src_row2))
    rt2 = (epi2, sched2, make_varlen_args(None, None, None, cu_seqlens_n=cu))
    gemm2 = lambda: fn2(w2, a, y.T, None, *rt2, None, None, buffers.slot_base, T_local * K, meta.slot_addr, *desc2)
    out, reduce = _gather_and_sum(buffers, scores, T_local, K, H)

    publish = lambda: ws.x_symm.copy_(x)  # before the prologue on the stream: its NVLink barrier orders it for the peers
    steps = (publish, prologue, gemm1, gemm2, lambda: ws.o_hdl.barrier(channel=1), reduce)

    def run():
        for step in steps:
            step()

    run.steps = steps
    run.step_names = ("x publish", "prologue", "gemm1", "gemm2", "barrier", "reduce")
    run.parts = SimpleNamespace(  # bench_stages.py's bounds: the landed rows, the postact and the GEMMs' arguments
        rows=rows, a=a, cu=cu, a_idx=a_idx, w1_il8=w1_il8, walk1=walk1, sem1=prologue.sem1, sem2=prologue.sem2, rt1=rt1,
        clusters=clusters, reduce=reduce, meta=meta, flags2=flags2, first_src_row2=first_src_row2, y=y,
        slots=buffers.slots, out=out,  # the reduce's operands (bench_stages' reduce + RMSNorm arm)
    )
    run()  # every kernel once (module loading) before the timed calls
    zero = lambda: (out.zero_(), buffers.slots.zero_())
    return {"quack_intra_swap": Kernel(run, _NOOP, out.float, zero, lambda ref: ref)}


def _quack_intra_swap_mxfp8(model, x, topk_idx, scores, w1_cat, w1_il, w2, ws, rank, W, T_local, args):
    """quack_intra_swap with MXFP8 operands end to end: x prequantized at setup, the publish copies it (fp8 rows +
    per-32 ue8m0 scale rows into symmetric buffers), the prologue's walk carries each reader's scale row, GEMM1's transfer warps land fp8 rows
    and scatter the scales into B's per-block atoms, its epilogue requantizes the SwiGLU postact along I into fp8 +
    atoms (quack gated_quant_mod pair_dim "m"), GEMM2 consumes them as its block-scaled B; weights quantized once.
    Checked against reference_mxfp8."""
    H, I, K, E_local, dev = model.H, model.I, model.topk, model.E // W, x.device
    TK, max_rows = W * T_local * K, routed_rows(topk_idx, rank, W, E_local, ws.ep_group)
    clusters = get_max_active_clusters(CLUSTER[0] * CLUSTER[1], get_device_capacity(dev))
    fp8, e8m0 = cutlass.Float8E4M3FN, torch.float8_e8m0fnu
    fn1, Cls1, block_rows = compile_gemm1_gated_swap(fp8, model, W, T_local, intra_pull=H, gather_B=False, mx=True)
    fn2, block_rows2, cluster_tile_m = compile_gemm2_swap(fp8, model, W, T_local, intra_push=H, mx=True)
    assert block_rows2 == block_rows
    # weights: gate/up rows interleaved per 8 (values and scale rows alike); blocked scale atoms per expert
    w1_q, w1_s, _ = quantize_mx(w1_cat)
    w1_q_il = interleave_gate_up(w1_q)
    w1_s_il = interleave_gate_up(w1_s.view(torch.uint8)).view(e8m0)
    sfa1 = torch.stack([blocked_sf(w1_s_il[e]).view(torch.uint8) for e in range(E_local)]).view(e8m0)
    w2_q, w2_s, _ = quantize_mx(w2)
    sfa2 = torch.stack([blocked_sf(w2_s[e]).view(torch.uint8) for e in range(E_local)]).view(e8m0)
    # x's fp8 rows and scale rows, symmetric: the peers' pulls read both
    xq_symm = symm_mem.empty(T_local, H, dtype=torch.float8_e4m3fn, device=dev)
    xs_symm = symm_mem.empty(T_local, H // 32, dtype=torch.uint8, device=dev)
    hq, hs = symm_mem.rendezvous(xq_symm, group=ws.ep_group), symm_mem.rendezvous(xs_symm, group=ws.ep_group)
    xq_peers = [hq.get_buffer(q, (T_local, H), torch.float8_e4m3fn) for q in range(W)]
    xs_peers = [hs.get_buffer(q, (T_local, H // 32), torch.uint8) for q in range(W)]
    buffers = make_combine_buffers(T_local, K, H, torch.bfloat16, ws.ep_group)
    prologue, meta, walk1 = make_prologue(
        topk_idx, xq_peers, rank, W, T_local, K, E_local, max_rows, block_rows, buffers.slot_base, ws.ep_group,
        x_sf_peer_bufs=xs_peers, slot_row_bytes=H * buffers.slots.element_size(),  # bf16 slots behind fp8 x rows
    )
    cu, s_scatter, a_idx = meta.cu, meta.s_scatter, meta.a_idx
    atoms = walk1.sf_atom_rows // 128
    rows = torch.empty(max_rows, H, dtype=torch.float8_e4m3fn, device=dev)  # B rows packed per expert (cu order)
    sf1 = torch.zeros(walk1.flags.numel() * atoms, H // 128, 32, 4, 4, dtype=e8m0, device=dev)  # B's per-block atoms
    a = torch.empty(max_rows, I, dtype=torch.float8_e4m3fn, device=dev)  # the postact, GEMM2's B
    sf_a = torch.zeros(1, walk1.flags.numel() * atoms, I // 128, 32, 4, 4, dtype=e8m0, device=dev)  # its atoms
    sched1 = make_scheduler_args(clusters, 8, prologue.sem1, dispatch_args=(walk1.flags, walk1.first_src_row, a_idx))
    varlen = make_varlen_args(None, None, None, cu_seqlens_n=cu, cu_tiles_n=walk1.cu_tiles)
    rt1 = (Cls1.EpilogueArguments(postact=a.T, postact_sf=sf_a), sched1, varlen)
    desc = lambda: torch.empty(torch.cuda.get_device_properties(dev).multi_processor_count * 16, dtype=torch.int64, device=dev)
    desc1, desc2 = desc(), desc()
    gemm1 = lambda: fn1(w1_q_il, rows, None, None, *rt1, sfa1, sf1[None], walk1.args, desc1)
    y = torch.empty(max_rows, H, dtype=torch.bfloat16, device=dev)  # D's shape for the kernel; never written
    flags2 = torch.zeros(E_local, -(-W * T_local // block_rows), dtype=torch.int32, device=dev)  # the scheduler's; never published
    first_src_row2 = torch.tensor([((rank + 1) % W) * TK // W], dtype=torch.int32, device=dev)
    epi2 = GemmDefaultEpiMixin.EpilogueArguments(add_to_output=None, rounding_mode=None)
    sched2 = make_scheduler_args(clusters, 8, prologue.sem2, combine_args=(flags2, s_scatter, first_src_row2))
    gemm2 = lambda: fn2(
        w2_q, a, y.T, None, epi2, sched2, varlen, sfa2, sf_a, buffers.slot_base, T_local * K, meta.slot_addr, desc2,
    )
    out, reduce = _gather_and_sum(buffers, scores, T_local, K, H)

    xq, xs = to_mx(x, 32)  # x arrives quantized (the previous layer's epilogue); only the copy is timed

    def publish():  # before the prologue on the stream (its NVLink barrier orders it)
        xq_symm.copy_(xq)
        xs_symm.copy_(xs.view(torch.uint8))

    steps = (publish, prologue, gemm1, gemm2, lambda: ws.o_hdl.barrier(channel=1), reduce)

    def run():
        for step in steps:
            step()

    run.steps = steps
    run.step_names = ("x publish", "prologue", "gemm1", "gemm2", "barrier", "reduce")
    run.parts = SimpleNamespace(  # the stages' tensors for stage checks
        rows=rows, sf1=sf1, a=a, sf_a=sf_a, walk1=walk1, meta=meta, cu=cu, xq=xq, xs=xs, xq_symm=xq_symm, xs_symm=xs_symm,
        w1_q_il=w1_q_il, w1_q=w1_q, w1_s=w1_s, w2_q=w2_q, w2_s=w2_s, sfa1=sfa1, sfa2=sfa2, buffers=buffers, out=out,
        clusters=clusters, a_idx=a_idx, sem2=prologue.sem2, flags2=flags2, first_src_row2=first_src_row2, y=y, reduce=reduce,
    )
    run()  # every kernel once (module loading) before the timed calls
    zero = lambda: (out.zero_(), buffers.slots.zero_())
    ref = lambda ref_bf16: reference_mxfp8(x, topk_idx, scores, model.E, I)  # its own quantized reference
    return {"quack_intra_swap_mxfp8": Kernel(run, _NOOP, out.float, zero, ref)}


BASELINES = {
    "sonic": _sonic, "deepep_deepgemm": _deepep_deepgemm, "deepep_te": _deepep_te, "megamoe": _megamoe,
    "megamoe_mxfp8": _megamoe_mxfp8, "megamoe_cutedsl": _megamoe_cutedsl, "quack_inter": _quack_inter, "quack_inter_swap": _quack_inter_swap,
    "quack_intra_swap": _quack_intra_swap, "quack_intra_swap_mxfp8": _quack_intra_swap_mxfp8,
}
# --graph: every arm's call is captured into a CUDA graph (all arms warm up eagerly first, then the captures) and the
# replays are timed; the eager Kernel records stay referenced so the captured launches' tensors outlive the capture.
GRAPH_ARMS = (
    "quack_inter", "quack_inter_swap", "quack_intra_swap", "quack_intra_swap_mxfp8", "megamoe",
    "megamoe_mxfp8", "sonic",
)
ABLATION_ARMS = set()  # timing-only arms (numerics invalid, checks skipped); none at present


def main():
    p = argparse.ArgumentParser()
    p.add_argument("baselines", nargs="+", choices=list(BASELINES))
    p.add_argument("--model", choices=list(MODEL_CONFIGS), default="dsv3", help="shape: H, I, E, top-k (MODEL_CONFIGS)")
    p.add_argument("--t-local", type=int, default=8192)
    p.add_argument("--comm-sms", type=int, default=16, help="quack_inter: SMs of the pull / push partition; the GEMMs get the rest")
    p.add_argument("--ref_check", action="store_true", help="each arm's fresh output vs the reference (see the header)")
    p.add_argument("--r2r_check", action="store_true", help="run each arm twice; bit-identity of the two outputs")
    p.add_argument("--no-canary", action="store_true", help="skip the sonic contention canary (memory: one arm alone beside a co-tenant)")
    p.add_argument("--sonic-metadata", action="store_true", help="quack arms: sonic's metadata pipeline instead of routing_metadata.py (A/B)")
    p.add_argument("--src-order", choices=("ring", "ascending"), default="ring", help="quack_intra_swap: the source rank each expert's pull / store walk starts at: rank+1 (ring) or 0")
    p.add_argument("--graph", action="store_true", help="capture every arm into a CUDA graph (warm-ups first, then all captures); the replays are timed")
    args = p.parse_args()

    rank, W = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    dev = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(dev)
    dist.init_process_group("nccl", device_id=dev)
    model = MODEL_CONFIGS[args.model]
    H, I, E, topk = model
    assert E % W == 0, f"{args.model}: E={E} not divisible by W={W}"
    T_local, E_local = args.t_local, E // W
    ws = SymmMemManager(None, dev)._get_or_alloc(T_local, H, topk, E_local, torch.bfloat16, DispatchMode.RANK_DEDUP_DISPATCH_TRITON)
    g = torch.Generator(device=dev).manual_seed(123 + rank)
    x = torch.randn(T_local, H, generator=g, device=dev, dtype=torch.bfloat16)
    topk_idx = torch.rand(T_local, E, generator=g, device=dev).topk(topk, dim=-1).indices.to(torch.int32)
    scores = torch.rand(T_local, topk, generator=g, device=dev).softmax(-1)  # fp32, as sonic's router emits
    local = [expert_weights(e, H, I, dev) for e in range(rank * E_local, (rank + 1) * E_local)]
    w1_cat = torch.stack([w1 for w1, _ in local])  # (E_local, 2I, H) rows [gate; up]
    w2 = torch.stack([w2 for _, w2 in local])  # (E_local, H, I)
    del local
    w1_il = (  # row 2j = gate j, 2j+1 = up j; the quack arms and sonic read it (7.5 GB at dsv3 W=2: skipped otherwise)
        w1_cat.view(E_local, 2, I, H).transpose(1, 2).reshape(E_local, 2 * I, H).contiguous()
        if not args.no_canary or any(b.startswith("quack") or b == "sonic" for b in args.baselines) else None
    )
    # This rank's expert rows (the report's FLOP count): the metadata once, outside the timed region.
    topk_idx_g = _ag_routing_decision(ws, topk_idx)
    meta = compute_dispatch_metadata(topk_idx_g, my_rank=rank, E_local=E_local, emit_combine=False)
    consumer = _build_consumer_metadata(
        expert_indices=meta["expert_local_padded"], token_indices=meta["a2a_token_indices"], TK=W * T_local * topk,
        E_local=E_local, s_reverse_idx_symm=ws.s_rev_symm,
    )
    total_m = int(consumer["expert_frequency_offset"][E_local])
    del topk_idx_g, meta, consumer
    ref = reference(x, topk_idx, scores, E, I) if args.ref_check else None
    shared = (model, x, topk_idx, scores, w1_cat, w1_il, w2, ws, rank, W, T_local, args)

    with torch.no_grad():
        canary = BASELINES["sonic"](*shared) if "sonic" in args.baselines or not args.no_canary else None
        kerns = {}
        for name in args.baselines:
            kerns.update(canary if name == "sonic" else BASELINES[name](*shared))
        if "sonic" not in kerns and canary is not None:
            kerns["sonic"] = canary["sonic"]  # timed next to the baselines: the contention canary
        del shared, w1_cat  # freed unless a DeepEP arm reads it (the Mega MoE arms hold their own copies)
        if args.graph:
            graphed = [name for name in GRAPH_ARMS if name in kerns]
            assert set(kerns) <= set(graphed), f"--graph needs every arm captured, not {set(kerns) - set(graphed)}"
            for name in graphed:  # every arm's eager warm-ups first (compiles, autotunes, lazy symm-mem allocations)
                for _ in range(3):
                    kerns[name].run()
            torch.cuda.synchronize()
            dist.barrier()
            eager = {name: kerns[name] for name in graphed}  # keeps each arm's run closure (its tensors) alive for the replays
            for name, kern in eager.items():
                kerns[name] = kern._replace(run=capture_graph(kern.run, warmup=0), label=kern.label + " (graph)")

        checks = {name: {} for name in kerns}
        for name, kern in kerns.items():
            if name in ABLATION_ARMS:
                continue
            if args.ref_check:
                checks[name]["ref"] = ref_check(kern, ref)
            if args.r2r_check:
                checks[name]["r2r"] = r2r_check(kern)

        arms = {name: kern.run for name, kern in kerns.items()}
        ms = bench_all(arms, ws.x_hdl)
        unstable = [None] * W  # rerun set unioned over ranks: every rank must call the same collectives
        dist.all_gather_object(unstable, [k for k, v in ms.items() if arm_stats(v)[3]])
        rerun = [k for k in arms if any(k in u for u in unstable)]
        if rerun:
            again = bench_all({k: arms[k] for k in dict.fromkeys(rerun + [k for k in ("sonic",) if k in arms])}, ws.x_hdl)
            ms.update({k: v for k, v in again.items() if not arm_stats(v)[3]})

    flops = 2 * total_m * H * I * 3  # GEMM1 (N = 2I) + GEMM2
    stats = {k: arm_stats(v) for k, v in ms.items()}
    torch.cuda.synchronize()
    gathered = [None] * W
    dist.all_gather_object(gathered, (stats, checks))
    if rank == 0:
        med = {k: max((s[k] for s, _ in gathered), key=lambda t: t[0]) for k in ms}  # slowest rank sets the collective
        print(
            f"<bf16 MoE EP forward {args.model} (H={H} I={I} E={E} top-{topk}) | W={W} T_local={T_local} total_m={total_m} "
            f"(rank 0) | timed: x publish -> out, router outside | warm {WARM_S * 1e3:.0f} ms, {REPS} x {WINDOW_S * 1e3:.0f} ms "
            f"blocks, rerun={rerun}>"
        )
        for k, (m, lo, hi, _) in med.items():
            mean = sum(s[k][0] for s, _ in gathered) / W
            flag = "  UNSTABLE" if any(s[k][3] for s, _ in gathered) else ""
            vs = f"  vs sonic {m / med['sonic'][0] - 1:+.1%}" if k != "sonic" and "sonic" in med else ""
            print(
                f"  {k + kerns[k].label + ':':<44} {m * 1e3:7.0f} us  [{lo * 1e3:.0f}-{hi * 1e3:.0f}]  "
                f"{flops / m / 1e9:5.0f} TFLOP/s  rank-mean {mean * 1e3:.0f} us{vs}{flag}"
            )
        for name in kerns:
            if "ref" in checks[name]:
                ok = all(c[name]["ref"][0] for _, c in gathered)
                err = max(c[name]["ref"][1] for _, c in gathered)
                print(f"  {name} ref: {'ok' if ok else 'MISMATCH'} max rel err {err:.3e}")
            if "r2r" in checks[name]:
                ok = all(c[name]["r2r"][0] for _, c in gathered)
                delta = max(c[name]["r2r"][1] for _, c in gathered)
                print(f"  {name} r2r: {'bitwise' if ok else 'MISMATCH'} max|delta|={delta:.3e}")
    for kern in kerns.values():
        kern.free()
    del kerns, arms  # captured graphs (NCCL work inside; arms holds their replays) must go before the process group, else destroy hangs
    torch.cuda.synchronize()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
