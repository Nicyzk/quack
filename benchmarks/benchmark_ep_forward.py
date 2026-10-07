"""Run full MoE EP forward baselines through one interface (quack.bench.ep's protocol and Kernel record).

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
  quack_inter      the top-k all-gather, our routing metadata kernels (quack.ep.routing_metadata: the same tables
                   as sonic's, 2 launches), then our inter-SM pipeline: the device walks,
                   fused dispatch + GEMM1 (quack.gemm_sm100_ep_inter's pull on --comm-sms SMs, quack's gated GEMM
                   with the SwiGLU epilogue on the rest, dispatch scheduler), fused GEMM2 + combine
                   (quack.gemm_sm100_ep_inter's push on the same partition, the GEMM on the rest), barrier, sonic's
                   gather-and-sum with the
                   scores. GEMM1 on 256x512 (2,1) static tiles, GEMM2 on 256x256 (2,1): its K is short, so the
                   256x512 tile's single tmem accumulator stage left the epilogue exposed (-5 % GEMM alone at 256x256).
  quack_intra      the same stages with the intra-SM kernels on all SMs, no partitions: the gated GEMM1 whose CTAs
                   carry 4 transfer warps pulling the walk (quack.gemm_sm100_ep_intra_dispatch), then the GEMM2
                   whose CTAs push the walk in 2 KB chunks (quack.gemm_sm100_ep_intra_combine), barrier, the same
                   gather-and-sum.

--ref_check compares each arm's fresh output with a bf16-matmul reference (max rel err < 2^-6: three bf16 rounding
points and a K-term weighted sum), --r2r_check runs each arm twice for bit-identity.

Usage (sonicmoe and DeepEP are sibling checkouts on PYTHONPATH; DeepGEMM and Transformer Engine installed):
    PYTHONPATH=~/quack:~/sonic-moe-pr54:~/DeepEP LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH \\
        torchrun --nproc_per_node=4 --standalone benchmarks/benchmark_ep_forward.py \\
        sonic deepep_deepgemm deepep_te megamoe quack_inter quack_intra --comm-sms 16 --ref_check
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
import torch.nn.functional as F
from cutlass import Int32, Int64
from quack.compile_utils import make_fake_tensor
from quack.cute_dsl_utils import get_device_capacity, get_max_active_clusters, torch2cute_dtype_map
from quack.epilogue.library import linear_act_mod
from quack.gemm_default_epi import GemmDefaultEpiMixin, GemmDefaultSm100
from quack.gemm_runtime.host import FakeArgCtx
from quack.gemm_tvm_ffi_utils import (
    make_fake_gemm_tensors,
    make_fake_scheduler_args,
    make_fake_varlen_args,
    make_scheduler_args,
    make_varlen_args,
)
from quack.rounding import RoundingMode
from sonicmoe.distributed_utils import CombineMode, DispatchMode, RuntimeEPConfig, SymmMemManager
from sonicmoe.enums import ActivationType
from sonicmoe.functional.distributed import build_rank_dedup_a_idx, compute_dispatch_metadata
from sonicmoe.functional.ep import _ag_routing_decision, _build_consumer_metadata, _moe_ep_forward_inner

from quack.bench.ep import (
    CLUSTER, SM100, TILE, _NOOP, _run_synced, MODEL_CONFIGS, REPS, WARM_S, WINDOW_S, GemmDefaultSm100IntraCombine,
    Kernel, SMStream, _gather_and_sum, arm_stats, bench_all, capture_graph, compile_gemm2, gemm2_args, r2r_check,
)
from quack.ep.prologue import make_prologue
from quack.ep.routed_dispatch import make_routed_buffers
from quack.ep.routing_metadata import make_routing_metadata
from quack.gemm_sm100_ep_inter import (
    make_combine_buffers, make_combine_push, make_combine_walk_device, make_dispatch_pull, make_dispatch_walk,
)
from quack.gemm_sm100_ep_intra_dispatch import GemmSm100 as GemmSm100IntraDispatch

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


_compiled = {}
GEMM1_TILE = (256, 512) if SM100 else TILE  # GEMM1's K is 3.5x GEMM2's: the exposed epilogue is small, the wide tile wins


def compile_gemm1_gated(dtype, tile=GEMM1_TILE, cluster=CLUSTER, intra_pull=0, gather_A=True, routed=()):
    """(fn, Cls, block_rows): quack's gated (SwiGLU) epilogue kernel class, minted by the same frontend gemm_gated uses
    (acc pairs over adjacent N, the postact tile half of N, no D store), compiled as GEMM1: varlen M, gather A, the
    dispatch scheduler argument, static persistence. Launch:
    fn(A, B, None, None, Cls.EpilogueArguments(mAuxOut=postact), scheduler, varlen, None, None). intra_pull = the
    hidden size: the same epilogue on the intra-SM dispatch kernel (transfer warps pulling A's rows; five walk tensors
    after the stream). gather_A=False: plain TMA A loads over the same
    varlen rows (the routed arm). routed = (E, W, K): MegaMoE's in-kernel routed dispatch (routed_dispatch.py)
    on the intra kernel, plain TMA A; the launch then takes make_routed_buffers' tensors after the walk placeholders."""
    key = (dtype, tile, cluster, intra_pull, gather_A, routed)
    if key not in _compiled:
        sm = get_device_capacity(torch.device("cuda"))[0]
        assert sm >= 9, "the dispatch-scheduler gated GEMM needs quack's SM90 or SM100 kernel"
        mod = linear_act_mod("swiglu", gated=True, has_c=False, has_rowvec=False, has_colvec=False)
        Cls = mod._mint((), sm, True, False, (), RoundingMode.RN, (), False)
        kw = {}
        if intra_pull:  # the minted class rebased from quack's GemmSm100 onto the intra dispatch kernel
            assert sm >= 10, "the intra-SM dispatch kernel is SM100 only"
            attrs = {k: v for k, v in vars(Cls).items() if k not in ("__dict__", "__weakref__")}
            Cls = type(Cls.__name__ + "_intra", (Cls.__bases__[0], GemmSm100IntraDispatch), attrs)
            kw = {"intra_dispatch_pull": intra_pull, "intra_dispatch_routed": routed}
        gemm = Cls(cutlass.Float32, dtype, tile, (*cluster, 1), gather_A=gather_A, use_clc_persistence=False, **kw)
        gemm.b_transposed = gemm.a_transposed = gemm.cd_transposed = False  # as quack's compile_gemm_kernel sets
        gemm.cd_packed = None
        mA, mB, mD, mC, m, n, k, l = make_fake_gemm_tensors(dtype, dtype, None, None, "k", "k", "n", "n", varlen_m=True, gather_A=gather_A)
        aux = next(op for op in Cls._epi_ops if op.name == "mAuxOut")
        epi = Cls.EpilogueArguments(mAuxOut=aux.host_fake_arg((dtype, "n"), FakeArgCtx(m, n, k, l, True, True)))
        sched = make_fake_scheduler_args(False, False, l, has_dispatch=True)
        varlen = make_fake_varlen_args(True, False, True, m)  # mAIdx in both modes: the dispatch scheduler reads it
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        i32 = lambda: make_fake_tensor(Int32, (cute.sym_int(),), leading_dim=0, divisibility=1)
        i64 = lambda: make_fake_tensor(Int64, (cute.sym_int(),), leading_dim=0, divisibility=1)
        walk_fakes = (i64(), i32(), i32(), i32(), i32(), i32()) if intra_pull else ()
        if routed:  # the routed tensors (make_routed_buffers' order) + T and rank
            walk_fakes += (i32(), i64(), i32(), i64(), i64(), i64(), i64(), i64(), i32(), i64(), i32(), i32(), i32(), i64(), i32(), Int32(0), Int32(0))
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


def swap_tile_n(model, W, T_local):
    """The swap kernels' token block: the fewest 256-row blocks covering the expected rows per expert (T_local * W *
    top-k / E), each sized to the expected rows plus their square root, rounded up to 32 (full blocks stay in the
    efficient MMA-N class; 16-odd N runs 5-8 % slower per row), in [64, 256]. W=4: 160 at 1024 tokens, 256 from 2048."""
    expected = T_local * W * model.topk / model.E
    blocks = math.ceil(expected / 256)
    return int(min(256, max(64, math.ceil((expected + expected**0.5) / blocks / 32) * 32)))


def compile_gemm1_gated_swap(dtype, model, W, T_local, intra_pull=0, gather_B=True):
    """(fn, Cls, block_rows): the gated epilogue on the swap-AB kernel — weights on M (interleave_gate_up's rows,
    pairs over M), tokens on N (varlen_n even split), the dispatch scheduler, dynamic persistence (resident grid +
    atomic tile counter: CLC's arrival order without CLC's oversized grid, which would cancel transfer warps; static
    loses ~300 µs of L2 reuse). gather_B: B rows by index over the dedup pool (TMA
    gather4; costs ~90 µs at W=4 1024); False: one plain TMA box over rows packed
    per expert, which on the intra kernel the transfer warps land per reader (intra_dispatch_scatter). Launch:
    fn(w1_il8, rows, None, None, Cls.EpilogueArguments(mAuxOut=postact.T), scheduler, varlen) with a zeroed int32
    semaphore in the scheduler arguments. block_rows = tile N, the unit the even-split blocks and their flags are cut
    from. intra_pull = the hidden size: the same on the intra-SM dispatch kernel (transfer warps pulling the walk;
    six walk tensors after the varlen argument, as compile_gemm1_gated)."""
    tile_n = swap_tile_n(model, W, T_local)  # the token block is derived from the shape, never caller-chosen
    tile = (256, tile_n)
    key = ("swap", dtype, intra_pull, gather_B, tile_n, SWAP_N_VARIANTS, SWAP_DYNAMIC_N)
    if key not in _compiled:
        assert SM100, "the swap-AB gated kernel is SM100 only"
        mod = linear_act_mod("swiglu", gated=True, has_c=False, has_rowvec=False, has_colvec=False)
        Cls = mod._mint((), 10, True, False, (), RoundingMode.RN, (), False, pair_dim="m")
        kw = {"varlen_n_dynamic_n": True} if SWAP_DYNAMIC_N else {}
        if intra_pull:  # the minted class rebased from quack's GemmSm100 onto the intra dispatch kernel
            attrs = {k: v for k, v in vars(Cls).items() if k not in ("__dict__", "__weakref__")}
            Cls = type(Cls.__name__ + "_intra", (Cls.__bases__[0], GemmSm100IntraDispatch), attrs)
            kw.update({"intra_dispatch_pull": intra_pull, "intra_dispatch_scatter": not gather_B})
        gemm = Cls(cutlass.Float32, dtype, tile, (*CLUSTER, 1), use_clc_persistence=False, gather_B=gather_B,
                   varlen_n_variants=SWAP_N_VARIANTS, **kw)
        gemm.b_transposed = gemm.a_transposed = gemm.cd_transposed = False
        gemm.cd_packed = None
        mA, mB, mD, mC, m, n, k, l = make_fake_gemm_tensors(dtype, dtype, None, None, "k", "k", "n", "n", varlen_n=True, gather_B=gather_B)
        aux = next(op for op in Cls._epi_ops if op.name == "mAuxOut")
        epi = Cls.EpilogueArguments(mAuxOut=aux.host_fake_arg((dtype, "m"), FakeArgCtx(m, n, k, l, False, False)))
        sched = make_fake_scheduler_args(True, False, l, has_dispatch=True, has_dispatch_row_key=not gather_B)
        varlen = make_fake_varlen_args(False, False, False, None, varlen_n=True, gather_B=gather_B)
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        i32 = lambda: make_fake_tensor(Int32, (cute.sym_int(),), leading_dim=0, divisibility=1)
        i64 = lambda: make_fake_tensor(Int64, (cute.sym_int(),), leading_dim=0, divisibility=1)
        walk_fakes = (i64(), i32(), i32(), i32(), i32(), i32()) if intra_pull else ()
        if intra_pull:  # the 17 routed placeholders (15 tensors, T, rank), then the per-CTA tensormap slots (L2 promotion 256B)
            walk_fakes += (None,) * 15 + (Int32(0), Int32(0), i64())
        fn = cute.compile(gemm, mA, mB, mD, mC, epi, sched, varlen, stream, None, None, *walk_fakes, options="--enable-tvm-ffi")
        _compiled[key] = (fn, Cls, tile_n)
    return _compiled[key]


def compile_gemm2_swap(dtype, model, W, T_local, intra_push=0):
    """(fn, block_rows, cluster_tile_m): the plain swap-AB kernel for GEMM2 — weights (H, I) on M, tokens on N
    (varlen_n even split), D = y.T (m-major), the combine scheduler (+1 per stored CTA tile, so a block's flag target
    is CLUSTER CTAs x ceil(H / cluster_tile_m)), dynamic persistence (zeroed semaphore in sched, as
    compile_gemm1_gated_swap). Launch: fn(w2, a, y.T, None, epi, sched, varlen).
    intra_push = the hidden size: the intra-SM combine kernel whose epilogue stores each token's slice of a subtile
    straight into the origins' slots (no y, no flags, no transfer warps); after the varlen argument five None
    placeholders, the slot base table (world,) int64, the slots per rank and the per-row slot addresses."""
    tile_n = swap_tile_n(model, W, T_local)  # the same block as GEMM1's (asserted at the call sites)
    tile = (256, tile_n)
    key = ("swap2", dtype, intra_push, tile_n, SWAP_N_VARIANTS, SWAP_DYNAMIC_N)
    if key not in _compiled:
        assert SM100, "the swap-AB kernel is SM100 only"
        dyn = {"varlen_n_dynamic_n": True} if SWAP_DYNAMIC_N else {}
        if intra_push:
            gemm = GemmDefaultSm100IntraCombine(
                cutlass.Float32, dtype, tile, (*CLUSTER, 1), use_clc_persistence=False, intra_combine_push=intra_push,
                varlen_n_variants=SWAP_N_VARIANTS, combine_epilogue_store=True, **dyn,
            )
            i64 = lambda: make_fake_tensor(Int64, (cute.sym_int(),), leading_dim=0, divisibility=1)
            walk_fakes = (None,) * 5 + (i64(), Int32(0), i64(), i64())  # ..., slot addresses, tensormap slots
        else:
            gemm = GemmDefaultSm100(cutlass.Float32, dtype, tile, (*CLUSTER, 1), use_clc_persistence=False,
                                    varlen_n_variants=SWAP_N_VARIANTS, **dyn)
            walk_fakes = ()
        gemm.b_transposed = gemm.a_transposed = gemm.cd_transposed = False
        gemm.cd_packed = None
        mA, mB, mD, mC, m, n, k, l = make_fake_gemm_tensors(dtype, dtype, dtype, None, "k", "k", "m", "n", varlen_n=True)
        epi = GemmDefaultEpiMixin.EpilogueArguments(add_to_output=False, rounding_mode=RoundingMode.RN)
        sched = make_fake_scheduler_args(True, False, l, has_combine=True)
        varlen = make_fake_varlen_args(False, False, False, None, varlen_n=True)
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        fn = cute.compile(gemm, mA, mB, mD, mC, epi, sched, varlen, stream, None, None, *walk_fakes, options="--enable-tvm-ffi")
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


def _quack_metadata(x, topk_idx, ws, rank, W, K, E_local, sonic_metadata=False):
    """The metadata the quack walks and GEMMs read, in fixed buffers: (topk_g, meta, prologue) with meta =
    routing_metadata's tables (cu, s_reverse, s_scatter, a_idx, first_src_row); prologue() = x publish, top-k
    all-gather, barrier, then the two metadata kernels (dispatch/routing_metadata.py). sonic_metadata: the same
    tables from sonic's compute_dispatch_metadata + _build_consumer_metadata + build_rank_dedup_a_idx (12 launches)
    instead, for the A/B."""
    T_local, dev = x.shape[0], x.device
    TK, max_rows = W * T_local * K, W * T_local * min(K, E_local)
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
    TK, max_rows = W * T_local * K, W * T_local * min(K, E_local)
    comm = SMStream(dev.index, args.comm_sms)
    assert comm.stream is not None, "--comm-sms must leave SMs for the GEMMs"
    rest = comm.complement()
    topk_g, meta, prologue = _quack_metadata(x, topk_idx, ws, rank, W, K, E_local, args.sonic_metadata)
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
    TK, max_rows = W * T_local * K, W * T_local * min(K, E_local)
    comm = SMStream(dev.index, args.comm_sms)
    assert comm.stream is not None, "--comm-sms must leave SMs for the GEMMs"
    rest = comm.complement()
    topk_g, meta, prologue = _quack_metadata(x, topk_idx, ws, rank, W, K, E_local, args.sonic_metadata)
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
    TK, max_rows = W * T_local * K, W * T_local * min(K, E_local)
    clusters = get_max_active_clusters(CLUSTER[0] * CLUSTER[1], get_device_capacity(dev))
    w1_il8 = interleave_gate_up(w1_cat)
    fn1, Cls1, block_rows = compile_gemm1_gated_swap(dtype, model, W, T_local, intra_pull=H, gather_B=False)
    fn2, block_rows2, cluster_tile_m = compile_gemm2_swap(dtype, model, W, T_local, intra_push=H)
    assert block_rows2 == block_rows
    buffers = make_combine_buffers(T_local, K, H, torch.bfloat16, ws.ep_group)
    prologue, meta, walk1 = make_prologue(
        topk_idx, ws.x_peer_bufs, rank, W, T_local, K, E_local, max_rows, block_rows, buffers.slot_base, ws.ep_group
    )
    cu, s_scatter, a_idx = meta.cu, meta.s_scatter, meta.a_idx
    rows = torch.empty(max_rows, H, dtype=torch.bfloat16, device=dev)  # B rows packed per expert (cu order)
    a = torch.empty(max_rows, I, dtype=torch.bfloat16, device=dev)
    sched1 = make_scheduler_args(clusters, 8, prologue.sem1, dispatch_args=(walk1.flags, walk1.first_src_row, a_idx))
    rt1 = (Cls1.EpilogueArguments(mAuxOut=a.T), sched1, make_varlen_args(None, None, None, cu_seqlens_n=cu))
    # per-CTA copies of the weight tensormaps (the kernels patch L2 promotion to 256B): 128 B slots
    desc = lambda: torch.empty(torch.cuda.get_device_properties(dev).multi_processor_count * 16, dtype=torch.int64, device=dev)
    desc1, desc2 = desc(), desc()
    gemm1 = lambda: fn1(
        w1_il8, rows, None, None, *rt1, None, None, walk1.src_address, walk1.pool_row, walk1.gate_ptr, walk1.gates,
        walk1.num_rows, walk1.dst_row, *((None,) * 15), 0, 0, desc1,
    )
    y = torch.empty(max_rows, H, dtype=torch.bfloat16, device=dev)  # D's shape for the kernel; never written
    flags2 = torch.zeros(E_local, -(-W * T_local // block_rows), dtype=torch.int32, device=dev)  # the scheduler's; never published
    first_src_row2 = torch.tensor([((rank + 1) % W) * TK // W], dtype=torch.int32, device=dev)
    epi2 = GemmDefaultEpiMixin.EpilogueArguments(add_to_output=None, rounding_mode=None)
    sched2 = make_scheduler_args(clusters, 8, prologue.sem2, combine_args=(flags2, s_scatter, first_src_row2))
    rt2 = (epi2, sched2, make_varlen_args(None, None, None, cu_seqlens_n=cu))
    gemm2 = lambda: fn2(
        w2, a, y.T, None, *rt2, None, None, None, None, None, None, None, buffers.slot_base, T_local * K, meta.slot_addr, desc2
    )
    out, reduce = _gather_and_sum(buffers, scores, T_local, K, H)

    publish = lambda: ws.x_symm.copy_(x)  # before the prologue on the stream: its NVLink barrier orders it for the peers
    steps = (publish, prologue, gemm1, gemm2, lambda: ws.o_hdl.barrier(channel=1), reduce)

    def run():
        for step in steps:
            step()

    run.steps = steps
    run.step_names = ("x publish", "prologue", "gemm1", "gemm2", "barrier", "reduce")
    run()  # every kernel once (module loading) before the timed calls
    zero = lambda: (out.zero_(), buffers.slots.zero_())
    return {"quack_intra_swap": Kernel(run, _NOOP, out.float, zero, lambda ref: ref)}


def _quack_intra(model, x, topk_idx, scores, w1_cat, w1_il, w2, ws, rank, W, T_local, args):
    """quack_inter's stages with the intra-SM kernels on all SMs: the gated GEMM1 whose CTAs carry 4 transfer warps
    pulling the walk, then the GEMM2 whose CTAs push the walk, one stream, no partitions."""
    H, I, K, E_local, dev = model.H, model.I, model.topk, model.E // W, x.device
    dtype = torch2cute_dtype_map[x.dtype]
    TK, max_rows = W * T_local * K, W * T_local * min(K, E_local)
    clusters = get_max_active_clusters(CLUSTER[0] * CLUSTER[1], get_device_capacity(dev))
    topk_g, meta, prologue = _quack_metadata(x, topk_idx, ws, rank, W, K, E_local, args.sonic_metadata)
    cu, s_scatter, recv_pos, a_idx = meta.cu, meta.s_scatter, meta.s_reverse, meta.a_idx
    fn1, Cls1, block_rows = compile_gemm1_gated(dtype, intra_pull=H)
    build1, walk1 = make_dispatch_walk(topk_g, s_scatter, recv_pos, meta.first_src_row, rank, a_idx, cu, ws.x_peer_bufs, block_rows)
    pool = torch.empty(W * T_local, H, dtype=torch.bfloat16, device=dev)
    a = torch.empty(max_rows, I, dtype=torch.bfloat16, device=dev)
    sched1 = make_scheduler_args(clusters, 8, None, dispatch_args=(walk1.flags, walk1.first_src_row))
    rt1 = (Cls1.EpilogueArguments(mAuxOut=a), sched1, make_varlen_args(cu, None, a_idx))
    gemm1 = lambda: fn1(
        pool, w1_il, None, None, *rt1, None, None, walk1.src_address, walk1.pool_row, walk1.gate_ptr, walk1.gates, walk1.num_rows,
        walk1.dst_row,
    )
    fn2, block_rows2, tile_n = compile_gemm2(dtype, intra_push=H)
    assert block_rows2 == block_rows
    y = torch.empty(max_rows, H, dtype=torch.bfloat16, device=dev)
    flags2 = torch.zeros(E_local, -(-W * T_local // block_rows), dtype=torch.int32, device=dev)
    slot_of_row = s_scatter
    first_src_row2 = torch.tensor([((rank + 1) % W) * TK // W], dtype=torch.int32, device=dev)
    buffers = make_combine_buffers(T_local, K, H, torch.bfloat16, ws.ep_group)
    build2, walk2 = make_combine_walk_device(slot_of_row, cu, flags2, first_src_row2, buffers, T_local, K, tile_n=tile_n)
    rt2 = gemm2_args(cu, flags2, slot_of_row, first_src_row2)
    gemm2 = lambda: fn2(
        a, w2, y, None, *rt2, None, None, walk2.row, walk2.dst_address, walk2.gate_address, walk2.flag_address, walk2.num_rows
    )
    out, reduce = _gather_and_sum(buffers, scores, T_local, K, H)

    steps = (prologue, build1, walk1.flags.zero_, gemm1, build2, flags2.zero_, gemm2, lambda: ws.o_hdl.barrier(channel=1), reduce)

    def run():
        for step in steps:
            step()

    run.steps = steps  # the stages, for capture bisection
    run.step_names = ("prologue", "build1", "flags1.zero", "gemm1", "build2", "flags2.zero", "gemm2", "barrier", "reduce")
    run()  # every kernel once (module loading) before the timed calls
    zero = lambda: (out.zero_(), buffers.slots.zero_())
    return {"quack_intra": Kernel(run, _NOOP, out.float, zero, lambda ref: ref)}


def _quack_intra_routed(model, x, topk_idx, scores, w1_cat, w1_il, w2, ws, rank, W, T_local, args):
    """quack_intra with MegaMoE's in-kernel routed dispatch (dispatch/routed_dispatch.py): no top-k all-gather,
    metadata kernels or walk builders; GEMM1's transfer warps count the local top-k, exchange the per-expert counts
    over NVLink and pull K rows per token (routed, no dedup) into an expert-major pool that GEMM1 reads with plain
    TMA. The prologue is the x publish and four memsets; the combine side is quack_intra's."""
    H, I, K, E, E_local, dev = model.H, model.I, model.topk, model.E, model.E // W, x.device
    dtype = torch2cute_dtype_map[x.dtype]
    TK, max_rows = W * T_local * K, W * T_local * min(K, E_local)
    clusters = get_max_active_clusters(CLUSTER[0] * CLUSTER[1], get_device_capacity(dev))
    fn1, Cls1, block_rows = compile_gemm1_gated(dtype, intra_pull=H, gather_A=False, routed=(E, W, K))
    buf = make_routed_buffers(T_local, K, E, E_local, max_rows, -(-W * T_local // block_rows), ws.x_peer_bufs, ws.ep_group)
    buf.topk = topk_idx.reshape(-1).contiguous()  # int32 global expert ids
    pool = torch.empty(max_rows, H, dtype=torch.bfloat16, device=dev)
    a = torch.empty(max_rows, I, dtype=torch.bfloat16, device=dev)
    a_idx = torch.arange(max_rows, dtype=torch.int32, device=dev)  # only the scheduler's first-tile lookup reads it
    first_src_row = torch.zeros(1, dtype=torch.int32, device=dev)
    sched1 = make_scheduler_args(clusters, 8, None, dispatch_args=(buf.flags, first_src_row))
    rt1 = (Cls1.EpilogueArguments(mAuxOut=a), sched1, make_varlen_args(buf.cu, None, a_idx))
    i32_1, i64_1 = torch.zeros(1, dtype=torch.int32, device=dev), torch.zeros(1, dtype=torch.int64, device=dev)
    trace = None
    if os.environ.get("ROUTED_DEBUG"):  # hang localization: the kernel writes phase markers into a file-backed,
        import ctypes  # device-mapped host buffer another process polls (this one may block behind the hung stream)

        trace = torch.from_file(f"{os.environ['ROUTED_DEBUG']}.rank{rank}", shared=True, size=1024, dtype=torch.int32)
        trace.zero_()
        rc = ctypes.CDLL("libcudart.so").cudaHostRegister(ctypes.c_void_p(trace.data_ptr()), 4096, 2)  # cudaHostRegisterMapped
        print(f"[rank {rank}] routed trace file {os.environ['ROUTED_DEBUG']}.rank{rank} register rc={rc}", flush=True)
        i64_1[0] = trace.data_ptr()
    gemm1 = lambda: fn1(
        pool, w1_il, None, None, *rt1, None, None, i64_1, i32_1, i32_1, i32_1, i32_1, i32_1,  # the walk placeholders
        buf.topk, buf.send_count, buf.src_idx, buf.idx_base, buf.recv_count, buf.recv_count_base, buf.recv_sum,
        buf.recv_sum_base, buf.signal, buf.signal_base, buf.barrier_counter, buf.grid_counter, buf.cu, buf.x_base,
        buf.slot_of_row, T_local, rank,
    )
    fn2, block_rows2, tile_n = compile_gemm2(dtype, intra_push=H)
    assert block_rows2 == block_rows
    y = torch.empty(max_rows, H, dtype=torch.bfloat16, device=dev)
    flags2 = torch.zeros(E_local, -(-W * T_local // block_rows), dtype=torch.int32, device=dev)
    first_src_row2 = torch.tensor([((rank + 1) % W) * TK // W], dtype=torch.int32, device=dev)
    buffers = make_combine_buffers(T_local, K, H, torch.bfloat16, ws.ep_group)
    build2, walk2 = make_combine_walk_device(buf.slot_of_row, buf.cu, flags2, first_src_row2, buffers, T_local, K, tile_n=tile_n)
    rt2 = gemm2_args(buf.cu, flags2, buf.slot_of_row, first_src_row2)
    gemm2 = lambda: fn2(
        a, w2, y, None, *rt2, None, None, walk2.row, walk2.dst_address, walk2.gate_address, walk2.flag_address, walk2.num_rows
    )
    out, reduce = _gather_and_sum(buffers, scores, T_local, K, H)

    steps = (lambda: ws.x_symm.copy_(x), buf.reset, gemm1, build2, flags2.zero_, gemm2, lambda: ws.o_hdl.barrier(channel=1), reduce)

    def run():
        for step in steps:
            step()

    run.steps = steps  # the stages, for capture bisection
    run.step_names = ("x publish", "reset", "gemm1", "build2", "flags2.zero", "gemm2", "barrier", "reduce")
    if trace is not None:  # GEMM1 alone, warm, then its in-kernel phase timestamps (routed_stamp) per launch, in us
        steps[0]()
        steps[1]()
        gemm1()
        torch.cuda.synchronize()
        names = ("prep", "barrier", "cta0 pull", "all pulls", "released", "first gate", "last gate")
        for _ in range(5):
            steps[1]()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            gemm1()
            e.record()
            torch.cuda.synchronize()
            t = trace.view(torch.int64)[32:40].tolist()  # tags 0..7: transfer entry, prep, barrier, cta0 pull, all pulls, released, first/last gate
            d = " ".join(f"{n} {(v - t[0]) / 1e3:.0f}" for n, v in zip(names, t[1:]))
            print(f"[rank {rank}] gemm1 {s.elapsed_time(e) * 1e3:.0f} us | {d}", flush=True)
        os._exit(0)
    run()  # every kernel once (module loading) before the timed calls
    zero = lambda: (out.zero_(), buffers.slots.zero_())
    return {"quack_intra_routed": Kernel(run, _NOOP, out.float, zero, lambda ref: ref)}


BASELINES = {
    "sonic": _sonic, "deepep_deepgemm": _deepep_deepgemm, "deepep_te": _deepep_te, "megamoe": _megamoe,
    "quack_inter": _quack_inter, "quack_inter_swap": _quack_inter_swap,
    "quack_intra": _quack_intra, "quack_intra_swap": _quack_intra_swap, "quack_intra_routed": _quack_intra_routed,
}
# --graph: every arm's call is captured into a CUDA graph (all arms warm up eagerly first, then the captures) and the
# replays are timed; the eager Kernel records stay referenced so the captured launches' tensors outlive the capture.
GRAPH_ARMS = ("quack_inter", "quack_inter_swap", "quack_intra", "quack_intra_swap", "quack_intra_routed", "megamoe", "sonic")
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
