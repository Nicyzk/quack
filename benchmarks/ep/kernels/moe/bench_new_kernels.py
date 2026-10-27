"""The fused dispatch + fc1 stage of bench_forward's quack_intra_swap (bf16) with other activations: the same walk,
landed rows and dispatch scheduler, the swap-AB GEMM1 minted with quack's gated (swiglu, geglu: weights (2I, H)
interleaved, pairs over M) or plain (relu_sq: weights (I, H), the postact row for row) activation epilogue.
One routing, bench_all's protocol; prologue + GEMM1 per call (the prologue kernel resets the flags and semaphores).
--formats mxfp8 / nvfp4: the stage with quantized operands (x and w1 in the format, x's rows + scale rows pulled
into the per-block SF atoms, bf16 postact; nvfp4: one per-tensor scale each, x's over all ranks, their product the
epilogue's alpha) beside its bounds, the bare GEMM1 over the landed rows (gates pre-satisfied, no transfer warps) and
the pull alone. A plain activation (relu_sq) also gets its unfused pair: the stage storing the raw preact, then the
activation as a graphed torch.compile pass.

--ref_check: the postact rows (cu order) vs act(rows @ w1^T) in fp32 on the landed rows (the formats: on the rows
requantized as the senders did), max rel err < 2^-6.

  torchrun --nproc_per_node=W bench_new_kernels.py --t-local 1024 --activations swiglu geglu relu_sq --ref_check
  torchrun --nproc_per_node=W bench_new_kernels.py --t-local 1024 --formats nvfp4 --activations swiglu geglu --ref_check
  PYTHONPATH=~/nccl-extensions/python-overlay:$PYTHONPATH CUDA_HOME=/usr/local/cuda-13.3 torchrun --nproc_per_node=W \
      bench_new_kernels.py --formats nvfp4 --activations geglu --baselines nccl_ep_te --nccl-ep-sms 16,32,64,all --ref_check
  (nccl_ep_te needs the libnccl_ep.so rebuilt against the env's NCCL, ~/nccl-extensions/README-local.md, and nvcc for its JIT)
"""
import argparse
import itertools
import os

import cutlass
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.nn.functional as F
from cutlass import Float32

import bench_forward as bf
from bench_dispatch import MODEL_CONFIGS, REPS, WARM_S, WINDOW_S, arm_stats, bench_all
from bench_forward import DispatchMode, SymmMemManager, expert_weights
from combine.combine_push import make_combine_buffers
from dispatch.prologue import make_prologue
from dispatch.pull import make_dispatch_pull
from quack.activation import act_fn_map, gate_fn_map
from quack.gemm_tvm_ffi_utils import make_scheduler_args, make_varlen_args

FORMATS = ("mxfp8", "nvfp4")  # the swiglu fc1 stage with quantized operands, plus its GEMM-alone and pull-alone bounds

TORCH_ACT = {
    "swiglu": lambda g, u: F.silu(g) * u,
    "geglu": lambda g, u: F.gelu(g) * u,
    "reglu": lambda g, u: F.relu(g) * u,
    "relu_sq": lambda x: F.relu(x) ** 2,
    "relu": F.relu,
    "silu": F.silu,
    "gelu_tanh_approx": lambda x: F.gelu(x, approximate="tanh"),
    "none": lambda h: h,  # no activation epilogue: the raw (2I-wide) preact is stored
}


def reference_postact(activation, rows, cu, w1_cat, I):
    """(total_m, I) fp32: the activation of each expert's landed rows times its weights."""
    gated, full = activation in gate_fn_map, activation == "none"
    out = torch.empty(int(cu[-1]), 2 * I if full else I, dtype=torch.float32, device=rows.device)
    for e in range(w1_cat.shape[0]):
        lo, hi = int(cu[e]), int(cu[e + 1])
        if hi == lo:
            continue
        h = rows[lo:hi].float() @ (w1_cat[e] if gated or full else w1_cat[e, :I]).float().T
        out[lo:hi] = TORCH_ACT[activation](*h.chunk(2, dim=-1)) if gated else TORCH_ACT[activation](h)
    return out


def fc1_stage_quantized(
    fmt, activation, model, x, topk_idx, w1_cat, ws, rank, W, T_local, block_rows, clusters, desc, rows_bf16, first_src_rank=None
):
    """The fused dispatch + fc1 stage (a gated activation, bf16 postact) with quantized operands and its own walk over
    the packed rows + scale rows, plus its bounds: the bare GEMM1 over the rows the stage landed (gates pre-satisfied)
    and the pull alone. fmt "mxfp8": e4m3 + ue8m0 per 32 (quack to_mx); "nvfp4": e2m1 + ue4m3 per 16 with one
    per-tensor scale each, x's from the amax over all ranks (every rank's rows share the scale the receiving GEMM
    applies), alpha = pts_w * pts_x. first_src_rank = make_prologue's (the source rank each expert's walk starts at;
    rows_bf16 must be landed in the same order). Returns (arms, postact, reference fn, keep): the caller holds `keep`
    for as long as the arms run (the symmetric x buffers; the walk carries their addresses, not references)."""
    H, I, K, E_local, dev = model.H, model.I, model.topk, model.E // W, x.device
    max_rows = W * T_local * min(K, E_local)
    e8m0 = torch.float8_e8m0fnu
    gated = activation in gate_fn_map
    plain = activation in act_fn_map  # non-gated activation: weights (I, H), the postact row for row
    # "none": the plain GEMM epilogue stores the 2I-wide preact (weights not interleaved)
    w1_src = w1_cat[:, :I].contiguous() if plain else w1_cat
    if fmt == "nvfp4":
        amax = x.abs().amax()
        dist.all_reduce(amax, op=dist.ReduceOp.MAX, group=ws.ep_group)
        xq, xs, pts_x, _ = bf.quantize_nvfp4(x, amax=amax)
        w1_q, w1_s, pts_w, w1_dq = bf.quantize_nvfp4(w1_src)
        w1_q_il = bf.interleave_gate_up(w1_q.view(torch.uint8)).view(torch.float4_e2m1fn_x2) if gated else w1_q
        w1_s_il = bf.interleave_gate_up(w1_s) if gated else w1_s
        x_dtype, sf_dtype, cute_dtype, cols, sf_cols = torch.float4_e2m1fn_x2, torch.float8_e4m3fn, cutlass.Float4E2M1FN, H // 2, H // 16
        rows_dq = bf.quantize_nvfp4(rows_bf16, amax=amax)[3]
        alpha_kw = {"alpha": Float32(float(pts_w * pts_x))}
    else:
        xq, xs, _ = bf.quantize_mx(x)
        w1_q, w1_s, w1_dq = bf.quantize_mx(w1_src)
        w1_q_il = bf.interleave_gate_up(w1_q) if gated else w1_q
        w1_s_il = bf.interleave_gate_up(w1_s.view(torch.uint8)) if gated else w1_s.view(torch.uint8)
        x_dtype, sf_dtype, cute_dtype, cols, sf_cols = torch.float8_e4m3fn, e8m0, cutlass.Float8E4M3FN, H, H // 32
        rows_dq = bf.quantize_mx(rows_bf16)[2]
        alpha_kw = {}
    sfa = torch.stack([bf.blocked_sf(w1_s_il[e].view(e8m0)).view(torch.uint8) for e in range(E_local)]).view(sf_dtype)
    xq_symm = symm_mem.empty(T_local, cols, dtype=x_dtype, device=dev)
    xs_symm = symm_mem.empty(T_local, sf_cols, dtype=torch.uint8, device=dev)
    hq, hs = symm_mem.rendezvous(xq_symm, group=ws.ep_group), symm_mem.rendezvous(xs_symm, group=ws.ep_group)
    xq_peers = [hq.get_buffer(q, (T_local, cols), x_dtype) for q in range(W)]
    xs_peers = [hs.get_buffer(q, (T_local, sf_cols), torch.uint8) for q in range(W)]
    xq_symm.copy_(xq.view(x_dtype))  # published once: x arrives quantized, the stage starts at the prologue
    xs_symm.copy_(xs.view(torch.uint8))
    buffers = make_combine_buffers(T_local, K, H, torch.bfloat16, ws.ep_group)  # the prologue's slot table (unused here)
    prologue, meta, walk = make_prologue(
        topk_idx, xq_peers, rank, W, T_local, K, E_local, max_rows, block_rows, buffers.slot_base, ws.ep_group,
        x_sf_peer_bufs=xs_peers, slot_row_bytes=H * 2, first_src_rank=first_src_rank,
    )
    atoms = walk.sf_atom_rows // 128
    rows = torch.empty(max_rows, cols, dtype=x_dtype, device=dev)
    sf1 = torch.zeros(walk.flags.numel() * atoms, sf_cols // 4, 32, 4, 4, dtype=sf_dtype, device=dev)
    a = torch.empty(max_rows, 2 * I if activation == "none" else I, dtype=torch.bfloat16, device=dev)
    varlen = make_varlen_args(None, None, None, cu_seqlens_n=meta.cu, cu_tiles_n=walk.cu_tiles)
    act_kw = {"activation": activation if gated or plain else None, "gated": gated}
    fn, Cls, _ = bf.compile_gemm1_gated_swap(
        cute_dtype, model, W, T_local, intra_pull=H, gather_B=False, mx=fmt, quant_postact=False, **act_kw
    )
    sched = make_scheduler_args(clusters, 8, prologue.sem1, dispatch_args=(walk.flags, walk.first_src_row, meta.a_idx))
    epi = Cls.EpilogueArguments(mAuxOut=a.T, **alpha_kw)
    gemm1 = lambda: fn(w1_q_il, rows, None, None, epi, sched, varlen, sfa, sf1[None], walk.args, desc)
    fn0, Cls0, _ = bf.compile_gemm1_gated_swap(
        cute_dtype, model, W, T_local, intra_pull=0, gather_B=False, mx=fmt, quant_postact=False, **act_kw
    )
    a0 = torch.empty_like(a)
    flags_full, sem0 = torch.full_like(walk.flags, 1 << 30), torch.zeros(1, dtype=torch.int32, device=dev)
    sched0 = make_scheduler_args(clusters, 8, sem0, dispatch_args=(flags_full, walk.first_src_row, meta.a_idx))
    epi0 = Cls0.EpilogueArguments(mAuxOut=a0.T, **alpha_kw)
    gemm0 = lambda: fn0(w1_q_il, rows, None, None, epi0, sched0, varlen, sfa, sf1[None])
    pull_go = make_dispatch_pull(xq_peers, rank, "intra", None, rows, scatter=True, a_format=fmt, sf_pool=sf1)[0](walk)
    arms = {
        f"dispatch+gemm1 {activation} {fmt}": lambda: (prologue(), gemm1()),
        f"gemm1 alone {activation} {fmt}": lambda: (sem0.zero_(), gemm0()),
        f"pull alone {fmt}": lambda: (prologue(), pull_go()),
    }
    want = lambda: reference_postact(activation, rows_dq, meta.cu, w1_dq, I)
    posts = {f"dispatch+gemm1 {activation} {fmt}": (a, want)}
    if plain or gated:  # the unfused pair: the same stage storing the raw preact, then the activation as its own pass
        fn_p, Cls_p, _ = bf.compile_gemm1_gated_swap(
            cute_dtype, model, W, T_local, intra_pull=H, gather_B=False, mx=fmt, quant_postact=False,
            activation=None, gated=False,
        )
        # gated: the 2I-wide preact over the interleaved weights (interleave_gate_up: per 16 columns, 8 gate then 8 up),
        # the pass pairs them
        pre, post = torch.empty(max_rows, 2 * I if gated else I, dtype=torch.bfloat16, device=dev), torch.empty_like(a)
        epi_p = Cls_p.EpilogueArguments(mAuxOut=pre.T, **alpha_kw)
        gemm1_p = lambda: fn_p(w1_q_il, rows, None, None, epi_p, sched, varlen, sfa, sf1[None], walk.args, desc)
        act = TORCH_ACT[activation]

        def gated_pass(h):
            g = h.view(*h.shape[:-1], -1, 2, 8)
            return act(g[..., 0, :], g[..., 1, :]).reshape(*h.shape[:-1], -1)

        act_c = torch.compile(lambda h, out: out.copy_(gated_pass(h) if gated else act(h)))
        n_rows = rows_bf16.shape[0]  # the landed rows (meta.cu is only filled once the prologue has run)
        act_g = bf.capture_graph(lambda: act_c(pre[:n_rows], post[:n_rows]))  # graphed: eager hides under host enqueue
        arms[f"dispatch+gemm1 {activation}-preact {fmt}"] = lambda: (prologue(), gemm1_p())
        arms[f"{activation} pass {fmt}"] = act_g
        posts[f"{activation} pass {fmt}"] = (post, want)
    keep = (xq_symm, xs_symm, hq, hs, xq_peers, xs_peers, buffers)
    return arms, posts, keep


class NcclRouting:
    """The dispatch schedule of an NCCL all-to-all baseline (Mixture-of-Kittens' bench_nccl_torch): per call, the
    per-expert counts are exchanged (one all_to_all), the token copies are sorted by expert and the send / receive
    splits, the grouped (expert-major) row order and the per-expert offsets are derived; the splits' .tolist() is the
    baseline's host sync."""

    def __init__(self, topk_idx, E_local, group):
        self.group, self.W, self.E_local = group, dist.get_world_size(group), E_local
        self.flat = topk_idx.flatten().to(torch.int64)
        self.sort_indices = torch.argsort(self.flat)
        self.local_counts = torch.zeros(self.W * E_local, dtype=torch.int32, device=topk_idx.device)
        self.local_counts.scatter_add_(0, self.flat, torch.ones_like(self.flat, dtype=torch.int32))
        self.recv_counts = torch.empty(self.W, E_local, dtype=torch.int32, device=topk_idx.device)
        self.block_ids = torch.arange(self.W * E_local, dtype=torch.int32, device=topk_idx.device)

    def build(self):
        dist.all_to_all_single(self.recv_counts.view(-1), self.local_counts, group=self.group)
        counts = self.recv_counts  # (source rank, local expert)
        send = self.local_counts.view(self.W, self.E_local).sum(1, dtype=torch.int32)
        recv = counts.sum(1, dtype=torch.int32)
        per_expert = counts.sum(0, dtype=torch.int32)
        sizes = torch.cat((send, recv, per_expert)).tolist()  # the one host sync: a2a splits + the per-expert rows
        send_splits, recv_splits, rows = sizes[: self.W], sizes[self.W : 2 * self.W], sizes[2 * self.W :]
        num_recv = sum(recv_splits)
        counts_flat = counts.flatten()
        recv_blocks = torch.repeat_interleave(self.block_ids, counts_flat, output_size=num_recv)
        block_starts = torch.cumsum(counts_flat, 0, dtype=torch.int32) - counts_flat
        within = torch.arange(num_recv, dtype=torch.int32, device=counts.device) - block_starts[recv_blocks]
        cu = torch.cumsum(per_expert, 0, dtype=torch.int32)
        expert_starts = cu - per_expert
        source_starts = torch.cumsum(counts, 0) - counts
        recv_to_grouped = ((expert_starts.unsqueeze(0) + source_starts).flatten()[recv_blocks] + within).to(torch.int64)
        return send_splits, recv_splits, num_recv, recv_to_grouped, torch.cat((cu.new_zeros(1), cu)), rows


def fc1_stage_te(kind, fmt, activation, model, x, topk_idx, w1_cat, ws, rank, W, T_local, sms=None):
    """fc1_stage_unfused's TE arms, see there; sms = NCCL EP's SM budget (nccl_ep_te)."""
    import transformer_engine.pytorch as te
    import transformer_engine_torch as tex
    from transformer_engine.common.recipe import MXFP8BlockScaling, NVFP4BlockScaling
    from transformer_engine.pytorch.cpp_extensions import general_grouped_gemm, general_grouped_gemm_for_grouped_tensor
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer, MXFP8Tensor
    from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer, NVFP4Tensor
    from transformer_engine.pytorch.tensor.storage.grouped_tensor_storage import GroupedTensorStorage

    assert fmt in FORMATS, f"{kind}: --formats mxfp8 / nvfp4"
    H, I, K, E_local, dev = model.H, model.I, model.topk, model.E // W, x.device
    max_rows = W * T_local * min(K, E_local)
    mx, gated = fmt == "mxfp8", activation in gate_fn_map
    w1_src = w1_cat if gated or activation == "none" else w1_cat[:, :I].contiguous()  # none: the 2I-wide (gated-shape) preact
    n_out = w1_src.shape[1]
    te_act = {"swiglu": tex.swiglu, "geglu": tex.geglu, "relu_sq": tex.srelu, "none": lambda h, _: h}[activation]
    e4m3, e2m1 = tex.DType.kFloat8E4M3, tex.DType.kFloat4E2M1
    dq = lambda t: t.dequantize(dtype=torch.float32)
    want_from = lambda rows_dq, cu, w_dq: reference_postact(activation, rows_dq, cu, torch.stack(w_dq), I)
    state = {}

    def quantizer(amax_group=None):  # the recipe's forward rowwise quantizer (NVFP4's RHT is columnwise-only)
        if mx:
            return MXFP8Quantizer(e4m3, rowwise=True, columnwise=False)
        return NVFP4Quantizer(e2m1, rowwise=True, columnwise=False, with_amax_reduction=amax_group is not None, amax_reduction_group=amax_group)

    if kind == "deepep_te":
        import deep_ep

        align = 128  # TE's grouped quantize: every expert's rows a multiple of 128, the padding zeroed
        buffer = deep_ep.ElasticBuffer(ws.ep_group, num_max_tokens_per_rank=T_local, hidden=H, num_topk=K, explicitly_destroy=True)
        num_sms = buffer.get_theoretical_num_sms(model.E, K)
        idx64 = topk_idx.to(deep_ep.topk_idx_t)
        ones = torch.ones(T_local, K, dtype=torch.float32, device=dev)
        common = dict(num_sms=num_sms, num_max_tokens_per_rank=T_local, num_experts=model.E, expert_alignment=align, do_zero_padding=True)
        recipe = MXFP8BlockScaling() if mx else NVFP4BlockScaling()
        # TE 2.19's grouped-tensor GroupedLinear (as MoK's bench_deepep_te.py): its grouped GEMM is not ordered with
        # the current stream on either side (tmp repro 2026-10-17); kept as the baseline's own path
        lin = te.GroupedLinear(E_local, H, n_out, bias=False, params_dtype=torch.bfloat16, device=dev, use_grouped_tensor=True)
        for e in range(E_local):
            getattr(lin, f"weight{e}").data = w1_src[e]

        def run():
            recv_x, _, _, handle, _ = buffer.dispatch(x=x, topk_idx=idx64, topk_weights=ones, do_expand=True, do_cpu_sync=False, do_handle_copy=False, **common)
            # psum[i] = where expert i's valid rows end; its padded start = the previous end rounded up
            psum = handle.psum_num_recv_tokens_per_expert.to(torch.int64)
            start = torch.cat((psum.new_zeros(1), psum[:-1])).add(align - 1).div(align, rounding_mode="floor") * align
            splits = (psum - start).add(align - 1).div(align, rounding_mode="floor") * align
            with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
                h = lin(recv_x, splits)
            state["out"] = te_act(h, None)
            state["recv_x"], state["cu"] = recv_x, torch.cat((start, (start[-1] + splits[-1]).view(1)))

        def want():  # the operands as the module quantized them: each expert's slice its own tensor (own amax)
            in_q, w_q = lin._get_quantizers()[:2]
            cu = state["cu"].tolist()
            rows = torch.cat([dq(in_q[e](state["recv_x"][cu[e] : cu[e + 1]])) for e in range(E_local) if cu[e + 1] > cu[e]])
            return want_from(rows, state["cu"], [dq(w_q[e](w1_src[e])) for e in range(E_local)])

        return run, lambda: state["out"][: int(state["cu"][-1])], want, (buffer, lin, ones)

    assert kind in ("nccl_te", "nccl_ep_te"), kind
    q_x = quantizer(None if mx else ws.ep_group)
    xq = q_x(x)
    xd, xs = xq._rowwise_data, xq._rowwise_scale_inv[:T_local]  # the rows' bytes, one scale row per row
    cols, sfc = xd.shape[1], xs.shape[1]
    amax = getattr(xq, "_amax_rowwise", None)
    w_q = [quantizer()(w1_src[e]) for e in range(E_local)]
    routing = NcclRouting(topk_idx, E_local, ws.ep_group)
    pre = torch.empty(max_rows, n_out, dtype=torch.bfloat16, device=dev)
    r128 = lambda n: (n + 127) // 128 * 128

    def qview(data, sinv, rows):  # a quantized view over the rows' bytes and their scale rows (128-row padded)
        if mx:
            return MXFP8Tensor((rows, H), torch.bfloat16, rowwise_data=data, rowwise_scale_inv=sinv, columnwise_data=None,
                               columnwise_scale_inv=None, fp8_dtype=e4m3, quantizer=q_x, with_gemm_swizzled_scales=False, requires_grad=False)
        return NVFP4Tensor((rows, H), torch.bfloat16, rowwise_data=data, rowwise_scale_inv=sinv, columnwise_data=None,
                           columnwise_scale_inv=None, amax_rowwise=amax, amax_columnwise=None, fp4_dtype=e2m1, quantizer=q_x,
                           with_gemm_swizzled_scales=False, requires_grad=False)

    if kind == "nccl_ep_te":
        import nccl.ep as ep
        from nccl.ep.interop.torch import get_nccl_comm_from_group

        align = 128  # TE's scale rows per expert start 128-aligned: NCCL EP pads each expert's slot zone to it
        rows_l = routing.build()[5]  # valid rows per local expert; routing is fixed, so built once (setup, as the handle)
        cu = torch.tensor([0, *itertools.accumulate(rows_l)], dtype=torch.int32, device=dev)
        starts = [sum(r128(n) for n in rows_l[:e]) for e in range(E_local)]
        budget = W * T_local * K + (align - 1) * E_local  # static recv slots: every expert's zone padded
        comm = get_nccl_comm_from_group(ws.ep_group)
        group = ep.Group.create(comm, ep.GroupConfig(
            algorithm=ep.Algorithm.HIGH_THROUGHPUT, num_experts=model.E, max_dispatch_tokens_per_rank=T_local,
            max_recv_tokens_per_rank=budget, max_token_bytes=(cols + sfc + 15) // 16 * 16, num_topk=K, max_num_sms=sms or 0,
        ))
        counts, offsets, total = (torch.zeros(n, dtype=torch.int32, device=dev) for n in (E_local, E_local, 1))
        info = ep.LayoutInfo(expert_counters=ep.Tensor(counts), expert_offsets=ep.Tensor(offsets), recv_total_counter=ep.Tensor(total))
        idx64 = ep.Tensor(topk_idx.to(torch.int64).contiguous())  # the handle keeps the descriptor: it must outlive it
        handle = group.create_handle(ep.Layout.EXPERT_MAJOR, idx64, layout_info=info,
                                     config=ep.HandleConfig(dispatch_output_per_expert_alignment=align), stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        assert offsets.tolist() == starts and int(total) == starts[-1] + r128(rows_l[-1]), (offsets.tolist(), starts, int(total))
        recv_rows = torch.empty(budget, cols, dtype=torch.uint8, device=dev)
        recv_sf = torch.zeros(budget, sfc, dtype=torch.uint8, device=dev)  # only landed rows are written: the padding stays 0
        slot_w = torch.empty(budget, dtype=torch.float32, device=dev)  # HT forward dispatch requires the per-slot weight output
        ones = torch.ones(T_local, K, dtype=torch.float32, device=dev)
        d_in = ep.DispatchInputs(tokens=ep.Tensor(xd.view(torch.float8_e4m3fn)), topk_weights=ep.Tensor(ones), scales=ep.Tensor(xs))
        d_out = ep.DispatchOutputs(tokens=ep.Tensor(recv_rows.view(torch.float8_e4m3fn)), topk_weights=ep.Tensor(slot_w), scales=ep.Tensor(recv_sf))
        d_cfg = ep.DispatchConfig(quantization_recipe=ep.DispatchQuantizationRecipe.FWD)  # the bytes + scale rows forwarded as is
        views = [qview(recv_rows[s : s + n], recv_sf[s : s + r128(n)], n) for s, n in zip(starts, rows_l)]
        num_recv, total = int(cu[-1]), starts[-1] + r128(rows_l[-1])
        # the GEMM leg = TE's single-launch grouped GEMM (GroupedLinear's path): one quantized tensor over the zones,
        # members = each zone's valid rows (tensor_offsets in logical elements), output compact in cu order. cuBLASLt
        # wants GEMM-swizzled scales: the weights' once, the landed (compact) scale rows per call; the swizzle op
        # writes a new scale buffer and re-points the tensor at it (and is a no-op once its flag is set), so the
        # view over the zones is rebuilt per call and recv_sf stays compact for the reference
        for w in w_q:
            tex.swizzle_scales_for_gemm_(w)
        first_dims = torch.tensor(rows_l, dtype=torch.int64, device=dev)
        in_off = torch.tensor([*(s * H for s in starts), total * H], dtype=torch.int64, device=dev)
        out_off = torch.tensor([int(c) * n_out for c in cu.tolist()], dtype=torch.int64, device=dev)
        out = GroupedTensorStorage(shape=(num_recv, n_out), dtype=torch.bfloat16, num_tensors=E_local, quantizer=None,
                                   data=pre[:num_recv].reshape(-1), first_dims=first_dims, tensor_offsets=out_off)
        amax_g = None if amax is None else amax.expand(E_local).contiguous()

        def dispatch():
            handle.dispatch(d_in, d_out, config=d_cfg, stream=torch.cuda.current_stream())

        def gemm():
            whole = qview(recv_rows[:total], recv_sf[:total], total)
            tex.swizzle_scales_for_gemm_(whole)
            a = GroupedTensorStorage(shape=(total, H), dtype=torch.bfloat16, num_tensors=E_local, quantizer=q_x,
                                     data=whole._rowwise_data.reshape(-1), scale_inv=whole._rowwise_scale_inv.reshape(-1),
                                     amax=amax_g, first_dims=first_dims, tensor_offsets=in_off, with_gemm_swizzled_scales=True)
            general_grouped_gemm_for_grouped_tensor(w_q, a, out, layout="TN")
            state["out"] = te_act(pre[:num_recv], None)

        def run():
            dispatch()
            gemm()

        run.legs = {"nccl_ep alone": dispatch, "te fc1 alone": gemm}

        def want():  # the operand as delivered: each expert's zone, dequantized (the weights requantized: w_q is swizzled)
            return want_from(torch.cat([dq(v) for v, n in zip(views, rows_l) if n]), cu, [dq(quantizer()(w1_src[e])) for e in range(E_local)])

        keep = (group, handle, comm, info, idx64, d_in, d_out, recv_rows, recv_sf, slot_w, ones, pre, xq, w_q, views, out)
        return run, lambda: state["out"], want, keep

    token_of_copy = routing.sort_indices // K
    send_rows = torch.empty(T_local * K, cols, dtype=torch.uint8, device=dev)
    send_sf = torch.empty(T_local * K, sfc, dtype=torch.uint8, device=dev)
    recv_rows = torch.empty(max_rows, cols, dtype=torch.uint8, device=dev)
    recv_sf = torch.empty(max_rows, sfc, dtype=torch.uint8, device=dev)
    grouped_rows = torch.empty(max_rows, cols, dtype=torch.uint8, device=dev)
    grouped_sf = torch.empty(max_rows, sfc, dtype=torch.uint8, device=dev)
    sf_pad = torch.zeros((max_rows + 127) // 128 * 128 + E_local * 128, sfc, dtype=torch.uint8, device=dev)
    experts = torch.arange(E_local, device=dev, dtype=torch.int32)

    def run():
        send_splits, recv_splits, num_recv, recv_to_grouped, cu, rows_l = routing.build()
        torch.index_select(xd, 0, token_of_copy, out=send_rows)
        torch.index_select(xs, 0, token_of_copy, out=send_sf)
        dist.all_to_all_single(recv_rows[:num_recv], send_rows, recv_splits, send_splits, group=ws.ep_group)
        dist.all_to_all_single(recv_sf[:num_recv], send_sf, recv_splits, send_splits, group=ws.ep_group)
        grouped_rows[:num_recv].index_copy_(0, recv_to_grouped, recv_rows[:num_recv])
        grouped_sf[:num_recv].index_copy_(0, recv_to_grouped, recv_sf[:num_recv])
        # TE's scale rows per expert start 128-aligned: expert i's at (cu[i] // 128 + i) * 128
        expert_of_row = torch.repeat_interleave(experts, cu[1:] - cu[:-1], output_size=num_recv)
        padded_rows = ((cu[:-1] // 128 + experts) * 128 - cu[:-1])[expert_of_row] + torch.arange(num_recv, device=dev, dtype=torch.int32)
        sf_pad.zero_()
        sf_pad.index_copy_(0, padded_rows.to(torch.int64), grouped_sf[:num_recv])
        if state.get("rows") != rows_l:  # the per-expert views over the fixed buffers, rebuilt when the routing changes
            cu_l = [0, *itertools.accumulate(rows_l)]
            state["rows"], state["views"] = rows_l, [
                qview(grouped_rows[cu_l[i] : cu_l[i + 1]], sf_pad[(cu_l[i] // 128 + i) * 128 :][: r128(rows_l[i])], rows_l[i])
                for i in range(E_local)
            ]
        general_grouped_gemm(w_q, state["views"], [pre[:num_recv]], [None] * E_local, torch.bfloat16, single_output=True, m_splits=rows_l)
        state["out"] = te_act(pre[:num_recv], None)
        state["cu"], state["num_recv"] = cu, num_recv

    def want():  # the operand as delivered: the regrouped bytes + scale rows, dequantized
        n = state["num_recv"]
        sf = torch.zeros(r128(n), sfc, dtype=torch.uint8, device=dev)
        sf[:n] = grouped_sf[:n]
        return want_from(dq(qview(grouped_rows[:n], sf, n)), state["cu"], [dq(t) for t in w_q])

    keep = (send_rows, send_sf, recv_rows, recv_sf, grouped_rows, grouped_sf, sf_pad, pre, xq, w_q)
    return run, lambda: state["out"], want, keep


def fc1_stage_unfused(kind, activation, model, x, topk_idx, w1_cat, ws, rank, W, T_local, fmt=None, sms=None):
    """Unfused baselines for the dispatch + fc1 stage, NCCL all_to_all for the dispatch (NcclRouting's schedule per
    call) then a grouped GEMM and the activation in torch:
      "nccl_ep_te" (fmt): NVIDIA nccl-extensions' NCCL EP (`nccl.ep`, high-throughput mode, expert-major layout,
        `sms` SMs) dispatching the TE-quantized rows + scale rows as bytes into 128-aligned per-expert zones, then
        TE's single-launch grouped-tensor GEMM over the zones (GroupedLinear's path: the landed scale rows swizzled
        for cuBLASLt per call) and the activation as a TE op; the handle (routing) is built at setup. Its
        `run.legs` = the dispatch alone and the swizzle + GEMM + activation alone;
      "nccl_quack_nvfp4": x and w1 in NVFP4 (quack's recipe, x's per-tensor scale over all ranks), the packed rows
        and e4m3 scale rows over NCCL as bytes, the received scales relaid into quack's per-expert padded SF layout,
        quack's grouped varlen-M blockscaled GEMM (compile_blockscaled_gemm_tvm_ffi, 128x128 tiles), alpha and the
        activation in torch;
      "nccl_torch_bf16": bf16 rows over NCCL, torch.nn.functional.grouped_mm, the activation in torch;
      "nccl_te" (fmt mxfp8 / nvfp4): x quantized with TE's forward quantizer before the send (nvfp4: one per-tensor
        scale over all ranks), the data + scale rows over NCCL as bytes, regrouped into per-expert quantized views,
        TE's cuBLASLt grouped GEMM (general_grouped_gemm), the activation as a TE op;
      "deepep_te" (fmt): DeepEP dispatch of the bf16 rows (expanded, 128-aligned, zeroed padding: TE's grouped
        quantize), TE GroupedLinear under the format's recipe (it quantizes the landed rows per expert; nvfp4: 2D
        weights), the activation as a TE op. Checked on the dequantized operands the GEMM consumed.
    Returns (arm, postact, reference fn, keep)."""
    from quack.blockscaled.quantize import pack_scale_2d_to_blocked_contig
    from quack.blockscaled.utils import compile_blockscaled_gemm_tvm_ffi

    H, I, K, E_local, dev = model.H, model.I, model.topk, model.E // W, x.device
    max_rows = W * T_local * min(K, E_local)
    if kind in ("nccl_te", "deepep_te", "nccl_ep_te"):
        return fc1_stage_te(kind, fmt, activation, model, x, topk_idx, w1_cat, ws, rank, W, T_local, sms)
    routing = NcclRouting(topk_idx, E_local, ws.ep_group)
    token_of_copy = routing.sort_indices // K
    act_fn = TORCH_ACT[activation]
    if kind == "nccl_quack_nvfp4":
        amax = x.abs().amax()
        dist.all_reduce(amax, op=dist.ReduceOp.MAX, group=ws.ep_group)
        xq, xs, pts_x, x_dq = bf.quantize_nvfp4(x, amax=amax)
        w1_q, w1_s, pts_w, w1_dq = bf.quantize_nvfp4(w1_cat)
        alpha = float(pts_w * pts_x)
        send_rows = torch.empty(T_local * K, H // 2, dtype=torch.uint8, device=dev)
        send_sf = torch.empty(T_local * K, H // 16, dtype=torch.uint8, device=dev)
        recv_rows = torch.empty(max_rows, H // 2, dtype=torch.uint8, device=dev)
        recv_sf = torch.empty(max_rows, H // 16, dtype=torch.uint8, device=dev)
        grouped_rows = torch.empty(max_rows, H // 2, dtype=torch.uint8, device=dev)
        grouped_sf = torch.empty(max_rows, H // 16, dtype=torch.uint8, device=dev)
        sf_pad = torch.zeros((max_rows + 127) // 128 * 128 + E_local * 128, H // 16, dtype=torch.uint8, device=dev)
        pre = torch.empty(max_rows, 2 * I, dtype=torch.bfloat16, device=dev)
        mB = w1_q.view(torch.uint8).permute(1, 2, 0).view(torch.float4_e2m1fn_x2)  # (2I, H/2, E_local), batch last
        sfb = torch.stack([bf.blocked_sf(w1_s[e].view(torch.float8_e8m0fnu)).view(torch.uint8) for e in range(E_local)])
        sfb = sfb.view(torch.float8_e4m3fn)
        mA = grouped_rows.view(torch.float4_e2m1fn_x2)
        sfa_blocked = pack_scale_2d_to_blocked_contig(sf_pad.view(1, -1, H // 16).view(torch.float8_e4m3fn))
        run_gemm = compile_blockscaled_gemm_tvm_ffi(
            cutlass.Float4E2M1FN, cutlass.Float8E4M3FN, 16, cutlass.BFloat16, (128, 128), (1, 1), mA, mB, pre,
            sfa_blocked, sfb, use_clc_persistence=True, varlen_m=True,
        )
        xq_u8, xs_u8 = xq.view(torch.uint8), xs
        experts = torch.arange(E_local, device=dev, dtype=torch.int32)
        state = {}

        def run():
            send_splits, recv_splits, num_recv, recv_to_grouped, cu, _ = routing.build()
            torch.index_select(xq_u8, 0, token_of_copy, out=send_rows)
            torch.index_select(xs_u8, 0, token_of_copy, out=send_sf)
            dist.all_to_all_single(recv_rows[:num_recv], send_rows, recv_splits, send_splits, group=ws.ep_group)
            dist.all_to_all_single(recv_sf[:num_recv], send_sf, recv_splits, send_splits, group=ws.ep_group)
            grouped_rows[:num_recv].index_copy_(0, recv_to_grouped, recv_rows[:num_recv])
            grouped_sf[:num_recv].index_copy_(0, recv_to_grouped, recv_sf[:num_recv])
            # quack's padded SF layout: expert i's scale rows start at (cu[i] // 128 + i) * 128
            expert_of_row = torch.repeat_interleave(experts, cu[1:] - cu[:-1], output_size=num_recv)
            padded_rows = ((cu[:-1] // 128 + experts) * 128 - cu[:-1])[expert_of_row] + torch.arange(num_recv, device=dev, dtype=torch.int32)
            sf_pad.zero_()
            sf_pad.index_copy_(0, padded_rows.to(torch.int64), grouped_sf[:num_recv])
            sfa_blocked.copy_(pack_scale_2d_to_blocked_contig(sf_pad.view(1, -1, H // 16).view(torch.float8_e4m3fn)))
            run_gemm(mA[:num_recv], mB, pre[:num_recv], sfa_blocked, sfb, cu)
            g, u = (pre[:num_recv].float() * alpha).chunk(2, dim=-1)
            state["out"] = act_fn(g, u)
            state["cu"], state["num_recv"] = cu, num_recv

        def want():  # the bf16 token rows in the baseline's grouped order (the same schedule), requantized as sent
            send_splits, recv_splits, num_recv, recv_to_grouped, cu, _ = routing.build()
            recv_bf16 = torch.empty(num_recv, H, dtype=torch.bfloat16, device=dev)
            dist.all_to_all_single(recv_bf16, x[token_of_copy], recv_splits, send_splits, group=ws.ep_group)
            rows_bf16 = torch.empty_like(recv_bf16).index_copy_(0, recv_to_grouped, recv_bf16)
            return reference_postact(activation, bf.quantize_nvfp4(rows_bf16, amax=amax)[3], cu, w1_dq, I)

        keep = (send_rows, send_sf, recv_rows, recv_sf, grouped_rows, grouped_sf, sf_pad, pre, mB, sfb, sfa_blocked, xq, xs)
        return run, lambda: state["out"], want, keep
    assert kind == "nccl_torch_bf16", kind
    send_rows = torch.empty(T_local * K, H, dtype=torch.bfloat16, device=dev)
    recv_rows = torch.empty(max_rows, H, dtype=torch.bfloat16, device=dev)
    grouped_rows = torch.empty(max_rows, H, dtype=torch.bfloat16, device=dev)
    w1_t = w1_cat.transpose(1, 2).contiguous()  # (E_local, H, 2I)
    state = {}

    def run():
        send_splits, recv_splits, num_recv, recv_to_grouped, cu, _ = routing.build()
        torch.index_select(x, 0, token_of_copy, out=send_rows)
        dist.all_to_all_single(recv_rows[:num_recv], send_rows, recv_splits, send_splits, group=ws.ep_group)
        grouped_rows[:num_recv].index_copy_(0, recv_to_grouped, recv_rows[:num_recv])
        pre = F.grouped_mm(grouped_rows[:num_recv], w1_t, offs=cu[1:])
        g, u = pre.chunk(2, dim=-1)
        state["out"] = act_fn(g, u)
        state["cu"], state["num_recv"] = cu, num_recv

    def want():
        return reference_postact(activation, grouped_rows[: state["num_recv"]], state["cu"], w1_cat, I)

    return run, lambda: state["out"], want, (send_rows, recv_rows, grouped_rows, w1_t)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=list(MODEL_CONFIGS), default="dsv3")
    p.add_argument("--t-local", type=int, default=1024)
    p.add_argument("--activations", nargs="+", default=["swiglu", "geglu", "relu_sq"], choices=sorted(TORCH_ACT))
    p.add_argument("--formats", nargs="+", default=["bf16"], choices=["none", "bf16", *FORMATS], help="operand formats of the fused stage; mxfp8 / nvfp4 rows carry their bounds and take gated activations only; none: baselines only")
    p.add_argument("--baselines", nargs="*", default=[], choices=["nccl_quack_nvfp4", "nccl_torch_bf16", "nccl_te", "deepep_te", "nccl_ep_te"], help="unfused dispatch + fc1 baselines (see fc1_stage_unfused), one per activation; nccl_te / deepep_te / nccl_ep_te one per --formats format")
    p.add_argument("--nccl-ep-sms", type=str, default="16,32,64,all", help="nccl_ep_te: NCCL EP's SM budget, one arm per comma-separated value (its HT default 16; all = the device's SM count: nothing runs beside the dispatch in the unfused chain)")
    p.add_argument("--ref_check", action="store_true")
    args = p.parse_args()
    args.sonic_metadata = False

    rank, W = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    dev = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(dev)
    dist.init_process_group("nccl", device_id=dev)
    model = MODEL_CONFIGS[args.model]
    H, I, E, K = model
    T_local, E_local = args.t_local, E // W
    max_rows = W * T_local * min(K, E_local)
    ws = SymmMemManager(None, dev)._get_or_alloc(T_local, H, K, E_local, torch.bfloat16, DispatchMode.RANK_DEDUP_DISPATCH_TRITON)
    g = torch.Generator(device=dev).manual_seed(123 + rank)
    x = torch.randn(T_local, H, generator=g, device=dev, dtype=torch.bfloat16)
    topk_idx = torch.rand(T_local, E, generator=g, device=dev).topk(K, dim=-1).indices.to(torch.int32)
    scores = torch.rand(T_local, K, generator=g, device=dev).softmax(-1)
    local = [expert_weights(e, H, I, dev) for e in range(rank * E_local, (rank + 1) * E_local)]
    w1_cat = torch.stack([w1 for w1, _ in local])
    w2 = torch.stack([w2 for _, w2 in local])
    del local
    shared = (model, x, topk_idx, scores, w1_cat, None, w2, ws, rank, W, T_local, args)

    with torch.no_grad():
        run = bf.BASELINES["quack_intra_swap"](*shared)["quack_intra_swap"].run  # the walk, the landed rows, the scheduler
        pt = run.parts
        prologue = run.steps[run.step_names.index("prologue")]
        _, sched1, varlen = pt.rt1
        walk1, rows, cu = pt.walk1, pt.rows, pt.cu
        total_m = int(cu[-1].item())
        w1_plain = w1_cat[:, :I].contiguous()  # non-gated fc1: the gate half's weights
        desc = torch.empty(torch.cuda.get_device_properties(dev).multi_processor_count * 16, dtype=torch.int64, device=dev)
        arms, posts, flops, keep = {}, {}, {}, []
        block_rows = bf.compile_gemm1_gated_swap(cutlass.BFloat16, model, W, T_local, intra_pull=H, gather_B=False)[2]
        for fmt in [f for f in args.formats if f in FORMATS]:
            for act in args.activations:
                fmt_arms, fmt_posts, pinned = fc1_stage_quantized(
                    fmt, act, model, x, topk_idx, w1_cat, ws, rank, W, T_local, block_rows, pt.clusters, desc, rows[:total_m]
                )
                arms.update(fmt_arms)
                posts.update(fmt_posts)
                keep.append(pinned)
                gemm_flops = 2 * total_m * H * (I if act in act_fn_map else 2 * I)
                for name in fmt_arms:
                    flops[name] = None if name.startswith("pull") or " pass " in name else gemm_flops
        te_kinds = ("nccl_te", "deepep_te", "nccl_ep_te")  # one arm per quantized format
        n_sm = torch.cuda.get_device_properties(dev).multi_processor_count
        ep_sms = [n_sm if s == "all" else int(s) for s in args.nccl_ep_sms.split(",")]
        for kind in args.baselines:
            for act in args.activations:
                for fmt in [f for f in args.formats if f in FORMATS] if kind in te_kinds else [None]:
                    assert kind in te_kinds or act in gate_fn_map, f"{kind}: gated activations only, got {act}"
                    for sms in ep_sms if kind == "nccl_ep_te" else [None]:
                        run_b, out_b, want_b, pinned = fc1_stage_unfused(kind, act, model, x, topk_idx, w1_cat, ws, rank, W, T_local, fmt, sms)
                        tag = f" {act}" + (f" {fmt}" if fmt else "")
                        name = f"{kind}{'' if sms is None else f'@{sms}'}{tag}"
                        arms[name], flops[name] = run_b, 2 * total_m * H * (2 * I if act in gate_fn_map else I)
                        keep.append(pinned)
                        posts[name] = (out_b, want_b)
                        for leg, fn in getattr(run_b, "legs", {}).items():  # the GEMM leg is the same at every SM count
                            if leg.startswith("nccl_ep") or sms == ep_sms[0]:
                                leg_name = f"{leg}{f'@{sms}' if leg.startswith('nccl_ep') else ''}{tag}"
                                arms[leg_name], flops[leg_name] = fn, None
        for act in args.activations if "bf16" in args.formats else ():
            gated = act in gate_fn_map
            assert gated or act in act_fn_map, act
            fn, Cls, _ = bf.compile_gemm1_gated_swap(cutlass.BFloat16, model, W, T_local, intra_pull=H, gather_B=False,
                                                     activation=act, gated=gated)
            a = torch.empty(max_rows, I, dtype=torch.bfloat16, device=dev)
            epi = Cls.EpilogueArguments(mAuxOut=a.T)
            w1 = pt.w1_il8 if gated else w1_plain

            def gemm1(fn=fn, w1=w1, epi=epi):
                fn(w1, rows, None, None, epi, sched1, varlen, None, None, walk1.args, desc)

            name = f"dispatch+gemm1 {act}"
            arms[name] = lambda gemm1=gemm1: (prologue(), gemm1())
            posts[name] = (a, lambda act=act: reference_postact(act, rows, cu, w1_cat, I))
            flops[name] = 2 * total_m * H * (2 * I if gated else I)
        for fn in arms.values():  # every kernel once (module loading) before the timed calls
            fn()
        torch.cuda.synchronize()
        dist.barrier()
        checks = {}
        if args.ref_check:
            for name, (a, want_fn) in posts.items():
                if not callable(a):
                    a.zero_()
                torch.cuda.synchronize()
                dist.barrier()
                arms[name]()
                torch.cuda.synchronize()
                dist.barrier()
                want = want_fn()
                got = a().float() if callable(a) else a[:total_m].float()  # a baseline's output is allocated per call
                err = ((got - want).abs().max() / want.abs().max()).item()
                checks[name] = (err < 2**-6, err)
        ms = bench_all(arms, ws.x_hdl)
        unstable = [None] * W
        dist.all_gather_object(unstable, [k for k, v in ms.items() if arm_stats(v)[3]])
        rerun = [k for k in arms if any(k in u for u in unstable)]
        if rerun:
            again = bench_all({k: arms[k] for k in rerun}, ws.x_hdl)
            ms.update({k: v for k, v in again.items() if not arm_stats(v)[3]})

    stats = {k: arm_stats(v) for k, v in ms.items()}
    torch.cuda.synchronize()
    gathered = [None] * W
    dist.all_gather_object(gathered, (stats, checks))
    if rank == 0:
        med = {k: max((s[k] for s, _ in gathered), key=lambda t: t[0]) for k in ms}
        print(
            f"<bf16 fused dispatch + fc1 {args.model} (H={H} I={I} E={E} top-{K}) | W={W} T_local={T_local} total_m={total_m} "
            f"(rank 0) | prologue + GEMM1 per call | warm {WARM_S * 1e3:.0f} ms, {REPS} x {WINDOW_S * 1e3:.0f} ms blocks, rerun={rerun}>"
        )
        for k, (m, lo, hi, _) in med.items():
            flag = "  UNSTABLE" if any(s[k][3] for s, _ in gathered) else ""
            rate = f"{flops[k] / m / 1e9:5.0f} TFLOP/s" if flops[k] else ""
            print(f"  {k + ':':<30} {m * 1e3:7.0f} us  [{lo * 1e3:.0f}-{hi * 1e3:.0f}]  {rate}{flag}")
        for name in checks:
            ok = all(c[name][0] for _, c in gathered)
            err = max(c[name][1] for _, c in gathered)
            print(f"  {name} ref: {'ok' if ok else 'MISMATCH'} max rel err {err:.3e}")
    dist.barrier()
    os._exit(0)


if __name__ == "__main__":
    main()
