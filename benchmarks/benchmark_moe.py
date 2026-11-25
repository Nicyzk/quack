# Copyright (c) 2026, QuACK team.
"""MoE expert-parallel benchmark: the dispatch and combine operators (quack/distributed/dispatch.py,
combine.py) fused into the grouped expert GEMMs vs NCCL all-to-all + torch grouped GEMMs. world_size ranks,
each holding E / world_size experts; uniform top-k routing seeded per rank.

Stages (--stage):
  forward       x -> out: DispatchRunner.prepare (the x publish + one launch: top-k exchange, routing, walk), FC1
                with SwiGLU and the pull fused, FC2 with the combine slot store fused, a barrier, reduce_over_k
                with the router scores
  dispatch_fc1  prepare + FC1 (--activation swiglu | relu_sq | none; --dtype bfloat16 | mxfp8_e4m3 | nvfp4)
  fc2_combine   FC2 + combine + barrier + reduce_over_k with the RMSNorm statistic, rstd finalized (one
                routing, a random FC2 input)
  backward      dout -> dx, dscores, dW1, dW2 (bfloat16, SwiGLU, SM100): prepare of dout + the FC2 dgrad with
                the pull and dSwiGLU fused, the score grads reduce-scattered, the FC1 dgrad with the combine
                slot store fused, a barrier, reduce_over_k, the two weight grads as varlen-K GEMMs; the
                forward's saved activations are built once, outside the timed call

--shared-expert (forward, backward): one shared expert, a dense SwiGLU MLP of the same widths on every local
token, added to the routed result: unfused dense GEMMs through quack's interface in quack_ep (gemm_act, gemm;
gemm_dact in the backward), torch in nccl_seq; its FLOPs and weight grads (dw1s, dw2s) are counted and checked.

Methods:
  quack_ep  the fused operators. SM100: swap-AB (tokens on N, cu_seqlens_n), tile_M 256 x tile_N, cluster
             (2,1), tile_N from the expected rows per expert (default_tile_n); SM90: varlen_m (tokens on M,
             cu_seqlens_m), 128x256 one CTA
  nccl_seq   NCCL all_to_all_single + torch._grouped_mm in bfloat16 (every --dtype), timed last as the
             contention canary; on the same (dequantized) operands it is the reference quack_ep is checked
             against before timing

Protocol = the TP benches' (quack.bench.bench_utils.bench_l2_rotate_dist): one captured 200-call window over
rotated per-call inputs, warmed by replaying itself, cross-rank max, --rounds rounds. TFLOPS on the stage's
GEMM FLOPs over rank 0's routed rows.

  torchrun --nproc_per_node=8 benchmarks/benchmark_moe.py --stage forward
"""

import argparse
import math
import os

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from quack.bench.bench_utils import bench_l2_rotate_dist, l2_rotate_arg_sets_dist
from quack.blockscaled.operand import BlockScaledOperand
from quack.blockscaled.quantize import nvfp4_per_tensor_scale, unpack_scale_blocked_to_2d
from quack.cute_dsl_utils import get_device_capacity
from quack.distributed.combine import CombineRunner
from quack.distributed.dispatch import DispatchRunner
from quack.distributed.reduce_over_k import reduce_over_k
from quack.distributed.reduce_over_k_epilogue import sqsum_partial
from quack.epilogue.library import dswiglu_moe_swap_mod, linear_act_mod
from quack.gemm import gemm as quack_gemm
from quack.gemm_config import cta_tile_shape_m
from quack.gemm_interface import (
    act_to_pytorch_fn_map,
    gated_to_pytorch_fn_map,
    gemm,
    gemm_act,
    gemm_dact,
)
from quack.rmsnorm import rmsnorm_ref

DEFAULT_SHAPES = [
    (7168, 3072, 384, 6),  # DeepSeek-V4-Pro
    (7168, 2048, 384, 8),  # Kimi K2 / K2.7
    (6144, 2048, 256, 8),  # GLM-5.2
]  # (hidden, intermediate, experts, top-k)
EPS = 1e-6  # rmsnorm_ref's default


def _print0(rank, *args):
    if rank == 0:
        print(*args, flush=True)


def _check(rank, name, d, ref, tol) -> bool:
    err = (d.float() - ref).abs().max().item()
    flag = torch.tensor([1.0 if err > tol else 0.0], device=d.device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    if flag.item() > 0:
        _print0(
            rank,
            f"  [{name}] SKIP: correctness check failed (err={err:.3e} "
            f"tol={tol:.3e}) — not timing a wrong result",
        )
        return False
    _print0(rank, f"  [{name}] correctness OK (err={err:.2e})")
    return True


def default_tile_n(t_local, world_size, topk, experts):
    """The swap-AB GEMMs' tile_N (the even-split token block): the fewest 256-row blocks covering the expected
    rows per expert, each sized to the expected rows plus their square root, rounded up to 32, in [64, 256]."""
    expected = t_local * world_size * topk / experts
    blocks = math.ceil(expected / 256)
    return int(min(256, max(64, math.ceil((expected + expected**0.5) / blocks / 32) * 32)))


def interleave_gate_up(w, r):
    """(E, 2I, X) [gate; up] -> gate / up rows interleaved per r: 8 for the swap-AB M-paired store, 1 for the
    N-paired (quack.gemm_interface._concat_interleave)."""
    E, I2, X = w.shape
    g, u = w.chunk(2, dim=1)
    return torch.stack(
        [g.view(E, I2 // (2 * r), r, X), u.view(E, I2 // (2 * r), r, X)], dim=2
    ).reshape(E, I2, X)


def torch_act(h, activation):
    if activation in gated_to_pytorch_fn_map:
        return gated_to_pytorch_fn_map[activation](*h.chunk(2, dim=-1))
    return act_to_pytorch_fn_map[activation](h)


def bench_shape(shape, t_local, rank, world_size, group, sync_hdl, device, args):
    H, I, E, K = shape
    e_local, bf16 = E // world_size, torch.bfloat16
    sm100 = get_device_capacity(device)[0] >= 10
    bs_format = None if args.dtype == "bfloat16" else args.dtype
    assert bs_format is None or (sm100 and args.stage == "dispatch_fc1"), (
        "block-scaled: the dispatch_fc1 stage on SM100"
    )
    assert args.stage != "backward" or sm100, (
        "backward: the swap-AB operators (cu_seqlens_n is SM100+)"
    )
    activation = "swiglu" if args.stage in ("forward", "backward") else args.activation
    activation = None if activation == "none" else activation
    tile_n = args.tile_n or default_tile_n(t_local, world_size, K, E)
    # SM100 swap-AB: (A, B) = (weights, rows), tokens on N; SM90 varlen_m: (rows, weights), tokens on M
    gemm_cfg = (
        dict(tile_M=256, tile_N=tile_n, cluster_M=2, cluster_N=1)
        if sm100
        else dict(tile_M=128, tile_N=256, cluster_M=1, cluster_N=1)
    )
    block_rows = tile_n if sm100 else 128  # the gated unit along the ragged dim
    cu_kw = "cu_seqlens_n" if sm100 else "cu_seqlens_m"

    shared_flops = 0
    if (
        args.shared_expert
    ):  # one dense SwiGLU MLP on every local token: the same weights on every rank
        assert args.stage in ("forward", "backward"), (
            "--shared-expert: the forward and backward stages"
        )
        torch.manual_seed(4321)
        w1s = torch.randn(2 * I, H, device=device, dtype=bf16) / math.sqrt(H)  # [gate; up]
        w2s = torch.randn(H, I, device=device, dtype=bf16) / math.sqrt(I)
        w1s_il = interleave_gate_up(w1s[None], 1)[
            0
        ]  # (gate_i, up_i) pairs: gemm_act's gated layout
        w1s_t, w2s_t = w1s_il.T.contiguous(), w2s.T.contiguous()  # (H, 2I), (I, H): gemm's (K, N)
        a_shared = torch.empty(t_local, I, device=device, dtype=bf16)
        y_shared = torch.empty(t_local, H, device=device, dtype=bf16)
        shared_flops = 2 * t_local * H * 2 * I + 2 * t_local * I * H
    torch.manual_seed(1000 + rank)
    topk = torch.rand(t_local, E, device=device).topk(K, dim=-1).indices.to(torch.int32)
    scores = torch.rand(t_local, K, device=device).softmax(-1)
    x = torch.randn(t_local, H, device=device, dtype=bf16)
    w1 = torch.randn(e_local, 2 * I, H, device=device, dtype=bf16) / math.sqrt(H)  # [gate; up]
    w2 = torch.randn(e_local, H, I, device=device, dtype=bf16) / math.sqrt(I)
    topk_g = torch.empty(world_size, t_local, K, dtype=torch.int32, device=device)
    dist.all_gather_into_tensor(topk_g, topk, group=group)
    max_rows = (
        -(-int((topk_g // e_local == rank).sum()) // 512) * 512
    )  # this routing's rows, rounded
    n1 = I if activation == "relu_sq" else 2 * I  # FC1's weight rows
    w1_ref = w1[:, :n1]

    # nccl_seq: each (token, k) sent to its expert's rank, packed per local expert in the dispatch runner's
    # row order (source rank, token, k), the grouped GEMMs, and back
    f = topk.reshape(-1).long()
    dst = f // e_local
    order = torch.argsort(dst, stable=True)
    send_tok = order // K
    in_split = torch.bincount(dst, minlength=world_size)
    out_split = torch.empty_like(in_split)
    dist.all_to_all_single(out_split, in_split, group=group)
    in_split, out_split = in_split.tolist(), out_split.tolist()
    recv_e = torch.empty(sum(out_split), dtype=torch.long, device=device)
    dist.all_to_all_single(
        recv_e, (f[order] % e_local).contiguous(), out_split, in_split, group=group
    )
    perm = torch.argsort(recv_e, stable=True)
    inv_perm = torch.argsort(perm)
    offs = torch.bincount(recv_e, minlength=e_local).cumsum(0).to(torch.int32)

    def nccl_dispatch(x_t):
        recv = x_t.new_empty(sum(out_split), H)
        dist.all_to_all_single(recv, x_t[send_tok], out_split, in_split, group=group)
        return recv[perm]

    def grouped_mm(rows, w):  # w (e_local, N, K)
        return torch._grouped_mm(rows, w.transpose(1, 2), offs=offs)

    def nccl_combine(y, s_t):
        back = y.new_empty(len(order), H)
        dist.all_to_all_single(back, y[inv_perm], in_split, out_split, group=group)
        out = torch.zeros(t_local, H, dtype=torch.float32, device=device)
        back = back.float() if s_t is None else back.float() * s_t.reshape(-1)[order, None]
        return out.index_add_(0, send_tok, back).to(bf16)

    # quack_ep
    dispatch = DispatchRunner(
        t_local, K, H, bs_format or bf16, max_rows, block_rows, e_local, group=group
    )
    combine = CombineRunner(t_local, K, H, bf16, max_rows, group=group)
    a = torch.empty(max_rows, n1 if activation is None else I, device=device, dtype=bf16)
    w1_k = interleave_gate_up(w1, 8 if sm100 else 1) if activation == "swiglu" else w1_ref
    epi, sf_kw, x_in, x_ref = {}, {}, (x,), x
    if bs_format is not None:
        pts_w = pts_x = None
        if bs_format == "nvfp4":  # per-tensor scales, x's shared by every rank (one alpha)
            amax = x.abs().amax().float()
            dist.all_reduce(amax, op=dist.ReduceOp.MAX, group=group)
            pts_w, pts_x = nvfp4_per_tensor_scale(w1_ref.abs().amax()), nvfp4_per_tensor_scale(amax)
            epi["alpha"] = float(pts_w * pts_x)
        wop_ref = BlockScaledOperand.quantize(w1_ref, bs_format, per_tensor_scale=pts_w)
        wop = BlockScaledOperand.quantize(w1_k, bs_format, per_tensor_scale=pts_w)
        xop = BlockScaledOperand.quantize(x, bs_format, per_tensor_scale=pts_x)
        vec = xop.format.sf_vec_size
        x_in = (xop.qdata, unpack_scale_blocked_to_2d(xop.scale[None], t_local, H // vec)[0])
        sf_kw = dict(
            SFA=wop.scale, SFB=dispatch.sf_pool, bs_format_a=bs_format, bs_format_b=bs_format
        )
        w1_k = wop.qdata
        w1_ref = wop_ref.dequantize(torch.float32).to(
            bf16
        )  # quantization is row-local: the same values
        x_ref = xop.dequantize(torch.float32).to(bf16)
    mod = None
    if activation is not None:
        mod = linear_act_mod(
            activation,
            gated=activation in gated_to_pytorch_fn_map,
            has_c=False,
            has_rowvec=False,
            has_colvec=False,
            has_alpha=bs_format == "nvfp4",
        )

    def fc1(dist_op_args):
        A, B = (w1_k, dispatch.rows) if sm100 else (dispatch.rows, w1_k)
        kw = {cu_kw: dispatch.cu, "dist_op": dispatch.op, "dist_op_args": dist_op_args, **sf_kw}
        out_a = a.T if sm100 else a  # tokens on N: the store writes m-major
        if mod is None:
            quack_gemm(A, B, out_a, None, None, **gemm_cfg, alpha=epi.get("alpha", 1.0), **kw)
        else:
            mod.gemm(A, B, None, None, **gemm_cfg, epi_args={**epi, "mAuxOut": out_a}, **kw)

    def fc2(a_t):
        A, B = (w2, a_t) if sm100 else (a_t, w2)
        quack_gemm(
            A,
            B,
            combine.d,
            None,
            None,
            **gemm_cfg,
            **{cu_kw: dispatch.cu},
            dist_op=combine.op,
            dist_op_args=combine.prepare(dispatch.s_scatter),
        )

    out = torch.empty(t_local, H, device=device, dtype=bf16)
    partials = torch.zeros(1, t_local, -(-H // 1024), device=device, dtype=torch.float32)
    rstd = torch.empty(t_local, 1, device=device, dtype=torch.float32)
    dispatch.prepare(x_in[0], topk, *x_in[1:])  # the routing fc2_combine times against
    torch.cuda.synchronize(device)
    total = int(dispatch.cu[-1])
    rows = torch.tensor([total], device=device)
    dist.broadcast(rows, 0, group=group)  # FLOPs over rank 0's rows
    rows = int(rows.item())

    if args.stage == "forward":
        flops = 2 * rows * H * n1 + 2 * rows * I * H + shared_flops

        def run_quack_ep(x_t, s_t):
            fc1(dispatch.prepare(x_t, topk))
            fc2(a)
            sync_hdl.barrier(channel=1)  # every rank's drains have landed
            reduce_over_k(combine.slots, s_t, out)
            if args.shared_expert:  # two dense GEMMs and an add into the routed output
                gemm_act(x_t, w1s_t, activation="swiglu", postact_out=a_shared, store_preact=False)
                gemm(a_shared, w2s_t, out=y_shared)
                out.add_(y_shared)
            return out

        def run_nccl_seq(x_t, s_t):
            h = torch_act(grouped_mm(nccl_dispatch(x_t), w1), activation)
            y = nccl_combine(grouped_mm(h, w2), s_t)
            if args.shared_expert:
                y = y + torch_act(x_t @ w1s.T, activation) @ w2s.T
            return y

        sets = l2_rotate_arg_sets_dist(x, scores)
        ref = run_nccl_seq(*sets[0])
    elif args.stage == "dispatch_fc1":
        flops = 2 * rows * H * n1

        def run_quack_ep(*x_t):
            fc1(dispatch.prepare(x_t[0], topk, *x_t[1:]))
            return a[:total]

        def run_nccl_seq(*x_t):  # bfloat16 in every --dtype
            return torch_act(
                grouped_mm(nccl_dispatch(x if bs_format else x_t[0]), w1[:, :n1]), activation
            )

        sets = l2_rotate_arg_sets_dist(*x_in)
        ref = torch_act(grouped_mm(nccl_dispatch(x_ref), w1_ref), activation)
    elif args.stage == "backward":
        # the two dgrads and the two wgrads
        flops = 2 * (2 * rows * H * n1 + 2 * rows * I * H + shared_flops)
        # the forward's saved activations, packed per local expert: x rows, scores, the preact
        x_rows = nccl_dispatch(x)
        score = torch.zeros(max_rows, device=device, dtype=torch.float32)
        recv_s = torch.empty(total, device=device, dtype=torch.float32)
        dist.all_to_all_single(recv_s, scores.reshape(-1)[order], out_split, in_split, group=group)
        score[:total] = recv_s[perm]
        w1_pw = interleave_gate_up(
            w1, 1
        )  # pairwise (gate_i, up_i) rows: the dgated epilogue's layout
        h_ref = grouped_mm(x_rows, w1)
        h = torch.zeros(max_rows, 2 * I, device=device, dtype=bf16)
        h[:total] = grouped_mm(x_rows, w1_pw)
        w2t = w2.transpose(
            1, 2
        ).contiguous()  # (E, I, H): the transposed weight copies, once per step
        w1t = w1_pw.transpose(1, 2).contiguous()  # (E, H, 2I)
        dout = torch.randn(t_local, H, device=device, dtype=bf16)
        if (
            args.shared_expert
        ):  # its saved preact, in gemm_dact's pairs and in [gate | up] for the reference
            hs, hs_ref = x @ w1s_il.T, x @ w1s.T
            dhs = torch.empty(t_local, 2 * I, device=device, dtype=bf16)
            dw1s, dw2s = torch.empty_like(w1s), torch.empty_like(w2s)
            dw1s_ref, dw2s_ref = torch.empty_like(w1s), torch.empty_like(w2s)
        dh = torch.zeros(max_rows, 2 * I, device=device, dtype=bf16)
        a_s = torch.zeros(max_rows, I, device=device, dtype=bf16)
        cta_tile_m = cta_tile_shape_m(256, 2, get_device_capacity(device)[0], False)
        dscore = torch.zeros(-(-I // cta_tile_m), max_rows, device=device, dtype=torch.float32)
        slot = dispatch.s_scatter[:total].long()
        ds_g = torch.zeros(world_size * t_local * K, device=device, dtype=torch.float32)
        ds = torch.empty(t_local * K, device=device, dtype=torch.float32)
        dw1, dw2 = torch.empty_like(w1), torch.empty_like(w2)
        dw1_ref, dw2_ref, ds_ref = torch.empty_like(w1), torch.empty_like(w2), torch.empty_like(ds)
        bounds = [0] + offs.tolist()

        def run_quack_ep(dout_t):
            # dispatch of dout fused with the fc2 dgrad and dSwiGLU: dh, a * s, the score grad partials
            dswiglu_moe_swap_mod.gemm(
                w2t,
                dispatch.rows,
                dh.T,
                h.T,
                epi_args={"score": score, "postact": a_s.T, "dscore": dscore},
                **gemm_cfg,
                cu_seqlens_n=dispatch.cu,
                dist_op=dispatch.op,
                dist_op_args=dispatch.prepare(dout_t, topk),
            )
            ds_g[slot] = dscore.sum(0)[:total]
            dist.reduce_scatter_tensor(ds, ds_g, group=group)
            # the fc1 dgrad drained into the origins' slots, then the sum over k
            quack_gemm(
                w1t,
                dh,
                combine.d,
                None,
                None,
                **gemm_cfg,
                cu_seqlens_n=dispatch.cu,
                dist_op=combine.op,
                dist_op_args=combine.prepare(dispatch.s_scatter),
            )
            sync_hdl.barrier(channel=1)
            reduce_over_k(combine.slots, None, out)
            # the weight grads as varlen-K GEMMs over the packed rows
            wgrad = dict(tile_M=128, tile_N=128, cluster_M=1, cluster_N=1, cu_seqlens_k=dispatch.cu)
            quack_gemm(dispatch.rows[:total].T, a_s[:total].T, dw2, None, None, **wgrad)
            quack_gemm(dh[:total].T, x_rows.T, dw1, None, None, **wgrad)
            if (
                args.shared_expert
            ):  # dense: the dgrad with dSwiGLU, the two wgrads, the fc1 dgrad added to dx
                gemm_dact(
                    dout_t, w2s_t.T, hs, activation="swiglu", dx_out=dhs, postact_out=a_shared
                )
                gemm(dout_t.T, a_shared, out=dw2s)
                gemm(dhs.T, x, out=dw1s)
                gemm(dhs, w1s_t.T, out=y_shared)
                out.add_(y_shared)
            return out

        def run_nccl_seq(dout_t):
            dy = nccl_dispatch(dout_t)
            da = grouped_mm(dy, w2t).float()
            g, u = h_ref.float().chunk(2, dim=-1)
            sg = torch.sigmoid(g)
            da_s = da * score[:total, None]
            dh_t = torch.cat([da_s * u * sg * (1 + g * (1 - sg)), da_s * g * sg], dim=-1).to(bf16)
            back = torch.empty(len(order), device=device, dtype=torch.float32)
            dist.all_to_all_single(
                back, (da * g * sg * u).sum(-1)[inv_perm], in_split, out_split, group=group
            )
            ds_ref[order] = back
            as_t = (g * sg * u * score[:total, None]).to(bf16)
            for e in range(e_local):  # torch has no grouped varlen-K GEMM
                r = slice(bounds[e], bounds[e + 1])
                torch.matmul(dy[r].T, as_t[r], out=dw2_ref[e])
                torch.matmul(dh_t[r].T, x_rows[r], out=dw1_ref[e])
            dx = nccl_combine(grouped_mm(dh_t, w1.transpose(1, 2)), None)
            if args.shared_expert:
                das = (dout_t @ w2s).float()
                gs, us = hs_ref.float().chunk(2, dim=-1)
                sgs = torch.sigmoid(gs)
                dhs_t = torch.cat(
                    [das * us * sgs * (1 + gs * (1 - sgs)), das * gs * sgs], dim=-1
                ).to(bf16)
                torch.matmul(dout_t.T, (gs * sgs * us).to(bf16), out=dw2s_ref)
                torch.matmul(dhs_t.T, x, out=dw1s_ref)
                dx = dx + dhs_t @ w1s
            return dx

        sets = l2_rotate_arg_sets_dist(dout)
        ref = run_nccl_seq(*sets[0])
    else:  # fc2_combine
        flops = 2 * rows * I * H
        a.copy_(torch.randn_like(a) / math.sqrt(I))

        def run_quack_ep(a_t, s_t):
            fc2(a_t)
            sync_hdl.barrier(channel=1)
            sums = reduce_over_k(
                combine.slots, s_t, out, epilogue=sqsum_partial, epi_args={"sqsum": partials}
            )
            torch.rsqrt(sums["sqsum"].view(t_local, 1) / H + EPS, out=rstd)
            return out

        def run_nccl_seq(a_t, s_t):
            y = nccl_combine(grouped_mm(a_t[:total], w2), s_t)
            return y, rmsnorm_ref(y)

        sets = l2_rotate_arg_sets_dist(a, scores)
        ref = run_nccl_seq(*sets[0])[0]

    # loose correctness checks before timing: quack_ep against nccl_seq on the same operands
    _print0(rank, f"H={H} I={I} E={E} topk={K} T={t_local} rows={rows} tile_N={tile_n}")
    ref = ref.float()
    d = run_quack_ep(*sets[0])
    torch.cuda.synchronize(device)
    ok = _check(rank, "quack_ep", d, ref, ref.abs().max().item() * 5e-2 + 1e-2)
    if ok and args.stage == "fc2_combine":  # the statistic, finalized
        rstd_ref = torch.rsqrt(ref.square().mean(-1, keepdim=True) + EPS)
        ok = _check(
            rank, "quack_ep rstd", rstd, rstd_ref, rstd_ref.abs().max().item() * 5e-2 + 1e-2
        )
    if ok and args.stage == "backward":  # the other grads; dw1 in the pairwise rows
        grads = [
            ("dscores", ds, ds_ref),
            ("dw1", dw1, interleave_gate_up(dw1_ref, 1)),
            ("dw2", dw2, dw2_ref),
        ]
        if args.shared_expert:
            grads += [
                ("dw1s", dw1s, interleave_gate_up(dw1s_ref[None], 1)[0]),
                ("dw2s", dw2s, dw2s_ref),
            ]
        for name, got, grad_ref in grads:
            tol = grad_ref.float().abs().max().item() * 5e-2 + 1e-2
            ok = _check(rank, f"quack_ep {name}", got, grad_ref.float(), tol) and ok
    if not ok:
        dist.barrier(group=group)
        return

    results = {}
    for name, fn in (("quack_ep", run_quack_ep), ("nccl_seq", run_nccl_seq)):  # the reference last
        results[name] = [bench_l2_rotate_dist(fn, sets) for _ in range(args.rounds)]  # ms per call
    for name, ms in results.items():
        _print0(
            rank, f"  {name:<10} {min(ms):8.3f} ms  {flops / (min(ms) * 1e-3) / 1e12:6.0f} TFLOPS"
        )
    dist.barrier(group=group)  # peers may still be pulling from this rank's buffers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", default="forward", choices=["forward", "dispatch_fc1", "fc2_combine", "backward"]
    )
    parser.add_argument(
        "--shapes",
        type=str,
        default=";".join(",".join(map(str, s)) for s in DEFAULT_SHAPES),
        help="hidden,intermediate,experts,topk;...",
    )
    parser.add_argument("--t-local", type=int, nargs="+", default=[2048, 4096, 8192])
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "mxfp8_e4m3", "nvfp4"])
    parser.add_argument("--activation", default="swiglu", choices=["swiglu", "relu_sq", "none"])
    parser.add_argument(
        "--tile-n", type=int, default=None, help="SM100 token block (default_tile_n)"
    )
    parser.add_argument("--shared-expert", action="store_true")
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank % torch.cuda.device_count())
    device = torch.device("cuda", rank % torch.cuda.device_count())
    dist.init_process_group("nccl", device_id=device)
    group = dist.group.WORLD
    sync_buf = symm_mem.empty(1, dtype=torch.int32, device=device)
    sync_hdl = symm_mem.rendezvous(sync_buf, group=group)  # the in-chain device barrier
    if rank == 0:
        print(f"MoE EP world_size={world_size} {args.stage} {args.dtype}, ms per call (best round)")
    for shape in args.shapes.split(";"):
        for t_local in args.t_local:
            bench_shape(
                tuple(map(int, shape.split(","))),
                t_local,
                rank,
                world_size,
                group,
                sync_hdl,
                device,
                args,
            )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
