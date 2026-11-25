# Copyright (C) 2026, Tri Dao.
"""The MoE EP backward on two ranks, sonic's chain on the swap-AB operators through quack.gemm.gemm (bf16):
  1. dispatch of dout fused with the fc2 dgrad: A = W2^T per expert, B = the pulled dout rows packed per expert
     (DispatchPull), D = da^T;
  2. dswiglu with the router score in one torch pass: dh, the per-row score grads (scattered to their slots and
     reduce-scattered to the tokens' ranks) and a * s;
  3. the fc1 dgrad fused with combine: A = W1^T on dh, each token's dx slice drained into its origin's slot
     (CombineStore), then reduce_over_k;
  4. the weight grads as varlen-K GEMMs over the packed rows (dW2 = dy^T (a s), dW1 = dh^T x).
Every grad is checked against an fp32 autograd reference over the whole problem (max rel err < 2^-6, the forward
benches' bound). The forward's saved activations (h, a) come from the same math in torch.

    torchrun --nproc_per_node=2 tests/test_moe_backward.py
    pytest tests/test_moe_backward.py -x
"""

import os
import subprocess
import sys

import pytest
import torch


def routing(topk_g, rank, e_local):
    """Rank `rank`'s tables from the gathered top-k (W, T_local, K) of global expert ids, the forward's (the
    dispatch runner's order): cu (E_local + 1), s_scatter (row -> global slot). A row's token is its slot // K."""
    f = topk_g.reshape(-1)
    mine = (f // e_local == rank).nonzero().squeeze(1)
    e = f[mine] - rank * e_local
    s_scatter = mine[torch.argsort(e, stable=True)]
    counts = torch.bincount(e, minlength=e_local)
    cu = torch.cat([torch.zeros(1, dtype=torch.int64, device=f.device), counts.cumsum(0)])
    return cu.to(torch.int32), s_scatter.to(torch.int32)


REL_TOL = 2.0**-6  # vs fp32 autograd, the forward benches' bound


def swiglu(h):
    g, u = h.chunk(2, dim=-1)
    return torch.nn.functional.silu(g) * u


def pairwise(w):
    """(E, 2I, X) [gate; up] -> rows interleaved pairwise (gate_i, up_i): the packed-pair layout the dgated
    epilogue reads its preact in and writes dh in."""
    E, I2, X = w.shape
    g, u = w.chunk(2, dim=1)
    return torch.stack([g, u], dim=2).reshape(E, I2, X)


def reference_grads(x, scores, topk_g, dout, w1, w2, topk):
    """fp32 autograd over every rank's tokens and every expert: (dx, dscores, dw1, dw2)."""
    x = x.float().detach().requires_grad_()
    scores = scores.float().detach().requires_grad_()
    w1 = w1.float().detach().requires_grad_()
    w2 = w2.float().detach().requires_grad_()
    f = topk_g.reshape(-1).long()
    tokens, s = torch.arange(f.numel(), device=x.device) // topk, scores.reshape(-1)
    y = torch.zeros_like(x)
    for e in range(w1.shape[0]):
        slots = (f == e).nonzero().squeeze(1)
        t = tokens[slots]
        y = y.index_add(0, t, s[slots, None] * (swiglu(x[t] @ w1[e].T) @ w2[e].T))
    (y * dout.float()).sum().backward()
    return x.grad, scores.grad.view_as(scores), w1.grad, w2.grad


def _run_rank():
    import torch.distributed as dist

    from quack.distributed.combine import CombineRunner
    from quack.distributed.dispatch import DispatchRunner
    from quack.distributed.reduce_over_k import reduce_over_k
    from quack.cute_dsl_utils import get_device_capacity
    from quack.epilogue.library import dswiglu_moe_swap_mod
    from quack.gemm import gemm
    from quack.gemm_config import cta_tile_shape_m

    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    device = torch.device("cuda", rank % torch.cuda.device_count())
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")
    group = dist.group.WORLD
    bf16 = torch.bfloat16
    H, I = 1024, 512
    e_local, topk, t_local = 4, 2, 512
    tile_n = 256  # the swap-AB operators' block: the GEMM's tile N
    E = world * e_local

    torch.manual_seed(0)  # the same routing, scores, weights and tokens on every rank
    topk_g = (
        torch.stack([torch.randperm(E, device=device)[:topk] for _ in range(world * t_local)])
        .view(world, t_local, topk)
        .to(torch.int32)
    )
    scores_g = torch.rand(world, t_local, topk, device=device).softmax(-1)
    w1 = torch.randn(E, 2 * I, H, device=device, dtype=bf16) * 0.05
    w2 = torch.randn(E, H, I, device=device, dtype=bf16) * 0.05
    x_all = torch.randn(world * t_local, H, device=device, dtype=bf16)
    dout_all = torch.randn(world * t_local, H, device=device, dtype=bf16)
    mine = slice(rank * e_local, (rank + 1) * e_local)
    w1_loc, w2_loc = w1[mine], w2[mine]

    # The routing and the forward's saved activations, packed per local expert
    cu, s_scatter = routing(topk_g, rank, e_local)
    total = max_rows = s_scatter.numel()
    slot = s_scatter[:total].long()
    x_rows = x_all[slot // topk]
    # the GEMM's operands span the routing's row bound (max_rows); rows past total are never scheduled
    score = torch.zeros(max_rows, device=device, dtype=torch.float32)
    score[:total] = scores_g.reshape(-1)[slot]
    w1_pw = pairwise(w1_loc)  # the preact in pairwise (gate_i, up_i) rows: the packed-pair layout
    h = torch.zeros(max_rows, 2 * I, device=device, dtype=bf16)
    for e in range(e_local):
        r = slice(int(cu[e]), int(cu[e + 1]))
        h[r] = x_rows[r] @ w1_pw[e].T

    # 1. dout in symmetric memory, dispatched into the fc2 dgrad
    w2t = w2_loc.transpose(
        1, 2
    ).contiguous()  # (E, I, H), K-major: the transposed weight copy, once per step
    w1t = w1_pw.transpose(1, 2).contiguous()  # (E, H, 2I), K in the pairwise order dh comes in
    dispatch = DispatchRunner(t_local, topk, H, bf16, max_rows, tile_n, e_local)
    combine = CombineRunner(t_local, topk, H, bf16, max_rows)
    dh = torch.zeros(max_rows, 2 * I, device=device, dtype=bf16)  # packed pairwise, as h
    a_s = torch.zeros(max_rows, I, device=device, dtype=bf16)
    # the score grad's per-CTA-M-tile partials (the sink rule: M tiles of the per-CTA tile, N columns)
    cta_tile_m = cta_tile_shape_m(256, 2, get_device_capacity(device)[0], False)
    dscore = torch.zeros(-(-I // cta_tile_m), max_rows, device=device, dtype=torch.float32)
    dx = torch.empty(t_local, H, device=device, dtype=bf16)
    me = slice(rank * t_local, (rank + 1) * t_local)
    args_d = dispatch.prepare(dout_all[me], topk_g[rank])  # publishes dout, routes on the device
    rows_d = dispatch.rows
    # 1 + 2 fused: the dgated epilogue on the fc2 dgrad reads the preact (C), writes dh (D), a * s and the
    # per-token score grad (a row-vector reduce over I)
    dswiglu_moe_swap_mod.gemm(
        w2t,
        rows_d,
        dh.T,
        h.T,
        epi_args={"score": score, "postact": a_s.T, "dscore": dscore},
        tile_M=256,
        tile_N=tile_n,
        cluster_M=2,
        cluster_N=1,
        cu_seqlens_n=cu,
        dist_op=dispatch.op,
        dist_op_args=args_d,
    )
    ds_g = torch.zeros(world * t_local * topk, device=device, dtype=torch.float32)
    ds_g[slot] = dscore.sum(0)[:total]  # the partials folded over the M tiles
    ds = torch.empty(t_local * topk, device=device, dtype=torch.float32)
    dist.reduce_scatter_tensor(ds, ds_g, group=group)
    # 3. fc1 dgrad drained into the origins' slots, then the sum over k
    gemm(
        w1t,
        dh,
        combine.d,
        None,
        None,
        256,
        tile_n,
        2,
        1,
        cu_seqlens_n=cu,
        dist_op=combine.op,
        dist_op_args=combine.prepare(s_scatter),  # a row's slot id is its routing slot
    )
    torch.cuda.synchronize(device)
    dist.barrier()  # every rank's drains have landed
    reduce_over_k(combine.slots, None, dx)
    # 4. the weight grads as varlen-K GEMMs over the packed rows: dW2_e = dy_e^T (a s)_e, dW1_e = dh_e^T x_e;
    # A (m, total_k) m-major, B (n, total_k) n-major = the rows' transposes
    dw1, dw2 = torch.empty_like(w1_loc), torch.empty_like(w2_loc)
    cu32 = cu.to(torch.int32)
    gemm(rows_d[:total].T, a_s[:total].T, dw2, None, None, 128, 128, 1, 1, cu_seqlens_k=cu32)
    gemm(
        dh[:total].T, x_rows.T, dw1, None, None, 128, 128, 1, 1, cu_seqlens_k=cu32
    )  # pairwise rows
    torch.cuda.synchronize(device)

    dx_ref, ds_ref, dw1_ref, dw2_ref = reference_grads(
        x_all, scores_g, topk_g, dout_all, w1, w2, topk
    )
    got = {"dx": dx, "dscores": ds.view(t_local, topk), "dw1": dw1, "dw2": dw2}
    ref = {
        "dx": dx_ref[me],
        "dscores": ds_ref[rank],
        "dw1": pairwise(dw1_ref[mine]),
        "dw2": dw2_ref[mine],
    }
    failures = []
    for name in got:
        rel = float((got[name].float() - ref[name]).abs().max() / ref[name].abs().max())
        print(f"rank{rank} {name}: max rel err {rel:.2e} (tol {REL_TOL:.2e})", flush=True)
        if rel > REL_TOL:
            failures.append(f"rank{rank} {name}: max rel err {rel:.4f}")
    dist.barrier()  # peers may still be pulling dout / draining slots
    dist.destroy_process_group()
    if failures:
        print("\n".join(failures), file=sys.stderr)
        sys.exit(1)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 10,
    reason="the chain runs on the swap-AB operators (cu_seqlens_n is SM100+)",
)
def test_moe_backward():
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=2",
        "--standalone",
        __file__,
    ]
    subprocess.run(cmd, check=True, timeout=900)


if __name__ == "__main__":
    _run_rank()
