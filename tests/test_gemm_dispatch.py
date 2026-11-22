# Copyright (C) 2026, Tri Dao.
"""Dispatch + fc1 through mod.gemm on two ranks: every rank's tokens live in a symmetric buffer, the GEMM's transfer
warps pull the rows this rank's experts read into every reader expert's row of B (DispatchPull, the swap-AB
grouped GEMM: tokens are B rows packed per expert in even-split N blocks), the tiles gate on the landed counts, and
the gated SwiGLU postact matches a torch reference over the routed tokens. Block-scaled formats pull each token's
scale row into the GEMM's per-block SFB atoms alongside (the scale lane).

    torchrun --nproc_per_node=2 tests/test_gemm_dispatch.py mxfp8
    pytest tests/test_gemm_dispatch.py -x
"""

import os
import subprocess
import sys

import pytest
import torch

H, I = 1024, 512
E_LOCAL, TOPK, T_LOCAL = 4, 2, 512
TILE_N = 256
FORMATS = {"bf16": None, "mxfp8": "mxfp8_e4m3", "nvfp4": "nvfp4"}  # BlockScaledFormat names


def interleave_gate_up(w):
    """(E, 2I, X) [gate; up] -> rows interleaved per 8 (the M-paired gated store's layout)."""
    E, I2, X = w.shape
    g, u = w.chunk(2, dim=1)
    return torch.stack([g.view(E, I2 // 16, 8, X), u.view(E, I2 // 16, 8, X)], dim=2).reshape(
        E, I2, X
    )


def split_gate_up(w_il):
    """Inverse of interleave_gate_up: (E, 2I, X) interleaved -> (gate (E, I, X), up (E, I, X))."""
    E, I2, X = w_il.shape
    gu = w_il.view(E, I2 // 16, 2, 8, X)
    return gu[:, :, 0].reshape(E, I2 // 2, X), gu[:, :, 1].reshape(E, I2 // 2, X)


def routing(topk_g, rank):
    """Rank `rank`'s tables from the gathered top-k (W, T_local, K) of global expert ids, the routing_metadata
    contract: cu (E_local + 1), s_scatter (row -> slot), s_reverse (slot -> row, this rank's slots only). A row's
    token is its slot // K (slots are (source rank, token, k) flattened)."""
    f = topk_g.reshape(-1)
    mine = (f // E_LOCAL == rank).nonzero().squeeze(1)
    e = f[mine] - rank * E_LOCAL
    s_scatter = mine[torch.argsort(e, stable=True)]
    counts = torch.bincount(e, minlength=E_LOCAL)
    cu = torch.cat([torch.zeros(1, dtype=torch.int64, device=f.device), counts.cumsum(0)])
    s_reverse = torch.zeros(f.numel(), dtype=torch.int64, device=f.device)
    s_reverse[s_scatter] = torch.arange(s_scatter.numel(), device=f.device)
    i32 = lambda t: t.to(torch.int32)
    return i32(cu), i32(s_scatter), i32(s_reverse)


def _run_rank(fmt):
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem
    import torch.nn.functional as F

    from quack.blockscaled.operand import BlockScaledOperand
    from quack.blockscaled.quantize import nvfp4_per_tensor_scale, unpack_scale_blocked_to_2d
    from quack.distributed.dispatch import DispatchRunner
    from quack.epilogue.library import linear_act_mod
    from quack.gemm_tvm_ffi_utils import varlen_n_block_atoms

    name = FORMATS[fmt]
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    device = torch.device("cuda", rank % torch.cuda.device_count())
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")
    group = dist.group.WORLD

    torch.manual_seed(0)  # the same routing and weights on every rank
    topk_g = (
        torch.stack(
            [torch.randperm(world * E_LOCAL, device=device)[:TOPK] for _ in range(world * T_LOCAL)]
        )
        .view(world, T_LOCAL, TOPK)
        .to(torch.int32)
    )
    w_il = interleave_gate_up(
        torch.randn(E_LOCAL, 2 * I, H, device=device, dtype=torch.bfloat16) * 0.05
    )
    torch.manual_seed(100 + rank)
    x = torch.randn(T_LOCAL, H, device=device, dtype=torch.bfloat16)
    x_all = torch.empty(world * T_LOCAL, H, device=device, dtype=torch.bfloat16)
    dist.all_gather_into_tensor(x_all, x)
    # Quantization is row-local, so every rank quantizes the gathered x the same way: its own rows go out, the
    # dequantized whole is the reference. nvfp4's per-tensor scales are shared and folded into alpha.
    epi_args, sf_kw, x_sf, sf_pool = {}, {}, None, None
    if name is None:
        w_q, w_dq, x_q, x_dq = w_il, w_il.float(), x_all, x_all.float()
    else:
        pts_w = pts_x = None
        if fmt == "nvfp4":
            pts_w = nvfp4_per_tensor_scale(w_il.abs().amax())
            pts_x = nvfp4_per_tensor_scale(x_all.abs().amax())
            epi_args["alpha"] = float(pts_w * pts_x)
        wop = BlockScaledOperand.quantize(w_il, name, per_tensor_scale=pts_w)
        xop = BlockScaledOperand.quantize(x_all, name, per_tensor_scale=pts_x)
        w_q, w_dq, x_q, x_dq = (
            wop.qdata,
            wop.dequantize(torch.float32),
            xop.qdata,
            xop.dequantize(torch.float32),
        )
        vec = xop.format.sf_vec_size
        x_sf = unpack_scale_blocked_to_2d(xop.scale[None], world * T_LOCAL, H // vec)[
            0
        ]  # 1-byte scale rows
        ncols = -(-world * T_LOCAL // TILE_N)  # the walk's flag columns bound the block slots
        sf_pool = torch.zeros(
            1,
            E_LOCAL * ncols * varlen_n_block_atoms(TILE_N),
            H // (4 * vec),
            32,
            4,
            4,
            dtype=x_sf.dtype,
            device=device,
        )
        sf_kw = dict(SFA=wop.scale, SFB=sf_pool, bs_format_a=name, bs_format_b=name)

    def symm_rows(t):
        """This rank's rows of t in symmetric memory: (the buffer typed as t, its rendezvous handle)."""
        buf = symm_mem.empty(
            T_LOCAL, t.shape[1] * t.element_size(), dtype=torch.uint8, device=device
        )
        buf.copy_(t[rank * T_LOCAL : (rank + 1) * T_LOCAL].view(torch.uint8))
        return buf.view(t.dtype), symm_mem.rendezvous(buf, group=group)

    x_loc, x_hdl = symm_rows(x_q)
    sf_loc, sf_hdl = (None, None) if x_sf is None else symm_rows(x_sf)
    cu, s_scatter, s_reverse = routing(topk_g, rank)
    total = int(cu[-1])
    dispatch = DispatchRunner(x_loc, x_hdl, TILE_N, sf_loc, sf_hdl, sf_pool)
    postact = torch.zeros(total, I, device=device, dtype=torch.bfloat16)
    epi_args["mAuxOut"] = postact.T
    mod = linear_act_mod(
        "swiglu",
        gated=True,
        has_c=False,
        has_rowvec=False,
        has_colvec=False,
        has_alpha=fmt == "nvfp4",
    )
    dist.barrier()  # every rank's rows are written before any pull reads them
    rows, args = dispatch.prepare(topk_g, s_scatter, s_reverse, cu)
    mod.gemm(
        w_q,
        rows,
        None,
        None,
        epi_args=epi_args,
        tile_M=256,
        tile_N=TILE_N,
        cluster_M=2,
        cluster_N=1,
        cu_seqlens_n=cu,
        dist_op=dispatch.op,
        dist_op_args=args,
        **sf_kw,
    )
    torch.cuda.synchronize(device)

    failures = []
    x_rows = x_dq[
        (s_scatter.long() // TOPK)[:total]
    ]  # the GEMM's B rows: each slot's token, in row order
    gate, up = split_gate_up(w_dq)
    for e in range(E_LOCAL):
        xe = x_rows[int(cu[e]) : int(cu[e + 1])]
        ref = F.silu(xe @ gate[e].T) * (xe @ up[e].T)
        err = (postact[int(cu[e]) : int(cu[e + 1])].float() - ref).abs().max().item()
        tol = 2 * (ref.bfloat16().float() - ref).abs().max().item() + 1e-3
        if err > tol:
            failures.append(f"rank{rank} expert {e}: err={err} tol={tol}")
    dist.barrier()  # peers may still be pulling from x
    dist.destroy_process_group()
    if failures:
        print("\n".join(failures), file=sys.stderr)
        sys.exit(1)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
@pytest.mark.parametrize("fmt", list(FORMATS))
def test_gemm_dispatch(fmt):
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=2",
        "--standalone",
        __file__,
        fmt,
    ]
    subprocess.run(cmd, check=True, timeout=600)


if __name__ == "__main__":
    _run_rank(sys.argv[1] if len(sys.argv) > 1 else "bf16")
