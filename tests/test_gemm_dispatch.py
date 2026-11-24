# Copyright (C) 2026, Tri Dao.
"""Multi-rank correctness tests for DispatchRunner (quack/distributed/dispatch.py): MoE dispatch fused into the
grouped FC1 GEMM. Every rank's tokens live in a symmetric buffer; the GEMM's transfer warps pull the rows this
rank's experts read into every reader expert's row of the token operand, the tiles gate on the landed counts,
and the fused activation's postact matches a torch reference over the routed tokens. Tokens on either ragged
dim: "n" = swap-AB (B rows in even-split N blocks, the M-paired gated store; SM100+), "m" = varlen_m (A rows in
fixed tile_M blocks, the N-paired store). Block-scaled formats ("n") pull each token's scale row into the
GEMM's per-block SFB atoms alongside (the scale lane).

Runs under torchrun (one rank per GPU); the pytest entry point spawns torchrun as a subprocess on the free GPUs.

    torchrun --nproc_per_node=2 tests/test_gemm_dispatch.py
    pytest tests/test_gemm_dispatch.py -x
"""

import os
import subprocess
import sys

import pytest
import torch


def interleave_gate_up(w, ragged_dim):
    """(E, 2I, X) [gate; up] -> gate / up rows interleaved per 8 (ragged dim "n": the M-paired gated store) or
    per row ("m": the N-paired store, row 2j = gate j)."""
    E, I2, X = w.shape
    g, u = w.chunk(2, dim=1)
    r = 8 if ragged_dim == "n" else 1
    return torch.stack(
        [g.view(E, I2 // (2 * r), r, X), u.view(E, I2 // (2 * r), r, X)], dim=2
    ).reshape(E, I2, X)


def routing(topk_g, rank, e_local):
    """Rank `rank`'s tables from the gathered top-k (W, T_local, K) of global expert ids, the routing_metadata
    contract: cu (E_local + 1), s_scatter (row -> slot), s_reverse (slot -> row, this rank's slots only). A row's
    token is its slot // K (slots are (source rank, token, k) flattened)."""
    f = topk_g.reshape(-1)
    mine = (f // e_local == rank).nonzero().squeeze(1)
    e = f[mine] - rank * e_local
    s_scatter = mine[torch.argsort(e, stable=True)]
    counts = torch.bincount(e, minlength=e_local)
    cu = torch.cat([torch.zeros(1, dtype=torch.int64, device=f.device), counts.cumsum(0)])
    s_reverse = torch.zeros(f.numel(), dtype=torch.int64, device=f.device)
    s_reverse[s_scatter] = torch.arange(s_scatter.numel(), device=f.device)
    i32 = lambda t: t.to(torch.int32)
    return i32(cu), i32(s_scatter), i32(s_reverse)


def _run_rank():
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem
    import torch.nn.functional as F

    from quack.blockscaled.operand import BlockScaledOperand
    from quack.blockscaled.quantize import nvfp4_per_tensor_scale, unpack_scale_blocked_to_2d
    from quack.cute_dsl_utils import get_device_capacity
    from quack.distributed.dispatch import DispatchRunner
    from quack.epilogue.library import linear_act_mod
    from quack.gemm_tvm_ffi_utils import varlen_n_block_atoms

    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    device = torch.device("cuda", rank % torch.cuda.device_count())
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")
    group = dist.group.WORLD
    sm100 = get_device_capacity(device)[0] >= 10
    failures = []

    H, I = 1024, 512
    e_local, topk, t_local = 4, 2, 512
    # (ragged dim, block): the GEMM's tile along the ragged dim, the gated unit. "n" (SM100+): 256 and the
    # 16-granule odd even-split (192); "m": the per-CTA tile_M
    blocks = ([("n", 256), ("n", 192)] if sm100 else []) + [("m", 128)]
    gated_acts, plain_acts = ("swiglu", "geglu"), ("relu_sq",)
    formats = {"mxfp8": "mxfp8_e4m3", "nvfp4": "nvfp4"}  # BlockScaledFormat names

    torch.manual_seed(0)  # the same routing and weights on every rank
    topk_g = (
        torch.stack(
            [torch.randperm(world * e_local, device=device)[:topk] for _ in range(world * t_local)]
        )
        .view(world, t_local, topk)
        .to(torch.int32)
    )
    w_cat = torch.randn(e_local, 2 * I, H, device=device, dtype=torch.bfloat16) * 0.05  # [gate; up]
    w_plain = w_cat[:, :I].contiguous()
    torch.manual_seed(100 + rank)
    x = torch.randn(t_local, H, device=device, dtype=torch.bfloat16)
    x_all = torch.empty(world * t_local, H, device=device, dtype=torch.bfloat16)
    dist.all_gather_into_tensor(x_all, x)
    cu, s_scatter, s_reverse = routing(topk_g, rank, e_local)
    total = int(cu[-1])
    token_of_row = (s_scatter.long() // topk)[:total]  # each row's token in the gathered x

    def symm_rows(t):
        """This rank's rows of t in symmetric memory: (the buffer typed as t, its rendezvous handle)."""
        buf = symm_mem.empty(
            t_local, t.shape[1] * t.element_size(), dtype=torch.uint8, device=device
        )
        buf.copy_(t[rank * t_local : (rank + 1) * t_local].view(torch.uint8))
        return buf.view(t.dtype), symm_mem.rendezvous(buf, group=group)

    def check(label, postact, w_dq, x_dq, activation, gated):
        """The postact against the fp32 reference over the routed tokens, within twice the bf16 rounding error."""
        x_rows = x_dq[token_of_row]
        for e in range(e_local):
            rows = slice(int(cu[e]), int(cu[e + 1]))
            h = x_rows[rows] @ w_dq[e].T
            if gated:
                g, u = h[:, :I], h[:, I:]
                ref = F.silu(g) * u if activation == "swiglu" else F.gelu(g, approximate="tanh") * u
            else:
                ref = h.relu().square()
            err = (postact[rows].float() - ref).abs().max().item()
            tol = 2 * (ref.bfloat16().float() - ref).abs().max().item() + 1e-3
            if err > tol:
                failures.append(f"rank{rank} {label} expert {e}: err={err} tol={tol}")

    def fc1(ragged_dim, block, w_q, rows, dispatch, args, mod, epi_args):
        """The grouped FC1 in the ragged dim's operand placement: swap-AB (weights on M) or varlen_m."""
        kw = dict(epi_args=epi_args, dist_op=dispatch.op, dist_op_args=args)
        if ragged_dim == "n":
            mod.gemm(
                w_q,
                rows,
                None,
                None,
                tile_M=256,
                tile_N=block,
                cluster_M=2,
                cluster_N=1,
                cu_seqlens_n=cu,
                **kw,
            )
        else:
            mod.gemm(
                rows,
                w_q,
                None,
                None,
                tile_M=block,
                tile_N=256,
                cluster_M=1,
                cluster_N=1,
                cu_seqlens_m=cu,
                **kw,
            )
        torch.cuda.synchronize(device)

    # --- bf16: every activation on every (ragged dim, block) this arch admits ---
    x_loc, x_hdl = symm_rows(x_all)
    for ragged_dim, block in blocks:
        dispatch = DispatchRunner(x_loc, x_hdl, block, ragged_dim=ragged_dim)
        for activation in gated_acts + plain_acts:
            gated = activation in gated_acts
            w = interleave_gate_up(w_cat, ragged_dim) if gated else w_plain
            postact = torch.zeros(total, I, device=device, dtype=torch.bfloat16)
            mod = linear_act_mod(
                activation, gated=gated, has_c=False, has_rowvec=False, has_colvec=False
            )
            dist.barrier()  # every rank's rows are written before any pull reads them
            rows, args = dispatch.prepare(topk_g, s_scatter, s_reverse, cu)
            aux = postact.T if ragged_dim == "n" else postact  # the M-paired store writes m-major
            fc1(ragged_dim, block, w, rows, dispatch, args, mod, {"mAuxOut": aux})
            w_ref = w_cat.float() if gated else w_plain.float()
            check(
                f"dispatch bf16 {ragged_dim}{block} {activation}",
                postact,
                w_ref,
                x_all.float(),
                activation,
                gated,
            )
    dist.barrier()  # peers may still be pulling from x

    # --- block-scaled rows (swap-AB, SM100): the scale lane fills the GEMM's per-block SFB atoms ---
    if sm100:
        for fmt, name in formats.items():
            ragged_dim, block = "n", 256
            w_il = interleave_gate_up(w_cat, ragged_dim)
            epi_args = {}
            pts_w = pts_x = None
            if fmt == "nvfp4":  # per-tensor scales shared by every rank, folded into alpha
                pts_w = nvfp4_per_tensor_scale(w_il.abs().amax())
                pts_x = nvfp4_per_tensor_scale(x_all.abs().amax())
                epi_args["alpha"] = float(pts_w * pts_x)
            wop = BlockScaledOperand.quantize(w_il, name, per_tensor_scale=pts_w)
            xop = BlockScaledOperand.quantize(
                x_all, name, per_tensor_scale=pts_x
            )  # row-local: every rank agrees
            vec = xop.format.sf_vec_size
            x_sf = unpack_scale_blocked_to_2d(xop.scale[None], world * t_local, H // vec)[
                0
            ]  # 1-byte scale rows
            ncols = -(-world * t_local // block)  # the walk's flag columns bound the block slots
            sf_pool = torch.zeros(
                1,
                e_local * ncols * varlen_n_block_atoms(block),
                H // (4 * vec),
                32,
                4,
                4,
                dtype=x_sf.dtype,
                device=device,
            )
            xq_loc, xq_hdl = symm_rows(xop.qdata)
            sf_loc, sf_hdl = symm_rows(x_sf)
            dispatch = DispatchRunner(
                xq_loc, xq_hdl, block, sf_loc, sf_hdl, sf_pool, ragged_dim=ragged_dim
            )
            postact = torch.zeros(total, I, device=device, dtype=torch.bfloat16)
            mod = linear_act_mod(
                "swiglu",
                gated=True,
                has_c=False,
                has_rowvec=False,
                has_colvec=False,
                has_alpha=fmt == "nvfp4",
            )
            dist.barrier()
            rows, args = dispatch.prepare(topk_g, s_scatter, s_reverse, cu)
            sf_kw = dict(SFA=wop.scale, SFB=sf_pool, bs_format_a=name, bs_format_b=name)
            kw = dict(
                epi_args={**epi_args, "mAuxOut": postact.T},
                dist_op=dispatch.op,
                dist_op_args=args,
                **sf_kw,
            )
            mod.gemm(
                wop.qdata,
                rows,
                None,
                None,
                tile_M=256,
                tile_N=block,
                cluster_M=2,
                cluster_N=1,
                cu_seqlens_n=cu,
                **kw,
            )
            torch.cuda.synchronize(device)
            # the dequantized operands are the reference: gate / up rows back from the per-8 interleave
            w_dq = wop.dequantize(torch.float32).view(e_local, I // 8, 2, 8, H)
            w_dq = torch.cat(
                [w_dq[:, :, 0].reshape(e_local, I, H), w_dq[:, :, 1].reshape(e_local, I, H)], dim=1
            )
            check(
                f"dispatch {fmt} {ragged_dim}{block} swiglu",
                postact,
                w_dq,
                xop.dequantize(torch.float32),
                "swiglu",
                True,
            )
            dist.barrier()

    fail_count = torch.tensor([len(failures)], device=device)
    dist.all_reduce(fail_count)
    if failures:
        print("\n".join(failures), file=sys.stderr, flush=True)
    dist.destroy_process_group()
    if int(fail_count.item()) > 0:
        sys.exit(1)
    if rank == 0:
        print("ALL DISPATCH+GEMM CHECKS PASSED", flush=True)


def _visible_gpus() -> str:
    """GPUs with no running compute processes (the node is shared), else all of them."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        free = [
            line.split(",")[0].strip()
            for line in out.strip().splitlines()
            if int(line.split(",")[1]) < 5
        ]
        return ",".join(free)
    except Exception:
        return ",".join(str(i) for i in range(torch.cuda.device_count()))


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs >= 2 GPUs")
def test_gemm_dispatch_multirank():
    gpus = _visible_gpus()
    nproc = min(len(gpus.split(",")), 8)
    assert nproc >= 2, f"not enough free GPUs: {gpus}"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpus
    env.setdefault("PYTHONPATH", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={nproc}",
            "--standalone",
            os.path.abspath(__file__),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=1500,
    )
    if res.returncode != 0:
        print(res.stdout[-4000:])
        print(res.stderr[-4000:], file=sys.stderr)
    assert res.returncode == 0
    assert "ALL DISPATCH+GEMM CHECKS PASSED" in res.stdout


if __name__ == "__main__":
    _run_rank()
