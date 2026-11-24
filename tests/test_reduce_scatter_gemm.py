# Copyright (c) 2026, QuACK team.
"""Multi-rank correctness tests for ReduceScatterRunner (quack/distributed/reduce_scatter.py): GEMM +
reduce-scatter, every rank ending with its M slab of the full-K product, with the epilogue classes the
operator admits running in its comm warps after the reduce.

Runs under torchrun (one rank per GPU); the pytest entry point spawns torchrun as a subprocess on the
free GPUs.

    torchrun --nproc_per_node=2 tests/test_reduce_scatter_gemm.py
    pytest tests/test_reduce_scatter_gemm.py -x
"""

import os
import subprocess
import sys

import pytest
import torch

NUM_ITERS = 3  # back-to-back launches: flag reset, exit barrier, PDL ordering


def _run_rank():
    import torch.distributed as dist

    from quack.cute_dsl_utils import get_device_capacity
    from quack.distributed.reduce_scatter import ReduceScatterRunner
    from quack.epilogue.frontend import gemm_epilogue
    from quack.epilogue.library import identity_epi, linear_act_mod, sq_reduce_mod
    from quack.epilogue.ops import ColVecLoad, RowVecLoad, RowVecReduce
    from quack.gemm import gemm
    from quack.gemm_config import GemmConfig
    from quack.rms_final_reduce import rms_final_reduce
    from quack.tile_scheduler import RasterOrderOption

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    device = torch.device("cuda", rank % torch.cuda.device_count())
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    failures = []

    dtype = torch.bfloat16
    tile_m, tile_n = 128, 256
    m, n, k = 2048, 1024, 1024  # k per rank; m is a whole number of tiles per slab for world <= 8
    capacity = get_device_capacity(device)
    sm100 = capacity[0] in (10, 11)
    slab = m // world
    rows = slice(rank * slab, (rank + 1) * slab)
    raster = RasterOrderOption.AlongN  # the reducers' measured raster (S2 schedule)
    launch = dict(tile_M=tile_m, tile_N=tile_n, cluster_M=1, cluster_N=1, raster_order=raster)

    def check(label, out, ref, base, mult=3):
        """out vs the fp32 ref, within mult x the error of the same-dtype baseline (cuBLAS + NCCL's
        output). mult 3, not quack's usual 2: the switch's bf16 ld_reduce truncates (up to 2 ulp
        toward zero) where the baseline rounds once, which is exactly 2x at world 2."""
        tol = (base.float() - ref).abs().max().item() + 1e-4
        err = (out.float() - ref).abs().max().item()
        if err > mult * tol:
            failures.append(f"rank{rank} {label}: err={err} tol={tol}")

    def operands(seed):
        torch.manual_seed(seed + rank)
        a = torch.randn(m, k, dtype=dtype, device=device) / 8
        b = torch.randn(n, k, dtype=dtype, device=device) / 8
        return a, b

    def full_ref(a, b):
        """fp32 full-K product over the ranks' K shards."""
        a_all = [torch.empty_like(a) for _ in range(world)]
        b_all = [torch.empty_like(b) for _ in range(world)]
        dist.all_gather(a_all, a)
        dist.all_gather(b_all, b)
        return sum((ar.float() @ br.float().T) for ar, br in zip(a_all, b_all))

    def nccl_rs(a, b):
        """cuBLAS + NCCL baseline: this rank's slab of the reduce-scattered bf16 partials."""
        out = torch.empty(slab, n, dtype=dtype, device=device)
        dist.reduce_scatter_single(out, a @ b.T)
        return out

    # --- D only, back-to-back launches, at the one-CTA tile and (SM100) the benched 2-CTA tile: the stripe each
    # comm warp group reduces is tile_m / world rows ---
    tiles = [(tile_m, tile_n, 1)] + ([(256, 256, 2)] if sm100 else [])
    for tm, tn, cm in tiles:
        rs = ReduceScatterRunner(
            m, n, dtype, tm // cm, tn
        )  # the flags are per CTA tile: tile_M / cluster_M
        for it in range(NUM_ITERS):
            a, b = operands(1000 + 10 * it)
            gemm(
                a,
                b,
                rs.d,
                None,
                None,
                tm,
                tn,
                cm,
                1,
                raster_order=raster,
                dist_op=rs.op,
                dist_op_args=rs.prepare(),
            )
            torch.cuda.synchronize(device)
            check(f"rs {tm}x{tn} c{cm} iter={it}", rs.d, full_ref(a, b)[rows], nccl_rs(a, b))

    # --- residual C, applied after the reduce: this rank's slab ---
    rs = ReduceScatterRunner(m, n, dtype, tile_m, tile_n)
    for it in range(NUM_ITERS):
        a, b = operands(4000 + 10 * it)
        c = torch.randn(slab, n, dtype=dtype, device=device)
        gemm(
            a,
            b,
            rs.d,
            c,
            None,
            tile_m,
            tile_n,
            1,
            1,
            raster_order=raster,
            dist_op=rs.op,
            dist_op_args=rs.prepare(),
        )
        torch.cuda.synchronize(device)
        check(f"rs+c iter={it}", rs.d, full_ref(a, b)[rows] + c.float(), nccl_rs(a, b) + c)

    # --- through an epilogue module (dist_op= / dist_op_args= on the torch-facing call) ---
    cfg = GemmConfig(
        tile_m=tile_m,
        tile_n=tile_n,
        pingpong=False,
        is_dynamic_persistent=False,
        cluster_m=1,
        cluster_n=1,
        raster_order=raster,
        device_capacity=capacity[0],
    )
    rs = ReduceScatterRunner(m, n, dtype, tile_m, tile_n)
    for it in range(NUM_ITERS):
        a, b = operands(6000 + 10 * it)
        identity_epi(a, b.T, out={"D": rs.d}, config=cfg, dist_op=rs.op, dist_op_args=rs.prepare())
        torch.cuda.synchronize(device)
        check(f"mod rs iter={it}", rs.d, full_ref(a, b)[rows], nccl_rs(a, b))

    # --- activation + aux output: the mod's EVT runs in the comm warps after the reduce; the aux follows D ---
    relu_mod = linear_act_mod("relu", gated=False, has_c=False, has_rowvec=False, has_colvec=False)
    rs = ReduceScatterRunner(m, n, dtype, tile_m, tile_n)
    post = torch.zeros(slab, n, dtype=dtype, device=device)
    for it in range(NUM_ITERS):
        a, b = operands(8000 + 10 * it)
        relu_mod.gemm(
            a,
            b,
            rs.d,
            None,
            epi_args={"mAuxOut": post},
            dist_op=rs.op,
            dist_op_args=rs.prepare(),
            **launch,
        )
        torch.cuda.synchronize(device)
        ref, base = full_ref(a, b)[rows], nccl_rs(a, b)
        check(f"rs+relu D iter={it}", rs.d, ref, base)
        check(f"rs+relu iter={it}", post, ref.relu(), base.relu())

    # --- vector operands in the comm warps' frame: a per-column bias, a per-row scale (D's rows: the slab) ---
    @gemm_epilogue(ops={"bias": RowVecLoad("bias"), "scale": ColVecLoad("scale")})
    def vec_epi(acc, bias, scale):
        return {"D": acc * scale + bias}

    torch.manual_seed(42)  # rank-invariant vectors
    bias = torch.randn(n, device=device, dtype=torch.float32)
    scale = torch.randn(m, device=device, dtype=torch.float32)[rows].contiguous()
    rs = ReduceScatterRunner(m, n, dtype, tile_m, tile_n)
    for it in range(NUM_ITERS):
        a, b = operands(8500 + 10 * it)
        vec_epi.gemm(
            a,
            b,
            rs.d,
            None,
            epi_args={"bias": bias, "scale": scale},
            dist_op=rs.op,
            dist_op_args=rs.prepare(),
            **launch,
        )
        torch.cuda.synchronize(device)
        ref = full_ref(a, b)[rows] * scale[:, None] + bias
        check(f"rs+vec iter={it}", rs.d, ref, nccl_rs(a, b).float() * scale[:, None] + bias)

    # --- reduce output in the comm warps: per-stripe column max_abs of the reduced slab ---
    @gemm_epilogue(reduces={"colmax": RowVecReduce("colmax", combine="max_abs")})
    def rowmax_epi(acc):
        return {"D": acc, "colmax": acc}

    stripe = tile_m // world  # the comm warps' frame: one reduce row per stripe of the slab
    rs = ReduceScatterRunner(m, n, dtype, tile_m, tile_n)
    colmax = torch.zeros(m // tile_m, n, dtype=torch.float32, device=device)
    for it in range(NUM_ITERS):
        a, b = operands(3000 + 10 * it)
        rowmax_epi.gemm(
            a,
            b,
            rs.d,
            None,
            epi_args={"colmax": colmax},
            dist_op=rs.op,
            dist_op_args=rs.prepare(),
            **launch,
        )
        torch.cuda.synchronize(device)
        ref, base = full_ref(a, b)[rows], nccl_rs(a, b).float()
        check(f"rs+rowmax D iter={it}", rs.d, ref, base)
        stripe_max = lambda x: x.view(m // tile_m, stripe, n).abs().amax(1)
        check(f"rs+rowmax colmax iter={it}", colmax, stripe_max(ref), stripe_max(base))

    # --- residual + RMSNorm statistic (gemm_rms's shape): D = (reduce + C) * weight, the slab's per-tile
    # squared-sum partials, rstd finalized by rms_final_reduce ---
    rms_mod = sq_reduce_mod(has_c=True, has_rowvec=True, has_aux=False)
    rstd_ref = lambda x: torch.rsqrt(x.square().mean(-1) + 1e-6)
    rs = ReduceScatterRunner(m, n, dtype, tile_m, tile_n)
    partials = torch.zeros(slab, n // tile_n, dtype=torch.float32, device=device)
    for it in range(NUM_ITERS):
        a, b = operands(3500 + 10 * it)
        c = torch.randn(slab, n, dtype=dtype, device=device)
        rms_mod.gemm(
            a,
            b,
            rs.d,
            c,
            epi_args={"mRowVecBroadcast": bias, "mColVecReduce": partials},
            dist_op=rs.op,
            dist_op_args=rs.prepare(),
            **launch,
        )
        rstd = rms_final_reduce(partials, scale=1.0 / n)
        torch.cuda.synchronize(device)
        ref, base = full_ref(a, b)[rows] + c.float(), (nccl_rs(a, b) + c).float()
        check(f"rs+rms D iter={it}", rs.d, ref * bias, base * bias)
        check(f"rs+rms rstd iter={it}", rstd, rstd_ref(ref), rstd_ref(base))

    if sm100:  # block-scaled operands and a quantized D: the SM100 formats (SM90 has no block-scaled hardware)
        from quack.blockscaled.operand import BlockScaledOperand
        from quack.blockscaled.quantize import (
            dequant_operand,
            nvfp4_per_tensor_scale,
            unpack_scale_blocked_to_2d,
        )

        # --- mxfp8 and nvfp4 operands: the reducers are format-agnostic (bf16 partials, the EVT after the sum);
        # nvfp4's per-tensor scales ride alpha, applied after the reduce, so they are the same on every rank ---
        for name in ("mxfp8_e4m3", "nvfp4"):
            rs = ReduceScatterRunner(m, n, dtype, tile_m, tile_n)
            for it in range(NUM_ITERS):
                a, b = operands(9000 + 10 * it)
                pts_a = pts_b = None
                if name == "nvfp4":
                    amax = torch.stack([a.abs().amax(), b.abs().amax()])
                    dist.all_reduce(amax, op=dist.ReduceOp.MAX)
                    pts_a, pts_b = nvfp4_per_tensor_scale(amax[0]), nvfp4_per_tensor_scale(amax[1])
                a_op = BlockScaledOperand.quantize(a, name, per_tensor_scale=pts_a)
                b_op = BlockScaledOperand.quantize(b, name, per_tensor_scale=pts_b)
                gemm(
                    a_op.qdata,
                    b_op.qdata,
                    rs.d,
                    None,
                    None,
                    tile_m,
                    tile_n,
                    1,
                    1,
                    raster_order=raster,
                    alpha=float(pts_a * pts_b) if name == "nvfp4" else 1.0,
                    dist_op=rs.op,
                    dist_op_args=rs.prepare(),
                    SFA=a_op.scale,
                    SFB=b_op.scale,
                    bs_format_a=name,
                    bs_format_b=name,
                )
                torch.cuda.synchronize(device)
                a_dq, b_dq = (op.dequantize(torch.float32).to(dtype) for op in (a_op, b_op))
                check(f"rs {name} iter={it}", rs.d, full_ref(a_dq, b_dq)[rows], nccl_rs(a_dq, b_dq))

        # --- requant: D quantized by its SF op in the comm warps, after the sum (bf16 wire) ---
        for d_dtype, sf_dtype, vec, bin_half in (
            (
                torch.float8_e4m3fn,
                torch.float8_e8m0fnu,
                32,
                16,
            ),  # half the widest e4m3 bin, in scale units
            (torch.float4_e2m1fn_x2, torch.float8_e4m3fn, 16, 1),  # half the widest e2m1 bin
        ):
            rs = ReduceScatterRunner(m, n, d_dtype, tile_m, tile_n)
            sfd = torch.empty(
                1, slab // 128, n // (4 * vec), 32, 4, 4, dtype=sf_dtype, device=device
            )
            for it in range(NUM_ITERS):
                a, b = operands(9500 + 10 * it + vec)
                gemm(
                    a,
                    b,
                    rs.d,
                    None,
                    None,
                    tile_m,
                    tile_n,
                    1,
                    1,
                    raster_order=raster,
                    dist_op=rs.op,
                    dist_op_args=rs.prepare(),
                    SFD=sfd,
                )
                torch.cuda.synchronize(device)
                ref = full_ref(a, b)[rows]
                scale = (
                    unpack_scale_blocked_to_2d(sfd, slab, n // vec)[0]
                    .float()
                    .repeat_interleave(vec, -1)
                )
                out = rs.d.float() if vec == 32 else dequant_operand(rs.d)
                err = (out * scale - ref).abs()
                bound = (
                    scale * bin_half * 1.05 + 3 * (nccl_rs(a, b).float() - ref).abs().max()
                )  # + the wire's error
                if not (err <= bound).all() or torch.isnan(scale).any():
                    failures.append(
                        f"rank{rank} rs requant vec{vec} iter={it}: err={err.max().item()} bound={bound.max().item()}"
                    )

    fail_count = torch.tensor([len(failures)], device=device)
    dist.all_reduce(fail_count)
    if failures:
        print("\n".join(failures), file=sys.stderr, flush=True)
    dist.destroy_process_group()
    if int(fail_count.item()) > 0:
        sys.exit(1)
    if rank == 0:
        print("ALL GEMM+RS CHECKS PASSED", flush=True)


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
def test_reduce_scatter_gemm_multirank():
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
        timeout=1200,
    )
    if res.returncode != 0:
        print(res.stdout[-4000:])
        print(res.stderr[-4000:], file=sys.stderr)
    assert res.returncode == 0
    assert "ALL GEMM+RS CHECKS PASSED" in res.stdout


if __name__ == "__main__":
    _run_rank()
