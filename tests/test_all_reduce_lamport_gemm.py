# Copyright (c) 2026, QuACK team.
"""Multi-rank correctness tests for AllReduceOneShotRunner (quack/distributed/all_reduce_lamport.py): GEMM +
one-shot (Lamport) all-reduce, every rank ending with the whole full-K product; no flags, the data is the
flag, and every rank reduces whole tiles, so aux outputs and a quantized D land complete and local as under
RS. More launches than slots, so the 3-slot rotation and the re-arm are exercised.

Runs under torchrun (one rank per GPU); the pytest entry point spawns torchrun as a subprocess on the
free GPUs.

    torchrun --nproc_per_node=2 tests/test_all_reduce_lamport_gemm.py
    pytest tests/test_all_reduce_lamport_gemm.py -x
"""

import os
import subprocess
import sys

import pytest
import torch

NUM_ITERS = 5  # > the 3 slots: rotation and re-arm


def _run_rank():
    import torch.distributed as dist

    from quack.cute_dsl_utils import get_device_capacity
    from quack.distributed.all_reduce_lamport import AllReduceOneShotRunner
    from quack.epilogue.frontend import gemm_epilogue
    from quack.epilogue.library import identity_epi, linear_act_mod
    from quack.epilogue.ops import ColVecLoad, RowVecLoad, RowVecReduce
    from quack.gemm import gemm
    from quack.gemm_config import GemmConfig
    from quack.tile_scheduler import RasterOrderOption

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    device = torch.device("cuda", rank % torch.cuda.device_count())
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    failures = []

    dtype = torch.bfloat16
    tile_m, tile_n = 128, 256
    m, n, k = 2048, 1024, 1024  # k per rank
    capacity = get_device_capacity(device)
    sm100 = capacity[0] in (10, 11)
    raster = RasterOrderOption.AlongN  # the reducers' measured raster (S2 schedule)
    launch = dict(tile_M=tile_m, tile_N=tile_n, cluster_M=1, cluster_N=1, raster_order=raster)

    def check(label, out, ref, base, mult=3):
        """out vs the fp32 ref, within mult x the error of the same-dtype baseline (cuBLAS + NCCL's output);
        mult 3: the switch's bf16 ld_reduce truncates where the baseline rounds once (2x at world 2)."""
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

    def nccl_ar(a, b):
        out = a @ b.T
        dist.all_reduce(out)
        return out

    # --- D only and D + a replicated residual C, more launches than slots, at the one-CTA tile and (SM100) the
    # benched 2-CTA tile (the comm warps' frame: whole tiles) ---
    tiles = [(tile_m, tile_n, 1)] + ([(256, 256, 2)] if sm100 else [])
    for (tm, tn, cm), has_c in [(t, c) for t in tiles for c in (False, True)]:
        ar1 = AllReduceOneShotRunner(
            m, n, dtype, tm // cm
        )  # the slots are per CTA tile: tile_M / cluster_M
        for it in range(NUM_ITERS):
            a, b = operands(6000 + 10 * it)
            c = None
            if has_c:
                torch.manual_seed(6000 + 10 * it)  # rank-invariant residual
                c = torch.randn(m, n, dtype=dtype, device=device)
            gemm(
                a,
                b,
                ar1.d,
                c,
                None,
                tm,
                tn,
                cm,
                1,
                raster_order=raster,
                dist_op=ar1.op,
                dist_op_args=ar1.prepare(),
            )
            torch.cuda.synchronize(device)
            ref, base = full_ref(a, b), nccl_ar(a, b)
            if has_c:
                ref, base = ref + c.float(), base + c
            check(f"lamport{'+c' if has_c else ''} {tm}x{tn} c{cm} iter={it}", ar1.d, ref, base)

    # --- the decode shape: M = 64 tokens on the 64x64 one-CTA tile the paper's decode rows run (quack's tuner never
    # proposes tile_m 64, so nothing else exercises it); world <= 4: a 64-row tile has a 16-row stripe floor ---
    if world <= 4:
        m_dec = 64
        ar1 = AllReduceOneShotRunner(m_dec, n, dtype, 64)
        for it in range(NUM_ITERS):
            torch.manual_seed(6500 + 10 * it + rank)
            a = torch.randn(m_dec, k, dtype=dtype, device=device) / 8
            b = torch.randn(n, k, dtype=dtype, device=device) / 8
            gemm(
                a,
                b,
                ar1.d,
                None,
                None,
                64,
                64,
                1,
                1,
                raster_order=raster,
                dist_op=ar1.op,
                dist_op_args=ar1.prepare(),
            )
            torch.cuda.synchronize(device)
            check(f"lamport decode 64x64 iter={it}", ar1.d, full_ref(a, b), nccl_ar(a, b))

    # --- through an epilogue module ---
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
    ar1 = AllReduceOneShotRunner(m, n, dtype, tile_m)
    for it in range(NUM_ITERS):
        a, b = operands(7000 + 10 * it)
        identity_epi(
            a, b.T, out={"D": ar1.d}, config=cfg, dist_op=ar1.op, dist_op_args=ar1.prepare()
        )
        torch.cuda.synchronize(device)
        check(f"mod lamport iter={it}", ar1.d, full_ref(a, b), nccl_ar(a, b))

    # --- activation + aux output: whole tiles per rank, so the aux lands complete and local ---
    relu_mod = linear_act_mod("relu", gated=False, has_c=False, has_rowvec=False, has_colvec=False)
    ar1 = AllReduceOneShotRunner(m, n, dtype, tile_m)
    post = torch.zeros(m, n, dtype=dtype, device=device)
    for it in range(NUM_ITERS):
        a, b = operands(8700 + 10 * it)
        relu_mod.gemm(
            a,
            b,
            ar1.d,
            None,
            epi_args={"mAuxOut": post},
            dist_op=ar1.op,
            dist_op_args=ar1.prepare(),
            **launch,
        )
        torch.cuda.synchronize(device)
        ref, base = full_ref(a, b), nccl_ar(a, b)
        check(f"lamport+relu D iter={it}", ar1.d, ref, base)
        check(f"lamport+relu iter={it}", post, ref.relu(), base.relu())

    # --- vector operands in the comm warps' frame: a per-column bias, a per-row scale ---
    @gemm_epilogue(ops={"bias": RowVecLoad("bias"), "scale": ColVecLoad("scale")})
    def vec_epi(acc, bias, scale):
        return {"D": acc * scale + bias}

    torch.manual_seed(42)  # rank-invariant vectors
    bias = torch.randn(n, device=device, dtype=torch.float32)
    scale = torch.randn(m, device=device, dtype=torch.float32)
    ar1 = AllReduceOneShotRunner(m, n, dtype, tile_m)
    for it in range(NUM_ITERS):
        a, b = operands(8500 + 10 * it)
        vec_epi.gemm(
            a,
            b,
            ar1.d,
            None,
            epi_args={"bias": bias, "scale": scale},
            dist_op=ar1.op,
            dist_op_args=ar1.prepare(),
            **launch,
        )
        torch.cuda.synchronize(device)
        ref = full_ref(a, b) * scale[:, None] + bias
        check(f"lamport+vec iter={it}", ar1.d, ref, nccl_ar(a, b).float() * scale[:, None] + bias)

    # --- reduce output in the comm warps: per-tile column max_abs (the frame: whole tiles) ---
    @gemm_epilogue(reduces={"colmax": RowVecReduce("colmax", combine="max_abs")})
    def rowmax_epi(acc):
        return {"D": acc, "colmax": acc}

    ar1 = AllReduceOneShotRunner(m, n, dtype, tile_m)
    colmax = torch.zeros(m // tile_m, n, dtype=torch.float32, device=device)
    for it in range(NUM_ITERS):
        a, b = operands(3700 + 10 * it)
        rowmax_epi.gemm(
            a,
            b,
            ar1.d,
            None,
            epi_args={"colmax": colmax},
            dist_op=ar1.op,
            dist_op_args=ar1.prepare(),
            **launch,
        )
        torch.cuda.synchronize(device)
        ref, base = full_ref(a, b), nccl_ar(a, b).float()
        check(f"lamport+rowmax D iter={it}", ar1.d, ref, base)
        tile_max = lambda x: x.view(m // tile_m, tile_m, n).abs().amax(1)
        check(f"lamport+rowmax colmax iter={it}", colmax, tile_max(ref), tile_max(base))

    if sm100:  # block-scaled operands and a quantized D: the SM100 formats (SM90 has no block-scaled hardware)
        from quack.blockscaled.operand import BlockScaledOperand
        from quack.blockscaled.quantize import (
            dequant_operand,
            nvfp4_per_tensor_scale,
            unpack_scale_blocked_to_2d,
        )

        for name in ("mxfp8_e4m3", "nvfp4"):
            ar1 = AllReduceOneShotRunner(m, n, dtype, tile_m)
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
                    ar1.d,
                    None,
                    None,
                    tile_m,
                    tile_n,
                    1,
                    1,
                    raster_order=raster,
                    alpha=float(pts_a * pts_b) if name == "nvfp4" else 1.0,
                    dist_op=ar1.op,
                    dist_op_args=ar1.prepare(),
                    SFA=a_op.scale,
                    SFB=b_op.scale,
                    bs_format_a=name,
                    bs_format_b=name,
                )
                torch.cuda.synchronize(device)
                a_dq, b_dq = (op.dequantize(torch.float32).to(dtype) for op in (a_op, b_op))
                check(f"lamport {name} iter={it}", ar1.d, full_ref(a_dq, b_dq), nccl_ar(a_dq, b_dq))

        # --- requant: D quantized by its SF op, whole tiles per rank so the SF op runs locally as under RS ---
        for d_dtype, sf_dtype, vec, bin_half in (
            (
                torch.float8_e4m3fn,
                torch.float8_e8m0fnu,
                32,
                16,
            ),  # half the widest e4m3 bin, in scale units
            (torch.float4_e2m1fn_x2, torch.float8_e4m3fn, 16, 1),  # half the widest e2m1 bin
        ):
            ar1 = AllReduceOneShotRunner(m, n, d_dtype, tile_m)
            sfd = torch.empty(1, m // 128, n // (4 * vec), 32, 4, 4, dtype=sf_dtype, device=device)
            for it in range(NUM_ITERS):
                a, b = operands(9700 + 10 * it + vec)
                gemm(
                    a,
                    b,
                    ar1.d,
                    None,
                    None,
                    tile_m,
                    tile_n,
                    1,
                    1,
                    raster_order=raster,
                    dist_op=ar1.op,
                    dist_op_args=ar1.prepare(),
                    SFD=sfd,
                )
                torch.cuda.synchronize(device)
                ref = full_ref(a, b)
                scale = (
                    unpack_scale_blocked_to_2d(sfd, m, n // vec)[0]
                    .float()
                    .repeat_interleave(vec, -1)
                )
                out = ar1.d.float() if vec == 32 else dequant_operand(ar1.d)
                err = (out * scale - ref).abs()
                bound = (
                    scale * bin_half * 1.05 + 3 * (nccl_ar(a, b).float() - ref).abs().max()
                )  # + the wire's error
                if not (err <= bound).all() or torch.isnan(scale).any():
                    failures.append(
                        f"rank{rank} lamport requant vec{vec} iter={it}: err={err.max().item()} bound={bound.max().item()}"
                    )

    fail_count = torch.tensor([len(failures)], device=device)
    dist.all_reduce(fail_count)
    if failures:
        print("\n".join(failures), file=sys.stderr, flush=True)
    dist.destroy_process_group()
    if int(fail_count.item()) > 0:
        sys.exit(1)
    if rank == 0:
        print("ALL GEMM+AR ONE-SHOT CHECKS PASSED", flush=True)


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
def test_all_reduce_lamport_gemm_multirank():
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
    assert "ALL GEMM+AR ONE-SHOT CHECKS PASSED" in res.stdout


if __name__ == "__main__":
    _run_rank()
