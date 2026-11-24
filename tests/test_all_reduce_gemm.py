# Copyright (c) 2026, QuACK team.
"""Multi-rank correctness tests for AllReduceRunner (quack/distributed/reduce_scatter.py): GEMM + two-shot
all-reduce, every rank ending with the whole full-K product; each rank reduces one stripe of every tile and
commits it into every rank's D by multimem store, so D and reduce outputs replicate and an aux output does
not (refused on the host).

Runs under torchrun (one rank per GPU); the pytest entry point spawns torchrun as a subprocess on the
free GPUs.

    torchrun --nproc_per_node=2 tests/test_all_reduce_gemm.py
    pytest tests/test_all_reduce_gemm.py -x
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
    from quack.distributed.reduce_scatter import AllReduceRunner
    from quack.epilogue.frontend import gemm_epilogue
    from quack.epilogue.library import identity_epi, linear_act_mod
    from quack.epilogue.ops import ColVecLoad, ColVecReduce, RowVecLoad
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
    m, n, k = 2048, 1024, 1024  # k per rank; m is a whole number of tiles for world <= 8
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

    # --- D only, back-to-back launches, at the one-CTA tile and (SM100) the benched 2-CTA tile: the stripe each
    # comm warp group reduces is tile_m / world rows ---
    tiles = [(tile_m, tile_n, 1)] + ([(256, 256, 2)] if sm100 else [])
    for tm, tn, cm in tiles:
        ar = AllReduceRunner(
            m, n, dtype, tm // cm, tn
        )  # the flags are per CTA tile: tile_M / cluster_M
        for it in range(NUM_ITERS):
            a, b = operands(2000 + 10 * it)
            gemm(
                a,
                b,
                ar.d,
                None,
                None,
                tm,
                tn,
                cm,
                1,
                raster_order=raster,
                dist_op=ar.op,
                dist_op_args=ar.prepare(),
            )
            torch.cuda.synchronize(device)
            check(f"ar {tm}x{tn} c{cm} iter={it}", ar.d, full_ref(a, b), nccl_ar(a, b))

    # --- the decode shape: M = 64 tokens on the 64x64 one-CTA tile the paper's decode rows run (quack's tuner never
    # proposes tile_m 64, so nothing else exercises it); world <= 4: a 64-row tile has a 16-row stripe floor ---
    if world <= 4:
        m_dec = 64
        ar = AllReduceRunner(m_dec, n, dtype, 64, 64)
        for it in range(NUM_ITERS):
            torch.manual_seed(2500 + 10 * it + rank)
            a = torch.randn(m_dec, k, dtype=dtype, device=device) / 8
            b = torch.randn(n, k, dtype=dtype, device=device) / 8
            gemm(
                a,
                b,
                ar.d,
                None,
                None,
                64,
                64,
                1,
                1,
                raster_order=raster,
                dist_op=ar.op,
                dist_op_args=ar.prepare(),
            )
            torch.cuda.synchronize(device)
            check(f"ar decode 64x64 iter={it}", ar.d, full_ref(a, b), nccl_ar(a, b))

    # --- residual C, applied after the reduce: replicated, so rank-invariant ---
    ar = AllReduceRunner(m, n, dtype, tile_m, tile_n)
    for it in range(NUM_ITERS):
        a, b = operands(5000 + 10 * it)
        torch.manual_seed(5000 + 10 * it)
        c = torch.randn(m, n, dtype=dtype, device=device)
        gemm(
            a,
            b,
            ar.d,
            c,
            None,
            tile_m,
            tile_n,
            1,
            1,
            raster_order=raster,
            dist_op=ar.op,
            dist_op_args=ar.prepare(),
        )
        torch.cuda.synchronize(device)
        check(f"ar+c iter={it}", ar.d, full_ref(a, b) + c.float(), nccl_ar(a, b) + c)

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
    ar = AllReduceRunner(m, n, dtype, tile_m, tile_n)
    for it in range(NUM_ITERS):
        a, b = operands(6000 + 10 * it)
        identity_epi(a, b.T, out={"D": ar.d}, config=cfg, dist_op=ar.op, dist_op_args=ar.prepare())
        torch.cuda.synchronize(device)
        check(f"mod ar iter={it}", ar.d, full_ref(a, b), nccl_ar(a, b))

    # --- vector operands in the comm warps' frame: a per-column bias, a per-row scale (D's rows: all of M) ---
    @gemm_epilogue(ops={"bias": RowVecLoad("bias"), "scale": ColVecLoad("scale")})
    def vec_epi(acc, bias, scale):
        return {"D": acc * scale + bias}

    torch.manual_seed(42)  # rank-invariant vectors
    bias = torch.randn(n, device=device, dtype=torch.float32)
    scale = torch.randn(m, device=device, dtype=torch.float32)
    ar = AllReduceRunner(m, n, dtype, tile_m, tile_n)
    for it in range(NUM_ITERS):
        a, b = operands(8500 + 10 * it)
        vec_epi.gemm(
            a,
            b,
            ar.d,
            None,
            epi_args={"bias": bias, "scale": scale},
            dist_op=ar.op,
            dist_op_args=ar.prepare(),
            **launch,
        )
        torch.cuda.synchronize(device)
        ref = full_ref(a, b) * scale[:, None] + bias
        check(f"ar+vec iter={it}", ar.d, ref, nccl_ar(a, b).float() * scale[:, None] + bias)

    # --- reduce output replicated into every rank (the gemm_rms shape: D + per-row sum of squares): each
    # rank reduces one stripe of every tile, so a row's slot is complete on one rank; AllReduce.store_sink
    # multimem-stores it into every rank's copy (runner.reduce_output) ---
    @gemm_epilogue(reduces={"sumsq": ColVecReduce("sumsq", scaled=True)})
    def sumsq_epi(acc):
        return {"D": acc, "sumsq": (acc, acc)}

    ar = AllReduceRunner(m, n, dtype, tile_m, tile_n)
    sumsq, sumsq_view = ar.reduce_output((m, n // tile_n), torch.float32)
    for it in range(NUM_ITERS):
        a, b = operands(3500 + 10 * it)
        sumsq_epi.gemm(
            a,
            b,
            ar.d,
            None,
            epi_args={"sumsq": sumsq_view},
            dist_op=ar.op,
            dist_op_args=ar.prepare(),
            **launch,
        )
        torch.cuda.synchronize(device)
        ref, base = full_ref(a, b), nccl_ar(a, b).float()
        check(f"ar+sumsq D iter={it}", ar.d, ref, base)
        check(f"ar+sumsq iter={it}", sumsq.sum(-1), (ref * ref).sum(-1), (base * base).sum(-1))

    # --- declared limits, refused on the host: an aux output would land on the reducing rank alone; a
    # quantized D's scale bytes are not replicated ---
    relu_mod = linear_act_mod("relu", gated=False, has_c=False, has_rowvec=False, has_colvec=False)
    ar = AllReduceRunner(m, n, dtype, tile_m, tile_n)
    a, b = operands(100)
    post = torch.zeros(m, n, dtype=dtype, device=device)
    try:
        relu_mod.gemm(
            a,
            b,
            ar.d,
            None,
            epi_args={"mAuxOut": post},
            dist_op=ar.op,
            dist_op_args=ar.prepare(),
            **launch,
        )
        failures.append(f"rank{rank} ar+aux: not refused")
    except ValueError:
        pass
    if sm100:
        arq = AllReduceRunner(m, n, torch.float8_e4m3fn, tile_m, tile_n)
        sfd = torch.empty(
            1, m // 128, n // 128, 32, 4, 4, dtype=torch.float8_e8m0fnu, device=device
        )
        try:
            gemm(
                a,
                b,
                arq.d,
                None,
                None,
                tile_m,
                tile_n,
                1,
                1,
                raster_order=raster,
                dist_op=arq.op,
                dist_op_args=arq.prepare(),
                SFD=sfd,
            )
            failures.append(f"rank{rank} ar+requant: not refused")
        except (ValueError, AssertionError):
            pass

    if sm100:  # block-scaled operands: the reducers are format-agnostic (bf16 partials, the EVT after the sum)
        from quack.blockscaled.operand import BlockScaledOperand
        from quack.blockscaled.quantize import nvfp4_per_tensor_scale

        for name in ("mxfp8_e4m3", "nvfp4"):
            ar = AllReduceRunner(m, n, dtype, tile_m, tile_n)
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
                    ar.d,
                    None,
                    None,
                    tile_m,
                    tile_n,
                    1,
                    1,
                    raster_order=raster,
                    alpha=float(pts_a * pts_b) if name == "nvfp4" else 1.0,
                    dist_op=ar.op,
                    dist_op_args=ar.prepare(),
                    SFA=a_op.scale,
                    SFB=b_op.scale,
                    bs_format_a=name,
                    bs_format_b=name,
                )
                torch.cuda.synchronize(device)
                a_dq, b_dq = (op.dequantize(torch.float32).to(dtype) for op in (a_op, b_op))
                check(f"ar {name} iter={it}", ar.d, full_ref(a_dq, b_dq), nccl_ar(a_dq, b_dq))

    fail_count = torch.tensor([len(failures)], device=device)
    dist.all_reduce(fail_count)
    if failures:
        print("\n".join(failures), file=sys.stderr, flush=True)
    dist.destroy_process_group()
    if int(fail_count.item()) > 0:
        sys.exit(1)
    if rank == 0:
        print("ALL GEMM+AR CHECKS PASSED", flush=True)


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
def test_all_reduce_gemm_multirank():
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
    assert "ALL GEMM+AR CHECKS PASSED" in res.stdout


if __name__ == "__main__":
    _run_rank()
