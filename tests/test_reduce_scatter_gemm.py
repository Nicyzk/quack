# Copyright (c) 2026, QuACK team.
"""Multi-rank correctness tests for the comm-object GEMM (quack/distributed/reduce_scatter.py):
GEMM + reduce-scatter, GEMM + all-reduce, and a reduce epi op running in the comm warps.

Runs under torchrun (one rank per GPU); the pytest entry point spawns torchrun as a
subprocess on the free GPUs.

    torchrun --nproc_per_node=2 tests/test_reduce_scatter_gemm.py
    pytest tests/test_reduce_scatter_gemm.py -x
"""

import os
import subprocess
import sys

import pytest
import torch

TILE_M, TILE_N = 128, 256
NUM_ITERS = 3  # back-to-back launches: flag reset, exit barrier, PDL ordering


def _compile(gemm_cls, comm, epi_fake, dtype, has_c=False, own_d=False):
    """The dense 2D GEMM of gemm_cls with the comm object, compiled on fake tensors."""
    import cutlass
    import cutlass.cute as cute

    from quack.gemm_tvm_ffi_utils import (
        make_fake_gemm_tensors,
        make_fake_scheduler_args,
        make_fake_varlen_args,
    )

    gemm = gemm_cls(
        cutlass.Float32, dtype, (TILE_M, TILE_N), (1, 1, 1), use_clc_persistence=False, comm=comm
    )
    gemm.b_transposed = gemm.a_transposed = gemm.cd_transposed = False
    gemm.cd_packed = None
    mA, mB, mD, mC, m, n, k, l = make_fake_gemm_tensors(
        dtype, dtype, dtype, None, "k", "k", "n", "n", batched=False
    )
    from quack.compile_utils import make_fake_tensor

    def own():
        return make_fake_tensor(
            dtype, (cute.sym_int(), cute.sym_int()), leading_dim=1, divisibility=8
        )

    if has_c:  # its own shape: slab-shaped under RS, full under AR
        mC = own()
    if own_d:  # the comm's wire has its own extent (lamport: 3 x world mirrors stacked along M)
        mD = own()
    sched = make_fake_scheduler_args(False, False, l)
    varlen = make_fake_varlen_args(False, False, False, None)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        gemm,
        mA,
        mB,
        mD,
        mC,
        epi_fake,
        sched,
        varlen,
        stream,
        None,
        None,
        comm.fake_args(dtype),
        options="--enable-tvm-ffi",
    )


def _run_rank():
    import cutlass.cute as cute
    import torch.distributed as dist
    from cutlass import Float32

    from quack.compile_utils import make_fake_tensor
    from quack.cute_dsl_utils import (
        get_device_capacity,
        get_max_active_clusters,
        torch2cute_dtype_map,
    )
    from quack.distributed.reduce_scatter import AllReduceRunner, ReduceScatterRunner
    from quack.epilogue.frontend import gemm_epilogue
    from quack.epilogue.ops import RowVecReduce
    from quack.gemm_default_epi import GemmDefaultEpiMixin, GemmDefaultSm90, GemmDefaultSm100
    from quack.gemm_tvm_ffi_utils import make_scheduler_args
    from quack.rounding import RoundingMode

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    device = torch.device("cuda", rank % torch.cuda.device_count())
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    failures = []

    dtype = torch.bfloat16
    cdtype = torch2cute_dtype_map[dtype]
    m, n, k = 2048, 1024, 1024  # k per rank; m is a whole number of tiles per slab for world <= 8
    capacity = get_device_capacity(device)
    gemm_cls = GemmDefaultSm90 if capacity[0] == 9 else GemmDefaultSm100
    clusters = get_max_active_clusters(1, capacity)
    sched_rt = make_scheduler_args(clusters, 8, None)
    epi_fake = GemmDefaultEpiMixin.EpilogueArguments(
        add_to_output=False, rounding_mode=RoundingMode.RN
    )
    epi_rt = GemmDefaultEpiMixin.EpilogueArguments(add_to_output=None, rounding_mode=None)
    slab = m // world

    def check(label, out, ref, base, mult=3):
        """out vs the fp32 ref, within mult x the error of the same-dtype baseline (cuBLAS + NCCL's
        output). mult 3, not quack's usual 2: the switch's bf16 ld_reduce truncates (up to 2 ulp
        toward zero) where the baseline rounds once, which is exactly 2x at world 2."""
        tol = (base.float() - ref).abs().max().item() + 1e-4
        err = (out.float() - ref).abs().max().item()
        if err > mult * tol:
            failures.append(f"rank{rank} {label}: err={err} tol={tol}")

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

    def nccl_ar(a, b):
        out = a @ b.T
        dist.all_reduce(out)
        return out

    # --- reduce-scatter: each rank ends with its M slab of the full-K product ---
    rs = ReduceScatterRunner(m, n, dtype, TILE_M, TILE_N, clusters)
    fn = _compile(gemm_cls, rs.comm, epi_fake, cdtype)
    for it in range(NUM_ITERS):
        torch.manual_seed(1000 + 10 * it + rank)
        a = torch.randn(m, k, dtype=dtype, device=device) / 8
        b = torch.randn(n, k, dtype=dtype, device=device) / 8
        fn(a, b, rs.wire, None, epi_rt, sched_rt, None, None, None, rs.args())
        torch.cuda.synchronize(device)
        rows = slice(rank * slab, (rank + 1) * slab)
        check(f"rs iter={it}", rs.out, full_ref(a, b)[rows], nccl_rs(a, b))

    # --- all-reduce: every rank ends with the whole full-K product ---
    ar = AllReduceRunner(m, n, dtype, TILE_M, TILE_N, clusters)
    fn = _compile(gemm_cls, ar.comm, epi_fake, cdtype)
    for it in range(NUM_ITERS):
        torch.manual_seed(2000 + 10 * it + rank)
        a = torch.randn(m, k, dtype=dtype, device=device) / 8
        b = torch.randn(n, k, dtype=dtype, device=device) / 8
        fn(a, b, ar.wire, None, epi_rt, sched_rt, None, None, None, ar.args())
        torch.cuda.synchronize(device)
        check(f"ar iter={it}", ar.out, full_ref(a, b), nccl_ar(a, b))

    # --- C operand, applied after the reduce: RS adds this rank's slab residual, AR a replicated one ---
    rs = ReduceScatterRunner(m, n, dtype, TILE_M, TILE_N, clusters)
    fn = _compile(gemm_cls, rs.comm, epi_fake, cdtype, has_c=True)
    for it in range(NUM_ITERS):
        torch.manual_seed(4000 + 10 * it + rank)
        a = torch.randn(m, k, dtype=dtype, device=device) / 8
        b = torch.randn(n, k, dtype=dtype, device=device) / 8
        c = torch.randn(slab, n, dtype=dtype, device=device)
        fn(a, b, rs.wire, c, epi_rt, sched_rt, None, None, None, rs.args())
        torch.cuda.synchronize(device)
        rows = slice(rank * slab, (rank + 1) * slab)
        check(f"rs+c iter={it}", rs.out, full_ref(a, b)[rows] + c.float(), nccl_rs(a, b) + c)

    ar = AllReduceRunner(m, n, dtype, TILE_M, TILE_N, clusters)
    fn = _compile(gemm_cls, ar.comm, epi_fake, cdtype, has_c=True)
    for it in range(NUM_ITERS):
        torch.manual_seed(5000 + 10 * it + rank)
        a = torch.randn(m, k, dtype=dtype, device=device) / 8
        b = torch.randn(n, k, dtype=dtype, device=device) / 8
        torch.manual_seed(5000 + 10 * it)  # rank-invariant residual
        c = torch.randn(m, n, dtype=dtype, device=device)
        fn(a, b, ar.wire, c, epi_rt, sched_rt, None, None, None, ar.args())
        torch.cuda.synchronize(device)
        check(f"ar+c iter={it}", ar.out, full_ref(a, b) + c.float(), nccl_ar(a, b) + c)

    # --- one-shot Lamport all-reduce: no flags, the data is the flag; more launches than slots so the
    # 3-slot rotation and the re-arm are exercised ---
    from quack.distributed.lamport import LamportRunner

    for has_c in (False, True):
        lp = LamportRunner(m, n, dtype, TILE_M)
        fn = _compile(gemm_cls, lp.comm, epi_fake, cdtype, has_c=has_c, own_d=True)
        for it in range(max(NUM_ITERS, 5)):
            torch.manual_seed(6000 + 10 * it + rank)
            a = torch.randn(m, k, dtype=dtype, device=device) / 8
            b = torch.randn(n, k, dtype=dtype, device=device) / 8
            c = None
            if has_c:
                torch.manual_seed(6000 + 10 * it)  # rank-invariant residual
                c = torch.randn(m, n, dtype=dtype, device=device)
            lp.before_launch()
            fn(a, b, lp.wire, c, epi_rt, sched_rt, None, None, None, lp.args())
            torch.cuda.synchronize(device)
            ref, base = full_ref(a, b), nccl_ar(a, b)
            if has_c:
                ref, base = ref + c.float(), base + c
            check(f"lamport{'+c' if has_c else ''} iter={it}", lp.out, ref, base)

    # --- through an epilogue module: comm= / comm_args= on the torch-facing call ---
    from quack.epilogue.library import identity_epi
    from quack.gemm_config import GemmConfig

    cfg = GemmConfig(
        tile_m=TILE_M,
        tile_n=TILE_N,
        pingpong=False,
        is_dynamic_persistent=False,
        cluster_m=1,
        cluster_n=1,
        device_capacity=capacity[0],
    )
    rs = ReduceScatterRunner(m, n, dtype, TILE_M, TILE_N, clusters)
    for it in range(NUM_ITERS):
        torch.manual_seed(6000 + 10 * it + rank)
        a = torch.randn(m, k, dtype=dtype, device=device) / 8
        b = torch.randn(n, k, dtype=dtype, device=device) / 8
        out = identity_epi(a, b.T, config=cfg, comm=rs)["D"]  # the runner's result, not its wire
        torch.cuda.synchronize(device)
        rows = slice(rank * slab, (rank + 1) * slab)
        check(f"mod rs iter={it}", out, full_ref(a, b)[rows], nccl_rs(a, b))

    lp = LamportRunner(m, n, dtype, TILE_M)  # the wire's own extent passes the module path's checks
    for it in range(max(NUM_ITERS, 5)):
        torch.manual_seed(7000 + 10 * it + rank)
        a = torch.randn(m, k, dtype=dtype, device=device) / 8
        b = torch.randn(n, k, dtype=dtype, device=device) / 8
        out = identity_epi(a, b.T, config=cfg, comm=lp)["D"]
        torch.cuda.synchronize(device)
        check(f"mod lamport iter={it}", out, full_ref(a, b), nccl_ar(a, b))

    # --- reduce epi op in the comm warps: per-stripe column max_abs of the reduced slab ---
    @gemm_epilogue(reduces={"colmax": RowVecReduce("colmax", combine="max_abs")})
    def rowmax_epi(acc):
        return {"D": acc, "colmax": acc}

    cls = rowmax_epi._mint((), capacity[0], False, False, (), RoundingMode.RN, (), False)
    colmax_fake = make_fake_tensor(
        Float32, (cute.sym_int(), cute.sym_int(), cute.sym_int()), leading_dim=2
    )
    rs = ReduceScatterRunner(m, n, dtype, TILE_M, TILE_N, clusters)
    fn = _compile(cls, rs.comm, cls.EpilogueArguments(colmax=colmax_fake), cdtype)
    stripe = TILE_M // world  # the comm warps' frame: one reduce row per stripe of the slab
    colmax = torch.zeros(1, m // TILE_M, n, dtype=torch.float32, device=device)
    for it in range(NUM_ITERS):
        torch.manual_seed(3000 + 10 * it + rank)
        a = torch.randn(m, k, dtype=dtype, device=device) / 8
        b = torch.randn(n, k, dtype=dtype, device=device) / 8
        epi = cls.EpilogueArguments(colmax=colmax)
        fn(a, b, rs.wire, None, epi, sched_rt, None, None, None, rs.args())
        torch.cuda.synchronize(device)
        rows = slice(rank * slab, (rank + 1) * slab)
        ref, base = full_ref(a, b)[rows], nccl_rs(a, b).float()
        check(f"rs+rowmax D iter={it}", rs.out, ref, base)
        stripe_max = lambda x: x.view(m // TILE_M, stripe, n).abs().amax(1)
        check(f"rs+rowmax colmax iter={it}", colmax[0], stripe_max(ref), stripe_max(base))

    fail_count = torch.tensor([len(failures)], device=device)
    dist.all_reduce(fail_count)
    if failures:
        print("\n".join(failures), file=sys.stderr, flush=True)
    dist.destroy_process_group()
    if int(fail_count.item()) > 0:
        sys.exit(1)
    if rank == 0:
        print("ALL RS/AR GEMM CHECKS PASSED", flush=True)


def _visible_gpus() -> str:
    """All GPUs, minus any with a large foreign allocation (shared node)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        free = [
            line.split(",")[0].strip()
            for line in out.strip().splitlines()
            if int(line.split(",")[1]) < 20000
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
    assert "ALL RS/AR GEMM CHECKS PASSED" in res.stdout


if __name__ == "__main__":
    _run_rank()
