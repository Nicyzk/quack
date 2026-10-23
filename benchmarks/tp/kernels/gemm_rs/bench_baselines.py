"""Run external-library GEMM+RS baselines through one interface.

These are NOT our kernels -- they are third-party / upstream implementations we
benchmark against. Each is registered in BASELINES with a tiny spec describing
how it must be launched:

  tk         ThunderKittens matmul_reduce_scatter
             ~/ThunderKittens/kernels/parallel/gemm_rs  (NCCL init; needs GPU env)
  blackwell  cutlass distributed_gemm_reduce_scatter_blackwell (fused, LDMC RS)
             ~/cutlass/examples/python/CuTeDSL/cute/blackwell/kernel/distributed  (torch symm_mem)
  82         cutlass 82_blackwell_distributed_gemm -- prebuilt C++ binary
             ~/cutlass/build/examples/82_blackwell_distributed_gemm  (subprocess; self-manages TP)
  flux       ByteDance flux GemmRS -- H100 (sm90)-only, own conda env
             (subprocess to h100_lib_runner.py under that env's torchrun)
  triton_dist Triton-distributed gemm_rs -- H100 (sm90)-only, own conda env
             (subprocess, same runner)

tk/blackwell run in-process under torchrun and are timed against a cuBLAS+NCCL
GEMM+RS reference, by default under the L2-cold rotate protocol shared with
ag_gemm (do_bench_l2_rotate_all: cloned (A, B) k-shards rotate, symm
workspaces stay fixed). 82 self-manages its GPUs and prints its own numbers,
so we just shell out to it (no torchrun). quack's tile comes from quack's stock
autotuner on the plain GEMM at the rank-local shape, on rank 0
(utils.tune_quack_tile, sonic's method); the sweep is that winner plus the
256x256 / 256x512 c(2,1) fallbacks, each at swizzle 1 and 8, and the cutlass arm
sweeps the same set minus what its can_implement rejects (tile N > 256).

K is the global contraction dim; it is sharded across TP ranks (k_local = K/world)
and A,B ~ N(0, 1/sqrt(K)). Per-rank FLOPs = 2*M*N*k_local.

Each in-process factory returns a Kernel: run/free plus the rank's A,B k-shards
and read_out/zero handles, so --ref_check / --r2r_check work generically for any
kernel (see ref_check/r2r_check). 82 is an external binary we only time, so its
--ref_check / --r2r_check are N/A.

Shapes come from --model / --site (utils.MODEL_CONFIGS, SITES): global N and K, K split
by the world size (k_local = K/TP).

Usage:
    GPU=B300 torchrun --nproc_per_node=8 bench_baselines.py tk        --model llama3_70b --site out --m 8192
    torchrun --nproc_per_node=8 bench_baselines.py cutlass --model llama3_70b --site out --m 8192 --ref_check
    bench_baselines.py 82 --model llama3_70b --site out --m 8192       # plain python; binary handles TP
"""
import argparse
import math
import os
import re
import subprocess
import sys
import time
from collections import namedtuple
from functools import partial

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "misc"))
from bench_rotate import do_bench_l2_rotate_all, l2_rotate_arg_sets, l2_rotate_count  # noqa: E402
from utils import (  # noqa: E402
    MODEL_CONFIGS,
    SITES,
    do_bench_counts,
    clean_print,
    parse_quack_configs,
    tune_quack_tile,
    quack_config_label,
    torchrun_init,
    torchrun_finalize,
    torchrun_init_nvshmem,
    torchrun_finalize_nvshmem,
)

_NOOP = lambda: None  # noqa: E731
# quack config candidates (tile x swizzle; static only — epi_reduce_mode requires static
# persistence); main overrides from --quack_configs
quack_configs = parse_quack_configs(clc_ok=False)

# run/free: timed call + cleanup. A,B: the rank's k-shards (M,k_local)/(N,k_local) for the
# fp32 reference. read_out: rank's output slab (m//world, N) as fp32. zero: reset the output buffer.
# calls: iterations one run() performs (graph arms replay several); ms is divided by it.
# raw: the device-resident output run() writes (tensor or holder dict) -- epilogue-
# composing benches (bench_llama_rs_oproj) read it without read_out's fp32 copy.
# fn: run on caller-supplied k-shards (the rotate protocol's per-set launch); None for
# arms whose operands live in their own buffers.
Kernel = namedtuple("Kernel", ["run", "free", "A", "B", "read_out", "zero", "calls",
                               "raw", "fn"],
                    defaults=[1, None, None])

def _cutlass_configs():
    """The quack sweep's set for the cutlass arm: (tile_m, tile_n, cluster_m, cluster_n,
    swizzle), scheduler dropped (the example has one); its can_implement rejects the rest."""
    return list(dict.fromkeys(c[:5] for c in quack_configs))


def _operands(dtype, M, N, K, world):
    """The rank's (M, k_local)/(N, k_local) k-shards ~ N(0, 1/sqrt(K))."""
    k_local = K // world
    A = torch.randn(M, k_local, dtype=dtype, device="cuda") / math.sqrt(K)
    B = torch.randn(N, k_local, dtype=dtype, device="cuda") / math.sqrt(K)
    return A, B


def _quack(dtype, M, N, K, rank, world):
    """quack gemm with epi_reduce_mode="reduce_scatter" (branch intra-sm).
    Eager like every other arm here — one kernel launch per iteration, the
    FFI floor hides under the burst backlog (the AG-side graph compensated
    the AllGatherRunner's multi-op host chain, which RS doesn't have)."""
    from quack.distributed.gemm_epi_reduce import make_epi_reduce_args
    from quack.gemm import gemm as quack_gemm

    A, B = _operands(dtype, M, N, K, world)
    tile_m, tile_n, cluster_m, cluster_n, swizzle_default, clc = quack_configs[0]
    d_arg, era = make_epi_reduce_args(
        "reduce_scatter", dtype, M, N, 1, tile_m, tile_n, cluster_m, world)

    def fn(a, b, swizzle=swizzle_default):  # caller-order (l, m/n, k_local)
        quack_gemm(a.unsqueeze(0), b.unsqueeze(0), d_arg, None, None, tile_m, tile_n,
                   cluster_m, cluster_n, is_dynamic_persistent=clc, max_swizzle_size=swizzle,
                   epi_reduce_mode="reduce_scatter", epi_reduce_args=era)

    return Kernel(lambda: fn(A, B), _NOOP, A, B, lambda: d_arg[0].float(), d_arg.zero_, fn=fn)


def _tk_import(world):
    """Load TK gemm_rs _C for this world size. NUM_DEVICES is compile-time in
    gemm_rs_h100.cu, so _C is per-world; _C_nd{2,4,8} are cross-built alongside
    the default _C (nd8). Falls back to _C if no per-world build exists."""
    import glob
    import importlib.util

    d = os.path.expanduser("~/ThunderKittens/kernels/parallel/gemm_rs")
    cand = glob.glob(os.path.join(d, f"_C_nd{world}.*.so")) or glob.glob(os.path.join(d, "_C.*.so"))
    assert cand, f"no TK gemm_rs _C for world={world}; build _C_nd{world} (NUM_DEVICES={world})"
    spec = importlib.util.spec_from_file_location("_C", cand[0])
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.TKParallelTensor, m.matmul_reduce_scatter


def _tk_operands(dtype, M, N, K, rank, world):
    """Standardized (A, B_std) shared with quack, plus B in TK's kernel layout
    (H100: (k_local, N); Blackwell: (N, k_local))."""
    gpu = os.environ.get("GPU")
    assert gpu in ("B200", "B300", "H100"), "set GPU=B200|B300|H100 for tk"
    A, B_std = _operands(dtype, M, N, K, world)  # A (M,k_local), B_std (N,k_local)
    B = B_std.T.contiguous() if gpu == "H100" else B_std
    return A, B_std, B


def _tk(dtype, M, N, K, rank, world):
    """ThunderKittens matmul_reduce_scatter (per-world _C; standardized operands)."""
    TKParallelTensor, matmul_reduce_scatter = _tk_import(world)
    A, B_std, B = _tk_operands(dtype, M, N, K, rank, world)

    C_tk = TKParallelTensor((M // world, N), dtype=torch.bfloat16,
                            local_rank=rank, local_world_size=world, multicast=False)
    C_tk.data_.zero_()
    barrier = TKParallelTensor((1, 1), dtype=torch.int,
                               local_rank=rank, local_world_size=world, multicast=True)
    barrier.data_.zero_()

    def fn(a, b):  # a=(M,k_local), b=TK-layout B
        matmul_reduce_scatter(a, b, C_tk, barrier)

    # Kernel.B = B_std (caller order) for ref_check; fn consumes the TK-layout B.
    return Kernel(lambda: fn(A, B), _NOOP, A, B_std, lambda: C_tk.data_.float(),
                  C_tk.data_.zero_, raw=C_tk.data_, fn=fn)


def _blackwell_build(dtype, M, N, K, rank, world, A, B, n_sets=1, configs=None):
    """cutlass distributed_gemm_reduce_scatter_blackwell (fused, LDMC), compiled
    once per config (tile_m, tile_n, cluster_m, cluster_n, swizzle) in configs
    (default: the M-divisible 256x256 c(2,1) sw8); configs the example's
    can_implement rejects are skipped and named once, the survivors come back
    as the 4th result; n_sets FULLY independent sets from repeated allocate_tensors calls
    (the example owns the cute views), each seeded with A/B — every set carries
    its own operands, comm workspace AND flags, so the rotation turns over all
    args like quack's autotuner contract (no shared symmetric state across the
    captured launches). Returns (launch(args, swizzle), per-set kernel args,
    set-0 tensors).

    Upstream rewrote this example (2026-08): the kernel takes rank_id/num_ranks
    at construction, comm buffers come from its own allocate_tensors (torch
    symm_mem, not nvshmem), and the flag set is unicast + multicast + one
    unicast view per peer. allocate_tensors seeds A/B with small INTEGERS, whose
    low entropy settles a higher clock than randn — overwrite with this bench's
    randn/sqrt(K) operands so every arm sees the same distribution."""
    sys.path.insert(0, os.path.expanduser(
        "~/cutlass/examples/python/CuTeDSL/cute/blackwell/kernel/distributed"))
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as cutlass_utils
    import cutlass.cute.testing as testing
    from distributed_gemm_reduce_scatter_blackwell import (
        Sm100PersistentDenseGemmReduceScatterLDMCKernel as KernelCls,
        allocate_tensors,
    )

    ab_dtype = c_dtype = cutlass.BFloat16
    acc_dtype = cutlass.Float32
    if configs is None:  # the example rejects partial tiles: the cluster tile must divide M and N
        configs = [(256, 256, 2, 1, 8) if M % 256 == 0 else
                   (128, 256, 1, 1, 8) if M % 128 == 0 else (64, 64, 1, 1, 8)]
    k_local = K // world
    mnkl = (M, N, k_local, 1)

    gemms = {}
    for cfg in configs:
        tm, tn, cm, cn, sw = cfg
        use_2cta = cm % 2 == 0 and tm in (128, 256)  # quack's 2-CTA MMA rule (cta_tile_shape_m)
        gemm = KernelCls(
            acc_dtype, c_dtype, use_2cta, (tm, tn), (cm, cn), True,
            rank_id=rank, num_ranks=world, reduce_scatter="LDMC", swizzle_size=sw)
        gemm.__call__.set_name_prefix(f"bench_rs_ldmc_r{rank}_w{world}_{tm}x{tn}_c{cm}x{cn}_sw{sw}")
        try:
            gemm.can_implement(mnkl, ab_dtype, c_dtype, "k", "k", "n")
        except testing.CantImplementError as e:
            clean_print(f"  cutlass skips {cutlass_config_label(cfg)}: {e}", print_once=True)
            continue
        gemms[cfg] = gemm
    assert gemms, f"cutlass RS: no config in {configs} is implementable at {mnkl}"

    sets = []
    for _ in range(n_sets):
        t = allocate_tensors(
            mnkl=mnkl, ab_dtype=ab_dtype, c_dtype=c_dtype, a_major="k", b_major="k",
            c_major="n", world_size=world, num_workspace=1,
            device=torch.device("cuda", torch.cuda.current_device()),
            slot_init_mode="test", global_rank=rank)
        t["_anchors"][0][0].copy_(A.unsqueeze(-1))
        t["_anchors"][1][0].copy_(B.unsqueeze(-1))
        sets.append(t)
    t0 = sets[0]
    def _args(t):  # every arg from this set's own symmetric allocation
        return dict(
            a=t["cute_tensor_a_list"][0], b=t["cute_tensor_b_list"][0], c=t["cute_tensor_c"],
            comm_in_multicast_tensor=t["cute_tensor_comm_in_mc"],
            comm_out_unicast_tensor=t["cute_tensor_comm_out_uc"],
            barrier_flag_unicast=t["cute_tensor_flag_unicast"],
            barrier_flag_multicast=t["cute_tensor_flag_multicast"],
            barrier_flag_unicast_per_peer=t["cute_tensor_flag_unicast_per_peer"])
    arg_list = [_args(t) for t in sets]

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    hw = cutlass_utils.HardwareInfo()
    compiled = {cfg: cute.compile(g, max_active_clusters=hw.get_max_active_clusters(cfg[2] * cfg[3]),
                                  stream=stream, **arg_list[0]) for cfg, g in gemms.items()}
    kept = list(gemms)

    def launch(args, cfg=kept[0]):
        # stream at call time, not bake: capture swaps the current stream
        compiled[cfg](**args, stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream))

    return launch, arg_list, t0, kept


def _blackwell(dtype, M, N, K, rank, world, A=None, B=None):
    """cutlass RS Kernel. A/B: optional (M, k_local)/(N, k_local) bf16 operand
    overrides (default: fresh randn/sqrt(K)) — pipeline benches pass their
    standardized inputs."""
    if A is None or B is None:
        A0, B0 = _operands(torch.bfloat16, M, N, K, world)
        A, B = (A0 if A is None else A), (B0 if B is None else B)
    launch, (args,), t, _kept = _blackwell_build(dtype, M, N, K, rank, world, A, B,
                                                 configs=_cutlass_configs())
    # comm_out is this rank's (M/world, N, L) slab; torch owns every allocation.
    out = t["torch_tensor_comm_out_gpu"]
    read_out = lambda: out.permute(2, 0, 1)[0].float()  # noqa: E731
    return Kernel(lambda: launch(args), _NOOP, A, B, read_out, out.zero_)


def cutlass_config_label(cfg):
    tm, tn, cm, cn, sw = cfg
    return f"{tm}x{tn} c({cm},{cn}) sw{sw}"


def _blackwell_l2_rotate_inputs(dtype, M, N, K, rank, world):
    """(fn, arg_sets, sweep) for do_bench_l2_rotate_all: one operand set per
    rotation slot; fn(args) launches the shared compiled kernel on it, the
    sweep ranks the quack sweep's configs the example can implement."""
    A, B = _operands(dtype, M, N, K, world)
    launch, arg_list, t, kept = _blackwell_build(dtype, M, N, K, rank, world, A, B,
                                                 n_sets=l2_rotate_count(A, B),
                                                 configs=_cutlass_configs())
    sweep = [(cutlass_config_label(c), partial(launch, cfg=c)) for c in kept]
    return launch, [(args,) for args in arg_list], (t,), sweep


def _nccl(dtype, M, N, K, rank, world):
    """cuBLAS GEMM + NCCL reduce-scatter reference."""
    A, B = _operands(torch.bfloat16, M, N, K, world)
    D_full = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    D_rs = torch.empty(M // world, N, dtype=torch.bfloat16, device="cuda")

    def fn(a, b):
        torch.matmul(a, b.T, out=D_full)
        dist.reduce_scatter_tensor(D_rs, D_full)

    return Kernel(lambda: fn(A, B), _NOOP, A, B, lambda: D_rs.float(), D_rs.zero_, fn=fn)


def _legs_l2_rotate_inputs(dtype, M, N, K, rank, world):
    """The bound legs under do_bench_l2_rotate_all: cuBLAS GEMM = the solo
    per-rank (M, N, k_local) GEMM; NCCL RS = reduce_scatter of the bf16 (M, N)
    partial alone. main reports the slower one as the bound (perfect-overlap
    floor)."""
    A, B = _operands(dtype, M, N, K, world)
    D_full = torch.empty(M, N, dtype=dtype, device="cuda")
    D_rs = torch.empty(M // world, N, dtype=dtype, device="cuda")

    def gemm_leg(a, b):
        torch.matmul(a, b.T, out=D_full)

    def rs_leg(a, b):
        dist.reduce_scatter_tensor(D_rs, D_full)

    sweep = [("cuBLAS GEMM", gemm_leg), ("NCCL RS", rs_leg)]
    return gemm_leg, l2_rotate_arg_sets(A, B), (D_full, D_rs), sweep


def _asynctp(dtype, M, N, K, rank, world):
    """PyTorch async-TP: micro-pipelined reduce_scatter(A @ B) via symmetric memory
    (the decomposition torchtitan's enable_async_tensor_parallel applies; private API,
    torch 2.13). A (M, k_local) contiguous with scatter_dim=0 is the zero-copy layout."""
    import torch.distributed._symmetric_memory as symm

    gn = dist.group.WORLD.group_name
    symm.enable_symm_mem_for_group(gn)
    A, B = _operands(torch.bfloat16, M, N, K, world)
    B_kn = B.T.contiguous()  # a COPY: operand overrides must refresh it too
    res = {"B_kn": B_kn}

    def fn(a, b_kn):
        # op-level entry per the official async-TP usage; same impl as the python fn
        res["out"] = torch.ops.symm_mem.fused_matmul_reduce_scatter(a, b_kn, "sum", 0, gn)

    # output is freshly allocated each call, so zero is a no-op
    return Kernel(lambda: fn(A, B_kn), _NOOP, A, B, lambda: res["out"].float(), _NOOP,
                  raw=res, fn=fn)


def _kernel_l2_rotate_inputs(factory, dtype, M, N, K, rank, world, swizzles=None):
    """(fn, arg_sets[, sweep]) for do_bench_l2_rotate_all from a Kernel with fn:
    cloned (A, B) k-shards are the L2-turnover payload; the plain-tensor arms
    (NCCL ref) have no symmetric state, so only A/B rotate. swizzles: raster
    swizzle candidates for the sweep (the best is reported)."""
    kern = factory(dtype, M, N, K, rank, world)
    out = (kern.fn, l2_rotate_arg_sets(kern.A, kern.B), (kern,))
    if swizzles:
        out += ([(f"sw{s}", partial(kern.fn, swizzle=s)) for s in swizzles],)
    return out


def _quack_rs_l2_rotate_inputs(dtype, M, N, K, rank, world):
    """quack RS rotate helper that turns over ALL args (quack autotuner
    contract): each of n_sets slots gets its own cloned (A, B) k-shards AND its
    own symmetric (d_arg, era) per candidate tile — output + workspace + flags
    depend on the tile — so no symmetric state is shared across the captured
    launches. n_sets is sized by the (A, B) footprint and capped by free memory
    in l2_rotate_count; each era is a collective alloc, so all ranks build the
    same count in the same order. The sweep runs every --quack_configs entry."""
    from quack.distributed.gemm_epi_reduce import make_epi_reduce_args
    from quack.gemm import gemm as quack_gemm

    tiles = sorted({c[:4] for c in quack_configs})
    n_sets = l2_rotate_count(*_operands(dtype, M, N, K, world))
    sets = []
    for _ in range(n_sets):
        A, B = _operands(dtype, M, N, K, world)
        eras = {
            t: make_epi_reduce_args("reduce_scatter", dtype, M, N, 1, t[0], t[1], t[2], world)
            for t in tiles
        }
        sets.append((A, B, eras))

    def make_fn(cfg):
        tile_m, tile_n, cluster_m, cluster_n, swizzle, clc = cfg

        def fn(a, b, eras):
            d_arg, era = eras[cfg[:4]]
            quack_gemm(a.unsqueeze(0), b.unsqueeze(0), d_arg, None, None, tile_m, tile_n,
                       cluster_m, cluster_n, is_dynamic_persistent=clc,
                       max_swizzle_size=swizzle, epi_reduce_mode="reduce_scatter",
                       epi_reduce_args=era)
        return fn

    arg_sets = [tuple(s) for s in sets]
    sweep = [(quack_config_label(c), make_fn(c)) for c in quack_configs]
    return sweep[0][1], arg_sets, sets, sweep


def _asynctp_l2_rotate_inputs(dtype, M, N, K, rank, world):
    """asynctp consumes B as its (k_local, N) copy, so that is what rotates."""
    kern = _asynctp(dtype, M, N, K, rank, world)
    return kern.fn, l2_rotate_arg_sets(kern.A, kern.raw["B_kn"]), (kern,)


def _tk_rs_l2_rotate_inputs(dtype, M, N, K, rank, world, swizzles=None):
    """TK matmul_reduce_scatter rotate helper (matches the quack RS contract):
    each of n_sets slots gets its own cloned (A, B) k-shards AND its own
    symmetric C_tk/barrier, so no symmetric state is shared across the captured
    launches. B is TK-layout ((k_local, N) on H100). n_sets is rank-synced in
    l2_rotate_count; each TKParallelTensor is collective, so all ranks build the
    same count in the same order."""
    TKParallelTensor, matmul_reduce_scatter = _tk_import(world)
    n_sets = l2_rotate_count(*_operands(dtype, M, N, K, world))
    sets = []
    for _ in range(n_sets):
        A, _B_std, B = _tk_operands(dtype, M, N, K, rank, world)
        C_tk = TKParallelTensor((M // world, N), dtype=torch.bfloat16,
                                local_rank=rank, local_world_size=world, multicast=False)
        C_tk.data_.zero_()
        barrier = TKParallelTensor((1, 1), dtype=torch.int,
                                   local_rank=rank, local_world_size=world, multicast=True)
        barrier.data_.zero_()
        sets.append((A, B, C_tk, barrier))

    def fn(a, b, C_tk, barrier):
        matmul_reduce_scatter(a, b, C_tk, barrier)

    return fn, [tuple(s) for s in sets], sets


BASELINES = {
    "quack":       {"factory": _quack,     "nvshmem": False},
    "tk":          {"factory": _tk,        "nvshmem": False},
    "cutlass":     {"factory": _blackwell, "nvshmem": False},
    "asynctp":     {"factory": _asynctp,   "nvshmem": False},
    "legs":        {"factory": None,       "nvshmem": False},  # bound legs, timing-only
    "82":          {"binary": True},
    "flux":        {"h100_lib": True},
    "triton_dist": {"h100_lib": True},
}

H100_LIB_ENVS = {
    "flux": "/scratch/gpfs/TRIDAO/zy4586/envs/flux",
    "triton_dist": "/scratch/gpfs/TRIDAO/zy4586/envs/triton-dist-main",  # upstream main 7908e4e, 2026-09-18
}


def run_h100_lib(lib, M, N, K):
    """flux / Triton-distributed only build for H100 (sm90) and live in their
    own conda envs, so shell out to h100_lib_runner.py under that env's
    torchrun (plain-python invocation, like 82; world = visible GPU count)."""
    cap = torch.cuda.get_device_capability()
    if cap != (9, 0):
        sys.exit(f"{lib} baseline is H100 (sm90)-only; this GPU is sm{cap[0]}{cap[1]}")
    torchrun = os.path.join(H100_LIB_ENVS[lib], "bin", "torchrun")
    runner = os.path.join(os.path.dirname(os.path.abspath(__file__)), "h100_lib_runner.py")
    env = os.environ.copy()
    if lib == "flux":
        # report IPC peer buffers as local so torch's from_blob device check passes
        env["LD_PRELOAD"] = "/scratch/gpfs/TRIDAO/zy4586/envs/ipc_attr_shim.so"
        # flux's loader looks in python/flux/lib, the build installs into lib64
        env["LD_LIBRARY_PATH"] = os.path.expanduser("~/flux/python/flux/lib64") + ":" + env.get(
            "LD_LIBRARY_PATH", ""
        )
    else:  # nvshmem's host lib is not on the loader path; env/lib = the libstdc++ (>= GLIBCXX_3.4.30) libtriton links
        env["LD_LIBRARY_PATH"] = ":".join([
            os.path.join(H100_LIB_ENVS[lib], "lib/python3.12/site-packages/nvidia/nvshmem/lib"),
            os.path.join(H100_LIB_ENVS[lib], "lib"), env.get("LD_LIBRARY_PATH", "")])
    # "--": torchrun abbrev-matches --m/--n/--k to its own flags without it
    subprocess.run([torchrun, f"--nproc_per_node={torch.cuda.device_count()}", "--", runner,
                    lib, f"--m={M}", f"--n={N}", f"--k={K}"], check=True, env=env)


def tflops(ms, M, N, K, world):
    return 2 * M * N * (K // world) / (ms * 1e9)


def _run_synced(kern):
    """Zero, barrier, run, sync, barrier -- one fresh cross-rank-consistent output."""
    kern.zero()
    torch.cuda.synchronize()
    dist.barrier()
    kern.run()
    torch.cuda.synchronize()
    dist.barrier()


def ref_check(kern, rank, world):
    """Max rel err of kern's output vs an fp32 full-K reference (all-gather k-shards, bmm, slice M-band)."""
    _run_synced(kern)
    got = kern.read_out()
    A_shards = [torch.empty_like(kern.A) for _ in range(world)]
    B_shards = [torch.empty_like(kern.B) for _ in range(world)]
    dist.all_gather(A_shards, kern.A)
    dist.all_gather(B_shards, kern.B)
    A_full = torch.cat(A_shards, dim=1).float()  # (M, K)
    B_full = torch.cat(B_shards, dim=1).float()  # (N, K)
    m0, mloc = rank * (kern.A.shape[0] // world), kern.A.shape[0] // world
    ref = A_full[m0:m0 + mloc] @ B_full.T        # (mloc, N)
    denom = ref.abs()
    mask = denom > 1e-3
    return ((got - ref).abs()[mask] / denom[mask]).max().item() if mask.any() else 0.0


def r2r_check(kern):
    """Run twice; return (bit-identical, max |delta|) of the rank's output slab."""
    _run_synced(kern)
    o1 = kern.read_out().clone()
    _run_synced(kern)
    o2 = kern.read_out().clone()
    return torch.equal(o1, o2), (o1 - o2).abs().max().item()


def run_82(M, N, K):
    """Shell out to the prebuilt cutlass 82_blackwell binary (timing-only, like the .sh;
    it self-manages TP. Its built-in verify is bf16-noisy, so we always --skip_verify)."""
    # the .rs_tp4 build (RS TilingA, K-sharded) — main's example binary is a
    # DIFFERENT decomposition (M/N-sharded, K full) and not comparable.
    # Worktree removed 2026-08-15; rebuild from cutlass branch 82-tp4
    # (worktree + cmake 100a|103a gate) or pass BIN=.
    # TP is compile-time (using TP = _N); per-TP binaries .rs_tp{2,4} next to the build
    tp = torch.cuda.device_count()
    binary = os.environ.get("BIN", os.path.expanduser(
        "~/tmp/cutlass-82/build/examples/82_blackwell_distributed_gemm/"
        f"82_blackwell_distributed_gemm.rs_tp{tp}"))
    # the binary's harness rotates cloned operand sets in its timed loop
    # (quack _pick_l2_rotate_count semantics) with one event pair + cross-device max
    print(f"<82_blackwell | {M}x{K}x{N} | L2-rotate (in-binary)>", flush=True)
    p = subprocess.run([binary, f"--m={M}", f"--n={N}", f"--k={K}", "--data_dist=randn",
                        "--warmup-iterations=100", "--iterations=500", "--skip_verify"],
                       check=True, capture_output=True, text=True)
    print(p.stdout, end="", flush=True)
    ms = float(re.search(r"Avg runtime: ([\d.]+) ms", p.stdout).group(1))
    print(f"  {'82:':<14} {ms:.3f} ms | {tflops(ms, M, N, K, tp):.2f} TFLOp/s", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("baseline", choices=list(BASELINES))
    p.add_argument("--model", choices=list(MODEL_CONFIGS), required=True, help="shape source (MODEL_CONFIGS)")
    p.add_argument("--site", choices=("out", "down"), required=True, help="row-parallel GEMM (SITES)")
    p.add_argument("--m", type=int, required=True, help="rows (2048 prefill, 8192 training)")
    p.add_argument("--ref_check", action="store_true",
                   help="Max rel err of the kernel & reference vs an fp32 full-K GEMM.")
    p.add_argument("--r2r_check", action="store_true",
                   help="Run kernel & reference twice; report bit-identity and max |delta|.")
    p.add_argument("--eager", action="store_true",
                   help="Time kernel & reference eager (capture=False), as the flux/triton_dist runners do.")
    p.add_argument("--rounds", type=int, default=1,
                   help="re-time the winning config N-1 more times back-to-back (self-shadow only); "
                        "the reference is timed once, last, as the contention canary")
    p.add_argument("--quack_configs", default=None,
                   help="quack sweep: 'tm,tn,cm,cn[,sw,static];...' (default: quack's autotuner winner + "
                        "256x256/256x512 c(2,1) x swizzle {1,8}; static only under epi_reduce); "
                        "the cutlass arm sweeps the same set minus what it cannot implement")
    args = p.parse_args()
    M, (N, K) = args.m, SITES[args.site](MODEL_CONFIGS[args.model])
    global quack_configs
    quack_configs = parse_quack_configs(args.quack_configs, clc_ok=False)

    spec = BASELINES[args.baseline]
    if spec.get("binary"):
        run_82(M, N, K)
        if args.ref_check or args.r2r_check:
            print("  (82: ref/r2r check N/A -- external binary, timing-only)", flush=True)
        return
    if spec.get("h100_lib"):
        run_h100_lib(args.baseline, M, N, K)
        if args.ref_check or args.r2r_check:
            print(f"  ({args.baseline}: ref/r2r check N/A -- external env, timing-only)", flush=True)
        return

    if spec["nvshmem"]:
        torchrun_init_nvshmem()
        finalize = torchrun_finalize_nvshmem
    else:
        torchrun_init()
        finalize = torchrun_finalize
    rank, world = dist.get_rank(), dist.get_world_size()
    assert M % world == 0 and K % world == 0
    if args.quack_configs is None and args.baseline in ("quack", "cutlass"):
        tile = tune_quack_tile(M, N, K // world)
        tm, tn, cm, cn = tile
        cta_m = tm // 2 if cm % 2 == 0 and tm in (128, 256) else tm
        full = M % cta_m == 0 and N % tn == 0  # the RS epilogue takes full CTA tiles only
        quack_configs = list(dict.fromkeys(parse_quack_configs(";".join(
            ",".join(map(str, t)) for t in ([tile] if full else []) + [(256, 256, 2, 1), (256, 512, 2, 1)]),
            clc_ok=False)))
        clean_print(f"  quack tuner: {tm}x{tn} c({cm},{cn}){'' if full else ' (partial tiles, dropped)'}",
                    print_once=True)
    # Seed so every arm (each a separate invocation) draws the SAME standardized
    # A,B ~ N(0, 1/sqrt(K)); matches gemm_ar's shared-input convention.
    torch.manual_seed(1111 + rank)

    # Default protocol: the quack-autotuner L2-cold rotate methodology
    # (do_bench_l2_rotate_all, as ag_gemm), so configs are picked and reported
    # under one regime; arms without a rotate helper fall back to counts and
    # say so. Checks always run through the plain factories (rotate is timing-only).
    # capture=False for asynctp: the symm_mem op host-blocks and is not graph-safe.
    ROTATE_INPUTS = {
        "quack":   (_quack_rs_l2_rotate_inputs, True),
        "tk":      (_tk_rs_l2_rotate_inputs, False),
        "cutlass": (_blackwell_l2_rotate_inputs, True),
        "asynctp": (_asynctp_l2_rotate_inputs, False),
        "legs":    (_legs_l2_rotate_inputs, True),
    }
    rotate_inputs, rotate_capture = ROTATE_INPUTS.get(args.baseline, (None, True))
    rotate_capture = rotate_capture and not args.eager
    need_plain = args.ref_check or args.r2r_check or rotate_inputs is None
    assert not (need_plain and spec["factory"] is None), "legs: timing-only, no checks"
    kern = spec["factory"](torch.bfloat16, M, N, K, rank, world) if need_plain else None
    ref = _nccl(torch.bfloat16, M, N, K, rank, world) if need_plain else None

    checks = []
    if args.ref_check:
        checks.append(("rel err", f"{ref_check(kern, rank, world):.4e} (|ref|>1e-3)",
                       f"{ref_check(ref, rank, world):.4e}"))
    if args.r2r_check:
        kb, kd = r2r_check(kern)
        rb, rd = r2r_check(ref)
        checks.append(("r2r", f"bit-identical={kb} max|Δ|={kd:.4e}",
                       f"bit-identical={rb} max|Δ|={rd:.4e}"))

    sweep_ms = []
    if rotate_inputs is not None:
        from quack.autotuner import _gpu_warmup

        proto = "L2-rotate" if rotate_capture else "L2-rotate (eager-timed)"
        out = rotate_inputs(torch.bfloat16, M, N, K, rank, world)
        fn, arg_sets, _keep = out[:3]
        sweep = out[3] if len(out) > 3 else None
        ref_fn, ref_arg_sets, _ref_keep = _kernel_l2_rotate_inputs(
            _nccl, torch.bfloat16, M, N, K, rank, world)
        _gpu_warmup()
        kwarg_sets = [{}] * len(arg_sets)
        if sweep:  # config candidates: report each, rank the best
            for label, fn_v in sweep:
                dist.barrier()
                sweep_ms.append((label, do_bench_l2_rotate_all(
                    fn_v, arg_sets, kwarg_sets, {}, capture=rotate_capture)))
            pick = max if args.baseline == "legs" else min  # legs: the bound is the slower leg
            best_label, ms = pick(sweep_ms, key=lambda r: r[1])
        else:
            dist.barrier()
            ms = do_bench_l2_rotate_all(fn, arg_sets, kwarg_sets, {}, capture=rotate_capture)
        dist.barrier()
        rounds = []  # (arm_ms, {leg: ms} | None) per extra round, winner config only.
        # Rounds run back-to-back right after the sweep; the ref chain is timed ONCE at the
        # end (its ~100 ms full-power window depresses the next timing's clocks ~5% at
        # heavy shapes -- the 82 binary self-times in isolation, so arm rounds must too).
        best_fn = dict(sweep)[best_label] if sweep_ms else fn
        for _ in range(args.rounds - 1):
            legs = None
            if args.baseline == "legs":  # every leg per round; the bound is the slower one
                legs = {}
                for label, fn_v in sweep:
                    dist.barrier()
                    legs[label] = do_bench_l2_rotate_all(fn_v, arg_sets, kwarg_sets, {}, capture=rotate_capture)
                v = max(legs.values())
            else:
                dist.barrier()
                v = do_bench_l2_rotate_all(best_fn, arg_sets, kwarg_sets, {}, capture=rotate_capture)
            rounds.append((v, legs))
        dist.barrier()
        ref_ms = do_bench_l2_rotate_all(ref_fn, ref_arg_sets, [{}] * len(ref_arg_sets), {},
                                        capture=rotate_capture)
    else:
        proto = "counts — not rotate-capable yet"
        rounds = []
        dist.barrier()
        time.sleep(1.0)  # prevent throttling
        ms = do_bench_counts(kern.run, calls_per_inv=kern.calls)
        dist.barrier()
        time.sleep(1.0)
        ref_ms = do_bench_counts(ref.run)

    clean_print("=" * 80, print_once=True)
    clean_print(f"<BF16 GEMM+RS {args.baseline} | world={world} | {M}x{K}x{N} | {proto}>",
                print_once=True)
    for label, v_ms in sweep_ms:
        clean_print(f"  {args.baseline + ' ' + label + ':':<22} {v_ms:.3f} ms | {tflops(v_ms, M, N, K, world):.2f} TFLOp/s", print_once=True)
    if sweep_ms:
        kern_name = (f"bound ({best_label})" if args.baseline == "legs"
                     else f"{args.baseline} best ({best_label})")
    else:
        kern_name = args.baseline
    clean_print(f"  {kern_name + ':':<14} {ms:.3f} ms | {tflops(ms, M, N, K, world):.2f} TFLOp/s", print_once=True)
    clean_print(f"  {'cuBLAS+NCCL:':<14} {ref_ms:.3f} ms | {tflops(ref_ms, M, N, K, world):.2f} TFLOp/s", print_once=True)
    for i, (v, legs) in enumerate(rounds, 1):
        for label, lv in (legs or {}).items():
            clean_print(f"  {args.baseline + ' ' + label + f' round{i}:':<22} {lv:.3f} ms | {tflops(lv, M, N, K, world):.2f} TFLOp/s", print_once=True)
        clean_print(f"  {args.baseline + f' round{i}:':<22} {v:.3f} ms | {tflops(v, M, N, K, world):.2f} TFLOp/s", print_once=True)
    for label, kv, rv in checks:
        clean_print(f"  {args.baseline + ' ' + label + ':':<22} {kv}", print_once=True)
        clean_print(f"  {'cuBLAS+NCCL ' + label + ':':<22} {rv}", print_once=True)

    if kern is not None:
        kern.free()
        ref.free()
    dist.barrier()
    finalize()


if __name__ == "__main__":
    main()
