# EP benches

The expert-parallel benches behind the MoE tables (forward, forward + shared expert, backward, NVFP4 and
MXFP8 fused dispatch + fc1, fc2 + combine + RMSNorm, the walk-order ablations, and the H100 fused stages):
our fused kernels against the baselines. Mirrored from duckpond's `kernels/` with the same layout. Two
edits against the duckpond copies: the kernels take their inline-PTX helpers from `quack.ep.ptx` instead
of NVIDIA's megamoe tree (same PTX, verified string for string), and the benches put this directory's
`kernels/` package root on `sys.path` themselves.

## Layout

| dir | what |
|---|---|
| `kernels/moe/bench_forward.py` | W=8 forward, routed experts: `quack_intra_swap` (ours), `megamoe` (DeepGEMM), `deepep_deepgemm`, `deepep_te`, `sonic` |
| `kernels/moe/bench_forward_shared_expert.py` | + 1 shared expert: `quack_intra_swap`, `mok`, `megamoe`, DeepEP arms |
| `kernels/moe/bench_backward.py` | backward: `quack_intra_swap`, `mok`, `deepep_te` |
| `kernels/moe/bench_new_kernels.py` | the fused dispatch + fc1 stage in MXFP8 / NVFP4 with other activations (`none` = the SwiGLU-shaped fc1 with no activation), its bounds, its unfused pair (raw preact + the activation as a pass), and the NCCL EP + TE / NCCL + TE / DeepEP + TE baselines |
| `kernels/moe/bench_fused_dispatch_gemm.py`, `bench_fused_gemm_combine.py` | the two fused stages alone (B300 and H100): `quack_intra_swap`, `tk`, `sonic`, the legs; fc2 + combine + RMSNorm |
| `kernels/moe/bench_ablations.py` | tile walk order ablations (`walk`, `src-order`) and the source-order ablation of the quantized fused stage (`src-order-fc1`) |
| `kernels/moe/h100_flux_*_runner.py` | the Comet (flux) arms of the H100 stage tables, run under flux's env |
| `kernels/moe/dispatch/`, `combine/` | our prologue (routing metadata + walk), the intra-SM pull, the combine push and walk builders |
| `kernels/moe/fused_gemm_dispatch/`, `fused_gemm_combine/` | the fused GEMM kernels, SM100 and SM90: swap-AB varlen-N gated GEMM1 with the intra-SM pull; GEMM2 with the epilogue slot store |
| `kernels/moe/reduce_over_k/` | the gather-and-sum reduce with the fused residual + RMSNorm |
| `kernels/all/` | quack main's GEMM with the comm-object surface, used by the non-swap `quack_intra` arm only |
| `kernels/misc/` | `utils.py` (torchrun init, symmetric memory, graph capture), `nccl_env.py` (NCCL channel floor for every reference) |
| `kernels/moe/bench_dispatch.py`, `bench_combine.py` | the shared protocol (warm-up, rotated 500 ms blocks, slowest rank's median, ref checks) |

Models (`--model`): `dsv3`, `dsv3_e128`, `kimi_k2`, `dsv4_pro`, `glm_5_2`, `qwen3_5_397b`, `qwen3_235b`,
`mixtral_8x22b`, `olmoe`.

## Running

```bash
conda activate ml-env-cu13                              # torch cu130, cutlass-dsl 4.7, TE 2.19, DeepGEMM, nvshmem
export PYTHONPATH=<this branch's checkout>:~/sonic-moe-pr54
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH    # DeepGEMM's libstdc++
export TRITON_CACHE_AUTOTUNING=1                        # sonic's triton autotune choices persist across launches
cd benchmarks/ep/kernels/moe
torchrun --nproc_per_node=8 bench_forward.py quack_intra_swap megamoe deepep_deepgemm deepep_te --model dsv4_pro --t-local 4096 --no-canary --ref_check
# mok: one arm per comm-SM count x minibatch size, the table's cell = the best of the sweep
torchrun --nproc_per_node=8 bench_forward_shared_expert.py quack_intra_swap mok megamoe --model dsv4_pro --t-local 4096 --mok-comm-sms 16,20,24,28,32,36,40,44 --mok-minibatch 2048,4096,8192,16384 --no-canary --ref_check
torchrun --nproc_per_node=8 bench_backward.py quack_intra_swap mok deepep_te --model dsv4_pro --t-local 4096 --mok-comm-sms 16,20,24,28,32,36,40,44 --mok-minibatch 2048,4096,8192,16384 --ref_check
# the fc1 tables: NVFP4 = the SwiGLU-shaped fc1 with no activation, MXFP8 = relu²; NCCL EP (nccl_ep_te) and DeepEP
# (deepep_te) in separate invocations (DeepEP's import asserts on any mapped libnccl*, which NCCL EP's library trips)
PYTHONPATH=~/nccl-extensions/python-overlay:$PYTHONPATH CUDA_HOME=/usr/local/cuda-13.3 \
  torchrun --nproc_per_node=8 bench_new_kernels.py --model dsv4_pro --t-local 4096 --formats nvfp4 --activations none --baselines nccl_ep_te --nccl-ep-sms 16 --ref_check
PYTHONPATH=~/nccl-extensions/python-overlay:$PYTHONPATH CUDA_HOME=/usr/local/cuda-13.3 \
  torchrun --nproc_per_node=8 bench_new_kernels.py --model dsv4_pro --t-local 4096 --formats mxfp8 --activations relu_sq --baselines nccl_ep_te --nccl-ep-sms 16 --ref_check
torchrun --nproc_per_node=8 bench_new_kernels.py --model dsv4_pro --t-local 4096 --formats nvfp4 --activations none --baselines deepep_te --ref_check
torchrun --nproc_per_node=8 bench_fused_dispatch_gemm.py quack_intra_swap tk --model dsv4_pro --t-local 4096 --ref_check
torchrun --nproc_per_node=8 bench_fused_gemm_combine.py quack_intra_swap --model dsv4_pro --t-local 4096 --ref_check
torchrun --nproc_per_node=8 bench_ablations.py walk --model dsv4_pro --t-local 4096
torchrun --nproc_per_node=8 bench_ablations.py src-order-fc1 --model dsv4_pro --t-local 4096 --format nvfp4 --activation geglu --ref_check
```

NCCL EP's high-throughput dispatch deadlocks at 8 ranks above 16 SMs on this node (the tables use 16); the
`nccl_ep_te` chain = the dispatch of the pre-quantized rows + scale rows as bytes, then TE's single-launch
grouped-tensor GEMM over the landed zones (the landed scale rows swizzled for cuBLASLt per call) and the
activation as a TE op; its `nccl_ep alone` / `te fc1 alone` legs print beside the chain.

`--nproc_per_node` is the EP world size. Every arm ref-checks against an fp32 reference of the same routing.
The tables were collected with all 8 GPUs idle; the benches print a stability flag per block and the
results file leaves unstable cells out. H100 runs go through sbatch; the flux arms run from
`h100_flux_*_runner.py` under the flux env (see the docstring of each runner for the exact command).

## Versions

The B300 tables: della-tridao (8x B300, driver 610.57.04), conda env `ml-env-cu13`. Every baseline runs on
the same torch + NCCL as our kernels.

| component | version |
|---|---|
| torch | 2.13.0+cu130; its `torch.distributed` loads the env's `nvidia-nccl-cu13` 2.31.2 (the NCCL of the NCCL + TE column and of every reference; `nccl_env.py` sets its channel floor) |
| nvidia-cutlass-dsl | 4.7.0 (+ apache-tvm-ffi 0.1.12): the quack kernels |
| transformer_engine | 2.19.0 (`transformer_engine` + `transformer_engine_cu13` wheels; `transformer_engine_torch` built from the PyPI sdist against the env's torch, NCCL and cuDNN 9.20 headers): every TE grouped GEMM, the NVFP4 / MXFP8 recipes |
| nccl-extensions (NCCL EP) | PyPI 0.1.0, source tag `nccl-extensions-v0.1.0` (library reports 0.2.0); `libnccl_ep.so` rebuilt against NCCL 2.31.2 with CUDA 13.3 nvcc, which its JIT also needs (`CUDA_HOME=/usr/local/cuda-13.3`); nccl4py 0.3.1 |
| DeepEP | V2 (NCCL GIN backend), `~/DeepEP` at 01dc3aa (2026-08-04), built locally |
| DeepGEMM | 2.8.0+local, built from `~/DeepGEMM` at 78b6900 (2026-09-14) |
| mixture-of-kittens (MoK) | 0.1.0, `~/mixture-of-kittens` at 22fc95a, editable install |
| sonic-moe | `~/sonic-moe-pr54` (PR #54 checkout) |
| ThunderKittens (ParallelKittens) | `~/ThunderKittens` at 67845f5f (2026-09-12) |
| nvidia-nvshmem-cu13 / nvshmem4py-cu13, triton | 3.4.5 / 0.3.0, 3.7.1 |
| flux (Comet) | H100 only, its own env (below) |

## External baselines and dependencies

| what | where |
|---|---|
| sonic-moe (routing metadata, the `sonic` arms, the reference) | `~/sonic-moe-pr54` on `PYTHONPATH` |
| DeepEP V2 (`deepep_*` arms, the DeepEP + TE baselines) | `~/DeepEP`, added to `sys.path` by the benches |
| DeepGEMM (`megamoe`, `deepep_deepgemm`) | pip in the env; needs `LD_LIBRARY_PATH=$CONDA_PREFIX/lib` |
| Transformer Engine 2.19 (`deepep_te`, the TE grouped GEMMs, the NVFP4 / MXFP8 recipes) | pip in the env (core wheels; `transformer_engine_torch` built from the PyPI sdist against the env's torch, NCCL and cuDNN headers) |
| NCCL EP, NVIDIA nccl-extensions (`nccl_ep_te`) | `~/nccl-extensions`: `libnccl_ep.so` rebuilt against the env's NCCL (the PyPI wheel's HT mode faults with it; recipe in its `README-local.md`) under a private copy of the `nccl` package, `python-overlay/`, put first on `PYTHONPATH`; its JIT needs nvcc (`CUDA_HOME`) |
| Mixture-of-Kittens (`mok`) | `~/mixture-of-kittens`, installed editable in the env |
| ThunderKittens moe_dispatch_gemm (`tk` in the fused dispatch + GEMM1 bench, `_C_nd<W>` builds) | `~/ThunderKittens/kernels/parallel/moe_dispatch_gemm` |
| flux / Comet (H100 stage tables) | conda env `/scratch/gpfs/TRIDAO/zy4586/envs/flux` + `LD_PRELOAD` of `ipc_attr_shim.so` (source in the tp mirror's `kernels/misc/`) |
| nvshmem python bindings, triton | in the env |

Not mirrored: NVIDIA's cutedsl megamoe tree (the `megamoe_cutedsl` arm of bench_forward is not runnable
here; `megamoe` is DeepGEMM's and is), the dispatch-only ThunderKittens kernel of bench_dispatch's `tk`
arm, and the older inter-SM combine variants (`combine_rs.py`, `inter_combine_deepep.py`).
