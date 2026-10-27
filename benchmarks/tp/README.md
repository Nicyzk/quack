# TP benches

The tensor-parallel benches behind the GEMM+RS, GEMM+AR and AG+GEMM tables (B300 and H100): our quack arms
(`epi_reduce` reduce-scatter / all-reduce, `AllGatherRunner`) against the baselines. Mirrored from duckpond's
`kernels/` with the same layout; the only edit is that `sys.path` roots are relative to this directory, so the
tree runs from any checkout of this branch. Branch `tp` = `intra-sm` + this directory; refresh the kernels
with `git merge intra-sm`.

## Layout

| dir | benches | quack arm |
|---|---|---|
| `kernels/gemm_rs/` | `bench_baselines.py` (bf16), `bench_mxfp8_rs.py`, `bench_llama_rs_oproj.py` (+ residual + RMSNorm) | epi_reduce reduce-scatter |
| `kernels/gemm_ar/` | `bench_baselines.py` (decode, M=64), `bench_llama_ar.py` (+ residual + RMSNorm, decode M=64) | epi_reduce all-reduce 2-shot / lamport 1-shot |
| `kernels/ag_gemm/` | `bench_baselines.py` (bf16), `bench_mxfp8_ag.py`, `bench_llama_ag_swiglu.py` (+ SwiGLU) | AllGatherRunner |
| `kernels/ag/`, `kernels/rs/`, `kernels/ar/` | the standalone collective legs: NCCL all-gather, our multimem reduce-scatter and two-shot all-reduce; `ar/ar_norm_*.py` = the AR + RMSNorm kernels of `bench_llama_ar.py` | |
| `kernels/misc/` | `utils.py` (torchrun init, symmetric memory, graph capture), `bench_rotate.py` (the L2-rotate timing protocol), `nccl_env.py` (NCCL channel floor for every reference), `te_ub.py` (Transformer Engine Userbuffers: the AG ring-exchange and RS pipeline overlap behind the `te_ub` / `te` arms), `sync.py`, `ipc_attr_shim.c` (flux, see below) | |

Each `bench_baselines.py` docstring lists its arms and what they measure.

## Running

```bash
conda activate ml-env-cu13                      # B300: torch cu130, cutlass-dsl 4.7, quack deps
export PYTHONPATH=<this branch's checkout>      # the quack package; the benches add their own dirs
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
cd benchmarks/tp/kernels
torchrun --nproc_per_node=8 gemm_rs/bench_baselines.py quack --model llama3_70b --site out  --m 16384 --ref_check --rounds 3
torchrun --nproc_per_node=8 ag_gemm/bench_baselines.py quack --model llama3_70b --site qkv  --m 16384 --ref_check --rounds 3
torchrun --nproc_per_node=8 gemm_ar/bench_baselines.py quack_2shot --model llama3_70b --site out --m 64 --ref_check --rounds 3
# the Transformer Engine Userbuffers columns (TE's overlap wants one connection per device); RS: one --te-rs-sms per process, the cell = the best of 16 / 32 / 48
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=8 ag_gemm/bench_baselines.py te_ub --model llama3_70b --site qkv --m 16384 --ref_check --rounds 3
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=8 gemm_rs/bench_baselines.py te_ub --model llama3_70b --site out --m 16384 --te-rs-sms 32 --ref_check --rounds 3
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=8 gemm_rs/bench_mxfp8_rs.py te --m 16384 --n 8192 --k 8192 --te-rs-sms 32 --ref_check --rounds 3
torchrun --nproc_per_node=8 gemm_ar/bench_llama_ar.py quack --m 64 --n 8192 --k 8192 --ref_check   # one baseline per process: quack / cublas_nccl / flashinfer / quack_ar_norm_1shot
```

One arm per process; `--nproc_per_node` is the TP degree. `--ref_check` compares the arm and its reference
against an fp32 full-K GEMM. On torchrun 2.4–2.8 put `--` before the script, or torchrun abbrev-matches
`--m` to its own flags. H100 runs go through sbatch (pli / tridao_group), not salloc.

The scheduler-order ablation (GEMM+RS, column-major vs row-major tile order) is not a flag: the raster is a
trace-time constant in `quack/gemm_tvm_ffi_utils.py` (`make_scheduler_args` forces `RasterOrderOption.AlongN`,
the column-major production order, under epi_reduce). The row-major cells are the same `gemm_rs/bench_baselines.py
quack` run with that line set to `AlongM`, under `QUACK_CACHE_ENABLED=0` so the jit cache does not serve the
old cubin; revert afterwards.

## External baselines

Not in this tree; each arm locates its dependency by the path below and fails with a clear import error if
it is missing.

| arm | where |
|---|---|
| `cutlass` / `cutlass_1shot` / `cutlass_2shot` (DistGEMM examples, our `cutlass_1shot_batch` reuses the lamport example's GEMM) | `~/cutlass/examples/python/CuTeDSL/cute/blackwell/kernel/{distributed,dense_gemm}` |
| `82` (cutlass 82_blackwell_distributed_gemm, prebuilt C++ binary, self-manages TP) | `~/cutlass/build/examples/82_blackwell_distributed_gemm` |
| `tk` (ThunderKittens, per-world `_C_nd<TP>` builds) | `~/ThunderKittens/kernels/parallel/{gemm_rs,ag_gemm,gemm_ar}` |
| `flux` (H100 only; subprocess into `h100_lib_runner.py` under flux's env) | conda env `/scratch/gpfs/TRIDAO/zy4586/envs/flux` + `LD_PRELOAD` of `ipc_attr_shim.so` built from `kernels/misc/ipc_attr_shim.c` (torch's `from_blob` rejects flux's IPC peer buffers without it) |
| `triton_dist` (H100 only; same runner) | conda env `/scratch/gpfs/TRIDAO/zy4586/envs/triton-dist-main`; needs `LD_LIBRARY_PATH` to its `nvidia/nvshmem/lib` |
| `cublas+flashinfer`, `cutlass+flashinfer`, `flashinfer` (allreduce_fusion in `bench_llama_ar.py`) | `flashinfer` in the bench env |
| `te_ub` (`te` in the MXFP8 RS bench): Transformer Engine 2.19 Userbuffers | `transformer_engine` 2.19 in the bench env (core wheels; `transformer_engine_torch` built from the PyPI sdist against the env's torch, NCCL and cuDNN headers); run under `CUDA_DEVICE_MAX_CONNECTIONS=1` |
| `asynctp` | torch's symmetric-memory async-TP, in the bench env |
| multimem RS / AR legs | `nvshmem` python bindings in the bench env |
