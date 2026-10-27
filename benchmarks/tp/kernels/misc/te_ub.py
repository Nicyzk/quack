"""Transformer Engine Userbuffers (comm + GEMM overlap) as a baseline arm for the AG+GEMM and GEMM+RS benches.

TE 2.19's `initialize_ub` builds the Userbuffers communicators once per process (torch-distributed bootstrap, no
MPI); `te.Linear` then overlaps its all-gather (column-parallel, `ub_overlap_ag`: the "ring_exchange" P2P ring on
`ag_sms` SMs) or reduce-scatter (row-parallel, `ub_overlap_rs`: the "pipeline" chunked GEMM on `rs_sms` SMs) with
the cuBLASLt GEMM. Both paths take MXFP8 operands under `te.fp8_autocast`; the forward captures into a CUDA graph.
The comm buffer is (M, width) bf16: width = K for the all-gather (the gathered input), N for the reduce-scatter (the
GEMM output); one geometry per process (TE keeps the communicators global)."""
import torch
import torch.distributed as dist

_STATE = {}
RS_SMS = 16  # the RS pipeline's comm SMs (TE's default); the benches set it from --te-rs-sms, one value per process


def init_ub(M, width, world, rs_sms=None, ag_sms=1):
    """initialize_ub once: qkv_fprop = the AG ring_exchange, proj_fprop = the RS pipeline, for the bf16 and FP8 modes."""
    from transformer_engine.pytorch.module.base import UserBufferQuantizationMode, initialize_ub

    rs_sms = RS_SMS if rs_sms is None else rs_sms
    key = (M, width, world, rs_sms, ag_sms)
    if _STATE.get("key") == key:
        return
    assert "key" not in _STATE, f"Userbuffers already initialized for {_STATE['key']}, asked for {key}"
    cfg = {
        "qkv_fprop": dict(method="ring_exchange", is_reduce_scatter=False, num_sm=ag_sms, cga_size=1, set_sm_margin=False,
                          num_splits=world, aggregate=False, atomic_gemm=False, use_ce=True, fp8_buf=False),
        "proj_fprop": dict(method="pipeline", is_reduce_scatter=True, num_sm=rs_sms, cga_size=2, set_sm_margin=True,
                           num_splits=4, aggregate=False, atomic_gemm=False, use_ce=True, fp8_buf=False),
    }
    initialize_ub([M, width], world, dtype=torch.bfloat16, ub_cfgs=[cfg, cfg],
                  quantization_modes=[UserBufferQuantizationMode.NONE, UserBufferQuantizationMode.FP8],
                  bootstrap_backend="nccl")
    _STATE["key"] = key


def ag_linear(K, n_local, world, weight):
    """Column-parallel te.Linear with the Userbuffers all-gather: x shard (M/world, K) -> (M, n_local); weight (n_local, K)."""
    import transformer_engine.pytorch as te

    lin = te.Linear(K, n_local * world, bias=False, params_dtype=torch.bfloat16, device="cuda", parallel_mode="column",
                    tp_group=dist.group.WORLD, tp_size=world, sequence_parallel=True, ub_overlap_ag=True, ub_name="qkv")
    lin.weight.data = weight.contiguous()
    return lin


def rs_linear(K, N, world, weight):
    """Row-parallel te.Linear with the Userbuffers reduce-scatter: x (M, K/world) -> (M/world, N); weight (N, K/world)."""
    import transformer_engine.pytorch as te

    lin = te.Linear(K, N, bias=False, params_dtype=torch.bfloat16, device="cuda", parallel_mode="row",
                    tp_group=dist.group.WORLD, tp_size=world, sequence_parallel=True, ub_overlap_rs=True, ub_name="proj")
    lin.weight.data = weight.contiguous()
    return lin
