"""Distributed helpers for the fused reduce-scatter / all-reduce GEMM path
(epi_reduce_mode): torch.distributed setup, symmetric / multicast tensor
allocation via torch symm_mem, the barrier flags the kernel's comm layer reads,
and the multimem intrinsic dispatch its reduce uses. These are the runtime
contract for calling GemmSm100(epi_reduce_mode=...), not benchmarking helpers.
"""

import math
import os
from typing import Optional, Tuple

import torch
import torch.distributed as dist

import cutlass
import cutlass.utils as utils
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cute.typing import Int32, Pointer
from cutlass.cutlass_dsl import T, dsl_user_op


def multimem_ld_reduce_128b(dtype):
    """128-bit multimem.ld_reduce(add) variant for dtype; all return 4x b32 (x, y, z, w)."""
    if dtype == cutlass.Float16:
        return utils.distributed.multimem_ld_reduce_8xf16
    if dtype == cutlass.Float32:
        return utils.distributed.multimem_ld_reduce_4xf32
    if dtype == cutlass.BFloat16:
        return utils.distributed.multimem_ld_reduce_8xbf16
    if dtype == cutlass.Float8E4M3FN:
        return utils.distributed.multimem_ld_reduce_16xe4m3
    if dtype == cutlass.Float8E5M2:
        return utils.distributed.multimem_ld_reduce_16xe5m2
    raise NotImplementedError(f"multimem_ld_reduce_128b: unsupported dtype {dtype}")


# Lamport (data-as-flag) primitives: the relaxed gpu-scope 16 B load that polls a
# mirror, and the plain 16 B store that re-arms a slot with the sentinel.


@dsl_user_op
def ld_relaxed_gpu_global_4xu32(
    ptr: Pointer, *, loc: Optional[ir.Location] = None, ip: Optional[ir.InsertionPoint] = None
) -> Tuple[Int32, Int32, Int32, Int32]:
    ptr_int = ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    regs = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(i32,i32,i32,i32)>"),
        [ptr_int],
        "ld.relaxed.gpu.global.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )
    x, y, z, w = [llvm.extractvalue(T.i32(), regs, [i]) for i in range(4)]
    return x, y, z, w


@dsl_user_op
def st_weak_global_4xu32(
    ptr: Pointer,
    x: Int32,
    y: Int32,
    z: Int32,
    w: Int32,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    ptr_int = ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    vals = [Int32(v).ir_value(loc=loc, ip=ip) for v in (x, y, z, w)]
    llvm.inline_asm(
        None,
        [ptr_int, *vals],
        "st.weak.global.v4.u32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


def init_distributed():
    """torchrun NCCL setup (idempotent), matching the cutlass distributed
    examples' init_distributed. Returns (global_rank, world_size, device)."""
    if not dist.is_initialized():
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        dist.init_process_group(
            backend="nccl", world_size=world_size, rank=global_rank, device_id=device
        )
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", global_rank))
    device = torch.device("cuda", local_rank)
    return global_rank, world_size, device


def clean_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def _wrap_device_ptr(ptr, shape, torch_dtype):
    """CUDA-typed non-owning view of a raw device pointer (e.g. a symm_mem
    multicast VA) on the current device; the symm_mem handle owns the memory."""
    dev = torch.device("cuda", torch.cuda.current_device())
    nbytes = math.prod(shape) * torch_dtype.itemsize
    storage = torch._C._construct_storage_from_data_pointer(ptr, dev, nbytes)
    return torch.empty(0, dtype=torch_dtype, device=dev).set_(storage).view(shape)


def make_symm_mem_tensor(shape, torch_dtype, permute):
    """Uninitialized symmetric tensor plus its multicast view via torch symm_mem.

    No host staging — an output or workspace buffer needs no initial contents.
    shape is in allocation (row-major) order and permute applies to both returns,
    so shape=(l, m, n) with permute=(1, 2, 0) gives kernel-order (m, n, l)
    n-major views. torch owns the allocation — no explicit free."""
    import torch.distributed._symmetric_memory as symm_mem

    base = symm_mem.empty(tuple(shape), dtype=torch_dtype, device="cuda")
    hdl = symm_mem.rendezvous(base, group=dist.group.WORLD)
    mc = _wrap_device_ptr(hdl.multicast_ptr, tuple(base.shape), base.dtype)
    return base.permute(permute), mc.permute(permute)


def make_symm_mem_flags(num_flags):
    """Zero-filled symmetric int32 flag array via torch symm_mem: this rank's
    local array (every wait reads it), its multicast view (broadcast signals),
    and per-peer P2P views (slot r writes rank r's array; slot rank aliases the
    local one). Each rank zero-fills its own copy before the (collective)
    rendezvous."""
    import torch.distributed._symmetric_memory as symm_mem

    flags = symm_mem.empty((num_flags,), dtype=torch.int32, device="cuda")
    flags.fill_(0)
    hdl = symm_mem.rendezvous(flags, group=dist.group.WORLD)
    flags_mc = _wrap_device_ptr(hdl.multicast_ptr, (num_flags,), flags.dtype)
    flags_per_peer = tuple(
        _wrap_device_ptr(hdl.buffer_ptrs[r], (num_flags,), flags.dtype)
        for r in range(dist.get_world_size())
    )
    return flags, flags_mc, flags_per_peer
