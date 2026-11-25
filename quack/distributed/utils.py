# Copyright (c) 2026, QuACK team.
"""Host helpers shared by the distributed operators' runners: symmetric allocations, their multicast views
and peer addresses, and the dtype rules the runners size their buffers by."""

import math

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from quack.cute_dsl_utils import torch2cute_dtype_map


def symm_alloc(shape, dtype, group=None, zero=False):
    """(local, handle) of a symmetric allocation on the current device over `group` (None = the world
    group). The handle pins the mapping: the caller keeps it as long as peers address the buffer."""
    local = symm_mem.empty(tuple(shape), dtype=dtype, device=torch.cuda.current_device())
    if zero:
        local.zero_()
    return local, symm_mem.rendezvous(local, group=dist.group.WORLD if group is None else group)


def multicast_view(handle, shape, dtype):
    """Non-owning tensor over the allocation's multicast address: a store lands in every rank's copy."""
    dev = torch.device("cuda", torch.cuda.current_device())
    storage = torch._C._construct_storage_from_data_pointer(
        handle.multicast_ptr, dev, math.prod(shape) * dtype.itemsize
    )
    return torch.empty(0, dtype=dtype, device=dev).set_(storage).view(tuple(shape))


def peer_bases(handle):
    """(world_size,) int64: every rank's address of the allocation."""
    return torch.tensor(handle.buffer_ptrs, dtype=torch.int64, device=torch.cuda.current_device())


def torch_dtype_of(cute_dtype):
    """The torch dtype of a cute dtype (torch2cute_dtype_map, reversed)."""
    return next(t for t, c in torch2cute_dtype_map.items() if c is cute_dtype)


def d_cols(n, dtype):
    """D's stored columns for n values: fp4 packs two values a byte."""
    return n // 2 if dtype == torch.float4_e2m1fn_x2 else n
