"""
Cross-GPU barrier kernel: multicast release-increment, spin until the local
flag reaches world_size (>= wait), then atomically consume world_size tokens.

Reuse-safe with a single instance: release needs world_size*phase cumulative
tokens, a phase p+1 token cannot exist before every rank issued phase p, and
early tokens bank in the counter. (The old CAS==world_size reset wedged when
a fast rank's next-phase +1 landed before a slow rank's CAS fired.)

No acquire fence: the rendezvous orders execution, and cross-GPU visibility
comes from the grid boundary (producer kernel commits to L2 before the
barrier) plus fresh uncached peer reads -- verified by the race tests.
"""

import torch

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.runtime import from_dlpack

import nvshmem.core


class BarrierAllKernel:
    @cute.jit
    def __call__(self, flag, flag_mc, world_size: cutlass.Constexpr, stream):
        self.kernel(flag, flag_mc, world_size).launch(grid=[1, 1, 1], block=[32, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, flag: cute.Tensor, flag_mc: cute.Tensor, world_size: cutlass.Constexpr):
        tidx = cute.arch.thread_idx()[0]
        if tidx == 0:
            utils.distributed.multimem_red_add1(
                flag_mc.iterator, scope="sys", order="release"
            )
            utils.distributed.spin_lock_ld_lt_relaxed_wait(
                flag.iterator,
                expected_val=world_size,
                scope="sys",
            )
            cute.arch.atomic_add(
                flag.iterator.llvm_ptr, cutlass.Int32(-world_size), sem="relaxed", scope="sys"
            )


def make_barrier(world_size):
    """Allocate a flag, compile the kernel; returns (barrier_fn, free_fn)."""
    flag = nvshmem.core.tensor((1,), dtype=torch.int32)
    flag.fill_(0)
    flag_mc = nvshmem.core.get_multicast_tensor(nvshmem.core.Teams.TEAM_NODE, flag)
    fd = from_dlpack(flag, enable_tvm_ffi=True)
    fmd = from_dlpack(flag_mc, enable_tvm_ffi=True)
    # TVM-FFI env stream (quack style): launches on the current stream, captured into the graph.
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    fn = cute.compile(BarrierAllKernel(), fd, fmd, world_size, stream, options="--enable-tvm-ffi")

    def free():
        nvshmem.core.free_tensor(flag_mc)
        nvshmem.core.free_tensor(flag)

    return lambda: fn(flag, flag_mc), free
