# Copyright (c) 2026, QuACK team.
"""Distributed (multi-GPU overlapped) GEMMs.

Each operator is a DistOp the GEMM takes as ``dist_op=`` with its per-call bundle ``dist_op_args=``; a Runner
owns the operator's buffers and returns the bundle from ``prepare()``.

Package layout:
- all_gather.py: the AllGather operator (its bundle AllGatherArguments), AllGatherRunner (ce-push transport,
  prepare() context; the module docstring is the full design record) and BlockScaledAllGatherRunner (adds a
  scale-factor lane under the same arrival flags, for blockscaled GEMMs)
- reduce_scatter.py: ReduceScatter and AllReduce (two-shot), their bundles and runners
- all_reduce_lamport.py: AllReduceLamport (one-shot all-reduce), its bundle and runner
- dispatch.py, dispatch_walk.py: DispatchPull (MoE dispatch into a grouped GEMM's token operand), its bundle,
  DispatchRunner and the device routing kernel its prepare() launches
- combine.py: CombineStore (MoE combine out of a grouped GEMM's epilogue), its bundle and runner
- reduce_over_k.py, reduce_over_k_epilogue.py: the combine's sum over a token's K slots (a single-GPU kernel)
  and its fused epilogues
- ptx.py: the inline-asm gmem primitives the operators share; utils.py: the runners' host helpers
"""

from quack.distributed.all_gather import (
    AllGather,
    AllGatherArguments,
    AllGatherRunner,
    BlockScaledAllGatherRunner,
)
from quack.distributed.all_reduce_lamport import (
    AllReduceLamport,
    AllReduceOneShotRunner,
    LamportArguments,
)
from quack.distributed.combine import CombineArguments, CombineRunner, CombineStore
from quack.distributed.dispatch import DispatchArguments, DispatchPull, DispatchRunner
from quack.distributed.reduce_over_k import reduce_over_k
from quack.distributed.reduce_scatter import (
    AllReduce,
    AllReduceArguments,
    AllReduceRunner,
    ReduceScatter,
    ReduceScatterArguments,
    ReduceScatterRunner,
)

__all__ = [
    "AllGather",
    "AllGatherArguments",
    "AllGatherRunner",
    "AllReduce",
    "AllReduceArguments",
    "AllReduceLamport",
    "AllReduceOneShotRunner",
    "AllReduceRunner",
    "BlockScaledAllGatherRunner",
    "CombineArguments",
    "CombineRunner",
    "CombineStore",
    "DispatchArguments",
    "DispatchPull",
    "DispatchRunner",
    "LamportArguments",
    "ReduceScatter",
    "ReduceScatterArguments",
    "ReduceScatterRunner",
    "reduce_over_k",
]
