# MoE dispatch fused into the GEMM through gemm_sm100.py's comm object: 4 transfer warps per CTA run
# dispatch/pull.py's pull_rows over the device walk (dispatch/walk.py), landing each row once in the pool the GEMM's
# gather-A reads and bumping the scheduler's flags[expert, block]. The gate is data: the scheduler params carry
# quack ep's DispatchSchedulerArguments, so the AB-load warp's wait_tile blocks on the block's flag and the
# scheduler walks the blocks in landing order. Usage: GemmSm100(..., comm=DispatchPull(pull_bytes=...)), launch
# with comm_args=DispatchArguments(walk.args, pool).
import sys
from pathlib import Path
from typing import NamedTuple, Optional

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, const_expr
from quack.compile_utils import make_fake_tensor
from quack.cute_dsl_utils import mlir_namedtuple

sys.path.insert(0, str(Path(__file__).parent.parent / "moe"))
from dispatch.pull import pull_chunk_bytes, pull_rows, stage_bytes  # noqa: E402
from dispatch.walk import WalkArguments, fake_walk_arguments  # noqa: E402
from gemm_sm100 import Comm  # noqa: E402

TRANSFER_WARPS = 4


@mlir_namedtuple
class DispatchArguments(NamedTuple):
    """The transfer warps' bundle (the GEMM's comm_args)."""

    walk: WalkArguments  # walk.make_dispatch_walk / make_prologue's walk.args
    pool: cute.Tensor  # the rows the pull writes: the gather-A pool (= mA) or the expert-packed rows (scatter)
    sf_pool: Optional[cute.Tensor] = None  # block-scaled: the SF atoms the pulled scales land in (the GEMM's SFB)


def fake_dispatch_arguments(pool_dtype, sf=False):
    """DispatchArguments of fake tensors for cute.compile."""
    pool = make_fake_tensor(pool_dtype, (cute.sym_int(), cute.sym_int()), leading_dim=1, divisibility=8)
    sf_pool = make_fake_tensor(cutlass.Uint8, (cute.sym_int(),), leading_dim=0, divisibility=16) if sf else None
    return DispatchArguments(fake_walk_arguments(sf), pool, sf_pool)


class DispatchPull(Comm):
    """The dispatch pull as the GEMM's comm warp group. pull_bytes = one row's bytes; scatter = the row lands in
    every reader's expert-major row (walk.dst_row) instead of the pool row; sf_words = 4-byte scale words per row
    for block-scaled rows (K / (4 sf_vec_size)), 0 = none."""

    num_warps = TRANSFER_WARPS
    regs = (232, 120, 40)  # 4 warp groups share 512: epilogue 232, the GEMM's others 120 each, the transfer loop 40
    flag_batch = 1  # publish each row as it lands: ~6 rows per warp, batching only delays every block's readiness

    def __init__(self, pull_bytes: int, scatter: bool = False, sf_words: int = 0):
        self.pull_bytes = pull_bytes
        self.pull_chunk = pull_chunk_bytes(pull_bytes, cutlass.Uint8)  # a chunk of the row per stage: the GEMM keeps its A/B stages
        self.scatter = scatter
        self.sf_words = sf_words

    def smem_bytes(self, gemm):
        return stage_bytes(self.pull_chunk, cutlass.Uint8, TRANSFER_WARPS)

    def smem_struct(self, gemm):
        class PullStorage:
            stage: cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, TRANSFER_WARPS * self.pull_chunk], 128]
            mbar: cute.struct.Align[cute.struct.MemRange[cutlass.Int64, TRANSFER_WARPS], 8]

        return cute.struct(PullStorage)

    @cute.jit
    def comm_warps(
        self, gemm, comm_args, storage, tile_sched_params, TileSchedulerCls, epilogue_params, epi_smem_tensors,
        varlen_manager, warp_idx: Int32, tidx: Int32,
    ):
        """The transfer warps: this CTA's share of the walk, rows into the pool, flags bumped as rows land."""
        chunk = self.pull_chunk
        stage = storage.comm.stage.get_tensor(cute.make_layout((TRANSFER_WARPS, chunk), stride=(chunk, 1)))
        transfer_warp = warp_idx - gemm.comm_warp_ids[0]
        mbar = storage.comm.mbar.data_ptr() + transfer_warp
        if tidx % 32 == Int32(0):
            cute.arch.mbarrier_init(mbar, 1)
        cute.arch.sync_warp()
        bid, gd = cute.arch.block_idx(), cute.arch.grid_dim()
        cta = bid[0] + gd[0] * (bid[1] + gd[1] * bid[2])
        num_ctas = gd[0] * gd[1] * gd[2]
        sf_pool_base = Int64(0)
        if const_expr(self.sf_words > 0):
            sf_pool_base = comm_args.sf_pool.iterator.toint()
        pull_rows(
            comm_args.pool.iterator.toint(),
            Int32(self.pull_bytes),
            comm_args.walk,
            tile_sched_params.dispatch.flags,
            stage[transfer_warp, None],
            mbar,
            cta * Int32(TRANSFER_WARPS) + transfer_warp,
            num_ctas * Int32(TRANSFER_WARPS),
            tidx % 32,
            self.flag_batch,
            self.scatter,
            self.sf_words,
            sf_pool_base,
        )
