"""Fused-communication (epi_reduce_mode) GEMM epilogue pieces; nothing here runs on
its own — gemm_sm100's kernel binds each piece into the shared GemmBase machinery.
Sections below: host contract / reducer tile scheduler / cross-launch exit barrier /
multimem reduce + store. The split-rank protocol itself (flag contract, producer
partial commit, reducer spin + epoch counters) lives in gemm_base's
epilogue_split_rank / split_rank_partial_commit, the cross-rank siblings of the
split-K pair.

Dataflow: both warp groups run epilogue_split_rank. The producer's finalize action
is split_rank_partial_commit — d_dtype fragment stripes into the symmetric
workspace (_frag_stripe_op, split-K's addressing), then the tile signal. The
reducer walks producer tiles intersecting its slab and runs epilogue() between two
bound functions: stripe multimem ld_reduce in (same addressing, mc view),
EVT/C_load/aux TileStores unchanged in the middle, real-address store out
(reduce_scatter: this rank's D slab; all_reduce: 16B multimem_st chunks through
the mc partition). Pipelines: epi_pipeline stages C for the reducer;
epi_reduce_store_pipeline backs the reducer's aux stores."""

from typing import Callable, NamedTuple, Optional

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import Int32, const_expr

from quack.cute_dsl_utils import mlir_namedtuple
from quack.fast_math import FastDivmod


# ---- host contract ----


@mlir_namedtuple
class EpiReduceArguments(NamedTuple):
    """Comm-side tensors for epi_reduce_mode. workspace/tile_flags/counters share one
    problem shape's cluster-rounded (E, ntile_m, ntile_n, L) tile domain; sync_barrier
    is per resident epi-reduce CTA slot, with num_sms allocation remaining a safe
    upper bound."""

    mD_mc: Optional[cute.Tensor] = None  # multicast view of symmetric D
    mD_peers: Optional[tuple] = None  # per-rank views of symmetric D
    # d_dtype partial stripes, (cta_M * cta_N, ntile_m, ntile_n, L) symmetric
    workspace: Optional[cute.Tensor] = None
    workspace_mc: Optional[cute.Tensor] = None
    # producer -> consumer, one flag per workspace tile slot
    tile_flags: Optional[cute.Tensor] = None
    tile_flags_mc: Optional[cute.Tensor] = None
    sync_barrier: Optional[cute.Tensor] = None  # exit barrier, one slot per resident CTA
    sync_barrier_mc: Optional[cute.Tensor] = None
    # consumer-private epoch bases, indexed like tile_flags
    consumer_counters: Optional[cute.Tensor] = None


# ---- reducer tile scheduler ----
# The reducer warps and the epi-load warp (C staging) walk the same slab-order
# static persistent schedule.


@mlir_namedtuple
class EpiReduceSchedulerParams(NamedTuple):
    tile_sched_params: utils.PersistentTileSchedulerParams
    num_persistent_clusters: Int32

    @staticmethod
    def create(problem_shape_ntile_mnl, cluster_shape_mnk, max_active_clusters):
        assert cluster_shape_mnk[2] == 1, (
            "EpiReduceSchedulerParams assumes cluster_shape_mnk[2] == 1"
        )
        tile_sched_params = utils.PersistentTileSchedulerParams(
            problem_shape_ntile_mnl, cluster_shape_mnk
        )
        num_persistent_clusters = cutlass.min(
            cute.size(tile_sched_params.problem_layout_ncluster_mnl),
            max_active_clusters,
        )
        return EpiReduceSchedulerParams(tile_sched_params, num_persistent_clusters)


@cute.jit
def clc_block_to_static_scheduler_coord(cluster_shape_mn):
    """CLC launch grid is grid=(cl_m * ncl_mn, cl_n, batch): block_idx is already the
    hierarchical coord, so peel the cluster-local part with FastDivmod and
    linearize the cluster coord through its layout.
    Returns (linear persistent cluster id, CTA m in cluster, CTA n in cluster)."""
    bidx, bidy, bidz = cute.arch.block_idx()
    gdx, gdy, gdz = cute.arch.grid_dim()
    cl_m, cl_n = cluster_shape_mn
    cl_m_fdd, cl_n_fdd = FastDivmod(cl_m), FastDivmod(cl_n)
    c_m, cta_m = divmod(bidx, cl_m_fdd)
    c_n, cta_n = divmod(bidy, cl_n_fdd)
    grid_m, _ = divmod(gdx, cl_m_fdd)
    grid_n, _ = divmod(gdy, cl_n_fdd)
    cluster_layout = cute.make_layout((grid_m, grid_n, gdz))
    return cluster_layout((c_m, c_n, bidz)), cta_m, cta_n


@cute.jit
def make_epi_reduce_tile_scheduler(params: EpiReduceSchedulerParams):
    tile_sched_params = params.tile_sched_params
    cluster_shape_mn = tile_sched_params.cluster_shape_mn
    cl_m, cl_n = cluster_shape_mn
    cluster_id, cta_m, cta_n = clc_block_to_static_scheduler_coord(cluster_shape_mn)
    return utils.StaticPersistentTileScheduler.create(
        tile_sched_params,
        (cta_m, cta_n, cluster_id),
        (cl_m, cl_n, params.num_persistent_clusters),
    )


# ---- cross-launch exit sync barrier: one slot per resident CTA ----


@cute.jit
def epi_reduce_exit_slot(params: EpiReduceSchedulerParams) -> Int32:
    # Keep the block_idx-derived coords inside one jit: returning the tuple to the
    # kernel and re-consuming it in a second jit mis-materializes the slot -> OOB write.
    cluster_shape_mn = params.tile_sched_params.cluster_shape_mn
    cluster_id, cta_m, cta_n = clc_block_to_static_scheduler_coord(cluster_shape_mn)
    slot_layout = cute.make_layout((*cluster_shape_mn, params.num_persistent_clusters))
    return slot_layout((cta_m, cta_n, cluster_id))


# ---- stripe reduce / commit ----
# The reducer's two data-movement callbacks, both in the PRODUCER's r2s fragment
# geometry (the stripe contract).


@cute.jit
def stripe_reduce_subtile(
    stripe_op: Callable,
    ws_mc_ptr: cute.Pointer,
    num_threads: cutlass.Constexpr[int],
    tidx: Int32,
    tRS_cD: cute.Tensor,
    row_lo: Int32,
    row_hi: Int32,
    col_limit: Int32,
    subtile_layout: cute.Layout,
    tRS_rD: cute.Tensor,
    epi_coord: cute.Coord,
) -> None:
    """Reduce this subtile's workspace stripes across all ranks into tRS_rD;
    passed to epilogue() as load_acc_subtile by the reducer warps. stripe_op is
    the bound _frag_stripe_op — the identical addressing the producers stored
    with, read through the workspace's mc view. Lanes outside [row_lo, row_hi)
    (foreign slab rows / M tail) or past col_limit are zeroed: keeps visit
    reductions exact, and foreign rows belong to their owning rank."""
    frag_elems = cute.size(tRS_rD)
    epi_idx = subtile_layout(epi_coord)
    tmp = cute.make_rmem_tensor(tRS_rD.layout.shape, ws_mc_ptr.dtype)
    stripe_op(
        "multimem_ld_reduce", ws_mc_ptr + epi_idx * num_threads * frag_elems,
        tidx, num_threads, tmp,
    )
    tRS_rD.store(tmp.load().to(tRS_rD.element_type))
    tRS_cD_cur = tRS_cD[None, None, None, epi_coord[0], epi_coord[1]]
    for i in cutlass.range_constexpr(frag_elems):
        crd = tRS_cD_cur[i]
        if crd[0] < row_lo or crd[0] >= row_hi or crd[1] >= col_limit:
            tRS_rD[i] = 0.0


@cute.jit
def commit_frag_subtile(
    tRS_gD: cute.Tensor,
    tRS_gD_mc: Optional[cute.Tensor],
    tRS_cD: cute.Tensor,
    row_lo: Int32,
    row_hi: Int32,
    col_limit: Int32,
    full_tile: cutlass.Boolean,
    all_reduce: cutlass.Constexpr[bool],
    tRS_rD: cute.Tensor,
    epi_coord: cute.Coord,
) -> None:
    """Store the reduced, post-EVT subtile at real (m, n) — reduce_scatter: plain
    generic stores to this rank's D slab (single-writer-local now that partials
    live in the workspace); all_reduce: 16B multimem_st through the mc partition
    to every rank. Passed to epilogue() as commit_D by the reducer warps. Owns
    d_dtype conversion and slab/edge predication.

    The AR chunks lean on the tmem-load atom handing each thread contiguous
    n-runs: coalesce fragment and gmem partition (order-preserving), certify 16B
    with max_common_vector, then one multimem_st per chunk. Chunks never straddle
    rows (cta_n % vec == 0) and never straddle N (host asserts N % vec == 0 for
    AR), so the first element's coords predicate the whole chunk."""
    tRS_gD_cur = tRS_gD[None, None, None, epi_coord[0], epi_coord[1]]
    if const_expr(tRS_rD.element_type != tRS_gD.element_type):
        tmp_out = cute.make_rmem_tensor(tRS_rD.layout.shape, tRS_gD.element_type)
        tmp_out.store(tRS_rD.load().to(tRS_gD.element_type))
    else:
        tmp_out = tRS_rD
    tRS_cD_cur = tRS_cD[None, None, None, epi_coord[0], epi_coord[1]]
    if const_expr(not all_reduce):
        if full_tile:
            cute.autovec_copy(tmp_out, tRS_gD_cur)
        else:
            for i in cutlass.range_constexpr(cute.size(tmp_out)):
                crd = tRS_cD_cur[i]
                if crd[0] >= row_lo and crd[0] < row_hi and crd[1] < col_limit:
                    tRS_gD_cur[i] = tmp_out[i]
    else:
        gD_mc_cur = tRS_gD_mc[None, None, None, epi_coord[0], epi_coord[1]]
        gD_mc_f = cute.coalesce(gD_mc_cur)
        vec = const_expr(
            min(
                cute.max_common_vector(gD_mc_f, cute.coalesce(tmp_out)),
                128 // tRS_gD.element_type.width,
            )
        )
        assert vec == 128 // tRS_gD.element_type.width, (
            "all_reduce commit needs 16B fragment n-runs (tmem-load atom too narrow)"
        )
        gD_mc_v = cute.zipped_divide(gD_mc_f, vec)
        for v in cutlass.range_constexpr(cute.size(tmp_out) // vec):
            crd = tRS_cD_cur[v * vec]
            if crd[0] >= row_lo and crd[0] < row_hi and crd[1] < col_limit:
                chunk = cute.make_tensor(tmp_out.iterator + v * vec, cute.make_layout(vec))
                v32 = cute.recast_tensor(chunk, cutlass.Int32)
                utils.distributed.multimem_st_4xb32(
                    gD_mc_v[None, v].iterator,
                    v32[0].ir_value(),
                    v32[1].ir_value(),
                    v32[2].ir_value(),
                    v32[3].ir_value(),
                )
