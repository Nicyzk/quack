"""Fused-communication (epi_reduce_mode) GEMM epilogue pieces; nothing here runs on
its own — gemm_sm100's kernel binds each piece into the shared GemmBase machinery.
Sections below: host contract / reducer tile scheduler / cross-launch exit barrier /
workspace tile ops (partial store / cross-rank reduce / D commit). The split-rank
protocol itself (flag contract, producer partial commit, reducer spin + epoch
counters) lives in gemm_base's epilogue_split_rank / split_rank_partial_commit, the
cross-rank siblings of the split-K pair.

Dataflow: both warp groups run epilogue_split_rank. The producer's finalize action
is split_rank_partial_commit — the partial D tile in d_dtype at its real (m, n) in
the flat symmetric workspace (frag_tile_op), then the tile signal. The reducer
walks its own tiles (reduce_scatter: slab-anchored; all_reduce: producer tiles)
and runs epilogue() between two bound functions: multimem ld_reduce of its tile
through the workspace mc view in (reduce_frag_subtile), EVT/C_load/aux TileStores
unchanged in the middle, real-address store out (reduce_scatter: this rank's D
slab; all_reduce: 16B multimem_st chunks through the mc partition). Pipelines:
epi_pipeline stages C for the reducer; epi_reduce_store_pipeline backs the
reducer's aux stores."""

from typing import Literal, NamedTuple, Optional

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import Int32, const_expr

from quack.cute_dsl_utils import mlir_namedtuple
from quack.dist_utils import multimem_ld_reduce_128b
from quack.fast_math import FastDivmod


# ---- host contract ----


@mlir_namedtuple
class EpiReduceArguments(NamedTuple):
    """Comm-side tensors for epi_reduce_mode. workspace is a flat (M_pad, N_pad, L)
    d_dtype tensor addressed by real coordinates, padded so any cta tile anchored in
    [0, M) x [0, N) is fully in bounds; tile_flags live on the cluster-rounded
    (ntile_m, ntile_n, L) producer tile grid; sync_barrier is per resident
    epi-reduce CTA slot, with num_sms allocation remaining a safe upper bound."""

    mD_mc: Optional[cute.Tensor] = None  # multicast view of symmetric D
    mD_peers: Optional[tuple] = None  # per-rank views of symmetric D
    # d_dtype partial D, flat (M_pad, N_pad, L) symmetric
    workspace: Optional[cute.Tensor] = None
    workspace_mc: Optional[cute.Tensor] = None
    # producer -> consumer, one flag per (m, n, batch) producer tile
    tile_flags: Optional[cute.Tensor] = None
    tile_flags_mc: Optional[cute.Tensor] = None
    sync_barrier: Optional[cute.Tensor] = None  # exit barrier, one slot per resident CTA
    sync_barrier_mc: Optional[cute.Tensor] = None
    # consumer-private epoch bases, indexed by the reducer's own tile coord
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


# ---- workspace tile ops: partial store / cross-rank reduce / D commit ----
# Workspaces are flat (M_pad, N_pad, L), addressed by real coordinates: padding
# keeps every cta tile anchored inside the problem fully in bounds (no predication
# on workspace access), and vectors run along N so arbitrary M anchors keep
# alignment. frag_tile_op is THE workspace access op — split-K's commit/fold bind
# its local ops (gemm_base), split-rank's producer and reducer share it with only
# the view differing (producer: its (m, n) tile; reducer: slab-anchored for
# reduce_scatter, through the mc view).


@cute.jit
def frag_tile_op(
    op: cutlass.Constexpr[Literal["store", "red_add", "load_add", "multimem_ld_reduce"]],
    tRS_gWs: cute.Tensor,
    tRS_rF: cute.Tensor,
    epi_coord: cute.Coord,
) -> None:
    """One epi subtile of fragment tRS_rF vs the partitioned workspace view
    tRS_gWs. The caller picks the workspace and the view; this op only moves data:

    - "store":    plain store of the fragment (partial commits, split 0's init)
    - "red_add":  one-way L2 red.add, no read-back (later splits' commits)
    - "load_add": read the tile and add it INTO the fragment (split-K fold)
    - "multimem_ld_reduce": read through a multicast view, reducing across ranks
      into the fragment (split-rank fold; tRS_gWs must be the mc view)

    Chunks are max_common_vector-sized, 16B-capped; the local ops take any width
    down to scalar (SM90 wgmma f32 runs are 8B), multimem is hardware-fixed at 16B.
    """
    gWs_cur = tRS_gWs[None, None, None, epi_coord[0], epi_coord[1]]
    gWs_f = cute.coalesce(gWs_cur)
    vec = const_expr(
        min(cute.max_common_vector(gWs_f, cute.coalesce(tRS_rF)), 128 // tRS_gWs.element_type.width)
    )
    if const_expr(op == "multimem_ld_reduce"):
        assert vec == 128 // tRS_gWs.element_type.width, (
            "multimem_ld_reduce needs 16B fragment n-runs (tmem-load atom too narrow)"
        )
    gWs_v = cute.zipped_divide(gWs_f, (vec,))
    if const_expr(op == "load_add"):
        frag = cute.make_rmem_tensor(tRS_rF.layout.shape, tRS_gWs.element_type)
    else:
        frag = tRS_rF
    for v in cutlass.range_constexpr(cute.size(tRS_rF) // vec):
        chunk = cute.make_tensor(frag.iterator + v * vec, cute.make_layout(vec))
        if const_expr(op == "store"):
            cute.autovec_copy(chunk, gWs_v[None, v])
        elif const_expr(op == "red_add"):
            if const_expr(vec == 1):
                # nvvm.atomicrmw rejects 1-wide vectors; pass the scalar.
                cute.arch.atomic_add(gWs_v[None, v].iterator, chunk[0])
            else:
                cute.arch.atomic_add(gWs_v[None, v].iterator, chunk.load())
        elif const_expr(op == "load_add"):
            cute.autovec_copy(gWs_v[None, v], chunk)
        else:
            x, y, z, w = multimem_ld_reduce_128b(tRS_rF.element_type)(gWs_v[None, v].iterator)
            chunk_i32 = cute.recast_tensor(chunk, cutlass.Int32)
            chunk_i32[0], chunk_i32[1], chunk_i32[2], chunk_i32[3] = x, y, z, w
    if const_expr(op == "load_add"):
        tRS_rF.store(tRS_rF.load() + frag.load())


@cute.jit
def reduce_frag_subtile(
    tRS_gWs_mc: cute.Tensor,
    tRS_cD: cute.Tensor,
    row_lo: Int32,
    row_hi: Int32,
    col_limit: Int32,
    tRS_rD: cute.Tensor,
    epi_coord: cute.Coord,
) -> None:
    """Reduce this subtile's workspace tile across all ranks into tRS_rD
    (frag_tile_op through the mc view); passed to epilogue() as load_acc_subtile
    by the reducer warps. Lanes outside [row_lo, row_hi) (foreign slab rows / M
    tail) or past col_limit are zeroed: keeps visit reductions exact."""
    tmp = cute.make_rmem_tensor(tRS_rD.layout.shape, tRS_gWs_mc.element_type)
    frag_tile_op("multimem_ld_reduce", tRS_gWs_mc, tmp, epi_coord)
    tRS_rD.store(tmp.load().to(tRS_rD.element_type))
    tRS_cD_cur = tRS_cD[None, None, None, epi_coord[0], epi_coord[1]]
    for i in cutlass.range_constexpr(cute.size(tRS_rD)):
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
        gD_mc_v = cute.zipped_divide(gD_mc_f, (vec,))
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
