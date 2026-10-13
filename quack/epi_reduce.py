"""Fused-communication (epi_reduce_mode) GEMM epilogue pieces; nothing here runs on
its own — gemm_sm100's kernel binds each piece into the shared GemmBase machinery.
Sections below: host contract / cross-launch exit barrier / flag map / multimem reduce +
store / lamport one-shot all_reduce.

Dataflow (cutlass-aligned, no separate reducer schedule): the producer warp group
runs epilogue_split_rank — split_rank_partial_commit stores register-direct
workspace-dtype partials into the padded symmetric workspace, then the tile
signal. The comm warps replay their own CTA's tile sequence
(gemm_sm100.epi_reduce_comm): per tile they wait the flag, multimem-ld_reduce a
1/world slice of the workspace, and store it — multimem_st broadcast into
symmetric D (all_reduce: every rank takes slice ``rank`` of its own tile) or a
plain local store into the slab-shaped D (reduce_scatter: cutlass's owner remap —
the CTA at (m, n) reduces stripe m // tiles_per_rank of the owned tile
m % tiles_per_rank + rank * tiles_per_rank)."""

import math
from typing import NamedTuple, Optional

import torch

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import Int32, const_expr
from cutlass.cutlass_dsl import any_, while_generate, yield_out

from quack.cute_dsl_utils import mlir_namedtuple
from quack.dist_utils import (
    ld_relaxed_gpu_global_4xu32,
    multimem_ld_reduce_128b,
    st_weak_global_4xu32,
)


# ---- host contract ----


@mlir_namedtuple
class EpiReduceArguments(NamedTuple):
    """Comm-side tensors for epi_reduce_mode. The flag array is sized to one
    problem shape (the tile->slot mapping) plus a num_sms exit-slot tail (one per
    resident CTA, cross-launch exit barrier)."""

    mD_mc: Optional[cute.Tensor] = None  # multicast view of symmetric D — all_reduce only
    # Partials workspace (any multimem-reducible dtype, default d_dtype),
    # (M_pad, N_pad, L) at real (m, n) coords: this rank's view (producer store
    # target) and its multicast view (reducer ld_reduce).
    workspace: Optional[cute.Tensor] = None
    workspace_mc: Optional[cute.Tensor] = None
    # producer -> consumer flags, ceil(M/cta_M) * ceil(N/cta_N) * L entries plus
    # the exit-slot tail. Every wait/consume reads the local array; signals write
    # through the multicast view (all_reduce, exit barrier) or the owner's
    # per-peer view (reduce_scatter, whose tiles have a single consuming rank).
    tile_flags: Optional[cute.Tensor] = None
    tile_flags_mc: Optional[cute.Tensor] = None
    tile_flags_per_peer: Optional[tuple] = None
    # all_reduce_lamport: (1,) int32 launch counter the host bumps before every
    # launch (slot % 3 = ping, +1 = pong). The workspace then holds 3 * world * L
    # (m, n) mirrors: workspace_mc is the TMA multicast store target, workspace the
    # local mirrors the comm warps poll. No tile flags.
    slot: Optional[cute.Tensor] = None


def epi_reduce_workspace_shape(m, n, cta_m, cta_n):
    """Padded workspace extents; the pad keeps every workspace access
    in-allocation with no predication, and nothing consumes it. M carries one
    extra cta_m block: fully OOB phantom cluster CTAs (2-CTA pairing is along M)
    store into the dead last block. Comm reads are tile-aligned (full tiles
    enforced in validate), so no other pad is needed."""
    m_pad = ((m + cta_m - 1) // cta_m + 1) * cta_m
    n_pad = (n + cta_n - 1) // cta_n * cta_n
    return m_pad, n_pad


def pick_num_comm_warps(cta_m: int, tile_N: int, ws_width: int, tile_split: int) -> Optional[int]:
    """cutlass's _pick_num_comm_warp_for_128b: the largest W in 4..1 whose 32*W threads
    split the (cta_m / tile_split, tile_N) slice evenly into whole 128 b atoms (one or
    more per thread); tile_split = world under the flag protocols, 1 under lamport.
    None when no count does."""
    slab = (cta_m // tile_split) * tile_N
    atom = 128 // ws_width
    for w in (4, 3, 2, 1):
        if slab % (32 * w) == 0 and (slab // (32 * w)) % atom == 0:
            return w
    return None


def validate_epi_reduce_args(
    epi_reduce_args, D, mode, m, n, l, tile_M, tile_N, cluster_M, num_ranks
):
    """Guard the comm bundle against this call's geometry — the epi_reduce sibling of
    validate_ag_geometry, called per launch from every frontend (warm plan-cache hits
    skip trace-time asserts, so the host is the only per-call check). Everything here
    is a mismatch the kernel can only corrupt or hang on: multimem vector width,
    kernel-order comm views, and flag/counter capacities (an under-sized flag array
    is a silent OOB multimem write). m is the full GEMM M (from A): D carries only
    the slab under reduce_scatter."""
    era = epi_reduce_args
    if D is None:
        raise ValueError("epi_reduce_mode requires D (the output tensor)")
    if D.stride(-1) != 1:
        # Pending the aligned-mode mirror: m-major D (swap_ab) ran on the
        # removed slab walk and is temporarily rejected.
        raise ValueError("epi_reduce_mode: D must be n-major")
    use_2cta = cluster_M % 2 == 0 and tile_M in (128, 256)
    cta_m = tile_M // (2 if use_2cta else 1)
    # Cutlass-aligned comm contract: full tiles (comm reads/stores are
    # unpredicated whole slices), whole-row slice quantum, and (RS) slabs of
    # whole CTA tiles so the owner remap is exact.
    if m % num_ranks:
        raise ValueError(f"epi_reduce_mode: m ({m}) must be divisible by world ({num_ranks})")
    if m % cta_m or n % tile_N:
        raise ValueError(
            f"epi_reduce_mode requires full tiles: m % {cta_m} == 0 and n % {tile_N} == 0"
        )
    if cta_m % num_ranks:
        raise ValueError(f"epi_reduce_mode: cta_m ({cta_m}) must be divisible by world")
    if mode == "reduce_scatter" and (m // num_ranks) % cta_m:
        raise ValueError(f"reduce_scatter: the slab (m/world) must be whole CTA tiles of {cta_m}")
    vec = 16 // D.element_size()
    if n % vec:
        raise ValueError(f"epi_reduce_mode: n ({n}) must be divisible by {vec} (16 B vectors)")
    d_rows = m // num_ranks if mode == "reduce_scatter" else m
    if D.shape[-2] != d_rows or D.shape[-1] != n:
        raise ValueError(
            f"epi_reduce_mode={mode}: D rows x cols ({d_rows}, {n}) expected, "
            f"got ({D.shape[-2]}, {D.shape[-1]})"
        )
    lamport = mode == "all_reduce_lamport"
    ws_mnl = (*epi_reduce_workspace_shape(m, n, cta_m, tile_N), 3 * num_ranks * l if lamport else l)
    for name, t in (("workspace", era.workspace), ("workspace_mc", era.workspace_mc)):
        if t is None or tuple(t.shape) != ws_mnl:
            raise ValueError(
                f"epi_reduce_args.{name}: kernel-order padded (m, n, l) {ws_mnl} expected, "
                f"got {None if t is None else tuple(t.shape)}"
            )
    if era.workspace.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"epi_reduce_args.workspace dtype {era.workspace.dtype} not multimem-reducible")
    if era.workspace_mc.dtype != era.workspace.dtype or era.workspace.stride(1) != 1:
        raise ValueError("epi_reduce_args.workspace must be n-major, workspace_mc same dtype")
    if mode == "all_reduce" and era.workspace.dtype != D.dtype:
        # Not inherent: the broadcast commit hardcodes 16 B multimem_st of D-dtype
        # atoms; teach it 8/32 B atoms if a workload needs mismatched-dtype AR.
        raise ValueError("all_reduce requires workspace dtype == D dtype")
    ws_vec = 16 // era.workspace.element_size()
    if n % ws_vec:
        raise ValueError(
            f"epi_reduce_mode: n ({n}) must be divisible by {ws_vec} (workspace vectors)"
        )
    # The comm warps cover the (cta_m/world, tile_N) slice (whole tile under lamport) in
    # 128 b atoms; the kernel is built with the same pick.
    tile_split = 1 if lamport else num_ranks
    if pick_num_comm_warps(cta_m, tile_N, era.workspace.element_size() * 8, tile_split) is None:
        raise ValueError(
            f"no comm-warp count (4..1) covers the ({cta_m // tile_split} x {tile_N} "
            f"{era.workspace.dtype}) slice in whole 128 b atoms"
        )
    if mode in ("reduce_scatter", "all_reduce_lamport"):
        if era.mD_mc is not None:
            raise ValueError(f"{mode} commits to plain local D: mD_mc must be None")
    elif era.mD_mc is None or tuple(era.mD_mc.shape) != (m, n, l):
        raise ValueError(
            f"all_reduce broadcast needs mD_mc, kernel-order (m, n, l) {(m, n, l)}, got "
            f"{None if era.mD_mc is None else tuple(era.mD_mc.shape)}"
        )
    if lamport:
        if era.slot is None or era.slot.numel() != 1 or era.slot.dtype != torch.int32:
            raise ValueError("all_reduce_lamport needs epi_reduce_args.slot: a (1,) int32 counter")
        return
    n_tiles = (n + tile_N - 1) // tile_N
    ntiles = ((m + cta_m - 1) // cta_m) * n_tiles * l
    num_sms = torch.cuda.get_device_properties(D.device).multi_processor_count
    # tile flags + exit-slot tail (one per resident CTA)
    if era.tile_flags_per_peer is None or len(era.tile_flags_per_peer) != num_ranks:
        raise ValueError(f"epi_reduce_args.tile_flags_per_peer must hold {num_ranks} peer views")
    for t in (era.tile_flags, era.tile_flags_mc, *era.tile_flags_per_peer):
        if t is None or t.numel() < ntiles + num_sms:
            raise ValueError(f"epi_reduce_args tile flags need >= {ntiles} + {num_sms} entries")


# ---- flag map + cross-launch exit barrier ----


@cute.jit
def epi_reduce_exit_slot(exit_base: Int32) -> Int32:
    """This CTA's exit-barrier slot in the flag tail (one per CTA of the grid,
    past the tile flags at exit_base). Keep the block_idx/grid_dim reads and their
    use inside one jit returning a scalar: materialized separately they
    mis-compute the slot, which writes out of bounds."""
    bidx, bidy, bidz = cute.arch.block_idx()
    gdx, gdy, _ = cute.arch.grid_dim()
    return exit_base + bidx + gdx * (bidy + gdy * bidz)


def epi_reduce_flag_slot(m_tile, n_tile, batch, flag_ntile_mn):
    """Flag index of CTA tile (m_tile, n_tile, batch): M-major linear id over
    the real (unpadded) tile grid. The single definition of the producer ->
    reducer flag map; extents come from the problem shape, never from padded
    buffer shapes."""
    return m_tile + flag_ntile_mn[0] * (n_tile + flag_ntile_mn[1] * batch)


def epi_reduce_slab(m_tile, tiles_per_slab):
    """reduce_scatter splits the tile grid along M into one slab per rank. Returns
    (slab, pos) for tile row m_tile, which both sides of the protocol read: the
    producer signals rank ``slab``, the only rank that consumes the tile; the comm
    warps at m_tile reduce stripe ``slab`` of the tile at ``pos`` in their own slab.
    The single definition of who owns what."""
    return m_tile // tiles_per_slab, m_tile % tiles_per_slab


# ---- multimem reduce + store ----
# Tile-agnostic (no GEMM state; a standalone RS kernel could bind them), under a
# three-part contract: the reduce reads a symmetric padded workspace through its mc
# view (unpredicated — full tiles are enforced in validate_epi_reduce_args); the
# partition's value atom is one contiguous 128b vector (n-major, N % (16B/elem) == 0);
# subtile (mi, ni) owns the even fragment block rows [mi*chunk, (mi+1)*chunk) x cols
# [ni*sub_loop_n, (ni+1)*sub_loop_n).
#
# The pair is the epilogue's contract: the reduce fills registers, the commit drains
# them, and the epi ops run in between once epilogue() can be framed on the slice.
# Splitting there is also what lets the workspace hold wider partials than D — the
# commit owns the d_dtype conversion.


@cute.jit
def multimem_reduce_subtile(
    frgWs_mc: cute.Tensor,
    tRS_rD: cute.Tensor,
    epi_coord: cute.Coord,
    # load_acc_subtile signature compat (acc prepass); a multimem load has nothing to release.
    no_release: cutlass.Constexpr[bool] = False,
) -> None:
    """Reduce this subtile's workspace partials across all ranks into tRS_rD via
    multimem ld_reduce; bound as epilogue()'s load_acc_subtile by the comm warps.
    Every load of the subtile is issued before the commit's stores, which is what
    keeps a latency-bound visit off one switch round trip per atom."""
    _atom, chunk, sub_loop_n = tRS_rD.shape
    ld_reduce = multimem_ld_reduce_128b(frgWs_mc.element_type)
    tmp_results = cute.make_rmem_tensor((4, chunk, sub_loop_n), cutlass.Int32)
    for ii in cutlass.range_constexpr(chunk):
        i = epi_coord[0] * chunk + ii
        for jj in cutlass.range_constexpr(sub_loop_n):
            j = epi_coord[1] * sub_loop_n + jj
            mc_ptr = frgWs_mc[None, i, j].iterator
            x, y, z, w = ld_reduce(mc_ptr)
            tmp_results[0, ii, jj] = x
            tmp_results[1, ii, jj] = y
            tmp_results[2, ii, jj] = z
            tmp_results[3, ii, jj] = w
    tmp_rD = cute.recast_tensor(tmp_results, frgWs_mc.element_type)
    tRS_rD.store(tmp_rD.load().to(tRS_rD.element_type))


def _subtile_to_dtype(tRS_rD, dtype):
    """d_dtype-converted register copy (tRS_rD itself when dtypes already match)."""
    if const_expr(tRS_rD.element_type == dtype):
        return tRS_rD
    tmp_out = cute.make_rmem_tensor(tRS_rD.layout.shape, dtype)
    tmp_out.store(tRS_rD.load().to(dtype))
    return tmp_out


@cute.jit
def signal_tile_broadcast(
    tile_flags_mc: cute.Tensor, tile_id: Int32, in_bounds: cutlass.Boolean
) -> None:
    """Announce a finished partial to every rank's copy of the tile's flag: the
    all_reduce consumer set, where each rank reduces a stripe of every tile."""
    if in_bounds:
        with cute.arch.elect_one():
            utils.distributed.multimem_red_add1(
                lock_ptr=tile_flags_mc.iterator + tile_id, scope="gpu", order="release"
            )


@cute.jit
def signal_tile_owner(
    tile_flags_per_peer: tuple,
    tile_id: Int32,
    owner_rank: Int32,
    in_bounds: cutlass.Boolean,
) -> None:
    """Announce a finished partial to the owning rank's copy only: the
    reduce_scatter consumer set, where one rank reduces the tile. The DSL cannot
    index a tuple of tensors by a runtime value, so the constexpr loop
    materializes the world-way dispatch."""
    if in_bounds:
        with cute.arch.elect_one():
            for r in cutlass.range_constexpr(len(tile_flags_per_peer)):
                if owner_rank == r:
                    utils.distributed.red_add1(
                        lock_ptr=tile_flags_per_peer[r].iterator + tile_id,
                        order="release",
                        scope="gpu",
                    )


@cute.jit
def commit_subtile_local(
    frgD: cute.Tensor,
    tRS_rD: cute.Tensor,
    epi_coord: cute.Coord,
) -> None:
    """reduce_scatter commit: vectorized stores of the reduced (post-EVT) subtile into
    this rank's plain slab-shaped D. Passed to epilogue() as commit_D. Owns the
    d_dtype conversion, so the workspace may hold wider partials than D."""
    _atom, chunk, sub_loop_n = tRS_rD.shape
    tmp_out = _subtile_to_dtype(tRS_rD, frgD.element_type)
    for ii in cutlass.range_constexpr(chunk):
        i = epi_coord[0] * chunk + ii
        for jj in cutlass.range_constexpr(sub_loop_n):
            j = epi_coord[1] * sub_loop_n + jj
            cute.autovec_copy(tmp_out[None, ii, jj], frgD[None, i, j])


@cute.jit
def commit_subtile_broadcast(
    frgD_mc: cute.Tensor,
    tRS_rD: cute.Tensor,
    epi_coord: cute.Coord,
) -> None:
    """all_reduce commit: multimem_st broadcast of the reduced (post-EVT) subtile to
    every rank's symmetric D. Passed to epilogue() as commit_D. The 128b broadcast
    atom is why all_reduce requires workspace dtype == D dtype."""
    _atom, chunk, sub_loop_n = tRS_rD.shape
    tmp_out = _subtile_to_dtype(tRS_rD, frgD_mc.element_type)
    out_i32 = cute.recast_tensor(tmp_out, cutlass.Int32)
    for ii in cutlass.range_constexpr(chunk):
        i = epi_coord[0] * chunk + ii
        for jj in cutlass.range_constexpr(sub_loop_n):
            j = epi_coord[1] * sub_loop_n + jj
            utils.distributed.multimem_st_4xb32(
                frgD_mc[None, i, j].iterator,
                out_i32[0, ii, jj].ir_value(),
                out_i32[1, ii, jj].ir_value(),
                out_i32[2, ii, jj].ir_value(),
                out_i32[3, ii, jj].ir_value(),
            )


# ---- lamport one-shot all_reduce (cutlass Lamport) ----
# Every rank TMA-multicasts its -0.0-scrubbed partial into slot[ping][rank] of every
# rank's 3-slot workspace; the comm warps poll their local slot[ping][0..world) per
# 16 B — the data is the flag — sum in f32 and store into plain local D. No tile
# flags, no exit barrier. The producer re-arms slot[pong] per tile before that tile's
# store: peers write pong next call, and only after receiving every tile of this one.

# 16 B atoms in flight per sentinel wait in the lamport consumer (1 = serial spin).
LAMPORT_BATCH = 8


def sentinel_word(dtype) -> int:
    """-0.0 in every lane of one 32-bit word, as a signed int32: lamport's
    "not arrived" sentinel (16-bit dtypes pack two lanes)."""
    word = 0x80008000 if dtype.width == 16 else 0x80000000
    return word - (1 << 32)


def lamport_mirror_batches(slot, num_ranks: int, l, batch, phase: int):
    """Workspace batch indices of the (m, n) mirrors in slot (slot + phase) % 3, one
    per source rank: phase 0 = ping (this call's stores / polls), 1 = pong (re-arm)."""
    s = (slot + phase) % 3
    return tuple((s * num_ranks + r) * l + batch for r in range(num_ranks))


@cute.jit
def lamport_store_subtile(
    tiled_copy_r2s: cute.TiledCopy, tRS_rD: cute.Tensor, tRS_sWs_stage: cute.Tensor
) -> None:
    """Scrub the sentinel from the converted partial: x + 0.0 is bit-exact for
    every value except -0.0 -> +0.0 under round-to-nearest."""
    src = tRS_rD.to(tRS_sWs_stage.element_type)
    vec = src.load()
    src.store(vec + cute.full_like(vec, 0.0))
    cute.copy(tiled_copy_r2s, src, tRS_sWs_stage)


@cute.jit
def lamport_rearm_tile(frg_clear: tuple) -> None:
    """Store the sentinel over this tile in every local mirror of slot[pong]."""
    neg_zero = sentinel_word(frg_clear[0].element_type)
    _atom, loop_m, loop_n = frg_clear[0].shape
    for r in cutlass.range_constexpr(len(frg_clear)):
        for i in cutlass.range(loop_m, unroll=1):
            for j in cutlass.range(loop_n, unroll=1):
                st_weak_global_4xu32(
                    frg_clear[r][None, i, j].iterator, neg_zero, neg_zero, neg_zero, neg_zero
                )


@cute.jit
def lamport_reduce_subtile(
    frg_in: tuple,
    tRS_rD: cute.Tensor,
    epi_coord: cute.Coord,
    no_release: cutlass.Constexpr[bool] = False,
) -> None:
    """Batched poll over the (atom, source mirror) lattice, atom-major: issue
    LAMPORT_BATCH 16 B loads back to back, spin reloading them until no last word
    is the sentinel (the multicast TMA store lands atomically per 16 B, so the
    probe that sees data carries the whole atom), then sum the landed atoms in
    tRS_rD's dtype. A batch may cross mirrors; the lattice is unrolled at trace
    time, so 4 * LAMPORT_BATCH u32 of loads are live besides the accumulator."""
    _atom, chunk, sub_loop_n = tRS_rD.shape
    ws_dtype = frg_in[0].element_type
    neg_zero = sentinel_word(ws_dtype)
    num_peers = len(frg_in)
    tmp = cute.make_rmem_tensor((4, chunk, sub_loop_n), cutlass.Int32)
    tmp_ws = cute.recast_tensor(tmp, ws_dtype)
    # Flat accumulator in tmp_ws's profile: tRS_rD's atom mode is hierarchical, so
    # only the final store crosses profiles (as the ld_reduce path's store does).
    acc = cute.make_rmem_tensor(tmp_ws.shape, tRS_rD.element_type)
    L = chunk * sub_loop_n * num_peers
    W = LAMPORT_BATCH
    for g in cutlass.range_constexpr((L + W - 1) // W):
        q0, q1 = g * W, min(L, (g + 1) * W)
        n = q1 - q0
        ptrs = []
        state = []
        for q in cutlass.range_constexpr(q0, q1):
            c, r = q // num_peers, q % num_peers
            ii, jj = c // sub_loop_n, c % sub_loop_n
            ptr = frg_in[r][None, epi_coord[0] * chunk + ii, epi_coord[1] * sub_loop_n + jj].iterator
            ptrs.append(ptr)
            state.extend(ld_relaxed_gpu_global_4xu32(ptr))
        spin = while_generate(state, lambda *rg: any_(rg[4 * k + 3] == neg_zero for k in range(n)))
        with spin:
            nxt = []
            for k in cutlass.range_constexpr(n):
                nxt.extend(ld_relaxed_gpu_global_4xu32(ptrs[k]))
            yield_out(nxt)
        res = spin.results
        for k in cutlass.range_constexpr(n):
            c, r = (q0 + k) // num_peers, (q0 + k) % num_peers
            ii, jj = c // sub_loop_n, c % sub_loop_n
            tmp[0, ii, jj] = res[4 * k]
            tmp[1, ii, jj] = res[4 * k + 1]
            tmp[2, ii, jj] = res[4 * k + 2]
            tmp[3, ii, jj] = res[4 * k + 3]
            vals = tmp_ws[None, ii, jj].load().to(tRS_rD.element_type)
            if const_expr(r == 0):
                acc[None, ii, jj].store(vals)
            else:
                acc[None, ii, jj].store(acc[None, ii, jj].load() + vals)
    tRS_rD.store(acc.load())
