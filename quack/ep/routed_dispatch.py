# MegaMoE's in-kernel routed dispatch for the intra GEMM1 (DeepGEMM sm100_bf16_mega_moe: dispatch_prep /
# dispatch_barrier / dispatch_pull), run by the GEMM's 4 transfer warps before any tile is scheduled:
#   prep     each CTA histograms its slice of the local top-k, takes base slots per expert with one packed atomic and
#            writes every (token, k) index word into the destination rank's (local expert, source rank) slot list;
#   barrier  grid sync, CTA 0 pushes the per-expert counts to the destinations, NVLink barrier, CTA 0 writes the
#            expert offsets (cu_seqlens), grid sync;
#   pull     row by row in expert-major order (source ranks interleaved as MegaMoE): TMA-load the source row over
#            NVLink into the warp's smem stage, bulk-store it to the expert-contiguous pool, record its combine slot,
#            bump the tile's dispatch flag (lane-parked, batched releases as dispatch_pull.pull_rows).
# Routed: K rows per token cross NVLink, no dedup; the rows land expert-major, so GEMM1 reads A with plain TMA.
from types import SimpleNamespace

import torch
import torch.distributed._symmetric_memory as symm_mem
import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import AddressSpace
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import Int32, Int64, T, dsl_user_op
from quack.ep.ptx import (
    cp_async_bulk_s2g, fns_b32, red_add_relaxed_sys_u64_raw, red_add_release_gpu_s32, red_add_release_sys_s32,
    software_grid_sync, stg_b32_raw, stg_b64_raw, tma_load_1d,
)

PREP_BARRIER_ID = 12  # the transfer warps' CTA-local barrier (10 = software_grid_sync's, 11 = dispatch -> scheduler)
DISPATCH_TO_SCHED_BARRIER_ID = 11


@cute.jit
def routed_mark(debug_base: Int64, tag: cutlass.Constexpr[int], value: Int32):
    """The same marker store from any warp of CTA (0, 0, 0), lane 0 (the GEMM-side phases)."""
    bidx, bidy, bidz = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    if (debug_base != Int64(0)) & (bidx == 0) & (bidy == 0) & (bidz == 0) & (tidx % 32 == Int32(0)):
        stg_b32_raw(debug_base + Int64(tag * 4), value)


@dsl_user_op
def read_globaltimer(*, loc=None, ip=None) -> Int64:
    """%globaltimer: ns, comparable across SMs."""
    return Int64(llvm.inline_asm(T.i64(), [], "mov.u64 $0, %globaltimer;", "=l", has_side_effects=True, asm_dialect=0, loc=loc, ip=ip))


@cute.jit
def routed_stamp(debug_base: Int64, tag: cutlass.Constexpr[int]):
    """Phase timestamp: CTA (0, 0, 0), lane 0 of the calling warp stores %globaltimer at 64-bit word 32 + tag."""
    bidx, bidy, bidz = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    if (debug_base != Int64(0)) & (bidx == 0) & (bidy == 0) & (bidz == 0) & (tidx % 32 == Int32(0)):
        stg_b64_raw(debug_base + Int64(256 + tag * 8), read_globaltimer())


@cute.jit
def _trace(tag: cutlass.Constexpr[int], debug_base: Int64, cta: Int32, warp: Int32, lane: Int32, value: Int32):
    """Hang localization: CTA 0 warp 0 lane 0 stores `value` at word `tag` of a host-mapped (pinned) buffer the host
    polls while the kernel runs (device printf never flushes from a hung kernel). debug_base 0 = off."""
    if (debug_base != Int64(0)) & (cta == Int32(0)) & (warp == Int32(0)) & (lane == Int32(0)):
        stg_b32_raw(debug_base + Int64(tag * 4), value)


@cute.jit
def routed_prep(
    smem_count: cute.Tensor,  # (E,) int32 smem scratch
    topk: cute.Tensor,  # (T * K,) int32 global expert ids of this rank's tokens
    send_count: cute.Tensor,  # (E,) int64, zeroed before launch: high 32 = CTAs arrived, low 32 = tokens
    idx_base: cute.Tensor,  # (W,) int64 base of every rank's slot lists ((E_local, W, T * K) int32)
    rank: Int32,
    T: Int32,
    cta: Int32,
    num_ctas: Int32,
    warp: Int32,
    lane: Int32,
    E: cutlass.Constexpr[int],
    E_local: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    W: cutlass.Constexpr[int],
    debug_base: Int64,
):
    tid = warp * Int32(32) + lane
    _trace(0, debug_base, cta, warp, lane, T)  # prep entered
    count = smem_count.iterator
    i = tid
    while i < Int32(E):
        (count + i).store(Int32(0))
        i = i + Int32(128)
    cute.arch.barrier(barrier_id=PREP_BARRIER_ID, number_of_threads=128)
    tokens_per_warp: cutlass.Constexpr[int] = 32 // K
    active_lanes: cutlass.Constexpr[int] = tokens_per_warp * K
    base = (cta * Int32(4) + warp) * Int32(tokens_per_warp)
    stride = num_ctas * Int32(4 * tokens_per_warp)
    t = base
    while t < T:
        tok = t + lane // Int32(K)
        if (lane < Int32(active_lanes)) & (tok < T):
            e = topk[tok * Int32(K) + lane % Int32(K)]
            if e >= Int32(0):
                cute.arch.atomic_add(count + e, Int32(1), sem="relaxed", scope="cta")
        cute.arch.sync_warp()
        t = t + stride
    cute.arch.barrier(barrier_id=PREP_BARRIER_ID, number_of_threads=128)
    e = tid
    while e < Int32(E):
        local = (count + e).load()
        old = cute.arch.atomic_add(
            send_count.iterator + e, (Int64(1) << Int64(32)) | (Int64(local) & Int64(0xFFFFFFFF)), sem="relaxed", scope="gpu"
        )
        (count + e).store(Int32(old & Int64(0xFFFFFFFF)))  # this CTA's base slot per expert
        e = e + Int32(128)
    cute.arch.barrier(barrier_id=PREP_BARRIER_ID, number_of_threads=128)
    TK = T * Int32(K)
    t = base
    while t < T:
        tok = t + lane // Int32(K)
        if (lane < Int32(active_lanes)) & (tok < T):
            k = lane % Int32(K)
            e = topk[tok * Int32(K) + k]
            if e >= Int32(0):
                dst = e // Int32(E_local)
                slot = cute.arch.atomic_add(count + e, Int32(1), sem="relaxed", scope="cta")
                off = ((e % Int32(E_local)) * Int32(W) + rank) * TK + slot
                stg_b32_raw(idx_base[dst] + Int64(off) * Int64(4), tok * Int32(K) + k)
        cute.arch.sync_warp()
        t = t + stride


@cute.jit
def routed_barrier(
    send_count: cute.Tensor,  # (E,) int64
    recv_count_base: cute.Tensor,  # (W,) int64 base of every rank's (W * E_local,) int64 per-source counts
    recv_sum_base: cute.Tensor,  # (W,) int64 base of every rank's (E_local,) int64 packed totals
    recv_sum: cute.Tensor,  # (E_local,) int64 this rank's totals (peers add into it)
    signal: cute.Tensor,  # (2,) int32 this rank's NVLink barrier slots
    signal_base: cute.Tensor,  # (W,) int64
    barrier_counter: cute.Tensor,  # (1,) int32 persistent across launches (sense-reversing, never reset)
    grid_counter: cute.Tensor,  # (1,) int32 software grid sync slot (self-resetting)
    cu: cute.Tensor,  # (E_local + 1,) int32 out: the expert offsets, written by CTA 0
    rank: Int32,
    cta: Int32,
    num_ctas: Int32,
    warp: Int32,
    lane: Int32,
    E: cutlass.Constexpr[int],
    E_local: cutlass.Constexpr[int],
    W: cutlass.Constexpr[int],
    debug_base: Int64,
):
    tid = warp * Int32(32) + lane
    _trace(1, debug_base, cta, warp, lane, num_ctas)  # prep done
    software_grid_sync(grid_counter, cta, num_ctas, tid, num_threads=128)  # every CTA's index words are written
    _trace(2, debug_base, cta, warp, lane, Int32(1))  # grid sync 1 passed
    if cta == Int32(0):
        e = tid
        while e < Int32(E):
            status = cute.arch.load(send_count.iterator + e, Int64, sem="relaxed", scope="gpu")
            dst = e // Int32(E_local)
            le = e % Int32(E_local)
            stg_b64_raw(recv_count_base[dst] + Int64(rank * Int32(E_local) + le) * Int64(8), status & Int64(0xFFFFFFFF))
            red_add_relaxed_sys_u64_raw(recv_sum_base[dst] + Int64(le) * Int64(8), status)
            e = e + Int32(128)
        cute.arch.fence_acq_rel_sys()
    cute.arch.barrier(barrier_id=PREP_BARRIER_ID, number_of_threads=128)
    _trace(3, debug_base, cta, warp, lane, Int32(1))  # counts pushed
    if (cta == Int32(0)) & (warp == Int32(0)):  # NVLink barrier: sense-reversing ping-pong on two slots, as the port
        status = barrier_counter[0] & Int32(3)
        phase = status & Int32(1)
        delta, target = Int32(1), Int32(W)
        if (status >> Int32(1)) != Int32(0):
            delta, target = Int32(-1), Int32(0)
        if lane < Int32(W):
            red_add_release_sys_s32(signal_base[lane] + Int64(phase) * Int64(4), delta)
        cute.arch.sync_warp()
        if lane == Int32(0):
            barrier_counter[0] = status + Int32(1)
            while cute.arch.load(signal.iterator + phase, Int32, sem="acquire", scope="sys") != target:
                pass
            _trace(4, debug_base, cta, warp, lane, target)  # NVLink barrier passed
            acc = Int32(0)  # the expert offsets from the (now complete) totals
            cu[0] = acc
            for le in cutlass.range(E_local, unroll=1):
                acc = acc + Int32(cute.arch.load(recv_sum.iterator + le, Int64, sem="relaxed", scope="sys") & Int64(0xFFFFFFFF))
                cu[le + Int32(1)] = acc
            _trace(5, debug_base, cta, warp, lane, acc)  # cu written: total rows
        cute.arch.sync_warp()
    software_grid_sync(grid_counter, cta, num_ctas, tid, num_threads=128)  # the offsets are visible everywhere
    _trace(6, debug_base, cta, warp, lane, Int32(1))  # grid sync 2 passed


@cute.jit
def routed_pull(
    recv_count: cute.Tensor,  # (W * E_local,) int64 rows per (source rank, local expert)
    recv_sum: cute.Tensor,  # (E_local,) int64 packed totals
    src_idx: cute.Tensor,  # (E_local * W * T * K,) int32 this rank's slot lists
    x_base: cute.Tensor,  # (W,) int64 base of every rank's x buffer
    pool_base: Int64,  # address of pool row 0; expert-contiguous, rows row_bytes apart
    row_bytes: Int32,
    slot_of_row: cute.Tensor,  # (rows,) int32 out: source rank * T * K + token * K + k of each pool row
    flags: cute.Tensor,  # (E_local, M clusters) int32 dispatch flags (the scheduler's), zeroed before launch
    stage: cute.Tensor,  # this warp's smem row
    mbarrier: cute.Pointer,  # this warp's smem mbarrier, initialized with arrive count 1
    global_warp: Int32,
    global_warps: Int32,
    lane: Int32,
    T: Int32,
    K: cutlass.Constexpr[int],
    W: cutlass.Constexpr[int],
    E_local: cutlass.Constexpr[int],
    block_rows: cutlass.Constexpr[int],
    flag_batch: cutlass.Constexpr[int],
    debug_base: Int64,
):
    """MegaMoE's token_in loop with contiguous expert packing: pool row = the row's position in the expert-major order;
    within an expert the source ranks are interleaved round-robin over the ranks still holding rows."""
    experts_per_lane: cutlass.Constexpr[int] = (E_local + 31) // 32
    TK = T * Int32(K)
    stored = cute.make_rmem_tensor(cute.make_layout(experts_per_lane), Int32)  # rows of expert (i * 32 + lane)
    for i in cutlass.range_constexpr(experts_per_lane):
        stored[i] = Int32(0)
        e = Int32(i * 32) + lane
        if e < Int32(E_local):
            stored[i] = Int32(recv_sum[e] & Int64(0xFFFFFFFF))
    cute.arch.sync_warp()
    cur_e = Int32(-1)
    e_start, e_end = Int32(0), Int32(0)
    rank_rows = Int32(0)  # lane < W: rows of the current expert from source rank `lane`
    pending, rows_parked, lane_gate = Int32(0), Int32(0), Int32(0)
    phase = Int32(0)
    row = global_warp
    while cur_e < Int32(E_local):
        old_e = cur_e
        while (row >= e_end) & (cur_e < Int32(E_local)):  # advance to the expert holding this row
            cur_e = cur_e + Int32(1)
            if cur_e < Int32(E_local):
                e_start = e_end
                mine = Int32(0)
                for i in cutlass.range_constexpr(experts_per_lane):
                    if cur_e == Int32(i * 32) + lane:
                        mine = stored[i]
                e_end = e_end + cute.arch.shuffle_sync(mine, cur_e % Int32(32))
        if cur_e < Int32(E_local):
            if old_e != cur_e:
                rank_rows = Int32(0)
                if lane < Int32(W):
                    rank_rows = Int32(recv_count[lane * Int32(E_local) + cur_e] & Int64(0xFFFFFFFF))
            idx_in_e = row - e_start
            # (source rank, index in its list) of row idx_in_e: rounds over the ranks still holding rows
            slot, offset, remaining = idx_in_e, Int32(0), rank_rows
            src, idx_in_rank, decided = Int32(0), Int32(0), Int32(0)
            for _ in cutlass.range_constexpr(W + 1):
                if decided == Int32(0):
                    active = remaining > Int32(0)
                    mask = cute.arch.vote_ballot_sync(active)
                    n_active = Int32(cute.arch.popc(Int32(mask)))
                    v = Int32(0x7FFFFFFF)
                    if active:
                        v = remaining
                    length = Int32(cute.arch.warp_redux_sync(v, "min"))
                    if n_active > Int32(0):
                        round_rows = length * n_active
                        if slot < round_rows:
                            src = fns_b32(Int32(mask), Int32(0), slot % n_active + Int32(1))
                            idx_in_rank = offset + slot // n_active
                            decided = Int32(1)
                        else:
                            slot = slot - round_rows
                            offset = offset + length
                            remaining = cutlass.max(remaining - length, Int32(0))
                    else:
                        decided = Int32(1)
            word = src_idx[(cur_e * Int32(W) + src) * TK + idx_in_rank]
            token = word // Int32(K)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(mbarrier, row_bytes)
                tma_load_1d(stage.iterator, x_base[src] + Int64(token) * Int64(row_bytes), mbarrier, row_bytes)
            cute.arch.sync_warp()
            cute.arch.mbarrier_wait(mbarrier, phase)
            dst = cute.make_ptr(stage.element_type, pool_base + Int64(row) * Int64(row_bytes), AddressSpace.gmem, assumed_align=16)
            with cute.arch.elect_one():
                cp_async_bulk_s2g(dst, stage.iterator, row_bytes)
                slot_of_row[row] = src * TK + word
            cute.arch.sync_warp()
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(0, read=True)  # stage free for the next load; landing waited at flush
            cute.arch.sync_warp()
            gate = Int32(cute.crd2idx((cur_e, idx_in_e // Int32(block_rows)), flags.layout))
            if lane == pending:
                lane_gate = gate
            pending = pending + Int32(1)
            rows_parked = rows_parked + Int32(1)
            if (rows_parked == Int32(flag_batch)) | (pending == Int32(32)):
                cute.arch.cp_async_bulk_wait_group(0, read=False)  # the parked rows' bytes have landed
                if lane < pending:
                    red_add_release_gpu_s32(flags.iterator + lane_gate, Int32(1))
                pending, rows_parked = Int32(0), Int32(0)
            cute.arch.sync_warp()
            phase = phase ^ Int32(1)
            if (debug_base != Int64(0)) & (global_warp == Int32(0)) & (lane == Int32(0)):
                stg_b32_raw(debug_base + Int64(7 * 4), row)  # pull progress of warp 0
            row = row + global_warps
    if pending > Int32(0):
        cute.arch.cp_async_bulk_wait_group(0, read=False)
        if lane < pending:
            red_add_release_gpu_s32(flags.iterator + lane_gate, Int32(1))
    cute.arch.sync_warp()
    if (debug_base != Int64(0)) & (global_warp == Int32(0)) & (lane == Int32(0)):
        stg_b32_raw(debug_base + Int64(8 * 4), Int32(1))  # pull done (warp 0)


@cute.jit
def routed_tail(
    send_count: cute.Tensor,  # (E,) int64
    recv_count: cute.Tensor,  # (W * E_local,) int64
    recv_sum: cute.Tensor,  # (E_local,) int64
    grid_counter: cute.Tensor,
    cta: Int32,
    num_ctas: Int32,
    warp: Int32,
    lane: Int32,
    E: cutlass.Constexpr[int],
    E_local: cutlass.Constexpr[int],
    W: cutlass.Constexpr[int],
    debug_base: Int64,
):
    """After every CTA's pull: CTA 0 zeroes the exchange buffers for the next forward, as MegaMoE's kernel tail. A
    host memset would race with a faster peer's pushes; here no peer pushes again before the pipeline's end-of-forward
    barrier, and no warp of this rank reads them after the grid sync."""
    tid = warp * Int32(32) + lane
    software_grid_sync(grid_counter, cta, num_ctas, tid, num_threads=128)
    _trace(15, debug_base, cta, warp, lane, Int32(1))  # every CTA's pull is done
    routed_stamp(debug_base, 4)
    if cta == Int32(0):
        i = tid
        while i < Int32(E):
            send_count[i] = Int64(0)
            i = i + Int32(128)
        i = tid
        while i < Int32(W * E_local):
            recv_count[i] = Int64(0)
            i = i + Int32(128)
        i = tid
        while i < Int32(E_local):
            recv_sum[i] = Int64(0)
            i = i + Int32(128)


def make_routed_buffers(T_local, K, E, E_local, max_rows, num_clusters, x_peer_bufs, group):
    """Per rank: the buffers the routed dispatch exchanges through (symmetric where peers write) and every rank's base
    addresses; reset() = the per-forward memsets (barrier_counter and grid_counter are self-resetting)."""
    dev = torch.cuda.current_device()
    W = group.size()
    i64 = lambda *shape: symm_mem.empty(*shape, dtype=torch.int64, device=dev)
    src_idx = symm_mem.empty(E_local * W * T_local * K, dtype=torch.int32, device=dev)
    recv_count, recv_sum = i64(W * E_local), i64(E_local)
    signal = symm_mem.empty(2, dtype=torch.int32, device=dev)
    hdls = [symm_mem.rendezvous(t, group=group) for t in (src_idx, recv_count, recv_sum, signal)]
    bases = lambda hdl, t: torch.tensor(
        [hdl.get_buffer(q, t.shape, t.dtype).data_ptr() for q in range(W)], dtype=torch.int64, device=dev
    )
    signal.zero_()
    b = SimpleNamespace(
        topk=None, send_count=torch.zeros(E, dtype=torch.int64, device=dev),
        src_idx=src_idx, idx_base=bases(hdls[0], src_idx),
        recv_count=recv_count, recv_count_base=bases(hdls[1], recv_count),
        recv_sum=recv_sum, recv_sum_base=bases(hdls[2], recv_sum),
        signal=signal, signal_base=bases(hdls[3], signal),
        barrier_counter=torch.zeros(1, dtype=torch.int32, device=dev), grid_counter=torch.zeros(1, dtype=torch.int32, device=dev),
        cu=torch.zeros(E_local + 1, dtype=torch.int32, device=dev),
        x_base=torch.tensor([t.data_ptr() for t in x_peer_bufs], dtype=torch.int64, device=dev),
        slot_of_row=torch.empty(max_rows, dtype=torch.int32, device=dev),
        flags=torch.zeros(E_local, num_clusters, dtype=torch.int32, device=dev),
        hdls=hdls,
    )

    def reset():  # the exchange buffers are zeroed in the kernel's tail (routed_tail); only the local flags here
        b.flags.zero_()

    b.reset = reset
    return b
