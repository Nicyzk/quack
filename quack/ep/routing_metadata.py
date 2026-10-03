"""The dispatch metadata our pipeline reads, from the gathered top-k, in two cute kernels (replaces sonic's
compute_dispatch_metadata + _build_consumer_metadata + build_rank_dedup_a_idx: 12 launches, ~105 us at W=4 8192).

Slot f = (source rank w, token t, k) flattened over (W, T_local, K); its global expert e_g = topk[f], destination
dst = e_g // E_local, local expert e = e_g - dst * E_local. On rank `rank` (same conventions as sonic's tables, so the
walks, the GEMM and the checks are unchanged):
  cu[e]         = exclusive prefix over local experts of the number of slots with dst == rank (E_local + 1 entries)
  s_reverse[f]  = cu[e] + rank of f among the rank's slots of expert e, in f order (rows of an expert are in
                  (source rank, token, k) order); only written for slots with dst == rank
  s_scatter[r]  = f of the slot in row r (r < cu[E_local])
  a_idx[r]      = the pool row of row r's token: pool rows are the rank-dedup receive positions, tokens with >= 1
                  slot to this rank packed in (source rank, token) order (sonic's pair_offset + rank_dedup_recv_pos)
  first_src_row = the pool row where source rank (rank + 1) % W's tokens start (sonic's pair_offset[(rank+1)%W, rank])
Tiles of TILE_TOKENS tokens (TILE_TOKENS * K slots): count_kernel writes per tile the slot counts per local expert
and the number of tokens present to this rank; emit_kernel prefixes them (every CTA reads the small count table) and
writes the tables. A token's k slots have distinct experts, so a slot's rank inside a tile is a warp ballot."""
from types import SimpleNamespace

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass.cutlass_dsl import Int32
from quack.compile_utils import make_fake_tensor

TILE_TOKENS = 256  # tokens per tile = threads per CTA: one token (K slots) per thread


class RoutingMetadata:
    def __init__(self, topk_k, e_local, num_ctas):
        self.topk_k, self.e_local, self.num_ctas = topk_k, e_local, num_ctas
        self.num_warps = TILE_TOKENS // 32
        assert e_local <= TILE_TOKENS, "the per-expert prefix is computed one expert per thread"

    @cute.jit
    def __call__(self, topk, counts, present, cu, s_reverse, s_scatter, a_idx, first_src_row, rank: Int32,
                 tokens_per_rank: Int32, world: Int32, stream):
        launch = dict(grid=[self.num_ctas, 1, 1], block=[TILE_TOKENS, 1, 1], smem=16384, stream=stream)  # scratch 8 KB + cu 1 KB
        self.count_kernel(topk, counts, present, rank, tokens_per_rank).launch(**launch)
        self.emit_kernel(topk, counts, present, cu, s_reverse, s_scatter, a_idx, first_src_row, rank, tokens_per_rank,
                         world).launch(**launch)

    @cute.jit
    def tile_expert_ranks(self, topk, tile, tidx, lane, warp, rank: Int32, scratch, expert_rank, my_slots):
        """For thread tidx's token (tile * TILE_TOKENS + tidx): expert_rank[k] = rank of slot k among the tile's
        slots of the same local expert (f order) for the slots destined to this rank, my_slots = their bitmask;
        scratch[(e, warp)] gets each warp's count of expert e (the caller prefixes it). One ballot per expert."""
        lanes_before = (Int32(1) << lane) - Int32(1)
        token = tile * Int32(TILE_TOKENS) + tidx
        local = cute.make_rmem_tensor(cute.make_layout(self.topk_k), Int32)
        mask = Int32(0)
        for k in cutlass.range_constexpr(self.topk_k):
            e_g = topk[token * Int32(self.topk_k) + Int32(k)]
            e = e_g - rank * Int32(self.e_local)
            mine = (e >= Int32(0)) & (e < Int32(self.e_local))
            local[k] = e
            if mine:
                mask = mask | Int32(1 << k)
            expert_rank[k] = Int32(0)
        for e in cutlass.range(self.e_local, unroll=1):
            has = Int32(0)
            for k in cutlass.range_constexpr(self.topk_k):
                if (local[k] == e) & (((mask >> k) & Int32(1)) == Int32(1)):
                    has = Int32(1)
            ballot = cute.arch.vote_ballot_sync(has == Int32(1))
            before = cute.arch.popc(ballot & lanes_before)
            if lane == Int32(0):
                scratch[e * Int32(self.num_warps) + warp] = cute.arch.popc(ballot)
            for k in cutlass.range_constexpr(self.topk_k):
                if (local[k] == e) & (((mask >> k) & Int32(1)) == Int32(1)):
                    expert_rank[k] = before
        return mask

    @cute.kernel
    def count_kernel(self, topk, counts, present, rank: Int32, tokens_per_rank: Int32):
        """counts[tile, e] = the tile's slots of local expert e destined to this rank; present[tile] = its tokens
        with >= 1 such slot."""
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp, lane = tidx // 32, tidx % 32
        scratch = utils.SmemAllocator().allocate_tensor(Int32, cute.make_layout(TILE_TOKENS * self.num_warps), byte_alignment=16)
        expert_rank = cute.make_rmem_tensor(cute.make_layout(self.topk_k), Int32)
        my_slots = cute.make_rmem_tensor(cute.make_layout(1), Int32)
        n_tiles = Int32(cute.size(present))
        tile = Int32(bidx)
        while tile < n_tiles:
            mask = self.tile_expert_ranks(topk, tile, tidx, lane, warp, rank, scratch, expert_rank, my_slots)
            present_ballot = cute.arch.vote_ballot_sync(mask != Int32(0))
            if lane == Int32(0):
                scratch[Int32(self.e_local * self.num_warps) + warp] = cute.arch.popc(present_ballot)
            cute.arch.sync_threads()
            if tidx <= Int32(self.e_local):  # thread e sums expert e's warp counts; thread e_local the present tokens
                total = Int32(0)
                for w in cutlass.range_constexpr(self.num_warps):
                    total = total + scratch[tidx * Int32(self.num_warps) + Int32(w)]
                if tidx < Int32(self.e_local):
                    counts[tile, tidx] = total
                else:
                    present[tile] = total
            cute.arch.sync_threads()
            tile = tile + Int32(self.num_ctas)

    @cute.kernel
    def emit_kernel(self, topk, counts, present, cu, s_reverse, s_scatter, a_idx, first_src_row, rank: Int32,
                    tokens_per_rank: Int32, world: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp, lane = tidx // 32, tidx % 32
        allocator = utils.SmemAllocator()
        scratch = allocator.allocate_tensor(Int32, cute.make_layout(TILE_TOKENS * self.num_warps), byte_alignment=16)
        cu_s = allocator.allocate_tensor(Int32, cute.make_layout(TILE_TOKENS + 1), byte_alignment=16)  # cu, then the tile's expert prefixes
        expert_rank = cute.make_rmem_tensor(cute.make_layout(self.topk_k), Int32)
        my_slots = cute.make_rmem_tensor(cute.make_layout(1), Int32)
        n_tiles = Int32(cute.size(present))
        tiles_per_rank = tokens_per_rank // Int32(TILE_TOKENS)
        # cu: thread e totals expert e over every tile, then a serial prefix (E_local + 1 values) by thread 0
        if tidx < Int32(self.e_local):
            total = Int32(0)
            for j in cutlass.range(n_tiles, unroll=4):
                total = total + counts[j, tidx]
            cu_s[tidx] = total
        cute.arch.sync_threads()
        if tidx == Int32(0):
            acc = Int32(0)
            for e in cutlass.range(self.e_local, unroll=1):
                c = cu_s[e]
                cu_s[e] = acc
                acc = acc + c
            cu_s[self.e_local] = acc
            if bidx == Int32(0):
                # first_src_row: tokens present to this rank from the ranks before (rank + 1) % world
                next_rank = (rank + Int32(1)) % world
                first = Int32(0)
                for j in cutlass.range(next_rank * tiles_per_rank, unroll=4):
                    first = first + present[j]
                first_src_row[0] = first
        cute.arch.sync_threads()
        if bidx == Int32(0):
            if tidx <= Int32(self.e_local):
                cu[tidx] = cu_s[tidx]
        tile = Int32(bidx)
        while tile < n_tiles:
            # prefixes over the tiles before this one: per expert (thread e) and the present tokens (thread e_local)
            if tidx <= Int32(self.e_local):
                before = Int32(0)
                for j in cutlass.range(tile, unroll=4):
                    if tidx < Int32(self.e_local):
                        before = before + counts[j, tidx]
                    else:
                        before = before + present[j]
                cu_s[Int32(TILE_TOKENS + 1) - Int32(1) - tidx] = before  # stash at the top of cu_s, backwards, past cu
            mask = self.tile_expert_ranks(topk, tile, tidx, lane, warp, rank, scratch, expert_rank, my_slots)
            present_ballot = cute.arch.vote_ballot_sync(mask != Int32(0))
            present_before = cute.arch.popc(present_ballot & ((Int32(1) << lane) - Int32(1)))
            if lane == Int32(0):
                scratch[Int32(self.e_local * self.num_warps) + warp] = cute.arch.popc(present_ballot)
            cute.arch.sync_threads()
            token = tile * Int32(TILE_TOKENS) + tidx
            # pool row of this token = present tokens before this tile + before this warp + before this lane
            pool_row = cu_s[Int32(TILE_TOKENS + 1) - Int32(1) - Int32(self.e_local)] + present_before
            for w in cutlass.range_constexpr(self.num_warps):
                if Int32(w) < warp:
                    pool_row = pool_row + scratch[Int32(self.e_local * self.num_warps) + Int32(w)]
            for k in cutlass.range_constexpr(self.topk_k):
                if ((mask >> k) & Int32(1)) == Int32(1):
                    f = token * Int32(self.topk_k) + Int32(k)
                    e = topk[f] - rank * Int32(self.e_local)
                    row = cu_s[e] + cu_s[Int32(TILE_TOKENS + 1) - Int32(1) - e] + expert_rank[k]
                    for w in cutlass.range_constexpr(self.num_warps):
                        if Int32(w) < warp:
                            row = row + scratch[e * Int32(self.num_warps) + Int32(w)]
                    s_reverse[f] = row
                    s_scatter[row] = f
                    a_idx[row] = pool_row
            cute.arch.sync_threads()
            tile = tile + Int32(self.num_ctas)


def make_routing_metadata(topk, rank, W, T_local, K, E_local, max_rows):
    """(build, meta) for topk = the gathered top-k (W * T_local * K,) int32 (rewritten per call): build() launches
    both kernels; meta = (cu (E_local + 1), s_reverse (W * T_local * K), s_scatter (max_rows), a_idx (max_rows),
    first_src_row (1)), all int32, filled by build()."""
    dev = topk.device
    assert topk.dtype == torch.int32 and topk.numel() == W * T_local * K
    assert T_local % TILE_TOKENS == 0, "tiles must not straddle source ranks"
    n_tiles = W * T_local // TILE_TOKENS
    i32 = lambda *shape: torch.empty(*shape, dtype=torch.int32, device=dev)
    counts, present = i32(n_tiles, E_local), i32(n_tiles)
    meta = SimpleNamespace(cu=i32(E_local + 1), s_reverse=i32(W * T_local * K), s_scatter=i32(max_rows), a_idx=i32(max_rows), first_src_row=i32(1))
    num_ctas = min(n_tiles, torch.cuda.get_device_properties(dev).multi_processor_count)
    kernel = RoutingMetadata(K, E_local, num_ctas)
    v = lambda: make_fake_tensor(Int32, (cute.sym_int(),), leading_dim=0, divisibility=1)
    m = lambda: make_fake_tensor(Int32, (cute.sym_int(), cute.sym_int()), leading_dim=1, divisibility=1)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    fn = cute.compile(kernel, v(), m(), v(), v(), v(), v(), v(), v(), Int32(rank), Int32(T_local), Int32(W), stream, options="--enable-tvm-ffi")
    build = lambda: fn(topk, counts, present, meta.cu, meta.s_reverse, meta.s_scatter, meta.a_idx, meta.first_src_row, rank, T_local, W)
    return build, meta
