"""Run EP dispatch baselines through one interface (sibling of ag_gemm/bench_baselines.py, same protocol).

Every arm produces sonic's rank-dedup receive layout: a (W * T_local, H) buffer packed by source rank ascending,
hit tokens ascending inside each block, so its rows compare bitwise to sonic's. Shape from --model (MODEL_CONFIGS:
H, I, E, top-k), bf16, uniform routing (distinct top-k per token, seeded per rank); routing metadata is built once
outside the timed region; W comes from the launch. Cells match results/2026-09-21/dispatch.md.

  sonic       sonic-moe PR54 rank-dedup dispatch, triton (rank_dedup_dispatch_triton): the reference
  tk          ThunderKittens dispatch-only kernel (dispatch/dispatch_tk/dispatch_tk.cu), fed the same
              packed-by-source slot table; SMs via --sms (green-context partition)
  deepep      DeepEP V2 elastic dispatch (~/DeepEP): dedup by construction, 1D bulk-copy push on --sms SMs (its own
              num_sms; DeepEP's bandwidth-model default when --sms is unset)
  quack_inter dispatch/pull.py: MegaMoE's pull loop at 1 CTA/SM on --sms over the device walk (dispatch/walk.py) in GEMM1's
              tile order (expert first, shifted source rank inside; one flag per M cluster), the walk build in the
              arm; the SM's smem full of rows in flight (16 warps on B300), the inter-SM leg
  quack_intra the same pull with 4 warps / 56 KB stage per SM: what the transfer warps look like inside the GEMM

Each factory returns a Kernel: run / free plus read_out (the output to compare: a tensor or a tuple, e.g. the pool
rows and the flags), zero (reset it, so a stale result cannot pass) and ref (sonic's rows -> what read_out must
equal), so --ref_check / --r2r_check work generically. The chosen baselines are timed together by bench_all (the
fixed protocol above it: probe-sized warm-up at each arm's own load, rotated blocks, per-block stability flag with
one rerun), with sonic's dispatch as the contention canary and a barrier-only arm as the per-call floor.
Copy-engine arms: bench_dispatch_ce.py.

Usage:
    PYTHONPATH=~/quack:~/sonic-moe-pr54:~/DeepEP torchrun --nproc_per_node=8 --standalone bench_dispatch.py quack_inter tk deepep --ref_check
    ... bench_dispatch.py quack_inter --model kimi_k2 --r2r_check
    ... bench_dispatch.py quack_inter deepep --sms 24
"""
import os

os.environ.setdefault("TRITON_CACHE_AUTOTUNING", "1")  # sonic's triton autotune choices persist across launches

import argparse
import contextlib
import copy
import glob
import importlib.util
import statistics
import sys
import time
from collections import namedtuple
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # the kernels package root
from kernels.misc import nccl_env  # noqa: E402, F401  (NCCL channel floor for every reference)
from cuda.bindings import driver
from sonicmoe.distributed_utils import DispatchMode, SymmMemManager
from sonicmoe.functional.distributed import build_rank_dedup_a_idx, compute_dispatch_metadata, rank_dedup_dispatch_triton
from sonicmoe.functional.distributed.collectives import _check
from sonicmoe.functional.ep import _ag_routing_decision, _build_consumer_metadata

ModelConfig = namedtuple("ModelConfig", ["H", "I", "E", "topk"])  # hidden, MoE intermediate, experts, top-k
MODEL_CONFIGS = {
    "dsv3": ModelConfig(7168, 2048, 256, 8),
    "dsv3_e128": ModelConfig(7168, 2048, 128, 8),  # dsv3 shapes, half the experts: varies experts/rank alone
    "kimi_k2": ModelConfig(7168, 2048, 384, 8),  # K2 / K2.5 / K2.7 share the MoE shape (+1 shared expert)
    "dsv4_pro": ModelConfig(7168, 3072, 384, 6),  # DeepSeek-V4-Pro (+1 shared expert)
    "glm_5_2": ModelConfig(6144, 2048, 256, 8),  # GLM-5.2 (+1 shared expert)
    "qwen3_5_397b": ModelConfig(4096, 1024, 512, 10),  # Qwen3.5-397B-A17B (+1 shared expert of 1024)
    "qwen3_235b": ModelConfig(4096, 1536, 128, 8),
    "qwen3_next_80b": ModelConfig(2048, 512, 512, 10),  # Qwen3-Next-80B-A3B (+1 shared expert): the push-bound end
    "mixtral_8x22b": ModelConfig(6144, 16384, 8, 2),
    "olmoe": ModelConfig(2048, 1024, 64, 8),
}
_NOOP = lambda: None  # noqa: E731

# run/free: timed call + cleanup. read_out: () -> the output to compare, a tensor or a tuple of tensors. zero: reset
# it. ref: sonic's receive rows (n_recv, H) -> what read_out must equal (None: timing-only arm). label: suffix for
# the report line. halves: for an arm made of two parts, the list of (start, end, start, end) events run() records
# per call.
Kernel = namedtuple("Kernel", ["run", "free", "read_out", "zero", "ref", "label", "halves"], defaults=[None, None, None, "", None])


class SMStream:
    """Torch stream confined to ~num_sms SMs via a green context; the whole GPU if num_sms >= total.
    Copied from sonic-moe PR #54 benchmarks/moe_sm_tflops.py (SMStream); the count rounds up to the
    device's co-scheduling alignment, so read .sms for what was granted."""

    def __init__(self, device_index: int, num_sms: int):
        _check(driver.cuInit(0))
        self.device_index = device_index
        self.dev = _check(driver.cuDeviceGet(device_index))
        res = _check(driver.cuDeviceGetDevResource(self.dev, driver.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM))
        self.total = int(res.sm.smCount)
        self.rest = None  # SMs the split left over (see complement)
        if num_sms >= self.total:
            self.sms, self.stream, self._gctx = self.total, None, None
            return
        align = int(res.sm.smCoscheduledAlignment) or 8
        n = max(align, ((num_sms + align - 1) // align) * align)
        groups, _nb, self.rest = _check(driver.cuDevSmResourceSplitByCount(1, res, 0, n))
        self.sms, self.stream, self._gctx = self._green(groups[0])

    def _green(self, part):
        desc = _check(driver.cuDevResourceGenerateDesc([part], 1))
        gctx = _check(driver.cuGreenCtxCreate(desc, self.dev, driver.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM))
        cu_stream = _check(driver.cuGreenCtxStreamCreate(gctx, driver.CUstream_flags.CU_STREAM_NON_BLOCKING, 0))
        return int(part.sm.smCount), torch.cuda.ExternalStream(int(cu_stream), device=torch.device("cuda", self.device_index)), gctx

    def complement(self):
        """Stream on the SMs this partition left over, disjoint from it (two kernels co-run, one on each)."""
        other = copy.copy(self)
        other.sms, other.stream, other._gctx = self._green(self.rest)
        other.rest = None
        return other

    def ctx(self):
        return torch.cuda.stream(self.stream) if self.stream is not None else contextlib.nullcontext()


# The protocol (fixed, not flags: two runs must be comparable). Per arm and block: WARM_S of warm-up at the arm's
# own load right before a WINDOW_S timed block, REPS blocks with the arm order rotated per block, the block timed in
# SAMPLES chunks. An arm is unstable when a block median is off the arm's median by more than SPREAD, or a block's
# p90 / median exceeds TAIL (co-tenant burst or straggler rank).
WARM_S, WINDOW_S, REPS, SAMPLES = 0.2, 0.5, 5, 100
SPREAD, TAIL = 0.03, 1.10


def bench_all(fns, hdl):
    """sonic's EP timing (benchmarks/distributed/bench-ep-nvlink.py bench_fn) under the protocol above. Every call
    is {symm-mem barrier, fn}. A 20-call probe per arm (host-timed, rank-MAX'd, after 2 untimed calls) sizes its
    warm-up and window as fixed call counts equal on every rank (a time-based loop would desync the barriers). Returns
    {name: [[chunk ms, ...] per block]} per rank; see arm_stats."""
    counts = {}
    for k, fn in fns.items():
        for _ in range(2):  # an arm's first calls may compile or autotune (triton): keep them out of the probe
            hdl.barrier()
            fn()
        dist.barrier()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            hdl.barrier()
            fn()
        torch.cuda.synchronize()
        per_call = torch.tensor([(time.perf_counter() - t0) / 20], device="cuda")
        dist.all_reduce(per_call, op=dist.ReduceOp.MAX)
        s = per_call.item()
        counts[k] = (max(10, int(WARM_S / s)), max(20, int(WINDOW_S / s)))
    ms = {k: [] for k in fns}
    names = list(fns)
    for rep in range(REPS):
        shift = rep % len(names)  # rotate the order per block: no arm always inherits the same predecessor's clocks
        for k in names[shift:] + names[:shift]:
            fn, (warm, iters) = fns[k], counts[k]
            for _ in range(warm):
                hdl.barrier()
                fn()
            dist.barrier()
            chunk = max(1, iters // SAMPLES)
            events = []
            for i in range(iters):
                if i % chunk == 0:
                    events.append(torch.cuda.Event(enable_timing=True))
                    events[-1].record()
                hdl.barrier()
                fn()
            events.append(torch.cuda.Event(enable_timing=True))
            events[-1].record()
            torch.cuda.synchronize()
            bounds = list(range(0, iters, chunk)) + [iters]
            ms[k].append([events[j].elapsed_time(events[j + 1]) / (bounds[j + 1] - bounds[j]) for j in range(len(bounds) - 1)])
    return ms


def arm_stats(blocks):
    """(median, low, high, unstable) of an arm: the median of its block medians and their range, and whether a
    block median is off by more than SPREAD or a block's p90 / median exceeds TAIL."""
    meds = [statistics.median(b) for b in blocks]
    med = statistics.median(meds)
    tails = [sorted(b)[int(0.9 * (len(b) - 1))] / statistics.median(b) for b in blocks]
    unstable = max(abs(m - med) for m in meds) > SPREAD * med or max(tails) > TAIL
    return med, min(meds), max(meds), unstable


def _tup(v):
    return v if isinstance(v, tuple) else (v,)


def _run_synced(kern):
    """Zero, barrier, run, sync, barrier: one fresh output, no rank still pushing into a buffer being zeroed."""
    kern.zero()
    torch.cuda.synchronize()
    dist.barrier()
    kern.run()
    torch.cuda.synchronize()
    dist.barrier()


def _compare(a, b):
    """(bitwise equal, max |delta|) over matching tuples of tensors."""
    pairs = list(zip(_tup(a), _tup(b)))
    return all(torch.equal(x, y) for x, y in pairs), max((x.float() - y.float()).abs().max().item() for x, y in pairs)


def ref_check(kern, rows):
    """(bitwise equal, max |delta|) of kern's fresh output vs its reference from sonic's rows."""
    _run_synced(kern)
    return _compare(kern.read_out(), kern.ref(rows))


def r2r_check(kern):
    """Run twice; (bit-identical, max |delta|) of the two outputs."""
    _run_synced(kern)
    first = tuple(t.clone() for t in _tup(kern.read_out()))
    _run_synced(kern)
    return _compare(kern.read_out(), first)


def n_recv(meta, rank):
    """Rows this rank receives (sonic's dedup: each hit token once per receiving rank)."""
    return int(meta["pair_count"][:, rank].sum())


# Every factory: (model, x, topk_idx_g, meta, ws, sm_stream, rank, W, T_local, args) -> Kernel. x: this rank's
# (T_local, H) tokens; topk_idx_g: every rank's routing (W, T_local, K); meta: sonic's metadata bundle; ws: the
# symm-mem workspace (x_symm, x_peer_bufs, ep_group, x_hdl); sm_stream: the --sms partition.
def _sonic(model, x, topk_idx_g, meta, ws, sm_stream, rank, W, T_local, args):
    """sonic's rank-dedup dispatch (triton) into recv (W * T_local, H), the reference: its rows are the ref itself.
    The call is sonic-moe PR #54 sonicmoe/functional/ep.py:95 _do_dispatch's RANK_DEDUP branch, without the
    backward / autograd, AG / A2A and inter-node paths."""
    recv = torch.empty(W * T_local, model.H, dtype=torch.bfloat16, device=x.device)
    n = n_recv(meta, rank)

    def run():
        rank_dedup_dispatch_triton(
            x_symm=ws.x_symm,
            dst_rank_flat=meta["dst_rank_flat"],
            pair_present_mask=meta["pair_present_mask"],
            rank_dedup_recv_pos=meta["rank_dedup_recv_pos"],
            recv_packed=recv,
            K=model.topk,
            group=ws.ep_group,
            peer_bufs=ws.x_peer_bufs,
            my_rank=rank,
        )

    return Kernel(run, _NOOP, lambda: recv[:n], recv.zero_, lambda rows: rows)


def _tk(model, x, topk_idx_g, meta, ws, sm_stream, rank, W, T_local, args):
    """ThunderKittens' dispatch-only kernel (dispatch/dispatch_tk/dispatch_tk.cu, H hardcoded; NUM_DEVICES is
    compile-time, so `make ND={W}` there builds _C_nd{W}), fed sonic's packed-by-source dedup slot table, x staged
    once into TK's IPC-shared tensor. The receiver walks sources rank+1, rank+2, ..., self, so no source serves two
    receivers at once (ascending measured 3.1x slower at W=8, results/2026-09-14/dispatch_gemm1.md)."""
    H, E_local, dev = model.H, model.E // W, x.device
    assert H == 7168, "dispatch_tk.cu hardcodes H=7168"
    d = Path(__file__).parent / "dispatch/dispatch_tk"
    cand = glob.glob(str(d / f"_C_nd{W}.*.so"))
    assert cand, f"no TK dispatch module for world={W}; run `make ND={W}` in {d}"
    spec = importlib.util.spec_from_file_location("_C", cand[0])
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    pre_tokens = m.TKParallelTensor((T_local, H), dtype=torch.bfloat16, local_rank=rank, local_world_size=W, multicast=False)
    pre_tokens.data_.copy_(x)
    src_dev, src_tok = [], []
    for q in range(W):  # block q = tokens of rank q that hit me, ascending
        tokens = (topk_idx_g[q] // E_local == rank).any(-1).nonzero().flatten().to(torch.int32)
        src_dev.append(torch.full_like(tokens, q))
        src_tok.append(tokens)
    src_dev, src_tok = torch.cat(src_dev), torch.cat(src_tok)
    n = int(src_tok.numel())
    assert n == n_recv(meta, rank)
    n_padded = (n + 15) // 16 * 16
    pull = torch.full((n_padded, 2), -1, dtype=torch.int32, device=dev)
    pull[:n, 0], pull[:n, 1] = src_dev, src_tok
    post_tokens = torch.empty(W * T_local, H, dtype=torch.bfloat16, device=dev)
    barrier = m.TKParallelTensor((2, 1024, 1024), dtype=torch.int, local_rank=rank, local_world_size=W, multicast=True)  # TK's 2 MB minimum
    barrier.data_.zero_()
    dist.barrier()
    torch.cuda.synchronize()
    # 16-slot blocks grouped by the source of their first slot, visited in source-ring order.
    block_src = src_dev[torch.arange(0, n, 16, device=dev)].tolist()
    ring = [(rank + 1 + i) % W for i in range(W)]
    block_order = torch.tensor(
        [b for q in ring for b in range(n_padded // 16) if block_src[b] == q], dtype=torch.int32, device=dev
    )
    assert block_order.numel() == n_padded // 16

    def run(_keep=(pre_tokens, barrier, m)):
        m.dispatch(pre_tokens, post_tokens, pull, block_order, barrier, n_padded)

    return Kernel(run, _NOOP, lambda: post_tokens[:n], post_tokens.zero_, lambda rows: rows)


def _deepep(model, x, topk_idx_g, meta, ws, sm_stream, rank, W, T_local, args):
    """DeepEP V2 elastic dispatch (~/DeepEP): dedup by construction, 1D bulk-copy push on num_sms SMs = the --sms
    partition's count (DeepEP's bandwidth-model default when --sms is unset). One full dispatch at setup computes
    the routing metadata and yields the handle; the timed call passes it back, so only the transport runs. Rows
    come out in DeepEP's order: ref maps sonic's rows to it through recv_src_metadata (source rank, token)."""
    sys.path.insert(0, os.path.expanduser("~/DeepEP"))
    import deep_ep

    E, E_local, dev = model.E, model.E // W, x.device
    buffer = deep_ep.ElasticBuffer(ws.ep_group, num_max_tokens_per_rank=T_local, hidden=model.H, explicitly_destroy=True)
    num_sms = sm_stream.sms if args.sms else buffer.get_theoretical_num_sms(E, model.topk)
    idx64 = topk_idx_g[rank].to(deep_ep.topk_idx_t)
    weights = torch.ones(idx64.shape, dtype=torch.float32, device=dev)
    common = dict(num_sms=num_sms, num_max_tokens_per_rank=T_local, num_experts=E, expert_alignment=1)
    out = {}
    out["recv_x"], _, _, handle, _ = buffer.dispatch(x=x, topk_idx=idx64, topk_weights=weights, do_cpu_sync=True, **common)
    torch.cuda.synchronize()
    n = int(handle.num_recv_tokens)
    gid = handle.recv_src_metadata[:n].reshape(n, -1)[:, 0].to(torch.int64)  # src_rank * T_local + src_token
    sonic_row = torch.full((W, T_local), -1, dtype=torch.int64, device=dev)
    for q in range(W):
        tokens = (topk_idx_g[q] // E_local == rank).any(-1).nonzero().flatten()
        sonic_row[q, tokens] = int(meta["pair_offset"][q, rank]) + torch.arange(tokens.numel(), device=dev)
    rows_of = sonic_row[gid // T_local, gid % T_local]
    assert bool((rows_of >= 0).all()) and n == n_recv(meta, rank)

    def run():
        out["recv_x"] = buffer.dispatch(x=x, handle=handle, do_cpu_sync=False, **common)[0]

    return Kernel(
        run, buffer.destroy, lambda: out["recv_x"][:n], lambda: out["recv_x"].zero_(), lambda rows: rows[rows_of],
        label=f" (deepep_sms={num_sms})",
    )


def _quack(model, x, topk_idx_g, meta, ws, sm_stream, rank, W, T_local, args, mode):
    """Our pull (dispatch/pull.py) over the device walk, mode "inter" / "intra" (see the module header),
    on --sms; the walk build (per routing, whole GPU) is part of the arm. Output = the pool rows and the flags, each
    at its cluster's row count (walk.targets)."""
    from dispatch.pull import make_dispatch_pull
    from dispatch.walk import make_dispatch_walk

    cu = meta["expert_frequency_offset"]
    a_idx = meta["x_idx_expanded_remap_for_rank_dedup"][: int(cu[model.E // W])]
    build, walk = make_dispatch_walk(
        topk_idx_g, meta["s_scatter_idx"], meta["recv_pos"], meta["pair_offset"], rank, a_idx, cu, ws.x_peer_bufs
    )
    run_pull, pool = make_dispatch_pull(ws.x_peer_bufs, rank, mode, sm_stream.sms)
    pull = run_pull(walk)
    n = n_recv(meta, rank)
    full = torch.cuda.Stream(device=x.device)  # the build is metadata work on the whole GPU; only the pull is on --sms

    def run():
        cur = torch.cuda.current_stream()
        full.wait_stream(cur)
        with torch.cuda.stream(full):
            build()
        cur.wait_stream(full)
        pull()

    return Kernel(
        run, _NOOP, lambda: (pool[:n], walk.flags), lambda: (pool.zero_(), walk.flags.zero_()),
        lambda rows: (rows, walk.targets),
    )


BASELINES = {
    "sonic": _sonic,
    "tk": _tk,
    "deepep": _deepep,
    "quack_inter": lambda *a: _quack(*a, "inter"),
    "quack_intra": lambda *a: _quack(*a, "intra"),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("baselines", nargs="+", choices=list(BASELINES))
    p.add_argument("--model", choices=list(MODEL_CONFIGS), default="dsv3", help="shape: H, I, E, top-k (MODEL_CONFIGS)")
    p.add_argument("--t-local", type=int, default=8192)
    p.add_argument(
        "--sms", type=int, default=0,
        help="SMs the transport gets: a green-context partition for the timed loop and DeepEP's num_sms "
        "(0 = the whole GPU; DeepEP's own default)",
    )
    p.add_argument("--ref_check", action="store_true", help="each arm's fresh output (zeroed first) vs sonic's rows")
    p.add_argument("--r2r_check", action="store_true", help="run each arm twice; bit-identity of the two outputs")
    args = p.parse_args()

    # Process group, shape, symm-mem workspace, routing and metadata (built once, outside the timed region).
    rank, W = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    dev = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(dev)
    dist.init_process_group("nccl", device_id=dev)
    model = MODEL_CONFIGS[args.model]
    H, _, E, topk = model
    assert E % W == 0, f"{args.model}: E={E} not divisible by W={W}"
    T_local, E_local = args.t_local, E // W
    sm_stream = SMStream(dev.index, args.sms if args.sms else 1 << 30)
    ws = SymmMemManager(None, dev)._get_or_alloc(
        T_local, H, topk, E_local, torch.bfloat16, DispatchMode.RANK_DEDUP_DISPATCH_TRITON
    )
    g = torch.Generator(device=dev).manual_seed(123 + rank)
    x = torch.randn(T_local, H, generator=g, device=dev, dtype=torch.bfloat16)
    # Uniform routing with distinct experts per token (real top-k never repeats an expert; DeepEP faults on repeats).
    topk_idx = torch.rand(T_local, E, generator=g, device=dev).topk(topk, dim=-1).indices.to(torch.int32)
    ws.x_symm.copy_(x)
    topk_idx_g = _ag_routing_decision(ws, topk_idx)
    ws.x_hdl.barrier()
    # Sonic's RANK_DEDUP dispatch metadata (PR #54 ep.py _moe_ep_forward_inner's prologue, the dispatch part only):
    # per-token destination ranks and dedup blocks (dst_rank_flat, pair_count / pair_offset, pair_present_mask,
    # rank_dedup_recv_pos), then the expert-grouped receive positions and GEMM1's gather index over them.
    meta = compute_dispatch_metadata(topk_idx_g, my_rank=rank, E_local=E_local, emit_combine=False)
    consumer = _build_consumer_metadata(
        expert_indices=meta["expert_local_padded"], token_indices=meta["a2a_token_indices"], TK=W * T_local * topk,
        E_local=E_local, s_reverse_idx_symm=ws.s_rev_symm,
    )
    meta["recv_pos"] = consumer["s_reverse_local"]
    meta["s_scatter_idx"] = consumer["s_scatter_idx"]
    meta["expert_frequency_offset"] = consumer["expert_frequency_offset"]
    meta["x_idx_expanded_remap_for_rank_dedup"] = build_rank_dedup_a_idx(
        dst_rank_flat=meta["dst_rank_flat"], s_reverse_local=consumer["s_reverse_local"],
        rank_dedup_recv_pos=meta["rank_dedup_recv_pos"], my_rank=rank, out=ws.x_idx_expanded_remap_for_rank_dedup_buf,
    )
    shared = (model, x, topk_idx_g, meta, ws, sm_stream, rank, W, T_local, args)

    ref = _sonic(*shared)
    ref.run()
    torch.cuda.synchronize()
    rows = ref.read_out().clone()  # sonic's receive rows: the reference for every arm
    kerns = {name: ref if name == "sonic" else BASELINES[name](*shared) for name in args.baselines}

    checks = {name: {} for name in kerns}
    for name, kern in kerns.items():
        if args.ref_check:
            checks[name]["ref"] = ref_check(kern, rows)
        if args.r2r_check:
            checks[name]["r2r"] = r2r_check(kern)

    arms = {name: kern.run for name, kern in kerns.items()}
    if "sonic" not in arms:
        arms["sonic"] = ref.run  # timed next to the baselines: the contention canary
    arms["barrier"] = _NOOP  # the floor inside every call: the symm-mem barrier plus the enqueue
    with sm_stream.ctx():
        ms = bench_all(arms, ws.x_hdl)
        # Unstable arms get one rerun next to the canary; a stable rerun replaces them. The rerun set is the union
        # over ranks: every rank must call the same collectives.
        unstable = [None] * W
        dist.all_gather_object(unstable, [k for k, v in ms.items() if arm_stats(v)[3]])
        rerun = [k for k in arms if any(k in u for u in unstable)]
        if rerun:
            again = bench_all({k: arms[k] for k in dict.fromkeys(rerun + ["sonic"])}, ws.x_hdl)
            ms.update({k: v for k, v in again.items() if k in rerun and not arm_stats(v)[3]})

    hits = (topk_idx_g // E_local == rank).any(-1)  # (W, T_local): token reaches my rank
    wire_rows = int(hits.sum() - hits[rank].sum())
    nbytes = wire_rows * H * 2
    stats = {k: arm_stats(v) for k, v in ms.items()}
    gathered = [None] * W
    dist.all_gather_object(gathered, (stats, checks))
    if rank == 0:
        print(
            f"<bf16 EP dispatch {args.model} (H={H} E={E} top-{topk}) | W={W} T_local={T_local} "
            f"sms={sm_stream.sms}/{sm_stream.total} | wire {wire_rows} rows ({nbytes / 1e6:.0f} MB) | "
            f"warm {WARM_S * 1e3:.0f} ms, {REPS} x {WINDOW_S * 1e3:.0f} ms blocks, rerun={rerun}>"
        )
        for k in ms:
            med, lo, hi, _ = max((g[k] for g, _ in gathered), key=lambda t: t[0])  # slowest rank sets the collective
            mean = sum(g[k][0] for g, _ in gathered) / W  # cross-rank mean of the medians (sonic's number)
            label = kerns[k].label if k in kerns else ""
            rate = "" if k == "barrier" else f"{nbytes / med / 1e6:.0f} GB/s"
            flag = "  UNSTABLE" if any(g[k][3] for g, _ in gathered) else ""
            print(
                f"  {k + label + ':':<28} {med * 1e3:7.0f} us  [{lo * 1e3:.0f}-{hi * 1e3:.0f}]  {rate:>10}  "
                f"rank-mean {mean * 1e3:.0f} us{flag}"
            )
        for name in kerns:
            for what in checks[name]:
                ok = all(c[name][what][0] for _, c in gathered)
                delta = max(c[name][what][1] for _, c in gathered)
                print(f"  {name} {what}: {'bitwise' if ok else 'MISMATCH'} max|delta|={delta:.3e}")
    for kern in kerns.values():
        kern.free()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
