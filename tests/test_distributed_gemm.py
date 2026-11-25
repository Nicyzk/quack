# Copyright (c) 2026, QuACK team.
"""Multi-rank correctness tests for AllGatherRunner (quack/distributed/).

Runs under torchrun (one rank per GPU); the pytest entry point spawns
torchrun as a subprocess on the available GPUs.

    torchrun --nproc_per_node=4 tests/test_distributed_gemm.py
    pytest tests/test_distributed_gemm.py -x
"""

import os
import subprocess
import sys

import pytest
import torch

NUM_ITERS = 6  # exercises epoch monotonicity + double-buffer reuse


def _ag_gemm(runner, a_shard, b, d=None):
    """Plain AG+GEMM (D = A_full @ B^T) through the prepare() context."""
    from quack.distributed import AllGather
    from quack.gemm import gemm as quack_gemm

    if d is None:
        d = torch.empty(runner.m_total, b.shape[0], dtype=runner.dtype, device=runner.device)
    with runner.prepare(a_shard) as (a_full, ag_args):
        # world_size == 1 yields no flags: a plain GEMM
        dist_op = AllGather() if ag_args is not None else None
        quack_gemm(a_full, b, d, None, None, 128, 256, 2, 1, dist_op=dist_op, dist_op_args=ag_args)
    return d


def _run_rank():
    import torch.distributed as dist

    from quack.distributed import AllGather, AllGatherRunner

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group("nccl")
    device = torch.device("cuda", rank % torch.cuda.device_count())

    failures = []

    def check_plain_result(label, runner, a_shard, b, d):
        """Check both the GEMM result and the runner's last gathered buffer."""
        a_full_ref = torch.empty(runner.m_total, b.shape[1], dtype=b.dtype, device=device)
        dist.all_gather_into_tensor(a_full_ref, a_shard)
        ref = a_full_ref.float() @ b.float().T
        baseline = ref.to(b.dtype).float()
        err = (d.float() - ref).abs().max().item()
        tol = (baseline - ref).abs().max().item() + 1e-4
        gathered_err = (runner.gathered_a() - a_full_ref).abs().max().item()
        if err > 2 * tol or gathered_err != 0:
            failures.append(f"rank{rank} {label}: err={err} tol={tol} gathered_err={gathered_err}")

    for dtype, shard_m, n, k, delay_comm, chunks in [
        (torch.bfloat16, 1024, 2048, 4096, False, 1),
        (torch.bfloat16, 512, 1024, 8192, True, 1),  # late sends: gate must hold
        (torch.bfloat16, 1024, 2048, 4096, False, 4),  # chunked sends
        (torch.bfloat16, 2048, 1024, 4096, True, 8),  # chunked + late sends
        (torch.float16, 2048, 3072, 2048, False, 2),
    ]:
        torch.manual_seed(1234)  # same B on all ranks
        b = (torch.randn(n, k, dtype=dtype, device=device) / 8).contiguous()
        ag = AllGatherRunner(shard_m, k, dtype, device=device, arrival_chunks=chunks)
        m_total = shard_m * world_size
        for it in range(NUM_ITERS):
            torch.manual_seed(1000 * it + rank)  # per-rank, per-iter shard
            a_shard = (torch.randn(shard_m, k, dtype=dtype, device=device) / 8).contiguous()

            # Reference: real all-gather + fp32 matmul
            a_full_ref = torch.empty(m_total, k, dtype=dtype, device=device)
            dist.all_gather_into_tensor(a_full_ref, a_shard)
            ref = a_full_ref.float() @ b.float().T
            baseline = (a_full_ref.float() @ b.float().T).to(dtype).float()

            if delay_comm:
                # Hold the TRANSPORT stream back so data arrival (and its flag
                # writes) is provably late and the in-kernel gate must wait.
                # (An earlier version slept copy_stream, which only delays the
                # post-GEMM reuse barrier — it never stressed the gate.)
                with torch.cuda.stream(ag.push_stream):
                    torch.cuda._sleep(int(50e6))  # ~25 ms at 2 GHz

            d = _ag_gemm(ag, a_shard, b)
            torch.cuda.synchronize(device)

            err = (d.float() - ref).abs().max().item()
            tol = (baseline - ref).abs().max().item() + 1e-4
            gathered_err = (ag.gathered_a() - a_full_ref).abs().max().item()
            if err > 2 * tol or gathered_err != 0:
                failures.append(
                    f"rank{rank} dtype={dtype} shard_m={shard_m} n={n} k={k} chunks={chunks} "
                    f"delay={delay_comm} iter={it}: err={err} tol={tol} "
                    f"gathered_err={gathered_err}"
                )
    # --- Abstraction check: AllGather + GEMM+SiLU epilogue via the
    # runner.prepare() CONTEXT — proves any epilogue-mod GEMM overlaps by
    # calling it inline in the with-body with the AllGather operator (same
    # route serves rope etc.).
    from quack.epilogue.library import linear_act_mod

    dtype, shard_m, n, k = torch.bfloat16, 1024, 2048, 4096
    torch.manual_seed(4321)
    b = (torch.randn(n, k, dtype=dtype, device=device) / 8).contiguous()
    runner = AllGatherRunner(shard_m, k, dtype, device=device)
    m_total = shard_m * world_size
    postact = torch.empty(m_total, n, dtype=dtype, device=device)

    for it in range(3):
        torch.manual_seed(77 * it + rank)
        a_shard = (torch.randn(shard_m, k, dtype=dtype, device=device) / 8).contiguous()
        a_full_ref = torch.empty(m_total, k, dtype=dtype, device=device)
        dist.all_gather_into_tensor(a_full_ref, a_shard)
        acc = a_full_ref.float() @ b.float().T
        ref = torch.nn.functional.silu(acc)
        baseline = torch.nn.functional.silu(acc.to(dtype).float())
        with runner.prepare(a_shard) as (a_full, ag_args):
            linear_act_mod(
                "silu", gated=False, has_c=False, has_rowvec=False, has_colvec=False
            ).gemm(
                a_full,
                b,
                None,
                None,
                epi_args=dict(mAuxOut=postact),
                tile_M=128,
                tile_N=256,
                cluster_M=2,
                cluster_N=1,
                dist_op=AllGather(),
                dist_op_args=ag_args,
            )
        torch.cuda.synchronize(device)
        err = (postact.float() - ref).abs().max().item()
        tol = (baseline - ref).abs().max().item() + 1e-4
        if err > 2 * tol:
            failures.append(f"rank{rank} AG+gemm_act(silu) iter={it}: err={err} tol={tol}")

    # --- The same route, the TP block's epilogues: gated SwiGLU (gate/up interleaved along N), RoPE on
    # the QKV output (an interleaved cos/sin tile operand), a quantized D (mxfp8 + SFD, SM100) ---
    from quack.cute_dsl_utils import get_device_capacity
    from quack.epilogue.library import rope_epi
    from quack.gemm import gemm as quack_gemm

    F = torch.nn.functional
    sm100 = get_device_capacity(device)[0] in (10, 11)
    torch.manual_seed(4322)
    b = (torch.randn(n, k, dtype=dtype, device=device) / 8).contiguous()
    table = torch.randn(m_total, n, dtype=dtype, device=device)
    postact = torch.empty(m_total, n // 2, dtype=dtype, device=device)
    d = torch.empty(m_total, n, dtype=dtype, device=device)
    if sm100:
        d_q = torch.empty(m_total, n, dtype=torch.float8_e4m3fn, device=device)
        sfd = torch.empty(
            1, m_total // 128, n // 128, 32, 4, 4, dtype=torch.float8_e8m0fnu, device=device
        )
    swiglu = linear_act_mod("swiglu", gated=True, has_c=False, has_rowvec=False, has_colvec=False)
    kw = dict(tile_M=128, tile_N=256, cluster_M=2, cluster_N=1, dist_op=AllGather())

    def check(label, got, ref, base):
        err = (got.float() - ref).abs().max().item()
        tol = (base - ref).abs().max().item() + 1e-4
        if err > 2 * tol:
            failures.append(f"rank{rank} {label}: err={err} tol={tol}")

    for it in range(3):
        torch.manual_seed(78 * it + rank)
        a_shard = (torch.randn(shard_m, k, dtype=dtype, device=device) / 8).contiguous()
        a_full_ref = torch.empty(m_total, k, dtype=dtype, device=device)
        dist.all_gather_into_tensor(a_full_ref, a_shard)
        acc = a_full_ref.float() @ b.float().T
        accq = acc.to(dtype).float()  # the baseline's one rounding
        with runner.prepare(a_shard) as (a_full, ag_args):
            swiglu.gemm(
                a_full, b, None, None, epi_args=dict(mAuxOut=postact), dist_op_args=ag_args, **kw
            )
        torch.cuda.synchronize(device)
        ref = F.silu(acc[:, 0::2]) * acc[:, 1::2]
        check(f"AG+swiglu iter={it}", postact, ref, F.silu(accq[:, 0::2]) * accq[:, 1::2])
        with runner.prepare(a_shard) as (a_full, ag_args):
            rope_epi.gemm(
                a_full, b, d, None, epi_args=dict(table=table), dist_op_args=ag_args, **kw
            )
        torch.cuda.synchronize(device)
        cos, sin = table.float()[:, 0::2], table.float()[:, 1::2]

        def rope(x):
            x1, x2 = x[:, 0::2], x[:, 1::2]
            return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).reshape(m_total, n)

        check(f"AG+rope iter={it}", d, rope(acc), rope(accq))
        if sm100:
            from quack.blockscaled.quantize import unpack_scale_blocked_to_2d

            with runner.prepare(a_shard) as (a_full, ag_args):
                # the 2-CTA tile is 256 rows: a 64-row CTA half would split SF vectors across threads
                quack_gemm(
                    a_full,
                    b,
                    d_q,
                    None,
                    None,
                    256,
                    256,
                    2,
                    1,
                    dist_op=AllGather(),
                    dist_op_args=ag_args,
                    SFD=sfd,
                )
            torch.cuda.synchronize(device)
            scale = unpack_scale_blocked_to_2d(sfd, m_total, n // 32)[0].float()
            scale = scale.repeat_interleave(32, -1)
            err = (d_q.float() * scale - acc).abs()
            # half the widest e4m3 bin (16 units of the scale) plus the baseline's rounding
            bound = scale * 16 * 1.05 + (accq - acc).abs().max()
            if not (err <= bound).all() or torch.isnan(scale).any():
                failures.append(f"rank{rank} AG+mxfp8-D iter={it}: err={err.max().item()}")

    # --- The TP block's remaining forward and backward epilogues on the gathered operand: the deferred-rstd
    # gate/up (per-row scale r, paired bias, SwiGLU; D = the affine preact), the dSwiGLU dgrad (acc = dout,
    # C = the saved packed preact, D = packed (dx, dy)), and the RMSNorm-backward partial around a dgrad
    # (a tile load y, the colvec rstd, the rowvec w, D = rstd * acc * w and the per-tile correction dots) ---
    from quack.epilogue.library import dswiglu_mod, norm_swiglu_mod, rms_bwd_partial_epi

    n = 1024  # != m_total: an unpinned (l, dim) vector is told apart from the other dim by its length
    torch.manual_seed(4323)
    b = (torch.randn(n, k, dtype=dtype, device=device) / 8).contiguous()
    r = (
        torch.rand(1, m_total, dtype=torch.float32, device=device) + 0.5
    )  # dense calls pass vectors (l, dim)
    bias = torch.randn(1, n, dtype=torch.float32, device=device)
    w = torch.randn(1, n, dtype=torch.float32, device=device)
    y_saved = torch.randn(m_total, n, dtype=dtype, device=device)
    preact = torch.randn(m_total, 2 * n, dtype=dtype, device=device)  # packed (x, y) pairs
    postact = torch.empty(m_total, n // 2, dtype=dtype, device=device)
    d_pre = torch.empty(m_total, n, dtype=dtype, device=device)
    post_bwd = torch.empty(
        m_total, n, dtype=dtype, device=device
    )  # dSwiGLU's recomputed postact, GEMM-N wide
    d_bwd = torch.empty(m_total, 2 * n, dtype=dtype, device=device)
    d_rms = torch.empty(m_total, n, dtype=dtype, device=device)
    dots = torch.zeros(m_total, n // 256, dtype=torch.float32, device=device)
    for it in range(2):
        torch.manual_seed(79 * it + rank)
        a_shard = (torch.randn(shard_m, k, dtype=dtype, device=device) / 8).contiguous()
        a_full_ref = torch.empty(m_total, k, dtype=dtype, device=device)
        dist.all_gather_into_tensor(a_full_ref, a_shard)
        acc = a_full_ref.float() @ b.float().T
        accq = acc.to(dtype).float()
        # r-scale + paired bias + SwiGLU (CODA's gate/up kernel with the rstd deferred into this GEMM)
        with runner.prepare(a_shard) as (a_full, ag_args):
            norm_swiglu_mod.gemm(
                a_full,
                b,
                d_pre,
                None,
                epi_args=dict(rstd=r, bias=bias, postact=postact),
                dist_op_args=ag_args,
                **kw,
            )
        torch.cuda.synchronize(device)
        v, vq = acc * r.T + bias, accq * r.T + bias
        check(f"AG+rscale+swiglu D iter={it}", d_pre, v, vq)
        check(
            f"AG+rscale+swiglu iter={it}",
            postact,
            F.silu(v[:, 0::2]) * v[:, 1::2],
            F.silu(vq[:, 0::2]) * vq[:, 1::2],
        )
        # dSwiGLU: the dgrad GEMM's output is dout per pair; C = the saved packed preact
        with runner.prepare(a_shard) as (a_full, ag_args):
            dswiglu_mod.gemm(
                a_full,
                b,
                d_bwd,
                preact,
                epi_args=dict(postact=post_bwd),
                dist_op_args=ag_args,
                **kw,
            )
        torch.cuda.synchronize(device)
        x, yy = preact.float()[:, 0::2], preact.float()[:, 1::2]
        sig = torch.sigmoid(x)

        def dswiglu_ref(dout):
            dx, dy = dout * yy * sig * (1 + x * (1 - sig)), dout * x * sig
            return torch.stack([dx, dy], -1).reshape(m_total, 2 * n), x * sig * yy

        d_ref, out_ref = dswiglu_ref(acc)
        d_base, _ = dswiglu_ref(accq)
        check(f"AG+dswiglu D iter={it}", d_bwd, d_ref, d_base)
        check(f"AG+dswiglu postact iter={it}", post_bwd, out_ref, out_ref.to(dtype).float())
        # RMSNorm backward partial: D = rstd * (acc * w), dots = per-tile sum over N of (acc * w) * (y * rstd)
        with runner.prepare(a_shard) as (a_full, ag_args):
            rms_bwd_partial_epi.gemm(
                a_full,
                b,
                d_rms,
                None,
                epi_args=dict(y=y_saved, rstd=r, w=w, dots=dots),
                dist_op_args=ag_args,
                **kw,
            )
        torch.cuda.synchronize(device)
        t, tq = acc * w, accq * w
        xhat = y_saved.float() * r.T
        check(f"AG+rms_bwd D iter={it}", d_rms, t * r.T, tq * r.T)
        tile_dots = lambda tt: (tt * xhat).view(m_total, n // 256, 256).sum(-1)
        check(f"AG+rms_bwd dots iter={it}", dots, tile_dots(t), tile_dots(tq))

    # --- Outer-capture smoke: the epoch is device-resident, so a caller may
    # capture any number of whole calls and replay.
    dtype, shard_m, n, k = torch.bfloat16, 1024, 1024, 4096
    torch.manual_seed(99)
    b = (torch.randn(n, k, dtype=dtype, device=device) / 8).contiguous()
    ag = AllGatherRunner(shard_m, k, dtype, device=device)
    m_total = shard_m * world_size
    a_in = torch.zeros(shard_m, k, dtype=dtype, device=device)
    d1 = torch.empty(m_total, n, dtype=dtype, device=device)
    d2 = torch.empty(m_total, n, dtype=dtype, device=device)
    for _ in range(2):  # warmup outside capture (also allocator warmup)
        _ag_gemm(ag, a_in, b, d1)
        _ag_gemm(ag, a_in, b, d2)
    torch.cuda.synchronize(device)
    dist.barrier()
    g = torch.cuda.CUDAGraph()
    # THREE calls — deliberately ODD, the hard case: replaying it puts two
    # same-parity calls back to back across the replay boundary, which the
    # old chained per-parity epoch bump silently corrupted (reused epoch
    # value + same buffer => gates release on stale data); the global-row
    # epoch stays monotone. Also exercises a real intra-capture reuse edge
    # (call 3 waits ev_reuse recorded by call 1's barrier — the i -> i+2
    # edge only appears in captures of >= 3 calls).
    with torch.cuda.graph(g):
        _ag_gemm(ag, a_in, b, d1)
        _ag_gemm(ag, a_in, b, d2)
        _ag_gemm(ag, a_in, b, d1)
    g_parity = ag.last_capture_parity  # baked final buffer, constant across replays
    for it in range(3):
        torch.manual_seed(500 * it + rank)
        a_in.copy_((torch.randn(shard_m, k, dtype=dtype, device=device) / 8))
        g.replay()
        torch.cuda.synchronize(device)
        a_full_ref = torch.empty(m_total, k, dtype=dtype, device=device)
        dist.all_gather_into_tensor(a_full_ref, a_in)
        ref = a_full_ref.float() @ b.float().T
        for name, dd in (("d1", d1), ("d2", d2)):
            err = (dd.float() - ref).abs().max().item()
            baseline = (a_full_ref.float() @ b.float().T).to(dtype).float()
            tol = (baseline - ref).abs().max().item() + 1e-4
            if err > 2 * tol:
                failures.append(f"rank{rank} outer-capture {name} it={it}: err={err} tol={tol}")

    # --- Mode-mixing through quiesce(): after ODD-count replays, quiesce()
    # must (a) fence the replay work so an eager call is safe and (b) resync
    # last_parity to the graph's BAKED final buffer (without the resync,
    # gathered_a() was observed selecting the WRONG buffer after odd
    # replays). Uses the SYNC-FREE resync (caller passes the capture-parity
    # token); the bare syncing variant is cross-checked against it below.
    ag.quiesce(last_parity=g_parity)
    torch.manual_seed(4242 + rank)
    a_in.copy_((torch.randn(shard_m, k, dtype=dtype, device=device) / 8))
    d_eager = _ag_gemm(ag, a_in, b)
    torch.cuda.synchronize(device)
    # bare quiesce() derives the buffer from the device snapshots — must
    # agree with the token path (validates both resync variants)
    shadow = ag.last_parity
    ag.quiesce()
    if ag.last_parity != shadow:
        failures.append(
            f"rank{rank} quiesce resync mismatch: device={ag.last_parity} token={shadow}"
        )

    # --- Cross-stream replays: quiesce(replay_stream=...) must order itself
    # after replays launched on a NON-current stream (without the edge the
    # device-epoch read misses the replay and lands one call behind).
    replay_stream = torch.cuda.Stream(device)
    torch.manual_seed(777 + rank)
    a_in.copy_((torch.randn(shard_m, k, dtype=dtype, device=device) / 8))
    torch.cuda.synchronize(device)
    with torch.cuda.stream(replay_stream):
        g.replay()
    ag.quiesce(replay_stream=replay_stream)
    shadow_before = ag.last_parity
    d_eager = _ag_gemm(ag, a_in, b)
    torch.cuda.synchronize(device)
    a_full_ref = torch.empty(m_total, k, dtype=dtype, device=device)
    dist.all_gather_into_tensor(a_full_ref, a_in)
    ref = a_full_ref.float() @ b.float().T
    baseline = (a_full_ref.float() @ b.float().T).to(dtype).float()
    tol = (baseline - ref).abs().max().item() + 1e-4
    err = (d_eager.float() - ref).abs().max().item()
    gathered_err = (ag.gathered_a() - a_full_ref).abs().max().item()
    if err > 2 * tol or gathered_err != 0 or shadow_before != g_parity:
        failures.append(
            f"rank{rank} cross-stream quiesce: err={err} tol={tol} "
            f"gathered_err={gathered_err} parity={shadow_before} (baked {g_parity})"
        )
    a_full_ref = torch.empty(m_total, k, dtype=dtype, device=device)
    dist.all_gather_into_tensor(a_full_ref, a_in)
    ref = a_full_ref.float() @ b.float().T
    baseline = (a_full_ref.float() @ b.float().T).to(dtype).float()
    tol = (baseline - ref).abs().max().item() + 1e-4
    err = (d_eager.float() - ref).abs().max().item()
    gathered_err = (ag.gathered_a() - a_full_ref).abs().max().item()
    if err > 2 * tol or gathered_err != 0:
        failures.append(
            f"rank{rank} quiesce mode-mix: err={err} tol={tol} gathered_err={gathered_err}"
        )

    # --- Odd graph x even replay count: graph buffer addresses are baked, so
    # a one-call graph always finishes in the SAME physical buffer. Two
    # replays advance the global epoch by an even count but do not change that
    # final physical buffer. Neither arithmetic call-count parity nor global
    # device-epoch parity may be used to select gathered_a(). Distinct eager
    # and replay inputs make a wrong physical-buffer choice observable.
    # capture_lockstep=True also covers the paced captured schedule
    # (per-call join edges left in the graph) end to end.
    parity_ag = AllGatherRunner(shard_m, k, dtype, device=device, capture_lockstep=True)
    parity_eager = torch.zeros(shard_m, k, dtype=dtype, device=device)
    parity_in = torch.zeros_like(parity_eager)
    parity_d = torch.empty(m_total, n, dtype=dtype, device=device)
    for _ in range(2):
        _ag_gemm(parity_ag, parity_eager, b, parity_d)
    torch.cuda.synchronize(device)
    dist.barrier()

    parity_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(parity_graph):
        _ag_gemm(parity_ag, parity_in, b, parity_d)
    parity_token = parity_ag.last_capture_parity

    torch.manual_seed(8100 + rank)
    parity_in.copy_(torch.randn_like(parity_in) / 8)
    parity_graph.replay()
    torch.manual_seed(8200 + rank)
    parity_in.copy_(torch.randn_like(parity_in) / 8)
    parity_graph.replay()

    parity_ag.quiesce(last_parity=parity_token)
    check_plain_result("quiesce token physical parity", parity_ag, parity_in, b, parity_d)
    parity_ag.quiesce()
    check_plain_result("quiesce device physical parity", parity_ag, parity_in, b, parity_d)

    # --- Repeated graph <-> eager transitions through the sync-free quiesce
    # path. Rank skew keeps one peer's eager GEMM live while faster ranks reach
    # the boundary; omitting the eager->graph fence corrupts the next replay.
    mix_ag = AllGatherRunner(shard_m, k, dtype, device=device)
    mix_graph_in = torch.zeros(shard_m, k, dtype=dtype, device=device)
    mix_eager_in = torch.zeros_like(mix_graph_in)
    mix_graph_d = torch.empty(m_total, n, dtype=dtype, device=device)
    mix_eager_d = torch.empty_like(mix_graph_d)
    for _ in range(2):
        _ag_gemm(mix_ag, mix_graph_in, b, mix_graph_d)
    torch.cuda.synchronize(device)
    dist.barrier()

    mix_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(mix_graph):
        _ag_gemm(mix_ag, mix_graph_in, b, mix_graph_d)
    mix_token = mix_ag.last_capture_parity

    for it in range(2):
        torch.manual_seed(9000 + 100 * it + rank)
        mix_graph_in.copy_(torch.randn_like(mix_graph_in) / 8)
        mix_graph.replay()
        mix_ag.quiesce(last_parity=mix_token)
        check_plain_result(f"quiesce graph->eager iter={it}", mix_ag, mix_graph_in, b, mix_graph_d)

        torch.manual_seed(9050 + 100 * it + rank)
        mix_eager_in.copy_(torch.randn_like(mix_eager_in) / 8)
        if rank == world_size - 1:
            torch.cuda._sleep(int(50e6))
        _ag_gemm(mix_ag, mix_eager_in, b, mix_eager_d)
        # pure fence before the next replay: no graph work since the last
        # eager call, so the runner's own parity is already correct
        mix_ag.quiesce(last_parity=mix_ag.last_parity)
        check_plain_result(f"quiesce eager->graph iter={it}", mix_ag, mix_eager_in, b, mix_eager_d)

    # --- world_size == 1 degenerate path: the fast path skips transport,
    # flags, and gate, but must STILL bump the device parity snapshots, or
    # bare quiesce()'s buffer discovery cannot see replays (measured
    # bare_error=5.19 before the fix; token path was always fine).
    solo_group = None
    for r in range(world_size):  # new_group is collective: all ranks, every group
        group_r = dist.new_group([r])
        if r == rank:
            solo_group = group_r
    # (rendezvous on a 1-rank group makes torch log a scary-looking but
    # benign "fail to export multicast handle ... Gracefully skipping"
    # warning trace — the driver refuses 1-device multicast objects)
    solo_ag = AllGatherRunner(shard_m, k, dtype, group=solo_group, device=device)
    solo_in = torch.zeros(shard_m, k, dtype=dtype, device=device)
    solo_d = torch.empty(shard_m, n, dtype=dtype, device=device)
    for _ in range(2):
        _ag_gemm(solo_ag, solo_in, b, solo_d)
    torch.cuda.synchronize(device)
    solo_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(solo_graph):
        _ag_gemm(solo_ag, solo_in, b, solo_d)
    for seed in (6100, 6200):  # two replays: baked parity vs count parity diverge
        torch.manual_seed(seed + rank)
        solo_in.copy_(torch.randn_like(solo_in) / 8)
        solo_graph.replay()
    solo_ag.quiesce()  # bare: device discovery must find the baked buffer
    torch.cuda.synchronize(device)
    ref = solo_in.float() @ b.float().T
    baseline = ref.to(dtype).float()
    tol = (baseline - ref).abs().max().item() + 1e-4
    err = (solo_d.float() - ref).abs().max().item()
    gathered_err = (solo_ag.gathered_a() - solo_in).abs().max().item()
    if err > 2 * tol or gathered_err != 0:
        failures.append(
            f"rank{rank} ws==1 bare quiesce: err={err} tol={tol} gathered_err={gathered_err}"
        )

    fail_count = torch.tensor([len(failures)], device=device)
    dist.all_reduce(fail_count)
    if failures:
        print("\n".join(failures), file=sys.stderr, flush=True)
    dist.destroy_process_group()
    if int(fail_count.item()) > 0:
        sys.exit(1)
    if rank == 0:
        print("ALL AG+GEMM CHECKS PASSED", flush=True)


def _visible_gpus() -> str:
    """All GPUs, minus any with a large foreign allocation (shared node)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        free = [
            line.split(",")[0].strip()
            for line in out.strip().splitlines()
            if int(line.split(",")[1]) < 20000
        ]
        return ",".join(free)
    except Exception:
        return ",".join(str(i) for i in range(torch.cuda.device_count()))


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs >= 2 GPUs")
def test_all_gather_gemm_multirank():
    gpus = _visible_gpus()
    nproc = min(len(gpus.split(",")), 8)
    assert nproc >= 2, f"not enough free GPUs: {gpus}"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpus
    env.setdefault("PYTHONPATH", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={nproc}",
            "--standalone",
            os.path.abspath(__file__),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=1200,
    )
    if res.returncode != 0:
        print(res.stdout[-4000:])
        print(res.stderr[-4000:], file=sys.stderr)
    assert res.returncode == 0
    assert "ALL AG+GEMM CHECKS PASSED" in res.stdout


if __name__ == "__main__":
    _run_rank()
