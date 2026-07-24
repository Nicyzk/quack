import torch
from quack.dist_utils import make_barrier_flags, make_symmetric_tensor
from quack.epi_reduce import EpiReduceArguments


def make_epi_reduce_args(mD_mc, mD_peers, m, n, l, cta_m, cta_n, cluster_m, cluster_n, num_ranks):
    """Allocate the epi_reduce_mode workspace/semaphores and assemble
    EpiReduceArguments. The workspace is a flat (M_pad, N_pad, L) d_dtype tensor
    addressed by real coordinates; flags and counters live on the cluster-rounded
    (ntile_m, ntile_n, L) tile grid (split-K's rounding: a boundary cluster's
    fully-OOB CTA still runs the commit protocol into its own dead slot). M_pad
    carries one extra cta_m row block so a reduce_scatter slab tile anchored at an
    unaligned slab origin stays fully in bounds — no predication on workspace
    access anywhere; N_pad keeps 16B vectors from straddling the edge.

    Call once at setup (symmetric allocs are collective) and reuse across
    launches: flags are monotonic — never reset — and consumer_counters hold each
    tile's epoch base, so never re-zero one without the other.

    Returns the torch-tensor flavor the TVM-FFI surfaces consume (EpiMod.gemm /
    quack.gemm.gemm); direct cute.compile callers build the cute flavor themselves.
    """
    assert m % num_ranks == 0, "epi_reduce_mode slab math needs M % num_ranks == 0"
    ntile_m = (m + cta_m - 1) // cta_m
    ntile_n = (n + cta_n - 1) // cta_n
    ntile_m = (ntile_m + cluster_m - 1) // cluster_m * cluster_m
    ntile_n = (ntile_n + cluster_n - 1) // cluster_n * cluster_n
    m_pad, n_pad = (ntile_m + 1) * cta_m, ntile_n * cta_n
    # d_dtype partial D; permuted so the kernel-facing view is (M_pad, N_pad, L)
    # with N contiguous.
    ws_base, ws_base_mc = make_symmetric_tensor((l, m_pad, n_pad), mD_peers[0].dtype)
    workspace = ws_base.permute(1, 2, 0)
    workspace_mc = ws_base_mc.permute(1, 2, 0)
    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    tile_flags, tile_flags_mc, _, _ = make_barrier_flags(ntile_m * ntile_n * l)
    sync_barrier, sync_barrier_mc, _, _ = make_barrier_flags(num_sms)
    counters = torch.zeros((l, ntile_m, ntile_n), dtype=torch.int32, device="cuda")
    tile_view = lambda t: t.view(l, ntile_m, ntile_n).permute(1, 2, 0)
    return EpiReduceArguments(
        mD_mc=mD_mc,
        mD_peers=tuple(mD_peers),
        workspace=workspace,
        workspace_mc=workspace_mc,
        tile_flags=tile_view(tile_flags),
        tile_flags_mc=tile_view(tile_flags_mc),
        sync_barrier=sync_barrier,
        sync_barrier_mc=sync_barrier_mc,
        consumer_counters=counters.permute(1, 2, 0),
    )
