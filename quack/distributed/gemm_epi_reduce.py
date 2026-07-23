import torch
from quack.dist_utils import make_barrier_flags, make_symmetric_tensor
from quack.epi_reduce import EpiReduceArguments


def make_epi_reduce_args(mD_mc, mD_peers, m, n, l, cta_m, cta_n, cluster_m, cluster_n, num_ranks):
    """Allocate the epi_reduce_mode workspace/semaphores and assemble
    EpiReduceArguments. Workspace, flags, and counters share the cluster-rounded
    (cta_m * cta_n, ntile_m, ntile_n, l) tile domain (split-K's rounding: a
    boundary cluster's fully-OOB CTA still runs the commit protocol into its own
    dead slot).

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
    num_tiles = ntile_m * ntile_n * l
    # d_dtype partial stripes; permuted so the kernel-facing view is
    # (E, ntile_m, ntile_n, L) with the stripe (E) contiguous.
    ws_base, ws_base_mc = make_symmetric_tensor(
        (l, ntile_n, ntile_m, cta_m * cta_n), mD_peers[0].dtype
    )
    workspace = ws_base.permute(3, 2, 1, 0)
    workspace_mc = ws_base_mc.permute(3, 2, 1, 0)
    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    tile_flags, tile_flags_mc, _, _ = make_barrier_flags(num_tiles)
    sync_barrier, sync_barrier_mc, _, _ = make_barrier_flags(num_sms)
    counters = torch.zeros(num_tiles, dtype=torch.int32, device="cuda")
    return EpiReduceArguments(
        mD_mc=mD_mc,
        mD_peers=tuple(mD_peers),
        workspace=workspace,
        workspace_mc=workspace_mc,
        tile_flags=tile_flags,
        tile_flags_mc=tile_flags_mc,
        sync_barrier=sync_barrier,
        sync_barrier_mc=sync_barrier_mc,
        consumer_counters=counters,
    )
