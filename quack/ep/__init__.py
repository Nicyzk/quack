"""Expert-parallel MoE support shared by the inter-SM transfer kernels (quack.gemm_sm100_ep_inter) and the
intra-SM fused GEMMs (quack.gemm_sm100_ep_intra_dispatch / _combine).

- ptx.py: the inline-PTX primitives of the transfer loops (1D bulk copies, release adds, software grid sync)
- routing_metadata.py: the dispatch tables (cu_seqlens, slot maps, pool rows) from the gathered top-k, 2 kernels
- prologue.py: the one-launch routing prologue (top-k exchange, resets, metadata, dispatch walk)
- routed_dispatch.py: MegaMoE's in-kernel routed dispatch (prep / barrier / pull) for the intra GEMM1
"""
