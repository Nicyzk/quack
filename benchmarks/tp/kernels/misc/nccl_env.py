"""NCCL defaults every benchmark's NCCL reference runs with (import before any process group).

NCCL 2.29.7 (torch 2.13+cu130) tunes the reference collectives of these tables — the 512 KiB
decode all-reduce, the 64 MiB reduce-scatter partial — onto ONE channel, 2x slower than NCCL
2.27.3 on the same GPUs (measured 2026-10-13 on della-tridao, same two GPUs, same minute). A
channel floor restores it (4 and 8 measured equal); protocol or algorithm overrides do not.
The cuBLAS+NCCL column must be the fastest honest NCCL, as every other arm gets its sweep, so
the floor applies to every reference and NCCL floor uniformly and is stated in the tables.
"""

import os

os.environ.setdefault("NCCL_MIN_NCHANNELS", "8")
