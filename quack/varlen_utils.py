# Copyright (c) 2025, Tri Dao.

from typing import Optional, NamedTuple
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Boolean, const_expr

from quack import copy_utils
from quack.cute_dsl_utils import mlir_namedtuple


# Grouping arguments together that should be passed to __call__
@mlir_namedtuple
class VarlenArguments(NamedTuple):
    mCuSeqlensM: Optional[cute.Tensor] = None
    mCuSeqlensK: Optional[cute.Tensor] = None
    # Variable-length N (swap-AB grouped GEMM: the ragged token dim rides B and D's N).
    mCuSeqlensN: Optional[cute.Tensor] = None
    mAIdx: Optional[cute.Tensor] = None
    # Per-sequence M-tile prefix (num_seqs + 1,): cumsum of
    # ceil(seqlen / cta_tile_M). Host-computed (device tensor) and only passed
    # when an epilogue op needs per-sequence tile indexing (M-fold reduce
    # sinks); the tile scheduler's m index is sequence-local, so a batchless
    # per-M-tile buffer needs this offset to keep sequences' rows disjoint.
    mCuTilesM: Optional[cute.Tensor] = None
    # varlen_n row gather (swap-AB tokens on B): B row of GEMM row i is mBIdx[i], (total_n,)
    mBIdx: Optional[cute.Tensor] = None


VARLEN_N_GRANULE = 64  # the varlen_n MMA-N variants' row granule (gemm_sm100.varlen_n_granule)


@cute.jit
def varlen_n_block_rows(n: Int32, tile_n: cutlass.Constexpr[int], granule: cutlass.Constexpr[int]) -> Int32:
    """varlen_n even split: n rows cut into ceil(n / tile_n) blocks of ceil(n / blocks) rows rounded up to
    `granule`, so no block is a thin tail. Shared by the kernel, the tile scheduler's gates and the walks."""
    num_blocks = cute.ceil_div(n, tile_n)
    return cute.ceil_div(cute.ceil_div(n, num_blocks), granule) * granule


class VarlenManager:
    @dataclass
    class Params:
        cu_seqlens_m: Optional[cute.Tensor] = None
        cu_seqlens_k: Optional[cute.Tensor] = None
        cu_seqlens_n: Optional[cute.Tensor] = None
        mAIdx: Optional[cute.Tensor] = None
        cu_tiles_m: Optional[cute.Tensor] = None
        mBIdx: Optional[cute.Tensor] = None

        @staticmethod
        @cute.jit
        def create(args: VarlenArguments, *, loc=None, ip=None) -> "VarlenManager.Params":
            return VarlenManager.Params(
                cu_seqlens_m=args.mCuSeqlensM,
                cu_seqlens_k=args.mCuSeqlensK,
                cu_seqlens_n=args.mCuSeqlensN,
                mAIdx=args.mAIdx,
                cu_tiles_m=args.mCuTilesM,
                mBIdx=args.mBIdx,
            )

    def __init__(
        self,
        params: Params,
        len_m_static: Int32,
        len_k_static: Int32,
        len_n_static: Int32,
        last_batch_idx: Int32 = Int32(-1),
        is_group_changed: Boolean = Boolean(True),
        *,
        loc=None,
        ip=None,
    ):
        self.params = params
        self._len_m_static = len_m_static
        self._len_k_static = len_k_static
        self._len_n_static = len_n_static
        self._last_batch_idx = last_batch_idx
        self._is_group_changed = is_group_changed
        self.varlen_m = const_expr(params.cu_seqlens_m is not None)
        self.varlen_k = const_expr(params.cu_seqlens_k is not None)
        self.varlen_n = const_expr(params.cu_seqlens_n is not None)
        self.gather_A = const_expr(params.mAIdx is not None)
        self.gather_B = const_expr(params.mBIdx is not None)
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(args: VarlenArguments, *, loc=None, ip=None) -> Params:
        num_varlen = sum(
            t is not None for t in (args.mCuSeqlensM, args.mCuSeqlensK, args.mCuSeqlensN)
        )
        assert num_varlen <= 1, "Only support one of varlen_m, varlen_k, varlen_n"
        assert not (args.mCuSeqlensN is not None and args.mAIdx is not None), (
            "varlen_n does not support gather_A"
        )
        assert args.mBIdx is None or args.mCuSeqlensN is not None, "gather_B needs varlen_n"
        return VarlenManager.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    @cute.jit
    def create(
        params: Params,
        len_m_static: Int32,
        len_k_static: Int32,
        len_n_static: Int32,
        *,
        loc=None,
        ip=None,
    ) -> "VarlenManager":
        return VarlenManager(
            params,
            len_m_static=len_m_static,
            len_k_static=len_k_static,
            len_n_static=len_n_static,
        )

    def len_m(self, batch_idx: Int32) -> Int32:
        if const_expr(self.varlen_m):
            return self.params.cu_seqlens_m[batch_idx + 1] - self.params.cu_seqlens_m[batch_idx]
        else:
            return self._len_m_static

    def len_k(self, batch_idx: Int32) -> Int32:
        if const_expr(self.varlen_k):
            return self.params.cu_seqlens_k[batch_idx + 1] - self.params.cu_seqlens_k[batch_idx]
        else:
            return self._len_k_static

    def len_n(self, batch_idx: Optional[Int32] = None) -> Int32:
        if const_expr(self.varlen_n):
            assert batch_idx is not None, "varlen_n: len_n needs the batch index"
            return self.params.cu_seqlens_n[batch_idx + 1] - self.params.cu_seqlens_n[batch_idx]
        else:
            return self._len_n_static

    # varlen_n even split (varlen_n_block_rows): block j covers rows [j * b, min((j + 1) * b, n)).
    def block_n_rows(self, batch_idx: Int32, tile_n: int, granule: int) -> Int32:
        return varlen_n_block_rows(self.len_n(batch_idx), tile_n, granule)

    def block_n_valid(self, batch_idx: Int32, pid_n: Int32, tile_n: int, granule: int) -> Int32:
        """Rows of block pid_n (the last block may be shorter)."""
        b = self.block_n_rows(batch_idx, tile_n, granule)
        return cutlass.min(b, self.len_n(batch_idx) - pid_n * b)

    def offset_block_B(
        self,
        mB_nkl: cute.Tensor,
        batch_idx: Int32,
        pid_n: Int32,
        tile_n: int,
        granule: int,
        row_shift: Int32 = Int32(0),
    ) -> cute.Tensor:
        """B at block pid_n's first row (+ row_shift; rows past the block are the next
        block's, masked at the stores)."""
        b = self.block_n_rows(batch_idx, tile_n, granule)
        offset = self.params.cu_seqlens_n[batch_idx] + pid_n * b + row_shift
        return cute.domain_offset((offset, None), mB_nkl)

    def offset_block_epi(
        self, mD_mnl: cute.Tensor, batch_idx: Int32, pid_n: Int32, tile_n: int, granule: int
    ) -> cute.Tensor:
        """D ragged to block pid_n: the TMA store clips columns past the block's rows."""
        b = self.block_n_rows(batch_idx, tile_n, granule)
        offset = self.params.cu_seqlens_n[batch_idx] + pid_n * b
        length = cutlass.min(b, self.len_n(batch_idx) - pid_n * b)
        ragged_rank = const_expr(cute.rank(mD_mnl))
        assert ragged_rank > 2, "varlen_n D must be a ragged TMA tensor"
        return copy_utils.offset_ragged_tensor(
            mD_mnl, offset, length, ragged_dim=1, ptr_shift=const_expr(ragged_rank == 3)
        )

    def tile_m_offset(self, batch_idx: Int32) -> Int32:
        """This sequence's first row in a per-M-tile buffer (cu_tiles_m prefix)."""
        return self.params.cu_tiles_m[batch_idx]

    def len_m_tiles(self, batch_idx: Int32) -> Int32:
        """Number of M tiles this sequence owns (from the cu_tiles_m prefix)."""
        return self.params.cu_tiles_m[batch_idx + 1] - self.params.cu_tiles_m[batch_idx]

    def offset_batch_A(self, mA_mkl: cute.Tensor, batch_idx: Int32) -> cute.Tensor:
        params = self.params
        if const_expr(self.varlen_m):
            offset = params.cu_seqlens_m[batch_idx]
            ragged_rank = const_expr(cute.rank(mA_mkl))
            if const_expr(ragged_rank == 2):  # Didn't create ragged tensor (gather_A)
                mA_mk = cute.domain_offset((offset, None), mA_mkl)
            else:
                # Ragged-for-TMA (zero-fill rows past the sequence end): loads
                # must use the 2-extra-dim wraparound form (ptr_shift lands the
                # descriptor base outside mapped memory, store-only).
                length = params.cu_seqlens_m[batch_idx + 1] - offset
                mA_mk = copy_utils.offset_ragged_tensor(
                    mA_mkl,
                    offset,
                    length,
                    ragged_dim=0,
                    ptr_shift=False,
                )
        elif const_expr(self.varlen_k):
            offset = params.cu_seqlens_k[batch_idx]
            ragged_rank = const_expr(cute.rank(mA_mkl))
            if const_expr(ragged_rank == 2):  # Didn't create ragged tensor
                mA_mk = cute.domain_offset((None, offset), mA_mkl)
            else:
                length = params.cu_seqlens_k[batch_idx + 1] - offset
                # rank 3 = 1-extra-dim (ptr_shift), rank 4 = 2-extra-dim
                ptr_shift = const_expr(ragged_rank == 3)
                mA_mk = copy_utils.offset_ragged_tensor(
                    mA_mkl,
                    offset,
                    length,
                    ragged_dim=1,
                    ptr_shift=ptr_shift,
                )
        else:
            mA_mk = mA_mkl[None, None, batch_idx]
        return mA_mk

    def offset_block_BIdx(
        self, batch_idx: Int32, pid_n: Int32, tile_n: int, granule: int, row_shift: Int32
    ) -> cute.Tensor:
        """mBIdx at block pid_n's first row + row_shift (a CTA's share of the block)."""
        b = self.block_n_rows(batch_idx, tile_n, granule)
        offset = self.params.cu_seqlens_n[batch_idx] + pid_n * b + row_shift
        return cute.domain_offset((offset,), self.params.mBIdx)

    def offset_batch_AIdx(self, batch_idx: Int32) -> cute.Tensor:
        params = self.params
        if const_expr(self.varlen_m):
            mAIdx_mk = cute.domain_offset((params.cu_seqlens_m[batch_idx],), params.mAIdx)
        elif const_expr(self.varlen_k):
            mAIdx_mk = cute.domain_offset((params.cu_seqlens_k[batch_idx],), params.mAIdx)
        else:
            mAIdx_mk = params.mAIdx[None, batch_idx]
        return mAIdx_mk

    def offset_batch_SFA(self, mSFA_mkl: cute.Tensor, batch_idx: Int32) -> cute.Tensor:
        """Offset SFA to this batch's tile-aligned region of the padded SF buffer.

        The padded offset, in tile units (128 source-M or source-K per tile),
        is simply `cu_seqlens[b] // 128 + b`. (Algebraically identical to
        `(cu_seqlens[b] + b*128) // 128 * 128` / 128.) We pass it as a
        compound coord `(0, offset_tile)` to `domain_offset` so the outer
        rm/rk mode is shifted in tile units — no `* 128` needed, and the
        compiler sees the tile alignment natively.
        """
        params = self.params
        tile = 128
        if const_expr(self.varlen_m):
            offset_tile = params.cu_seqlens_m[batch_idx] // tile + batch_idx
            return cute.domain_offset(((None, offset_tile), None), mSFA_mkl)
        elif const_expr(self.varlen_k):
            offset_tile = params.cu_seqlens_k[batch_idx] // tile + batch_idx
            return cute.domain_offset((None, (None, offset_tile)), mSFA_mkl)
        else:
            return mSFA_mkl[None, None, batch_idx]

    def offset_batch_SFB(self, mSFB_chunks: cute.Tensor, batch_idx: Int32) -> cute.Tensor:
        """Slice the (chunk, RK, RN, L) SFB chunk view to this batch.

        For varlen_k (L == 1) the batch offset is on the atom-k mode, with the
        same tile-aligned padded-layout arithmetic as offset_batch_SFA.
        (varlen_k blockscaled is mxfp8-only, where one SF atom covers
        4 sf-k blocks x sf_vec_size 32 = 128 K elements.)
        """
        params = self.params
        tile = 128
        if const_expr(self.varlen_k):
            offset_atom_k = params.cu_seqlens_k[batch_idx] // tile + batch_idx
            return cute.domain_offset((None, offset_atom_k, None), mSFB_chunks[None, None, None, 0])
        else:
            return mSFB_chunks[None, None, None, batch_idx]

    def offset_batch_B(self, mB_nkl: cute.Tensor, batch_idx: Int32) -> cute.Tensor:
        params = self.params
        if const_expr(self.varlen_n):
            # Rows past the sequence end are the next sequence's (masked at the stores).
            mB_nk = cute.domain_offset((params.cu_seqlens_n[batch_idx], None), mB_nkl)
        elif const_expr(self.varlen_k):
            offset = params.cu_seqlens_k[batch_idx]
            ragged_rank = const_expr(cute.rank(mB_nkl))
            if const_expr(ragged_rank == 2):  # Didn't create ragged tensor
                mB_nk = cute.domain_offset((None, offset), mB_nkl)
            else:
                length = params.cu_seqlens_k[batch_idx + 1] - offset
                ptr_shift = const_expr(ragged_rank == 3)
                mB_nk = copy_utils.offset_ragged_tensor(
                    mB_nkl,
                    offset,
                    length,
                    ragged_dim=1,
                    ptr_shift=ptr_shift,
                )
        else:
            mB_nk = mB_nkl[None, None, batch_idx]
        return mB_nk

    def offset_batch_epi(self, mD_mnl: cute.Tensor, batch_idx: Int32) -> cute.Tensor:
        params = self.params
        if const_expr(self.varlen_n):
            offset = params.cu_seqlens_n[batch_idx]
            ragged_rank = const_expr(cute.rank(mD_mnl))
            if const_expr(ragged_rank == 2):  # Didn't create ragged tensor
                mD_mn = cute.domain_offset((None, offset), mD_mnl)
            else:
                length = params.cu_seqlens_n[batch_idx + 1] - offset
                ptr_shift = const_expr(ragged_rank == 3)
                mD_mn = copy_utils.offset_ragged_tensor(
                    mD_mnl,
                    offset,
                    length,
                    ragged_dim=1,
                    ptr_shift=ptr_shift,
                )
        elif const_expr(self.varlen_m):
            offset = params.cu_seqlens_m[batch_idx]
            ragged_rank = const_expr(cute.rank(mD_mnl))
            if const_expr(ragged_rank == 2):  # Didn't create ragged tensor
                mD_mn = cute.domain_offset((offset, None), mD_mnl)
            else:
                length = params.cu_seqlens_m[batch_idx + 1] - offset
                ptr_shift = const_expr(ragged_rank == 3)
                mD_mn = copy_utils.offset_ragged_tensor(
                    mD_mnl,
                    offset,
                    length,
                    ragged_dim=0,
                    ptr_shift=ptr_shift,
                )
        else:
            mD_mn = mD_mnl[None, None, batch_idx]
        return mD_mn

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [
            self.params,
            self._len_m_static,
            self._len_k_static,
            self._len_n_static,
            self._last_batch_idx,
            self._is_group_changed,
        ]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip(
            [
                self.params,
                self._len_m_static,
                self._len_k_static,
                self._len_n_static,
                self._last_batch_idx,
                self._is_group_changed,
            ],
            self._values_pos,
        ):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return self.__class__(*(tuple(obj_list)), loc=self._loc)
