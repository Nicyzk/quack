"""Inline-PTX primitives of the EP transfer loops: 1D bulk copies with a per-warp mbarrier, release / relaxed
reductions and stores on raw int64 peer addresses, fns.b32, and a software grid sync (cuTeDSL has no cooperative
launch). Addresses are either cute pointers or Int64 byte addresses (peer buffers reached through a base table)."""
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import Int32, Int64, T, dsl_user_op

TMA_CACHE_HINT_EVICT_FIRST = 0x12F0000000000000  # cute::TMA::CacheHintSm100::EVICT_FIRST: peer rows are read once


def _address_value(pointer_or_address, *, loc=None, ip=None):
    if isinstance(pointer_or_address, Int64):
        return pointer_or_address.ir_value()
    return pointer_or_address.toint(loc=loc, ip=ip).ir_value()


@dsl_user_op
def tma_load_1d(dst_smem, src_gmem, mbar_smem, num_bytes, *, loc=None, ip=None) -> None:
    """cp.async.bulk gmem -> smem of num_bytes, completing on the smem mbarrier; src_gmem a pointer or Int64."""
    llvm.inline_asm(
        None,
        [
            dst_smem.toint(loc=loc, ip=ip).ir_value(),
            _address_value(src_gmem, loc=loc, ip=ip),
            num_bytes.ir_value(),
            mbar_smem.toint(loc=loc, ip=ip).ir_value(),
            Int64(TMA_CACHE_HINT_EVICT_FIRST).ir_value(),
        ],
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [$0], [$1], $2, [$3], $4;",
        "r,l,r,r,l",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def cp_async_bulk_s2g(dst_gmem, src_smem, num_bytes, *, loc=None, ip=None) -> None:
    """cp.async.bulk smem -> gmem in the current bulk group (the caller commits and waits the group)."""
    llvm.inline_asm(
        None,
        [
            dst_gmem.toint(loc=loc, ip=ip).ir_value(),
            src_smem.toint(loc=loc, ip=ip).ir_value(),
            num_bytes.ir_value(),
        ],
        "cp.async.bulk.global.shared::cta.bulk_group [$0], [$1], $2;",
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def red_add_release_gpu_s32(counter_ptr, value: Int32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [counter_ptr.toint(loc=loc, ip=ip).ir_value(), value.ir_value()],
        "red.release.gpu.global.add.s32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def red_add_release_sys_s32(addr: Int64, value: Int32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [addr.ir_value(), value.ir_value()],
        "red.release.sys.global.add.s32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def red_add_relaxed_sys_u64_raw(addr: Int64, val: Int64, *, loc=None, ip=None) -> None:
    """Relaxed sibling of the release add: the caller's barrier and a later release publish order it."""
    llvm.inline_asm(
        None,
        [addr.ir_value(), val.ir_value()],
        "red.relaxed.sys.global.add.u64 [$0], $1;",
        "l,l",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def stg_b32_raw(addr: Int64, val: Int32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [addr.ir_value(), val.ir_value()],
        "st.global.u32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def stg_b64_raw(addr: Int64, val: Int64, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [addr.ir_value(), val.ir_value()],
        "st.global.u64 [$0], $1;",
        "l,l",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def fns_b32(mask: Int32, base: Int32, n: Int32, *, loc=None, ip=None) -> Int32:
    """Bit position of the n-th set bit of mask at or after base (fns.b32)."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [mask.ir_value(), base.ir_value(), n.ir_value()],
            "fns.b32 $0, $1, $2, $3;",
            "=r,r,r,r",
            has_side_effects=False,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )
    )


_FINISH_SUM_TAG = 0x80000000


@dsl_user_op
@cute.jit
def software_grid_sync(counter_ptr, sm_idx, num_sms, tid_in_group, *, num_threads=None, loc=None, ip=None):
    """Grid-wide barrier over num_sms resident CTAs (DeepGEMM barrier.cuh grid_sync): one u32 slot whose bit 31 is
    the round phase and whose low bits count arrivals; CTA 0 adds (tag - (num_sms - 1)), the others 1, so the slot
    flips bit 31 once every CTA arrived and is primed for the next round (no reset). tid_in_group = the thread's
    index inside the participating group (its 0 issues the add and spins); num_threads = the group's size on named
    barrier 10 (None: the whole CTA on bar.sync 0)."""
    if num_threads is None:
        cute.arch.sync_threads(loc=loc, ip=ip)
    else:
        cute.arch.barrier(barrier_id=10, number_of_threads=num_threads, loc=loc, ip=ip)

    if cutlass.const_expr(isinstance(num_sms, int)):
        sm_zero_bits = (_FINISH_SUM_TAG - (num_sms - 1)) & 0xFFFFFFFF
        if cutlass.const_expr(sm_zero_bits >= 0x80000000):
            sm_zero_bits -= 0x100000000
        sm_zero_delta = Int32(sm_zero_bits)
    else:
        sm_zero_delta = Int32(-_FINISH_SUM_TAG) - (Int32(num_sms) - Int32(1))
    other_delta = Int32(1)

    if cutlass.const_expr(hasattr(counter_ptr, "iterator")):
        counter_ptr = counter_ptr.iterator
    llvm.inline_asm(
        None,
        [
            counter_ptr.toint(loc=loc, ip=ip).ir_value(),
            Int32(sm_idx).ir_value(),
            sm_zero_delta.ir_value(),
            other_delta.ir_value(),
            Int32(tid_in_group).ir_value(),
        ],
        (
            "{\n\t"
            ".reg .b32 %delta; .reg .b32 %old; .reg .b32 %cur;\n\t"
            ".reg .pred %not_leader; .reg .pred %is_sm0; .reg .pred %waiting;\n\t"
            "setp.ne.u32 %not_leader, $4, 0;\n\t"
            "@%not_leader bra DONE;\n\t"
            "setp.eq.u32 %is_sm0, $1, 0;\n\t"
            "selp.b32 %delta, $2, $3, %is_sm0;\n\t"
            "atom.release.gpu.global.add.u32 %old, [$0], %delta;\n\t"
            "SPIN:\n\t"
            "ld.relaxed.gpu.global.b32 %cur, [$0];\n\t"
            "xor.b32 %cur, %cur, %old;\n\t"
            "and.b32 %cur, %cur, 0x80000000;\n\t"
            "setp.eq.u32 %waiting, %cur, 0;\n\t"
            "@%waiting bra SPIN;\n\t"
            "fence.acq_rel.gpu;\n\t"
            "DONE:\n\t"
            "}"
        ),
        "l,r,r,r,r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )

    if cutlass.const_expr(num_threads is None):
        cute.arch.sync_threads(loc=loc, ip=ip)
    else:
        cute.arch.barrier(barrier_id=10, number_of_threads=num_threads, loc=loc, ip=ip)
