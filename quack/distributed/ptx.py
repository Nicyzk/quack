# gmem primitives the distributed operators share: the 16-byte relaxed load that polls a mirror (Lamport), the
# 16-byte plain stores that re-arm a slot or land a row slice in a peer's buffer (Lamport re-arm, combine), and
# the dispatch pull's row transport (1D bulk copies through smem, the release add that publishes a landed row).
from typing import Optional, Tuple

from cutlass import Int32, Int64
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cute.typing import Pointer
from cutlass.cutlass_dsl import T, dsl_user_op

TMA_CACHE_HINT_EVICT_FIRST = 0x12F0000000000000  # a pulled row is read once: do not keep it in L2


def _address_value(pointer_or_address, *, loc=None, ip=None):
    if isinstance(pointer_or_address, Int64):
        return pointer_or_address.ir_value()
    return pointer_or_address.toint(loc=loc, ip=ip).ir_value()


@dsl_user_op
def tma_load_1d(
    destination_smem, source_gmem, mbarrier_smem, num_bytes, *, loc=None, ip=None
) -> None:
    """A 1D gmem -> smem bulk copy completing on the smem mbarrier (evict-first cache hint)."""
    llvm.inline_asm(
        None,
        [
            destination_smem.toint(loc=loc, ip=ip).ir_value(),
            _address_value(source_gmem, loc=loc, ip=ip),
            num_bytes.ir_value(),
            mbarrier_smem.toint(loc=loc, ip=ip).ir_value(),
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
def cp_async_bulk_s2g(
    destination_gmem,
    source_smem,
    num_bytes,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """A 1D smem -> gmem bulk copy in the current bulk group."""
    llvm.inline_asm(
        None,
        [
            destination_gmem.toint(loc=loc, ip=ip).ir_value(),
            source_smem.toint(loc=loc, ip=ip).ir_value(),
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
def red_add_release_gpu_s32(
    counter_pointer,
    value: Int32,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """A release add on a gpu-scope counter: the publication of landed bytes."""
    llvm.inline_asm(
        None,
        [counter_pointer.toint(loc=loc, ip=ip).ir_value(), value.ir_value()],
        "red.release.gpu.global.add.s32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_relaxed_gpu_global_4xu32(
    ptr: Pointer, *, loc: Optional[ir.Location] = None, ip: Optional[ir.InsertionPoint] = None
) -> Tuple[Int32, Int32, Int32, Int32]:
    ptr_int = ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    regs = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(i32,i32,i32,i32)>"),
        [ptr_int],
        "ld.relaxed.gpu.global.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )
    x, y, z, w = [llvm.extractvalue(T.i32(), regs, [i]) for i in range(4)]
    return x, y, z, w


@dsl_user_op
def st_weak_global_4xu32(
    ptr: Pointer,
    x: Int32,
    y: Int32,
    z: Int32,
    w: Int32,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    st_weak_global_4xu32_at(ptr.toint(loc=loc, ip=ip), x, y, z, w, loc=loc, ip=ip)


@dsl_user_op
def st_weak_global_4xu32_at(
    addr: Int64,
    x: Int32,
    y: Int32,
    z: Int32,
    w: Int32,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """The 16 B store at a raw gmem address (a peer's buffer reached through its base address)."""
    vals = [Int32(v).ir_value(loc=loc, ip=ip) for v in (x, y, z, w)]
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), *vals],
        "st.weak.global.v4.u32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )
