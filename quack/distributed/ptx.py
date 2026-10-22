# 16-byte gmem primitives the distributed operators share: the relaxed load that polls a mirror (Lamport), the
# plain stores that re-arm a slot or land a row slice in a peer's buffer (Lamport re-arm, combine).
from typing import Optional, Tuple

from cutlass import Int32, Int64
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cute.typing import Pointer
from cutlass.cutlass_dsl import T, dsl_user_op


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
