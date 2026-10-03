# Copyright (c) 2025, Tri Dao.
"""Runtime-N UMMA for SM100: issue tcgen05.mma.cta_group::2.kind::f16 with hand-built descriptors so a varlen_n
tile runs its block's valid rows (16-granular) without per-size compiled atoms (megamoe's update_instr_desc_with_umma_n).
Operands: bf16 / f16 A and B, both K-major, 128 B swizzle, f32 accumulator; one 64-wide k-block is 4 MMAs of K = 16.
The smem descriptor encoding was checked against the DSL's make_umma_smem_desc (SBO 1024 B, LBO 16 B, version 1,
SWIZZLE_128B)."""

from cutlass import Int32, Int64
import cutlass.cute as cute
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm


@cute.jit
def smem_desc_sw128_k(smem_addr: Int32) -> Int64:
    """UMMA smem descriptor for a K-major SW128 tile at smem_addr (bytes): the 8-row x 128 B atoms are 1024 B apart."""
    desc = Int64((smem_addr >> 4) & Int32(0x3FFF))
    desc = desc | (Int64(1) << 16)  # leading byte offset (unused for swizzled K-major; the DSL encodes 1)
    desc = desc | (Int64(1024 >> 4) << 32)  # stride byte offset
    desc = desc | (Int64(1) << 46)  # descriptor version (sm100)
    desc = desc | (Int64(2) << 61)  # SWIZZLE_128B
    return desc


@cute.jit
def instr_desc_f16(n: Int32, m: int, a_bf16: bool, b_bf16: bool) -> Int32:
    """kind::f16 instruction descriptor: f32 accumulate, K-major A and B, M x N."""
    desc = Int32(1 << 4)  # c format f32
    if a_bf16:
        desc = desc | Int32(1 << 7)
    if b_bf16:
        desc = desc | Int32(1 << 10)
    return desc | ((n >> 3) << 17) | Int32((m >> 4) << 24)


@dsl_user_op
def umma_f16_2sm(tmem_addr: Int32, a_desc: Int64, b_desc: Int64, idesc: Int32, accumulate: Int32, *, loc=None, ip=None):
    """One tcgen05.mma.cta_group::2.kind::f16 from the leader CTA's elected thread; accumulate != 0 adds to D."""
    llvm.inline_asm(
        None,
        [tmem_addr.ir_value(loc=loc, ip=ip), a_desc.ir_value(loc=loc, ip=ip), b_desc.ir_value(loc=loc, ip=ip),
         idesc.ir_value(loc=loc, ip=ip), accumulate.ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $4, 0;\n\ttcgen05.mma.cta_group::2.kind::f16 [$0], $1, $2, $3, p;\n}",
        "r,l,l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def tcgen05_fence_after_thread_sync(*, loc=None, ip=None):
    llvm.inline_asm(None, [], "tcgen05.fence::after_thread_sync;", "", has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
