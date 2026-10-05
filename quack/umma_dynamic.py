# Copyright (c) 2025, Tri Dao.
"""Runtime-N UMMA for SM100: issue tcgen05.mma.cta_group::2 (kind::f16, or kind::mxf8f6f4.block_scale for MXFP8)
with hand-built descriptors so a varlen_n tile runs its block's valid rows (16-granular) without per-size compiled
atoms (megamoe's update_instr_desc_with_umma_n). Operands K-major, 128 B swizzle, f32 accumulator: bf16 / f16 (one
64-wide k-block is 4 MMAs of K = 16) or e4m3 with ue8m0 scales per 32 (one 128-wide k-block is 4 MMAs of K = 32).
The smem descriptor encoding was checked against the DSL's make_umma_smem_desc (SBO 1024 B, LBO 16 B, version 1,
SWIZZLE_128B); the instruction descriptors follow cute/arch/mma_sm100_desc.hpp."""

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


@cute.jit
def instr_desc_mxf8f6f4(n: Int32, m: int) -> Int32:
    """kind::mxf8f6f4 instruction descriptor: e4m3 A and B (format 0), ue8m0 scales, K-major, f32 accumulate, M x N;
    the SF ids (bits 4-5, 29-30) are filled per k-block by umma_mxf8f6f4_2sm."""
    desc = Int32(1 << 23)  # scale format ue8m0
    return desc | ((n >> 3) << 17) | Int32((m >> 4) << 24)


@cute.jit
def umma_mxf8f6f4_2sm(
    tmem_addr: Int32, a_desc: Int64, b_desc: Int64, idesc: Int32, sfa_addr: Int32, sfb_addr: Int32, accumulate: Int32
):
    """One tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale from the leader CTA's elected thread; sfa_addr /
    sfb_addr = the k-block's scale factors in tmem as the DSL's SF tensors address them: a 32-bit tmem word holds 4
    k-blocks' ue8m0 scales and the k-block's byte rides the address's bits 30-31, which the instruction wants as the
    descriptor's SF ids (CUTLASS make_runtime_instr_desc_block_scaled); accumulate != 0 adds to D."""
    idesc = idesc | (((sfa_addr >> 30) & 3) << 29) | (((sfb_addr >> 30) & 3) << 4)
    _umma_mxf8f6f4_2sm(tmem_addr, a_desc, b_desc, idesc, sfa_addr & 0x3FFFFFFF, sfb_addr & 0x3FFFFFFF, accumulate)


@dsl_user_op
def _umma_mxf8f6f4_2sm(
    tmem_addr: Int32, a_desc: Int64, b_desc: Int64, idesc: Int32, sfa_addr: Int32, sfb_addr: Int32,
    accumulate: Int32, *, loc=None, ip=None,
):
    llvm.inline_asm(
        None,
        [tmem_addr.ir_value(loc=loc, ip=ip), a_desc.ir_value(loc=loc, ip=ip), b_desc.ir_value(loc=loc, ip=ip),
         idesc.ir_value(loc=loc, ip=ip), sfa_addr.ir_value(loc=loc, ip=ip), sfb_addr.ir_value(loc=loc, ip=ip),
         accumulate.ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $6, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale [$0], $1, $2, $3, [$4], [$5], p;\n}",
        "r,l,l,r,r,r,r",
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
