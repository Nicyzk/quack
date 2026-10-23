"""Fused epilogues for reduce_over_k, quack's way: a plain Python fn over the per-element fp32 sum, lowered onto
quack's EpiOp machinery (quack.epilogue: ops, ComposableEpiMixin, the fn visit loop in visit.py, the fn
fingerprint keyer). The fn contract is @gemm_epilogue's: ``fn(acc, **operands) -> {"D": ..., <outputs/sinks>}``,
operand kinds inferred from the tensors with m = T and n = H ((1, H) row vector, (1, T) per-token column vector,
(T, H) / (1, T, H) tile, python float / 1-element tensor scalar), ``outputs=`` naming aux (T, H) stores and
``reduces=`` the per-tile partial reduces (ColVecReduce over H: partials (1, T, ceil(H / tile_h)), finalized on
return). No pairing / packed / prepass modes: those are GEMM-N notions.

    @reduce_epilogue(reduces={"sqsum": ColVecReduce("sqsum", scaled=True)})
    def rms_partial(acc, res):
        y = acc + res
        return {"D": y, "sqsum": (y, y)}

    sums = reduce_over_k(slots, scores, out, epilogue=rms_partial, epi_args={"res": x, "sqsum": partials})

Reused from quack unchanged: Scalar, RowVecLoad (the per-column vector, cp.async-staged through smem), ColVecReduce
(the per-token reduce: lane shuffle over the row's lanes, smem merge of the row's warps, per-CTA-tile partials). The
three ops below exist because quack's TileLoad / TileStore / ColVecLoad assume the GEMM's TMA staging and
128-row tiles; here a tile operand is one more predicated row load through the kernel's own thread partition and
the CTA has two rows."""
import hashlib
import inspect
from typing import NamedTuple

import cutlass
import cutlass.cute as cute
from cutlass import Float32, const_expr
from quack.compile_utils import make_fake_tensor
from quack.cute_dsl_utils import mlir_namedtuple, torch2cute_dtype_map
from quack.epilogue.frontend import _infer_kind, _pinned_visit_kind
from quack.epilogue.head_rmsnorm import HeadRstd  # noqa: F401 (re-export)
from quack.epilogue.ops import ColVecLoad, ColVecReduce, EpiOp, EpiSmemBytes, RowVecLoad, Scalar  # noqa: F401 (re-exports)
from quack.epilogue.visit import _EpiModMixinBase
from quack.gemm_runtime.identity import function_semantic_key
import quack.copy_utils as copy_utils

from .reduce_over_k import ReduceOverK, compile_reduce_over_k


class TokenVecLoad(ColVecLoad):
    """The (1, T) per-token operand: one fp32 per thread straight from gmem into a broadcast fragment (the CTA's
    rows are far fewer than its threads, so ColVecLoad's cp.async smem stage does not partition)."""

    def smem_bytes(self, arg_tensor, cta_tile_shape_mnk, epi_tile, warp_shape_mnk=None):
        return EpiSmemBytes()

    def smem_struct_field(self, gemm, params):
        return None

    def get_smem_tensor(self, gemm, params, storage_epi):
        return None

    def needs_async_fence(self):
        return False

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        tile_M, tile_N = ctx.tile_M, ctx.tile_N
        gVec = cute.local_tile(self._get_gmem_vec(param, ctx), (tile_M,), (ctx.tile_coord_mnkl[0],))
        row = ctx.partition_for_epilogue_fn(cute.make_identity_tensor((tile_M, tile_N)))[0][0]
        limit = min(ctx.varlen_manager.len_m(ctx.batch_idx) - ctx.tile_coord_mnkl[0] * tile_M, tile_M)
        v = Float32(0.0)
        if row < limit:
            v = Float32(gVec[row])
        bcast = cute.make_rmem_tensor(cute.make_layout((tile_M, tile_N), stride=(1, 0)), Float32)
        frag = cute.make_rmem_tensor(ctx.partition_for_epilogue_fn(bcast).layout, gemm.acc_dtype)
        frag[0] = v  # every element of the (one row, broadcast columns) fragment aliases register 0
        return frag[None, None, None, 0, 0]

    def begin_loop(self, gemm, state, epi_coord):
        return state


class _RowTile(EpiOp):
    """Host schema shared by the (T, H) / (1, T, H) tile operands and outputs."""

    def host_fake_arg(self, key, fctx):
        dtype, ndim = key
        shape = (fctx.l, fctx.m, fctx.n) if ndim == 3 else (fctx.m, fctx.n)
        return make_fake_tensor(dtype, shape, leading_dim=ndim - 1, divisibility=128 // dtype.width)

    def param_fields(self):
        return [(self.name, object, None)]

    def to_params(self, gemm, args):
        return {self.name: getattr(args, self.name)}

    @cute.jit
    def _tile(self, gemm, param, tile_coord_mnkl, tidx):
        mT = param[tile_coord_mnkl[3], None, None] if const_expr(cute.rank(param) == 3) else param
        tXpX, row_ok, _ = gemm.tile_predicate(mT.shape, tile_coord_mnkl, tidx)
        return gemm.tile_partition(mT, tile_coord_mnkl, tidx), tXpX, row_ok


class RowTileLoad(_RowTile):
    """A (T, H) tile operand (a residual): the kernel's predicated row load, same lanes as the x rows."""

    fn_port = "tile"

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        tXgT, tXpX, row_ok = self._tile(gemm, param, ctx.tile_coord_mnkl, ctx.tidx)
        frag = cute.make_rmem_tensor_like(tXgT)
        frag.fill(0.0)
        if row_ok:
            copy_utils.copy(tXgT, frag, pred=tXpX)
        return frag


class RowTileStore(_RowTile):
    """An aux (T, H) output in the tensor's dtype: convert + the kernel's predicated row store."""

    def is_tile_store(self):
        return True

    @cute.jit
    def store(self, gemm, param, frag, tile_coord_mnkl, tidx):
        tXgT, tXpX, row_ok = self._tile(gemm, param, tile_coord_mnkl, tidx)
        out = cute.make_rmem_tensor_like(tXgT)
        out.store(frag.load().to(param.element_type))
        if row_ok:
            copy_utils.copy(out, tXgT, pred=tXpX)


_KIND_TO_OP = {"row": RowVecLoad, "col": TokenVecLoad, "tile": RowTileLoad, "scalar": Scalar}


class _ReduceEpiMixin(_EpiModMixinBase):
    """quack's fn visit (epi_visit_subtile) over the minted class's _epi_ops; no split-K / prepass fields."""

    _extra_param_fields = ()
    epi_to_underlying_arguments = ReduceOverK.epi_to_underlying_arguments


class ReduceEpiMod:
    """A fn epilogue for reduce_over_k: validates the fn signature (EpiMod's rules), fingerprints it, mints one
    kernel class per operand-kind signature and arch, and owns the host launch."""

    def __init__(self, fn, outputs=(), ops=None, reduces=None, outs=None, extra_ops=(), prepass=None, prepass_outs=()):
        self.fn = fn
        self.outputs = tuple(outputs)
        self.ops = dict(ops or {})
        self.sinks = {**dict(reduces or {}), **dict(outs or {})}
        self.extra_ops = tuple(extra_ops)
        # prepass: quack's statistics sweep over the raw sum before the main fn (HeadRstd: per-row rstd in-kernel).
        self.prepass, self.prepass_outs = prepass, tuple(prepass_outs)
        if (prepass is None) != (not self.prepass_outs):
            raise ValueError("prepass= and prepass_outs= come together")
        self.prepass_operand_names = ()
        if prepass is not None:
            psig = list(inspect.signature(prepass).parameters)
            if not psig or psig[0] != "acc":
                raise ValueError("prepass fn must take 'acc' first")
            self.prepass_operand_names = tuple(psig[1:])
        for name, op in self.ops.items():
            if not isinstance(op, EpiOp) or op.name != name:
                raise ValueError(f"op for {name!r} must be an EpiOp named {name!r}")
        for name, op in self.sinks.items():
            if not isinstance(op, EpiOp) or op.fn_port != "sink" or op.name != name:
                raise ValueError(f"sink op for {name!r} must have fn_port == 'sink' and be named {name!r}")
        params = list(inspect.signature(fn).parameters)
        if not params or params[0] != "acc":
            raise ValueError("epilogue fn must take 'acc' first")
        self.operand_names = tuple(params[1:])
        all_names = set(self.operand_names) | set(self.outputs) | set(self.sinks)
        if all_names & {"acc", "D"}:
            raise ValueError("operand/output names may not use the reserved names acc, D")
        if len(set(self.outputs)) != len(self.outputs):
            raise ValueError("duplicate output names")
        for op in self.extra_ops:
            if not isinstance(op, EpiOp) or op.name in all_names:
                raise ValueError(f"extra op {op!r} must be an EpiOp with a fresh name")
        self.semantic_key = (
            function_semantic_key(fn),
            function_semantic_key(prepass) if prepass is not None else None,
            self.prepass_outs,
            self.outputs,
            tuple(op.cache_key() for _, op in sorted(self.ops.items())),
            tuple(op.cache_key() for _, op in sorted(self.sinks.items())),
            tuple(op.cache_key() for op in self.extra_ops),
        )
        self.semantic_digest = hashlib.sha256(repr(self.semantic_key).encode()).hexdigest()
        self._ident = f"{fn.__name__}_{self.semantic_digest[:16]}"
        self._minted = {}

    def mint(self, kind_sig, arch, arg_forms=()):
        """One kernel class per (operand kinds, arch, arg forms); arg_forms = ((name, form), ...) for the ops whose
        host value changes the compiled signature (HeadRstd's int width is a Constexpr field)."""
        key = (kind_sig, arch, arg_forms)
        if key in self._minted:
            return self._minted[key]
        epi_ops = [self.ops[name] if name in self.ops else _KIND_TO_OP[kind](name) for name, kind in kind_sig]
        epi_ops += [RowTileStore(name) for name in self.outputs]
        epi_ops += list(self.sinks.values()) + list(self.extra_ops)
        forms = dict(arg_forms)
        Args = NamedTuple("EpilogueArguments", [(op.name, op.arg_spec_type(const=forms.get(op.name) == "Const")) for op in epi_ops])
        Args.__new__.__defaults__ = (None,) * len(epi_ops)
        prepass_sig = tuple((n, k) for n, k in kind_sig if n in self.prepass_operand_names)
        cls_name = f"ReduceMod_{self._ident}_{'_'.join(k for _, k in kind_sig) or 'none'}{''.join(f'_{n}{f}' for n, f in arg_forms)}_sm{arch}"
        cls = type(
            cls_name,
            (_ReduceEpiMixin, ReduceOverK),
            {
                "_epi_ops": tuple(epi_ops),
                "_epi_mod_fn": staticmethod(self.fn),
                "_epi_mod_operands": kind_sig,
                "_epi_mod_outputs": self.outputs,
                "_epi_mod_sinks": tuple(self.sinks),
                "_epi_mod_prepass_fn": staticmethod(self.prepass) if self.prepass else None,
                "_epi_mod_prepass_operands": prepass_sig,
                "_epi_mod_prepass_outs": self.prepass_outs,
                "EpilogueArguments": mlir_namedtuple(Args),
                "__module__": __name__,
                "__qualname__": cls_name,
            },
        )
        self._minted[key] = cls
        return cls

    def __call__(self, x, w, out, epi_args, arch, **config):
        """reduce_over_k's epilogue launch: x (T, K, H), w (T, K) | None, out (T, H), epi_args by op name."""
        T, K, H = x.shape
        kind_sig = tuple(
            (name, _pinned_visit_kind(self.ops[name]) if name in self.ops else _infer_kind(name, epi_args[name], T, H))
            for name in self.operand_names
        )
        unknown = set(self.prepass_operand_names) - set(self.operand_names)
        if unknown:
            raise ValueError(f"prepass fn reads undeclared operands {sorted(unknown)}")
        for name in self.prepass_outs:
            if name not in set(self.operand_names) | set(self.sinks):
                raise ValueError(f"prepass out {name!r} must be a declared op")
        arg_forms = tuple((n, self.ops[n].host_arg_form(epi_args[n])) for n in self.operand_names if n in self.ops and self.ops[n].host_arg_form(epi_args[n]))
        cls = self.mint(kind_sig, arch, arg_forms)
        names = {op.name for op in cls._epi_ops}
        unknown = set(epi_args) - names
        if unknown:
            raise ValueError(f"epi_args {sorted(unknown)} name no operand, output or reduce of {self.fn.__name__}")
        missing = set(self.operand_names) | set(self.outputs) | set(self.sinks)
        missing -= set(epi_args)
        if missing:
            raise ValueError(f"epi_args is missing {sorted(missing)}")
        probe = cls(torch2cute_dtype_map[x.dtype], K, w is not None, arch, **config)  # the tile the ops validate against
        for op in cls._epi_ops:
            validate = getattr(op, "host_validate", None)
            if validate is not None and epi_args.get(op.name) is not None:
                validate(epi_args[op.name], m=T, n=H, tile_M=probe.tiler_mn[0], tile_N=probe.tiler_mn[1], batch=None, varlen_m=False, epi_args=epi_args)
        epi_keys = tuple((op.name, op.host_arg_key(epi_args.get(op.name))) for op in cls._epi_ops)
        fn = compile_reduce_over_k(torch2cute_dtype_map[x.dtype], K, w is not None, arch, cls=cls, epi_keys=epi_keys, **config)
        args = cls.EpilogueArguments(**{op.name: op.host_call_arg(epi_args.get(op.name), k) for op, (_, k) in zip(cls._epi_ops, epi_keys)})
        fn(x, w, out, args)
        return {name: op.host_finalize(epi_args[name]) for name, op in self.sinks.items() if getattr(op, "host_finalize", None)}


def reduce_epilogue(outputs=(), ops=None, reduces=None, outs=None, extra_ops=(), prepass=None, prepass_outs=()):
    """Decorator: an elementwise fn over the fp32 sum becomes a reduce_over_k epilogue (module docstring)."""
    return lambda fn: ReduceEpiMod(
        fn, outputs=outputs, ops=ops, reduces=reduces, outs=outs, extra_ops=extra_ops, prepass=prepass, prepass_outs=prepass_outs
    )
