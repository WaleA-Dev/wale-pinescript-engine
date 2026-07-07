"""
Code generator: ParsedScript -> Python strategy source (PineStrategy subclass).

Two-phase translation:
1. Hoisting — top-level assignments whose expressions are pure functions of
   price series / inputs / other hoisted vars (ta.*, math.*, arithmetic) are
   compiled to vectorized numpy in precompute().
2. Everything stateful (var declarations, := mutation, if-blocks, strategy.*
   order calls, position-dependent expressions) is compiled into on_bar(),
   executed per bar against the TradingView-style broker.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

from .parser import (
    Assign, BinOp, Bool, Call, ExprStatement, ForBlock, Ident, IfBlock, Index,
    Na, Num, ParsedScript, PineInput, RawStmt, Str, Ternary, TupleAssign, UnOp,
    Unsupported,
)

BUILTIN_SERIES = {"open", "high", "low", "close", "volume",
                  "hl2", "hlc3", "ohlc4", "hlcc4"}

# Chart-annotation namespaces: they only draw/log on the chart and can never
# affect order flow, so they translate to inert values with no warning.
DISPLAY_CALL_PREFIXES = (
    "table.", "label.", "line.", "box.", "polyline.", "linefill.",
    "color.", "chart.", "log.", "alert", "runtime.error",
)
DISPLAY_IDENT_PREFIXES = (
    "color.", "position.", "size.", "location.", "shape.", "text.",
    "format.", "display.", "extend.", "xloc.", "yloc.", "barmerge.",
    "font.", "scale.", "hline.", "plot.", "label.", "line.", "table.",
    "box.",
)
# Historical-backtest semantics for barstate.*: every bar in a backtest is a
# confirmed historical bar; the dataset's final bar is the "last" bar.
BARSTATE_SCALARS = {
    "barstate.islast": "(i == len(s.close) - 1)",
    "barstate.isfirst": "(i == 0)",
    "barstate.isconfirmed": "True",
    "barstate.ishistory": "True",
    "barstate.isrealtime": "False",
    "barstate.isnew": "True",
    "barstate.islastconfirmedhistory": "(i == len(s.close) - 1)",
}

# ta.* function name -> (vector fn name, signature adapter)
# adapter values: 'plain' (args pass through), or a template describing
# builtin series injections.
TA_PLAIN = {
    "ema", "sma", "rma", "wma", "hma", "swma", "alma", "stdev", "variance",
    "dev", "rsi", "macd", "bb", "highest", "lowest", "mom", "roc", "change",
    "sum", "cum", "avg", "crossover", "crossunder", "cross", "rising",
    "falling", "barssince", "valuewhen", "pivothigh", "pivotlow", "stoch",
    "cci", "linreg",
}
TA_INJECT = {
    "atr": ("atr", ["high", "low", "close"]),        # ta.atr(len)
    "tr": ("tr", ["high", "low", "close"]),
    "wpr": ("wpr", ["high", "low", "close"]),
    "mfi": ("mfi", None),                            # ta.mfi(src, len) -> mfi(src, volume, len)
    "vwma": ("vwma", None),                          # ta.vwma(src, len) -> vwma(src, volume, len)
    "supertrend": ("supertrend", ["high", "low", "close"]),
    "obv": ("obv", ["close", "volume"]),
}

MATH_VECTOR = {
    "math.max": "np.maximum", "math.min": "np.minimum", "math.abs": "np.abs",
    "math.floor": "np.floor", "math.ceil": "np.ceil", "math.round": "np.round",
    "math.sqrt": "np.sqrt", "math.pow": "np.power", "math.exp": "np.exp",
    "math.log": "np.log", "math.log10": "np.log10", "math.sign": "np.sign",
    "math.sin": "np.sin", "math.cos": "np.cos", "math.tan": "np.tan",
    "math.avg": "ta.avg",
    "nz": "__NZ__", "na": "__NA_FN__",
}
MATH_SCALAR = {
    "math.max": "max", "math.min": "min", "math.abs": "abs",
    "math.floor": "math.floor", "math.ceil": "math.ceil", "math.round": "round",
    "math.sqrt": "math.sqrt", "math.pow": "pow", "math.exp": "math.exp",
    "math.log": "math.log", "math.log10": "math.log10",
    "math.sign": "_sign", "math.sin": "math.sin", "math.cos": "math.cos",
    "math.tan": "math.tan", "math.avg": "_avg", "nz": "nz", "na": "na",
    "int": "int", "float": "float", "bool": "bool", "str.tostring": "str",
}

RESERVED = {"i", "s", "b", "p", "df", "n", "np", "ta", "math", "self",
            "NA", "na", "nz", "max", "min", "abs", "open", "high", "low",
            "close", "volume", "True", "False", "None",
            # Python builtins the generated code calls — a Pine variable named
            # `len` (ubiquitous) must not shadow them
            "len", "sum", "range", "int", "float", "bool", "str", "round",
            "pow", "any", "all", "map", "filter", "next", "type", "id"}

STRATEGY_CONSTS = {
    "strategy.long": "1",
    "strategy.short": "-1",
    "strategy.percent_of_equity": "'percent_of_equity'",
    "strategy.fixed": "'fixed'",
    "strategy.cash": "'cash_per_order'",
    "strategy.commission.percent": "'percent'",
    "strategy.direction.long": "1",
    "strategy.direction.short": "-1",
    "strategy.direction.all": "0",
}


class TranslationError(Exception):
    pass


class PineTranslator:
    def __init__(self, parsed: ParsedScript):
        self.script = parsed
        self.warnings: List[str] = list(parsed.warnings)
        self.inputs: Dict[str, PineInput] = {inp.name: inp for inp in parsed.inputs}
        self.functions: Dict[str, Any] = dict(getattr(parsed, "functions", {}) or {})
        self._stateful_ta_keys: List[str] = []
        if self.functions:
            self._expand_functions_in_stmts(parsed.statements)

        self.hoisted: Set[str] = set()
        self.var_state: Set[str] = set()
        self.loop_vars: Set[str] = set()      # top-level per-bar lets not hoisted
        self.block_locals: Set[str] = set()
        self.used_builtins: Set[str] = set()
        self._name_map: Dict[str, str] = {}
        # synthetic hoists: vectorizable ta.* subtrees found inside stateful
        # expressions, lifted into precompute()
        self._synth_by_code: Dict[str, str] = {}   # vec code -> synth name
        self._synth_order: List[tuple] = []        # (name, vec code)

    # ── user-function inlining ───────────────────────────────────────────────

    def _clone_sub(self, node, mapping, depth=0):
        """Deep-copy an expression, substituting param idents and inlining
        calls to single-expression user functions."""
        if node is None or depth > 16:
            return node
        if isinstance(node, Ident):
            return mapping.get(node.name, node)
        if isinstance(node, Call):
            if node.func in self.functions:
                params, fexpr = self.functions[node.func]
                args = [self._clone_sub(a, mapping, depth) for a in node.args]
                return self._clone_sub(fexpr, dict(zip(params, args)), depth + 1)
            return Call(func=node.func,
                        args=[self._clone_sub(a, mapping, depth) for a in node.args],
                        kwargs={k: self._clone_sub(v, mapping, depth)
                                for k, v in node.kwargs.items()})
        if isinstance(node, Index):
            return Index(base=self._clone_sub(node.base, mapping, depth),
                         index=self._clone_sub(node.index, mapping, depth))
        if isinstance(node, BinOp):
            return BinOp(op=node.op, left=self._clone_sub(node.left, mapping, depth),
                         right=self._clone_sub(node.right, mapping, depth))
        if isinstance(node, UnOp):
            return UnOp(op=node.op, operand=self._clone_sub(node.operand, mapping, depth))
        if isinstance(node, Ternary):
            return Ternary(cond=self._clone_sub(node.cond, mapping, depth),
                           if_true=self._clone_sub(node.if_true, mapping, depth),
                           if_false=self._clone_sub(node.if_false, mapping, depth))
        return node

    def _expand_functions_in_stmts(self, stmts):
        for st in stmts:
            if isinstance(st, (Assign, TupleAssign, ExprStatement)):
                st.expr = self._clone_sub(st.expr, {})
            elif isinstance(st, IfBlock):
                st.branches = [
                    (self._clone_sub(c, {}) if c is not None else None, b)
                    for c, b in st.branches]
                for _, b in st.branches:
                    self._expand_functions_in_stmts(b)
            elif isinstance(st, ForBlock):
                st.start = self._clone_sub(st.start, {})
                st.end = self._clone_sub(st.end, {})
                if st.step is not None:
                    st.step = self._clone_sub(st.step, {})
                self._expand_functions_in_stmts(st.body)

    # ── naming ───────────────────────────────────────────────────────────────

    def _py(self, name: str) -> str:
        if name in self._name_map:
            return self._name_map[name]
        py = name if name not in RESERVED else name + "_u"
        py = re.sub(r"\W", "_", py)
        self._name_map[name] = py
        return py

    # ── classification ──────────────────────────────────────────────────────

    def _collect_reassigned(self, stmts, out: Set[str]):
        for st in stmts:
            if isinstance(st, Assign) and st.kind == "reassign":
                out.add(st.target)
            elif isinstance(st, IfBlock):
                for _, body in st.branches:
                    self._collect_reassigned(body, out)
            elif isinstance(st, ForBlock):
                out.add(st.var)
                self._collect_reassigned(st.body, out)

    def _expr_idents(self, node, out: Set[str]):
        if node is None:
            return
        if isinstance(node, Ident):
            out.add(node.name)
        elif isinstance(node, Call):
            out.add(node.func)
            for a in node.args:
                self._expr_idents(a, out)
            for v in node.kwargs.values():
                self._expr_idents(v, out)
        elif isinstance(node, Index):
            self._expr_idents(node.base, out)
            self._expr_idents(node.index, out)
        elif isinstance(node, BinOp):
            self._expr_idents(node.left, out)
            self._expr_idents(node.right, out)
        elif isinstance(node, UnOp):
            self._expr_idents(node.operand, out)
        elif isinstance(node, Ternary):
            self._expr_idents(node.cond, out)
            self._expr_idents(node.if_true, out)
            self._expr_idents(node.if_false, out)
        elif isinstance(node, list):
            for x in node:
                self._expr_idents(x, out)

    def _is_vectorizable(self, node) -> bool:
        idents: Set[str] = set()
        self._expr_idents(node, idents)
        for name in idents:
            if name.startswith("strategy."):
                return False
            if name.startswith("request.") or name.startswith("array.") or \
               name.startswith("matrix.") or name.startswith("map.") or \
               name.startswith("barstate.") or name.startswith("timeframe.") or \
               name.startswith("syminfo.") or name.startswith("time"):
                return False
            if name.startswith("ta."):
                fn = name[3:]
                if fn not in TA_PLAIN and fn not in TA_INJECT:
                    return False
                continue
            if name in MATH_VECTOR or name.startswith("math."):
                continue
            if name in ("na", "nz", "int", "float", "bool"):
                continue
            if name in BUILTIN_SERIES or name == "bar_index":
                continue
            if name in self.inputs:
                continue
            if name in self.hoisted:
                continue
            if name in self.var_state or name in self.loop_vars or name in self.block_locals:
                return False
            # unknown identifier — not safe to hoist
            return False
        return True

    def _classify(self):
        reassigned: Set[str] = set()
        self._collect_reassigned(self.script.statements, reassigned)

        # collect block-local names (assigned with '=' inside if bodies)
        def collect_block_lets(stmts, top: bool):
            for st in stmts:
                if isinstance(st, IfBlock):
                    for _, body in st.branches:
                        collect_block_lets(body, False)
                elif isinstance(st, Assign) and st.kind == "let" and not top:
                    self.block_locals.add(st.target)
        collect_block_lets(self.script.statements, True)

        for st in self.script.statements:
            if isinstance(st, Assign):
                if st.kind == "var":
                    self.var_state.add(st.target)
                elif st.kind == "let":
                    if st.target in reassigned:
                        self.loop_vars.add(st.target)
                    elif self._is_vectorizable(st.expr):
                        self.hoisted.add(st.target)
                    else:
                        self.loop_vars.add(st.target)
            elif isinstance(st, TupleAssign):
                if all(t not in reassigned for t in st.targets) and \
                        self._is_vectorizable(st.expr):
                    self.hoisted.update(st.targets)
                else:
                    self.loop_vars.update(st.targets)

        # find used builtin series everywhere
        idents: Set[str] = set()
        for st in self.script.statements:
            self._stmt_idents(st, idents)
        self.used_builtins = {b for b in BUILTIN_SERIES if b in idents}
        self.used_builtins.update({"open", "high", "low", "close"})  # broker needs these
        for inp in self.inputs.values():
            if inp.type == "source" and str(inp.default) in BUILTIN_SERIES:
                self.used_builtins.add(str(inp.default))

    def _stmt_idents(self, st, out: Set[str]):
        if isinstance(st, Assign):
            self._expr_idents(st.expr, out)
        elif isinstance(st, TupleAssign):
            self._expr_idents(st.expr, out)
        elif isinstance(st, ExprStatement):
            self._expr_idents(st.expr, out)
        elif isinstance(st, IfBlock):
            for cond, body in st.branches:
                self._expr_idents(cond, out)
                for child in body:
                    self._stmt_idents(child, out)
        elif isinstance(st, ForBlock):
            self._expr_idents(st.start, out)
            self._expr_idents(st.end, out)
            if st.step is not None:
                self._expr_idents(st.step, out)
            for child in st.body:
                self._stmt_idents(child, out)

    # ── vector emitter ───────────────────────────────────────────────────────

    def _vec(self, node) -> str:
        if node is None:
            return "np.full(n, np.nan)"
        if isinstance(node, Num):
            return repr(node.value)
        if isinstance(node, Str):
            return repr(node.value)
        if isinstance(node, Bool):
            return "True" if node.value else "False"
        if isinstance(node, Na):
            return "np.nan"
        if isinstance(node, Ident):
            return self._vec_ident(node.name)
        if isinstance(node, UnOp):
            if node.op == "not":
                return f"~np.asarray({self._vec(node.operand)}, dtype=bool)"
            return f"(-{self._vec(node.operand)})"
        if isinstance(node, BinOp):
            l, r = self._vec(node.left), self._vec(node.right)
            if node.op == "and":
                return f"(np.asarray({l}, dtype=bool) & np.asarray({r}, dtype=bool))"
            if node.op == "or":
                return f"(np.asarray({l}, dtype=bool) | np.asarray({r}, dtype=bool))"
            return f"({l} {node.op} {r})"
        if isinstance(node, Ternary):
            return (f"np.where({self._vec(node.cond)}, "
                    f"{self._vec(node.if_true)}, {self._vec(node.if_false)})")
        if isinstance(node, Index):
            base = self._vec(node.base)
            k = self._vec(node.index)
            return f"ta.shift({base}, {k})"
        if isinstance(node, Call):
            return self._vec_call(node)
        raise TranslationError(f"Cannot vectorize node: {node!r}")

    def _vec_ident(self, name: str) -> str:
        if name in BUILTIN_SERIES:
            return {"open": "open_"}.get(name, name)
        if name == "bar_index":
            return "np.arange(n, dtype=float)"
        if name in self.inputs:
            inp = self.inputs[name]
            if inp.type == "source":
                return self._vec_ident(str(inp.default or "close"))
            return f"p[{self._py(name)!r}]"
        if name in self.hoisted:
            return self._py(name)
        if name in STRATEGY_CONSTS:
            return STRATEGY_CONSTS[name]
        raise TranslationError(f"Cannot vectorize identifier '{name}'")

    def _vec_call(self, node: Call) -> str:
        fn = node.func
        if fn.startswith("ta."):
            short = fn[3:]
            if short in TA_INJECT:
                target, inject = TA_INJECT[short]
                args = [self._vec(a) for a in node.args]
                if short in ("mfi", "vwma"):
                    # (src, len) -> (src, volume, len)
                    args = [args[0], "volume"] + args[1:]
                elif inject:
                    injected = [{"open": "open_"}.get(x, x) for x in inject]
                    args = injected + args
                return f"ta.{target}({', '.join(args)})"
            if short in TA_PLAIN:
                args = [self._vec(a) for a in node.args]
                kw = [f"{k}={self._vec(v)}" for k, v in node.kwargs.items()]
                return f"ta.{short}({', '.join(args + kw)})"
            raise TranslationError(f"Unsupported indicator ta.{short}")
        if fn == "na":
            return f"np.isnan(np.asarray({self._vec(node.args[0])}, dtype=float))"
        if fn == "nz":
            x = self._vec(node.args[0])
            repl = self._vec(node.args[1]) if len(node.args) > 1 else "0.0"
            return f"np.where(np.isnan(np.asarray({x}, dtype=float)), {repl}, {x})"
        if fn in MATH_VECTOR:
            mapped = MATH_VECTOR[fn]
            args = ", ".join(self._vec(a) for a in node.args)
            return f"{mapped}({args})"
        if fn.startswith("math."):
            args = ", ".join(self._vec(a) for a in node.args)
            return f"np.{fn[5:]}({args})"
        if fn in ("int", "float", "bool"):
            return f"({self._vec(node.args[0])})"
        raise TranslationError(f"Unsupported function '{fn}' in vector context")

    # ── sub-expression hoisting ──────────────────────────────────────────────

    def _contains_ta(self, node) -> bool:
        if isinstance(node, Call):
            if node.func.startswith("ta."):
                return True
            return (any(self._contains_ta(a) for a in node.args) or
                    any(self._contains_ta(v) for v in node.kwargs.values()))
        if isinstance(node, BinOp):
            return self._contains_ta(node.left) or self._contains_ta(node.right)
        if isinstance(node, UnOp):
            return self._contains_ta(node.operand)
        if isinstance(node, Ternary):
            return (self._contains_ta(node.cond) or self._contains_ta(node.if_true)
                    or self._contains_ta(node.if_false))
        if isinstance(node, Index):
            return self._contains_ta(node.base) or self._contains_ta(node.index)
        return False

    def _hoist_sub(self, node):
        """Lift maximal vectorizable ta.*-containing subtrees out of a stateful
        expression, replacing them with references to synthetic precomputed
        series."""
        if node is None or isinstance(node, (Num, Str, Bool, Na, Ident)):
            return node
        if self._contains_ta(node) and self._is_vectorizable(node):
            try:
                code = self._vec(node)
            except TranslationError:
                return node
            if code in self._synth_by_code:
                name = self._synth_by_code[code]
            else:
                # single underscore: double would be name-mangled inside the class
                name = f"_hx{len(self._synth_order)}"
                self._synth_by_code[code] = name
                self._synth_order.append((name, code))
                self.hoisted.add(name)
            return Ident(name)
        if isinstance(node, BinOp):
            return BinOp(node.op, self._hoist_sub(node.left), self._hoist_sub(node.right))
        if isinstance(node, UnOp):
            return UnOp(node.op, self._hoist_sub(node.operand))
        if isinstance(node, Ternary):
            return Ternary(self._hoist_sub(node.cond), self._hoist_sub(node.if_true),
                           self._hoist_sub(node.if_false))
        if isinstance(node, Index):
            return Index(self._hoist_sub(node.base), node.index)
        if isinstance(node, Call):
            return Call(node.func,
                        [self._hoist_sub(a) for a in node.args],
                        {k: self._hoist_sub(v) for k, v in node.kwargs.items()})
        return node

    def _scal_root(self, node, locals_in_scope: Set[str]) -> str:
        return self._scal(self._hoist_sub(node), locals_in_scope)

    # ── scalar emitter (on_bar) ──────────────────────────────────────────────

    def _scal(self, node, locals_in_scope: Set[str]) -> str:
        if node is None:
            return "NA"
        if isinstance(node, Num):
            return repr(node.value)
        if isinstance(node, Str):
            return repr(node.value)
        if isinstance(node, Bool):
            return "True" if node.value else "False"
        if isinstance(node, Na):
            return "NA"
        if isinstance(node, Ident):
            return self._scal_ident(node.name, locals_in_scope)
        if isinstance(node, UnOp):
            if node.op == "not":
                return f"(not {self._scal(node.operand, locals_in_scope)})"
            return f"(-{self._scal(node.operand, locals_in_scope)})"
        if isinstance(node, BinOp):
            l = self._scal(node.left, locals_in_scope)
            r = self._scal(node.right, locals_in_scope)
            op = {"and": "and", "or": "or"}.get(node.op, node.op)
            return f"({l} {op} {r})"
        if isinstance(node, Ternary):
            return (f"({self._scal(node.if_true, locals_in_scope)} "
                    f"if {self._scal(node.cond, locals_in_scope)} "
                    f"else {self._scal(node.if_false, locals_in_scope)})")
        if isinstance(node, Index):
            return self._scal_index(node, locals_in_scope)
        if isinstance(node, Call):
            return self._scal_call(node, locals_in_scope)
        raise TranslationError(f"Cannot translate node: {node!r}")

    def _scal_ident(self, name: str, locals_in_scope: Set[str]) -> str:
        if name in BUILTIN_SERIES:
            return f"s.{name}[i]"
        if name == "bar_index":
            return "i"
        if name == "strategy.position_size":
            return "b.position_size"
        if name == "strategy.position_avg_price":
            return "b.position_avg_price"
        if name == "strategy.equity":
            return "b.equity(s.close[i])"
        if name == "strategy.initial_capital":
            return "b.cfg.initial_capital"
        if name == "strategy.opentrades":
            return "(1 if b.position_size != 0 else 0)"
        if name == "strategy.closedtrades":
            return "b.closedtrades"
        if name == "strategy.wintrades":
            return "b.wintrades"
        if name == "strategy.losstrades":
            return "b.losstrades"
        if name == "strategy.eventrades":
            return "b.eventrades"
        if name in STRATEGY_CONSTS:
            return STRATEGY_CONSTS[name]
        if name in self.inputs:
            inp = self.inputs[name]
            if inp.type == "source":
                return self._scal_ident(str(inp.default or "close"), locals_in_scope)
            return f"p[{self._py(name)!r}]"
        if name in self.hoisted:
            return f"s.{self._py(name)}[i]"
        if name in self.var_state:
            return f"self.v[{self._py(name)!r}]"
        if name in locals_in_scope or name in self.loop_vars or name in self.block_locals:
            return self._py(name)
        if name in BARSTATE_SCALARS:
            return BARSTATE_SCALARS[name]
        if name.startswith(DISPLAY_IDENT_PREFIXES):
            return "NA"  # display-only constant; cannot affect trades
        self.warnings.append(f"Unknown identifier '{name}' treated as na")
        return "NA"

    def _scal_index(self, node: Index, locals_in_scope: Set[str]) -> str:
        if isinstance(node.base, Ident):
            name = node.base.name
            k = self._scal(node.index, locals_in_scope)
            if name == "strategy.position_size":
                return f"b.position_size_at(int({k}))"
            if name in BUILTIN_SERIES:
                return f"_sv(s.{name}, i - int({k}))"
            if name in self.hoisted:
                return f"_sv(s.{self._py(name)}, i - int({k}))"
            if name in self.var_state:
                return f"self.var_prev({self._py(name)!r}, i, int({k}))"
            if name in self.loop_vars or name in self.block_locals:
                self.warnings.append(
                    f"History access on per-bar variable '{name}' uses current value")
                return self._py(name)
        base = self._scal(node.base, locals_in_scope)
        k = self._scal(node.index, locals_in_scope)
        return f"_sv({base}, i - int({k}))" if base.startswith("s.") else base

    def _scal_call(self, node: Call, locals_in_scope: Set[str]) -> str:
        fn = node.func
        if fn in MATH_SCALAR:
            args = ", ".join(self._scal(a, locals_in_scope) for a in node.args)
            return f"{MATH_SCALAR[fn]}({args})"
        if fn.startswith("math."):
            args = ", ".join(self._scal(a, locals_in_scope) for a in node.args)
            return f"math.{fn[5:]}({args})"
        if fn == "ta.barssince" and len(node.args) == 1:
            key = f"bs{len(self._stateful_ta_keys)}"
            self._stateful_ta_keys.append(key)
            cond = self._scal(node.args[0], locals_in_scope)
            return f"self._barssince({key!r}, _truthy({cond}), i)"
        if fn == "ta.valuewhen" and len(node.args) in (2, 3):
            occ = node.args[2] if len(node.args) == 3 else None
            if occ is None or (isinstance(occ, Num) and occ.value == 0):
                key = f"vw{len(self._stateful_ta_keys)}"
                self._stateful_ta_keys.append(key)
                cond = self._scal(node.args[0], locals_in_scope)
                src = self._scal(node.args[1], locals_in_scope)
                return f"self._valuewhen({key!r}, _truthy({cond}), {src})"
        if fn.startswith("ta."):
            self.warnings.append(
                f"ta.{fn[3:]} used in stateful context cannot be computed per-bar — na")
            return "NA"
        if fn == "strategy.equity":
            return "b.equity(s.close[i])"
        if fn.startswith(DISPLAY_CALL_PREFIXES):
            return "NA"  # drawing/logging call; no effect on order flow
        self.warnings.append(f"Unsupported call '{fn}' treated as na")
        return "NA"

    # ── statement emitters ───────────────────────────────────────────────────

    def _emit_strategy_call(self, call: Call, locals_in_scope: Set[str]) -> Optional[str]:
        fn = call.func

        def arg_str(node) -> str:
            return repr(node.value) if isinstance(node, Str) else self._scal_root(node, locals_in_scope)

        if fn == "strategy.entry":
            entry_id = arg_str(call.args[0]) if call.args else "'Long'"
            direction = "1"
            dir_node = call.args[1] if len(call.args) > 1 else call.kwargs.get("direction")
            if dir_node is not None:
                if isinstance(dir_node, Ident) and "short" in dir_node.name:
                    direction = "-1"
                elif isinstance(dir_node, Ident) and "long" in dir_node.name:
                    direction = "1"
                else:
                    direction = self._scal_root(dir_node, locals_in_scope)
            qty = call.kwargs.get("qty")
            qty_part = f", qty={self._scal_root(qty, locals_in_scope)}" if qty is not None else ""
            return f"b.entry({entry_id}, {direction}{qty_part})"

        if fn in ("strategy.close", "strategy.close_all"):
            comment = call.kwargs.get("comment")
            c = f"comment={arg_str(comment)}" if comment is not None else ""
            if fn == "strategy.close_all":
                return f"b.close_all({c})"
            eid = arg_str(call.args[0]) if call.args else "None"
            return f"b.close({eid}{', ' + c if c else ''})"

        if fn == "strategy.exit":
            parts = []
            eid = arg_str(call.args[0]) if call.args else "'Exit'"
            parts.append(eid)
            if len(call.args) > 1:
                parts.append(f"from_entry={arg_str(call.args[1])}")
            for key in ("from_entry", "stop", "limit", "trail_points",
                        "trail_offset", "trail_price", "loss", "profit", "comment"):
                if key in call.kwargs:
                    val = call.kwargs[key]
                    v = arg_str(val) if key in ("from_entry", "comment") else \
                        self._scal_root(val, locals_in_scope)
                    if key == "trail_price":
                        key = "trail_points"  # approximated
                        self.warnings.append("trail_price approximated as trail_points")
                    parts.append(f"{key}={v}")
            return f"b.exit({', '.join(parts)})"

        if fn in ("strategy.cancel", "strategy.cancel_all"):
            return "b.cancel()"

        if fn in ("strategy", "indicator"):
            return None  # declaration, handled as metadata

        if fn.startswith("strategy."):
            self.warnings.append(f"Unsupported strategy call '{fn}' skipped")
            return None
        return None

    def _emit_stmts(self, stmts: List[Any], indent: str,
                    locals_in_scope: Set[str], lines: List[str]):
        emitted_any = False
        for st in stmts:
            if isinstance(st, Assign):
                if st.kind == "let" and st.target in self.hoisted:
                    continue  # already vectorized
                if st.kind == "var":
                    continue  # initialized in VAR_DEFAULTS
                target = st.target
                expr = self._scal_root(st.expr, locals_in_scope)
                if target in self.var_state:
                    lines.append(f"{indent}self.v[{self._py(target)!r}] = {expr}")
                else:
                    lines.append(f"{indent}{self._py(target)} = {expr}")
                    locals_in_scope.add(target)
                emitted_any = True
            elif isinstance(st, TupleAssign):
                if all(t in self.hoisted for t in st.targets):
                    continue
                self.warnings.append(
                    f"Line {st.line_no}: stateful tuple assignment not supported")
            elif isinstance(st, IfBlock):
                first = True
                for cond, body in st.branches:
                    if cond is None:
                        lines.append(f"{indent}else:")
                    else:
                        kw = "if" if first else "elif"
                        lines.append(f"{indent}{kw} _truthy({self._scal_root(cond, locals_in_scope)}):")
                    inner: List[str] = []
                    self._emit_stmts(body, indent + "    ", set(locals_in_scope), inner)
                    if not inner:
                        inner = [f"{indent}    pass"]
                    lines.extend(inner)
                    first = False
                emitted_any = True
            elif isinstance(st, ExprStatement):
                if isinstance(st.expr, Call):
                    code = self._emit_strategy_call(st.expr, locals_in_scope)
                    if code:
                        lines.append(f"{indent}{code}")
                        emitted_any = True
            elif isinstance(st, ForBlock):
                var_py = self._py(st.var)
                self.loop_vars.add(st.var)
                start = self._scal(st.start, locals_in_scope)
                end_ = self._scal(st.end, locals_in_scope)
                step = (self._scal(st.step, locals_in_scope)
                        if st.step is not None else "None")
                lines.append(f"{indent}for {var_py} in _pine_range({start}, {end_}, {step}):")
                inner_for: List[str] = []
                scope_for = set(locals_in_scope)
                scope_for.add(st.var)
                self._emit_stmts(st.body, indent + "    ", scope_for, inner_for)
                if not inner_for:
                    inner_for = [f"{indent}    pass"]
                lines.extend(inner_for)
                emitted_any = True
            elif isinstance(st, RawStmt):
                lines.append(f"{indent}{st.code}")
                emitted_any = True
            elif isinstance(st, Unsupported):
                lines.append(f"{indent}pass  # unsupported: {st.text[:70]}")
        return emitted_any

    # ── top-level generation ─────────────────────────────────────────────────

    def _sanitize_class_name(self, text: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9_ ]+", " ", text).strip()
        if not cleaned:
            return "AutoTranslatedStrategy"
        name = "".join(tok.capitalize() for tok in cleaned.split())
        if not name[0].isalpha():
            name = "S" + name
        return name + "Strategy"

    def _var_default_literal(self, st: Assign) -> str:
        node = st.expr
        if isinstance(node, Num):
            return repr(node.value)
        if isinstance(node, Bool):
            return "True" if node.value else "False"
        if isinstance(node, Str):
            return repr(node.value)
        if isinstance(node, Na):
            return "float('nan')"
        if isinstance(node, UnOp) and node.op == "-" and isinstance(node.operand, Num):
            return repr(-node.operand.value)
        if isinstance(node, Call) and node.func.startswith(DISPLAY_CALL_PREFIXES):
            return "float('nan')"  # display object handle; inert in a backtest
        self.warnings.append(
            f"var '{st.target}' initializer is dynamic; starts as na")
        return "float('nan')"

    def generate_python_code(self) -> str:
        self._classify()
        script = self.script
        cls = self._sanitize_class_name(script.name)
        meta = script.meta

        # TradingView documented defaults (Pine v6 reference, strategy()):
        # default_qty_type=strategy.fixed, default_qty_value=1, initial_capital=1000000
        initial_capital = meta.get("initial_capital") or 1000000.0
        qty_type = meta.get("qty_type") or "fixed"
        qty_value = meta.get("default_qty_value")
        if qty_value is None:
            qty_value = 1.0
        commission = meta.get("commission_value") or 0.0
        pyramiding = meta.get("pyramiding") or 1

        L: List[str] = []
        L.append('"""')
        L.append(f"Auto-transpiled from PineScript: {script.name}")
        L.append("Generated by Wale Pine Engine — event-driven TradingView semantics.")
        L.append('"""')
        L.append("")
        L.append("import math")
        L.append("")
        L.append("import numpy as np")
        L.append("import pandas as pd")
        L.append("")
        L.append("from src import pine_ta as ta")
        L.append("from src.strategies.pine_base import PineStrategy, NA, na, nz, _sv, _truthy, _pine_range")
        L.append("")
        L.append("")
        L.append("def _sign(x):")
        L.append("    return (x > 0) - (x < 0)")
        L.append("")
        L.append("")
        L.append("def _avg(*xs):")
        L.append("    return sum(xs) / len(xs)")
        L.append("")
        L.append("")
        L.append(f"class {cls}(PineStrategy):")
        L.append(f"    \"\"\"{script.name} (auto-transpiled)\"\"\"")
        L.append("")
        L.append(f"    INITIAL_CAPITAL = {float(initial_capital)!r}")
        L.append(f"    QTY_TYPE = {qty_type!r}")
        L.append(f"    QTY_VALUE = {float(qty_value)!r}")
        L.append(f"    COMMISSION_PCT = {float(commission)!r}")
        L.append(f"    PYRAMIDING = {int(pyramiding)}")

        # VAR_DEFAULTS
        var_items = []
        for st in script.statements:
            if isinstance(st, Assign) and st.kind == "var":
                var_items.append(f"{self._py(st.target)!r}: {self._var_default_literal(st)}")
        L.append("    VAR_DEFAULTS = {" + ", ".join(var_items) + "}")
        L.append("")

        # __init__ with params
        L.append("    def __init__(self, **params):")
        L.append("        super().__init__(**params)")
        if script.inputs:
            for inp in script.inputs:
                L.append(f"        self.params.setdefault({self._py(inp.name)!r}, {inp.default!r})")
        else:
            L.append("        pass")
        L.append("")

        # ── phase 1: vectorize user hoisted assignments (may demote on failure)
        vec_lines: List[str] = []
        exported = {b: ({"open": "open_"}.get(b, b)) for b in self.used_builtins}
        for st in script.statements:
            if isinstance(st, Assign) and st.kind == "let" and st.target in self.hoisted:
                try:
                    vec_lines.append(f"        {self._py(st.target)} = {self._vec(st.expr)}")
                    exported[st.target] = self._py(st.target)
                except TranslationError as e:
                    self.hoisted.discard(st.target)
                    self.loop_vars.add(st.target)
                    self.warnings.append(f"Could not vectorize '{st.target}': {e}")
            elif isinstance(st, TupleAssign) and all(t in self.hoisted for t in st.targets):
                try:
                    targets = ", ".join(self._py(t) for t in st.targets)
                    vec_lines.append(f"        {targets} = {self._vec(st.expr)}")
                    for t in st.targets:
                        exported[t] = self._py(t)
                except TranslationError as e:
                    for t in st.targets:
                        self.hoisted.discard(t)
                        self.loop_vars.add(t)
                    self.warnings.append(f"Could not vectorize tuple assign: {e}")

        # ── phase 2: emit on_bar body (collects synthetic ta.* sub-hoists)
        body: List[str] = []
        self._emit_stmts(script.statements, "        ", set(), body)
        if not body:
            body = ["        pass"]

        # synthetic sub-expressions lifted out of stateful code
        for name, code in self._synth_order:
            vec_lines.append(f"        {name} = {code}")
            exported[name] = name

        # ── assemble precompute
        L.append("    def precompute(self, df, p):")
        L.append("        n = len(df)")
        L.append("        close = df['close'].to_numpy(dtype=float)")
        L.append("        open_ = df['open'].to_numpy(dtype=float) if 'open' in df.columns else close")
        L.append("        high = df['high'].to_numpy(dtype=float) if 'high' in df.columns else close")
        L.append("        low = df['low'].to_numpy(dtype=float) if 'low' in df.columns else close")
        L.append("        volume = df['volume'].to_numpy(dtype=float) if 'volume' in df.columns else np.zeros(n)")
        if "hl2" in self.used_builtins:
            L.append("        hl2 = (high + low) / 2.0")
        if "hlc3" in self.used_builtins:
            L.append("        hlc3 = (high + low + close) / 3.0")
        if "ohlc4" in self.used_builtins:
            L.append("        ohlc4 = (open_ + high + low + close) / 4.0")
        if "hlcc4" in self.used_builtins:
            L.append("        hlcc4 = (high + low + close + close) / 4.0")
        L.append("")
        L.extend(vec_lines)
        L.append("")
        # keys must be the s.<attr> names used by on_bar: builtins keep Pine name,
        # hoisted vars use python-safe name
        export_pairs = []
        for name, py in exported.items():
            attr = name if name in BUILTIN_SERIES else self._py(name)
            export_pairs.append(f"{attr!r}: {py}")
        L.append("        return {" + ", ".join(sorted(set(export_pairs))) + "}")
        L.append("")

        # ── on_bar
        L.append("    def on_bar(self, i, s, b, p):")
        L.extend(body)
        L.append("")

        # param grid
        L.append("    def param_grid(self):")
        L.append("        return PARAM_GRID_DEFAULT")
        L.append("")
        L.append("")
        L.append(self._generate_param_grid())
        return "\n".join(L) + "\n"

    def _generate_param_grid(self) -> str:
        candidates: Dict[str, list] = {}
        for inp in self.script.inputs:
            key = self._py(inp.name)
            if inp.type == "int" and isinstance(inp.default, (int, float)):
                base = int(inp.default)
                candidates[key] = sorted({max(2, base - 5), base, base + 5})
            elif inp.type == "float" and isinstance(inp.default, (int, float)):
                base = float(inp.default)
                candidates[key] = [round(base * 0.9, 6), base, round(base * 1.1, 6)]
            elif inp.type == "bool":
                candidates[key] = [inp.default]
            else:
                candidates[key] = [inp.default]

        def grid_size(d):
            size = 1
            for vals in d.values():
                size *= max(1, len(vals))
            return size

        if grid_size(candidates) > 256:
            for idx, key in enumerate(list(candidates.keys())):
                if idx >= 4:
                    vals = candidates[key]
                    candidates[key] = [vals[len(vals) // 2]]

        lines = ["PARAM_GRID_DEFAULT = {"]
        for name, vals in candidates.items():
            lines.append(f"    {name!r}: {vals!r},")
        lines.append("}")
        return "\n".join(lines)
