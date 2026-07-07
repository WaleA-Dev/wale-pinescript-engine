"""
PineScript parser: lexer + expression AST + indentation-aware statement parser.

Produces a ParsedScript (metadata, inputs, statement tree) that the code
generator turns into an executable Python strategy. Supports the Pine v4/v5/v6
strategy subset that retail strategies actually use: inputs, ta.* indicators,
var state, := mutation, if/else blocks, ternaries, series history access, and
strategy.entry / exit / close order calls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ── Expression AST ───────────────────────────────────────────────────────────


@dataclass
class Num:
    value: float


@dataclass
class Str:
    value: str


@dataclass
class Bool:
    value: bool


@dataclass
class Na:
    pass


@dataclass
class Ident:
    name: str  # possibly dotted: "ta.ema", "strategy.position_size"


@dataclass
class Call:
    func: str
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Index:
    base: Any
    index: Any


@dataclass
class UnOp:
    op: str  # '-', 'not', '+'
    operand: Any


@dataclass
class BinOp:
    op: str
    left: Any
    right: Any


@dataclass
class Ternary:
    cond: Any
    if_true: Any
    if_false: Any


# ── Statement AST ────────────────────────────────────────────────────────────


@dataclass
class Assign:
    target: str
    expr: Any
    kind: str          # 'let' (=), 'var' (var x =), 'reassign' (:=)
    decl_type: str = ""  # float/int/bool/... if given
    line_no: int = -1


@dataclass
class TupleAssign:
    targets: List[str]
    expr: Any
    line_no: int = -1


@dataclass
class IfBlock:
    branches: List[Tuple[Optional[Any], List[Any]]]  # (cond|None for else, body)
    line_no: int = -1


@dataclass
class ExprStatement:
    expr: Any  # usually a Call
    line_no: int = -1


@dataclass
class Unsupported:
    text: str
    reason: str
    line_no: int = -1


@dataclass
class ForBlock:
    var: str
    start: Any
    end: Any
    step: Any                    # None -> auto direction, step 1
    body: List[Any] = field(default_factory=list)
    line_no: int = -1


@dataclass
class RawStmt:
    code: str                    # 'break' / 'continue'
    line_no: int = -1


@dataclass
class PineInput:
    name: str
    type: str       # int/float/bool/string
    default: Any
    title: str


@dataclass
class ParsedScript:
    name: str
    meta: Dict[str, Any]              # strategy() kwargs of interest
    inputs: List[PineInput]
    statements: List[Any]             # ordered statement AST
    warnings: List[str]
    functions: Dict[str, Any] = field(default_factory=dict)  # name -> (params, expr)


# ── Lexer ────────────────────────────────────────────────────────────────────

_TWO_CHAR_OPS = {"==", "!=", "<=", ">=", ":=", "=>"}
_ONE_CHAR_OPS = set("+-*/%<>=?:,()[]")
_KEYWORDS = {"and", "or", "not", "if", "else", "var", "varip", "for", "to", "by",
             "while", "true", "false", "na", "switch", "import", "export",
             "method", "type", "return", "break", "continue"}


@dataclass
class Token:
    kind: str   # NUM STR ID OP KW EOF
    value: str


def tokenize(text: str) -> List[Token]:
    tokens: List[Token] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch in " \t":
            i += 1
            continue
        if ch in "'\"":
            j = i + 1
            buf = []
            while j < n and text[j] != ch:
                if text[j] == "\\" and j + 1 < n:
                    buf.append(text[j + 1])
                    j += 2
                    continue
                buf.append(text[j])
                j += 1
            tokens.append(Token("STR", "".join(buf)))
            i = j + 1
            continue
        if ch.isdigit() or (ch == "." and i + 1 < n and text[i + 1].isdigit()):
            m = re.match(r"\d*\.?\d+(?:[eE][+-]?\d+)?", text[i:])
            tokens.append(Token("NUM", m.group(0)))
            i += m.end()
            continue
        if ch.isalpha() or ch == "_":
            m = re.match(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", text[i:])
            word = m.group(0)
            if word in _KEYWORDS:
                tokens.append(Token("KW", word))
            else:
                tokens.append(Token("ID", word))
            i += m.end()
            continue
        two = text[i:i + 2]
        if two in _TWO_CHAR_OPS:
            tokens.append(Token("OP", two))
            i += 2
            continue
        if ch in _ONE_CHAR_OPS:
            tokens.append(Token("OP", ch))
            i += 1
            continue
        # unknown char (e.g. '#texthex') — skip
        i += 1
    tokens.append(Token("EOF", ""))
    return tokens


# ── Expression parser (Pratt) ────────────────────────────────────────────────


class ExprParser:
    def __init__(self, tokens: List[Token]):
        self.toks = tokens
        self.pos = 0

    def peek(self) -> Token:
        return self.toks[self.pos]

    def next(self) -> Token:
        t = self.toks[self.pos]
        self.pos += 1
        return t

    def expect(self, kind: str, value: Optional[str] = None) -> Token:
        t = self.next()
        if t.kind != kind or (value is not None and t.value != value):
            raise SyntaxError(f"Expected {value or kind}, got {t.kind}:{t.value!r}")
        return t

    def at_op(self, *vals: str) -> bool:
        t = self.peek()
        return t.kind == "OP" and t.value in vals

    def at_kw(self, *vals: str) -> bool:
        t = self.peek()
        return t.kind == "KW" and t.value in vals

    # precedence climbing
    def parse(self) -> Any:
        return self.parse_ternary()

    def parse_ternary(self) -> Any:
        cond = self.parse_or()
        if self.at_op("?"):
            self.next()
            if_true = self.parse_ternary()
            self.expect("OP", ":")
            if_false = self.parse_ternary()
            return Ternary(cond, if_true, if_false)
        return cond

    def parse_or(self) -> Any:
        left = self.parse_and()
        while self.at_kw("or"):
            self.next()
            left = BinOp("or", left, self.parse_and())
        return left

    def parse_and(self) -> Any:
        left = self.parse_not()
        while self.at_kw("and"):
            self.next()
            left = BinOp("and", left, self.parse_not())
        return left

    def parse_not(self) -> Any:
        if self.at_kw("not"):
            self.next()
            return UnOp("not", self.parse_not())
        return self.parse_comparison()

    def parse_comparison(self) -> Any:
        left = self.parse_addsub()
        while self.at_op("==", "!=", "<", "<=", ">", ">="):
            op = self.next().value
            left = BinOp(op, left, self.parse_addsub())
        return left

    def parse_addsub(self) -> Any:
        left = self.parse_muldiv()
        while self.at_op("+", "-"):
            op = self.next().value
            left = BinOp(op, left, self.parse_muldiv())
        return left

    def parse_muldiv(self) -> Any:
        left = self.parse_unary()
        while self.at_op("*", "/", "%"):
            op = self.next().value
            left = BinOp(op, left, self.parse_unary())
        return left

    def parse_unary(self) -> Any:
        if self.at_op("-"):
            self.next()
            return UnOp("-", self.parse_unary())
        if self.at_op("+"):
            self.next()
            return self.parse_unary()
        return self.parse_postfix()

    def parse_postfix(self) -> Any:
        node = self.parse_primary()
        while True:
            if self.at_op("("):
                if not isinstance(node, Ident):
                    raise SyntaxError("Can only call named functions")
                node = self.parse_call(node.name)
            elif self.at_op("["):
                self.next()
                idx = self.parse()
                self.expect("OP", "]")
                node = Index(node, idx)
            else:
                return node

    def parse_call(self, func: str) -> Call:
        self.expect("OP", "(")
        args: List[Any] = []
        kwargs: Dict[str, Any] = {}
        while not self.at_op(")"):
            # kwarg: ID '=' expr  (but not '==')
            t = self.peek()
            nxt = self.toks[self.pos + 1] if self.pos + 1 < len(self.toks) else Token("EOF", "")
            if t.kind == "ID" and nxt.kind == "OP" and nxt.value == "=":
                key = self.next().value
                self.next()  # '='
                kwargs[key] = self.parse()
            else:
                args.append(self.parse())
            if self.at_op(","):
                self.next()
        self.expect("OP", ")")
        return Call(func, args, kwargs)

    def parse_primary(self) -> Any:
        t = self.peek()
        if t.kind == "NUM":
            self.next()
            return Num(float(t.value))
        if t.kind == "STR":
            self.next()
            return Str(t.value)
        if t.kind == "KW" and t.value in ("true", "false"):
            self.next()
            return Bool(t.value == "true")
        if t.kind == "KW" and t.value == "na":
            self.next()
            # na used as function: na(x)
            if self.at_op("("):
                self.next()
                arg = self.parse()
                self.expect("OP", ")")
                return Call("na", [arg])
            return Na()
        if t.kind == "ID":
            self.next()
            return Ident(t.value)
        if t.kind == "OP" and t.value == "(":
            self.next()
            inner = self.parse()
            self.expect("OP", ")")
            return inner
        if t.kind == "OP" and t.value == "[":
            # tuple literal in expressions (rare) — treat as list of exprs
            self.next()
            items = [self.parse()]
            while self.at_op(","):
                self.next()
                items.append(self.parse())
            self.expect("OP", "]")
            return items
        raise SyntaxError(f"Unexpected token {t.kind}:{t.value!r}")


def parse_expression(text: str) -> Any:
    p = ExprParser(tokenize(text))
    node = p.parse()
    return node


# ── Statement / script parser ────────────────────────────────────────────────

_CONT_END = re.compile(
    r"(?:=|:=|\band\b|\bor\b|\bnot\b|[+\-*/%,?:(]|==|!=|<=|>=|<|>)\s*$"
)
_CONT_START = re.compile(r"^\s*(?:\band\b|\bor\b|[?:+*/]|==|!=|<=|>=)")


class PineParser:
    """Parse Pine source into a ParsedScript."""

    def __init__(self, pine_code: str):
        self.code = pine_code or ""
        self.warnings: List[str] = []

    # — public —

    def parse(self) -> ParsedScript:
        self._functions: Dict[str, Any] = {}
        logical = self._logical_lines()
        meta, name = self._extract_meta(logical)
        inputs, logical = self._extract_inputs(logical)
        statements = self._parse_block(logical, 0, len(logical), base_indent=None)
        return ParsedScript(name=name, meta=meta, inputs=inputs,
                            functions=self._functions,
                            statements=statements, warnings=self.warnings)

    # — preprocessing —

    @staticmethod
    def _strip_comment(line: str) -> str:
        out = []
        quote = None
        i = 0
        while i < len(line):
            ch = line[i]
            if quote:
                out.append(ch)
                if ch == quote and line[i - 1] != "\\":
                    quote = None
            elif ch in "'\"":
                quote = ch
                out.append(ch)
            elif ch == "/" and i + 1 < len(line) and line[i + 1] == "/":
                break
            else:
                out.append(ch)
            i += 1
        return "".join(out)

    @staticmethod
    def _indent_of(line: str) -> int:
        n = 0
        for ch in line:
            if ch == " ":
                n += 1
            elif ch == "\t":
                n += 4
            else:
                break
        return n

    @staticmethod
    def _brackets_balanced(text: str) -> bool:
        depth = 0
        quote = None
        for i, ch in enumerate(text):
            if quote:
                if ch == quote and text[i - 1] != "\\":
                    quote = None
            elif ch in "'\"":
                quote = ch
            elif ch in "([":
                depth += 1
            elif ch in ")]":
                depth -= 1
        return depth <= 0

    def _logical_lines(self) -> List[Tuple[int, str, int]]:
        """Return [(indent, text, first_line_no)] with continuations joined."""
        raw = self.code.splitlines()
        stripped = [self._strip_comment(ln).rstrip() for ln in raw]
        out: List[Tuple[int, str, int]] = []
        i = 0
        while i < len(stripped):
            line = stripped[i]
            if not line.strip():
                i += 1
                continue
            indent = self._indent_of(line)
            text = line.strip()
            start = i
            while i + 1 < len(stripped):
                nxt = stripped[i + 1]
                if not nxt.strip():
                    # blank line ends continuation only if brackets are balanced
                    if self._brackets_balanced(text):
                        break
                    i += 1
                    continue
                nxt_indent = self._indent_of(nxt)
                joins = False
                if not self._brackets_balanced(text):
                    joins = True
                elif _CONT_END.search(text) and nxt_indent > indent:
                    joins = True
                elif _CONT_START.match(nxt) and nxt_indent > indent:
                    joins = True
                if not joins:
                    break
                text = text + " " + nxt.strip()
                i += 1
            out.append((indent, text, start + 1))
            i += 1
        return out

    # — metadata & inputs —

    def _extract_meta(self, logical: List[Tuple[int, str, int]]):
        meta: Dict[str, Any] = {}
        name = "UnnamedStrategy"
        for indent, text, ln in logical:
            m = re.match(r"^(strategy|indicator)\s*\(", text)
            if not m:
                continue
            try:
                call = parse_expression(text)
            except SyntaxError:
                break
            if not isinstance(call, Call):
                break
            if call.args and isinstance(call.args[0], Str):
                name = call.args[0].value
            elif "title" in call.kwargs and isinstance(call.kwargs["title"], Str):
                name = call.kwargs["title"].value

            def num(key, default=None):
                v = call.kwargs.get(key)
                if isinstance(v, Num):
                    return v.value
                if isinstance(v, UnOp) and v.op == "-" and isinstance(v.operand, Num):
                    return -v.operand.value
                return default

            # TV documented default when strategy() omits initial_capital
            meta["initial_capital"] = num("initial_capital", 1000000.0)
            meta["default_qty_value"] = num("default_qty_value", None)
            meta["commission_value"] = num("commission_value", 0.0)
            meta["pyramiding"] = int(num("pyramiding", 1) or 1)
            qty_type = call.kwargs.get("default_qty_type")
            if isinstance(qty_type, Ident):
                if "percent_of_equity" in qty_type.name:
                    meta["qty_type"] = "percent_of_equity"
                elif "cash" in qty_type.name:
                    meta["qty_type"] = "cash_per_order"
                else:
                    meta["qty_type"] = "fixed"
            break
        return meta, name

    def _extract_inputs(self, logical):
        inputs: List[PineInput] = []
        remaining: List[Tuple[int, str, int]] = []
        pat = re.compile(r"^([A-Za-z_]\w*)\s*=\s*input(?:\.(\w+))?\s*\(")
        for indent, text, ln in logical:
            m = pat.match(text)
            if not m or indent > 0:
                remaining.append((indent, text, ln))
                continue
            var_name, itype = m.group(1), (m.group(2) or "float")
            if itype in ("timeframe", "color", "session", "symbol"):
                self.warnings.append(
                    f"input.{itype} '{var_name}' treated as its default constant")
            try:
                rhs = text.split("=", 1)[1].strip()
                call = parse_expression(rhs)
            except SyntaxError as e:
                self.warnings.append(f"Could not parse input '{var_name}': {e}")
                continue
            default: Any = None
            title = var_name
            if isinstance(call, Call):
                if call.args:
                    default = self._literal_value(call.args[0])
                elif "defval" in call.kwargs:
                    default = self._literal_value(call.kwargs["defval"])
            if itype == "source":
                _a0 = call.args[0] if isinstance(call, Call) and call.args else None
                default = _a0.name if isinstance(_a0, Ident) else (default or "close")
                title_node = (call.args[1] if len(call.args) > 1 else
                              call.kwargs.get("title"))
                if isinstance(title_node, Str):
                    title = title_node.value
            if itype == "int" and isinstance(default, float):
                default = int(default)
            if itype == "bool" and not isinstance(default, bool):
                default = bool(default)
            if itype in ("string",) and default is None:
                default = ""
            if default is None and itype in ("float", "int"):
                default = 0
            inputs.append(PineInput(name=var_name, type=itype,
                                    default=default, title=title))
        return inputs, remaining

    @staticmethod
    def _literal_value(node) -> Any:
        if isinstance(node, Num):
            return node.value
        if isinstance(node, Str):
            return node.value
        if isinstance(node, Bool):
            return node.value
        if isinstance(node, Na):
            return float("nan")
        if isinstance(node, UnOp) and node.op == "-" and isinstance(node.operand, Num):
            return -node.operand.value
        if isinstance(node, Ident):
            return node.name  # e.g. close — leave symbolic
        return None

    # — statements —

    _IGNORED_CALLS = (
        "plot", "plotshape", "plotchar", "plotarrow", "plotcandle", "plotbar",
        "hline", "bgcolor", "barcolor", "fill", "alert", "alertcondition",
        "label", "line", "box", "table", "polyline", "linefill", "log",
        "max_bars_back", "barcolour", "indicator", "strategy.risk",
    )

    def _parse_block(self, logical, start, end, base_indent) -> List[Any]:
        stmts: List[Any] = []
        i = start
        while i < end:
            indent, text, ln = logical[i]
            if base_indent is None:
                base_indent = indent
            if indent < base_indent:
                break

            # if / else chains
            if re.match(r"^if\b", text):
                branches: List[Tuple[Optional[Any], List[Any]]] = []
                cond_text = text[2:].strip()
                body_start, body_end = self._body_span(logical, i + 1, end, indent)
                branches.append((self._parse_expr_safe(cond_text, ln),
                                 self._parse_block(logical, body_start, body_end, None)))
                i = body_end
                while i < end:
                    e_indent, e_text, e_ln = logical[i]
                    if e_indent != indent:
                        break
                    if re.match(r"^else\s+if\b", e_text):
                        cond_text = e_text[len("else if"):].strip()
                        bs, be = self._body_span(logical, i + 1, end, indent)
                        branches.append((self._parse_expr_safe(cond_text, e_ln),
                                         self._parse_block(logical, bs, be, None)))
                        i = be
                    elif re.match(r"^else\b", e_text):
                        bs, be = self._body_span(logical, i + 1, end, indent)
                        branches.append((None, self._parse_block(logical, bs, be, None)))
                        i = be
                        break
                    else:
                        break
                stmts.append(IfBlock(branches=branches, line_no=ln))
                continue

            if re.match(r"^(for|while|switch)\b", text):
                mfor = re.match(
                    r"^for\s+(?:(?:int|float)\s+)?([A-Za-z_]\w*)\s*=\s*(.+?)\s+to\s+(.+?)"
                    r"(?:\s+by\s+(.+))?$", text)
                if mfor:
                    bs, be = self._body_span(logical, i + 1, end, indent)
                    body = self._parse_block(logical, bs, be, None)
                    stmts.append(ForBlock(
                        var=mfor.group(1),
                        start=self._parse_expr_safe(mfor.group(2), ln),
                        end=self._parse_expr_safe(mfor.group(3), ln),
                        step=(self._parse_expr_safe(mfor.group(4), ln)
                              if mfor.group(4) else None),
                        body=body, line_no=ln))
                    i = be
                    continue
                bs, be = self._body_span(logical, i + 1, end, indent)
                self.warnings.append(
                    f"Line {ln}: '{text.split()[0]}' blocks are not supported — skipped")
                stmts.append(Unsupported(text=text, reason="loop/switch", line_no=ln))
                i = be
                continue

            mfn = re.match(r"^([A-Za-z_]\w*)\s*\(([^)]*)\)\s*=>\s*(.*)$", text)
            if mfn or text.startswith("method "):
                bs, be = self._body_span(logical, i + 1, end, indent)
                if mfn and mfn.group(3).strip() and bs == be and not text.startswith("method "):
                    # single-expression function: record it for call-site inlining
                    params = [pp.split("=")[0].strip().split(" ")[-1]
                              for pp in mfn.group(2).split(",") if pp.strip()]
                    fexpr = self._parse_expr_safe(mfn.group(3).strip(), ln, quiet=True)
                    if fexpr is not None:
                        self._functions[mfn.group(1)] = (params, fexpr)
                        i = be
                        continue
                self.warnings.append(
                    f"Line {ln}: multi-line user functions are not supported — skipped")
                stmts.append(Unsupported(text=text, reason="function", line_no=ln))
                i = be
                continue

            # assignment from a switch expression -> nested ternary
            msw = re.match(r"^([A-Za-z_]\w*)\s*=\s*switch\s+(.+)$", text)
            if msw:
                bs, be = self._body_span(logical, i + 1, end, indent)
                subject = self._parse_expr_safe(msw.group(2), ln)
                arms, default = [], Na()
                for j in range(bs, be):
                    _ai, a_text, a_ln = logical[j]
                    am = re.match(r"^(?:(.+?)\s*)?=>\s*(.+)$", a_text)
                    if not am:
                        continue
                    arm_expr = self._parse_expr_safe(am.group(2), a_ln)
                    if am.group(1) and am.group(1).strip():
                        arms.append((self._parse_expr_safe(am.group(1).strip(), a_ln),
                                     arm_expr))
                    else:
                        default = arm_expr
                out = default
                for pat, ae in reversed(arms):
                    out = Ternary(cond=BinOp(op="==", left=subject, right=pat),
                                  if_true=ae, if_false=out)
                stmts.append(Assign(target=msw.group(1), expr=out, kind="let",
                                    line_no=ln))
                i = be
                continue

            if text in ("break", "continue"):
                stmts.append(RawStmt(code=text, line_no=ln))
                i += 1
                continue

            # tuple assignment: [a, b] = expr
            m = re.match(r"^\[\s*([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)\s*\]\s*=\s*(.+)$", text)
            if m:
                targets = [t.strip() for t in m.group(1).split(",")]
                expr = self._parse_expr_safe(m.group(2), ln)
                stmts.append(TupleAssign(targets=targets, expr=expr, line_no=ln))
                i += 1
                continue

            # assignment: [var|varip] [type] name = / := expr
            m = re.match(
                r"^(?:(var|varip)\s+)?"
                r"(?:(float|int|bool|string|color|line|label|box|table)\s+)?"
                r"([A-Za-z_]\w*)\s*(:=|\+=|-=|\*=|/=|=(?!=))\s*(.+)$",
                text,
            )
            if m and not re.match(r"^(if|else|for|while|switch)\b", text):
                var_kw, decl_type, name, op, rhs = m.groups()
                # skip pure-graphics assignments
                if decl_type in ("color", "line", "label", "box", "table"):
                    i += 1
                    continue
                if op in ("+=", "-=", "*=", "/="):
                    kind = "reassign"
                    expr = BinOp(op=op[0], left=Ident(name=name),
                                 right=self._parse_expr_safe(rhs, ln))
                else:
                    kind = "var" if var_kw else ("reassign" if op == ":=" else "let")
                    expr = self._parse_expr_safe(rhs, ln)
                stmts.append(Assign(target=name, expr=expr, kind=kind,
                                    decl_type=decl_type or "", line_no=ln))
                i += 1
                continue

            # bare call statement
            call_name = text.split("(", 1)[0].strip()
            if any(call_name == ig or call_name.startswith(ig + ".") or
                   call_name.startswith(ig) and ig in ("label", "line", "box", "table", "log")
                   for ig in self._IGNORED_CALLS):
                i += 1
                continue
            expr = self._parse_expr_safe(text, ln, quiet=True)
            if expr is not None:
                stmts.append(ExprStatement(expr=expr, line_no=ln))
            else:
                stmts.append(Unsupported(text=text, reason="unparseable", line_no=ln))
            i += 1
        return stmts

    def _body_span(self, logical, start, end, parent_indent) -> Tuple[int, int]:
        """Find the [start, stop) span of lines more indented than parent."""
        j = start
        while j < end and logical[j][0] > parent_indent:
            j += 1
        return start, j

    def _parse_expr_safe(self, text: str, ln: int, quiet: bool = False):
        try:
            return parse_expression(text)
        except SyntaxError as e:
            if not quiet:
                self.warnings.append(f"Line {ln}: could not parse '{text[:60]}': {e}")
            return None
