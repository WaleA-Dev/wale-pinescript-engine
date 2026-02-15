"""
PineScript structural parser for universal translation.

This parser extracts strategy structure; it does not execute Pine code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class PineInput:
    """Represents one Pine input parameter."""

    name: str
    type: str
    default: Any
    title: str


@dataclass
class PineIndicator:
    """Represents one ta.* indicator assignment."""

    name: str
    function: str
    args: List[str]
    kwargs: Dict[str, Any]
    line_no: int = -1


@dataclass
class PineCondition:
    """Represents one boolean condition expression assignment."""

    name: str
    expression: str
    line_no: int = -1


@dataclass
class PineStrategy:
    """Container for parsed Pine strategy structure."""

    name: str
    inputs: List[PineInput]
    indicators: List[PineIndicator]
    conditions: List[PineCondition]
    entries: List[Dict[str, Any]]
    exits: List[Dict[str, Any]]


class PineParser:
    """
    Lightweight Pine strategy parser.

    Focuses on common strategy patterns:
    - strategy(...)
    - input.*
    - ta.* indicators
    - condition assignments
    - strategy.entry / strategy.exit / strategy.close
    """

    def __init__(self, pine_code: str):
        self.pine_code = pine_code or ""
        self.lines = [ln.rstrip() for ln in self.pine_code.splitlines()]
        self.clean_lines = [self._strip_inline_comment(ln).rstrip() for ln in self.lines]
        self._line_index = {idx: ln.strip() for idx, ln in enumerate(self.clean_lines)}
        self._strategy_decl_lines = self._locate_strategy_call_lines()
        self._input_call_lines = self._locate_input_call_lines()

    def parse(self) -> PineStrategy:
        return PineStrategy(
            name=self._parse_strategy_name(),
            inputs=self._parse_inputs(),
            indicators=self._parse_indicators(),
            conditions=self._parse_conditions(),
            entries=self._parse_entries(),
            exits=self._parse_exits(),
        )

    def _parse_strategy_name(self) -> str:
        args = self._extract_function_call_args("strategy")
        if args:
            m = re.search(r"['\"]([^'\"]+)['\"]", args)
            if m:
                return m.group(1).strip()
        pat = re.compile(r"strategy\s*\(\s*['\"]([^'\"]+)['\"]")
        for line in self._line_index.values():
            m = pat.search(line)
            if m:
                return m.group(1).strip()
        return "UnnamedStrategy"

    def _parse_inputs(self) -> List[PineInput]:
        out: List[PineInput] = []
        start_pat = re.compile(r"^([a-zA-Z_]\w*)\s*=\s*input\.(int|float|bool|string)\s*\(")
        i = 0
        while i < len(self.clean_lines):
            s = self.clean_lines[i].strip()
            if not s:
                i += 1
                continue
            m = start_pat.match(s)
            if not m:
                i += 1
                continue

            var_name, i_type = m.groups()
            open_idx = self.clean_lines[i].find("(", m.end() - 1)
            if open_idx < 0:
                i += 1
                continue

            inside, end_line = self._extract_parenthesized(i, open_idx)
            if inside is None:
                i += 1
                continue

            args = self._split_args(inside)
            default_raw = args[0].strip() if args else ""
            title_match = re.search(r"['\"]([^'\"]+)['\"]", inside)
            title = title_match.group(1) if title_match else var_name
            default = self._parse_value(default_raw, i_type)
            out.append(PineInput(name=var_name, type=i_type, default=default, title=title))
            i = end_line + 1
        return out

    def _parse_indicators(self) -> List[PineIndicator]:
        out: List[PineIndicator] = []
        # e.g. emaFast = ta.ema(close, 20)
        pat = re.compile(r"^([a-zA-Z_]\w*)\s*=\s*ta\.([a-zA-Z_]\w*)\s*\((.*)\)\s*$")
        for idx, line in self._line_index.items():
            s = line.strip()
            if not s or s.startswith("//"):
                continue
            m = pat.match(s)
            if not m:
                continue
            name, func, arg_text = m.groups()
            if func.lower() in {"crossover", "crossunder", "rising", "falling"}:
                # These are boolean conditions, not indicator series declarations.
                continue
            args = [a.strip() for a in self._split_args(arg_text)] if arg_text.strip() else []
            out.append(PineIndicator(name=name, function=func, args=args, kwargs={}, line_no=idx))
        return out

    def _parse_conditions(self) -> List[PineCondition]:
        out: List[PineCondition] = []
        assign_pat = re.compile(r"^(?:(?:var|float|int|bool|string)\s+)?([a-zA-Z_]\w*)\s*=\s*(.+)$")
        lines = self.clean_lines
        blocked = self._strategy_decl_lines | self._input_call_lines
        i = 0
        while i < len(lines):
            s = lines[i].strip()
            if not s or s.startswith("//"):
                i += 1
                continue
            if i in blocked:
                i += 1
                continue
            if s.startswith("strategy."):
                i += 1
                continue

            # Handle multiline assignments: `name =` followed by indented lines.
            multi_match = re.match(r"^(?:(?:var|float|int|bool|string)\s+)?([a-zA-Z_]\w*)\s*=\s*$", s)
            if multi_match:
                name = multi_match.group(1)
                expr_lines: List[str] = []
                j = i + 1
                while j < len(lines):
                    raw_next = lines[j]
                    stripped = raw_next.strip()
                    if not stripped or stripped.startswith("//"):
                        break
                    if re.match(r"^(?:(?:var|float|int|bool|string)\s+)?[a-zA-Z_]\w*\s*=", stripped):
                        break
                    if stripped.startswith("if "):
                        break
                    if re.match(r"^strategy\.(entry|exit|close)\b", stripped):
                        break
                    expr_lines.append(stripped)
                    j += 1
                if expr_lines:
                    expr = " ".join(expr_lines).strip()
                    ta_call = re.match(r"^ta\.([a-zA-Z_]\w*)\s*\(", expr)
                    if not (ta_call and ta_call.group(1).lower() not in {"crossover", "crossunder", "rising", "falling"}):
                        out.append(PineCondition(name=name, expression=expr, line_no=i))
                i = j
                continue

            m = assign_pat.match(s)
            if not m:
                i += 1
                continue
            name, expr = m.groups()
            expr = expr.strip()
            if expr.startswith("input."):
                # inputs/indicator assignments handled elsewhere
                i += 1
                continue
            ta_call = re.match(r"^ta\.([a-zA-Z_]\w*)\s*\(", expr)
            if ta_call and ta_call.group(1).lower() not in {"crossover", "crossunder", "rising", "falling"}:
                # handled in indicator parser
                i += 1
                continue
            out.append(PineCondition(name=name, expression=expr, line_no=i))
            i += 1
        return out

    def _parse_entries(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        entry_pat = re.compile(r"strategy\.entry\s*\((.+)\)")
        for idx, raw in self._line_index.items():
            s = raw.strip()
            m = entry_pat.search(s)
            if not m:
                continue
            args = self._split_args(m.group(1))
            entry_id = self._strip_quotes(args[0]) if args else f"Entry_{idx}"
            direction = "long"
            when_expr: Optional[str] = None
            for arg in args[1:]:
                a = arg.strip()
                if "strategy.short" in a:
                    direction = "short"
                if a.startswith("when="):
                    when_expr = a.split("=", 1)[1].strip()

            # if <cond> on prior line is a common style.
            prev = self._line_index.get(idx - 1, "").strip()
            if when_expr is None and prev.startswith("if "):
                when_expr = prev[3:].strip()

            out.append({"id": entry_id, "direction": direction, "when": when_expr})
        return out

    def _parse_exits(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        exit_pat = re.compile(r"strategy\.exit\s*\((.+)\)")
        close_pat = re.compile(r"strategy\.close\s*\((.+)\)")
        for idx, raw in self._line_index.items():
            s = raw.strip()

            em = exit_pat.search(s)
            if em:
                args = self._split_args(em.group(1))
                stop = None
                limit = None
                when = None
                for arg in args:
                    a = arg.strip()
                    if a.startswith("stop="):
                        stop = a.split("=", 1)[1].strip()
                    elif a.startswith("limit="):
                        limit = a.split("=", 1)[1].strip()
                    elif a.startswith("when="):
                        when = a.split("=", 1)[1].strip()
                if when is None:
                    prev = self._line_index.get(idx - 1, "").strip()
                    if prev.startswith("if "):
                        when = prev[3:].strip()
                out.append({"type": "exit", "stop": stop, "limit": limit, "when": when})
                continue

            cm = close_pat.search(s)
            if cm:
                args = self._split_args(cm.group(1))
                when = None
                for arg in args:
                    a = arg.strip()
                    if a.startswith("when="):
                        when = a.split("=", 1)[1].strip()
                if when is None:
                    prev = self._line_index.get(idx - 1, "").strip()
                    if prev.startswith("if "):
                        when = prev[3:].strip()
                out.append({"type": "close", "when": when})
        return out

    @staticmethod
    def _strip_quotes(value: str) -> str:
        s = value.strip()
        if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
            return s[1:-1]
        return s

    @staticmethod
    def _split_args(s: str) -> List[str]:
        """
        Split function arg list while respecting nested parentheses and quotes.
        """
        out: List[str] = []
        current: List[str] = []
        depth = 0
        quote: Optional[str] = None
        i = 0
        while i < len(s):
            ch = s[i]
            if quote:
                current.append(ch)
                if ch == quote and (i == 0 or s[i - 1] != "\\"):
                    quote = None
                i += 1
                continue
            if ch in ("'", '"'):
                quote = ch
                current.append(ch)
                i += 1
                continue
            if ch == "(":
                depth += 1
                current.append(ch)
                i += 1
                continue
            if ch == ")":
                depth = max(0, depth - 1)
                current.append(ch)
                i += 1
                continue
            if ch == "," and depth == 0:
                out.append("".join(current).strip())
                current = []
                i += 1
                continue
            current.append(ch)
            i += 1
        tail = "".join(current).strip()
        if tail:
            out.append(tail)
        return out

    @staticmethod
    def _strip_inline_comment(line: str) -> str:
        """Strip // comments while preserving quoted strings."""
        if not line:
            return line
        out: List[str] = []
        quote: Optional[str] = None
        i = 0
        while i < len(line):
            ch = line[i]
            if quote:
                out.append(ch)
                if ch == quote and (i == 0 or line[i - 1] != "\\"):
                    quote = None
                i += 1
                continue
            if ch in ("'", '"'):
                quote = ch
                out.append(ch)
                i += 1
                continue
            if ch == "/" and i + 1 < len(line) and line[i + 1] == "/":
                break
            out.append(ch)
            i += 1
        return "".join(out)

    def _extract_parenthesized(self, start_line: int, open_idx: int) -> tuple[Optional[str], int]:
        """Extract text inside balanced parentheses from a line/column start."""
        depth = 0
        quote: Optional[str] = None
        chunks: List[str] = []
        for i in range(start_line, len(self.clean_lines)):
            line = self.clean_lines[i]
            j = open_idx if i == start_line else 0
            while j < len(line):
                ch = line[j]
                if quote:
                    if ch == quote and (j == 0 or line[j - 1] != "\\"):
                        quote = None
                    if depth > 0 and not (depth == 1 and ch == ")"):
                        chunks.append(ch)
                    j += 1
                    continue
                if ch in ("'", '"'):
                    quote = ch
                    if depth > 0:
                        chunks.append(ch)
                    j += 1
                    continue
                if ch == "(":
                    depth += 1
                    if depth > 1:
                        chunks.append(ch)
                    j += 1
                    continue
                if ch == ")":
                    if depth <= 0:
                        j += 1
                        continue
                    depth -= 1
                    if depth == 0:
                        return "".join(chunks).strip(), i
                    chunks.append(ch)
                    j += 1
                    continue
                if depth > 0:
                    chunks.append(ch)
                j += 1
            if depth > 0:
                chunks.append("\n")
        return None, start_line

    def _extract_function_call_args(self, func_name: str) -> Optional[str]:
        """
        Extract top-level function call arguments with balanced parentheses.
        """
        text = "\n".join(self.clean_lines)
        match = re.search(rf"\b{re.escape(func_name)}\s*\(", text)
        if not match:
            return None

        depth = 0
        start = None
        in_single = False
        in_double = False
        escape = False

        for i in range(match.end() - 1, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if in_single:
                if ch == "'":
                    in_single = False
                continue
            if in_double:
                if ch == '"':
                    in_double = False
                continue
            if ch == "'":
                in_single = True
                continue
            if ch == '"':
                in_double = True
                continue
            if ch == "(":
                depth += 1
                if start is None:
                    start = i + 1
                continue
            if ch == ")":
                depth -= 1
                if depth == 0 and start is not None:
                    return text[start:i]
        return None

    def _locate_strategy_call_lines(self) -> set[int]:
        """Locate all lines that belong to top-level strategy(...) declaration."""
        spans: set[int] = set()
        for i, line in enumerate(self.clean_lines):
            s = line.strip()
            if not s:
                continue
            if re.search(r"(?<!\.)\bstrategy\s*\(", s):
                open_idx = line.find("(")
                _, end = self._extract_parenthesized(i, open_idx)
                for k in range(i, end + 1):
                    spans.add(k)
                break
        return spans

    def _locate_input_call_lines(self) -> set[int]:
        """Locate all lines that belong to multiline input.*(...) blocks."""
        spans: set[int] = set()
        start_pat = re.compile(r"^([a-zA-Z_]\w*)\s*=\s*input\.(int|float|bool|string)\s*\(")
        for i, line in enumerate(self.clean_lines):
            s = line.strip()
            if not s:
                continue
            if not start_pat.match(s):
                continue
            open_idx = line.find("(", line.find("input."))
            inside, end = self._extract_parenthesized(i, open_idx)
            if inside is None:
                continue
            for k in range(i, end + 1):
                spans.add(k)
        return spans

    @staticmethod
    def _parse_value(value_str: str, type_hint: str) -> Any:
        if value_str is None:
            return None
        s = value_str.strip()
        if type_hint == "int":
            try:
                return int(float(s))
            except Exception:
                return 0
        if type_hint == "float":
            try:
                return float(s)
            except Exception:
                return 0.0
        if type_hint == "bool":
            return s.lower() in ("true", "1", "yes", "y")
        return PineParser._strip_quotes(s)
