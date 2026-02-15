"""
Translate parsed Pine structure into Python strategy code.
"""

from __future__ import annotations

import re
from typing import List, Tuple

from .parser import PineCondition, PineIndicator, PineStrategy


class PineTranslator:
    """Generate Python strategy class source from a PineStrategy AST."""

    def __init__(self, parsed_strategy: PineStrategy):
        self.strategy = parsed_strategy
        self.input_defaults = {inp.name: inp.default for inp in parsed_strategy.inputs}

    def generate_python_code(self) -> str:
        parts: List[str] = []
        parts.append(self._generate_imports())
        parts.append(self._generate_class_header())
        parts.append(self._generate_init())
        parts.append(self._generate_signals_method())
        parts.append(self._generate_indicator_helpers())
        parts.append(self._generate_param_grid())
        return "\n\n".join(parts).strip() + "\n"

    def _generate_imports(self) -> str:
        return (
            "import numpy as np\n"
            "import pandas as pd\n"
            "from src.strategies.base import BaseStrategy"
        )

    def _sanitize_class_name(self, text: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9_ ]+", " ", text).strip()
        if not cleaned:
            return "AutoTranslatedStrategy"
        return "".join(tok.capitalize() for tok in cleaned.split()) + "Strategy"

    def _generate_class_header(self) -> str:
        cls = self._sanitize_class_name(self.strategy.name)
        return (
            f"class {cls}(BaseStrategy):\n"
            f"    \"\"\"\n"
            f"    Auto-translated from PineScript: {self.strategy.name}\n"
            f"\n"
            f"    WARNING: Auto-generated code. Review logic before production use.\n"
            f"    \"\"\""
        )

    def _generate_init(self) -> str:
        lines = ["    def __init__(self, **params):", "        super().__init__(**params)"]
        for inp in self.strategy.inputs:
            default = repr(inp.default)
            lines.append(f"        self.params.setdefault('{inp.name}', {default})")
        if not self.strategy.inputs:
            lines.append("        # No Pine inputs found; using defaults only.")
        return "\n".join(lines)

    def _is_number(self, token: str) -> bool:
        return bool(re.match(r"^-?\d+(\.\d+)?$", token.strip()))

    def _translate_token(self, token: str) -> str:
        """Translate single Pine identifier/literal into Python expression."""
        t = token.strip()
        if t in {"open", "high", "low", "close", "volume"}:
            return f"df['{t}']"
        if t in self.input_defaults:
            return f"self.params.get('{t}', {repr(self.input_defaults[t])})"
        return t

    def _resolve_param_or_literal(self, token: str, default_value=14) -> str:
        t = token.strip()
        if self._is_number(t):
            return t
        if t in self.input_defaults:
            return f"self.params.get('{t}', {repr(self.input_defaults[t])})"
        return t

    def _shift_if_series(self, expr: str) -> str:
        x = expr.strip()
        if (
            self._is_number(x)
            or x.startswith("self.params.get(")
            or x in {"True", "False", "None"}
            or (x.startswith("'") and x.endswith("'"))
            or (x.startswith('"') and x.endswith('"'))
        ):
            return x
        return f"({x}).shift(1)"

    def _translate_indicator(self, ind: PineIndicator) -> str:
        fn = ind.function.lower()
        args = ind.args

        if fn in {"ema", "sma", "rma"} and len(args) >= 2:
            src = self._translate_token(args[0])
            length_expr = self._resolve_param_or_literal(args[1], default_value=14)
            if fn == "ema":
                return f"{src}.ewm(span={length_expr}, adjust=False).mean()"
            if fn == "sma":
                return f"{src}.rolling({length_expr}).mean()"
            return f"{src}.ewm(alpha=1/{length_expr}, adjust=False).mean()"

        if fn in {"highest", "lowest"} and len(args) >= 2:
            src = self._translate_token(args[0])
            length_expr = self._resolve_param_or_literal(args[1], default_value=14)
            op = "max" if fn == "highest" else "min"
            return f"{src}.rolling({length_expr}).{op}()"

        if fn == "atr":
            p = args[0] if args else "14"
            p_expr = self._resolve_param_or_literal(p, default_value=14)
            return f"self._calc_atr(df, {p_expr})"

        if fn == "rsi":
            src = self._translate_token(args[0] if args else "close")
            p = args[1] if len(args) > 1 else "14"
            p_expr = self._resolve_param_or_literal(p, default_value=14)
            return f"self._calc_rsi({src}, {p_expr})"

        if fn == "macd":
            src = self._translate_token(args[0] if args else "close")
            fast = self._resolve_param_or_literal(args[1], 12) if len(args) > 1 else "12"
            slow = self._resolve_param_or_literal(args[2], 26) if len(args) > 2 else "26"
            sig = self._resolve_param_or_literal(args[3], 9) if len(args) > 3 else "9"
            return f"self._calc_macd({src}, fast={fast}, slow={slow}, signal={sig})[0]"

        # Unknown indicator call: keep translation executable.
        return "pd.Series(np.nan, index=df.index, dtype='float64')"

    def _replace_strategy_runtime(self, expr: str) -> str:
        out = expr
        out = re.sub(r"strategy\.position_avg_price\b", "__POS_AVG_PRICE__", out)
        out = re.sub(r"strategy\.position_size\s*\[\s*\d+\s*\]", "0", out)
        out = re.sub(r"strategy\.position_size\b", "0", out)
        out = out.replace("strategy.long", "1")
        out = out.replace("strategy.short", "-1")
        out = out.replace("strategy.percent_of_equity", "1")
        out = out.replace("strategy.commission.percent", "1")
        return out

    def _replace_crosses(self, expr: str) -> str:
        def repl_cross(match):
            a_raw = match.group(1).strip()
            b_raw = match.group(2).strip()
            a = self._translate_token(a_raw)
            b = self._translate_token(b_raw)
            a_prev = self._shift_if_series(a)
            b_prev = self._shift_if_series(b)
            return f"(({a}) > ({b})) & ({a_prev} <= {b_prev})"

        def repl_crossunder(match):
            a_raw = match.group(1).strip()
            b_raw = match.group(2).strip()
            a = self._translate_token(a_raw)
            b = self._translate_token(b_raw)
            a_prev = self._shift_if_series(a)
            b_prev = self._shift_if_series(b)
            return f"(({a}) < ({b})) & ({a_prev} >= {b_prev})"

        expr = re.sub(
            r"(?:ta\.)?crossover\s*\(\s*([^,]+?)\s*,\s*([^)]+?)\s*\)",
            repl_cross,
            expr,
        )
        expr = re.sub(
            r"(?:ta\.)?crossunder\s*\(\s*([^,]+?)\s*,\s*([^)]+?)\s*\)",
            repl_crossunder,
            expr,
        )
        return expr

    def _translate_condition(self, cond: PineCondition) -> str:
        expr = (cond.expression or "").strip()
        if not expr:
            return "pd.Series(False, index=df.index)"

        if "?" in expr and ":" in expr:
            return "False"

        expr = self._replace_strategy_runtime(expr)
        expr = self._replace_crosses(expr)

        expr = expr.replace("math.max(", "np.maximum(")
        expr = expr.replace("math.min(", "np.minimum(")

        # Pine literals / helpers
        expr = re.sub(r"\btrue\b", "True", expr, flags=re.IGNORECASE)
        expr = re.sub(r"\bfalse\b", "False", expr, flags=re.IGNORECASE)
        expr = re.sub(r"\bna\s*\(", "pd.isna(", expr)
        expr = re.sub(r"\bna\b", "np.nan", expr)

        # Pine series indexing: x[1] -> x.shift(1)
        expr = re.sub(r"\b([a-zA-Z_]\w*)\[(\d+)\]", r"\1.shift(\2)", expr)

        # Replace raw OHLC and input symbols.
        # Use negative lookbehind/lookahead to avoid double-wrapping df['close'] etc.
        for raw in ("close", "open", "high", "low", "volume"):
            expr = re.sub(rf"(?<!\[')(?<!\w)\b{raw}\b(?!'\])(?!\w)", f"df['{raw}']", expr)
        expr = expr.replace("__POS_AVG_PRICE__", "df['close']")
        for inp_name, inp_default in self.input_defaults.items():
            expr = re.sub(
                rf"(?<!\[')(?<!['\w])\b{re.escape(inp_name)}\b(?!['\w])(?!'\])",
                f"self.params.get('{inp_name}', {repr(inp_default)})",
                expr,
            )

        # Pine logical operators.
        expr = re.sub(r"\band\b", "&", expr)
        expr = re.sub(r"\bor\b", "|", expr)
        expr = re.sub(r"\bnot\b", "~", expr)
        expr = self._parenthesize_logical(expr)

        # Keep generated code executable for unsupported runtime namespaces.
        if any(tok in expr for tok in ("request.", "array.", "varip", "barstate.", "strategy.")):
            return "False"

        return expr

    def _parenthesize_logical(self, expr: str) -> str:
        """Wrap logical clauses to avoid Python precedence pitfalls with & and |."""
        parts = re.split(r"(\s*[&|]\s*)", expr)
        if len(parts) <= 1:
            return expr
        out: List[str] = []
        for part in parts:
            p = part.strip()
            if not p:
                continue
            if p in {"&", "|"}:
                out.append(p)
                continue
            if p.startswith("(") and p.endswith(")"):
                out.append(p)
            else:
                out.append(f"({p})")
        return " ".join(out)

    def _sorted_assignments(self) -> List[Tuple[int, str, object]]:
        items: List[Tuple[int, str, object]] = []
        for ind in self.strategy.indicators:
            items.append((getattr(ind, "line_no", -1), "indicator", ind))
        for cond in self.strategy.conditions:
            items.append((getattr(cond, "line_no", -1), "condition", cond))
        # When line numbers are equal/missing, keep indicator first.
        items.sort(key=lambda x: (x[0], 0 if x[1] == "indicator" else 1))
        return items

    def _generate_signals_method(self) -> str:
        lines = [
            "    def generate_signals(self, df):",
            '        """Generate vectorized bar-position signal."""',
            "        if df is None or len(df) == 0:",
            "            return pd.Series(dtype='int64')",
            "",
            "        def _as_mask(value):",
            "            if isinstance(value, pd.Series):",
            "                return value.fillna(False).astype(bool)",
            "            if isinstance(value, np.ndarray):",
            "                return pd.Series(value, index=df.index).fillna(False).astype(bool)",
            "            return pd.Series(bool(value), index=df.index)",
            "",
            "        # Computed assignments in source order",
        ]

        for _, kind, obj in self._sorted_assignments():
            if kind == "indicator":
                expr = self._translate_indicator(obj)
                lines.extend(
                    [
                        "        try:",
                        f"            {obj.name} = {expr}",
                        "        except Exception:",
                        f"            {obj.name} = pd.Series(np.nan, index=df.index, dtype='float64')",
                    ]
                )
            else:
                expr = self._translate_condition(obj)
                lines.extend(
                    [
                        "        try:",
                        f"            {obj.name} = {expr}",
                        "        except Exception:",
                        f"            {obj.name} = pd.Series(np.nan, index=df.index, dtype='float64')",
                    ]
                )

        lines.extend(
            [
                "",
                "        signal = pd.Series(0.0, index=df.index)",
            ]
        )

        for entry in self.strategy.entries:
            when = entry.get("when")
            direction = entry.get("direction", "long")
            sig_val = 1 if direction == "long" else -1
            if not when:
                continue
            when_expr = self._translate_condition(PineCondition(name="_tmp_entry", expression=when, line_no=-1))
            lines.extend(
                [
                    "        try:",
                    f"            signal.loc[_as_mask({when_expr})] = {sig_val}",
                    "        except Exception:",
                    "            pass",
                ]
            )

        # Handle close exits as flattening events.
        for ext in self.strategy.exits:
            when = ext.get("when")
            if not when:
                continue
            when_expr = self._translate_condition(PineCondition(name="_tmp_exit", expression=when, line_no=-1))
            lines.extend(
                [
                    "        try:",
                    f"            signal.loc[_as_mask({when_expr})] = 0",
                    "        except Exception:",
                    "            pass",
                ]
            )

        lines.extend(
            [
                "",
                "        signal = signal.replace(0, np.nan).ffill().fillna(0).astype(int)",
                "        return signal",
            ]
        )
        return "\n".join(lines)

    def _generate_indicator_helpers(self) -> str:
        return (
            "    def _calc_atr(self, df, period=14):\n"
            "        high = df['high']\n"
            "        low = df['low']\n"
            "        close = df['close']\n"
            "        tr1 = high - low\n"
            "        tr2 = (high - close.shift(1)).abs()\n"
            "        tr3 = (low - close.shift(1)).abs()\n"
            "        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)\n"
            "        return tr.ewm(alpha=1/period, adjust=False).mean()\n"
            "\n"
            "    def _calc_rsi(self, series, period=14):\n"
            "        delta = series.diff()\n"
            "        gain = delta.where(delta > 0, 0.0)\n"
            "        loss = -delta.where(delta < 0, 0.0)\n"
            "        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()\n"
            "        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()\n"
            "        rs = avg_gain / avg_loss\n"
            "        return 100 - (100 / (1 + rs))\n"
            "\n"
            "    def _calc_macd(self, series, fast=12, slow=26, signal=9):\n"
            "        ema_fast = series.ewm(span=fast, adjust=False).mean()\n"
            "        ema_slow = series.ewm(span=slow, adjust=False).mean()\n"
            "        macd = ema_fast - ema_slow\n"
            "        signal_line = macd.ewm(span=signal, adjust=False).mean()\n"
            "        hist = macd - signal_line\n"
            "        return macd, signal_line, hist"
        )

    def _generate_param_grid(self) -> str:
        candidates = {}
        for inp in self.strategy.inputs:
            if inp.type == "int":
                base = int(inp.default) if isinstance(inp.default, int) else 14
                candidates[inp.name] = [max(2, base - 5), base, base + 5]
            elif inp.type == "float":
                base = float(inp.default) if isinstance(inp.default, (int, float)) else 1.0
                candidates[inp.name] = [
                    round(base * 0.9, 6),
                    round(base, 6),
                    round(base * 1.1, 6),
                ]
            elif inp.type == "bool":
                candidates[inp.name] = [True, False]
            else:
                candidates[inp.name] = [inp.default]

        # Keep default grid tractable for overnight runs.
        def grid_size(d):
            size = 1
            for vals in d.values():
                size *= max(1, len(vals))
            return size

        max_default_grid = 256
        if grid_size(candidates) > max_default_grid:
            keys = list(candidates.keys())
            for idx, key in enumerate(keys):
                if idx >= 4:
                    # Freeze lower-priority dimensions to defaults.
                    candidates[key] = [candidates[key][len(candidates[key]) // 2]]
            # Secondary clamp if still too large.
            for key, vals in list(candidates.items()):
                if grid_size(candidates) <= max_default_grid:
                    break
                if len(vals) > 2:
                    candidates[key] = [vals[1]]

        lines = ["# Auto-generated parameter grid", "PARAM_GRID_DEFAULT = {"]
        if not candidates:
            lines.append("    # No Pine inputs detected.")
        for name, vals in candidates.items():
            lines.append(f"    '{name}': {repr(vals)},")
        lines.append("}")
        return "\n".join(lines)
