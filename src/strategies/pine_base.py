"""
Base class for transpiled Pine strategies.

Generated subclasses implement:
    precompute(df, p) -> dict[str, np.ndarray]   (vectorized series)
    on_bar(i, s, b, p)                            (per-bar stateful logic)

run() drives the TradingView-style broker loop. generate_signals() adapts the
result to the legacy vectorized interface so validation/optimization still work.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.pine_runtime import Broker, BrokerConfig, build_result
from src.strategies.base import BaseStrategy

NA = float("nan")


def na(x) -> bool:
    """Pine na() for scalars."""
    try:
        return x is None or math.isnan(x)
    except TypeError:
        return False


def nz(x, replacement=0.0):
    return replacement if na(x) else x


def _sv(arr, idx: int):
    """Safe series access: arr[idx] or nan when out of range."""
    if 0 <= idx < len(arr):
        return arr[idx]
    return NA


def _truthy(x) -> bool:
    """Pine condition semantics: na is false, everything else by bool()."""
    try:
        if x is None:
            return False
        if isinstance(x, float) and math.isnan(x):
            return False
        return bool(x)
    except (TypeError, ValueError):
        return False


def _pine_range(start, end, step=None):
    """Pine for-loop counter values: inclusive of `end`, auto direction."""
    a, b = float(start), float(end)
    s = float(step) if step not in (None, 0) else (1.0 if b >= a else -1.0)
    out, x = [], a
    if s > 0:
        while x <= b + 1e-9:
            out.append(x)
            x += s
    else:
        while x >= b - 1e-9:
            out.append(x)
            x += s
    return out


class PineStrategy(BaseStrategy):
    """Event-driven strategy transpiled from PineScript."""

    # Overridden by generated code from the strategy() declaration
    INITIAL_CAPITAL: float = 100000.0
    QTY_TYPE: str = "percent_of_equity"
    QTY_VALUE: float = 100.0
    COMMISSION_PCT: float = 0.0
    PYRAMIDING: int = 1

    # Persistent `var` declarations: name -> initializer (evaluated once)
    VAR_DEFAULTS: Dict[str, Any] = {}

    def precompute(self, df: pd.DataFrame, p: Dict[str, Any]) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def on_bar(self, i: int, s: SimpleNamespace, b: Broker, p: Dict[str, Any]) -> None:
        raise NotImplementedError

    # ── engine ───────────────────────────────────────────────────────────────

    def run(self, df: pd.DataFrame, commission_pct: float | None = None) -> Dict[str, Any]:
        n = len(df)
        cfg = BrokerConfig(
            initial_capital=self.INITIAL_CAPITAL,
            qty_type=self.QTY_TYPE,
            qty_value=self.QTY_VALUE,
            commission_pct=self.COMMISSION_PCT if commission_pct is None else commission_pct,
            pyramiding=self.PYRAMIDING,
        )
        b = Broker(cfg, n)
        if n == 0:
            return build_result(b, df)

        p = dict(self.params)
        self._bs_last: Dict[str, int] = {}
        self._vw_last: Dict[str, Any] = {}
        series = self.precompute(df, p)
        # Pine semantics: every named series is bar-indexed. Hoisted constants
        # (plain numbers, np.where over scalars -> 0-d arrays) must broadcast
        # to full length or s.x[i] access blows up.
        for k, v in list(series.items()):
            arr = np.asarray(v)
            series[k] = np.full(n, arr.item()) if arr.ndim == 0 else arr
        s = SimpleNamespace(**series)

        # persistent var state
        self.v: Dict[str, Any] = {k: (fn() if callable(fn) else fn)
                                  for k, fn in self.VAR_DEFAULTS.items()}
        # per-var history (value at each processed bar) for `x[k]` access
        self._varhist: Dict[str, np.ndarray] = {
            k: np.full(n, np.nan) for k in self.VAR_DEFAULTS
        }

        o = np.asarray(df["open"], dtype=float) if "open" in df.columns else np.asarray(df["close"], dtype=float)
        h = np.asarray(df["high"], dtype=float) if "high" in df.columns else o
        l = np.asarray(df["low"], dtype=float) if "low" in df.columns else o
        c = np.asarray(df["close"], dtype=float)

        bar_errors = 0
        first_error: str | None = None
        for i in range(n):
            b.begin_bar(i, o[i], h[i], l[i], c[i])
            try:
                self.on_bar(i, s, b, p)
            except Exception as exc:
                bar_errors += 1
                if first_error is None:
                    first_error = f"bar {i}: {type(exc).__name__}: {exc}"
            b.end_bar(i, c[i])
            for name, hist in self._varhist.items():
                val = self.v.get(name)
                try:
                    hist[i] = float(val) if val is not None else np.nan
                except (TypeError, ValueError):
                    hist[i] = np.nan

        result = build_result(b, df)
        result["broker"] = b
        result["bar_errors"] = bar_errors
        result["first_error"] = first_error
        return result

    def _barssince(self, key: str, cond: bool, i: int):
        """Stateful ta.barssince: bars since `cond` was last true (na if never)."""
        if cond:
            self._bs_last[key] = i
        last = self._bs_last.get(key)
        return NA if last is None else float(i - last)

    def _valuewhen(self, key: str, cond: bool, value):
        """Stateful ta.valuewhen(cond, src, 0): src value at the most recent
        bar where cond was true (na if never)."""
        if cond:
            self._vw_last[key] = value
        return self._vw_last.get(key, NA)

    def var_prev(self, name: str, i: int, k: int):
        """History access for persistent vars: name[k] at bar i."""
        idx = i - k
        if idx < 0:
            return NA
        return self._varhist[name][idx]

    # ── legacy interface ─────────────────────────────────────────────────────

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        result = self.run(df)
        pos = result["position_hist"]
        return pd.Series(np.sign(pos), index=df.index).astype(int)
