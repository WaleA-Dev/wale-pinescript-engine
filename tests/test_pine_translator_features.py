"""Feature tests for translator capabilities added by the corpus stress-test:
for-loops, compound assignment, single-line user functions, switch expressions,
input.source, TV-correct ta.cci, trade-count builtins, stateful ta.barssince,
and hoisted-scalar broadcasting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.pine_translator.pipeline import TranslationPipeline
from src.strategies.pine_base import PineStrategy, _pine_range
from src import pine_ta as ta


def _df(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    close = 100 + np.cumsum(rng.normal(0.05, 1.0, n))
    close = np.maximum(close, 5.0)
    open_ = np.concatenate(([close[0]], close[:-1]))
    high = np.maximum(open_, close) + 0.5
    low = np.minimum(open_, close) - 0.5
    idx = pd.date_range("2021-01-01", periods=n, freq="D")
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": np.full(n, 1e5)}, index=idx)


def _translate_run(pine: str, n: int = 300):
    r = TranslationPipeline().translate(pine, auto_save=False)
    assert r["success"] is True, r["issues"]
    ns: dict = {}
    exec(compile(r["python_code"], "<t>", "exec"), ns)
    cls = [o for o in ns.values() if isinstance(o, type)
           and issubclass(o, PineStrategy) and o is not PineStrategy][0]
    run = cls().run(_df(n))
    assert run["bar_errors"] == 0, run["first_error"]
    return r, run


def test_pine_range_matches_pine_for_semantics():
    assert _pine_range(0, 9) == [float(x) for x in range(10)]      # inclusive
    assert _pine_range(10, 0) == [float(x) for x in range(10, -1, -1)]  # auto down
    assert _pine_range(0, 1, 0.5) == [0.0, 0.5, 1.0]               # float step


def test_for_loop_translates_and_runs():
    pine = """
//@version=6
strategy("Loop", overlay=true)
sumv = 0.0
for i = 0 to 9
    sumv := sumv + close[i]
avg10 = sumv / 10
if close > avg10 and close[1] <= avg10
    strategy.entry("L", strategy.long)
if close < avg10
    strategy.close("L")
"""
    r, run = _translate_run(pine)
    assert "_pine_range(" in r["python_code"]
    assert not any("not supported" in w for w in r["warnings"]), r["warnings"]
    assert len(run["trades"]) > 0


def test_compound_assignment_desugars_to_reassign():
    pine = """
//@version=6
strategy("Plus", overlay=true)
var bars_held = 0
if strategy.position_size > 0
    bars_held += 1
else
    bars_held := 0
if ta.crossover(ta.ema(close, 5), ta.ema(close, 20))
    strategy.entry("L", strategy.long)
if bars_held >= 10
    strategy.close("L")
"""
    r, run = _translate_run(pine)
    assert r["warnings"] == [], r["warnings"]
    closed = [t for t in run["trades"] if t["exit_price"]]
    assert closed and all(t["bars_held"] >= 10 for t in closed)


def test_single_line_user_function_inlined():
    pine = """
//@version=6
strategy("Fn", overlay=true)
f(x, len) => ta.ema(x, len)
fast = f(close, 9)
slow = f(close, 21)
if ta.crossover(fast, slow)
    strategy.entry("L", strategy.long)
if ta.crossunder(fast, slow)
    strategy.close("L")
"""
    r, run = _translate_run(pine)
    assert r["warnings"] == [], r["warnings"]
    assert "ta.ema(close, 9.0)" in r["python_code"]     #真 inlined, not na
    assert len(run["trades"]) > 0


def test_switch_expression_becomes_ternary_chain():
    pine = """
//@version=6
strategy("Switch", overlay=true)
mode = input.string("Fast", "Mode", options=["Fast", "Slow"])
len = switch mode
    "Fast" => 5
    "Slow" => 40
    => 10
if ta.crossover(ta.ema(close, len), ta.ema(close, 50))
    strategy.entry("L", strategy.long)
if ta.crossunder(ta.ema(close, len), ta.ema(close, 50))
    strategy.close("L")
"""
    r, run = _translate_run(pine)
    assert not any("switch" in w for w in r["warnings"]), r["warnings"]
    assert len(run["trades"]) > 0


def test_input_source_resolves_to_series():
    pine = """
//@version=6
strategy("Src", overlay=true)
src = input.source(hl2, "Source")
if ta.crossover(ta.ema(src, 5), ta.ema(src, 20))
    strategy.entry("L", strategy.long)
if ta.crossunder(ta.ema(src, 5), ta.ema(src, 20))
    strategy.close("L")
"""
    r, run = _translate_run(pine)
    assert r["warnings"] == [], r["warnings"]
    assert "ta.ema(hl2, 5.0)" in r["python_code"]
    assert len(run["trades"]) > 0


def test_cci_signature_matches_tradingview():
    """ta.cci takes (source, length) — the source is used directly as the
    typical price, per the Pine v6 reference."""
    src = np.array([10.0, 11, 12, 13, 12, 11, 12, 13, 14, 15, 14, 13, 14, 15, 16,
                    17, 16, 15, 16, 17, 18, 19, 18, 17, 18], dtype=float)
    out = ta.cci(src, 20)
    assert out.shape == src.shape
    assert np.isfinite(out[-1])


def test_trade_count_builtins():
    pine = """
//@version=6
strategy("Counts", overlay=true)
if ta.crossover(ta.ema(close, 5), ta.ema(close, 20)) and strategy.closedtrades < 3
    strategy.entry("L", strategy.long)
if ta.crossunder(ta.ema(close, 5), ta.ema(close, 20))
    strategy.close("L")
"""
    r, run = _translate_run(pine)
    assert r["warnings"] == [], r["warnings"]
    assert "b.closedtrades" in r["python_code"]
    closed = [t for t in run["trades"] if t["exit_price"]]
    assert len(closed) <= 3 + 1   # cap enforced (one may still be open)


def test_barssince_stateful_helper():
    pine = """
//@version=6
strategy("BSince", overlay=true)
crossedUp = ta.crossover(ta.ema(close, 5), ta.ema(close, 20))
if crossedUp
    strategy.entry("L", strategy.long)
sinceEntry = ta.barssince(crossedUp and strategy.position_size > 0)
if strategy.position_size > 0 and sinceEntry >= 5
    strategy.close("L")
"""
    r, run = _translate_run(pine)
    assert "self._barssince(" in r["python_code"]
    assert not any("barssince" in w for w in r["warnings"]), r["warnings"]
    assert len(run["trades"]) > 0


def test_hoisted_scalar_broadcasts_to_series():
    """Ternaries over input scalars hoist to 0-d arrays; the runtime must
    broadcast them so s.x[i] access works (corpus failure class)."""
    pine = """
//@version=6
strategy("Scalar", overlay=true)
aggressive = input.bool(false, "Aggressive")
stop_pct = aggressive ? 4.5 : 14.0
if ta.crossover(ta.ema(close, 5), ta.ema(close, 20))
    strategy.entry("L", strategy.long)
if strategy.position_size > 0 and close < strategy.position_avg_price * (1 - stop_pct / 100)
    strategy.close("L")
"""
    r, run = _translate_run(pine)
    assert run["bar_errors"] == 0
    assert len(run["trades"]) > 0
