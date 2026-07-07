"""Tests for the Pine transpiler (lexer/parser/codegen) + pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.pine_translator.parser import PineParser, Assign, IfBlock
from src.pine_translator.pipeline import TranslationPipeline
from src.strategies.pine_base import PineStrategy


PINE_SIMPLE = """
//@version=5
strategy("EMA Cross", overlay=true)

fastLength = input.int(10, "Fast EMA")
slowLength = input.int(30, "Slow EMA")

fastEMA = ta.ema(close, fastLength)
slowEMA = ta.ema(close, slowLength)

longCondition = ta.crossover(fastEMA, slowEMA)
shortCondition = ta.crossunder(fastEMA, slowEMA)

if longCondition
    strategy.entry("Long", strategy.long)
if shortCondition
    strategy.entry("Short", strategy.short)
"""


def _trend_df(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    close = 100 + np.cumsum(rng.normal(0.05, 1.0, n))
    close = np.maximum(close, 5.0)
    open_ = np.concatenate(([close[0]], close[:-1]))
    high = np.maximum(open_, close) + 0.5
    low = np.minimum(open_, close) - 0.5
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": np.full(n, 1000.0)}, index=idx)


def _load_class(code: str):
    ns: dict = {}
    exec(compile(code, "<t>", "exec"), ns)
    for obj in ns.values():
        if isinstance(obj, type) and issubclass(obj, PineStrategy) and obj is not PineStrategy:
            return obj
    raise AssertionError("no strategy class generated")


def test_parse_simple_ema_cross():
    parsed = PineParser(PINE_SIMPLE).parse()
    assert parsed.name == "EMA Cross"
    assert len(parsed.inputs) == 2
    assert parsed.inputs[0].name == "fastLength"
    assert parsed.inputs[0].default == 10
    assigns = [s for s in parsed.statements if isinstance(s, Assign)]
    assert {a.target for a in assigns} >= {"fastEMA", "slowEMA", "longCondition"}
    ifs = [s for s in parsed.statements if isinstance(s, IfBlock)]
    assert len(ifs) == 2


def test_translation_pipeline_generates_runnable_code():
    result = TranslationPipeline().translate(PINE_SIMPLE, auto_save=False)
    assert result["success"] is True, result["issues"]
    assert "PineStrategy" in result["python_code"]
    cls = _load_class(result["python_code"])
    run = cls().run(_trend_df())
    assert run["bar_errors"] == 0
    assert run["metrics"]["num_trades"] > 0


PINE_MULTILINE = """
//@version=6
strategy(
     "Smart Trend Momentum Strategy v6",
     overlay = true,
     initial_capital = 10000
)

lenFast = input.int(20, "Fast", minval=2, maxval=200)
lenSlow = input.int(
    50,
    "Slow",
    minval=5,
    maxval=300,
    tooltip="multiline"
)
"""


def test_parser_handles_multiline_strategy_and_inputs():
    parsed = PineParser(PINE_MULTILINE).parse()
    assert parsed.name == "Smart Trend Momentum Strategy v6"
    assert parsed.meta["initial_capital"] == 10000
    assert any(inp.name == "lenSlow" and inp.default == 50 for inp in parsed.inputs)
    # strategy() argument lines should not leak into statements
    targets = [s.target for s in parsed.statements if isinstance(s, Assign)]
    assert "initial_capital" not in targets


PINE_STATEFUL_CROSS = """
//@version=5
strategy("Cross Gate", initial_capital=10000)
fastLen = input.int(5, "Fast")
slowLen = input.int(20, "Slow")
emaF = ta.ema(close, fastLen)
emaS = ta.ema(close, slowLen)
longCond = ta.crossover(emaF, emaS) and strategy.position_size == 0
if longCond
    strategy.entry("L", strategy.long)
if strategy.position_size > 0
    strategy.exit("X", from_entry="L", stop=strategy.position_avg_price * 0.97, limit=strategy.position_avg_price * 1.03)
"""


def test_ta_subexpression_inside_stateful_condition_is_hoisted():
    """ta.crossover mixed with strategy.position_size must not degrade to na."""
    result = TranslationPipeline().translate(PINE_STATEFUL_CROSS, auto_save=False)
    assert result["success"] is True, result["issues"]
    assert "_hx0" in result["python_code"]  # synthetic hoist present
    cls = _load_class(result["python_code"])
    df = _trend_df()
    run = cls().run(df)
    assert run["bar_errors"] == 0
    # entries must be gated by crossovers, not fire on every flat bar
    from src import pine_ta as ta
    c = df["close"].to_numpy(float)
    n_cross = int(ta.crossover(ta.ema(c, 5), ta.ema(c, 20)).sum())
    assert 0 < run["metrics"]["num_trades"] <= n_cross + 1


def test_var_state_and_trailing_exit_semantics():
    pine = """
//@version=5
strategy("VarState", initial_capital=10000)
var float entryPrice = na
ema9 = ta.ema(close, 9)
goLong = ta.crossover(close, ema9) and strategy.position_size == 0
if goLong
    strategy.entry("L", strategy.long)
    entryPrice := close
if strategy.position_size > 0 and close < entryPrice * 0.97
    strategy.close("L", comment="SL")
"""
    result = TranslationPipeline().translate(pine, auto_save=False)
    assert result["success"] is True, result["issues"]
    assert "VAR_DEFAULTS" in result["python_code"]
    cls = _load_class(result["python_code"])
    run = cls().run(_trend_df())
    assert run["bar_errors"] == 0


def test_translation_pipeline_handles_real_complex_examples():
    root = Path(__file__).resolve().parents[1]
    examples = [
        root / "newstrat.pine",
        root / "examples" / "saty_phase_strategy.pine",
    ]
    pipe = TranslationPipeline()
    for path in examples:
        code = path.read_text(encoding="utf-8")
        result = pipe.translate(code, auto_save=False)
        assert result["success"] is True, f"{path.name}: {result['issues']}"


def test_newstrat_produces_bracketed_trades():
    """The ATR stop/limit exits must actually fire (the old translator dropped them)."""
    root = Path(__file__).resolve().parents[1]
    code = (root / "newstrat.pine").read_text(encoding="utf-8")
    result = TranslationPipeline().translate(code, auto_save=False)
    cls = _load_class(result["python_code"])
    strat = cls()
    assert strat.INITIAL_CAPITAL == 10000
    assert strat.QTY_VALUE == 5
    assert strat.COMMISSION_PCT == 0.05
    run = strat.run(_trend_df(600))
    closed = [t for t in run["trades"] if t["exit_date"] != "OPEN"]
    assert closed, "no closed trades"
    assert all(("stop" in t["reason"] or "limit" in t["reason"]) for t in closed)


def test_tv_default_sizing_is_fixed_one_share():
    """strategy() without qty/capital args must use TV's documented defaults:
    default_qty_type=strategy.fixed, default_qty_value=1, initial_capital=1e6."""
    pine = """
//@version=6
strategy("Bare Defaults", overlay=true)
if ta.crossover(ta.ema(close, 5), ta.ema(close, 20))
    strategy.entry("L", strategy.long)
if ta.crossunder(ta.ema(close, 5), ta.ema(close, 20))
    strategy.close("L")
"""
    result = TranslationPipeline().translate(pine, auto_save=False)
    assert result["success"] is True, result["issues"]
    cls = _load_class(result["python_code"])
    assert cls.QTY_TYPE == "fixed"
    assert cls.QTY_VALUE == 1.0
    assert cls.INITIAL_CAPITAL == 1000000.0


def test_pivothigh_var_history_semantics():
    """ta.pivothigh confirmation lag + `var float x = na` + x[1] history reads.

    high series: pivots at bars 1 (3), 4 (4), 7 (5); each confirmed 1 bar later.
    Entry condition (new pivot higher than previous stored pivot) first becomes
    true on bar 5, so the market entry fills at bar 6's open — Pine semantics.
    """
    pine = """
//@version=6
strategy("VarHist", overlay=false)
var float lastHigh = na
float ph = ta.pivothigh(high, 1, 1)
bool newPivot = not na(ph)
if newPivot
    lastHigh := ph
bool cond = newPivot and not na(lastHigh[1]) and ph > lastHigh[1]
if cond
    strategy.entry("L", strategy.long)
"""
    result = TranslationPipeline().translate(pine, auto_save=False)
    assert result["success"] is True, result["issues"]
    cls = _load_class(result["python_code"])
    assert "lastHigh" in cls.VAR_DEFAULTS

    high = np.array([1, 3, 2, 1, 4, 2, 1, 5, 2], dtype=float)
    n = len(high)
    df = pd.DataFrame({
        "open": high - 0.5, "high": high, "low": high - 1.0,
        "close": high - 0.25, "volume": np.full(n, 100.0),
    }, index=pd.date_range("2021-01-01", periods=n, freq="D"))

    run = cls().run(df)
    assert run["bar_errors"] == 0, run["first_error"]
    trades = run["trades"]
    assert len(trades) == 1
    # cond fires on bar 5 (pivot 4 > stored pivot 3) -> fill at bar 6 open
    assert trades[0]["entry_bar"] == 6


def test_saty_divergence_variant_translates_and_runs():
    """User-class script: typed decls, table.*, display arithmetic, pivothigh
    divergence on a computed oscillator, var history — must translate and run
    with zero bar errors (no silent per-bar exceptions)."""
    root = Path(__file__).resolve().parents[1]
    code = (root / "examples" / "saty_phase_user.pine").read_text(encoding="utf-8")
    result = TranslationPipeline().translate(code, auto_save=False)
    assert result["success"] is True, result["issues"]
    cls = _load_class(result["python_code"])
    # declared explicitly in the script
    assert cls.INITIAL_CAPITAL == 100000.0
    # strategy() omits qty -> TV default: 1 share fixed
    assert cls.QTY_TYPE == "fixed" and cls.QTY_VALUE == 1.0
    run = cls().run(_trend_df(500))
    assert run["bar_errors"] == 0, run["first_error"]


def test_display_types_translate_silently_and_barstate_works():
    """Chart-annotation namespaces (color.*, table.*, label.*) must translate
    with ZERO warnings — they can't affect order flow. barstate.* must compile
    to real backtest semantics, not na."""
    pine = """
//@version=6
strategy("Display Types", overlay=true)
green = color.rgb(0,255,0)
var tbl = table.new(position.top_right, 1, 1)
var lbl = label.new(bar_index, close, "hi")
fast = ta.ema(close, 5)
slow = ta.ema(close, 20)
if barstate.islast
    table.cell(tbl, 0, 0, 'Last bar', bgcolor=green)
if ta.crossover(fast, slow) and not barstate.isfirst
    strategy.entry("L", strategy.long)
if ta.crossunder(fast, slow)
    strategy.close("L")
"""
    result = TranslationPipeline().translate(pine, auto_save=False)
    assert result["success"] is True, result["issues"]
    assert result["warnings"] == [], result["warnings"]
    assert result["manual_review_needed"] is False
    assert "(i == len(s.close) - 1)" in result["python_code"]  # barstate.islast
    cls = _load_class(result["python_code"])
    run = cls().run(_trend_df(300))
    assert run["bar_errors"] == 0, run["first_error"]


def test_saty_divergence_variant_zero_warnings():
    """The user's real-world script must translate completely clean."""
    root = Path(__file__).resolve().parents[1]
    code = (root / "examples" / "saty_phase_user.pine").read_text(encoding="utf-8")
    result = TranslationPipeline().translate(code, auto_save=False)
    assert result["success"] is True, result["issues"]
    assert result["warnings"] == [], result["warnings"]
    assert result["manual_review_needed"] is False


# ── input gates: wrong-language / non-strategy pastes fail with ONE clear issue ──

def test_pipeline_rejects_python_code(tmp_path):
    py_code = """import numpy as np
import pandas as pd
from src.strategies.base import BaseStrategy

class CustomStrategy(BaseStrategy):
    def generate_signals(self, df):
        signal = pd.Series(0, index=df.index)
        return signal
"""
    pipe = TranslationPipeline(strategy_dir=str(tmp_path))
    r = pipe.translate(py_code, auto_save=True)
    assert r["success"] is False
    assert len(r["issues"]) == 1
    assert "Python" in r["issues"][0]
    assert r["warnings"] == []                      # no confusing warning cascade
    assert not list(tmp_path.glob("*.py*"))         # nothing junk written to disk


def test_pipeline_rejects_indicator_scripts(tmp_path):
    pine = """//@version=6
indicator("My RSI", overlay=false)
r = ta.rsi(close, 14)
plot(r)
"""
    pipe = TranslationPipeline(strategy_dir=str(tmp_path))
    r = pipe.translate(pine, auto_save=True)
    assert r["success"] is False
    assert len(r["issues"]) == 1
    assert "indicator()" in r["issues"][0]
    assert not list(tmp_path.glob("*"))


def test_pipeline_rejects_strategy_without_entries(tmp_path):
    pine = """//@version=6
strategy("No Trades", overlay=true)
e = ta.ema(close, 21)
plot(e)
"""
    pipe = TranslationPipeline(strategy_dir=str(tmp_path))
    r = pipe.translate(pine, auto_save=True)
    assert r["success"] is False
    assert "strategy.entry" in r["issues"][0]
    assert not list(tmp_path.glob("*"))


def test_pipeline_rejects_empty_and_garbage_input(tmp_path):
    pipe = TranslationPipeline(strategy_dir=str(tmp_path))
    r = pipe.translate("   \n  ", auto_save=True)
    assert r["success"] is False and "Nothing to translate" in r["issues"][0]
    r = pipe.translate("hello world this is not code", auto_save=True)
    assert r["success"] is False and len(r["issues"]) == 1
    assert not list(tmp_path.glob("*"))


def test_pipeline_saves_only_on_success(tmp_path):
    pine = """//@version=6
strategy("Save Me", overlay=true)
if ta.crossover(ta.ema(close, 5), ta.ema(close, 20))
    strategy.entry("L", strategy.long)
"""
    pipe = TranslationPipeline(strategy_dir=str(tmp_path))
    r = pipe.translate(pine, auto_save=True)
    assert r["success"] is True, r["issues"]
    assert (tmp_path / "save_me.py").exists()
    assert (tmp_path / "save_me.pine").exists()     # original Pine kept for viewer
