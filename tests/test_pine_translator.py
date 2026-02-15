"""Tests for universal Pine parser + translation pipeline."""

from __future__ import annotations

from pathlib import Path

from src.pine_translator.parser import PineParser
from src.pine_translator.pipeline import TranslationPipeline


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


def test_parse_simple_ema_cross():
    strategy = PineParser(PINE_SIMPLE).parse()
    assert strategy.name == "EMA Cross"
    assert len(strategy.inputs) == 2
    assert strategy.inputs[0].name == "fastLength"
    assert strategy.inputs[0].default == 10
    assert len(strategy.indicators) >= 2
    assert len(strategy.entries) == 2


def test_translation_pipeline_generates_code():
    result = TranslationPipeline().translate(PINE_SIMPLE, auto_save=False)
    assert result["success"] is True
    assert "generate_signals" in result["python_code"]
    assert "BaseStrategy" in result["python_code"]


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
    assert any(inp.name == "lenSlow" and inp.default == 50 for inp in parsed.inputs)
    # strategy() argument lines should not leak into conditions.
    assert all(cond.name != "initial_capital" for cond in parsed.conditions)


def test_translation_pipeline_handles_real_complex_examples():
    root = Path(__file__).resolve().parents[1]
    examples = [
        root / "examples" / "new.pine",
        root / "newstrat.pine",
        root / "examples" / "saty_phase_strategy.pine",
    ]
    pipe = TranslationPipeline()
    for path in examples:
        code = path.read_text(encoding="utf-8")
        result = pipe.translate(code, auto_save=False)
        assert result["success"] is True, f"{path.name}: {result['issues']}"
        assert result["manual_review_needed"] is False
