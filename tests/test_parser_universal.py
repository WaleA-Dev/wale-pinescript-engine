"""Parser regression tests for universal Pine patterns."""

from src.parser import PineScriptParser


PINE_SNIPPET = """
//@version=6
strategy("Universal Parser Test", overlay=true)

preset = input.string("High WR", "Preset", options=["High WR", "High Sharpe", "Custom"])
custom_sl = input.float(15.0, "Custom Stop")
custom_trail = input.float(1.0, "Custom Trail")
custom_pt = input.float(3.0, "Custom PT")
custom_ob = input.bool(true, "Custom OB")

momentum_ema_fast = input.int(10, "Momentum EMA Fast")
momentum_ema_slow = input.int(30, "Momentum EMA Slow")

float stop_loss_pct = preset == "High WR" ? 12.5 : preset == "High Sharpe" ? 18.0 : custom_sl
float trailing_pct = preset == "High WR" ? 0.40 : preset == "High Sharpe" ? 0.5 : custom_trail
float profit_target_pct = preset == "High WR" ? 4.5 : preset == "High Sharpe" ? 3.0 : custom_pt
bool use_ob_exit = preset == "High WR" ? false : preset == "High Sharpe" ? true : custom_ob

entry_threshold = input.float(-50, "Entry Threshold")

pivot = ta.ema(close, 21)
atr14 = ta.atr(14)
raw_signal = ((close - pivot) / (3.0 * atr14)) * 100
oscillator = ta.ema(raw_signal, 3)
leaving_entry = oscillator[1] <= entry_threshold and oscillator > entry_threshold
leaving_extreme = oscillator[1] <= -50 and oscillator > -110
overbought_exit = oscillator[1] >= 100 and oscillator < 100
if use_ob_exit and overbought_exit
    strategy.close("Long", comment="OB")
"""


def test_parser_handles_typed_preset_ternary_and_oscillator_expressions():
    parser = PineScriptParser(pine_content=PINE_SNIPPET)
    params = parser.parse_params()

    assert params.stop_loss_pct == 12.5
    assert params.trailing_pct == 0.40
    assert params.profit_target_pct == 4.5
    assert params.use_ob_exit is False

    assert params.use_oscillator_entry is True
    assert params.entry_threshold == -50.0
    assert params.use_secondary_osc_entry is True
    assert params.secondary_prev_threshold == -50.0
    assert params.secondary_curr_threshold == -110.0

    # Momentum aliases should not toggle EMA crossover mode.
    assert params.momentum_ema_fast == 10
    assert params.momentum_ema_slow == 30
    assert params.use_ema_crossover is False

