#!/usr/bin/env python3
"""
Test script to verify the parser and engine work with Saty Phase strategy on SCHD.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from src.parser import PineScriptParser
from src.backtest import BacktestEngine, BacktestConfig

# Minimal Saty Phase PineScript code for parser testing
SATY_PHASE_PINE = '''
//@version=6
strategy("Saty Phase - ANTI-CONSOLIDATION V2", overlay=true, initial_capital=100000, default_qty_type=strategy.percent_of_equity, default_qty_value=100, commission_type=strategy.commission.percent, commission_value=0.1)

preset = input.string("High WR", title="Preset", options=["High WR", "High Sharpe", "Custom"])
use_extreme_entry = input.bool(true, title="Use Extreme Entry")
extreme_threshold = input.float(-110, title="Extreme Threshold")
ema_filter_enabled = input.bool(false, title="Enable EMA Filter")
ema_length = input.int(200, title="EMA Length")
slope_lookback = input.int(200, title="Slope Lookback")
slope_threshold = input.float(2.0, title="Slope Threshold")
consolidation_lookback = input.int(400, title="Consolidation Lookback")
consolidation_threshold = input.float(8.0, title="Consolidation Threshold")
momentum_fast_len = input.int(10, title="Momentum Fast")
momentum_slow_len = input.int(30, title="Momentum Slow")
adx_length = input.int(14, title="ADX Length")
adx_threshold = input.float(20, title="ADX Threshold")

stop_loss_pct = preset == "High WR" ? 14.0 : preset == "High Sharpe" ? 18.0 : 14.0
trailing_step_pct = preset == "High WR" ? 0.40 : preset == "High Sharpe" ? 0.50 : 0.40
profit_target_pct = preset == "High WR" ? 4.5 : preset == "High Sharpe" ? 3.0 : 4.5

ema21 = ta.ema(close, 21)
atr14 = ta.atr(14)
raw_oscillator = ((close - ema21) / (3 * atr14)) * 100
oscillator = ta.ema(raw_oscillator, 3)

entrySignal = ta.crossover(oscillator, -50)

if entrySignal and not is_consolidating
    strategy.entry("Long", strategy.long)

if strategy.position_size > 0
    if close <= entryPrice * (1 - stop_loss_pct / 100)
        strategy.close("Long", comment="SL")
    if ta.crossunder(oscillator, 100)
        strategy.close("Long", comment="OB")
'''

def test_parser():
    """Test that the parser extracts correct parameters."""
    print("=" * 60)
    print("PARSER TEST")
    print("=" * 60)

    parser = PineScriptParser(pine_content=SATY_PHASE_PINE)
    settings = parser.parse_strategy_settings()
    params = parser.parse_params()

    print(f"\n--- Strategy Settings ---")
    print(f"Title: {settings.title}")
    print(f"Initial Capital: ${settings.initial_capital:,.0f}")
    print(f"Qty Type: {settings.default_qty_type}")
    print(f"Qty Value: {settings.default_qty_value}%")
    print(f"Commission: {settings.commission_value}%")

    print(f"\n--- Parsed Parameters ---")
    print(f"Stop Loss: {params.stop_loss_pct}%")
    print(f"Trailing Step: {params.trailing_pct}%")
    print(f"Profit Target (trail activation): {params.profit_target_pct}%")
    print(f"Use Trailing Stop: {params.use_trailing_stop}")
    print(f"Use Oscillator Entry: {params.use_oscillator_entry}")
    print(f"Entry Threshold: {params.entry_threshold}")
    print(f"Use Extreme Entry: {params.use_extreme_entry}")
    print(f"Extreme Threshold: {params.extreme_threshold}")
    print(f"Use OB Exit: {params.use_ob_exit}")
    print(f"OB Threshold: {params.ob_threshold}")
    print(f"Use Consolidation Filter: {params.use_consolidation_filter}")
    print(f"EMA Filter Enabled: {params.use_ema_filter}")
    print(f"EMA Length: {params.ema_length}")
    print(f"ADX Length: {params.adx_length}")
    print(f"ADX Threshold: {params.adx_threshold}")
    print(f"Momentum Fast: {params.momentum_ema_fast}")
    print(f"Momentum Slow: {params.momentum_ema_slow}")
    print(f"Exit Type: {params.exit_type}")

    print(f"\n--- Custom Params ---")
    for k, v in params.custom_params.items():
        print(f"  {k} = {v}")

    # Verify key parameters
    errors = []
    if params.stop_loss_pct != 14.0:
        errors.append(f"stop_loss_pct: expected 14.0, got {params.stop_loss_pct}")
    if params.trailing_pct != 0.40:
        errors.append(f"trailing_pct: expected 0.40, got {params.trailing_pct}")
    if params.profit_target_pct != 4.5:
        errors.append(f"profit_target_pct: expected 4.5, got {params.profit_target_pct}")
    if not params.use_oscillator_entry:
        errors.append("use_oscillator_entry should be True")
    if params.entry_threshold != -50.0:
        errors.append(f"entry_threshold: expected -50.0, got {params.entry_threshold}")
    if not params.use_ob_exit:
        errors.append("use_ob_exit should be True")
    if params.ob_threshold != 100.0:
        errors.append(f"ob_threshold: expected 100.0, got {params.ob_threshold}")
    if not params.use_consolidation_filter:
        errors.append("use_consolidation_filter should be True")
    if params.exit_type != "strategy_close":
        errors.append(f"exit_type: expected 'strategy_close', got '{params.exit_type}'")

    if errors:
        print(f"\nFAILED - {len(errors)} errors:")
        for e in errors:
            print(f"  - {e}")
    else:
        print(f"\nPASSED - All parameters correctly parsed!")

    return params, settings


def test_engine_with_csv():
    """Test the engine with SCHD CSV data if available."""
    print("\n" + "=" * 60)
    print("ENGINE TEST")
    print("=" * 60)

    # Try to find SCHD data
    csv_paths = [
        "data/SCHD_1H_databento.csv",   # SCHD data from Databento
        "data/NDAQ_1H_TV_aligned.csv",   # Fallback: NDAQ data
    ]

    df = None
    for path in csv_paths:
        if os.path.exists(path):
            print(f"Loading data from {path}...")
            df = pd.read_csv(path)
            df['time'] = pd.to_datetime(df['time'])
            print(f"  {len(df)} bars, {df['time'].min()} to {df['time'].max()}")
            break

    if df is None:
        print("No test data available. Skipping engine test.")
        return

    # Parse the Saty Phase strategy
    parser = PineScriptParser(pine_content=SATY_PHASE_PINE)
    params = parser.parse_params()
    settings = parser.parse_strategy_settings()

    # Configure backtest
    config = BacktestConfig(
        initial_capital=settings.initial_capital,
        commission_pct=settings.commission_value,
        order_size_pct=settings.default_qty_value,
        qty_type="percent_of_equity",
    )

    # Debug: check oscillator values before running
    from src.indicators import ema, atr as atr_func
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    osc_ema = ema(close, params.oscillator_ema_len)
    osc_atr = atr_func(high, low, close, params.oscillator_atr_len)
    import numpy as np
    raw_osc = np.full(len(close), np.nan)
    for j in range(len(close)):
        if not np.isnan(osc_ema[j]) and not np.isnan(osc_atr[j]) and osc_atr[j] != 0:
            raw_osc[j] = ((close[j] - osc_ema[j]) / (params.oscillator_atr_mult * osc_atr[j])) * params.oscillator_scale
    smoothed = ema(raw_osc, params.oscillator_smooth_len)
    valid = smoothed[~np.isnan(smoothed)]
    if len(valid) > 0:
        print(f"\n--- Oscillator Debug ---")
        print(f"  Range: {valid.min():.1f} to {valid.max():.1f}")
        print(f"  Mean: {valid.mean():.1f}")
        below_thresh = (valid < params.entry_threshold).sum()
        above_thresh = (valid > params.entry_threshold).sum()
        print(f"  Below {params.entry_threshold}: {below_thresh} bars")
        print(f"  Above {params.entry_threshold}: {above_thresh} bars")
        # Count crossovers
        crosses = 0
        for j in range(1, len(smoothed)):
            if not np.isnan(smoothed[j]) and not np.isnan(smoothed[j-1]):
                if smoothed[j-1] <= params.entry_threshold and smoothed[j] > params.entry_threshold:
                    crosses += 1
        print(f"  Crossovers above {params.entry_threshold}: {crosses}")

    # Run backtest - first with consolidation filter
    print(f"\nRunning backtest WITH consolidation filter on {len(df)} bars...")
    engine = BacktestEngine(config=config, params=params)
    result = engine.run(df)
    print(f"  Trades: {result.total_trades}")

    # Run again WITHOUT consolidation filter to verify oscillator entry works
    import copy
    params2 = copy.deepcopy(params)
    params2.use_consolidation_filter = False
    print(f"\nRunning backtest WITHOUT consolidation filter...")
    engine2 = BacktestEngine(config=config, params=params2)
    result = engine2.run(df)

    print(f"\n--- Results ---")
    print(f"Total Trades: {result.total_trades}")
    print(f"Winning: {result.winning_trades}")
    print(f"Losing: {result.losing_trades}")
    print(f"Win Rate: {result.win_rate:.1f}%")
    print(f"Total P&L: ${result.total_pnl:,.2f}")
    print(f"Profit Factor: {result.profit_factor:.2f}")
    print(f"Max Drawdown: {result.max_drawdown_pct:.1f}%")

    if result.trades:
        print(f"\n--- First 5 Trades ---")
        for t in result.trades[:5]:
            signal = t.exit_signal.value if t.exit_signal else "Open"
            print(f"  #{t.trade_id}: Entry {t.entry_time} @ ${t.entry_price:.2f} -> "
                  f"Exit {t.exit_time} @ ${t.exit_price:.2f} ({signal}) P&L: ${t.pnl:.2f}")


if __name__ == '__main__':
    params, settings = test_parser()
    test_engine_with_csv()
