"""Tests for strategy signal generation: shape, values in {-1,0,1}, no NaN."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategies.donchian import DonchianBreakoutStrategy
from src.strategies.ema_crossover import EMACrossoverStrategy
from src.strategies.ndx_trader import NdxTraderStrategy


def _sample_ohlc(n=500, seed=42):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0.02, 0.5, n))
    close = np.maximum(close, 1.0)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + np.abs(rng.normal(0.1, 0.05, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0.1, 0.05, n))
    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "volume": rng.integers(1000, 10000, n),
    })


@pytest.mark.parametrize("cls,kwargs", [
    (DonchianBreakoutStrategy, {}),
    (DonchianBreakoutStrategy, {"lookback": 10}),
    (EMACrossoverStrategy, {}),
    (EMACrossoverStrategy, {"fast_period": 8, "slow_period": 20}),
    (NdxTraderStrategy, {}),
])
def test_signal_shape_and_values(cls, kwargs):
    df = _sample_ohlc()
    strategy = cls(**kwargs)
    signals = strategy.generate_signals(df)

    assert len(signals) == len(df), "Signal length must match data length"
    assert not np.any(np.isnan(signals)), "Signals must not contain NaN"
    assert set(np.unique(signals)).issubset({-1, 0, 1}), "Signals must be in {-1, 0, 1}"


def test_donchian_param_grid():
    s = DonchianBreakoutStrategy()
    grid = s.param_grid()
    assert "lookback" in grid
    assert len(grid["lookback"]) >= 2


def test_ema_crossover_param_grid():
    s = EMACrossoverStrategy()
    grid = s.param_grid()
    assert "fast_period" in grid
    assert "slow_period" in grid


def test_ndx_trader_param_grid():
    s = NdxTraderStrategy()
    grid = s.param_grid()
    assert "fast_len" in grid
    assert "slow_len" in grid


def test_base_strategy_helpers():
    df = _sample_ohlc(200)
    close = df["close"]

    ema = DonchianBreakoutStrategy.calc_ema(close, 10)
    assert len(ema) == len(close)
    assert not ema.isna().all()

    sma = DonchianBreakoutStrategy.calc_sma(close, 10)
    assert len(sma) == len(close)

    rsi = DonchianBreakoutStrategy.calc_rsi(close, 14)
    assert len(rsi) == len(close)
    valid_rsi = rsi.dropna()
    assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all()

    atr = DonchianBreakoutStrategy.calc_atr(df, 14)
    assert len(atr) == len(df)
    assert (atr.dropna() >= 0).all()
