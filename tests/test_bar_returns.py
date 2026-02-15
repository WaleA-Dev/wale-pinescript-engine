"""Tests for bar-return engine primitives."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.bar_returns import compute_bar_returns, compute_metrics
from src.strategies.donchian import DonchianBreakoutStrategy


def test_compute_bar_returns_basic():
    df = pd.DataFrame({"close": [100.0, 110.0, 121.0]})
    signals = [1, 1, 1]
    out = compute_bar_returns(df, signals=signals)
    # pct_change: NaN, 0.1, 0.1; strategy_return = shift(1)*pct = NaN, 1*0.1, 1*0.1
    assert np.isnan(out.loc[0, "strategy_return"])
    assert np.isclose(out.loc[1, "strategy_return"], 0.1)


def test_compute_metrics_manual():
    r = pd.Series([0.1, -0.05, 0.02, -0.01], dtype=float)
    m = compute_metrics(r)
    assert np.isclose(m["profit_factor"], 2.0)
    assert np.isfinite(m["sharpe_ratio"])
    assert 0 <= m["max_drawdown"] <= 1
    assert m["total_return"] > 0
    assert "win_rate" in m
    assert "num_trades" in m
    assert "calmar_ratio" in m


def test_donchian_signals_valid():
    df = pd.DataFrame({"close": [10, 11, 12, 11, 10, 9, 10, 11, 12]})
    sig = DonchianBreakoutStrategy(lookback=3).generate_signals(df)
    assert len(sig) == len(df)
    assert set(np.unique(sig)).issubset({-1, 0, 1})
