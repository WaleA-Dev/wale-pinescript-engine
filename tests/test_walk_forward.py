"""Tests for walk-forward engine: OOS segments non-overlapping with IS."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy
from src.validation.walk_forward import walk_forward_backtest


class SimpleStrategy(BaseStrategy):
    def __init__(self, **params):
        super().__init__(**params)

    def _set_defaults(self):
        self.params.setdefault("lookback", 3)

    def generate_signals(self, df):
        lr = np.log(df["close"]).diff()
        return np.sign(lr.rolling(int(self.params["lookback"])).mean()).fillna(0).astype(int).values

    def param_grid(self):
        return {"lookback": [2, 3, 4]}


def _ohlc(n=400, seed=11):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0.03, 0.5, size=n))
    close = np.maximum(close, 1.0)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + np.abs(rng.normal(0.06, 0.02, size=n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0.06, 0.02, size=n))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})


def test_walk_forward_oos_length():
    df = _ohlc(300)
    train_years = 0.5
    train_bars = int(train_years * 252)

    oos = walk_forward_backtest(
        df, SimpleStrategy, {"lookback": 3},
        train_years=train_years, retrain_days=15,
        param_grid={"lookback": [2, 3]}, bars_per_day=1,
    )
    assert len(oos) == len(df) - train_bars


def test_walk_forward_no_param_grid():
    df = _ohlc(300)
    train_years = 0.5
    train_bars = int(train_years * 252)

    oos = walk_forward_backtest(
        df, SimpleStrategy, {"lookback": 3},
        train_years=train_years, bars_per_day=1,
    )
    assert len(oos) == len(df) - train_bars
    assert not oos.isna().all()


def test_walk_forward_returns_are_numeric():
    df = _ohlc(300)
    oos = walk_forward_backtest(
        df, SimpleStrategy, {"lookback": 3},
        train_years=0.5, bars_per_day=1,
    )
    assert oos.dtype in [np.float64, np.float32]
