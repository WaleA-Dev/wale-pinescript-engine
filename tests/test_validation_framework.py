"""Tests for optimization and validation modules."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.optimization import grid_search
from src.strategies.base import BaseStrategy
from src.validation.in_sample_permutation import in_sample_permutation_test
from src.validation.walk_forward import walk_forward_backtest, walk_forward_permutation_test


def _ohlc(n=400, seed=11):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0.03, 0.5, size=n))
    close = np.maximum(close, 1.0)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + np.abs(rng.normal(0.06, 0.02, size=n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0.06, 0.02, size=n))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})


class MomentumSignStrategy(BaseStrategy):
    def __init__(self, lookback=3):
        super().__init__(lookback=lookback)
        self.lookback = int(lookback)

    def generate_signals(self, df):
        lr = np.log(df["close"]).diff()
        return np.sign(lr.rolling(self.lookback).mean()).fillna(0).astype(int)


def test_grid_search_returns_params():
    df = _ohlc(350, seed=22)
    params, pf = grid_search(df, MomentumSignStrategy, {"lookback": [2, 3, 4]})
    assert params["lookback"] in {2, 3, 4}
    assert np.isfinite(pf) or np.isinf(pf)


def test_in_sample_pvalue_math(monkeypatch):
    vals = iter([2.0, 1.0, 2.5, 2.0])  # real then perms

    def fake_grid_search(df, strategy_class, param_grid):
        return {"lookback": 3}, next(vals)

    monkeypatch.setattr("src.validation.in_sample_permutation.grid_search", fake_grid_search)
    monkeypatch.setattr("src.validation.in_sample_permutation.permute_bars", lambda df: df)

    res = in_sample_permutation_test(_ohlc(40), MomentumSignStrategy, {"lookback": [3]}, n_perms=3)
    # Corrected Phipson & Smyth: (1 + count_ge) / (1 + n_perms) = (1+2)/(1+3) = 0.75
    assert np.isclose(res["p_value"], 0.75)


def test_walk_forward_shape_and_pvalue(monkeypatch):
    df = _ohlc(300, seed=33)
    oos = walk_forward_backtest(
        df,
        MomentumSignStrategy,
        {"lookback": 3},
        train_years=0.5,
        retrain_days=15,
        param_grid={"lookback": [2, 3]},
        bars_per_day=1,
    )
    assert len(oos) == len(df) - int(0.5 * 252)

    vals = iter([3.0, 2.0, 4.0, 3.0])  # real then perms

    def fake_wf(**kwargs):
        return pd.Series([next(vals)], dtype=float)

    monkeypatch.setattr("src.validation.walk_forward.walk_forward_backtest", fake_wf)
    monkeypatch.setattr("src.validation.walk_forward.compute_metrics", lambda s: {"profit_factor": float(s.iloc[0])})
    monkeypatch.setattr("src.validation.walk_forward.permute_bars", lambda d, start_index=0: d)

    res = walk_forward_permutation_test(
        _ohlc(80),
        MomentumSignStrategy,
        {"lookback": 3},
        n_perms=3,
        train_years=0.1,
        bars_per_day=1,
    )
    # Corrected Phipson & Smyth: (1 + count_ge) / (1 + n_perms) = (1+2)/(1+3) = 0.75
    assert np.isclose(res["p_value"], 0.75)
