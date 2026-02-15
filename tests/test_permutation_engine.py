"""Tests for permutation algorithm."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.permutation import permute_bars


def _sample_ohlc(n=3000, seed=42):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0.0, 0.8, size=n))
    close = np.maximum(close, 1.0)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + np.abs(rng.normal(0.2, 0.08, size=n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0.2, 0.08, size=n))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})


def test_permute_preserves_bar_structure_and_anchors():
    df = _sample_ohlc(2000, seed=1)
    np.random.seed(7)
    p = permute_bars(df)
    assert float(p.iloc[0]["open"]) == float(df.iloc[0]["open"])
    assert float(p.iloc[-1]["close"]) == float(df.iloc[-1]["close"])
    assert np.all(p["high"].to_numpy() >= np.maximum(p["open"].to_numpy(), p["close"].to_numpy()))
    assert np.all(p["low"].to_numpy() <= np.minimum(p["open"].to_numpy(), p["close"].to_numpy()))


def test_permute_preserves_return_distribution_ks():
    df = _sample_ohlc(3500, seed=3)
    np.random.seed(9)
    p = permute_bars(df)
    r1 = np.diff(np.log(df["close"].to_numpy()))
    r2 = np.diff(np.log(p["close"].to_numpy()))
    assert ks_2samp(r1, r2).pvalue > 0.05
