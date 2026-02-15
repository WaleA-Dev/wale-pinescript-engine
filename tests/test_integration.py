"""End-to-end integration test: load data -> signals -> metrics -> permutation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.bar_returns import compute_bar_returns, compute_metrics
from src.data_loader import load_csv
from src.optimization import grid_search
from src.strategies.donchian import DonchianBreakoutStrategy


def _create_test_csv(path, n=300, seed=42):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0.02, 0.5, n))
    close = np.maximum(close, 1.0)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + np.abs(rng.normal(0.1, 0.05, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0.1, 0.05, n))
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "Date": dates.strftime("%Y-%m-%d"),
        "Open": open_, "High": high, "Low": low, "Close": close,
        "Volume": rng.integers(1000, 10000, n),
    })
    df.to_csv(path, index=False)


def test_end_to_end_load_signals_metrics(tmp_path):
    csv_file = tmp_path / "test_data.csv"
    _create_test_csv(csv_file)

    # Step 1: Load data
    df = load_csv(csv_file)
    assert len(df) == 300
    assert "open" in df.columns
    assert "close" in df.columns

    # Step 2: Generate signals
    strategy = DonchianBreakoutStrategy(lookback=10)
    signals = strategy.generate_signals(df)
    assert len(signals) == len(df)
    assert set(np.unique(signals)).issubset({-1, 0, 1})

    # Step 3: Compute returns and metrics
    result = compute_bar_returns(df, signals=signals)
    assert "strategy_return" in result.columns
    assert "equity_curve" in result.columns

    metrics = compute_metrics(result["strategy_return"])
    assert "profit_factor" in metrics
    assert "sharpe_ratio" in metrics
    assert "max_drawdown" in metrics
    assert np.isfinite(metrics["sharpe_ratio"])


def test_end_to_end_grid_search(tmp_path):
    csv_file = tmp_path / "test_data.csv"
    _create_test_csv(csv_file, n=200)

    df = load_csv(csv_file)
    best_params, best_score = grid_search(
        df, DonchianBreakoutStrategy, {"lookback": [10, 20, 30]}
    )
    assert "lookback" in best_params
    assert best_params["lookback"] in {10, 20, 30}
    assert np.isfinite(best_score) or np.isinf(best_score)
