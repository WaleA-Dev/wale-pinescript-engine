"""Walk-forward and walk-forward permutation validation (steps 3 and 4)."""

from __future__ import annotations

from typing import Dict, Iterable, List, Type

import numpy as np
import pandas as pd

from ..bar_returns import compute_bar_returns, compute_metrics
from ..optimization import grid_search
from ..permutation import permute_bars

OHLC = ["open", "high", "low", "close"]


def _train_bars(train_years: float, trading_days_per_year: int, bars_per_day: int) -> int:
    return int(train_years * trading_days_per_year * bars_per_day)


def walk_forward_backtest(
    df: pd.DataFrame,
    strategy_class: Type,
    best_params: Dict,
    train_years: float = 4,
    retrain_days: int = 30,
    param_grid: Dict[str, Iterable] | List[Dict] | None = None,
    trading_days_per_year: int = 252,
    bars_per_day: int = 1,
) -> pd.Series:
    """Step 3: realistic walk-forward out-of-sample returns."""
    train_bars = _train_bars(train_years, trading_days_per_year, bars_per_day)
    if train_bars < 1 or train_bars >= len(df):
        raise ValueError("Insufficient data for requested walk-forward training window")

    retrain_bars = max(1, int(retrain_days * bars_per_day))

    if param_grid is None:
        strategy = strategy_class(**best_params)
        signals = strategy.generate_signals(df)
        run_df = compute_bar_returns(df, signals=signals)
        return run_df["strategy_return"].iloc[train_bars:]

    signals = pd.Series(0.0, index=df.index, dtype="float64")
    current_params = dict(best_params)
    i = train_bars
    while i < len(df):
        train_slice = df.iloc[i - train_bars : i]
        current_params, _ = grid_search(train_slice, strategy_class, param_grid)

        j = min(i + retrain_bars, len(df))
        strategy = strategy_class(**current_params)
        partial_signals = strategy.generate_signals(df.iloc[:j])
        partial_series = pd.Series(partial_signals, index=df.index[:j], dtype="float64").fillna(0.0)
        signals.iloc[i:j] = partial_series.iloc[i:j]
        i = j

    run_df = compute_bar_returns(df, signals=signals.values)
    return run_df["strategy_return"].iloc[train_bars:]


def walk_forward_permutation_test(
    df: pd.DataFrame,
    strategy_class: Type,
    best_params: Dict,
    n_perms: int = 200,
    train_years: float = 4,
    retrain_days: int = 30,
    param_grid: Dict[str, Iterable] | List[Dict] | None = None,
    trading_days_per_year: int = 252,
    bars_per_day: int = 1,
) -> Dict:
    """Step 4: permutation significance on walk-forward OOS performance."""
    if n_perms <= 0:
        raise ValueError("n_perms must be > 0")

    train_bars = _train_bars(train_years, trading_days_per_year, bars_per_day)
    if train_bars >= len(df):
        raise ValueError("Insufficient data for requested walk-forward training window")

    real_oos_returns = walk_forward_backtest(
        df=df,
        strategy_class=strategy_class,
        best_params=best_params,
        train_years=train_years,
        retrain_days=retrain_days,
        param_grid=param_grid,
        trading_days_per_year=trading_days_per_year,
        bars_per_day=bars_per_day,
    )
    real_oos_pf = compute_metrics(real_oos_returns)["profit_factor"]

    perm_oos_pfs: List[float] = []
    for _ in range(int(n_perms)):
        df_perm = df.copy()
        tail = permute_bars(df.iloc[train_bars:].copy(), start_index=0)
        df_perm.loc[df_perm.index[train_bars:], OHLC] = tail[OHLC].to_numpy()

        perm_returns = walk_forward_backtest(
            df=df_perm,
            strategy_class=strategy_class,
            best_params=best_params,
            train_years=train_years,
            retrain_days=retrain_days,
            param_grid=param_grid,
            trading_days_per_year=trading_days_per_year,
            bars_per_day=bars_per_day,
        )
        perm_oos_pfs.append(float(compute_metrics(perm_returns)["profit_factor"]))

    # Correct permutation p-value per Phipson & Smyth (2010)
    count_ge = int(np.sum(np.asarray(perm_oos_pfs) >= real_oos_pf))
    p_value = float((1 + count_ge) / (1 + len(perm_oos_pfs)))
    threshold = 0.05  # Pre-specified alpha; no data-dependent adjustment
    return {
        "real_oos_pf": float(real_oos_pf),
        "perm_oos_pfs": perm_oos_pfs,
        "p_value": p_value,
        "passed": bool(p_value < threshold),
        "threshold": threshold,
    }
