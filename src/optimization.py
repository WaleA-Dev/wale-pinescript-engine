"""Grid search optimization for bar-return strategies."""

from __future__ import annotations

from itertools import product
from typing import Dict, Iterable, List, Tuple, Type

import numpy as np
import pandas as pd

from .bar_returns import compute_bar_returns, compute_metrics


def expand_param_grid(param_grid: Dict[str, Iterable] | List[Dict]) -> List[Dict]:
    """Expand dict/list parameter grid into list of param dictionaries."""
    if isinstance(param_grid, dict):
        keys = list(param_grid.keys())
        vals = [list(v) for v in param_grid.values()]
        if any(len(v) == 0 for v in vals):
            raise ValueError("param_grid values cannot be empty")
        return [dict(zip(keys, combo)) for combo in product(*vals)]

    if isinstance(param_grid, list):
        if not param_grid:
            raise ValueError("param_grid list cannot be empty")
        return [dict(p) for p in param_grid]

    raise TypeError("param_grid must be dict or list[dict]")


def grid_search(
    df: pd.DataFrame,
    strategy_class: Type,
    param_grid: Dict[str, Iterable] | List[Dict],
    metric: str = "profit_factor",
    top_n: int = 5,
) -> Tuple[Dict, float]:
    """Optimize parameters by maximizing the given metric."""
    param_sets = expand_param_grid(param_grid)
    total = len(param_sets)

    best_params = None
    best_score = -np.inf

    for i, params in enumerate(param_sets):
        if total > 50 and (i + 1) % max(1, total // 10) == 0:
            print(f"  Grid search: {i + 1}/{total} ({100 * (i + 1) // total}%)")

        strategy = strategy_class(**params)
        signals = strategy.generate_signals(df)
        run_df = compute_bar_returns(df, signals=signals)
        score = compute_metrics(run_df["strategy_return"])[metric]

        if np.isnan(score):
            continue
        if score > best_score:
            best_score = float(score)
            best_params = dict(params)

    if best_params is None:
        raise ValueError("No valid parameters evaluated")

    if total > 100:
        print(f"  Warning: grid search evaluated {total} combos")

    return best_params, best_score
