"""Donchian breakout strategy (vectorized)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseStrategy


class DonchianBreakoutStrategy(BaseStrategy):
    """
    Donchian breakout:
    - Long when close >= prior highest(close, lookback)
    - Short when close <= prior lowest(close, lookback)
    - Hold otherwise
    """

    def __init__(self, **params):
        super().__init__(**params)

    def _set_defaults(self):
        self.params.setdefault("lookback", 20)
        self.params.setdefault("source_col", "close")

    def param_grid(self) -> dict:
        return {"lookback": [10, 20, 30, 40]}

    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        lookback = int(self.params["lookback"])
        source_col = self.params["source_col"]
        if lookback < 2:
            raise ValueError("lookback must be >= 2")
        if source_col not in df.columns:
            raise ValueError(f"Missing source column '{source_col}'")

        src = pd.to_numeric(df[source_col], errors="coerce")
        highest = src.rolling(lookback, min_periods=lookback).max().shift(1)
        lowest = src.rolling(lookback, min_periods=lookback).min().shift(1)

        long_entry = src >= highest
        short_entry = src <= lowest
        raw = np.where(long_entry, 1.0, np.where(short_entry, -1.0, np.nan))

        signals = pd.Series(raw, index=df.index, dtype="float64").ffill().fillna(0.0)
        signals = signals.where(~(highest.isna() | lowest.isna()), 0.0)
        return signals.astype(int).values


PARAM_GRID_TIGHT = {"lookback": [10, 20, 30, 40]}
PARAM_GRID_WIDE = {"lookback": list(range(5, 101, 5))}
PARAM_GRID_DEFAULT = PARAM_GRID_TIGHT
