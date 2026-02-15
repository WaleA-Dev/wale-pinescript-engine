"""EMA crossover strategy (vectorized)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseStrategy


class EMACrossoverStrategy(BaseStrategy):
    """
    Long-only EMA crossover:
    - Enter long when fast EMA crosses above slow EMA
    - Exit to flat when fast EMA crosses below slow EMA
    """

    def __init__(self, **params):
        super().__init__(**params)

    def _set_defaults(self):
        self.params.setdefault("fast_period", 12)
        self.params.setdefault("slow_period", 26)
        self.params.setdefault("source_col", "close")

    def param_grid(self) -> dict:
        return {
            "fast_period": [8, 12, 16],
            "slow_period": [20, 26, 40],
        }

    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        fast_period = int(self.params["fast_period"])
        slow_period = int(self.params["slow_period"])
        source_col = self.params["source_col"]

        if fast_period >= slow_period:
            raise ValueError("fast_period must be < slow_period")
        if source_col not in df.columns:
            raise ValueError(f"Missing source column '{source_col}'")

        close = pd.to_numeric(df[source_col], errors="coerce")
        fast = close.ewm(span=fast_period, adjust=False, min_periods=fast_period).mean()
        slow = close.ewm(span=slow_period, adjust=False, min_periods=slow_period).mean()

        cross_up = (fast > slow) & (fast.shift(1) <= slow.shift(1))
        cross_down = (fast < slow) & (fast.shift(1) >= slow.shift(1))

        raw = np.where(cross_up, 1.0, np.where(cross_down, 0.0, np.nan))
        signal = pd.Series(raw, index=df.index, dtype="float64").ffill().fillna(0.0)
        signal = signal.where(~(fast.isna() | slow.isna()), 0.0)
        return signal.astype(int).values


PARAM_GRID_TIGHT = {
    "fast_period": [8, 12, 16],
    "slow_period": [20, 26, 40],
}
PARAM_GRID_WIDE = {
    "fast_period": list(range(5, 31, 2)),
    "slow_period": list(range(20, 101, 5)),
}
PARAM_GRID_DEFAULT = PARAM_GRID_TIGHT
