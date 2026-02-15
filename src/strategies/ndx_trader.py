"""NDX Trader strategy — EMA trend + RSI pullback entries with ATR risk."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseStrategy


class NdxTraderStrategy(BaseStrategy):
    """
    Translated from: Starter NDX Trader v1 (Frequent Trades, ATR Risk)

    Logic:
    - Long: fast EMA > slow EMA AND RSI crosses above rsi_pull_long
    - Short: fast EMA < slow EMA AND RSI crosses below rsi_pull_short
    - Holds position until opposing signal
    """

    def __init__(self, **params):
        super().__init__(**params)

    def _set_defaults(self):
        self.params.setdefault("fast_len", 20)
        self.params.setdefault("slow_len", 50)
        self.params.setdefault("rsi_len", 14)
        self.params.setdefault("rsi_pull_long", 45)
        self.params.setdefault("rsi_pull_short", 55)

    def param_grid(self) -> dict:
        return {
            "fast_len": [15, 20, 25],
            "slow_len": [40, 50, 60],
            "rsi_len": [10, 14],
            "rsi_pull_long": [40, 45, 50],
        }

    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        fast_len = int(self.params["fast_len"])
        slow_len = int(self.params["slow_len"])
        rsi_len = int(self.params["rsi_len"])
        rsi_pull_long = float(self.params["rsi_pull_long"])
        rsi_pull_short = float(self.params["rsi_pull_short"])

        close = pd.to_numeric(df["close"], errors="coerce")
        ema_fast = self.calc_ema(close, fast_len)
        ema_slow = self.calc_ema(close, slow_len)
        rsi = self.calc_rsi(close, rsi_len)

        trend_up = ema_fast > ema_slow
        trend_down = ema_fast < ema_slow

        # RSI cross above pull level during uptrend = long entry
        rsi_cross_up = (rsi > rsi_pull_long) & (rsi.shift(1) <= rsi_pull_long)
        long_signal = trend_up & rsi_cross_up

        # RSI cross below pull level during downtrend = short entry
        rsi_cross_down = (rsi < rsi_pull_short) & (rsi.shift(1) >= rsi_pull_short)
        short_signal = trend_down & rsi_cross_down

        raw = np.where(long_signal, 1.0, np.where(short_signal, -1.0, np.nan))
        signal = pd.Series(raw, index=df.index, dtype="float64").ffill().fillna(0.0)

        # Warmup: zero out first slow_len bars
        signal.iloc[:slow_len] = 0

        return signal.astype(int).values


PARAM_GRID_DEFAULT = {
    "fast_len": [15, 20, 25],
    "slow_len": [40, 50, 60],
    "rsi_len": [10, 14],
    "rsi_pull_long": [40, 45, 50],
}
