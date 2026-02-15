import numpy as np
import pandas as pd
from src.strategies.base import BaseStrategy

class EmaCrossSimpleStrategy(BaseStrategy):
    """
    Auto-translated from PineScript: EMA Cross Simple

    WARNING: Auto-generated code. Review logic before production use.
    """

    def __init__(self, **params):
        super().__init__(**params)
        self.params.setdefault('fastLength', 10)
        self.params.setdefault('slowLength', 30)

    def generate_signals(self, df):
        """Generate vectorized bar-position signal."""
        if df is None or len(df) == 0:
            return pd.Series(dtype='int64')

        def _as_mask(value):
            if isinstance(value, pd.Series):
                return value.fillna(False).astype(bool)
            if isinstance(value, np.ndarray):
                return pd.Series(value, index=df.index).fillna(False).astype(bool)
            return pd.Series(bool(value), index=df.index)

        # Computed assignments in source order
        try:
            fastEMA = df['close'].ewm(span=self.params.get('fastLength', 10), adjust=False).mean()
        except Exception:
            fastEMA = pd.Series(np.nan, index=df.index, dtype='float64')
        try:
            slowEMA = df['close'].ewm(span=self.params.get('slowLength', 30), adjust=False).mean()
        except Exception:
            slowEMA = pd.Series(np.nan, index=df.index, dtype='float64')
        try:
            longCondition = ((fastEMA) > (slowEMA)) & ((fastEMA).shift(1) <= (slowEMA).shift(1))
        except Exception:
            longCondition = pd.Series(np.nan, index=df.index, dtype='float64')
        try:
            shortCondition = ((fastEMA) < (slowEMA)) & ((fastEMA).shift(1) >= (slowEMA).shift(1))
        except Exception:
            shortCondition = pd.Series(np.nan, index=df.index, dtype='float64')

        signal = pd.Series(0.0, index=df.index)
        try:
            signal.loc[_as_mask(longCondition)] = 1
        except Exception:
            pass
        try:
            signal.loc[_as_mask(shortCondition)] = -1
        except Exception:
            pass

        signal = signal.replace(0, np.nan).ffill().fillna(0).astype(int)
        return signal

    def _calc_atr(self, df, period=14):
        high = df['high']
        low = df['low']
        close = df['close']
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.ewm(alpha=1/period, adjust=False).mean()

    def _calc_rsi(self, series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _calc_macd(self, series, fast=12, slow=26, signal=9):
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        hist = macd - signal_line
        return macd, signal_line, hist

# Auto-generated parameter grid
PARAM_GRID_DEFAULT = {
    'fastLength': [5, 10, 15],
    'slowLength': [25, 30, 35],
}
