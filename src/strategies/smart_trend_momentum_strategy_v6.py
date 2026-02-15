import numpy as np
import pandas as pd
from src.strategies.base import BaseStrategy

class SmartTrendMomentumStrategyV6Strategy(BaseStrategy):
    """
    Auto-translated from PineScript: Smart Trend Momentum Strategy v6

    WARNING: Auto-generated code. Review logic before production use.
    """

    def __init__(self, **params):
        super().__init__(**params)
        self.params.setdefault('emaFastLen', 20)
        self.params.setdefault('emaSlowLen', 50)
        self.params.setdefault('rsiLen', 14)
        self.params.setdefault('rsiMin', 45)
        self.params.setdefault('rsiMax', 70)
        self.params.setdefault('atrLen', 14)
        self.params.setdefault('slMult', 1.8)
        self.params.setdefault('tpMult', 2.8)
        self.params.setdefault('useShorts', False)

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
            emaFast = df['close'].ewm(span=self.params.get('emaFastLen', 20), adjust=False).mean()
        except Exception:
            emaFast = pd.Series(np.nan, index=df.index, dtype='float64')
        try:
            emaSlow = df['close'].ewm(span=self.params.get('emaSlowLen', 50), adjust=False).mean()
        except Exception:
            emaSlow = pd.Series(np.nan, index=df.index, dtype='float64')
        try:
            rsiVal = self._calc_rsi(df['close'], self.params.get('rsiLen', 14))
        except Exception:
            rsiVal = pd.Series(np.nan, index=df.index, dtype='float64')
        try:
            atrVal = self._calc_atr(df, self.params.get('atrLen', 14))
        except Exception:
            atrVal = pd.Series(np.nan, index=df.index, dtype='float64')
        try:
            upTrend = emaFast > emaSlow
        except Exception:
            upTrend = pd.Series(np.nan, index=df.index, dtype='float64')
        try:
            downTrend = emaFast < emaSlow
        except Exception:
            downTrend = pd.Series(np.nan, index=df.index, dtype='float64')
        try:
            longCond = (upTrend) & (rsiVal > self.params.get('rsiMin', 45)) & (rsiVal < self.params.get('rsiMax', 70)) & (0 == 0)
        except Exception:
            longCond = pd.Series(np.nan, index=df.index, dtype='float64')
        try:
            shortCond = (self.params.get('useShorts', False)) & (downTrend) & (rsiVal < (100 - self.params.get('rsiMin', 45))) & (0 == 0)
        except Exception:
            shortCond = pd.Series(np.nan, index=df.index, dtype='float64')
        try:
            longStop = df['close'] - atrVal * self.params.get('slMult', 1.8)
        except Exception:
            longStop = pd.Series(np.nan, index=df.index, dtype='float64')
        try:
            longTake = df['close'] + atrVal * self.params.get('tpMult', 2.8)
        except Exception:
            longTake = pd.Series(np.nan, index=df.index, dtype='float64')
        try:
            shortStop = df['close'] + atrVal * self.params.get('slMult', 1.8)
        except Exception:
            shortStop = pd.Series(np.nan, index=df.index, dtype='float64')
        try:
            shortTake = df['close'] - atrVal * self.params.get('tpMult', 2.8)
        except Exception:
            shortTake = pd.Series(np.nan, index=df.index, dtype='float64')

        signal = pd.Series(0.0, index=df.index)
        try:
            signal.loc[_as_mask(longCond)] = 1
        except Exception:
            pass
        try:
            signal.loc[_as_mask(shortCond)] = -1
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
    'emaFastLen': [15, 20, 25],
    'emaSlowLen': [45, 50, 55],
    'rsiLen': [9, 14, 19],
    'rsiMin': [40, 45, 50],
    'rsiMax': [70],
    'atrLen': [14],
    'slMult': [1.8],
    'tpMult': [2.8],
    'useShorts': [False],
}
