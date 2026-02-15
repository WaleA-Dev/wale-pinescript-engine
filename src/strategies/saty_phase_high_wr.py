import numpy as np
import pandas as pd
from src.strategies.base import BaseStrategy

class SatyPhaseHighWrStrategy(BaseStrategy):
    """
    Auto-translated from PineScript: Saty Phase - High WR

    WARNING: Auto-generated code. Review logic before production use.
    """

    def __init__(self, **params):
        super().__init__(**params)
        self.params.setdefault('stop_loss_pct', 14.0)
        self.params.setdefault('trailing_pct', 0.4)
        self.params.setdefault('profit_target_pct', 4.5)
        self.params.setdefault('use_ob_exit', False)
        self.params.setdefault('entry_threshold', -50.0)

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
            pivot = df['close'].ewm(span=21, adjust=False).mean()
        except Exception:
            pivot = pd.Series(np.nan, index=df.index, dtype='float64')
        try:
            atr14 = self._calc_atr(df, 14)
        except Exception:
            atr14 = pd.Series(np.nan, index=df.index, dtype='float64')
        try:
            raw_signal = ((df['close'] - pivot) / (3.0 * atr14)) * 100
        except Exception:
            raw_signal = pd.Series(np.nan, index=df.index, dtype='float64')
        try:
            oscillator = raw_signal.ewm(span=3, adjust=False).mean()
        except Exception:
            oscillator = pd.Series(np.nan, index=df.index, dtype='float64')
        try:
            leaving_entry = (oscillator.shift(1) <= self.params.get('entry_threshold', -50.0)) & (oscillator > self.params.get('entry_threshold', -50.0))
        except Exception:
            leaving_entry = pd.Series(np.nan, index=df.index, dtype='float64')
        try:
            leaving_extreme = (oscillator.shift(1) <= -100) & (oscillator > -100)
        except Exception:
            leaving_extreme = pd.Series(np.nan, index=df.index, dtype='float64')
        try:
            currentPnL = ((df['close'] - entryPrice) / entryPrice) * 100
        except Exception:
            currentPnL = pd.Series(np.nan, index=df.index, dtype='float64')
        try:
            stopLossPrice = entryPrice * (1 - self.params.get('stop_loss_pct', 14.0) / 100)
        except Exception:
            stopLossPrice = pd.Series(np.nan, index=df.index, dtype='float64')

        signal = pd.Series(0.0, index=df.index)
        try:
            signal.loc[_as_mask(((leaving_entry) | (leaving_extreme)) & (0 == 0))] = 1
        except Exception:
            pass
        try:
            signal.loc[_as_mask(df['low'] <= stopLossPrice)] = 0
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
    'stop_loss_pct': [12.6, 14.0, 15.4],
    'trailing_pct': [0.36, 0.4, 0.44],
    'profit_target_pct': [4.05, 4.5, 4.95],
    'use_ob_exit': [True, False],
    'entry_threshold': [-45.0, -50.0, -55.0],
}
