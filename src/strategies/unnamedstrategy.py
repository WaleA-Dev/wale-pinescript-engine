import numpy as np
import pandas as pd
from src.strategies.base import BaseStrategy

class UnnamedstrategyStrategy(BaseStrategy):
    """
    Auto-translated from PineScript: UnnamedStrategy

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

        # Indicator calculations
        emaFast = df['close'].ewm(span=self.params.get('emaFastLen', 20), adjust=False).mean()
        emaSlow = df['close'].ewm(span=self.params.get('emaSlowLen', 50), adjust=False).mean()
        rsiVal = self._calc_rsi(df['close'], self.params.get('rsiLen', 14))
        atrVal = self._calc_atr(df, self.params.get('atrLen', 14))

        # Condition calculations
        overlay = True,
        initial_capital = 10000,
        default_qty_type = False  # TODO: unsupported Pine runtime reference
        default_qty_value = 5,
        pyramiding = 1,
        commission_type = False  # TODO: unsupported Pine runtime reference
        commission_value = 0.05
        upTrend = emaFast > emaSlow
        downTrend = emaFast < emaSlow
        longCond = upTrend & rsiVal > self.params.get('rsiMin', 45) & rsiVal < self.params.get('rsiMax', 70) and
        shortCond = self.params.get('useShorts', False) & downTrend & rsiVal < (100 - self.params.get('rsiMin', 45)) and
        longStop = False  # TODO: unsupported Pine runtime reference
        longTake = False  # TODO: unsupported Pine runtime reference
        shortStop = False  # TODO: unsupported Pine runtime reference
        shortTake = False  # TODO: unsupported Pine runtime reference

        signal = pd.Series(0.0, index=df.index)
        signal.loc[longCond] = 1
        signal.loc[shortCond] = -1

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
    'emaFastLen': list(range(10, 35, 5)),
    'emaSlowLen': list(range(40, 65, 5)),
    'rsiLen': list(range(4, 29, 5)),
    'rsiMin': list(range(35, 60, 5)),
    'rsiMax': list(range(60, 85, 5)),
    'atrLen': list(range(4, 29, 5)),
    'slMult': [1.4400, 1.8000, 2.1600],
    'tpMult': [2.2400, 2.8000, 3.3600],
    'useShorts': [True, False],
}
