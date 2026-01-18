"""
Indicator Library - Exact PineScript Mathematical Equivalents

This module implements TradingView's indicator calculations with exact mathematical
equivalence. Key differences from common Python implementations:

1. EMA uses SMA seed for initialization (not starting from bar 0)
2. RMA uses alpha = 1/length (not 2/(length+1) like EMA)
3. ATR uses RMA smoothing (not EMA)
4. ADX uses the full DMI calculation with proper smoothing

All functions handle NaN values correctly and match PineScript's bar-by-bar behavior.
"""

import numpy as np
from typing import Union

ArrayLike = Union[np.ndarray, list]


def _ensure_array(series: ArrayLike) -> np.ndarray:
    """Convert input to numpy array with float dtype."""
    if isinstance(series, np.ndarray):
        return series.astype(float)
    return np.array(series, dtype=float)


def sma(series: ArrayLike, length: int) -> np.ndarray:
    """
    Simple Moving Average - exact PineScript ta.sma() equivalent.
    
    Args:
        series: Price or indicator series
        length: Lookback period
        
    Returns:
        SMA values with NaN for insufficient data
    """
    series = _ensure_array(series)
    n = len(series)
    out = np.full(n, np.nan, dtype=float)
    
    if length <= 0 or length > n:
        return out
    
    # Use cumsum for efficient calculation
    cumsum = np.zeros(n + 1)
    valid_mask = ~np.isnan(series)
    
    for i in range(n):
        if valid_mask[i]:
            cumsum[i + 1] = cumsum[i] + series[i]
        else:
            cumsum[i + 1] = cumsum[i]
    
    # Calculate SMA where we have enough valid values
    for i in range(length - 1, n):
        window = series[i - length + 1 : i + 1]
        if not np.isnan(window).any():
            out[i] = window.mean()
    
    return out


def ema(series: ArrayLike, length: int) -> np.ndarray:
    """
    Exponential Moving Average - exact PineScript ta.ema() equivalent.
    
    CRITICAL: PineScript initializes EMA with an SMA seed of the first 'length' values.
    Many Python implementations skip this, causing divergence on early bars.
    
    Formula: EMA[i] = alpha * price[i] + (1 - alpha) * EMA[i-1]
    Where: alpha = 2 / (length + 1)
    
    Args:
        series: Price or indicator series
        length: EMA period
        
    Returns:
        EMA values with NaN until SMA seed is available
    """
    series = _ensure_array(series)
    n = len(series)
    out = np.full(n, np.nan, dtype=float)
    
    if length <= 0 or length > n:
        return out
    
    alpha = 2.0 / (length + 1)
    
    # Find first valid window for SMA seed
    start_idx = None
    for i in range(length - 1, n):
        window = series[i - length + 1 : i + 1]
        if not np.isnan(window).any():
            out[i] = window.mean()  # SMA seed
            start_idx = i + 1
            break
    
    if start_idx is None:
        return out
    
    # Recursive EMA calculation
    for i in range(start_idx, n):
        if np.isnan(series[i]) or np.isnan(out[i - 1]):
            out[i] = np.nan
        else:
            out[i] = alpha * series[i] + (1 - alpha) * out[i - 1]
    
    return out


def rma(series: ArrayLike, length: int) -> np.ndarray:
    """
    Running Moving Average (Wilder's Smoothing) - exact PineScript ta.rma() equivalent.
    
    CRITICAL: RMA uses alpha = 1/length, NOT 2/(length+1) like EMA.
    This is used internally by ATR and ADX.
    
    Formula: RMA[i] = alpha * price[i] + (1 - alpha) * RMA[i-1]
    Where: alpha = 1 / length
    
    Args:
        series: Price or indicator series
        length: RMA period
        
    Returns:
        RMA values with NaN until SMA seed is available
    """
    series = _ensure_array(series)
    n = len(series)
    out = np.full(n, np.nan, dtype=float)
    
    if length <= 0 or length > n:
        return out
    
    alpha = 1.0 / length  # Different from EMA!
    
    # Find first valid window for SMA seed
    start_idx = None
    for i in range(length - 1, n):
        window = series[i - length + 1 : i + 1]
        if not np.isnan(window).any():
            out[i] = window.mean()  # SMA seed
            start_idx = i + 1
            break
    
    if start_idx is None:
        return out
    
    # Recursive RMA calculation
    for i in range(start_idx, n):
        if np.isnan(series[i]) or np.isnan(out[i - 1]):
            out[i] = np.nan
        else:
            out[i] = alpha * series[i] + (1 - alpha) * out[i - 1]
    
    return out


def tr(high: ArrayLike, low: ArrayLike, close: ArrayLike) -> np.ndarray:
    """
    True Range - exact PineScript ta.tr() equivalent.
    
    Formula: TR = max(high - low, |high - prev_close|, |low - prev_close|)
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        
    Returns:
        True Range values (first bar uses high - low)
    """
    high = _ensure_array(high)
    low = _ensure_array(low)
    close = _ensure_array(close)
    n = len(high)
    
    out = np.full(n, np.nan, dtype=float)
    
    # First bar: just high - low
    if n > 0 and not (np.isnan(high[0]) or np.isnan(low[0])):
        out[0] = high[0] - low[0]
    
    # Subsequent bars: max of three ranges
    for i in range(1, n):
        if np.isnan(high[i]) or np.isnan(low[i]) or np.isnan(close[i - 1]):
            continue
        
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        out[i] = max(hl, hc, lc)
    
    return out


def atr(high: ArrayLike, low: ArrayLike, close: ArrayLike, length: int) -> np.ndarray:
    """
    Average True Range - exact PineScript ta.atr() equivalent.
    
    CRITICAL: TradingView's ATR uses RMA smoothing, NOT EMA.
    Using EMA will cause drift over time and incorrect stop loss levels.
    
    Formula: ATR = RMA(TR, length)
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        length: ATR period
        
    Returns:
        ATR values
    """
    true_range = tr(high, low, close)
    return rma(true_range, length)


def adx(high: ArrayLike, low: ArrayLike, close: ArrayLike, length: int) -> np.ndarray:
    """
    Average Directional Index - exact PineScript ta.adx() equivalent.
    
    Full DMI calculation with proper smoothing:
    1. Calculate +DM and -DM
    2. Smooth with RMA
    3. Calculate +DI and -DI
    4. Calculate DX
    5. Smooth DX with RMA to get ADX
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        length: ADX period
        
    Returns:
        ADX values
    """
    high = _ensure_array(high)
    low = _ensure_array(low)
    close = _ensure_array(close)
    n = len(high)
    
    # Calculate directional movement
    plus_dm = np.full(n, np.nan, dtype=float)
    minus_dm = np.full(n, np.nan, dtype=float)
    
    for i in range(1, n):
        if np.isnan(high[i]) or np.isnan(high[i-1]) or np.isnan(low[i]) or np.isnan(low[i-1]):
            continue
        
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        
        plus_dm[i] = up_move if (up_move > down_move and up_move > 0) else 0.0
        minus_dm[i] = down_move if (down_move > up_move and down_move > 0) else 0.0
    
    # Smooth the components
    true_range = tr(high, low, close)
    smoothed_tr = rma(true_range, length)
    smoothed_plus_dm = rma(plus_dm, length)
    smoothed_minus_dm = rma(minus_dm, length)
    
    # Calculate +DI and -DI
    plus_di = np.full(n, np.nan, dtype=float)
    minus_di = np.full(n, np.nan, dtype=float)
    
    for i in range(n):
        if np.isnan(smoothed_tr[i]) or smoothed_tr[i] == 0:
            continue
        if not np.isnan(smoothed_plus_dm[i]):
            plus_di[i] = 100.0 * smoothed_plus_dm[i] / smoothed_tr[i]
        if not np.isnan(smoothed_minus_dm[i]):
            minus_di[i] = 100.0 * smoothed_minus_dm[i] / smoothed_tr[i]
    
    # Calculate DX
    dx = np.full(n, np.nan, dtype=float)
    for i in range(n):
        if np.isnan(plus_di[i]) or np.isnan(minus_di[i]):
            continue
        di_sum = plus_di[i] + minus_di[i]
        if di_sum == 0:
            dx[i] = 0.0
        else:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    # Smooth DX to get ADX
    return rma(dx, length)


def adx_components(high: ArrayLike, low: ArrayLike, close: ArrayLike, length: int) -> tuple:
    """
    ADX with +DI and -DI components - for strategies that need directional indicators.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        length: ADX period
        
    Returns:
        Tuple of (ADX, +DI, -DI) arrays
    """
    high = _ensure_array(high)
    low = _ensure_array(low)
    close = _ensure_array(close)
    n = len(high)
    
    # Calculate directional movement
    plus_dm = np.full(n, np.nan, dtype=float)
    minus_dm = np.full(n, np.nan, dtype=float)
    
    for i in range(1, n):
        if np.isnan(high[i]) or np.isnan(high[i-1]) or np.isnan(low[i]) or np.isnan(low[i-1]):
            continue
        
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        
        plus_dm[i] = up_move if (up_move > down_move and up_move > 0) else 0.0
        minus_dm[i] = down_move if (down_move > up_move and down_move > 0) else 0.0
    
    # Smooth the components
    true_range = tr(high, low, close)
    smoothed_tr = rma(true_range, length)
    smoothed_plus_dm = rma(plus_dm, length)
    smoothed_minus_dm = rma(minus_dm, length)
    
    # Calculate +DI and -DI
    plus_di = np.full(n, np.nan, dtype=float)
    minus_di = np.full(n, np.nan, dtype=float)
    
    for i in range(n):
        if np.isnan(smoothed_tr[i]) or smoothed_tr[i] == 0:
            continue
        if not np.isnan(smoothed_plus_dm[i]):
            plus_di[i] = 100.0 * smoothed_plus_dm[i] / smoothed_tr[i]
        if not np.isnan(smoothed_minus_dm[i]):
            minus_di[i] = 100.0 * smoothed_minus_dm[i] / smoothed_tr[i]
    
    # Calculate DX
    dx = np.full(n, np.nan, dtype=float)
    for i in range(n):
        if np.isnan(plus_di[i]) or np.isnan(minus_di[i]):
            continue
        di_sum = plus_di[i] + minus_di[i]
        if di_sum == 0:
            dx[i] = 0.0
        else:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    # Smooth DX to get ADX
    adx_values = rma(dx, length)
    
    return adx_values, plus_di, minus_di


def rsi(series: ArrayLike, length: int) -> np.ndarray:
    """
    Relative Strength Index - exact PineScript ta.rsi() equivalent.
    
    Formula: RSI = 100 - (100 / (1 + RS))
    Where: RS = RMA(gains, length) / RMA(losses, length)
    
    Args:
        series: Price series (typically close)
        length: RSI period
        
    Returns:
        RSI values (0-100)
    """
    series = _ensure_array(series)
    n = len(series)
    
    # Calculate price changes
    changes = np.full(n, np.nan, dtype=float)
    for i in range(1, n):
        if not np.isnan(series[i]) and not np.isnan(series[i-1]):
            changes[i] = series[i] - series[i-1]
    
    # Separate gains and losses
    gains = np.where(changes > 0, changes, 0.0)
    losses = np.where(changes < 0, -changes, 0.0)
    
    # Handle NaN propagation
    gains = np.where(np.isnan(changes), np.nan, gains)
    losses = np.where(np.isnan(changes), np.nan, losses)
    
    # Smooth with RMA
    avg_gain = rma(gains, length)
    avg_loss = rma(losses, length)
    
    # Calculate RSI
    out = np.full(n, np.nan, dtype=float)
    for i in range(n):
        if np.isnan(avg_gain[i]) or np.isnan(avg_loss[i]):
            continue
        if avg_loss[i] == 0:
            out[i] = 100.0 if avg_gain[i] > 0 else 50.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            out[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return out


def macd(series: ArrayLike, fast_length: int = 12, slow_length: int = 26, 
         signal_length: int = 9) -> tuple:
    """
    MACD - exact PineScript ta.macd() equivalent.
    
    Args:
        series: Price series (typically close)
        fast_length: Fast EMA period
        slow_length: Slow EMA period
        signal_length: Signal line EMA period
        
    Returns:
        Tuple of (MACD line, Signal line, Histogram)
    """
    series = _ensure_array(series)
    
    fast_ema = ema(series, fast_length)
    slow_ema = ema(series, slow_length)
    
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal_length)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def bollinger_bands(series: ArrayLike, length: int = 20, mult: float = 2.0) -> tuple:
    """
    Bollinger Bands - exact PineScript ta.bb() equivalent.
    
    Args:
        series: Price series (typically close)
        length: SMA period
        mult: Standard deviation multiplier
        
    Returns:
        Tuple of (Upper band, Middle band, Lower band)
    """
    series = _ensure_array(series)
    n = len(series)
    
    middle = sma(series, length)
    
    # Calculate standard deviation
    std = np.full(n, np.nan, dtype=float)
    for i in range(length - 1, n):
        window = series[i - length + 1 : i + 1]
        if not np.isnan(window).any():
            std[i] = np.std(window, ddof=0)  # Population std like PineScript
    
    upper = middle + mult * std
    lower = middle - mult * std
    
    return upper, middle, lower


def stochastic(high: ArrayLike, low: ArrayLike, close: ArrayLike, 
               k_length: int = 14, k_smooth: int = 1, d_smooth: int = 3) -> tuple:
    """
    Stochastic Oscillator - exact PineScript ta.stoch() equivalent.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        k_length: %K lookback period
        k_smooth: %K smoothing period
        d_smooth: %D smoothing period
        
    Returns:
        Tuple of (%K, %D)
    """
    high = _ensure_array(high)
    low = _ensure_array(low)
    close = _ensure_array(close)
    n = len(high)
    
    # Calculate raw %K
    raw_k = np.full(n, np.nan, dtype=float)
    for i in range(k_length - 1, n):
        h_window = high[i - k_length + 1 : i + 1]
        l_window = low[i - k_length + 1 : i + 1]
        
        if np.isnan(h_window).any() or np.isnan(l_window).any() or np.isnan(close[i]):
            continue
        
        highest = np.max(h_window)
        lowest = np.min(l_window)
        
        if highest == lowest:
            raw_k[i] = 50.0
        else:
            raw_k[i] = 100.0 * (close[i] - lowest) / (highest - lowest)
    
    # Smooth %K
    k = sma(raw_k, k_smooth) if k_smooth > 1 else raw_k
    
    # Calculate %D
    d = sma(k, d_smooth)
    
    return k, d


def highest(series: ArrayLike, length: int) -> np.ndarray:
    """
    Highest value over period - exact PineScript ta.highest() equivalent.
    
    Args:
        series: Price or indicator series
        length: Lookback period
        
    Returns:
        Rolling highest values
    """
    series = _ensure_array(series)
    n = len(series)
    out = np.full(n, np.nan, dtype=float)
    
    for i in range(length - 1, n):
        window = series[i - length + 1 : i + 1]
        if not np.isnan(window).any():
            out[i] = np.max(window)
    
    return out


def lowest(series: ArrayLike, length: int) -> np.ndarray:
    """
    Lowest value over period - exact PineScript ta.lowest() equivalent.
    
    Args:
        series: Price or indicator series
        length: Lookback period
        
    Returns:
        Rolling lowest values
    """
    series = _ensure_array(series)
    n = len(series)
    out = np.full(n, np.nan, dtype=float)
    
    for i in range(length - 1, n):
        window = series[i - length + 1 : i + 1]
        if not np.isnan(window).any():
            out[i] = np.min(window)
    
    return out


def crossover(series1: ArrayLike, series2: ArrayLike) -> np.ndarray:
    """
    Crossover detection - exact PineScript ta.crossover() equivalent.
    
    Returns True when series1 crosses above series2.
    
    Args:
        series1: First series
        series2: Second series (or constant)
        
    Returns:
        Boolean array where True indicates crossover
    """
    series1 = _ensure_array(series1)
    series2 = _ensure_array(series2) if hasattr(series2, '__len__') else np.full(len(series1), series2)
    n = len(series1)
    
    out = np.full(n, False, dtype=bool)
    
    for i in range(1, n):
        if (np.isnan(series1[i]) or np.isnan(series1[i-1]) or 
            np.isnan(series2[i]) or np.isnan(series2[i-1])):
            continue
        
        # Current bar: series1 > series2
        # Previous bar: series1 <= series2
        if series1[i] > series2[i] and series1[i-1] <= series2[i-1]:
            out[i] = True
    
    return out


def crossunder(series1: ArrayLike, series2: ArrayLike) -> np.ndarray:
    """
    Crossunder detection - exact PineScript ta.crossunder() equivalent.
    
    Returns True when series1 crosses below series2.
    
    Args:
        series1: First series
        series2: Second series (or constant)
        
    Returns:
        Boolean array where True indicates crossunder
    """
    series1 = _ensure_array(series1)
    series2 = _ensure_array(series2) if hasattr(series2, '__len__') else np.full(len(series1), series2)
    n = len(series1)
    
    out = np.full(n, False, dtype=bool)
    
    for i in range(1, n):
        if (np.isnan(series1[i]) or np.isnan(series1[i-1]) or 
            np.isnan(series2[i]) or np.isnan(series2[i-1])):
            continue
        
        # Current bar: series1 < series2
        # Previous bar: series1 >= series2
        if series1[i] < series2[i] and series1[i-1] >= series2[i-1]:
            out[i] = True
    
    return out
