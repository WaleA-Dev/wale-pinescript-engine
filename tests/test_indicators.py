"""
Tests for indicator implementations.

These tests verify that our indicator calculations match PineScript's behavior.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.indicators import (
    ema, rma, sma, atr, tr, adx, rsi, macd,
    bollinger_bands, stochastic, highest, lowest,
    crossover, crossunder
)


class TestSMA:
    """Tests for Simple Moving Average."""
    
    def test_basic_sma(self):
        """SMA should return correct average."""
        series = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sma(series, 3)
        
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert result[2] == pytest.approx(2.0)  # (1+2+3)/3
        assert result[3] == pytest.approx(3.0)  # (2+3+4)/3
        assert result[4] == pytest.approx(4.0)  # (3+4+5)/3
    
    def test_sma_with_nan(self):
        """SMA should handle NaN values."""
        series = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        result = sma(series, 3)
        
        # Windows containing NaN should produce NaN
        assert np.isnan(result[2])
        assert np.isnan(result[3])
        assert result[4] == pytest.approx(4.0)  # (3+4+5)/3


class TestEMA:
    """Tests for Exponential Moving Average."""
    
    def test_ema_initialization(self):
        """EMA should initialize with SMA seed."""
        series = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        result = ema(series, 3)
        
        # First valid EMA should be SMA of first 3 values
        assert result[2] == pytest.approx(2.0)  # (1+2+3)/3
        
        # Subsequent values use EMA formula
        alpha = 2.0 / (3 + 1)
        expected_3 = alpha * 4.0 + (1 - alpha) * 2.0
        assert result[3] == pytest.approx(expected_3)
    
    def test_ema_alpha(self):
        """EMA should use correct alpha = 2/(length+1)."""
        series = np.array([10.0] * 10 + [20.0] * 10)
        result = ema(series, 5)
        
        # After many bars at 20, EMA should approach 20
        assert result[-1] > 19.0


class TestRMA:
    """Tests for Running Moving Average (Wilder's Smoothing)."""
    
    def test_rma_alpha(self):
        """RMA should use alpha = 1/length (different from EMA)."""
        series = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        
        ema_result = ema(series, 3)
        rma_result = rma(series, 3)
        
        # RMA and EMA should differ after initialization
        # because alpha = 1/3 for RMA vs 2/4 for EMA
        assert rma_result[3] != ema_result[3]
    
    def test_rma_smoothing(self):
        """RMA should be smoother than EMA (slower response)."""
        series = np.array([10.0] * 5 + [20.0] * 20)
        
        ema_result = ema(series, 5)
        rma_result = rma(series, 5)
        
        # After step change, EMA should respond faster
        # So at same point, EMA should be closer to 20
        assert ema_result[10] > rma_result[10]


class TestTrueRange:
    """Tests for True Range calculation."""
    
    def test_tr_basic(self):
        """TR should be max of three ranges."""
        high = np.array([10.0, 12.0, 11.0])
        low = np.array([8.0, 9.0, 8.0])
        close = np.array([9.0, 11.0, 9.0])
        
        result = tr(high, low, close)
        
        # First bar: high - low = 2
        assert result[0] == pytest.approx(2.0)
        
        # Second bar: max(12-9, |12-9|, |9-9|) = 3
        assert result[1] == pytest.approx(3.0)
    
    def test_tr_gap_up(self):
        """TR should handle gap up correctly."""
        high = np.array([10.0, 15.0])
        low = np.array([8.0, 13.0])
        close = np.array([9.0, 14.0])
        
        result = tr(high, low, close)
        
        # Gap up: high - prev_close = 15 - 9 = 6
        # This should be larger than high - low = 2
        assert result[1] == pytest.approx(6.0)


class TestATR:
    """Tests for Average True Range."""
    
    def test_atr_uses_rma(self):
        """ATR should use RMA smoothing, not EMA."""
        high = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0])
        low = np.array([9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0])
        close = np.array([9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5])
        
        result = atr(high, low, close, 3)
        
        # ATR should be smoothed TR
        true_range = tr(high, low, close)
        expected = rma(true_range, 3)
        
        np.testing.assert_array_almost_equal(result, expected)


class TestADX:
    """Tests for Average Directional Index."""
    
    def test_adx_range(self):
        """ADX should be between 0 and 100."""
        np.random.seed(42)
        n = 100
        high = 100 + np.cumsum(np.random.randn(n) * 0.5)
        low = high - np.abs(np.random.randn(n))
        close = (high + low) / 2
        
        result = adx(high, low, close, 14)
        
        valid = result[~np.isnan(result)]
        assert all(0 <= v <= 100 for v in valid)
    
    def test_adx_trending(self):
        """ADX should be high in trending market."""
        # Strong uptrend
        n = 50
        high = np.arange(100, 100 + n, dtype=float)
        low = high - 0.5
        close = high - 0.25
        
        result = adx(high, low, close, 14)
        
        # ADX should be relatively high in strong trend
        valid = result[~np.isnan(result)]
        if len(valid) > 0:
            assert valid[-1] > 20  # Trending threshold


class TestRSI:
    """Tests for Relative Strength Index."""
    
    def test_rsi_range(self):
        """RSI should be between 0 and 100."""
        np.random.seed(42)
        series = 100 + np.cumsum(np.random.randn(100))
        
        result = rsi(series, 14)
        
        valid = result[~np.isnan(result)]
        assert all(0 <= v <= 100 for v in valid)
    
    def test_rsi_overbought(self):
        """RSI should be high after sustained gains."""
        # Consistent upward movement
        series = np.arange(100, 120, 0.5)
        
        result = rsi(series, 14)
        
        # RSI should be very high
        valid = result[~np.isnan(result)]
        if len(valid) > 0:
            assert valid[-1] > 70
    
    def test_rsi_oversold(self):
        """RSI should be low after sustained losses."""
        # Consistent downward movement
        series = np.arange(120, 100, -0.5)
        
        result = rsi(series, 14)
        
        # RSI should be very low
        valid = result[~np.isnan(result)]
        if len(valid) > 0:
            assert valid[-1] < 30


class TestCrossover:
    """Tests for crossover/crossunder detection."""
    
    def test_crossover_detection(self):
        """Crossover should detect when series1 crosses above series2."""
        series1 = np.array([1.0, 2.0, 3.0, 4.0, 3.0, 2.0])
        series2 = np.array([2.5, 2.5, 2.5, 2.5, 2.5, 2.5])
        
        result = crossover(series1, series2)
        
        # Crossover at index 2 (series1 goes from 2 to 3, crossing 2.5)
        assert not result[0]
        assert not result[1]
        assert result[2]
        assert not result[3]
    
    def test_crossunder_detection(self):
        """Crossunder should detect when series1 crosses below series2."""
        series1 = np.array([4.0, 3.0, 2.0, 1.0, 2.0, 3.0])
        series2 = np.array([2.5, 2.5, 2.5, 2.5, 2.5, 2.5])
        
        result = crossunder(series1, series2)
        
        # Crossunder at index 2 (series1 goes from 3 to 2, crossing 2.5)
        assert not result[0]
        assert not result[1]
        assert result[2]
        assert not result[3]


class TestHighestLowest:
    """Tests for highest/lowest functions."""
    
    def test_highest(self):
        """Highest should return rolling maximum."""
        series = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        result = highest(series, 3)
        
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert result[2] == pytest.approx(3.0)  # max(1,3,2)
        assert result[3] == pytest.approx(5.0)  # max(3,2,5)
        assert result[4] == pytest.approx(5.0)  # max(2,5,4)
    
    def test_lowest(self):
        """Lowest should return rolling minimum."""
        series = np.array([3.0, 1.0, 4.0, 2.0, 5.0])
        result = lowest(series, 3)
        
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert result[2] == pytest.approx(1.0)  # min(3,1,4)
        assert result[3] == pytest.approx(1.0)  # min(1,4,2)
        assert result[4] == pytest.approx(2.0)  # min(4,2,5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
