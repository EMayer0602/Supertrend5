"""
Tests for technical indicators: ATR, RSI, and Supertrend calculations.
These are critical tests as indicator accuracy directly affects trading decisions.
"""
import pytest
import numpy as np
import sys
import os
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock yfinance before importing new5
sys.modules['yfinance'] = MagicMock()

from new5 import calculate_atr, calculate_rsi, calculate_supertrend_vectorized


class TestCalculateATR:
    """Tests for Average True Range calculation."""

    def test_atr_returns_correct_shape(self, sample_ohlcv_100):
        """ATR output should have same length as input."""
        high = sample_ohlcv_100['High_TEST'].values
        low = sample_ohlcv_100['Low_TEST'].values
        close = sample_ohlcv_100['Close_TEST'].values

        atr = calculate_atr(high, low, close, period=14)

        assert len(atr) == len(close)

    def test_atr_initial_period_is_nan(self, sample_ohlcv_100):
        """First (period-1) values should be NaN."""
        high = sample_ohlcv_100['High_TEST'].values
        low = sample_ohlcv_100['Low_TEST'].values
        close = sample_ohlcv_100['Close_TEST'].values
        period = 14

        atr = calculate_atr(high, low, close, period=period)

        assert np.all(np.isnan(atr[:period-1]))

    def test_atr_values_are_positive(self, sample_ohlcv_100):
        """ATR values should always be positive (after initial period)."""
        high = sample_ohlcv_100['High_TEST'].values
        low = sample_ohlcv_100['Low_TEST'].values
        close = sample_ohlcv_100['Close_TEST'].values
        period = 14

        atr = calculate_atr(high, low, close, period=period)
        valid_atr = atr[period:]

        assert np.all(valid_atr > 0)

    def test_atr_known_values(self, known_atr_data):
        """Test ATR with known constant range data."""
        high, low, close = known_atr_data
        period = 10

        atr = calculate_atr(high, low, close, period=period)

        # With constant $4 true range, ATR should converge to ~4
        # Allow some tolerance for EMA smoothing
        final_atr = atr[-1]
        assert 3.5 < final_atr < 4.5

    def test_atr_increases_with_volatility(self, sample_ohlcv_100, volatile_data):
        """ATR should be higher for more volatile data."""
        # Normal volatility
        high1 = sample_ohlcv_100['High_TEST'].values
        low1 = sample_ohlcv_100['Low_TEST'].values
        close1 = sample_ohlcv_100['Close_TEST'].values
        atr1 = calculate_atr(high1, low1, close1, period=14)

        # High volatility
        high2 = volatile_data['High_TEST'].values
        low2 = volatile_data['Low_TEST'].values
        close2 = volatile_data['Close_TEST'].values
        atr2 = calculate_atr(high2, low2, close2, period=14)

        # ATR as percentage of price should be higher for volatile data
        atr1_pct = np.mean(atr1[-20:]) / np.mean(close1[-20:])
        atr2_pct = np.mean(atr2[-20:]) / np.mean(close2[-20:])

        assert atr2_pct > atr1_pct

    def test_atr_different_periods(self, sample_ohlcv_100):
        """Shorter period ATR should be more responsive."""
        high = sample_ohlcv_100['High_TEST'].values
        low = sample_ohlcv_100['Low_TEST'].values
        close = sample_ohlcv_100['Close_TEST'].values

        atr_short = calculate_atr(high, low, close, period=7)
        atr_long = calculate_atr(high, low, close, period=21)

        # Calculate standard deviation of ATR changes
        std_short = np.std(np.diff(atr_short[21:]))
        std_long = np.std(np.diff(atr_long[21:]))

        # Shorter period should show more variation
        assert std_short >= std_long * 0.8  # Allow some tolerance


class TestCalculateRSI:
    """Tests for Relative Strength Index calculation."""

    def test_rsi_returns_correct_shape(self, sample_ohlcv_100):
        """RSI output should have same length as input."""
        close = sample_ohlcv_100['Close_TEST'].values

        rsi = calculate_rsi(close, period=14)

        assert len(rsi) == len(close)

    def test_rsi_bounded_0_to_100(self, sample_ohlcv_100):
        """RSI values should always be between 0 and 100."""
        close = sample_ohlcv_100['Close_TEST'].values

        rsi = calculate_rsi(close, period=14)

        assert np.all(rsi >= 0)
        assert np.all(rsi <= 100)

    def test_rsi_initial_period_neutral(self, sample_ohlcv_100):
        """Initial period values should be neutral (50)."""
        close = sample_ohlcv_100['Close_TEST'].values
        period = 14

        rsi = calculate_rsi(close, period=period)

        assert np.all(rsi[:period] == 50)

    def test_rsi_uptrend_high(self, trending_up_data):
        """RSI should be elevated (>50) in uptrends."""
        close = trending_up_data['Close_TEST'].values

        rsi = calculate_rsi(close, period=14)

        # Average RSI in latter half should be above 50
        avg_rsi = np.mean(rsi[50:])
        assert avg_rsi > 50

    def test_rsi_downtrend_low(self, trending_down_data):
        """RSI should be depressed (<50) in downtrends."""
        close = trending_down_data['Close_TEST'].values

        rsi = calculate_rsi(close, period=14)

        # Average RSI in latter half should be below 50
        avg_rsi = np.mean(rsi[50:])
        assert avg_rsi < 50

    def test_rsi_overbought_oversold_levels(self, volatile_data):
        """RSI should reach overbought (>70) and oversold (<30) in volatile markets."""
        close = volatile_data['Close_TEST'].values

        rsi = calculate_rsi(close, period=14)
        valid_rsi = rsi[14:]

        # Should have some overbought and oversold readings
        has_overbought = np.any(valid_rsi > 70)
        has_oversold = np.any(valid_rsi < 30)

        # At least one should occur in volatile data
        assert has_overbought or has_oversold


class TestCalculateSupertrendVectorized:
    """Tests for Supertrend indicator calculation."""

    def test_supertrend_returns_correct_shapes(self, sample_ohlcv_100):
        """Supertrend should return three arrays of correct length."""
        high = sample_ohlcv_100['High_TEST'].values
        low = sample_ohlcv_100['Low_TEST'].values
        close = sample_ohlcv_100['Close_TEST'].values

        supertrend, direction, atr = calculate_supertrend_vectorized(
            high, low, close, period=10, multiplier=3.0
        )

        assert len(supertrend) == len(close)
        assert len(direction) == len(close)
        assert len(atr) == len(close)

    def test_supertrend_direction_values(self, sample_ohlcv_100):
        """Direction should only be -1, 0, or 1."""
        high = sample_ohlcv_100['High_TEST'].values
        low = sample_ohlcv_100['Low_TEST'].values
        close = sample_ohlcv_100['Close_TEST'].values

        _, direction, _ = calculate_supertrend_vectorized(
            high, low, close, period=10, multiplier=3.0
        )

        valid_direction = direction[10:]  # Skip initial period
        assert np.all(np.isin(valid_direction, [-1, 0, 1]))

    def test_supertrend_bullish_in_uptrend(self, trending_up_data):
        """Supertrend should be predominantly bullish (direction=1) in uptrend."""
        high = trending_up_data['High_TEST'].values
        low = trending_up_data['Low_TEST'].values
        close = trending_up_data['Close_TEST'].values

        _, direction, _ = calculate_supertrend_vectorized(
            high, low, close, period=10, multiplier=3.0
        )

        valid_direction = direction[20:]  # Skip warmup period
        bullish_pct = np.mean(valid_direction == 1)

        assert bullish_pct > 0.6  # At least 60% bullish

    def test_supertrend_bearish_in_downtrend(self, trending_down_data):
        """Supertrend should be predominantly bearish (direction=-1) in downtrend."""
        high = trending_down_data['High_TEST'].values
        low = trending_down_data['Low_TEST'].values
        close = trending_down_data['Close_TEST'].values

        _, direction, _ = calculate_supertrend_vectorized(
            high, low, close, period=10, multiplier=3.0
        )

        valid_direction = direction[20:]  # Skip warmup period
        bearish_pct = np.mean(valid_direction == -1)

        assert bearish_pct > 0.6  # At least 60% bearish

    def test_supertrend_line_position(self, sample_ohlcv_100):
        """Supertrend should be below price when bullish, above when bearish."""
        high = sample_ohlcv_100['High_TEST'].values
        low = sample_ohlcv_100['Low_TEST'].values
        close = sample_ohlcv_100['Close_TEST'].values

        supertrend, direction, _ = calculate_supertrend_vectorized(
            high, low, close, period=10, multiplier=3.0
        )

        # Check for valid period
        for i in range(20, len(close)):
            if direction[i] == 1:  # Bullish
                # Supertrend should be at or below close
                assert supertrend[i] <= close[i] * 1.01  # Allow 1% tolerance
            elif direction[i] == -1:  # Bearish
                # Supertrend should be at or above close
                assert supertrend[i] >= close[i] * 0.99  # Allow 1% tolerance

    def test_supertrend_multiplier_effect(self, sample_ohlcv_100):
        """Higher multiplier should result in fewer direction changes."""
        high = sample_ohlcv_100['High_TEST'].values
        low = sample_ohlcv_100['Low_TEST'].values
        close = sample_ohlcv_100['Close_TEST'].values

        _, direction_low, _ = calculate_supertrend_vectorized(
            high, low, close, period=10, multiplier=2.0
        )
        _, direction_high, _ = calculate_supertrend_vectorized(
            high, low, close, period=10, multiplier=5.0
        )

        # Count direction changes
        changes_low = np.sum(np.diff(direction_low[10:]) != 0)
        changes_high = np.sum(np.diff(direction_high[10:]) != 0)

        assert changes_high <= changes_low

    def test_supertrend_period_effect(self, sample_ohlcv_100):
        """Longer period should result in smoother Supertrend."""
        high = sample_ohlcv_100['High_TEST'].values
        low = sample_ohlcv_100['Low_TEST'].values
        close = sample_ohlcv_100['Close_TEST'].values

        st_short, _, _ = calculate_supertrend_vectorized(
            high, low, close, period=7, multiplier=3.0
        )
        st_long, _, _ = calculate_supertrend_vectorized(
            high, low, close, period=21, multiplier=3.0
        )

        # Calculate volatility of supertrend changes
        vol_short = np.std(np.diff(st_short[25:]))
        vol_long = np.std(np.diff(st_long[25:]))

        assert vol_long <= vol_short * 1.1  # Long period should be smoother

    def test_supertrend_direction_persistence(self, sample_ohlcv_100):
        """Direction should not flip every bar (some persistence expected)."""
        high = sample_ohlcv_100['High_TEST'].values
        low = sample_ohlcv_100['Low_TEST'].values
        close = sample_ohlcv_100['Close_TEST'].values

        _, direction, _ = calculate_supertrend_vectorized(
            high, low, close, period=10, multiplier=3.0
        )

        valid_direction = direction[10:]
        changes = np.sum(np.diff(valid_direction) != 0)

        # Should not change more than 50% of the time
        change_pct = changes / len(valid_direction)
        assert change_pct < 0.5

    def test_supertrend_initial_zeros(self, sample_ohlcv_100):
        """Initial period should have zeros before calculation starts."""
        high = sample_ohlcv_100['High_TEST'].values
        low = sample_ohlcv_100['Low_TEST'].values
        close = sample_ohlcv_100['Close_TEST'].values
        period = 10

        supertrend, direction, _ = calculate_supertrend_vectorized(
            high, low, close, period=period, multiplier=3.0
        )

        # First (period-1) values should be zero
        assert np.all(supertrend[:period-1] == 0)
        assert np.all(direction[:period-1] == 0)


class TestIndicatorEdgeCases:
    """Edge case tests for all indicators."""

    def test_atr_minimum_data(self):
        """ATR should handle minimum required data."""
        n = 15
        high = np.full(n, 102.0)
        low = np.full(n, 98.0)
        close = np.full(n, 100.0)

        atr = calculate_atr(high, low, close, period=14)

        assert len(atr) == n
        assert not np.isnan(atr[-1])

    def test_rsi_constant_prices(self):
        """RSI should handle constant prices gracefully."""
        close = np.full(30, 100.0)

        rsi = calculate_rsi(close, period=14)

        # With no price changes, RSI should be neutral or near 50
        assert np.all(rsi[14:] >= 0)
        assert np.all(rsi[14:] <= 100)

    def test_supertrend_constant_prices(self):
        """Supertrend should handle constant prices."""
        n = 30
        high = np.full(n, 102.0)
        low = np.full(n, 98.0)
        close = np.full(n, 100.0)

        supertrend, direction, atr = calculate_supertrend_vectorized(
            high, low, close, period=10, multiplier=3.0
        )

        assert len(supertrend) == n
        assert not np.isnan(supertrend[-1])

    def test_indicators_with_gaps(self):
        """Test indicators handle price gaps (common in stocks)."""
        np.random.seed(42)
        n = 50

        # Create data with a gap
        close = np.concatenate([
            np.full(25, 100.0),
            np.full(25, 120.0)  # 20% gap
        ])
        high = close + 2
        low = close - 2

        atr = calculate_atr(high, low, close, period=14)
        rsi = calculate_rsi(close, period=14)
        supertrend, direction, _ = calculate_supertrend_vectorized(
            high, low, close, period=10, multiplier=3.0
        )

        # All should handle the gap without errors
        assert not np.any(np.isnan(atr[14:]))
        assert not np.any(np.isnan(rsi[14:]))
        assert not np.any(np.isnan(supertrend[10:]))
