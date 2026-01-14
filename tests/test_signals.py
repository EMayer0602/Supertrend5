"""
Tests for signal generation logic.
Validates that buy/sell signals are correctly generated based on Supertrend direction changes.
"""
import pytest
import numpy as np
import sys
import os
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock yfinance before importing new5
sys.modules['yfinance'] = MagicMock()

from new5 import (
    generate_signals_vectorized,
    calculate_supertrend_vectorized,
    get_htf_trend
)


class TestGenerateSignalsVectorized:
    """Tests for signal generation function."""

    def test_signals_return_correct_shape(self, sample_ohlcv_100):
        """Buy and sell signals should have same length as input."""
        close = sample_ohlcv_100['Close_TEST'].values
        n = len(close)

        # Create mock direction
        direction = np.ones(n)
        direction[:n//2] = -1  # First half bearish, second half bullish
        supertrend = close  # Simplified

        buy_signals, sell_signals = generate_signals_vectorized(
            close, supertrend, direction, None, use_htf_filter=False
        )

        assert len(buy_signals) == n
        assert len(sell_signals) == n

    def test_signals_are_boolean(self, sample_ohlcv_100):
        """Signals should be boolean arrays."""
        close = sample_ohlcv_100['Close_TEST'].values
        n = len(close)

        direction = np.ones(n)
        direction[:n//2] = -1
        supertrend = close

        buy_signals, sell_signals = generate_signals_vectorized(
            close, supertrend, direction, None, use_htf_filter=False
        )

        assert buy_signals.dtype == bool
        assert sell_signals.dtype == bool

    def test_buy_signal_on_direction_change_to_bullish(self):
        """Buy signal should occur when direction changes from -1 to 1."""
        n = 20
        close = np.full(n, 100.0)
        supertrend = np.full(n, 100.0)

        # Direction: bearish then bullish
        direction = np.full(n, -1.0)
        direction[10:] = 1  # Switch to bullish at index 10

        buy_signals, sell_signals = generate_signals_vectorized(
            close, supertrend, direction, None, use_htf_filter=False
        )

        # Buy signal should be at index 10 (where direction changes to 1)
        assert buy_signals[10] == True
        assert np.sum(buy_signals) == 1  # Only one buy signal

    def test_sell_signal_on_direction_change_to_bearish(self):
        """Sell signal should occur when direction changes from 1 to -1."""
        n = 20
        close = np.full(n, 100.0)
        supertrend = np.full(n, 100.0)

        # Direction: bullish then bearish
        direction = np.full(n, 1.0)
        direction[10:] = -1  # Switch to bearish at index 10

        buy_signals, sell_signals = generate_signals_vectorized(
            close, supertrend, direction, None, use_htf_filter=False
        )

        # Sell signal should be at index 10 (where direction changes to -1)
        assert sell_signals[10] == True
        assert np.sum(sell_signals) == 1  # Only one sell signal

    def test_multiple_direction_changes(self):
        """Test multiple direction changes generate correct signals."""
        n = 50
        close = np.full(n, 100.0)
        supertrend = np.full(n, 100.0)

        # Direction: switches multiple times
        direction = np.array([
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  # 0-9: bearish
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,            # 10-19: bullish
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  # 20-29: bearish
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,            # 30-39: bullish
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  # 40-49: bearish
        ], dtype=float)

        buy_signals, sell_signals = generate_signals_vectorized(
            close, supertrend, direction, None, use_htf_filter=False
        )

        # Buy signals at 10, 30 (bearish to bullish)
        assert buy_signals[10] == True
        assert buy_signals[30] == True
        assert np.sum(buy_signals) == 2

        # Sell signals at 20, 40 (bullish to bearish)
        assert sell_signals[20] == True
        assert sell_signals[40] == True
        assert np.sum(sell_signals) == 2

    def test_no_signal_without_direction_change(self):
        """No signals should occur without direction changes."""
        n = 20
        close = np.full(n, 100.0)
        supertrend = np.full(n, 100.0)
        direction = np.ones(n)  # Constant bullish

        buy_signals, sell_signals = generate_signals_vectorized(
            close, supertrend, direction, None, use_htf_filter=False
        )

        assert np.sum(buy_signals) == 0
        assert np.sum(sell_signals) == 0


class TestHTFFilterModes:
    """Tests for different HTF filter modes."""

    def test_trend_following_mode_blocks_against_htf(self):
        """In trend_following mode, buy signals should only occur when HTF is bullish."""
        n = 30
        close = np.full(n, 100.0)
        supertrend = np.full(n, 100.0)

        # Direction changes to bullish at index 10 and 20
        direction = np.array([
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            -1, -1, -1, -1, -1, -1, -1, -1, 1, 1,
        ], dtype=float)

        # HTF is bearish for first half, bullish for second half
        htf_direction = np.array([
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        ], dtype=float)

        buy_signals, sell_signals = generate_signals_vectorized(
            close, supertrend, direction, htf_direction,
            use_htf_filter=True, filter_mode="trend_following"
        )

        # Buy at index 10 should be blocked (HTF bearish)
        assert buy_signals[10] == False
        # Buy at index 28 should pass (HTF bullish)
        assert buy_signals[28] == True

    def test_confirmation_mode_requires_htf_match(self):
        """In confirmation mode, both entry and exit require HTF alignment."""
        n = 30
        close = np.full(n, 100.0)
        supertrend = np.full(n, 100.0)

        direction = np.array([
            -1, -1, -1, -1, -1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        ], dtype=float)

        htf_direction = np.array([
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        ], dtype=float)

        buy_signals, sell_signals = generate_signals_vectorized(
            close, supertrend, direction, htf_direction,
            use_htf_filter=True, filter_mode="confirmation"
        )

        # Buy at 5 should pass (HTF bullish)
        assert buy_signals[5] == True
        # Sell at 15 should pass (HTF bearish)
        assert sell_signals[15] == True

    def test_exit_only_mode_allows_all_entries(self):
        """In exit_only mode, all entry signals should pass."""
        n = 30
        close = np.full(n, 100.0)
        supertrend = np.full(n, 100.0)

        direction = np.array([
            -1, -1, -1, -1, -1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, 1, 1, 1, 1, 1,
        ], dtype=float)

        htf_direction = np.array([
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        ], dtype=float)  # Always bearish

        buy_signals, sell_signals = generate_signals_vectorized(
            close, supertrend, direction, htf_direction,
            use_htf_filter=True, filter_mode="exit_only"
        )

        # All buy signals should pass regardless of HTF
        assert buy_signals[5] == True
        assert buy_signals[25] == True

    def test_htf_filter_disabled(self):
        """When HTF filter is disabled, signals should pass regardless of HTF."""
        n = 20
        close = np.full(n, 100.0)
        supertrend = np.full(n, 100.0)

        direction = np.full(n, -1.0)
        direction[10:] = 1

        htf_direction = np.full(n, -1.0)  # Always bearish

        buy_signals, _ = generate_signals_vectorized(
            close, supertrend, direction, htf_direction,
            use_htf_filter=False, filter_mode="trend_following"
        )

        # Buy signal should still occur even with bearish HTF when filter disabled
        assert buy_signals[10] == True


class TestSignalsWithRealData:
    """Tests using realistic synthetic data."""

    def test_signals_generated_in_trending_market(self, trending_up_data):
        """Should generate appropriate signals in trending market."""
        high = trending_up_data['High_TEST'].values
        low = trending_up_data['Low_TEST'].values
        close = trending_up_data['Close_TEST'].values

        supertrend, direction, _ = calculate_supertrend_vectorized(
            high, low, close, period=10, multiplier=3.0
        )

        buy_signals, sell_signals = generate_signals_vectorized(
            close, supertrend, direction, None, use_htf_filter=False
        )

        # In uptrend, should have more buy signals than sell after initial establishment
        total_buys = np.sum(buy_signals[20:])
        total_sells = np.sum(sell_signals[20:])

        # At least some signals should be generated
        assert total_buys + total_sells > 0

    def test_signals_in_ranging_market(self, ranging_data):
        """Should generate signals in ranging market (may have false signals)."""
        high = ranging_data['High_TEST'].values
        low = ranging_data['Low_TEST'].values
        close = ranging_data['Close_TEST'].values

        supertrend, direction, _ = calculate_supertrend_vectorized(
            high, low, close, period=10, multiplier=3.0
        )

        buy_signals, sell_signals = generate_signals_vectorized(
            close, supertrend, direction, None, use_htf_filter=False
        )

        # Ranging markets typically have more whipsaws
        total_signals = np.sum(buy_signals) + np.sum(sell_signals)
        assert total_signals >= 2  # Should have some signals due to oscillation

    def test_signal_timing_correctness(self, sample_ohlcv_100):
        """Verify signals occur at correct timing relative to direction changes."""
        high = sample_ohlcv_100['High_TEST'].values
        low = sample_ohlcv_100['Low_TEST'].values
        close = sample_ohlcv_100['Close_TEST'].values

        supertrend, direction, _ = calculate_supertrend_vectorized(
            high, low, close, period=10, multiplier=3.0
        )

        buy_signals, sell_signals = generate_signals_vectorized(
            close, supertrend, direction, None, use_htf_filter=False
        )

        # Verify each buy signal corresponds to direction change -1 -> 1
        for i in range(1, len(buy_signals)):
            if buy_signals[i]:
                assert direction[i-1] == -1 or direction[i-1] == 0
                assert direction[i] == 1

        # Verify each sell signal corresponds to direction change 1 -> -1
        for i in range(1, len(sell_signals)):
            if sell_signals[i]:
                assert direction[i-1] == 1 or direction[i-1] == 0
                assert direction[i] == -1


class TestHTFTrendCalculation:
    """Tests for HTF trend calculation and mapping."""

    def test_htf_trend_returns_series(self, sample_ohlcv_500):
        """get_htf_trend should return a pandas Series."""
        htf_trend = get_htf_trend(sample_ohlcv_500, 'TEST', period=10, multiplier=3.0)

        import pandas as pd
        assert isinstance(htf_trend, pd.Series)
        assert len(htf_trend) == len(sample_ohlcv_500)

    def test_htf_trend_values(self, sample_ohlcv_500):
        """HTF trend should contain valid direction values."""
        htf_trend = get_htf_trend(sample_ohlcv_500, 'TEST', period=10, multiplier=3.0)

        # Should be -1, 0, or 1
        unique_values = htf_trend.unique()
        for val in unique_values:
            assert val in [-1, 0, 1]

    def test_htf_trend_persistence(self, sample_ohlcv_500):
        """HTF trend should show persistence (doesn't change daily)."""
        htf_trend = get_htf_trend(sample_ohlcv_500, 'TEST', period=10, multiplier=3.0)

        # Count changes
        changes = (htf_trend.diff() != 0).sum()

        # HTF (weekly) should change less frequently than daily
        # Expect at most ~20% of days to have changes
        change_rate = changes / len(htf_trend)
        assert change_rate < 0.3  # Less than 30% change rate
