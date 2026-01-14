"""
Pytest fixtures for Supertrend Trading System tests.
Provides sample OHLCV data and configuration objects for testing.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock yfinance before importing new5
from unittest.mock import MagicMock
sys.modules['yfinance'] = MagicMock()

from new5 import TradingConfig, OptimizedTradingSystem


@pytest.fixture
def default_config():
    """Standard TradingConfig for tests."""
    return TradingConfig(
        symbol="TEST",
        initial_capital=10000.0,
        transaction_cost=0.001,
        st_period=10,
        st_multiplier=3.0,
        htf_period=10,
        htf_multiplier=3.0,
        use_htf_filter=True,
        days_back=365
    )


@pytest.fixture
def trading_system(default_config):
    """OptimizedTradingSystem instance for tests."""
    return OptimizedTradingSystem(default_config)


@pytest.fixture
def sample_ohlcv_100():
    """100 days of synthetic OHLCV data with realistic price movements."""
    np.random.seed(42)
    n = 100

    # Start price
    start_price = 100.0

    # Generate returns with slight upward drift
    returns = np.random.normal(0.001, 0.02, n)

    # Generate close prices
    close = start_price * np.cumprod(1 + returns)

    # Generate OHLC from close
    daily_range = np.random.uniform(0.01, 0.03, n) * close
    high = close + daily_range * np.random.uniform(0.3, 0.7, n)
    low = close - daily_range * np.random.uniform(0.3, 0.7, n)
    open_price = low + (high - low) * np.random.uniform(0.2, 0.8, n)

    # Ensure OHLC consistency
    high = np.maximum(high, np.maximum(open_price, close))
    low = np.minimum(low, np.minimum(open_price, close))

    volume = np.random.uniform(1000000, 5000000, n)

    dates = pd.date_range(end=datetime.now(), periods=n, freq='D')

    df = pd.DataFrame({
        'Open_TEST': open_price,
        'High_TEST': high,
        'Low_TEST': low,
        'Close_TEST': close,
        'Volume_TEST': volume
    }, index=dates)

    return df


@pytest.fixture
def sample_ohlcv_500():
    """500 days of synthetic OHLCV data for longer backtests."""
    np.random.seed(42)
    n = 500

    start_price = 100.0
    returns = np.random.normal(0.0005, 0.018, n)
    close = start_price * np.cumprod(1 + returns)

    daily_range = np.random.uniform(0.01, 0.03, n) * close
    high = close + daily_range * np.random.uniform(0.3, 0.7, n)
    low = close - daily_range * np.random.uniform(0.3, 0.7, n)
    open_price = low + (high - low) * np.random.uniform(0.2, 0.8, n)

    high = np.maximum(high, np.maximum(open_price, close))
    low = np.minimum(low, np.minimum(open_price, close))

    volume = np.random.uniform(1000000, 5000000, n)

    dates = pd.date_range(end=datetime.now(), periods=n, freq='D')

    df = pd.DataFrame({
        'Open_TEST': open_price,
        'High_TEST': high,
        'Low_TEST': low,
        'Close_TEST': close,
        'Volume_TEST': volume
    }, index=dates)

    return df


@pytest.fixture
def trending_up_data():
    """Data with clear upward trend for testing bullish signals."""
    np.random.seed(123)
    n = 100

    # Strong upward trend
    trend = np.linspace(100, 150, n)
    noise = np.random.normal(0, 1, n)
    close = trend + noise

    daily_range = np.random.uniform(1, 3, n)
    high = close + daily_range * 0.6
    low = close - daily_range * 0.4
    open_price = low + (high - low) * np.random.uniform(0.3, 0.7, n)

    high = np.maximum(high, np.maximum(open_price, close))
    low = np.minimum(low, np.minimum(open_price, close))

    volume = np.random.uniform(1000000, 5000000, n)

    dates = pd.date_range(end=datetime.now(), periods=n, freq='D')

    df = pd.DataFrame({
        'Open_TEST': open_price,
        'High_TEST': high,
        'Low_TEST': low,
        'Close_TEST': close,
        'Volume_TEST': volume
    }, index=dates)

    return df


@pytest.fixture
def trending_down_data():
    """Data with clear downward trend for testing bearish signals."""
    np.random.seed(456)
    n = 100

    # Strong downward trend
    trend = np.linspace(150, 100, n)
    noise = np.random.normal(0, 1, n)
    close = trend + noise

    daily_range = np.random.uniform(1, 3, n)
    high = close + daily_range * 0.4
    low = close - daily_range * 0.6
    open_price = low + (high - low) * np.random.uniform(0.3, 0.7, n)

    high = np.maximum(high, np.maximum(open_price, close))
    low = np.minimum(low, np.minimum(open_price, close))

    volume = np.random.uniform(1000000, 5000000, n)

    dates = pd.date_range(end=datetime.now(), periods=n, freq='D')

    df = pd.DataFrame({
        'Open_TEST': open_price,
        'High_TEST': high,
        'Low_TEST': low,
        'Close_TEST': close,
        'Volume_TEST': volume
    }, index=dates)

    return df


@pytest.fixture
def ranging_data():
    """Sideways market data for testing false signal filtering."""
    np.random.seed(789)
    n = 100

    # Mean-reverting sideways market
    close = 100 + 5 * np.sin(np.linspace(0, 4 * np.pi, n)) + np.random.normal(0, 1, n)

    daily_range = np.random.uniform(1, 2, n)
    high = close + daily_range * 0.5
    low = close - daily_range * 0.5
    open_price = low + (high - low) * np.random.uniform(0.3, 0.7, n)

    high = np.maximum(high, np.maximum(open_price, close))
    low = np.minimum(low, np.minimum(open_price, close))

    volume = np.random.uniform(1000000, 5000000, n)

    dates = pd.date_range(end=datetime.now(), periods=n, freq='D')

    df = pd.DataFrame({
        'Open_TEST': open_price,
        'High_TEST': high,
        'Low_TEST': low,
        'Close_TEST': close,
        'Volume_TEST': volume
    }, index=dates)

    return df


@pytest.fixture
def volatile_data():
    """High volatility data for testing Supertrend in volatile markets."""
    np.random.seed(321)
    n = 100

    # High volatility with random walk
    returns = np.random.normal(0.001, 0.05, n)  # 5% daily std
    close = 100 * np.cumprod(1 + returns)

    daily_range = np.random.uniform(0.03, 0.08, n) * close
    high = close + daily_range * np.random.uniform(0.4, 0.6, n)
    low = close - daily_range * np.random.uniform(0.4, 0.6, n)
    open_price = low + (high - low) * np.random.uniform(0.2, 0.8, n)

    high = np.maximum(high, np.maximum(open_price, close))
    low = np.minimum(low, np.minimum(open_price, close))

    volume = np.random.uniform(2000000, 10000000, n)

    dates = pd.date_range(end=datetime.now(), periods=n, freq='D')

    df = pd.DataFrame({
        'Open_TEST': open_price,
        'High_TEST': high,
        'Low_TEST': low,
        'Close_TEST': close,
        'Volume_TEST': volume
    }, index=dates)

    return df


@pytest.fixture
def known_atr_data():
    """Data with known ATR values for validation testing."""
    # Simple case: constant range for predictable ATR
    n = 30
    close = np.array([100.0] * n)
    high = np.array([102.0] * n)  # Constant $2 above close
    low = np.array([98.0] * n)    # Constant $2 below close
    # True range should be $4 for each day (high - low)
    # ATR should converge to ~$4

    return high, low, close


@pytest.fixture
def known_rsi_data():
    """Data with known RSI values for validation testing."""
    # 14 days of gains followed by 14 days of losses
    n = 30
    prices = np.concatenate([
        np.linspace(100, 114, 15),  # 14 gains
        np.linspace(114, 100, 15)   # 14 losses (back to start)
    ])
    return prices


@pytest.fixture
def sample_trades():
    """Sample trade list for statistics testing."""
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')

    return [
        {
            'entry_date': dates[10],
            'entry_price': 100.0,
            'exit_date': dates[20],
            'exit_price': 110.0,
            'profit_loss': 0.098,  # 10% gain minus 0.2% transaction costs
            'entry_index': 10,
            'exit_index': 20,
            'exit_reason': 'signal'
        },
        {
            'entry_date': dates[30],
            'entry_price': 108.0,
            'exit_date': dates[40],
            'exit_price': 102.0,
            'profit_loss': -0.058,  # ~5.6% loss plus transaction costs
            'entry_index': 30,
            'exit_index': 40,
            'exit_reason': 'signal'
        },
        {
            'entry_date': dates[50],
            'entry_price': 105.0,
            'exit_date': dates[60],
            'exit_price': 115.0,
            'profit_loss': 0.093,  # ~9.5% gain minus transaction costs
            'entry_index': 50,
            'exit_index': 60,
            'exit_reason': 'trailing_stop'
        },
        {
            'entry_date': dates[70],
            'entry_price': 112.0,
            'exit_date': dates[80],
            'exit_price': 118.0,
            'profit_loss': 0.052,  # ~5.4% gain minus transaction costs
            'entry_index': 70,
            'exit_index': 80,
            'exit_reason': 'signal'
        },
    ]


@pytest.fixture
def empty_trades():
    """Empty trade list for edge case testing."""
    return []


@pytest.fixture
def single_trade():
    """Single trade for minimum case testing."""
    dates = pd.date_range(end=datetime.now(), periods=50, freq='D')

    return [{
        'entry_date': dates[10],
        'entry_price': 100.0,
        'exit_date': dates[30],
        'exit_price': 120.0,
        'profit_loss': 0.198,  # 20% gain minus transaction costs
        'entry_index': 10,
        'exit_index': 30,
        'exit_reason': 'signal'
    }]
