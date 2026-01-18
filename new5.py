import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
import warnings
import asyncio

warnings.filterwarnings('ignore')

# Fix for Python 3.10+ event loop issue with ib_insync
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

from ib_insync import IB, Stock, util

# =============================================================================
# IB CONNECTION
# =============================================================================
_ib_connection = None

def get_ib_connection(host: str = '127.0.0.1', port: int = 7497, client_id: int = 21) -> IB:
    """Get or create IB connection"""
    global _ib_connection
    if _ib_connection is None or not _ib_connection.isConnected():
        _ib_connection = IB()
        _ib_connection.connect(host, port, clientId=client_id)
        print(f"Connected to TWS at {host}:{port}")
    return _ib_connection

def disconnect_ib():
    """Disconnect from IB"""
    global _ib_connection
    if _ib_connection and _ib_connection.isConnected():
        _ib_connection.disconnect()
        print("Disconnected from TWS")
    _ib_connection = None

def download_from_tws(symbol: str, days_back: int = 365) -> pd.DataFrame:
    """Download historical data from TWS"""
    ib = get_ib_connection()

    contract = Stock(symbol, 'SMART', 'USD')
    ib.qualifyContracts(contract)

    # Calculate duration
    if days_back <= 365:
        duration = f'{days_back} D'
    else:
        months = days_back // 30
        if months <= 12:
            duration = f'{months} M'
        else:
            years = months // 12
            duration = f'{years} Y'

    bars = ib.reqHistoricalData(
        contract,
        endDateTime='',
        durationStr=duration,
        barSizeSetting='1 day',
        whatToShow='TRADES',
        useRTH=True,
        formatDate=1
    )

    if not bars:
        raise ValueError(f"No data available for {symbol}")

    df = util.df(bars)
    df.set_index('date', inplace=True)

    # Rename columns to match expected format with symbol suffix
    df.rename(columns={
        'open': f'Open_{symbol}',
        'high': f'High_{symbol}',
        'low': f'Low_{symbol}',
        'close': f'Close_{symbol}',
        'volume': f'Volume_{symbol}'
    }, inplace=True)

    # Small delay to avoid pacing violations
    ib.sleep(0.3)

    return df

# =============================================================================
# SYMBOL LISTS - DOW 30, NASDAQ 100, ALL TICKERS
# =============================================================================

# DOW JONES 30 Components (as of 2024)
DOW_30 = [
    'AAPL', 'AMGN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS', 'DOW',
    'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM',
    'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'V', 'VZ', 'WBA', 'WMT'
]

# NASDAQ 100 Components (as of 2024)
NASDAQ_100 = [
    'AAPL', 'ABNB', 'ADBE', 'ADI', 'ADP', 'ADSK', 'AEP', 'AMAT', 'AMD', 'AMGN',
    'AMZN', 'ANSS', 'ARM', 'ASML', 'AVGO', 'AZN', 'BIIB', 'BKNG', 'BKR', 'CCEP',
    'CDNS', 'CDW', 'CEG', 'CHTR', 'CMCSA', 'COST', 'CPRT', 'CRWD', 'CSCO', 'CSGP',
    'CSX', 'CTAS', 'CTSH', 'DASH', 'DDOG', 'DLTR', 'DXCM', 'EA', 'EXC', 'FANG',
    'FAST', 'FTNT', 'GEHC', 'GFS', 'GILD', 'GOOG', 'GOOGL', 'HON', 'IDXX', 'ILMN',
    'INTC', 'INTU', 'ISRG', 'KDP', 'KHC', 'KLAC', 'LIN', 'LRCX', 'LULU', 'MAR',
    'MCHP', 'MDB', 'MDLZ', 'MELI', 'META', 'MNST', 'MRNA', 'MRVL', 'MSFT', 'MU',
    'NFLX', 'NVDA', 'NXPI', 'ODFL', 'ON', 'ORLY', 'PANW', 'PAYX', 'PCAR', 'PDD',
    'PEP', 'PYPL', 'QCOM', 'REGN', 'ROP', 'ROST', 'SBUX', 'SMCI', 'SNPS', 'SPLK',
    'TEAM', 'TMUS', 'TSLA', 'TTD', 'TTWO', 'TXN', 'VRSK', 'WDAY', 'XEL', 'ZS'
]

# Combined unique tickers from DOW 30 + NASDAQ 100
ALL_TICKERS = list(dict.fromkeys(DOW_30 + NASDAQ_100))  # Remove duplicates, preserve order


# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass
class TradingConfig:
    """Configuration for the trading system"""
    symbol: str = "MSFT"
    initial_capital: float = 10000.0
    transaction_cost: float = 0.001

    # Supertrend Parameters (will be optimized)
    st_period: int = 10
    st_multiplier: float = 3.0

    # HTF Filter Parameters
    htf_period: int = 10
    htf_multiplier: float = 3.0
    use_htf_filter: bool = True

    # Backtest Period
    days_back: int = 365  # 1 year


# =============================================================================
# OPTIMIZED SUPERTREND CALCULATION (Vectorized with NumPy)
# =============================================================================
def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Calculate Average True Range using vectorized operations"""
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]

    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)

    true_range = np.maximum(np.maximum(tr1, tr2), tr3)

    # EMA-based ATR for smoother results
    atr = np.zeros_like(true_range)
    atr[:period] = np.nan
    atr[period-1] = np.mean(true_range[:period])

    multiplier = 2 / (period + 1)
    for i in range(period, len(true_range)):
        atr[i] = true_range[i] * multiplier + atr[i-1] * (1 - multiplier)

    return atr


def calculate_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate RSI (Relative Strength Index) using vectorized operations"""
    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    # Calculate average gain/loss using EMA
    avg_gain = np.zeros(len(close))
    avg_loss = np.zeros(len(close))

    # Initial SMA
    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])

    # EMA for subsequent values
    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i-1]) / period

    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    rsi[:period] = 50  # Neutral for initial period

    return rsi


def calculate_supertrend_vectorized(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                                     period: int = 10, multiplier: float = 3.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Optimized Supertrend calculation using NumPy vectorization
    Returns: (supertrend, direction, atr)
    - direction: 1 = bullish (price above supertrend), -1 = bearish
    """
    n = len(close)
    atr = calculate_atr(high, low, close, period)

    hl2 = (high + low) / 2

    # Basic bands
    basic_upper = hl2 + (multiplier * atr)
    basic_lower = hl2 - (multiplier * atr)

    # Final bands
    final_upper = np.zeros(n)
    final_lower = np.zeros(n)
    supertrend = np.zeros(n)
    direction = np.zeros(n)

    # Initialize
    final_upper[period-1] = basic_upper[period-1]
    final_lower[period-1] = basic_lower[period-1]

    for i in range(period, n):
        # Final Upper Band
        if basic_upper[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]:
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = final_upper[i-1]

        # Final Lower Band
        if basic_lower[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]:
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = final_lower[i-1]

    # Calculate Supertrend and Direction
    for i in range(period, n):
        if i == period:
            if close[i] <= final_upper[i]:
                supertrend[i] = final_upper[i]
                direction[i] = -1
            else:
                supertrend[i] = final_lower[i]
                direction[i] = 1
        else:
            if supertrend[i-1] == final_upper[i-1]:
                if close[i] <= final_upper[i]:
                    supertrend[i] = final_upper[i]
                    direction[i] = -1
                else:
                    supertrend[i] = final_lower[i]
                    direction[i] = 1
            else:
                if close[i] >= final_lower[i]:
                    supertrend[i] = final_lower[i]
                    direction[i] = 1
                else:
                    supertrend[i] = final_upper[i]
                    direction[i] = -1

    return supertrend, direction, atr


# =============================================================================
# MOVING AVERAGE INDICATORS (JMA, KAMA, SMA, EMA)
# =============================================================================

def calculate_jma(series: pd.Series, period: int = 7, phase: int = 50) -> pd.Series:
    """Calculate Jurik Moving Average (JMA)"""
    phase_ratio = (phase + 100) / 200
    beta = 0.45 * (period - 1) / (0.45 * (period - 1) + 2)
    alpha = beta ** 3

    jma = pd.Series(index=series.index, dtype=float)
    jma.iloc[0] = series.iloc[0]

    e0 = series.iloc[0]
    e1 = 0
    e2 = 0

    for i in range(1, len(series)):
        e0 = (1 - alpha) * series.iloc[i] + alpha * e0
        e1 = (series.iloc[i] - e0) * (1 - beta) + beta * e1
        e2 = (e0 + phase_ratio * e1 - jma.iloc[i-1]) * ((1 - alpha) ** 2) + (alpha ** 2) * e2
        jma.iloc[i] = jma.iloc[i-1] + e2

    return jma


def calculate_kama(series: pd.Series, period: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    """Calculate Kaufman Adaptive Moving Average (KAMA)"""
    change = abs(series - series.shift(period))
    volatility = abs(series - series.shift(1)).rolling(window=period).sum()

    er = change / volatility
    er = er.fillna(0)

    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    kama = pd.Series(index=series.index, dtype=float)
    kama.iloc[period-1] = series.iloc[period-1]

    for i in range(period, len(series)):
        kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (series.iloc[i] - kama.iloc[i-1])

    return kama


def calculate_sma(series: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average"""
    return series.rolling(window=period).mean()


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average"""
    return series.ewm(span=period, adjust=False).mean()


def get_ma_crossover_signals(close: np.ndarray, fast_ma: np.ndarray, slow_ma: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Generate buy/sell signals from MA crossover"""
    n = len(close)
    buy_signals = np.zeros(n, dtype=bool)
    sell_signals = np.zeros(n, dtype=bool)

    for i in range(1, n):
        if np.isnan(fast_ma[i]) or np.isnan(slow_ma[i]):
            continue
        if np.isnan(fast_ma[i-1]) or np.isnan(slow_ma[i-1]):
            continue
        # Buy when fast crosses above slow
        if fast_ma[i] > slow_ma[i] and fast_ma[i-1] <= slow_ma[i-1]:
            buy_signals[i] = True
        # Sell when fast crosses below slow
        elif fast_ma[i] < slow_ma[i] and fast_ma[i-1] >= slow_ma[i-1]:
            sell_signals[i] = True

    return buy_signals, sell_signals


# =============================================================================
# HTF (HIGHER TIME FRAME) FILTER
# =============================================================================
def resample_to_weekly(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Resample daily data to weekly for HTF analysis"""
    ohlc_dict = {
        f'Open_{symbol}': 'first',
        f'High_{symbol}': 'max',
        f'Low_{symbol}': 'min',
        f'Close_{symbol}': 'last',
        f'Volume_{symbol}': 'sum'
    }

    weekly = df.resample('W').agg(ohlc_dict).dropna()
    return weekly


def get_htf_trend(daily_df: pd.DataFrame, symbol: str, period: int, multiplier: float) -> pd.Series:
    """
    Calculate HTF (Weekly) Supertrend and map back to daily timeframe
    Returns: Series with HTF direction aligned to daily index
    """
    weekly_df = resample_to_weekly(daily_df, symbol)

    if len(weekly_df) < period + 5:
        print("Warning: Not enough weekly data for HTF filter")
        return pd.Series(1, index=daily_df.index)

    high = weekly_df[f'High_{symbol}'].values
    low = weekly_df[f'Low_{symbol}'].values
    close = weekly_df[f'Close_{symbol}'].values

    _, htf_direction, _ = calculate_supertrend_vectorized(high, low, close, period, multiplier)

    # Create weekly direction series
    htf_series = pd.Series(htf_direction, index=weekly_df.index)

    # Forward-fill to daily timeframe
    daily_htf = htf_series.reindex(daily_df.index, method='ffill')
    daily_htf = daily_htf.fillna(method='bfill')

    return daily_htf


# =============================================================================
# VECTORIZED TRADING SIGNALS
# =============================================================================
def generate_signals_vectorized(close: np.ndarray, supertrend: np.ndarray,
                                 direction: np.ndarray, htf_direction: Optional[np.ndarray] = None,
                                 use_htf_filter: bool = True,
                                 filter_mode: str = "trend_following") -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate buy/sell signals using vectorized operations

    filter_mode options:
    - "trend_following": Only trade in HTF direction (Long only when HTF bullish)
    - "confirmation": Use HTF as confirmation (original strict mode)
    - "exit_only": Use all entries, but HTF filters exits

    Returns: (buy_signals, sell_signals) as boolean arrays
    """
    n = len(close)

    # Detect direction changes
    prev_direction = np.roll(direction, 1)
    prev_direction[0] = 0

    # Signal when direction changes from -1 to 1 (buy) or 1 to -1 (sell)
    raw_buy = (prev_direction == -1) & (direction == 1)
    raw_sell = (prev_direction == 1) & (direction == -1)

    if use_htf_filter and htf_direction is not None:
        if filter_mode == "trend_following":
            # Only trade in direction of HTF trend
            # Buy signals only when HTF is bullish, Sell signals to exit longs when direction changes
            # No short trading - only long when HTF bullish
            buy_signals = raw_buy & (htf_direction == 1)
            # Exit long when either: sell signal AND still in HTF bullish, OR HTF turns bearish
            sell_signals = raw_sell  # Always allow exits
        elif filter_mode == "confirmation":
            # Original strict mode - only enter when HTF confirms
            buy_signals = raw_buy & (htf_direction == 1)
            sell_signals = raw_sell & (htf_direction == -1)
        else:  # exit_only
            buy_signals = raw_buy
            sell_signals = raw_sell
    else:
        buy_signals = raw_buy
        sell_signals = raw_sell

    return buy_signals, sell_signals


# =============================================================================
# OPTIMIZED TRADING SYSTEM CLASS
# =============================================================================
class OptimizedTradingSystem:
    def __init__(self, config: TradingConfig):
        self.config = config

    def generate_trades(self, df: pd.DataFrame, buy_signals: np.ndarray,
                        sell_signals: np.ndarray, long_only: bool = False,
                        use_trailing_stop: bool = False, trailing_stop_pct: float = 0.05,
                        rsi: Optional[np.ndarray] = None, rsi_oversold: int = 30, rsi_overbought: int = 70) -> Tuple[List[Dict], List[Dict]]:
        """Generate trade lists from signals with optional trailing stop and RSI filter

        Args:
            long_only: If True, only generate long trades (no shorts)
            use_trailing_stop: If True, use trailing stop to lock in profits
            trailing_stop_pct: Trailing stop percentage (e.g., 0.05 = 5%)
            rsi: RSI values array (optional)
            rsi_oversold: RSI level considered oversold (buy opportunity)
            rsi_overbought: RSI level considered overbought (sell opportunity)
        """
        symbol = self.config.symbol
        close_col = f'Close_{symbol}'
        high_col = f'High_{symbol}'

        long_trades = []
        short_trades = []

        position = None  # None, 'long', 'short'
        entry_price = 0
        entry_date = None
        entry_idx = 0
        highest_since_entry = 0  # For trailing stop

        close_prices = df[close_col].values
        high_prices = df[high_col].values
        dates = df.index

        for i in range(len(df)):
            # Check trailing stop first (if in position)
            if use_trailing_stop and position == 'long':
                highest_since_entry = max(highest_since_entry, high_prices[i])
                trailing_stop_price = highest_since_entry * (1 - trailing_stop_pct)

                if close_prices[i] < trailing_stop_price:
                    # Trailing stop hit - exit position
                    profit_loss = (close_prices[i] - entry_price) / entry_price
                    profit_loss -= self.config.transaction_cost * 2
                    long_trades.append({
                        'entry_date': entry_date,
                        'entry_price': entry_price,
                        'exit_date': dates[i],
                        'exit_price': close_prices[i],
                        'profit_loss': profit_loss,
                        'entry_index': entry_idx,
                        'exit_index': i,
                        'exit_reason': 'trailing_stop'
                    })
                    position = None
                    highest_since_entry = 0
                    continue

            # RSI filter for entry
            rsi_allows_buy = True
            rsi_allows_sell = True
            if rsi is not None:
                # Only buy if RSI is not overbought (gives room to grow)
                rsi_allows_buy = rsi[i] < rsi_overbought
                # Only sell/short if RSI is not oversold (might bounce)
                rsi_allows_sell = rsi[i] > rsi_oversold

            if buy_signals[i] and rsi_allows_buy:
                # Close short position if exists
                if position == 'short':
                    profit_loss = (entry_price - close_prices[i]) / entry_price
                    profit_loss -= self.config.transaction_cost * 2
                    short_trades.append({
                        'entry_date': entry_date,
                        'entry_price': entry_price,
                        'exit_date': dates[i],
                        'exit_price': close_prices[i],
                        'profit_loss': profit_loss,
                        'entry_index': entry_idx,
                        'exit_index': i,
                        'exit_reason': 'signal'
                    })
                    position = None

                # Open long position
                if position is None:
                    position = 'long'
                    entry_price = close_prices[i]
                    entry_date = dates[i]
                    entry_idx = i
                    highest_since_entry = high_prices[i]

            elif sell_signals[i] and rsi_allows_sell:
                # Close long position if exists
                if position == 'long':
                    profit_loss = (close_prices[i] - entry_price) / entry_price
                    profit_loss -= self.config.transaction_cost * 2
                    long_trades.append({
                        'entry_date': entry_date,
                        'entry_price': entry_price,
                        'exit_date': dates[i],
                        'exit_price': close_prices[i],
                        'profit_loss': profit_loss,
                        'entry_index': entry_idx,
                        'exit_index': i,
                        'exit_reason': 'signal'
                    })
                    position = None
                    highest_since_entry = 0

                # Open short position only if not long_only mode
                if not long_only and position is None:
                    position = 'short'
                    entry_price = close_prices[i]
                    entry_date = dates[i]
                    entry_idx = i

        return long_trades, short_trades

    def calculate_equity_curve_vectorized(self, df: pd.DataFrame,
                                          long_trades: List[Dict],
                                          short_trades: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate equity curves using vectorized operations"""
        n = len(df)
        symbol = self.config.symbol
        close_col = f'Close_{symbol}'
        close_prices = df[close_col].values

        long_equity = np.full(n, self.config.initial_capital)
        short_equity = np.full(n, self.config.initial_capital)

        # Process long trades
        for trade in long_trades:
            entry_idx = trade['entry_index']
            exit_idx = trade['exit_index']
            entry_price = trade['entry_price']

            # Calculate daily equity during trade
            for i in range(entry_idx, exit_idx + 1):
                daily_return = (close_prices[i] - entry_price) / entry_price
                daily_return -= self.config.transaction_cost  # Entry cost
                if i == exit_idx:
                    daily_return -= self.config.transaction_cost  # Exit cost

                # Get capital at entry
                capital_at_entry = long_equity[entry_idx - 1] if entry_idx > 0 else self.config.initial_capital
                long_equity[i] = capital_at_entry * (1 + daily_return)

            # Carry forward equity after trade
            if exit_idx + 1 < n:
                for i in range(exit_idx + 1, n):
                    if i < n and (not any(t['entry_index'] <= i <= t['exit_index'] for t in long_trades if t != trade)):
                        long_equity[i] = long_equity[exit_idx]

        # Process short trades
        for trade in short_trades:
            entry_idx = trade['entry_index']
            exit_idx = trade['exit_index']
            entry_price = trade['entry_price']

            for i in range(entry_idx, exit_idx + 1):
                daily_return = (entry_price - close_prices[i]) / entry_price
                daily_return -= self.config.transaction_cost
                if i == exit_idx:
                    daily_return -= self.config.transaction_cost

                capital_at_entry = short_equity[entry_idx - 1] if entry_idx > 0 else self.config.initial_capital
                short_equity[i] = capital_at_entry * (1 + daily_return)

            if exit_idx + 1 < n:
                for i in range(exit_idx + 1, n):
                    if i < n and (not any(t['entry_index'] <= i <= t['exit_index'] for t in short_trades if t != trade)):
                        short_equity[i] = short_equity[exit_idx]

        # Fix equity curves - forward fill properly
        long_equity = self._fix_equity_curve(long_equity, long_trades)
        short_equity = self._fix_equity_curve(short_equity, short_trades)

        # Combined equity
        combined_equity = long_equity + short_equity - self.config.initial_capital

        # Buy and hold equity
        buy_hold_equity = (close_prices / close_prices[0]) * self.config.initial_capital

        return long_equity, short_equity, combined_equity, buy_hold_equity

    def _fix_equity_curve(self, equity: np.ndarray, trades: List[Dict]) -> np.ndarray:
        """Fix equity curve with proper forward filling"""
        result = np.full_like(equity, self.config.initial_capital)
        current_equity = self.config.initial_capital

        # Sort trades by entry index
        sorted_trades = sorted(trades, key=lambda x: x['entry_index'])

        trade_idx = 0
        in_trade = False
        trade_entry_equity = self.config.initial_capital

        for i in range(len(equity)):
            # Check if entering a trade
            if trade_idx < len(sorted_trades):
                trade = sorted_trades[trade_idx]
                if i == trade['entry_index']:
                    in_trade = True
                    trade_entry_equity = current_equity

                if in_trade and trade['entry_index'] <= i <= trade['exit_index']:
                    # During trade - calculate equity
                    pnl = trade['profit_loss'] * (i - trade['entry_index']) / max(1, trade['exit_index'] - trade['entry_index'])
                    result[i] = trade_entry_equity * (1 + pnl)

                    if i == trade['exit_index']:
                        current_equity = trade_entry_equity * (1 + trade['profit_loss'])
                        result[i] = current_equity
                        in_trade = False
                        trade_idx += 1
                else:
                    result[i] = current_equity
            else:
                result[i] = current_equity

        return result

    def calculate_statistics(self, trades: List[Dict], equity_curve: np.ndarray) -> Dict:
        """Calculate comprehensive trading statistics"""
        if not trades:
            return self._empty_statistics()

        profits = [t['profit_loss'] for t in trades if t['profit_loss'] > 0]
        losses = [t['profit_loss'] for t in trades if t['profit_loss'] <= 0]

        total_trades = len(trades)
        winning_trades = len(profits)
        losing_trades = len(losses)

        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        avg_profit = np.mean(profits) if profits else 0
        avg_loss = np.mean(losses) if losses else 0

        total_profit = sum(profits)
        total_loss = abs(sum(losses))

        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

        # Total return
        total_return = (equity_curve[-1] - self.config.initial_capital) / self.config.initial_capital

        # Max drawdown
        rolling_max = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - rolling_max) / rolling_max
        max_drawdown = abs(np.min(drawdowns))

        # Sharpe Ratio (annualized)
        returns = np.diff(equity_curve) / equity_curve[:-1]
        sharpe_ratio = np.sqrt(252) * (np.mean(returns) / np.std(returns)) if np.std(returns) > 0 else 0

        # Sortino Ratio
        negative_returns = returns[returns < 0]
        sortino_ratio = np.sqrt(252) * (np.mean(returns) / np.std(negative_returns)) if len(negative_returns) > 0 and np.std(negative_returns) > 0 else 0

        # Calmar Ratio
        calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0

        # Average trade duration
        durations = [(t['exit_index'] - t['entry_index']) for t in trades]
        avg_duration = np.mean(durations) if durations else 0

        # Expectancy
        expectancy = (win_rate * avg_profit) - ((1 - win_rate) * abs(avg_loss))

        return {
            "Total Trades": total_trades,
            "Winning Trades": winning_trades,
            "Losing Trades": losing_trades,
            "Win Rate": f"{win_rate:.1%}",
            "Avg Profit": f"{avg_profit:.2%}",
            "Avg Loss": f"{avg_loss:.2%}",
            "Profit Factor": f"{profit_factor:.2f}",
            "Total Return": f"{total_return:.2%}",
            "Max Drawdown": f"{max_drawdown:.2%}",
            "Sharpe Ratio": f"{sharpe_ratio:.2f}",
            "Sortino Ratio": f"{sortino_ratio:.2f}",
            "Calmar Ratio": f"{calmar_ratio:.2f}",
            "Avg Trade Duration": f"{avg_duration:.1f} days",
            "Expectancy": f"{expectancy:.2%}",
        }

    def _empty_statistics(self) -> Dict:
        return {
            "Total Trades": 0,
            "Winning Trades": 0,
            "Losing Trades": 0,
            "Win Rate": "0.0%",
            "Avg Profit": "0.00%",
            "Avg Loss": "0.00%",
            "Profit Factor": "0.00",
            "Total Return": "0.00%",
            "Max Drawdown": "0.00%",
            "Sharpe Ratio": "0.00",
            "Sortino Ratio": "0.00",
            "Calmar Ratio": "0.00",
            "Avg Trade Duration": "0.0 days",
            "Expectancy": "0.00%",
        }


# =============================================================================
# PARAMETER OPTIMIZATION
# =============================================================================
def optimize_parameters(df: pd.DataFrame, symbol: str,
                        period_range: range = range(7, 21, 2),
                        multiplier_range: np.ndarray = np.arange(2.0, 5.0, 0.5),
                        use_htf: bool = True,
                        long_only: bool = True,
                        filter_mode: str = "trend_following",
                        optimize_for: str = "return") -> Tuple[int, float, Dict]:
    """
    Grid search for optimal Supertrend parameters

    optimize_for: "return", "sharpe", "profit_factor", "win_rate"
    """
    print("\n" + "="*60)
    print("PARAMETER OPTIMIZATION")
    print("="*60)
    print(f"Mode: {'Long Only' if long_only else 'Long & Short'}")
    print(f"HTF Filter: {filter_mode if use_htf else 'Disabled'}")
    print(f"Optimizing for: {optimize_for}")

    close_col = f'Close_{symbol}'
    high_col = f'High_{symbol}'
    low_col = f'Low_{symbol}'

    high = df[high_col].values
    low = df[low_col].values
    close = df[close_col].values

    best_score = -np.inf
    best_period = 10
    best_multiplier = 3.0
    best_stats = {}

    results = []

    config = TradingConfig(symbol=symbol)
    system = OptimizedTradingSystem(config)

    total_combinations = len(period_range) * len(multiplier_range)
    current = 0

    for period in period_range:
        for multiplier in multiplier_range:
            current += 1

            try:
                # Calculate Supertrend
                supertrend, direction, _ = calculate_supertrend_vectorized(high, low, close, period, multiplier)

                # Get HTF trend if enabled
                htf_direction = None
                if use_htf:
                    htf_direction = get_htf_trend(df, symbol, period, multiplier).values

                # Generate signals
                buy_signals, sell_signals = generate_signals_vectorized(
                    close, supertrend, direction, htf_direction, use_htf, filter_mode
                )

                # Generate trades
                long_trades, short_trades = system.generate_trades(df, buy_signals, sell_signals, long_only)

                all_trades = long_trades + short_trades

                if len(all_trades) < 3:
                    continue

                # Calculate equity
                long_eq, short_eq, combined_eq, buy_hold = system.calculate_equity_curve_vectorized(
                    df, long_trades, short_trades
                )

                # Calculate metrics
                total_return = (combined_eq[-1] - config.initial_capital) / config.initial_capital
                buy_hold_return = (buy_hold[-1] - config.initial_capital) / config.initial_capital

                # Sharpe Ratio
                returns = np.diff(combined_eq) / combined_eq[:-1]
                returns = returns[~np.isnan(returns)]
                sharpe = np.sqrt(252) * (np.mean(returns) / np.std(returns)) if len(returns) > 0 and np.std(returns) > 0 else 0

                # Win Rate
                wins = len([t for t in all_trades if t['profit_loss'] > 0])
                win_rate = wins / len(all_trades) if all_trades else 0

                # Profit Factor
                total_profit = sum([t['profit_loss'] for t in all_trades if t['profit_loss'] > 0])
                total_loss = abs(sum([t['profit_loss'] for t in all_trades if t['profit_loss'] <= 0]))
                profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

                # Max Drawdown
                rolling_max = np.maximum.accumulate(combined_eq)
                drawdowns = (combined_eq - rolling_max) / rolling_max
                max_dd = abs(np.min(drawdowns))

                # Risk-adjusted return (Return / Max DD)
                risk_adj_return = total_return / max_dd if max_dd > 0 else total_return

                # Select score based on optimization target
                if optimize_for == "return":
                    score = total_return
                elif optimize_for == "sharpe":
                    score = sharpe
                elif optimize_for == "profit_factor":
                    score = profit_factor if profit_factor != float('inf') else 10
                elif optimize_for == "win_rate":
                    score = win_rate
                elif optimize_for == "risk_adjusted":
                    score = risk_adj_return
                else:
                    score = total_return

                results.append({
                    'period': period,
                    'multiplier': multiplier,
                    'sharpe': sharpe,
                    'return': total_return,
                    'win_rate': win_rate,
                    'profit_factor': profit_factor,
                    'max_dd': max_dd,
                    'trades': len(all_trades),
                    'score': score
                })

                # Update best if better score AND positive return AND outperforms buy & hold
                if score > best_score and total_return > 0 and len(all_trades) >= 3:
                    best_score = score
                    best_period = period
                    best_multiplier = multiplier
                    best_stats = {
                        'sharpe': sharpe,
                        'return': total_return,
                        'win_rate': win_rate,
                        'profit_factor': profit_factor,
                        'max_dd': max_dd,
                        'long_trades': len(long_trades),
                        'short_trades': len(short_trades)
                    }

            except Exception as e:
                pass

            # Progress
            if current % 10 == 0:
                print(f"Progress: {current}/{total_combinations} ({100*current/total_combinations:.0f}%)")

    # If no positive result found, find best among all
    if best_stats == {}:
        print("\nNo profitable strategy found. Selecting least negative...")
        best_result = max(results, key=lambda x: x['return']) if results else None
        if best_result:
            best_period = best_result['period']
            best_multiplier = best_result['multiplier']
            best_stats = best_result

    print(f"\nBest Parameters Found:")
    print(f"  Period: {best_period}")
    print(f"  Multiplier: {best_multiplier}")
    print(f"  Return: {best_stats.get('return', 0):.2%}")
    print(f"  Win Rate: {best_stats.get('win_rate', 0):.1%}")
    print(f"  Profit Factor: {best_stats.get('profit_factor', 0):.2f}")
    print(f"  Max Drawdown: {best_stats.get('max_dd', 0):.2%}")

    return best_period, best_multiplier, results


# =============================================================================
# VISUALIZATION
# =============================================================================
def create_visualization(df: pd.DataFrame, symbol: str, config: TradingConfig,
                         supertrend: np.ndarray, direction: np.ndarray,
                         htf_direction: np.ndarray,
                         buy_signals: np.ndarray, sell_signals: np.ndarray,
                         long_trades: List[Dict], short_trades: List[Dict],
                         long_equity: np.ndarray, short_equity: np.ndarray,
                         combined_equity: np.ndarray, buy_hold_equity: np.ndarray) -> go.Figure:
    """Create comprehensive trading visualization"""

    close_col = f'Close_{symbol}'
    high_col = f'High_{symbol}'
    low_col = f'Low_{symbol}'
    open_col = f'Open_{symbol}'

    # Skip initial period for cleaner visualization
    skip = max(config.st_period, config.htf_period) + 10
    plot_df = df.iloc[skip:]
    plot_indices = np.arange(skip, len(df))

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            f'{symbol} Price with Supertrend & HTF Filter',
            f'Equity Curves (Initial: ${config.initial_capital:,.0f})',
            'HTF Trend Direction'
        ),
        row_heights=[0.5, 0.35, 0.15]
    )

    # Row 1: Candlestick Chart
    fig.add_trace(
        go.Candlestick(
            x=plot_df.index,
            open=plot_df[open_col],
            high=plot_df[high_col],
            low=plot_df[low_col],
            close=plot_df[close_col],
            name='Price',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ),
        row=1, col=1
    )

    # Supertrend line with color based on direction
    for i in range(len(plot_df) - 1):
        idx = skip + i
        color = '#26a69a' if direction[idx] == 1 else '#ef5350'
        fig.add_trace(
            go.Scatter(
                x=[plot_df.index[i], plot_df.index[i+1]],
                y=[supertrend[idx], supertrend[idx+1]],
                mode='lines',
                line=dict(color=color, width=2),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=1
        )

    # Buy signals
    buy_dates = plot_df.index[buy_signals[skip:]]
    buy_prices = plot_df[close_col].values[buy_signals[skip:]]
    if len(buy_dates) > 0:
        fig.add_trace(
            go.Scatter(
                x=buy_dates,
                y=buy_prices,
                mode='markers',
                name='Buy Signal',
                marker=dict(symbol='triangle-up', size=15, color='#26a69a',
                           line=dict(width=2, color='white'))
            ),
            row=1, col=1
        )

    # Sell signals
    sell_dates = plot_df.index[sell_signals[skip:]]
    sell_prices = plot_df[close_col].values[sell_signals[skip:]]
    if len(sell_dates) > 0:
        fig.add_trace(
            go.Scatter(
                x=sell_dates,
                y=sell_prices,
                mode='markers',
                name='Sell Signal',
                marker=dict(symbol='triangle-down', size=15, color='#ef5350',
                           line=dict(width=2, color='white'))
            ),
            row=1, col=1
        )

    # Row 2: Equity Curves
    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=long_equity[skip:],
            mode='lines',
            name='Long Equity',
            line=dict(color='#26a69a', width=2)
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=short_equity[skip:],
            mode='lines',
            name='Short Equity',
            line=dict(color='#ef5350', width=2)
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=combined_equity[skip:],
            mode='lines',
            name='Combined Strategy',
            line=dict(color='#7c4dff', width=3)
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=buy_hold_equity[skip:],
            mode='lines',
            name='Buy & Hold',
            line=dict(color='#ffa726', width=2, dash='dash')
        ),
        row=2, col=1
    )

    # Row 3: HTF Direction
    htf_colors = ['#ef5350' if d == -1 else '#26a69a' for d in htf_direction[skip:]]
    fig.add_trace(
        go.Bar(
            x=plot_df.index,
            y=htf_direction[skip:],
            name='HTF Trend',
            marker_color=htf_colors
        ),
        row=3, col=1
    )

    # Layout
    fig.update_layout(
        title=dict(
            text=f'<b>{symbol} Supertrend Trading System with HTF Filter</b><br>'
                 f'<sup>Period: {config.st_period} | Multiplier: {config.st_multiplier} | HTF Filter: {"ON" if config.use_htf_filter else "OFF"}</sup>',
            x=0.5,
            font=dict(size=20)
        ),
        height=1000,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            bgcolor='rgba(255,255,255,0.8)'
        ),
        xaxis_rangeslider_visible=False,
        template='plotly_white',
        hovermode='x unified'
    )

    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Equity ($)", row=2, col=1)
    fig.update_yaxes(title_text="HTF Trend", row=3, col=1, tickvals=[-1, 0, 1], ticktext=['Bearish', 'Neutral', 'Bullish'])

    return fig


# =============================================================================
# MULTI-STRATEGY COMPARISON
# =============================================================================
def run_strategy_comparison(df: pd.DataFrame, symbol: str, config: TradingConfig):
    """Test multiple strategy configurations and find the best one"""
    print("\n" + "="*60)
    print("ADVANCED MULTI-STRATEGY COMPARISON")
    print("="*60)

    close_col = f'Close_{symbol}'
    high_col = f'High_{symbol}'
    low_col = f'Low_{symbol}'

    high = df[high_col].values
    low = df[low_col].values
    close = df[close_col].values

    # Calculate RSI once
    rsi = calculate_rsi(close, 14)

    # Calculate buy & hold return for comparison
    buy_hold_return = (close[-1] - close[0]) / close[0]
    print(f"\nBuy & Hold Return: {buy_hold_return:.2%}")
    print("Target: Beat this!\n")

    # Extended strategies - focus on catching big moves
    strategies = [
        # Basic strategies
        {"name": "Long Only (Basic)", "use_htf": False, "long_only": True, "filter_mode": "exit_only",
         "use_trailing": False, "trailing_pct": 0, "use_rsi": False},

        # With Trailing Stop - various levels
        {"name": "Long + Trailing 5%", "use_htf": False, "long_only": True, "filter_mode": "exit_only",
         "use_trailing": True, "trailing_pct": 0.05, "use_rsi": False},
        {"name": "Long + Trailing 12%", "use_htf": False, "long_only": True, "filter_mode": "exit_only",
         "use_trailing": True, "trailing_pct": 0.12, "use_rsi": False},
        {"name": "Long + Trailing 15%", "use_htf": False, "long_only": True, "filter_mode": "exit_only",
         "use_trailing": True, "trailing_pct": 0.15, "use_rsi": False},

        # With RSI Filter - buy oversold
        {"name": "Long + RSI Filter", "use_htf": False, "long_only": True, "filter_mode": "exit_only",
         "use_trailing": False, "trailing_pct": 0, "use_rsi": True},

        # Combined: RSI + wider Trailing
        {"name": "Long + RSI + Trail 12%", "use_htf": False, "long_only": True, "filter_mode": "exit_only",
         "use_trailing": True, "trailing_pct": 0.12, "use_rsi": True},

        # HTF strategies
        {"name": "HTF Long Only", "use_htf": True, "long_only": True, "filter_mode": "trend_following",
         "use_trailing": False, "trailing_pct": 0, "use_rsi": False},
        {"name": "HTF + Trail 15%", "use_htf": True, "long_only": True, "filter_mode": "trend_following",
         "use_trailing": True, "trailing_pct": 0.15, "use_rsi": False},
    ]

    results = []
    system = OptimizedTradingSystem(config)

    # Extended parameter search - very fine tuning
    period_range = range(5, 35, 1)  # Finer steps
    multiplier_range = np.arange(1.5, 8.0, 0.25)  # Finer multiplier steps

    for strat in strategies:
        print(f"Testing: {strat['name']}...")

        best_return = -np.inf
        best_config = None

        for period in period_range:
            for multiplier in multiplier_range:
                try:
                    supertrend, direction, _ = calculate_supertrend_vectorized(high, low, close, period, multiplier)

                    htf_direction = None
                    if strat['use_htf']:
                        htf_direction = get_htf_trend(df, symbol, period, multiplier).values

                    buy_signals, sell_signals = generate_signals_vectorized(
                        close, supertrend, direction, htf_direction, strat['use_htf'], strat['filter_mode']
                    )

                    # Use RSI if enabled
                    rsi_arr = rsi if strat['use_rsi'] else None

                    long_trades, short_trades = system.generate_trades(
                        df, buy_signals, sell_signals,
                        long_only=strat['long_only'],
                        use_trailing_stop=strat['use_trailing'],
                        trailing_stop_pct=strat['trailing_pct'],
                        rsi=rsi_arr
                    )

                    if len(long_trades) + len(short_trades) < 2:
                        continue

                    _, _, combined_eq, _ = system.calculate_equity_curve_vectorized(
                        df, long_trades, short_trades
                    )

                    total_return = (combined_eq[-1] - config.initial_capital) / config.initial_capital

                    if total_return > best_return:
                        best_return = total_return
                        best_config = {
                            'period': period,
                            'multiplier': multiplier,
                            'return': total_return,
                            'trades': len(long_trades) + len(short_trades),
                            'vs_buyhold': total_return - buy_hold_return
                        }

                except Exception:
                    pass

        if best_config:
            results.append({
                'strategy': strat['name'],
                'use_htf': strat['use_htf'],
                'long_only': strat['long_only'],
                'filter_mode': strat['filter_mode'],
                'use_trailing': strat['use_trailing'],
                'trailing_pct': strat['trailing_pct'],
                'use_rsi': strat['use_rsi'],
                **best_config
            })
            beat_marker = "✓ BEATS" if best_config['return'] > buy_hold_return else "✗"
            print(f"  Best: P={best_config['period']}, M={best_config['multiplier']:.1f}, Ret={best_config['return']:.2%} {beat_marker}")

    # Sort by return
    results.sort(key=lambda x: x['return'], reverse=True)

    print("\n" + "="*60)
    print("STRATEGY COMPARISON RESULTS (sorted by return)")
    print("="*60)
    print(f"{'Rank':<5} {'Strategy':<30} {'Return':>10} {'vs B&H':>10} {'Trades':>7}")
    print("-"*65)

    for i, r in enumerate(results, 1):
        vs_bh = r['return'] - buy_hold_return
        marker = "**" if r['return'] > buy_hold_return else ""
        print(f"{i:<5} {r['strategy']:<30} {r['return']:>9.2%} {vs_bh:>+9.2%} {r['trades']:>7} {marker}")

    # Return best that beats buy & hold, or overall best
    beating_strategies = [r for r in results if r['return'] > buy_hold_return]
    if beating_strategies:
        print(f"\n>>> {len(beating_strategies)} strategies beat Buy & Hold!")
        return beating_strategies[0]
    else:
        print(f"\n>>> No strategy beats Buy & Hold yet. Using best available.")
        return results[0] if results else None


# =============================================================================
# MAIN FUNCTION
# =============================================================================
def main():
    print("="*60)
    print("SUPERTREND TRADING SYSTEM v5.0 - OPTIMIZED WITH HTF FILTER")
    print("="*60)

    # Configuration - 1 year backtest
    config = TradingConfig(
        symbol="MSFT",
        initial_capital=10000.0,
        days_back=365,  # 1 year
        use_htf_filter=True
    )

    # Strategy settings - these will be overridden by the best strategy found
    LONG_ONLY = True  # Only long trades (better for bullish markets like MSFT)
    FILTER_MODE = "trend_following"  # trend_following, confirmation, exit_only
    OPTIMIZE_FOR = "return"  # return, sharpe, profit_factor, win_rate, risk_adjusted
    COMPARE_STRATEGIES = True  # Run multi-strategy comparison first

    print(f"\nSymbol: {config.symbol}")
    print(f"Initial Capital: ${config.initial_capital:,.2f}")
    print(f"Backtest Period: {config.days_back} days")

    # Download Data from TWS
    print(f"\nDownloading {config.symbol} data from TWS...")

    try:
        stock_data = download_from_tws(config.symbol, config.days_back)
    except Exception as e:
        print(f"Error downloading data: {e}")
        return

    if stock_data.empty:
        print("Error: No data downloaded")
        return

    print(f"Data loaded: {len(stock_data)} trading days")
    print(f"Date range: {stock_data.index[0].strftime('%Y-%m-%d')} to {stock_data.index[-1].strftime('%Y-%m-%d')}")

    # Column names
    close_col = f'Close_{config.symbol}'
    high_col = f'High_{config.symbol}'
    low_col = f'Low_{config.symbol}'

    high = stock_data[high_col].values
    low = stock_data[low_col].values
    close = stock_data[close_col].values

    # Additional strategy settings
    USE_TRAILING_STOP = False
    TRAILING_STOP_PCT = 0.08
    USE_RSI_FILTER = False

    # Run multi-strategy comparison first
    if COMPARE_STRATEGIES:
        best_strat = run_strategy_comparison(stock_data, config.symbol, config)
        if best_strat:
            LONG_ONLY = best_strat['long_only']
            FILTER_MODE = best_strat['filter_mode']
            config.use_htf_filter = best_strat['use_htf']
            config.st_period = best_strat['period']
            config.st_multiplier = best_strat['multiplier']
            config.htf_period = best_strat['period']
            config.htf_multiplier = best_strat['multiplier']
            USE_TRAILING_STOP = best_strat.get('use_trailing', False)
            TRAILING_STOP_PCT = best_strat.get('trailing_pct', 0.08)
            USE_RSI_FILTER = best_strat.get('use_rsi', False)
            print(f"\n>>> Using best strategy: {best_strat['strategy']}")
    else:
        # Parameter Optimization for single strategy
        print("\n" + "-"*60)
        best_period, best_multiplier, optimization_results = optimize_parameters(
            stock_data, config.symbol,
            period_range=range(7, 21, 2),
            multiplier_range=np.arange(2.0, 5.0, 0.5),
            use_htf=config.use_htf_filter,
            long_only=LONG_ONLY,
            filter_mode=FILTER_MODE,
            optimize_for=OPTIMIZE_FOR
        )
        config.st_period = best_period
        config.st_multiplier = best_multiplier
        config.htf_period = best_period
        config.htf_multiplier = best_multiplier

    print("\n" + "-"*60)
    print("RUNNING BACKTEST WITH OPTIMIZED PARAMETERS")
    print("-"*60)
    print(f"HTF Filter: {'Enabled' if config.use_htf_filter else 'Disabled'}")
    print(f"Mode: {'Long Only' if LONG_ONLY else 'Long & Short'}")
    print(f"Filter Mode: {FILTER_MODE}")
    print(f"Trailing Stop: {'Enabled (' + str(int(TRAILING_STOP_PCT*100)) + '%)' if USE_TRAILING_STOP else 'Disabled'}")
    print(f"RSI Filter: {'Enabled' if USE_RSI_FILTER else 'Disabled'}")
    print(f"Period: {config.st_period} | Multiplier: {config.st_multiplier}")

    # Calculate Supertrend with best parameters
    supertrend, direction, atr = calculate_supertrend_vectorized(
        high, low, close, config.st_period, config.st_multiplier
    )

    # Calculate HTF Trend
    htf_direction = get_htf_trend(
        stock_data, config.symbol, config.htf_period, config.htf_multiplier
    ).values

    # Calculate RSI if needed
    rsi = calculate_rsi(close, 14) if USE_RSI_FILTER else None

    # Generate Signals
    buy_signals, sell_signals = generate_signals_vectorized(
        close, supertrend, direction, htf_direction, config.use_htf_filter, FILTER_MODE
    )

    # Initialize Trading System
    system = OptimizedTradingSystem(config)

    # Generate Trades
    long_trades, short_trades = system.generate_trades(
        stock_data, buy_signals, sell_signals,
        long_only=LONG_ONLY,
        use_trailing_stop=USE_TRAILING_STOP,
        trailing_stop_pct=TRAILING_STOP_PCT,
        rsi=rsi
    )

    # Calculate Equity Curves
    long_equity, short_equity, combined_equity, buy_hold_equity = system.calculate_equity_curve_vectorized(
        stock_data, long_trades, short_trades
    )

    # Print Trade Details
    print(f"\n{'='*60}")
    print("TRADE DETAILS")
    print("="*60)

    if long_trades:
        print(f"\nLONG TRADES ({len(long_trades)} total):")
        print("-"*40)
        for i, trade in enumerate(long_trades, 1):
            print(f"  {i}. Entry: {trade['entry_date'].strftime('%Y-%m-%d')} @ ${trade['entry_price']:.2f}")
            print(f"     Exit:  {trade['exit_date'].strftime('%Y-%m-%d')} @ ${trade['exit_price']:.2f}")
            print(f"     P/L:   {trade['profit_loss']:.2%}")
    else:
        print("\nNo long trades executed")

    if short_trades:
        print(f"\nSHORT TRADES ({len(short_trades)} total):")
        print("-"*40)
        for i, trade in enumerate(short_trades, 1):
            print(f"  {i}. Entry: {trade['entry_date'].strftime('%Y-%m-%d')} @ ${trade['entry_price']:.2f}")
            print(f"     Exit:  {trade['exit_date'].strftime('%Y-%m-%d')} @ ${trade['exit_price']:.2f}")
            print(f"     P/L:   {trade['profit_loss']:.2%}")
    else:
        print("\nNo short trades executed")

    # Calculate Statistics
    print(f"\n{'='*60}")
    print("PERFORMANCE STATISTICS")
    print("="*60)

    long_stats = system.calculate_statistics(long_trades, long_equity)
    short_stats = system.calculate_statistics(short_trades, short_equity)
    all_trades = long_trades + short_trades
    combined_stats = system.calculate_statistics(all_trades, combined_equity)

    print("\n--- LONG TRADES ---")
    for key, value in long_stats.items():
        print(f"  {key}: {value}")

    print("\n--- SHORT TRADES ---")
    for key, value in short_stats.items():
        print(f"  {key}: {value}")

    print("\n--- COMBINED STRATEGY ---")
    for key, value in combined_stats.items():
        print(f"  {key}: {value}")

    # Buy & Hold comparison
    buy_hold_return = (buy_hold_equity[-1] - config.initial_capital) / config.initial_capital
    strategy_return = (combined_equity[-1] - config.initial_capital) / config.initial_capital

    print(f"\n{'='*60}")
    print("STRATEGY vs BUY & HOLD")
    print("="*60)
    print(f"  Strategy Return:   {strategy_return:.2%}")
    print(f"  Buy & Hold Return: {buy_hold_return:.2%}")
    print(f"  Outperformance:    {strategy_return - buy_hold_return:.2%}")
    print(f"  Final Equity:      ${combined_equity[-1]:,.2f}")

    # Create Visualization
    print(f"\n{'='*60}")
    print("GENERATING VISUALIZATION...")
    print("="*60)

    fig = create_visualization(
        stock_data, config.symbol, config,
        supertrend, direction, htf_direction,
        buy_signals, sell_signals,
        long_trades, short_trades,
        long_equity, short_equity, combined_equity, buy_hold_equity
    )

    # Save to HTML
    output_file = f"supertrend_{config.symbol}_results.html"
    fig.write_html(output_file)
    print(f"\nResults saved to: {output_file}")

    # Try to show plot (may fail in headless environments)
    try:
        import plotly.io as pio
        pio.renderers.default = 'browser'
        fig.show()
    except Exception as e:
        print(f"Could not open browser (headless environment). View the HTML file instead.")

    print("\n" + "="*60)
    print("BACKTEST COMPLETE")
    print("="*60)

    return {
        'config': config,
        'long_trades': long_trades,
        'short_trades': short_trades,
        'combined_equity': combined_equity,
        'buy_hold_equity': buy_hold_equity
    }


# =============================================================================
# MULTI-TICKER ANALYSIS - TEST ALL STRATEGIES
# =============================================================================

def test_strategy_return(stock_data: pd.DataFrame, buy_signals: np.ndarray, sell_signals: np.ndarray,
                         config: TradingConfig, system) -> float:
    """Helper: calculate return for a given signal set"""
    try:
        long_trades, _ = system.generate_trades(stock_data, buy_signals, sell_signals, long_only=True)
        if len(long_trades) < 2:
            return -np.inf
        _, _, combined_eq, _ = system.calculate_equity_curve_vectorized(stock_data, long_trades, [])
        return (combined_eq[-1] - config.initial_capital) / config.initial_capital
    except:
        return -np.inf


def test_ticker(symbol: str, days_back: int = 365) -> Dict:
    """
    Test a single ticker with ALL strategies and find the best one.
    Strategies: SUPERTREND, JMA, KAMA, EMA, SMA, BUYHOLD
    """
    config = TradingConfig(
        symbol=symbol,
        initial_capital=10000.0,
        days_back=days_back,
        use_htf_filter=False
    )

    try:
        # Download from TWS
        stock_data = download_from_tws(symbol, days_back)

        if stock_data.empty or len(stock_data) < 100:
            return {'symbol': symbol, 'error': 'No data'}

        close_col = f'Close_{symbol}'
        high_col = f'High_{symbol}'
        low_col = f'Low_{symbol}'

        high = stock_data[high_col].values
        low = stock_data[low_col].values
        close = stock_data[close_col].values
        close_series = stock_data[close_col]

        buy_hold_return = (close[-1] - close[0]) / close[0]

        system = OptimizedTradingSystem(config)

        # Track best result per strategy
        strategy_results = {}

        # =================================================================
        # 1. SUPERTREND Strategy
        # =================================================================
        best_st_return = -np.inf
        best_st_params = None
        for period in [10, 14, 20]:
            for mult in [2.0, 3.0, 4.0]:
                try:
                    supertrend, direction, _ = calculate_supertrend_vectorized(high, low, close, period, mult)
                    buy_signals, sell_signals = generate_signals_vectorized(close, supertrend, direction, None, False, "exit_only")
                    ret = test_strategy_return(stock_data, buy_signals, sell_signals, config, system)
                    if ret > best_st_return:
                        best_st_return = ret
                        best_st_params = {'period': period, 'multiplier': mult}
                except:
                    pass
        if best_st_return > -np.inf:
            strategy_results['SUPERTREND'] = {'return': best_st_return, 'params': best_st_params}

        # =================================================================
        # 2. JMA Strategy (JMA crossover)
        # =================================================================
        best_jma_return = -np.inf
        best_jma_params = None
        for fast in [7, 10, 14]:
            for slow in [21, 30, 50]:
                if fast >= slow:
                    continue
                try:
                    jma_fast = calculate_jma(close_series, fast).values
                    jma_slow = calculate_jma(close_series, slow).values
                    buy_signals, sell_signals = get_ma_crossover_signals(close, jma_fast, jma_slow)
                    ret = test_strategy_return(stock_data, buy_signals, sell_signals, config, system)
                    if ret > best_jma_return:
                        best_jma_return = ret
                        best_jma_params = {'fast': fast, 'slow': slow}
                except:
                    pass
        if best_jma_return > -np.inf:
            strategy_results['JMA'] = {'return': best_jma_return, 'params': best_jma_params}

        # =================================================================
        # 3. KAMA Strategy (KAMA crossover with SMA signal)
        # =================================================================
        best_kama_return = -np.inf
        best_kama_params = None
        for period in [10, 14, 20]:
            for signal in [10, 14, 21]:
                try:
                    kama = calculate_kama(close_series, period).values
                    signal_line = calculate_sma(close_series, signal).values
                    buy_signals, sell_signals = get_ma_crossover_signals(close, kama, signal_line)
                    ret = test_strategy_return(stock_data, buy_signals, sell_signals, config, system)
                    if ret > best_kama_return:
                        best_kama_return = ret
                        best_kama_params = {'period': period, 'signal': signal}
                except:
                    pass
        if best_kama_return > -np.inf:
            strategy_results['KAMA'] = {'return': best_kama_return, 'params': best_kama_params}

        # =================================================================
        # 4. EMA Strategy (EMA crossover)
        # =================================================================
        best_ema_return = -np.inf
        best_ema_params = None
        for fast in [8, 12, 20]:
            for slow in [21, 26, 50]:
                if fast >= slow:
                    continue
                try:
                    ema_fast = calculate_ema(close_series, fast).values
                    ema_slow = calculate_ema(close_series, slow).values
                    buy_signals, sell_signals = get_ma_crossover_signals(close, ema_fast, ema_slow)
                    ret = test_strategy_return(stock_data, buy_signals, sell_signals, config, system)
                    if ret > best_ema_return:
                        best_ema_return = ret
                        best_ema_params = {'fast': fast, 'slow': slow}
                except:
                    pass
        if best_ema_return > -np.inf:
            strategy_results['EMA'] = {'return': best_ema_return, 'params': best_ema_params}

        # =================================================================
        # 5. SMA Strategy (SMA crossover)
        # =================================================================
        best_sma_return = -np.inf
        best_sma_params = None
        for fast in [10, 20, 30]:
            for slow in [50, 100, 200]:
                if fast >= slow:
                    continue
                try:
                    sma_fast = calculate_sma(close_series, fast).values
                    sma_slow = calculate_sma(close_series, slow).values
                    buy_signals, sell_signals = get_ma_crossover_signals(close, sma_fast, sma_slow)
                    ret = test_strategy_return(stock_data, buy_signals, sell_signals, config, system)
                    if ret > best_sma_return:
                        best_sma_return = ret
                        best_sma_params = {'fast': fast, 'slow': slow}
                except:
                    pass
        if best_sma_return > -np.inf:
            strategy_results['SMA'] = {'return': best_sma_return, 'params': best_sma_params}

        # =================================================================
        # Find best ACTIVE strategy (NEVER assign BUYHOLD - always real strategy)
        # =================================================================
        best_strategy = None
        best_return = -np.inf
        best_params = {}

        # Find best among actual trading strategies
        for strat_name, strat_data in strategy_results.items():
            if strat_data['return'] > best_return:
                best_return = strat_data['return']
                best_strategy = strat_name
                best_params = strat_data['params']

        # If no strategy worked, default to SUPERTREND
        if best_strategy is None:
            best_strategy = 'SUPERTREND'
            best_return = 0
            best_params = {'period': 10, 'multiplier': 3.0}

        # Check if best strategy beats B&H
        beats_bh = best_return > buy_hold_return

        # Add BUYHOLD to strategy_results for reference only
        strategy_results['BUYHOLD'] = {'return': buy_hold_return, 'params': {}}

        return {
            'symbol': symbol,
            'buy_hold': buy_hold_return,
            'data_days': len(stock_data),
            # Best ACTIVE strategy - ALWAYS a real strategy, NEVER BUYHOLD
            'assigned_strategy': best_strategy,
            'assigned_params': best_params,
            'assigned_return': best_return,
            # Comparison with B&H
            'beats_bh': beats_bh,
            'outperformance': best_return - buy_hold_return,
            # ALL strategies with BEST PARAMS
            'all_strategies': strategy_results
        }
    except Exception as e:
        return {'symbol': symbol, 'error': str(e)}


def run_multi_ticker_analysis():
    """Test strategy on DOW 30 and NASDAQ 100"""
    print("="*80)
    print("MULTI-TICKER ANALYSIS - DOW 30 & NASDAQ 100")
    print("="*80)
    print(f"\nDOW 30: {len(DOW_30)} symbols")
    print(f"NASDAQ 100: {len(NASDAQ_100)} symbols")

    all_results = []

    # Test Dow Jones 30
    print("\n" + "-"*80)
    print(f"TESTING DOW JONES 30 ({len(DOW_30)} symbols)")
    print("-"*80)

    for i, symbol in enumerate(DOW_30, 1):
        print(f"[{i}/{len(DOW_30)}] Testing {symbol}...", end=" ")
        result = test_ticker(symbol, days_back=365)
        result['index'] = 'DOW30'
        all_results.append(result)

        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            marker = "✓ BEATS" if result['beats_bh'] else "✗"
            strat_ret = result.get('strategy_return', 0)
            print(f"B&H: {result['buy_hold']:.1%} | Strat: {strat_ret:.1%} | {result['assigned_strategy']} {marker}")

    # Test NASDAQ 100
    print("\n" + "-"*80)
    print(f"TESTING NASDAQ 100 ({len(NASDAQ_100)} symbols)")
    print("-"*80)

    for i, symbol in enumerate(NASDAQ_100, 1):
        # Skip if already tested in DOW 30
        if any(r['symbol'] == symbol for r in all_results):
            print(f"[{i}/{len(NASDAQ_100)}] {symbol}... (already tested in DOW 30)")
            continue

        print(f"[{i}/{len(NASDAQ_100)}] Testing {symbol}...", end=" ")
        result = test_ticker(symbol, days_back=365)
        result['index'] = 'NASDAQ100'
        all_results.append(result)

        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            marker = "✓ BEATS" if result['beats_bh'] else "✗"
            strat_ret = result.get('strategy_return', 0)
            print(f"B&H: {result['buy_hold']:.1%} | Strat: {strat_ret:.1%} | {result['assigned_strategy']} {marker}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY - ALL STRATEGIES")
    print("="*80)

    valid_results = [r for r in all_results if 'error' not in r]
    beating_bh = [r for r in valid_results if r['beats_bh']]

    # Count assignments by strategy
    strategy_counts = {}
    for r in valid_results:
        strat = r.get('assigned_strategy', 'UNKNOWN')
        strategy_counts[strat] = strategy_counts.get(strat, 0) + 1

    print(f"\nTotal tickers tested: {len(valid_results)}")
    print(f"Strategies that BEAT Buy & Hold: {len(beating_bh)} ({100*len(beating_bh)/len(valid_results):.1f}%)")

    print(f"\n--- STRATEGY ASSIGNMENTS ---")
    for strat in ['SUPERTREND', 'JMA', 'KAMA', 'EMA', 'SMA', 'BUYHOLD']:
        count = strategy_counts.get(strat, 0)
        pct = 100 * count / len(valid_results) if valid_results else 0
        print(f"  {strat:<12}: {count:3d} symbols ({pct:4.1f}%)")

    if beating_bh:
        print("\n--- TOP PERFORMERS (Beat Buy & Hold) ---")
        beating_bh.sort(key=lambda x: x['outperformance'] or 0, reverse=True)
        for r in beating_bh[:25]:  # Top 25
            strat_ret = r.get('strategy_return', 0)
            strat_name = r.get('assigned_strategy', '?')
            print(f"  {r['symbol']:<6} [{strat_name:<10}] Return: {strat_ret:+6.1%} vs B&H: {r['buy_hold']:+6.1%} (Out: {r['outperformance']:+.1%})")
        if len(beating_bh) > 25:
            print(f"  ... and {len(beating_bh) - 25} more")

    # List all assignments by strategy
    print("\n--- ASSIGNMENTS BY STRATEGY ---")
    for strat in ['SUPERTREND', 'JMA', 'KAMA', 'EMA', 'SMA']:
        assigned = [r for r in valid_results if r.get('assigned_strategy') == strat]
        if assigned:
            symbols = [r['symbol'] for r in assigned]
            print(f"\n{strat} ({len(assigned)}): {', '.join(symbols)}")

    buyhold_assigned = [r for r in valid_results if r.get('assigned_strategy') == 'BUYHOLD']
    if buyhold_assigned:
        symbols = [r['symbol'] for r in buyhold_assigned]
        print(f"\nBUYHOLD ({len(buyhold_assigned)}): {', '.join(symbols)}")

    # Average performance
    avg_bh = np.mean([r['buy_hold'] for r in valid_results])
    avg_strat = np.mean([r.get('assigned_return', 0) for r in valid_results])
    outperfs = [r['outperformance'] for r in valid_results if r.get('outperformance')]
    avg_outperf = np.mean(outperfs) if outperfs else 0

    print(f"\n--- AVERAGES ---")
    print(f"  Avg Buy & Hold Return:     {avg_bh:.1%}")
    print(f"  Avg Best Strategy Return:  {avg_strat:.1%}")
    print(f"  Avg Outperformance:        {avg_outperf:+.1%}")

    return all_results


# =============================================================================
# ENHANCED ANALYSIS - MARKET CHARACTERISTICS
# =============================================================================
def analyze_stock_characteristics(symbol: str, days_back: int = 365) -> Dict:
    """Analyze stock characteristics to determine strategy suitability"""
    try:
        # Download from TWS
        data = download_from_tws(symbol, days_back)

        if data.empty or len(data) < 100:
            return {'symbol': symbol, 'error': 'No data'}

        close_col = f'Close_{symbol}'
        high_col = f'High_{symbol}'
        low_col = f'Low_{symbol}'

        close = data[close_col].values
        high = data[high_col].values
        low = data[low_col].values

        # Calculate metrics
        returns = np.diff(close) / close[:-1]

        # Volatility (annualized)
        volatility = np.std(returns) * np.sqrt(252)

        # Total return
        total_return = (close[-1] - close[0]) / close[0]

        # Max Drawdown
        running_max = np.maximum.accumulate(close)
        drawdowns = (close - running_max) / running_max
        max_drawdown = abs(np.min(drawdowns))

        # Trend Strength (linear regression R²)
        x = np.arange(len(close))
        z = np.polyfit(x, close, 1)
        p = np.poly1d(z)
        ss_res = np.sum((close - p(x)) ** 2)
        ss_tot = np.sum((close - np.mean(close)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Trend direction from regression
        trend_direction = "bullish" if z[0] > 0 else "bearish"

        # Average True Range (as % of price)
        atr_pct = calculate_atr(high, low, close, 14)[-100:].mean() / close[-1]

        # Classify stock
        if volatility > 0.40:
            vol_class = "HIGH"
        elif volatility > 0.25:
            vol_class = "MEDIUM"
        else:
            vol_class = "LOW"

        if r_squared > 0.7:
            trend_class = "STRONG"
        elif r_squared > 0.4:
            trend_class = "MODERATE"
        else:
            trend_class = "WEAK"

        # Strategy recommendation
        if trend_class == "STRONG" and trend_direction == "bullish" and vol_class == "LOW":
            recommendation = "BUY_AND_HOLD"
            reason = "Strong uptrend, low volatility - B&H likely optimal"
        elif vol_class in ["HIGH", "MEDIUM"] or trend_class == "WEAK":
            recommendation = "SUPERTREND"
            reason = "High volatility or weak trend - Supertrend can add value"
        elif max_drawdown > 0.30:
            recommendation = "SUPERTREND"
            reason = "High drawdown risk - Supertrend provides protection"
        else:
            recommendation = "NEUTRAL"
            reason = "Mixed characteristics - test both approaches"

        return {
            'symbol': symbol,
            'volatility': volatility,
            'vol_class': vol_class,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'r_squared': r_squared,
            'trend_class': trend_class,
            'trend_direction': trend_direction,
            'atr_pct': atr_pct,
            'recommendation': recommendation,
            'reason': reason,
            'data_days': len(data)
        }
    except Exception as e:
        return {'symbol': symbol, 'error': str(e)}


def run_enhanced_analysis():
    """Run enhanced analysis with market classification"""
    print("="*80)
    print("ENHANCED MULTI-TICKER ANALYSIS WITH MARKET CLASSIFICATION")
    print("="*80)

    # Test sample of stocks
    test_stocks = [
        # High performers from previous test
        'NFLX', 'META', 'TSLA', 'AMD', 'ADBE', 'PYPL', 'BA', 'DIS',
        # Low performers from previous test
        'NVDA', 'AVGO', 'GS', 'CAT', 'AAPL', 'MSFT', 'JPM',
        # Additional variety
        'SPY', 'QQQ', 'ARKK', 'XLF', 'XLE'
    ]

    results = []
    strategy_results = []

    print("\n--- ANALYZING STOCK CHARACTERISTICS ---\n")

    for symbol in test_stocks:
        print(f"Analyzing {symbol}...", end=" ")
        char = analyze_stock_characteristics(symbol, 365)

        if 'error' in char:
            print(f"Error: {char['error']}")
            continue

        results.append(char)
        print(f"Vol: {char['vol_class']}, Trend: {char['trend_class']} ({char['trend_direction']}), Rec: {char['recommendation']}")

        # Also run strategy test
        strat_result = test_ticker(symbol, 365)
        if 'error' not in strat_result:
            strat_result.update(char)
            strategy_results.append(strat_result)

    # Analyze correlations
    print("\n" + "="*80)
    print("ANALYSIS RESULTS")
    print("="*80)

    # Group by recommendation
    supertrend_recs = [r for r in strategy_results if r['recommendation'] == 'SUPERTREND']
    bh_recs = [r for r in strategy_results if r['recommendation'] == 'BUY_AND_HOLD']
    neutral_recs = [r for r in strategy_results if r['recommendation'] == 'NEUTRAL']

    print("\n--- SUPERTREND RECOMMENDED STOCKS ---")
    if supertrend_recs:
        st_wins = [r for r in supertrend_recs if r.get('beats_bh', False)]
        print(f"Total: {len(supertrend_recs)}, Strategy wins: {len(st_wins)} ({100*len(st_wins)/len(supertrend_recs):.0f}%)")
        for r in sorted(supertrend_recs, key=lambda x: x.get('outperformance', 0) or 0, reverse=True):
            outperf = r.get('outperformance', 0) or 0
            marker = "✓" if r.get('beats_bh', False) else "✗"
            print(f"  {r['symbol']}: Vol={r['vol_class']}, B&H={r['buy_hold']:.1%}, Strat={r['strategy']:.1%}, Out={outperf:+.1%} {marker}")

    print("\n--- BUY & HOLD RECOMMENDED STOCKS ---")
    if bh_recs:
        st_wins = [r for r in bh_recs if r.get('beats_bh', False)]
        print(f"Total: {len(bh_recs)}, Strategy wins: {len(st_wins)} ({100*len(st_wins)/len(bh_recs):.0f}%)")
        for r in sorted(bh_recs, key=lambda x: x.get('outperformance', 0) or 0, reverse=True):
            outperf = r.get('outperformance', 0) or 0
            marker = "✓" if r.get('beats_bh', False) else "✗"
            print(f"  {r['symbol']}: Vol={r['vol_class']}, B&H={r['buy_hold']:.1%}, Strat={r['strategy']:.1%}, Out={outperf:+.1%} {marker}")

    print("\n--- NEUTRAL STOCKS ---")
    if neutral_recs:
        st_wins = [r for r in neutral_recs if r.get('beats_bh', False)]
        print(f"Total: {len(neutral_recs)}, Strategy wins: {len(st_wins)} ({100*len(st_wins)/len(neutral_recs):.0f}%)")
        for r in sorted(neutral_recs, key=lambda x: x.get('outperformance', 0) or 0, reverse=True):
            outperf = r.get('outperformance', 0) or 0
            marker = "✓" if r.get('beats_bh', False) else "✗"
            print(f"  {r['symbol']}: Vol={r['vol_class']}, B&H={r['buy_hold']:.1%}, Strat={r['strategy']:.1%}, Out={outperf:+.1%} {marker}")

    # Key insight
    print("\n" + "="*80)
    print("KEY INSIGHT")
    print("="*80)

    # Calculate accuracy of recommendation
    correct_st = len([r for r in supertrend_recs if r.get('beats_bh', False)])
    correct_bh = len([r for r in bh_recs if not r.get('beats_bh', False)])
    total_correct = correct_st + correct_bh
    total_recs = len(supertrend_recs) + len(bh_recs)

    if total_recs > 0:
        accuracy = 100 * total_correct / total_recs
        print(f"\nRecommendation Accuracy: {accuracy:.0f}% ({total_correct}/{total_recs})")
        print(f"  - Supertrend recommendations correct: {correct_st}/{len(supertrend_recs)}")
        print(f"  - Buy & Hold recommendations correct: {correct_bh}/{len(bh_recs)}")

    print("\n--- VOLATILITY vs STRATEGY PERFORMANCE ---")
    high_vol = [r for r in strategy_results if r['vol_class'] == 'HIGH']
    med_vol = [r for r in strategy_results if r['vol_class'] == 'MEDIUM']
    low_vol = [r for r in strategy_results if r['vol_class'] == 'LOW']

    for name, group in [('HIGH', high_vol), ('MEDIUM', med_vol), ('LOW', low_vol)]:
        if group:
            wins = len([r for r in group if r.get('beats_bh', False)])
            avg_out = np.mean([r.get('outperformance', 0) or 0 for r in group])
            print(f"  {name} Volatility: {wins}/{len(group)} beat B&H ({100*wins/len(group):.0f}%), Avg outperf: {avg_out:+.1%}")

    return strategy_results


def screen_for_supertrend_stocks():
    """Screen DOW 30 + NASDAQ 100 stocks to find best candidates for Supertrend strategy"""
    print("="*80)
    print("SUPERTREND STOCK SCREENER - DOW 30 + NASDAQ 100")
    print("="*80)
    print("\nScreening all DOW 30 and NASDAQ 100 stocks...")
    print("(Looking for HIGH volatility stocks where Supertrend beats Buy & Hold)\n")

    # Use all DOW 30 + NASDAQ 100 symbols (deduplicated)
    candidates = ALL_TICKERS
    print(f"Total candidates: {len(candidates)} unique symbols")

    suitable_stocks = []
    unsuitable_stocks = []

    for i, symbol in enumerate(candidates, 1):
        print(f"[{i}/{len(candidates)}] Screening {symbol}...", end=" ")

        # Analyze characteristics
        char = analyze_stock_characteristics(symbol, 365)
        if 'error' in char:
            print(f"Error: {char['error']}")
            continue

        # Test strategy (1 year backtest)
        result = test_ticker(symbol, 365)
        if 'error' in result:
            print(f"Error: {result['error']}")
            continue

        result.update(char)

        # Determine suitability based on volatility and performance
        outperf = result.get('outperformance', 0) or 0
        if char['vol_class'] == 'HIGH' and result.get('beats_bh', False):
            result['suitability'] = 'EXCELLENT'
            suitable_stocks.append(result)
            print(f"✓ EXCELLENT - Vol={char['vol_class']}, {result['assigned_strategy']}, Out={outperf:+.1%}")
        elif char['vol_class'] in ['HIGH', 'MEDIUM'] and result.get('beats_bh', False):
            result['suitability'] = 'GOOD'
            suitable_stocks.append(result)
            print(f"✓ GOOD - Vol={char['vol_class']}, {result['assigned_strategy']}, Out={outperf:+.1%}")
        elif char['vol_class'] == 'LOW':
            result['suitability'] = 'BUY_HOLD'
            unsuitable_stocks.append(result)
            print(f"→ B&H Better - Vol={char['vol_class']}, BUYHOLD")
        else:
            result['suitability'] = 'NEUTRAL'
            unsuitable_stocks.append(result)
            print(f"✗ Neutral - Vol={char['vol_class']}, {result.get('assigned_strategy', 'N/A')}")

    # Results
    print("\n" + "="*80)
    print("SCREENING RESULTS")
    print("="*80)

    print(f"\n--- BEST FOR SUPERTREND ({len(suitable_stocks)} stocks) ---")
    suitable_stocks.sort(key=lambda x: x.get('outperformance', 0) or 0, reverse=True)

    for r in suitable_stocks[:30]:  # Show top 30
        outperf = r.get('outperformance', 0) or 0
        suit = r.get('suitability', 'N/A')
        strat_ret = r.get('strategy_return', 0) or 0
        print(f"  {r['symbol']:<6} [{suit}] Vol={r['vol_class']}, Strat={strat_ret:.1%}, B&H={r['buy_hold']:.1%}, Out={outperf:+.1%}")

    print(f"\n--- RECOMMENDATION: USE BUY & HOLD ({len([u for u in unsuitable_stocks if u['vol_class'] == 'LOW'])} stocks) ---")
    low_vol = [u for u in unsuitable_stocks if u['vol_class'] == 'LOW']
    for r in low_vol:
        print(f"  {r['symbol']:<6} Vol={r['vol_class']}, B&H={r['buy_hold']:.1%} - Stable uptrend, B&H better")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nTotal screened: {len(candidates)}")
    print(f"Suitable for Supertrend: {len(suitable_stocks)} ({100*len(suitable_stocks)/len(candidates):.0f}%)")
    print(f"Better with Buy & Hold: {len(low_vol)} ({100*len(low_vol)/len(candidates):.0f}%)")

    if suitable_stocks:
        top5 = suitable_stocks[:5]
        print(f"\n>>> TOP 5 PICKS FOR SUPERTREND:")
        for i, r in enumerate(top5, 1):
            print(f"  {i}. {r['symbol']} - Outperformance: {r.get('outperformance', 0):+.1%}")

    return suitable_stocks, unsuitable_stocks


def optimize_all_tickers(days_back: int = 365, save_results: bool = True) -> Dict:
    """
    Optimize all DOW 30 + NASDAQ 100 tickers and assign each to best strategy.
    Strategies tested: SUPERTREND, JMA, KAMA, EMA, SMA, BUYHOLD

    Args:
        days_back: Number of days for backtest (default 365 = 1 year)
        save_results: Save results to JSON file

    Returns:
        Dict with assignments for each ticker
    """
    import json

    def sanitize_for_json(obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize_for_json(item) for item in obj]
        elif isinstance(obj, (np.bool_, np.generic)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    print("="*80)
    print("OPTIMIZE ALL TICKERS - MULTI-STRATEGY")
    print("Strategies: SUPERTREND, JMA, KAMA, EMA, SMA, BUYHOLD")
    print("="*80)
    print(f"\nTickers: {len(ALL_TICKERS)} (DOW 30 + NASDAQ 100)")
    print(f"Period: {days_back} days ({days_back/365:.1f} years)")
    print(f"Data Source: TWS")
    print("="*80)

    assignments = {}
    strategy_counts = {'SUPERTREND': 0, 'JMA': 0, 'KAMA': 0, 'EMA': 0, 'SMA': 0}
    beats_bh_count = 0
    error_count = 0

    for i, symbol in enumerate(ALL_TICKERS, 1):
        print(f"\n[{i}/{len(ALL_TICKERS)}] {symbol}...", end=" ")

        result = test_ticker(symbol, days_back)

        if 'error' in result:
            print(f"ERROR: {result['error']}")
            assignments[symbol] = {'strategy': 'ERROR', 'error': result['error']}
            error_count += 1
            continue

        assigned = result['assigned_strategy']
        bh_ret = result['buy_hold']
        strat_ret = result.get('assigned_return', 0)
        outperf = result.get('outperformance', 0)
        beats_bh = result.get('beats_bh', False)

        strategy_counts[assigned] = strategy_counts.get(assigned, 0) + 1
        if beats_bh:
            beats_bh_count += 1

        # Always show the REAL strategy, indicate if it beats B&H
        marker = "✓" if beats_bh else "✗"
        print(f"{assigned:<10} | Strat: {strat_ret:+.1%} vs B&H: {bh_ret:+.1%} | {marker}")

        assignments[symbol] = {
            'strategy': assigned,
            'params': result.get('assigned_params', {}),
            'strategy_return': strat_ret,
            'buy_hold_return': bh_ret,
            'beats_buyhold': beats_bh,
            'outperformance': outperf,
            'all_strategies': result.get('all_strategies', {})
        }

    # Summary
    print("\n" + "="*80)
    print("OPTIMIZATION SUMMARY")
    print("="*80)
    total_valid = len(ALL_TICKERS) - error_count
    print(f"\nTotal tickers: {len(ALL_TICKERS)}")
    print(f"Successfully tested: {total_valid}")
    print(f"Strategies that BEAT B&H: {beats_bh_count} ({100*beats_bh_count/total_valid:.1f}%)")
    print(f"\n--- STRATEGY ASSIGNMENTS (Best strategy per ticker) ---")
    for strat in ['SUPERTREND', 'JMA', 'KAMA', 'EMA', 'SMA']:
        count = strategy_counts.get(strat, 0)
        pct = 100 * count / total_valid if total_valid else 0
        print(f"  {strat:<12}: {count:3d} ({pct:4.1f}%)")
    print(f"  {'ERRORS':<12}: {error_count:3d}")

    # List assignments by strategy
    for strat in ['SUPERTREND', 'JMA', 'KAMA', 'EMA', 'SMA']:
        assigned_syms = [(sym, data) for sym, data in assignments.items() if data.get('strategy') == strat]
        if assigned_syms:
            print(f"\n--- {strat} ASSIGNMENTS ({len(assigned_syms)}) ---")
            for sym, data in sorted(assigned_syms, key=lambda x: x[1].get('strategy_return', 0), reverse=True):
                params = data.get('params', {})
                ret = data.get('strategy_return', 0)
                bh = data.get('buy_hold_return', 0)
                beats = "✓" if data.get('beats_buyhold', False) else "✗"
                param_str = ', '.join(f"{k}={v}" for k, v in params.items()) if params else "default"
                print(f"  {sym:<6}: Strat={ret:+6.1%} B&H={bh:+6.1%} {beats} | {param_str}")

    # Save results
    if save_results:
        output = {
            '_comment': f"Multi-strategy assignments for {len(ALL_TICKERS)} tickers - BEST strategy per ticker",
            '_optimization_date': datetime.now().strftime("%Y-%m-%d %H:%M"),
            '_days_back': days_back,
            '_strategies': ['SUPERTREND', 'JMA', 'KAMA', 'EMA', 'SMA'],
            '_summary': {
                **strategy_counts,
                'beats_buyhold': beats_bh_count,
                'errors': error_count
            },
            'assignments': assignments
        }

        with open('ticker_assignments.json', 'w') as f:
            json.dump(sanitize_for_json(output), f, indent=4)
        print(f"\nResults saved to: ticker_assignments.json")

    # Disconnect from TWS
    disconnect_ib()

    return assignments


# =============================================================================
# HTF COMPARISON & MULTI-PORTFOLIO SIMULATION
# =============================================================================

def test_ticker_htf_comparison(symbol: str, days_back: int = 365, end_offset_days: int = 0) -> Dict:
    """
    Test a single ticker with ALL strategies, comparing WITH and WITHOUT HTF filter.
    Returns best approach for each strategy variant.

    Args:
        symbol: Stock symbol
        days_back: Number of days of data to use for testing
        end_offset_days: Skip the last N days (for walk-forward: optimize on older data)
    """
    config = TradingConfig(
        symbol=symbol,
        initial_capital=10000.0,
        days_back=days_back + end_offset_days,  # Get extra data
        use_htf_filter=False
    )

    try:
        # Download more data than needed
        stock_data = download_from_tws(symbol, days_back + end_offset_days + 50)

        if stock_data.empty or len(stock_data) < 100:
            return {'symbol': symbol, 'error': 'No data'}

        # Trim to optimization period (exclude last end_offset_days)
        if end_offset_days > 0 and len(stock_data) > end_offset_days:
            stock_data = stock_data.iloc[:-end_offset_days]

        # Take only the last days_back days
        if len(stock_data) > days_back:
            stock_data = stock_data.iloc[-days_back:]

        if len(stock_data) < 100:
            return {'symbol': symbol, 'error': 'Not enough data after filtering'}

        close_col = f'Close_{symbol}'
        high_col = f'High_{symbol}'
        low_col = f'Low_{symbol}'

        high = stock_data[high_col].values
        low = stock_data[low_col].values
        close = stock_data[close_col].values
        close_series = stock_data[close_col]

        buy_hold_return = (close[-1] - close[0]) / close[0]

        system = OptimizedTradingSystem(config)

        # Get HTF direction for HTF-enabled tests
        htf_direction = None
        try:
            htf_series = get_htf_trend(stock_data, symbol, 10, 3.0)
            htf_direction = htf_series.values
        except:
            pass

        results = {}

        # =================================================================
        # Test each strategy WITH and WITHOUT HTF filter
        # =================================================================

        # SUPERTREND
        for use_htf in [False, True]:
            htf_label = "_HTF" if use_htf else ""
            best_return = -np.inf
            best_params = None
            for period in [10, 14, 20]:
                for mult in [2.0, 3.0, 4.0]:
                    try:
                        supertrend, direction, _ = calculate_supertrend_vectorized(high, low, close, period, mult)
                        htf_dir = htf_direction if use_htf else None
                        buy_signals, sell_signals = generate_signals_vectorized(
                            close, supertrend, direction, htf_dir, use_htf, "trend_following"
                        )
                        ret = test_strategy_return(stock_data, buy_signals, sell_signals, config, system)
                        if ret > best_return:
                            best_return = ret
                            best_params = {'period': period, 'multiplier': mult}
                    except:
                        pass
            if best_return > -np.inf:
                results[f'SUPERTREND{htf_label}'] = {'return': float(best_return), 'params': best_params, 'htf': use_htf}

        # JMA Crossover
        for use_htf in [False, True]:
            htf_label = "_HTF" if use_htf else ""
            best_return = -np.inf
            best_params = None
            for fast in [7, 10, 14]:
                for slow in [21, 30, 50]:
                    if fast >= slow:
                        continue
                    try:
                        jma_fast = calculate_jma(close_series, fast).values
                        jma_slow = calculate_jma(close_series, slow).values
                        buy_signals, sell_signals = get_ma_crossover_signals(close, jma_fast, jma_slow)
                        # Apply HTF filter manually if enabled
                        if use_htf and htf_direction is not None:
                            buy_signals = buy_signals & (htf_direction == 1)
                        ret = test_strategy_return(stock_data, buy_signals, sell_signals, config, system)
                        if ret > best_return:
                            best_return = ret
                            best_params = {'fast': fast, 'slow': slow}
                    except:
                        pass
            if best_return > -np.inf:
                results[f'JMA{htf_label}'] = {'return': float(best_return), 'params': best_params, 'htf': use_htf}

        # KAMA Crossover
        for use_htf in [False, True]:
            htf_label = "_HTF" if use_htf else ""
            best_return = -np.inf
            best_params = None
            for period in [10, 14, 20]:
                for signal in [10, 14, 21]:
                    try:
                        kama = calculate_kama(close_series, period).values
                        signal_line = calculate_sma(close_series, signal).values
                        buy_signals, sell_signals = get_ma_crossover_signals(close, kama, signal_line)
                        if use_htf and htf_direction is not None:
                            buy_signals = buy_signals & (htf_direction == 1)
                        ret = test_strategy_return(stock_data, buy_signals, sell_signals, config, system)
                        if ret > best_return:
                            best_return = ret
                            best_params = {'period': period, 'signal': signal}
                    except:
                        pass
            if best_return > -np.inf:
                results[f'KAMA{htf_label}'] = {'return': float(best_return), 'params': best_params, 'htf': use_htf}

        # EMA Crossover
        for use_htf in [False, True]:
            htf_label = "_HTF" if use_htf else ""
            best_return = -np.inf
            best_params = None
            for fast in [8, 12, 20]:
                for slow in [21, 26, 50]:
                    if fast >= slow:
                        continue
                    try:
                        ema_fast = calculate_ema(close_series, fast).values
                        ema_slow = calculate_ema(close_series, slow).values
                        buy_signals, sell_signals = get_ma_crossover_signals(close, ema_fast, ema_slow)
                        if use_htf and htf_direction is not None:
                            buy_signals = buy_signals & (htf_direction == 1)
                        ret = test_strategy_return(stock_data, buy_signals, sell_signals, config, system)
                        if ret > best_return:
                            best_return = ret
                            best_params = {'fast': fast, 'slow': slow}
                    except:
                        pass
            if best_return > -np.inf:
                results[f'EMA{htf_label}'] = {'return': float(best_return), 'params': best_params, 'htf': use_htf}

        # SMA Crossover
        for use_htf in [False, True]:
            htf_label = "_HTF" if use_htf else ""
            best_return = -np.inf
            best_params = None
            for fast in [10, 20, 30]:
                for slow in [50, 100, 200]:
                    if fast >= slow:
                        continue
                    try:
                        sma_fast = calculate_sma(close_series, fast).values
                        sma_slow = calculate_sma(close_series, slow).values
                        buy_signals, sell_signals = get_ma_crossover_signals(close, sma_fast, sma_slow)
                        if use_htf and htf_direction is not None:
                            buy_signals = buy_signals & (htf_direction == 1)
                        ret = test_strategy_return(stock_data, buy_signals, sell_signals, config, system)
                        if ret > best_return:
                            best_return = ret
                            best_params = {'fast': fast, 'slow': slow}
                    except:
                        pass
            if best_return > -np.inf:
                results[f'SMA{htf_label}'] = {'return': float(best_return), 'params': best_params, 'htf': use_htf}

        # Add B&H
        results['BUYHOLD'] = {'return': float(buy_hold_return), 'params': {}, 'htf': False}

        # Find overall best
        best_strategy = None
        best_return = -np.inf
        for strat_name, strat_data in results.items():
            if strat_data['return'] > best_return:
                best_return = strat_data['return']
                best_strategy = strat_name

        return {
            'symbol': symbol,
            'best_strategy': best_strategy,
            'best_return': float(best_return),
            'buy_hold': float(buy_hold_return),
            'all_results': results
        }
    except Exception as e:
        return {'symbol': symbol, 'error': str(e)}


def run_htf_comparison_and_categorize(days_back: int = 180, min_pnl: float = 0.30, end_offset_days: int = 0):
    """
    Run comprehensive HTF comparison on all tickers.
    Categorize by strategy with PnL >= min_pnl (default 30%).

    Args:
        days_back: Number of days for optimization period
        min_pnl: Minimum PnL threshold for categorization (default 30%)
        end_offset_days: Skip the last N days (for walk-forward optimization)

    Categories:
    - STRATEGY_HTF: Strategy with HTF filter, PnL >= 30%
    - STRATEGY_NOHTF: Strategy without HTF filter, PnL >= 30%
    - BUYHOLD: Buy & Hold is best with PnL >= 30%
    - UNDERPERFORM: Best PnL < 30%
    """
    import json

    def sanitize_for_json(obj):
        if isinstance(obj, dict):
            return {k: sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize_for_json(item) for item in obj]
        elif isinstance(obj, (np.bool_, np.generic)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    print("="*80)
    print("HTF COMPARISON & CATEGORIZATION (WALK-FORWARD)")
    print(f"Optimization Period: {days_back} days ({days_back/30:.0f} months)")
    if end_offset_days > 0:
        print(f"Data ends: {end_offset_days} days ago (for out-of-sample testing)")
    print(f"Minimum PnL threshold: {min_pnl:.0%}")
    print("="*80)

    # Categories
    categories = {
        'SUPERTREND_HTF': [],
        'SUPERTREND_NOHTF': [],
        'JMA_HTF': [],
        'JMA_NOHTF': [],
        'KAMA_HTF': [],
        'KAMA_NOHTF': [],
        'EMA_HTF': [],
        'EMA_NOHTF': [],
        'SMA_HTF': [],
        'SMA_NOHTF': [],
        'BUYHOLD': [],
        'UNDERPERFORM': []
    }

    all_results = {}
    error_count = 0

    for i, symbol in enumerate(ALL_TICKERS, 1):
        print(f"\n[{i}/{len(ALL_TICKERS)}] {symbol}...", end=" ")

        result = test_ticker_htf_comparison(symbol, days_back, end_offset_days)

        if 'error' in result:
            print(f"ERROR: {result['error']}")
            error_count += 1
            continue

        best = result['best_strategy']
        best_ret = result['best_return']
        bh_ret = result['buy_hold']

        all_results[symbol] = result

        # Categorize based on best strategy and PnL threshold
        if best_ret < min_pnl:
            categories['UNDERPERFORM'].append({
                'symbol': symbol,
                'best_strategy': best,
                'return': best_ret,
                'buy_hold': bh_ret
            })
            marker = "⚠"
            cat = "UNDERPERFORM"
        elif best == 'BUYHOLD':
            categories['BUYHOLD'].append({
                'symbol': symbol,
                'return': bh_ret,
                'params': {}
            })
            marker = "📈"
            cat = "BUYHOLD"
        else:
            # Determine category based on strategy name
            base_strat = best.replace('_HTF', '')
            has_htf = '_HTF' in best
            cat_key = f"{base_strat}_{'HTF' if has_htf else 'NOHTF'}"

            strat_data = result['all_results'].get(best, {})
            categories[cat_key].append({
                'symbol': symbol,
                'return': best_ret,
                'params': strat_data.get('params', {}),
                'buy_hold': bh_ret
            })
            marker = "✓" if has_htf else "○"
            cat = cat_key

        print(f"{best:<15} | {best_ret:+.1%} (B&H: {bh_ret:+.1%}) | {marker} {cat}")

    # Summary
    print("\n" + "="*80)
    print("CATEGORIZATION SUMMARY (PnL >= {:.0%})".format(min_pnl))
    print("="*80)

    total_valid = len(ALL_TICKERS) - error_count
    print(f"\nTotal tested: {total_valid} | Errors: {error_count}")

    print("\n--- CATEGORIES ---")
    for cat, items in categories.items():
        if items:
            avg_ret = np.mean([x['return'] for x in items]) if items else 0
            print(f"\n{cat}: {len(items)} symbols (avg return: {avg_ret:+.1%})")
            for item in sorted(items, key=lambda x: x['return'], reverse=True)[:5]:
                params_str = ', '.join(f"{k}={v}" for k, v in item.get('params', {}).items()) if item.get('params') else ""
                print(f"    {item['symbol']:<6}: {item['return']:+.1%} | {params_str}")
            if len(items) > 5:
                print(f"    ... and {len(items)-5} more")

    # Save categorized results
    output = {
        '_comment': 'HTF Comparison Results - Categorized by Strategy (Walk-Forward)',
        '_date': datetime.now().strftime("%Y-%m-%d %H:%M"),
        '_optimization_days': days_back,
        '_end_offset_days': end_offset_days,
        '_min_pnl': min_pnl,
        '_summary': {cat: len(items) for cat, items in categories.items()},
        'categories': categories,
        'all_results': {sym: res for sym, res in all_results.items() if 'error' not in res}
    }

    with open('htf_categorized_results.json', 'w') as f:
        json.dump(sanitize_for_json(output), f, indent=4)
    print(f"\nResults saved to: htf_categorized_results.json")

    return categories, all_results


def run_multi_portfolio_simulation(categories: Dict = None, days_back: int = 180):
    """
    Run multi-portfolio simulation based on categorized strategies.
    Each portfolio uses symbols from a specific category with their optimal parameters.
    """
    import json

    def sanitize_for_json(obj):
        if isinstance(obj, dict):
            return {k: sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize_for_json(item) for item in obj]
        elif isinstance(obj, (np.bool_, np.generic)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    # Load categories if not provided
    if categories is None:
        try:
            with open('htf_categorized_results.json', 'r') as f:
                data = json.load(f)
                categories = data.get('categories', {})
        except:
            print("ERROR: No categorized results found. Run --htf-compare first.")
            return

    print("\n" + "="*80)
    print("MULTI-PORTFOLIO SIMULATION")
    print(f"Period: {days_back} days ({days_back/30:.0f} months)")
    print("="*80)

    portfolio_results = {}
    initial_capital_per_stock = 10000.0

    for cat_name, symbols_data in categories.items():
        if not symbols_data or cat_name == 'UNDERPERFORM':
            continue

        print(f"\n--- Portfolio: {cat_name} ({len(symbols_data)} symbols) ---")

        portfolio_returns = []
        portfolio_details = []

        for item in symbols_data:
            symbol = item['symbol']
            params = item.get('params', {})
            stored_return = item.get('return', 0)

            # Use stored return from categorization
            portfolio_returns.append(stored_return)
            portfolio_details.append({
                'symbol': symbol,
                'return': stored_return,
                'params': params
            })
            print(f"  {symbol:<6}: {stored_return:+.1%}")

        if portfolio_returns:
            # Portfolio statistics
            avg_return = np.mean(portfolio_returns)
            total_return = np.sum(portfolio_returns) / len(portfolio_returns)  # Equal weighted
            min_return = np.min(portfolio_returns)
            max_return = np.max(portfolio_returns)
            winners = sum(1 for r in portfolio_returns if r > 0)
            win_rate = winners / len(portfolio_returns)

            portfolio_results[cat_name] = {
                'num_symbols': len(symbols_data),
                'avg_return': float(avg_return),
                'total_return': float(total_return),
                'min_return': float(min_return),
                'max_return': float(max_return),
                'win_rate': float(win_rate),
                'symbols': portfolio_details
            }

            print(f"\n  Portfolio Stats:")
            print(f"    Avg Return:  {avg_return:+.1%}")
            print(f"    Min/Max:     {min_return:+.1%} / {max_return:+.1%}")
            print(f"    Win Rate:    {win_rate:.0%} ({winners}/{len(portfolio_returns)})")

    # Overall comparison
    print("\n" + "="*80)
    print("PORTFOLIO COMPARISON")
    print("="*80)
    print(f"\n{'Portfolio':<20} {'Symbols':>8} {'Avg Ret':>10} {'Win Rate':>10} {'Best':>10} {'Worst':>10}")
    print("-"*70)

    sorted_portfolios = sorted(portfolio_results.items(), key=lambda x: x[1]['avg_return'], reverse=True)
    for name, stats in sorted_portfolios:
        print(f"{name:<20} {stats['num_symbols']:>8} {stats['avg_return']:>+9.1%} {stats['win_rate']:>9.0%} {stats['max_return']:>+9.1%} {stats['min_return']:>+9.1%}")

    # Save results
    output = {
        '_comment': 'Multi-Portfolio Simulation Results',
        '_date': datetime.now().strftime("%Y-%m-%d %H:%M"),
        '_days_back': days_back,
        'portfolios': portfolio_results
    }

    with open('multi_portfolio_results.json', 'w') as f:
        json.dump(sanitize_for_json(output), f, indent=4)
    print(f"\nResults saved to: multi_portfolio_results.json")

    return portfolio_results


def run_full_htf_analysis(days_back: int = 180, min_pnl: float = 0.30, end_offset_days: int = 0):
    """
    Run full HTF analysis: comparison, categorization, and portfolio simulation.

    Args:
        days_back: Number of days for optimization period
        min_pnl: Minimum PnL threshold
        end_offset_days: Skip last N days for walk-forward analysis
    """
    print("="*80)
    print("FULL HTF ANALYSIS & MULTI-PORTFOLIO SIMULATION")
    if end_offset_days > 0:
        print(f"WALK-FORWARD: Optimize {days_back} days, ending {end_offset_days} days ago")
    print("="*80)

    # Step 1: Run comparison and categorization
    categories, all_results = run_htf_comparison_and_categorize(days_back, min_pnl, end_offset_days)

    # Step 2: Run multi-portfolio simulation
    portfolio_results = run_multi_portfolio_simulation(categories, days_back)

    # Disconnect
    disconnect_ib()

    return categories, portfolio_results


def run_walk_forward_analysis(optimize_days: int = 270, test_days: int = 90, min_pnl: float = 0.30):
    """
    Walk-Forward Analysis:
    - Optimize strategies on historical data (optimize_days, ending test_days ago)
    - Then run portfolio simulation on the test period (last test_days)

    Args:
        optimize_days: Days for optimization (default: 9 months = 270 days)
        test_days: Days for out-of-sample testing (default: 3 months = 90 days)
        min_pnl: Minimum PnL threshold for categorization
    """
    print("="*80)
    print("WALK-FORWARD ANALYSIS")
    print("="*80)
    print(f"Optimization Period: {optimize_days} days ({optimize_days/30:.0f} months)")
    print(f"Test Period: {test_days} days ({test_days/30:.0f} months)")
    print(f"Min PnL Threshold: {min_pnl:.0%}")
    print("="*80)

    # Step 1: Optimize on historical data (ending test_days ago)
    print("\n>>> STEP 1: OPTIMIZATION (Historical Data)")
    categories, all_results = run_htf_comparison_and_categorize(
        days_back=optimize_days,
        min_pnl=min_pnl,
        end_offset_days=test_days
    )

    print("\n>>> STEP 2: OUT-OF-SAMPLE SIMULATION (Last {} days)".format(test_days))
    print("Run: python portfolio_simulation.py {}".format(test_days))

    # Disconnect
    disconnect_ib()

    return categories, all_results


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--multi":
        run_multi_ticker_analysis()
        disconnect_ib()
    elif len(sys.argv) > 1 and sys.argv[1] == "--enhanced":
        run_enhanced_analysis()
        disconnect_ib()
    elif len(sys.argv) > 1 and sys.argv[1] == "--screen":
        screen_for_supertrend_stocks()
        disconnect_ib()
    elif len(sys.argv) > 1 and sys.argv[1] == "--all":
        # Optimize all tickers
        optimize_all_tickers(days_back=365)
    elif len(sys.argv) > 1 and sys.argv[1] == "--htf-compare":
        # HTF comparison and categorization (6 months, 30% threshold)
        run_htf_comparison_and_categorize(days_back=180, min_pnl=0.30)
        disconnect_ib()
    elif len(sys.argv) > 1 and sys.argv[1] == "--htf-portfolio":
        # Multi-portfolio simulation from saved categories
        run_multi_portfolio_simulation(days_back=180)
        disconnect_ib()
    elif len(sys.argv) > 1 and sys.argv[1] == "--htf-full":
        # Full analysis: compare + categorize + portfolio simulation
        run_full_htf_analysis(days_back=180, min_pnl=0.30)
    elif len(sys.argv) > 1 and sys.argv[1] == "--walk-forward":
        # Walk-forward analysis: 9 months optimization, 3 months test
        run_walk_forward_analysis(optimize_days=270, test_days=90, min_pnl=0.30)
    else:
        main()
        disconnect_ib()
