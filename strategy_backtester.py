#!/usr/bin/env python3
"""
Multi-Strategy Backtester & Symbol Optimizer
=============================================
Tests all symbols with multiple strategies and assigns each to its best strategy.

Strategies: JMA, Supertrend, SMA, KAMA, EMA, Buy&Hold (each with/without HTF JMA filter)

Data Source: Interactive Brokers TWS (via ib_insync)

Optimization:
- Optimizes over last 6 months of data
- Supports daily OR hourly bars
- HTF filter periods: 6, 8, 12 (hours for hourly, days for daily)
- Manual trigger every 4 weeks
- Simulates 6 months after optimization

Position Sizing:
- Amount per trade = Capital / 30 / entry_price (rounded)
- Max 20 positions open at once
- Entry fee deducted at entry
- Exit fee deducted at exit

Usage:
    python strategy_backtester.py --symbols AAPL,MSFT,GOOGL --capital 100000
    python strategy_backtester.py --all --capital 100000 --timeframe hourly
    python strategy_backtester.py --optimize --timeframe daily
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
import argparse
import os
import asyncio

# Fix for Python 3.10+ event loop issue with ib_insync
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

from ib_insync import IB, Stock, util

# Global IB connection
_ib_connection = None

# =============================================================================
# CONSTANTS
# =============================================================================
DEFAULT_CAPITAL = 100000
POSITION_DIVISOR = 30  # Capital / 30 per trade
MAX_POSITIONS = 20
FEE_RATE = 0.001  # 0.1% per trade (entry + exit)
TRAILING_STOP_PCT = 0.12  # 12% trailing stop

# Optimization and simulation periods (in months)
OPTIMIZATION_MONTHS = 6
SIMULATION_MONTHS = 6

# HTF filter periods: 6, 8, 12 (hours for hourly data, days for daily data)
HTF_PERIODS = [6, 8, 12]

# Timeframes
TIMEFRAME_DAILY = 'daily'
TIMEFRAME_HOURLY = 'hourly'

# =============================================================================
# INDICATOR CALCULATIONS
# =============================================================================

def calculate_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 2.0) -> pd.DataFrame:
    """Calculate Supertrend indicator"""
    df = df.copy()

    # ATR calculation
    df['tr'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            abs(df['High'] - df['Close'].shift(1)),
            abs(df['Low'] - df['Close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(window=period).mean()

    # Basic bands
    hl2 = (df['High'] + df['Low']) / 2
    df['basic_ub'] = hl2 + (multiplier * df['atr'])
    df['basic_lb'] = hl2 - (multiplier * df['atr'])

    # Final bands with trailing logic
    df['final_ub'] = df['basic_ub']
    df['final_lb'] = df['basic_lb']

    for i in range(period, len(df)):
        if df['basic_ub'].iloc[i] < df['final_ub'].iloc[i-1] or df['Close'].iloc[i-1] > df['final_ub'].iloc[i-1]:
            df.loc[df.index[i], 'final_ub'] = df['basic_ub'].iloc[i]
        else:
            df.loc[df.index[i], 'final_ub'] = df['final_ub'].iloc[i-1]

        if df['basic_lb'].iloc[i] > df['final_lb'].iloc[i-1] or df['Close'].iloc[i-1] < df['final_lb'].iloc[i-1]:
            df.loc[df.index[i], 'final_lb'] = df['basic_lb'].iloc[i]
        else:
            df.loc[df.index[i], 'final_lb'] = df['final_lb'].iloc[i-1]

    # Supertrend direction
    df['supertrend'] = np.nan
    df['st_direction'] = 1  # 1 = bullish, -1 = bearish

    for i in range(period, len(df)):
        if df['Close'].iloc[i] > df['final_ub'].iloc[i-1]:
            df.loc[df.index[i], 'st_direction'] = 1
        elif df['Close'].iloc[i] < df['final_lb'].iloc[i-1]:
            df.loc[df.index[i], 'st_direction'] = -1
        else:
            df.loc[df.index[i], 'st_direction'] = df['st_direction'].iloc[i-1]

        if df['st_direction'].iloc[i] == 1:
            df.loc[df.index[i], 'supertrend'] = df['final_lb'].iloc[i]
        else:
            df.loc[df.index[i], 'supertrend'] = df['final_ub'].iloc[i]

    return df


def calculate_jma(series: pd.Series, period: int = 7, phase: int = 50) -> pd.Series:
    """Calculate Jurik Moving Average (JMA)"""
    # Simplified JMA approximation using EMA with adaptive smoothing
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


# =============================================================================
# STRATEGY SIGNALS
# =============================================================================

def get_supertrend_signals(df: pd.DataFrame, period: int = 10, multiplier: float = 2.0,
                           htf_filter: bool = False, htf_period: int = 200) -> pd.Series:
    """Generate Supertrend buy/sell signals"""
    df = calculate_supertrend(df, period, multiplier)

    signals = pd.Series(0, index=df.index)

    # Supertrend direction change signals
    for i in range(1, len(df)):
        if df['st_direction'].iloc[i] == 1 and df['st_direction'].iloc[i-1] == -1:
            signals.iloc[i] = 1  # Buy
        elif df['st_direction'].iloc[i] == -1 and df['st_direction'].iloc[i-1] == 1:
            signals.iloc[i] = -1  # Sell

    # HTF filter
    if htf_filter:
        htf_ma = calculate_jma(df['Close'], htf_period)
        # Only allow buys when price > HTF MA
        for i in range(len(df)):
            if signals.iloc[i] == 1 and df['Close'].iloc[i] < htf_ma.iloc[i]:
                signals.iloc[i] = 0

    return signals


def get_jma_signals(df: pd.DataFrame, period: int = 7, phase: int = 50, signal_period: int = 21,
                    htf_filter: bool = False, htf_period: int = 200) -> pd.Series:
    """Generate JMA crossover signals"""
    jma_fast = calculate_jma(df['Close'], period, phase)
    jma_slow = calculate_jma(df['Close'], signal_period, phase)

    signals = pd.Series(0, index=df.index)

    for i in range(1, len(df)):
        if jma_fast.iloc[i] > jma_slow.iloc[i] and jma_fast.iloc[i-1] <= jma_slow.iloc[i-1]:
            signals.iloc[i] = 1  # Buy
        elif jma_fast.iloc[i] < jma_slow.iloc[i] and jma_fast.iloc[i-1] >= jma_slow.iloc[i-1]:
            signals.iloc[i] = -1  # Sell

    if htf_filter:
        htf_ma = calculate_jma(df['Close'], htf_period)
        for i in range(len(df)):
            if signals.iloc[i] == 1 and df['Close'].iloc[i] < htf_ma.iloc[i]:
                signals.iloc[i] = 0

    return signals


def get_kama_signals(df: pd.DataFrame, period: int = 10, fast: int = 2, slow: int = 30,
                     signal_period: int = 10, htf_filter: bool = False, htf_period: int = 200) -> pd.Series:
    """Generate KAMA crossover signals"""
    kama = calculate_kama(df['Close'], period, fast, slow)
    signal_line = calculate_sma(kama, signal_period)

    signals = pd.Series(0, index=df.index)

    for i in range(1, len(df)):
        if pd.notna(kama.iloc[i]) and pd.notna(signal_line.iloc[i]):
            if kama.iloc[i] > signal_line.iloc[i] and kama.iloc[i-1] <= signal_line.iloc[i-1]:
                signals.iloc[i] = 1
            elif kama.iloc[i] < signal_line.iloc[i] and kama.iloc[i-1] >= signal_line.iloc[i-1]:
                signals.iloc[i] = -1

    if htf_filter:
        htf_ma = calculate_jma(df['Close'], htf_period)
        for i in range(len(df)):
            if signals.iloc[i] == 1 and df['Close'].iloc[i] < htf_ma.iloc[i]:
                signals.iloc[i] = 0

    return signals


def get_sma_signals(df: pd.DataFrame, fast: int = 20, slow: int = 50,
                    htf_filter: bool = False, htf_period: int = 200) -> pd.Series:
    """Generate SMA crossover signals"""
    sma_fast = calculate_sma(df['Close'], fast)
    sma_slow = calculate_sma(df['Close'], slow)

    signals = pd.Series(0, index=df.index)

    for i in range(1, len(df)):
        if pd.notna(sma_fast.iloc[i]) and pd.notna(sma_slow.iloc[i]):
            if sma_fast.iloc[i] > sma_slow.iloc[i] and sma_fast.iloc[i-1] <= sma_slow.iloc[i-1]:
                signals.iloc[i] = 1
            elif sma_fast.iloc[i] < sma_slow.iloc[i] and sma_fast.iloc[i-1] >= sma_slow.iloc[i-1]:
                signals.iloc[i] = -1

    if htf_filter:
        htf_ma = calculate_jma(df['Close'], htf_period)
        for i in range(len(df)):
            if signals.iloc[i] == 1 and df['Close'].iloc[i] < htf_ma.iloc[i]:
                signals.iloc[i] = 0

    return signals


def get_ema_signals(df: pd.DataFrame, fast: int = 12, slow: int = 26,
                    htf_filter: bool = False, htf_period: int = 200) -> pd.Series:
    """Generate EMA crossover signals"""
    ema_fast = calculate_ema(df['Close'], fast)
    ema_slow = calculate_ema(df['Close'], slow)

    signals = pd.Series(0, index=df.index)

    for i in range(1, len(df)):
        if ema_fast.iloc[i] > ema_slow.iloc[i] and ema_fast.iloc[i-1] <= ema_slow.iloc[i-1]:
            signals.iloc[i] = 1
        elif ema_fast.iloc[i] < ema_slow.iloc[i] and ema_fast.iloc[i-1] >= ema_slow.iloc[i-1]:
            signals.iloc[i] = -1

    if htf_filter:
        htf_ma = calculate_jma(df['Close'], htf_period)
        for i in range(len(df)):
            if signals.iloc[i] == 1 and df['Close'].iloc[i] < htf_ma.iloc[i]:
                signals.iloc[i] = 0

    return signals


def get_buyhold_signals(df: pd.DataFrame, htf_filter: bool = False, htf_period: int = 200) -> pd.Series:
    """Generate Buy & Hold signals (buy at start, hold forever or use HTF filter)"""
    signals = pd.Series(0, index=df.index)

    if htf_filter:
        htf_ma = calculate_jma(df['Close'], htf_period)
        in_position = False

        for i in range(htf_period, len(df)):
            if not in_position and df['Close'].iloc[i] > htf_ma.iloc[i]:
                signals.iloc[i] = 1
                in_position = True
            elif in_position and df['Close'].iloc[i] < htf_ma.iloc[i]:
                signals.iloc[i] = -1
                in_position = False
    else:
        # Simple buy at start
        signals.iloc[50] = 1  # Buy after warmup period

    return signals


# =============================================================================
# BACKTESTER
# =============================================================================

class Backtester:
    """Backtests a strategy on a single symbol"""

    def __init__(self, initial_capital: float = DEFAULT_CAPITAL, fee_rate: float = FEE_RATE,
                 trailing_stop_pct: float = TRAILING_STOP_PCT):
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.trailing_stop_pct = trailing_stop_pct

    def run(self, df: pd.DataFrame, signals: pd.Series) -> Dict:
        """Run backtest and return results"""
        capital = self.initial_capital
        position = 0
        entry_price = 0
        entry_date = None
        highest_price = 0

        trades = []
        equity_curve = [capital]

        for i in range(len(df)):
            date = df.index[i]
            price = df['Close'].iloc[i]
            signal = signals.iloc[i]

            # Check trailing stop
            if position > 0:
                highest_price = max(highest_price, price)
                stop_price = highest_price * (1 - self.trailing_stop_pct)

                if price < stop_price:
                    # Trailing stop hit
                    exit_value = position * price
                    exit_fee = exit_value * self.fee_rate
                    capital += exit_value - exit_fee

                    pnl = (price - entry_price) * position - (entry_price * position * self.fee_rate) - exit_fee
                    pnl_pct = (price - entry_price) / entry_price * 100

                    trades.append({
                        'symbol': df.attrs.get('symbol', 'N/A'),
                        'entry_date': entry_date,
                        'entry_price': entry_price,
                        'exit_date': date,
                        'exit_price': price,
                        'qty': position,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'exit_reason': 'Trailing Stop'
                    })

                    position = 0
                    entry_price = 0
                    highest_price = 0

            # Process signals
            if signal == 1 and position == 0:  # Buy signal
                # Position sizing: Capital / 30 / price
                trade_capital = capital / POSITION_DIVISOR
                qty = int(trade_capital / price)

                if qty > 0:
                    entry_value = qty * price
                    entry_fee = entry_value * self.fee_rate
                    capital -= entry_value + entry_fee

                    position = qty
                    entry_price = price
                    entry_date = date
                    highest_price = price

            elif signal == -1 and position > 0:  # Sell signal
                exit_value = position * price
                exit_fee = exit_value * self.fee_rate
                capital += exit_value - exit_fee

                pnl = (price - entry_price) * position - (entry_price * position * self.fee_rate) - exit_fee
                pnl_pct = (price - entry_price) / entry_price * 100

                trades.append({
                    'symbol': df.attrs.get('symbol', 'N/A'),
                    'entry_date': entry_date,
                    'entry_price': entry_price,
                    'exit_date': date,
                    'exit_price': price,
                    'qty': position,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'exit_reason': 'Signal'
                })

                position = 0
                entry_price = 0
                highest_price = 0

            # Record equity
            current_equity = capital + (position * price if position > 0 else 0)
            equity_curve.append(current_equity)

        # Close any open position at end
        if position > 0:
            price = df['Close'].iloc[-1]
            exit_value = position * price
            exit_fee = exit_value * self.fee_rate
            capital += exit_value - exit_fee

            pnl = (price - entry_price) * position - (entry_price * position * self.fee_rate) - exit_fee
            pnl_pct = (price - entry_price) / entry_price * 100

            trades.append({
                'symbol': df.attrs.get('symbol', 'N/A'),
                'entry_date': entry_date,
                'entry_price': entry_price,
                'exit_date': df.index[-1],
                'exit_price': price,
                'qty': position,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'exit_reason': 'End of Period'
            })

        # Calculate metrics
        return self._calculate_metrics(trades, equity_curve)

    def _calculate_metrics(self, trades: List[Dict], equity_curve: List[float]) -> Dict:
        """Calculate performance metrics"""
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_pnl': 0.0,
                'avg_winner': 0.0,
                'avg_loser': 0.0,
                'profit_factor': 0.0,
                'max_drawdown': 0.0,
                'max_drawdown_pct': 0.0,
                'sharpe_ratio': 0.0,
                'total_return_pct': 0.0,
                'trades': trades
            }

        pnls = [t['pnl'] for t in trades]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p < 0]

        total_trades = len(trades)
        winning_trades = len(winners)
        losing_trades = len(losers)
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0

        total_pnl = sum(pnls)
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
        avg_winner = sum(winners) / len(winners) if winners else 0
        avg_loser = sum(losers) / len(losers) if losers else 0

        gross_profit = sum(winners) if winners else 0
        gross_loss = abs(sum(losers)) if losers else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0

        # Max drawdown
        peak = equity_curve[0]
        max_dd = 0
        max_dd_pct = 0
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            dd = peak - equity
            dd_pct = dd / peak * 100 if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
                max_dd_pct = dd_pct

        # Sharpe ratio (simplified)
        returns = pd.Series(equity_curve).pct_change().dropna()
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

        total_return_pct = (equity_curve[-1] - equity_curve[0]) / equity_curve[0] * 100

        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'avg_winner': avg_winner,
            'avg_loser': avg_loser,
            'profit_factor': profit_factor,
            'max_drawdown': max_dd,
            'max_drawdown_pct': max_dd_pct,
            'sharpe_ratio': sharpe,
            'total_return_pct': total_return_pct,
            'final_equity': equity_curve[-1],
            'trades': trades
        }


# =============================================================================
# STRATEGY OPTIMIZER
# =============================================================================

# Base strategies with default parameters
STRATEGIES = {
    'SUPERTREND': {'func': get_supertrend_signals, 'params': {'period': 10, 'multiplier': 2.0}},
    'JMA': {'func': get_jma_signals, 'params': {'period': 7, 'phase': 50, 'signal_period': 21}},
    'KAMA': {'func': get_kama_signals, 'params': {'period': 10, 'fast': 2, 'slow': 30, 'signal_period': 10}},
    'SMA': {'func': get_sma_signals, 'params': {'fast': 20, 'slow': 50}},
    'EMA': {'func': get_ema_signals, 'params': {'fast': 12, 'slow': 26}},
    'BUYHOLD': {'func': get_buyhold_signals, 'params': {}},
}

# Parameter grids for optimization - each symbol gets its optimal parameters
PARAM_GRIDS = {
    'SUPERTREND': {
        'period': [7, 10, 14],
        'multiplier': [1.5, 2.0, 2.5, 3.0]
    },
    'JMA': {
        'period': [5, 7, 10],
        'phase': [50],
        'signal_period': [14, 21, 30]
    },
    'KAMA': {
        'period': [10, 14, 20],
        'fast': [2],
        'slow': [30],
        'signal_period': [5, 10, 14]
    },
    'SMA': {
        'fast': [10, 20, 30],
        'slow': [50, 100, 200]
    },
    'EMA': {
        'fast': [8, 12, 20],
        'slow': [21, 26, 50]
    },
    'BUYHOLD': {}
}


def generate_param_combinations(param_grid: Dict) -> List[Dict]:
    """Generate all combinations of parameters from a grid"""
    if not param_grid:
        return [{}]

    keys = list(param_grid.keys())
    values = list(param_grid.values())

    combinations = []
    from itertools import product

    for combo in product(*values):
        combinations.append(dict(zip(keys, combo)))

    return combinations


# =============================================================================
# IB CONNECTION FUNCTIONS
# =============================================================================

def get_ib_connection(host: str = '127.0.0.1', port: int = 7497, client_id: int = 20) -> IB:
    """Get or create IB connection

    Args:
        host: TWS host (default localhost)
        port: TWS port (7497 for paper, 7496 for live)
        client_id: Unique client ID
    """
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


# =============================================================================
# DATA DOWNLOAD FUNCTIONS (from TWS)
# =============================================================================

def download_data(symbol: str, timeframe: str = TIMEFRAME_DAILY, months: int = 12) -> pd.DataFrame:
    """Download historical data for a symbol from TWS

    Args:
        symbol: Stock ticker
        timeframe: 'daily' or 'hourly'
        months: Number of months of data to download
    """
    ib = get_ib_connection()

    # Create contract
    contract = Stock(symbol, 'SMART', 'USD')
    ib.qualifyContracts(contract)

    # Calculate duration string
    if months <= 12:
        duration = f'{months} M'
    else:
        years = months // 12
        duration = f'{years} Y'

    # Bar size based on timeframe
    if timeframe == TIMEFRAME_HOURLY:
        bar_size = '1 hour'
    else:
        bar_size = '1 day'

    # Request historical data
    bars = ib.reqHistoricalData(
        contract,
        endDateTime='',  # Now
        durationStr=duration,
        barSizeSetting=bar_size,
        whatToShow='TRADES',
        useRTH=True,  # Regular trading hours only
        formatDate=1
    )

    if not bars:
        raise ValueError(f"No data available for {symbol}")

    # Convert to DataFrame
    df = util.df(bars)
    df.set_index('date', inplace=True)

    # Rename columns to match expected format
    df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }, inplace=True)

    return df


def download_data_bulk(symbols: List[str], timeframe: str = TIMEFRAME_DAILY,
                       months: int = 12) -> Dict[str, pd.DataFrame]:
    """Download historical data for multiple symbols from TWS

    Args:
        symbols: List of stock tickers
        timeframe: 'daily' or 'hourly'
        months: Number of months of data to download

    Returns:
        Dict mapping symbol to DataFrame
    """
    ib = get_ib_connection()
    data = {}

    # Bar size based on timeframe
    if timeframe == TIMEFRAME_HOURLY:
        bar_size = '1 hour'
    else:
        bar_size = '1 day'

    # Calculate duration string
    if months <= 12:
        duration = f'{months} M'
    else:
        years = months // 12
        duration = f'{years} Y'

    for symbol in symbols:
        try:
            contract = Stock(symbol, 'SMART', 'USD')
            ib.qualifyContracts(contract)

            bars = ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow='TRADES',
                useRTH=True,
                formatDate=1
            )

            if bars:
                df = util.df(bars)
                df.set_index('date', inplace=True)
                df.rename(columns={
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                }, inplace=True)
                data[symbol] = df

            # Small delay to avoid pacing violations
            ib.sleep(0.5)

        except Exception as e:
            print(f"  Error downloading {symbol}: {e}")

    return data


def split_optimization_simulation(df: pd.DataFrame, opt_months: int = 6, sim_months: int = 6) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into optimization and simulation periods

    Args:
        df: Full historical data
        opt_months: Months for optimization (first period)
        sim_months: Months for simulation (second period)

    Returns:
        (optimization_df, simulation_df)
    """
    total_days = len(df)

    # Calculate split point (optimization period first, then simulation)
    opt_days = int(total_days * opt_months / (opt_months + sim_months))

    opt_df = df.iloc[:opt_days].copy()
    sim_df = df.iloc[opt_days:].copy()

    return opt_df, sim_df


def test_symbol_all_strategies(symbol: str, df: pd.DataFrame, capital: float = DEFAULT_CAPITAL,
                                optimize_params: bool = True) -> Dict:
    """Test a symbol with all strategies and ALL parameter combinations

    Args:
        symbol: Stock ticker
        df: Historical price data
        capital: Initial capital
        optimize_params: If True, test all parameter combinations from PARAM_GRIDS
                        If False, only test default parameters (faster)

    Returns:
        Dict with results for each strategy/param combination
    """
    backtester = Backtester(initial_capital=capital)
    results = {}

    df.attrs['symbol'] = symbol

    for strat_name, strat_config in STRATEGIES.items():
        func = strat_config['func']

        # Get parameter combinations to test
        if optimize_params and strat_name in PARAM_GRIDS:
            param_combinations = generate_param_combinations(PARAM_GRIDS[strat_name])
        else:
            # Use default params only
            param_combinations = [strat_config['params'].copy()]

        # Test each parameter combination
        for param_idx, params in enumerate(param_combinations):
            # Create a unique key for this parameter combination
            if optimize_params and len(param_combinations) > 1:
                param_str = '_'.join(f"{k}{v}" for k, v in params.items())
                base_key = f"{strat_name}_{param_str}"
            else:
                base_key = strat_name

            # Without HTF filter
            try:
                signals = func(df, **params, htf_filter=False)
                result = backtester.run(df, signals)
                result['params'] = params.copy()
                result['htf_filter'] = False
                result['htf_period'] = None
                results[base_key] = result
            except Exception as e:
                results[base_key] = {
                    'error': str(e), 'win_rate': 0, 'total_return_pct': -100,
                    'params': params.copy(), 'htf_filter': False, 'htf_period': None
                }

            # With HTF filter (each period)
            for htf_period in HTF_PERIODS:
                htf_key = f'{base_key}_HTF{htf_period}'
                try:
                    signals = func(df, **params, htf_filter=True, htf_period=htf_period)
                    result = backtester.run(df, signals)
                    result['params'] = params.copy()
                    result['htf_filter'] = True
                    result['htf_period'] = htf_period
                    results[htf_key] = result
                except Exception as e:
                    results[htf_key] = {
                        'error': str(e), 'win_rate': 0, 'total_return_pct': -100,
                        'params': params.copy(), 'htf_filter': True, 'htf_period': htf_period
                    }

    return results


def find_best_strategy(results: Dict, min_trades: int = 3) -> Tuple[str, Dict]:
    """Find the best strategy based on combined score

    Returns:
        (strategy_key, result_dict) where result_dict contains:
        - All performance metrics
        - 'params': the optimal parameters used
        - 'htf_filter': whether HTF filter was used
        - 'htf_period': the HTF period (or None)
        - 'base_strategy': the base strategy name (SUPERTREND, JMA, etc.)
    """
    best_strategy = None
    best_score = -float('inf')
    best_result = None

    for strat_name, result in results.items():
        if 'error' in result:
            continue

        if result.get('total_trades', 0) < min_trades:
            continue

        # Optimize for maximum PnL (total return)
        score = result.get('total_return_pct', -100)

        if score > best_score:
            best_score = score
            best_strategy = strat_name
            best_result = result.copy()

    # Extract base strategy name
    if best_result:
        # Find base strategy (first part before any '_' that matches STRATEGIES keys)
        for base_name in STRATEGIES.keys():
            if best_strategy.startswith(base_name):
                best_result['base_strategy'] = base_name
                break

    return best_strategy, best_result


def optimize_all_symbols(symbols: List[str], capital: float = DEFAULT_CAPITAL,
                         months: int = 12, timeframe: str = TIMEFRAME_DAILY) -> Dict:
    """Optimize all symbols and assign to best strategies with optimal parameters

    Args:
        symbols: List of stock tickers
        capital: Initial capital
        months: Number of months of data (default 12 = 1 year)
        timeframe: 'daily' or 'hourly'
    """
    assignments = {}
    all_results = {}

    # Calculate total combinations
    total_combos = sum(
        len(generate_param_combinations(PARAM_GRIDS.get(s, {}))) * (len(HTF_PERIODS) + 1)
        for s in STRATEGIES.keys()
    )
    print(f"\nTesting {len(symbols)} symbols with ~{total_combos} strategy/param combinations each...")
    print(f"Data source: TWS ({months} months, {timeframe})")
    print("=" * 80)

    for i, symbol in enumerate(symbols):
        print(f"\n[{i+1}/{len(symbols)}] Testing {symbol}...")

        try:
            # Download data from TWS
            df = download_data(symbol, timeframe, months)

            if df.empty or len(df) < 100:
                print(f"  Skipping {symbol}: insufficient data")
                continue

            # Calculate Buy & Hold return for comparison
            buy_hold_return = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100
            print(f"  Buy & Hold: {buy_hold_return:.1f}%")

            # Test all strategies with all parameter combinations
            results = test_symbol_all_strategies(symbol, df, capital, optimize_params=True)
            all_results[symbol] = results

            # Find best strategy with optimal parameters
            best_strat, best_result = find_best_strategy(results)

            if best_strat:
                # Extract base strategy and parameters
                base_strat = best_result.get('base_strategy', best_strat.split('_')[0])
                opt_params = best_result.get('params', {})
                htf_filter = best_result.get('htf_filter', False)
                htf_period = best_result.get('htf_period', None)
                strategy_return = best_result.get('total_return_pct', 0)

                # Check if strategy beats Buy & Hold
                if strategy_return > buy_hold_return:
                    # Strategy wins - use it
                    assignments[symbol] = {
                        'strategy': base_strat,
                        'strategy_key': best_strat,
                        'params': opt_params,
                        'htf_filter': htf_filter,
                        'htf_period': htf_period,
                        'win_rate': best_result.get('win_rate', 0),
                        'profit_factor': best_result.get('profit_factor', 0),
                        'total_return_pct': strategy_return,
                        'sharpe_ratio': best_result.get('sharpe_ratio', 0),
                        'total_trades': best_result.get('total_trades', 0),
                        'buy_hold_return': buy_hold_return,
                        'beats_buyhold': True
                    }

                    htf_info = f" + HTF{htf_period}" if htf_filter else ""
                    print(f"  Best: {base_strat}{htf_info} | Params: {opt_params}")
                    print(f"        Return: {strategy_return:.1f}% vs B&H: {buy_hold_return:.1f}% -> BEATS B&H")
                else:
                    # Buy & Hold wins - assign BUYHOLD
                    assignments[symbol] = {
                        'strategy': 'BUYHOLD',
                        'strategy_key': 'BUYHOLD',
                        'params': {},
                        'htf_filter': False,
                        'htf_period': None,
                        'win_rate': 100.0,  # B&H is always 1 "winning" trade
                        'profit_factor': float('inf') if buy_hold_return > 0 else 0,
                        'total_return_pct': buy_hold_return,
                        'sharpe_ratio': 0,
                        'total_trades': 1,
                        'buy_hold_return': buy_hold_return,
                        'beats_buyhold': False,
                        'best_strategy_return': strategy_return
                    }

                    print(f"  Best strategy: {base_strat} = {strategy_return:.1f}%")
                    print(f"        B&H: {buy_hold_return:.1f}% -> USING BUY & HOLD")
            else:
                # No profitable strategy found - use Buy & Hold
                assignments[symbol] = {
                    'strategy': 'BUYHOLD',
                    'strategy_key': 'BUYHOLD',
                    'params': {},
                    'htf_filter': False,
                    'htf_period': None,
                    'total_return_pct': buy_hold_return,
                    'buy_hold_return': buy_hold_return,
                    'beats_buyhold': False,
                    'reason': 'No profitable strategy found'
                }
                print(f"  No profitable strategy -> USING BUY & HOLD ({buy_hold_return:.1f}%)")

        except Exception as e:
            print(f"  Error: {e}")
            assignments[symbol] = {'strategy': 'EXCLUDED', 'error': str(e)}

    return assignments, all_results


def update_stock_categories(assignments: Dict, output_file: str = 'stock_categories.json'):
    """Update stock_categories.json with new assignments and per-symbol optimized parameters"""
    # Load existing
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            categories = json.load(f)
    else:
        categories = {'strategies': {}, 'ticker_settings': {}, 'watchlist': {'tickers': []}}

    # Clear existing ticker assignments
    for strat_name in categories.get('strategies', {}):
        if 'tickers' in categories['strategies'][strat_name]:
            categories['strategies'][strat_name]['tickers'] = []

    # Initialize symbol_params section for per-symbol optimized parameters
    if 'symbol_params' not in categories:
        categories['symbol_params'] = {}

    # Assign tickers to strategies and store their optimal parameters
    for symbol, data in assignments.items():
        strat = data.get('strategy', 'EXCLUDED')

        if strat not in categories['strategies']:
            categories['strategies'][strat] = {
                'description': f'{strat} strategy',
                'settings': {},
                'tickers': []
            }

        if symbol not in categories['strategies'][strat].get('tickers', []):
            categories['strategies'][strat]['tickers'].append(symbol)

        # Store per-symbol optimized parameters
        if strat != 'EXCLUDED' and 'params' in data:
            categories['symbol_params'][symbol] = {
                'strategy': strat,
                'params': data.get('params', {}),
                'htf_filter': data.get('htf_filter', False),
                'htf_period': data.get('htf_period', None),
                'metrics': {
                    'win_rate': data.get('win_rate', 0),
                    'profit_factor': data.get('profit_factor', 0),
                    'total_return_pct': data.get('total_return_pct', 0),
                    'sharpe_ratio': data.get('sharpe_ratio', 0),
                }
            }

    # Sort tickers
    for strat_name in categories['strategies']:
        if 'tickers' in categories['strategies'][strat_name]:
            categories['strategies'][strat_name]['tickers'].sort()

    # Update metadata
    total_optimized = sum(1 for s, d in assignments.items() if d.get('strategy') != 'EXCLUDED')
    categories['_comment'] = f"Auto-optimized {total_optimized} symbols with individual parameters"
    categories['_last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Save
    with open(output_file, 'w') as f:
        json.dump(categories, f, indent=4)

    print(f"\nUpdated {output_file}")

    # Print summary
    print("\nStrategy Assignment Summary:")
    print("-" * 40)
    for strat_name, strat_data in categories['strategies'].items():
        tickers = strat_data.get('tickers', [])
        if tickers:
            print(f"{strat_name}: {len(tickers)} symbols")

    print(f"\nPer-symbol optimized parameters: {len(categories.get('symbol_params', {}))} symbols")


# =============================================================================
# OPTIMIZE AND SIMULATE
# =============================================================================

def optimize_and_simulate(symbols: List[str], capital: float = DEFAULT_CAPITAL,
                          timeframe: str = TIMEFRAME_DAILY) -> Dict:
    """
    Full optimization workflow:
    1. Download 12 months of data
    2. Use first 6 months for optimization (finding best strategy)
    3. Use last 6 months for simulation (validating)
    4. Store optimized parameters per symbol
    """
    results = {}
    optimized_params = {}

    print(f"\n{'='*80}")
    print(f"OPTIMIZATION & SIMULATION")
    print(f"{'='*80}")
    print(f"Timeframe: {timeframe.upper()}")
    print(f"HTF Periods: {HTF_PERIODS}")
    print(f"Optimization: {OPTIMIZATION_MONTHS} months")
    print(f"Simulation: {SIMULATION_MONTHS} months")
    print(f"Symbols: {len(symbols)}")
    print(f"{'='*80}\n")

    for i, symbol in enumerate(symbols):
        print(f"\n[{i+1}/{len(symbols)}] {symbol}")
        print("-" * 40)

        try:
            # Download full data (12 months)
            df = download_data(symbol, timeframe, OPTIMIZATION_MONTHS + SIMULATION_MONTHS)
            print(f"  Data: {len(df)} bars ({timeframe})")

            if len(df) < 100:
                print(f"  SKIP: Insufficient data")
                continue

            # Split into optimization and simulation periods
            opt_df, sim_df = split_optimization_simulation(df, OPTIMIZATION_MONTHS, SIMULATION_MONTHS)
            print(f"  Optimization: {len(opt_df)} bars | Simulation: {len(sim_df)} bars")

            # Phase 1: Optimize on first 6 months
            print(f"  Phase 1: Optimizing...")
            opt_results = test_symbol_all_strategies(symbol, opt_df, capital)
            best_strat, best_opt_result = find_best_strategy(opt_results)

            if not best_strat:
                print(f"  SKIP: No profitable strategy in optimization")
                continue

            print(f"  Best Strategy: {best_strat}")
            print(f"    Opt Win Rate: {best_opt_result.get('win_rate', 0):.1f}%")
            print(f"    Opt Return: {best_opt_result.get('total_return_pct', 0):.1f}%")

            # Phase 2: Simulate on last 6 months with best strategy
            print(f"  Phase 2: Simulating...")
            backtester = Backtester(initial_capital=capital)

            # Get the strategy function and OPTIMIZED params from the result
            base_strat = best_opt_result.get('base_strategy')
            if not base_strat:
                # Fallback: extract from strategy name
                for name in STRATEGIES.keys():
                    if best_strat.startswith(name):
                        base_strat = name
                        break

            if base_strat and base_strat in STRATEGIES:
                func = STRATEGIES[base_strat]['func']

                # Use the OPTIMIZED parameters from the best result
                opt_params = best_opt_result.get('params', STRATEGIES[base_strat]['params'].copy())
                htf_filter = best_opt_result.get('htf_filter', False)
                htf_period = best_opt_result.get('htf_period', None)

                # Build full params for simulation
                sim_params = opt_params.copy()
                sim_params['htf_filter'] = htf_filter
                if htf_filter and htf_period:
                    sim_params['htf_period'] = htf_period

                print(f"    Params: {opt_params}")
                print(f"    HTF Filter: {htf_filter}" + (f" (period={htf_period})" if htf_filter else ""))

                # Run simulation
                sim_df.attrs['symbol'] = symbol
                signals = func(sim_df, **sim_params)
                sim_result = backtester.run(sim_df, signals)

                print(f"    Sim Win Rate: {sim_result.get('win_rate', 0):.1f}%")
                print(f"    Sim Return: {sim_result.get('total_return_pct', 0):.1f}%")
                print(f"    Sim Trades: {sim_result.get('total_trades', 0)}")

                # Store results with OPTIMIZED parameters
                results[symbol] = {
                    'strategy': base_strat,
                    'strategy_key': best_strat,
                    'optimization': {
                        'win_rate': best_opt_result.get('win_rate', 0),
                        'return_pct': best_opt_result.get('total_return_pct', 0),
                        'trades': best_opt_result.get('total_trades', 0),
                    },
                    'simulation': {
                        'win_rate': sim_result.get('win_rate', 0),
                        'return_pct': sim_result.get('total_return_pct', 0),
                        'trades': sim_result.get('total_trades', 0),
                        'profit_factor': sim_result.get('profit_factor', 0),
                    },
                    'params': opt_params,
                    'htf_filter': htf_filter,
                    'htf_period': htf_period,
                    'timeframe': timeframe
                }

                # Store optimized params for this symbol
                optimized_params[symbol] = {
                    'strategy': base_strat,
                    'params': opt_params,
                    'htf_filter': htf_filter,
                    'htf_period': htf_period,
                    'timeframe': timeframe,
                    'optimization_metrics': {
                        'win_rate': best_opt_result.get('win_rate', 0),
                        'return_pct': best_opt_result.get('total_return_pct', 0),
                    },
                    'simulation_metrics': {
                        'win_rate': sim_result.get('win_rate', 0),
                        'return_pct': sim_result.get('total_return_pct', 0),
                    },
                    'optimized_date': datetime.now().strftime("%Y-%m-%d")
                }

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    return results, optimized_params


def save_optimized_params(params: Dict, filename: str = 'optimized_params.json'):
    """Save optimized parameters to JSON file"""
    output = {
        '_comment': f"Optimized parameters for {len(params)} symbols",
        '_last_optimized': datetime.now().strftime("%Y-%m-%d %H:%M"),
        '_next_optimization': (datetime.now() + timedelta(weeks=4)).strftime("%Y-%m-%d"),
        'symbols': params
    }

    with open(filename, 'w') as f:
        json.dump(output, f, indent=4)

    print(f"\nSaved optimized parameters to {filename}")


# =============================================================================
# MAIN
# =============================================================================

DEFAULT_SYMBOLS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'NFLX', 'ADBE',
    'CRM', 'ORCL', 'INTC', 'CSCO', 'QCOM', 'AVGO', 'TXN', 'MU', 'AMAT', 'LRCX',
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'V', 'MA', 'PYPL', 'SQ', 'COIN',
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD', 'MPC', 'VLO', 'PSX', 'OXY',
    'JNJ', 'UNH', 'PFE', 'MRK', 'ABBV', 'LLY', 'BMY', 'AMGN', 'GILD', 'MRNA',
    'DIS', 'CMCSA', 'NFLX', 'T', 'VZ', 'TMUS',
    'NKE', 'SBUX', 'MCD', 'HD', 'LOW', 'TGT', 'WMT', 'COST',
    'BA', 'LMT', 'RTX', 'NOC', 'GD',
    'UBER', 'LYFT', 'ABNB', 'BKNG', 'EXPE',
    'PLTR', 'SNOW', 'CRWD', 'ZS', 'NET', 'DDOG', 'MDB', 'SHOP'
]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-Strategy Backtester & Optimizer')
    parser.add_argument('--symbols', type=str, help='Comma-separated list of symbols')
    parser.add_argument('--all', action='store_true', help='Test all default symbols')
    parser.add_argument('--capital', type=float, default=DEFAULT_CAPITAL, help='Initial capital')
    parser.add_argument('--timeframe', type=str, default='daily', choices=['daily', 'hourly'],
                        help='Timeframe: daily or hourly')
    parser.add_argument('--optimize', action='store_true', help='Run full optimization + simulation')
    parser.add_argument('--output', type=str, default='stock_categories.json', help='Output file')

    args = parser.parse_args()

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(',')]
    elif args.all:
        symbols = DEFAULT_SYMBOLS
    else:
        print("Usage:")
        print("  python strategy_backtester.py --symbols AAPL,MSFT,GOOGL")
        print("  python strategy_backtester.py --all")
        print("  python strategy_backtester.py --all --timeframe hourly")
        print("  python strategy_backtester.py --all --optimize --timeframe daily")
        exit(1)

    # Remove duplicates
    symbols = list(dict.fromkeys(symbols))

    print(f"Strategy Backtester & Optimizer")
    print(f"================================")
    print(f"Capital: ${args.capital:,.2f}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Symbols: {len(symbols)}")

    if args.optimize:
        # Run full optimization + simulation workflow
        print(f"\nMode: OPTIMIZE + SIMULATE (6 months each)")
        results, optimized_params = optimize_and_simulate(symbols, args.capital, args.timeframe)

        # Save optimized parameters
        save_optimized_params(optimized_params, 'optimized_params.json')

        # Update stock categories based on results
        assignments = {sym: {'strategy': data['strategy']} for sym, data in results.items()}
        update_stock_categories(assignments, args.output)

        # Print summary
        print(f"\n{'='*80}")
        print("OPTIMIZATION SUMMARY")
        print(f"{'='*80}")
        print(f"Symbols optimized: {len(results)}")

        if results:
            avg_opt_wr = sum(r['optimization']['win_rate'] for r in results.values()) / len(results)
            avg_sim_wr = sum(r['simulation']['win_rate'] for r in results.values()) / len(results)
            print(f"Avg Optimization Win Rate: {avg_opt_wr:.1f}%")
            print(f"Avg Simulation Win Rate: {avg_sim_wr:.1f}%")

        # Disconnect from TWS
        disconnect_ib()
    else:
        # Run standard optimization (1 year data from TWS)
        print(f"\nMode: STANDARD (test all strategies)")
        assignments, all_results = optimize_all_symbols(symbols, args.capital, months=12, timeframe=args.timeframe)

        # Update categories file
        update_stock_categories(assignments, args.output)

    # Disconnect from TWS
    disconnect_ib()

    print("\nDone!")
