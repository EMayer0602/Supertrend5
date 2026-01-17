#!/usr/bin/env python3
"""
Multi-Strategy Backtester & Symbol Optimizer
=============================================
Tests all symbols with multiple strategies and assigns each to its best strategy.

Strategies: JMA, Supertrend, SMA, KAMA, EMA, Buy&Hold (each with/without HTF JMA filter)

Position Sizing:
- Amount per trade = Capital / 30 / entry_price (rounded)
- Max 20 positions open at once
- Entry fee deducted at entry
- Exit fee deducted at exit
- Capital = Capital + daily_pnl

Usage:
    python strategy_backtester.py --symbols AAPL,MSFT,GOOGL --capital 100000
    python strategy_backtester.py --all --capital 100000
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
import argparse
import os

# =============================================================================
# CONSTANTS
# =============================================================================
DEFAULT_CAPITAL = 100000
POSITION_DIVISOR = 30  # Capital / 30 per trade
MAX_POSITIONS = 20
FEE_RATE = 0.001  # 0.1% per trade (entry + exit)
TRAILING_STOP_PCT = 0.12  # 12% trailing stop

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

STRATEGIES = {
    'SUPERTREND': {'func': get_supertrend_signals, 'params': {'period': 10, 'multiplier': 2.0}},
    'JMA': {'func': get_jma_signals, 'params': {'period': 7, 'phase': 50, 'signal_period': 21}},
    'KAMA': {'func': get_kama_signals, 'params': {'period': 10, 'fast': 2, 'slow': 30, 'signal_period': 10}},
    'SMA': {'func': get_sma_signals, 'params': {'fast': 20, 'slow': 50}},
    'EMA': {'func': get_ema_signals, 'params': {'fast': 12, 'slow': 26}},
    'BUYHOLD': {'func': get_buyhold_signals, 'params': {}},
}

HTF_PERIODS = [50, 100, 150, 200]


def test_symbol_all_strategies(symbol: str, df: pd.DataFrame, capital: float = DEFAULT_CAPITAL) -> Dict:
    """Test a symbol with all strategies and return results"""
    backtester = Backtester(initial_capital=capital)
    results = {}

    df.attrs['symbol'] = symbol

    for strat_name, strat_config in STRATEGIES.items():
        func = strat_config['func']
        params = strat_config['params'].copy()

        # Without HTF filter
        try:
            signals = func(df, **params, htf_filter=False)
            result = backtester.run(df, signals)
            results[strat_name] = result
        except Exception as e:
            results[strat_name] = {'error': str(e), 'win_rate': 0, 'total_return_pct': -100}

        # With HTF filter (each period)
        for htf_period in HTF_PERIODS:
            try:
                signals = func(df, **params, htf_filter=True, htf_period=htf_period)
                result = backtester.run(df, signals)
                results[f'{strat_name}_HTF{htf_period}'] = result
            except Exception as e:
                results[f'{strat_name}_HTF{htf_period}'] = {'error': str(e), 'win_rate': 0, 'total_return_pct': -100}

    return results


def find_best_strategy(results: Dict, min_trades: int = 3) -> Tuple[str, Dict]:
    """Find the best strategy based on combined score"""
    best_strategy = None
    best_score = -float('inf')
    best_result = None

    for strat_name, result in results.items():
        if 'error' in result:
            continue

        if result.get('total_trades', 0) < min_trades:
            continue

        # Combined score: win_rate * 0.3 + profit_factor * 0.3 + total_return_pct * 0.2 + sharpe * 0.2
        win_rate = result.get('win_rate', 0)
        pf = min(result.get('profit_factor', 0), 10)  # Cap profit factor
        ret = result.get('total_return_pct', -100)
        sharpe = result.get('sharpe_ratio', 0)

        score = win_rate * 0.3 + pf * 10 * 0.3 + ret * 0.2 + sharpe * 5 * 0.2

        if score > best_score:
            best_score = score
            best_strategy = strat_name
            best_result = result

    return best_strategy, best_result


def optimize_all_symbols(symbols: List[str], capital: float = DEFAULT_CAPITAL,
                         period: str = '2y') -> Dict:
    """Optimize all symbols and assign to best strategies"""
    assignments = {}
    all_results = {}

    print(f"\nTesting {len(symbols)} symbols with {len(STRATEGIES) * (len(HTF_PERIODS) + 1)} strategy variants...")
    print("=" * 80)

    for i, symbol in enumerate(symbols):
        print(f"\n[{i+1}/{len(symbols)}] Testing {symbol}...")

        try:
            # Download data
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)

            if df.empty or len(df) < 100:
                print(f"  Skipping {symbol}: insufficient data")
                continue

            # Test all strategies
            results = test_symbol_all_strategies(symbol, df, capital)
            all_results[symbol] = results

            # Find best strategy
            best_strat, best_result = find_best_strategy(results)

            if best_strat:
                assignments[symbol] = {
                    'strategy': best_strat,
                    'win_rate': best_result.get('win_rate', 0),
                    'profit_factor': best_result.get('profit_factor', 0),
                    'total_return_pct': best_result.get('total_return_pct', 0),
                    'sharpe_ratio': best_result.get('sharpe_ratio', 0),
                    'total_trades': best_result.get('total_trades', 0)
                }
                print(f"  Best: {best_strat} | Win: {best_result.get('win_rate', 0):.1f}% | "
                      f"PF: {best_result.get('profit_factor', 0):.2f} | "
                      f"Return: {best_result.get('total_return_pct', 0):.1f}%")
            else:
                print(f"  No suitable strategy found")
                assignments[symbol] = {'strategy': 'EXCLUDED', 'reason': 'No profitable strategy'}

        except Exception as e:
            print(f"  Error: {e}")
            assignments[symbol] = {'strategy': 'EXCLUDED', 'error': str(e)}

    return assignments, all_results


def update_stock_categories(assignments: Dict, output_file: str = 'stock_categories.json'):
    """Update stock_categories.json with new assignments"""
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

    # Assign tickers to strategies
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

    # Sort tickers
    for strat_name in categories['strategies']:
        if 'tickers' in categories['strategies'][strat_name]:
            categories['strategies'][strat_name]['tickers'].sort()

    # Update metadata
    categories['_comment'] = f"Auto-optimized {len(assignments)} symbols"
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
    'NKE', 'SBUX', 'MCD', 'HD', 'LOW', 'TGT', 'WMT', 'COST', 'AMZN',
    'BA', 'LMT', 'RTX', 'NOC', 'GD',
    'UBER', 'LYFT', 'ABNB', 'BKNG', 'EXPE',
    'PLTR', 'SNOW', 'CRWD', 'ZS', 'NET', 'DDOG', 'MDB', 'SHOP'
]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-Strategy Backtester & Optimizer')
    parser.add_argument('--symbols', type=str, help='Comma-separated list of symbols')
    parser.add_argument('--all', action='store_true', help='Test all default symbols')
    parser.add_argument('--capital', type=float, default=DEFAULT_CAPITAL, help='Initial capital')
    parser.add_argument('--period', type=str, default='2y', help='Data period (1y, 2y, 5y)')
    parser.add_argument('--output', type=str, default='stock_categories.json', help='Output file')

    args = parser.parse_args()

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(',')]
    elif args.all:
        symbols = DEFAULT_SYMBOLS
    else:
        print("Usage: python strategy_backtester.py --symbols AAPL,MSFT,GOOGL")
        print("       python strategy_backtester.py --all")
        exit(1)

    # Remove duplicates
    symbols = list(dict.fromkeys(symbols))

    print(f"Strategy Backtester & Optimizer")
    print(f"================================")
    print(f"Capital: ${args.capital:,.2f}")
    print(f"Period: {args.period}")
    print(f"Symbols: {len(symbols)}")

    # Run optimization
    assignments, all_results = optimize_all_symbols(symbols, args.capital, args.period)

    # Update categories file
    update_stock_categories(assignments, args.output)

    print("\nDone!")
