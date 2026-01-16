#!/usr/bin/env python3
"""
Stock-Strategy Matcher
======================
Analyzes stocks and assigns them to the best performing strategy.

Strategies:
1. Supertrend Classic - Basic Supertrend
2. Supertrend + RSI - With RSI filter
3. Supertrend + Trend - With 200 SMA filter
4. Supertrend Conservative - Tight stops, wide TP
5. Supertrend Aggressive - Wide stops, tight TP
6. Buy & Hold - Benchmark

For each stock, tests all strategies and recommends the best one
based on: Win Rate, Profit Factor, and Total Return.
"""

import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import json
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# INDICATORS
# =============================================================================
def calculate_atr(high, low, close, period=14):
    """Calculate ATR"""
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    atr = np.zeros_like(tr)
    atr[period-1] = np.mean(tr[:period])
    alpha = 2 / (period + 1)
    for i in range(period, len(tr)):
        atr[i] = tr[i] * alpha + atr[i-1] * (1 - alpha)
    return atr


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    deltas = np.diff(close, prepend=close[0])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.zeros(len(close))
    avg_loss = np.zeros(len(close))
    if len(close) > period:
        avg_gain[period] = np.mean(gains[1:period+1])
        avg_loss[period] = np.mean(losses[1:period+1])
        for i in range(period + 1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i]) / period
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    rsi[:period+1] = 50
    return rsi


def calculate_sma(data, period):
    """Calculate SMA"""
    sma = np.zeros_like(data)
    for i in range(period - 1, len(data)):
        sma[i] = np.mean(data[i-period+1:i+1])
    return sma


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend"""
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    hl2 = (high + low) / 2
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr
    final_upper = np.copy(upper)
    final_lower = np.copy(lower)
    supertrend = np.zeros(n)
    direction = np.zeros(n)

    for i in range(period, n):
        if upper[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]:
            final_upper[i] = upper[i]
        else:
            final_upper[i] = final_upper[i-1]
        if lower[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]:
            final_lower[i] = lower[i]
        else:
            final_lower[i] = final_lower[i-1]

    for i in range(period, n):
        if i == period:
            supertrend[i] = final_upper[i] if close[i] <= final_upper[i] else final_lower[i]
            direction[i] = -1 if close[i] <= final_upper[i] else 1
        else:
            if supertrend[i-1] == final_upper[i-1]:
                supertrend[i] = final_upper[i] if close[i] <= final_upper[i] else final_lower[i]
                direction[i] = -1 if close[i] <= final_upper[i] else 1
            else:
                supertrend[i] = final_lower[i] if close[i] >= final_lower[i] else final_upper[i]
                direction[i] = 1 if close[i] >= final_lower[i] else -1

    return supertrend, direction


# =============================================================================
# STRATEGY DEFINITIONS
# =============================================================================
STRATEGIES = {
    'supertrend_classic': {
        'name': 'Supertrend Classic',
        'description': 'Basic Supertrend (Period=10, Mult=3.0)',
        'params': {'st_period': 10, 'st_multiplier': 3.0, 'use_rsi': False, 'use_trend': False,
                   'stop_loss': 0.05, 'take_profit': 0.10}
    },
    'supertrend_rsi': {
        'name': 'Supertrend + RSI',
        'description': 'Supertrend with RSI filter (30-70)',
        'params': {'st_period': 10, 'st_multiplier': 3.0, 'use_rsi': True, 'use_trend': False,
                   'stop_loss': 0.04, 'take_profit': 0.08}
    },
    'supertrend_trend': {
        'name': 'Supertrend + Trend',
        'description': 'Supertrend with 200 SMA trend filter',
        'params': {'st_period': 10, 'st_multiplier': 3.0, 'use_rsi': False, 'use_trend': True,
                   'stop_loss': 0.04, 'take_profit': 0.08}
    },
    'supertrend_full': {
        'name': 'Supertrend Full',
        'description': 'Supertrend + RSI + Trend filter (58% WR target)',
        'params': {'st_period': 10, 'st_multiplier': 3.0, 'use_rsi': True, 'use_trend': True,
                   'stop_loss': 0.03, 'take_profit': 0.06}
    },
    'supertrend_conservative': {
        'name': 'Supertrend Conservative',
        'description': 'Tight stop loss, wide take profit',
        'params': {'st_period': 14, 'st_multiplier': 4.0, 'use_rsi': True, 'use_trend': True,
                   'stop_loss': 0.02, 'take_profit': 0.08}
    },
    'supertrend_aggressive': {
        'name': 'Supertrend Aggressive',
        'description': 'Wide stop loss, more trades',
        'params': {'st_period': 8, 'st_multiplier': 2.5, 'use_rsi': False, 'use_trend': False,
                   'stop_loss': 0.06, 'take_profit': 0.06}
    },
}


# =============================================================================
# BACKTEST ENGINE
# =============================================================================
def backtest_strategy(data: pd.DataFrame, symbol: str, params: Dict) -> Dict:
    """Run backtest for a strategy on a symbol"""
    close = data[f'Close_{symbol}'].values
    high = data[f'High_{symbol}'].values
    low = data[f'Low_{symbol}'].values
    dates = data.index

    # Calculate indicators
    supertrend, direction = calculate_supertrend(
        high, low, close,
        params['st_period'],
        params['st_multiplier']
    )
    rsi = calculate_rsi(close, 14)
    sma200 = calculate_sma(close, 200)

    # Generate signals
    prev_direction = np.roll(direction, 1)
    prev_direction[0] = 0

    trades = []
    position = None
    entry_price = 0
    stop_loss = 0
    take_profit = 0

    start_idx = 200 if params.get('use_trend', False) else params['st_period']

    for i in range(start_idx, len(data)):
        # Check exits
        if position == 'long':
            hit_stop = close[i] <= stop_loss
            hit_tp = close[i] >= take_profit
            exit_signal = prev_direction[i] == 1 and direction[i] == -1

            if hit_stop or hit_tp or exit_signal:
                pnl = (close[i] - entry_price) / entry_price - 0.002
                trades.append({
                    'pnl': pnl,
                    'is_winner': pnl > 0
                })
                position = None

        # Check entries
        if position is None and prev_direction[i] == -1 and direction[i] == 1:
            # Apply filters
            can_enter = True

            if params.get('use_trend', False):
                can_enter = can_enter and close[i] > sma200[i]

            if params.get('use_rsi', False):
                can_enter = can_enter and 30 < rsi[i] < 70

            if can_enter:
                position = 'long'
                entry_price = close[i]
                stop_loss = entry_price * (1 - params['stop_loss'])
                take_profit = entry_price * (1 + params['take_profit'])

    # Calculate stats
    if not trades:
        return {'win_rate': 0, 'profit_factor': 0, 'total_return': 0, 'trades': 0}

    winners = sum(1 for t in trades if t['is_winner'])
    win_rate = winners / len(trades)

    profits = sum(t['pnl'] for t in trades if t['pnl'] > 0)
    losses = abs(sum(t['pnl'] for t in trades if t['pnl'] <= 0))
    profit_factor = profits / losses if losses > 0 else 10

    total_return = sum(t['pnl'] for t in trades)

    return {
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'total_return': total_return,
        'trades': len(trades)
    }


# =============================================================================
# STOCK-STRATEGY MATCHER
# =============================================================================
class StockStrategyMatcher:
    """Matches stocks to their best performing strategy"""

    def __init__(self, symbols: List[str], days_back: int = 1825):
        self.symbols = symbols
        self.days_back = days_back
        self.results = {}
        self.assignments = {}

    def analyze_all(self) -> Dict:
        """Analyze all stocks with all strategies"""
        print("="*70)
        print("STOCK-STRATEGY MATCHER")
        print("="*70)
        print(f"Symbols: {len(self.symbols)}")
        print(f"Strategies: {len(STRATEGIES)}")
        print()

        for symbol in self.symbols:
            print(f"Analyzing {symbol}...", end=" ")

            try:
                # Download data
                end_date = datetime.now()
                start_date = end_date - timedelta(days=self.days_back)
                data = yf.download(symbol, start=start_date, end=end_date, progress=False)

                if data.empty or len(data) < 300:
                    print("Not enough data")
                    continue

                # Flatten columns
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = ['_'.join(col).strip() for col in data.columns.values]
                else:
                    data.columns = [f'{col}_{symbol}' for col in data.columns]

                # Test all strategies
                symbol_results = {}
                best_score = -float('inf')
                best_strategy = None

                for strat_id, strat_info in STRATEGIES.items():
                    result = backtest_strategy(data, symbol, strat_info['params'])

                    if result['trades'] >= 5:  # Minimum trades
                        # Score: weighted combination
                        score = (result['win_rate'] * 0.4 +
                                min(result['profit_factor'] / 5, 1) * 0.3 +
                                min(result['total_return'] + 0.5, 1) * 0.3)

                        symbol_results[strat_id] = {
                            'name': strat_info['name'],
                            **result,
                            'score': score
                        }

                        if score > best_score and result['win_rate'] >= 0.50:
                            best_score = score
                            best_strategy = strat_id

                self.results[symbol] = symbol_results

                if best_strategy:
                    self.assignments[symbol] = {
                        'strategy': best_strategy,
                        'strategy_name': STRATEGIES[best_strategy]['name'],
                        **symbol_results[best_strategy]
                    }
                    print(f"-> {STRATEGIES[best_strategy]['name']} (WR: {symbol_results[best_strategy]['win_rate']:.0%})")
                else:
                    print("No suitable strategy")

            except Exception as e:
                print(f"Error: {e}")

        return self.assignments

    def print_report(self):
        """Print detailed report"""
        print("\n" + "="*70)
        print("STRATEGY ASSIGNMENTS")
        print("="*70)

        # Group by strategy
        by_strategy = {}
        for symbol, info in self.assignments.items():
            strat = info['strategy']
            if strat not in by_strategy:
                by_strategy[strat] = []
            by_strategy[strat].append((symbol, info))

        for strat_id, symbols in sorted(by_strategy.items()):
            strat_name = STRATEGIES[strat_id]['name']
            print(f"\n{strat_name.upper()}")
            print("-" * 50)

            for symbol, info in sorted(symbols, key=lambda x: x[1]['win_rate'], reverse=True):
                print(f"  {symbol:<6} | WR: {info['win_rate']:>5.1%} | "
                      f"PF: {info['profit_factor']:>5.2f} | "
                      f"Return: {info['total_return']:>+6.1%} | "
                      f"Trades: {info['trades']}")

        # Summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)

        for strat_id in STRATEGIES:
            count = len(by_strategy.get(strat_id, []))
            print(f"  {STRATEGIES[strat_id]['name']:<25}: {count} stocks")

        # Best performers
        print("\n" + "-"*50)
        print("TOP 10 BY WIN RATE:")
        sorted_by_wr = sorted(self.assignments.items(),
                              key=lambda x: x[1]['win_rate'], reverse=True)
        for symbol, info in sorted_by_wr[:10]:
            print(f"  {symbol:<6} | {info['strategy_name']:<25} | WR: {info['win_rate']:.1%}")

    def save_assignments(self, filename: str = "strategy_assignments.json"):
        """Save assignments to JSON file"""
        output = {
            'generated': datetime.now().isoformat(),
            'assignments': self.assignments,
            'strategies': {k: v['name'] for k, v in STRATEGIES.items()}
        }

        with open(filename, 'w') as f:
            json.dump(output, f, indent=2, default=str)

        print(f"\nSaved to {filename}")


# =============================================================================
# PORTFOLIO SYMBOLS FROM CONFIG
# =============================================================================
def load_portfolio_symbols():
    """Load symbols from portfolio_config.json"""
    try:
        with open('portfolio_config.json', 'r') as f:
            config = json.load(f)
        symbols = [p['symbol'] for p in config['positions'] if p['qty'] != 0]
        return symbols
    except:
        return None


# =============================================================================
# MAIN
# =============================================================================
def main():
    import sys

    # Default symbols or from portfolio
    portfolio_symbols = load_portfolio_symbols()

    if portfolio_symbols:
        print(f"Loaded {len(portfolio_symbols)} symbols from portfolio_config.json")
        symbols = portfolio_symbols
    else:
        # Default test symbols
        symbols = [
            'MSFT', 'AAPL', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
            'JPM', 'V', 'JNJ', 'UNH', 'HD', 'PG', 'MA', 'DIS',
            'PYPL', 'NFLX', 'AMD', 'CRM', 'ADBE'
        ]

    # Run matcher
    matcher = StockStrategyMatcher(symbols, days_back=1825)
    matcher.analyze_all()
    matcher.print_report()
    matcher.save_assignments()


if __name__ == "__main__":
    main()
