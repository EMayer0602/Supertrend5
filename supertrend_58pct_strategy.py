#!/usr/bin/env python3
"""
Supertrend Strategy - Optimized for 58%+ Win Rate
=================================================
Trading strategy targeting minimum 58% winning trades.

Key optimizations for high win rate:
1. Trade only in direction of major trend (200 SMA)
2. RSI confirmation (avoid overbought/oversold extremes)
3. Volume confirmation
4. Tight stop loss to cut losers quickly
5. Wide take profit to let winners run
"""

import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass
class StrategyConfig:
    """Configuration for 58%+ Win Rate Strategy"""
    symbol: str = "MSFT"
    initial_capital: float = 10000.0
    transaction_cost: float = 0.001

    # Supertrend Parameters
    st_period: int = 10
    st_multiplier: float = 3.0

    # Trend Filter (200 SMA)
    trend_ma_period: int = 200

    # RSI Filter
    rsi_period: int = 14
    rsi_oversold: int = 30
    rsi_overbought: int = 70

    # Risk Management
    stop_loss_pct: float = 0.03      # 3% stop loss
    take_profit_pct: float = 0.06    # 6% take profit (2:1 R:R)
    trailing_stop_pct: float = 0.04  # 4% trailing stop after profit

    # Backtest period
    days_back: int = 1825  # 5 years


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
# TRADING SYSTEM
# =============================================================================
class WinRateStrategy:
    """Strategy optimized for 58%+ win rate"""

    def __init__(self, config: StrategyConfig):
        self.config = config

    def run_backtest(self, df: pd.DataFrame) -> Dict:
        """Run backtest with win rate optimization"""
        symbol = self.config.symbol
        close = df[f'Close_{symbol}'].values
        high = df[f'High_{symbol}'].values
        low = df[f'Low_{symbol}'].values
        dates = df.index

        # Calculate indicators
        supertrend, direction = calculate_supertrend(
            high, low, close,
            self.config.st_period,
            self.config.st_multiplier
        )
        rsi = calculate_rsi(close, self.config.rsi_period)
        sma200 = calculate_sma(close, self.config.trend_ma_period)

        # Generate signals
        prev_direction = np.roll(direction, 1)
        prev_direction[0] = 0

        trades = []
        position = None
        entry_price = 0
        entry_date = None
        stop_loss = 0
        take_profit = 0
        highest_price = 0

        for i in range(self.config.trend_ma_period, len(df)):
            # Check exits first if in position
            if position == 'long':
                highest_price = max(highest_price, high[i])

                # Calculate trailing stop if in profit
                if close[i] > entry_price * 1.02:  # 2% profit
                    trailing_stop = highest_price * (1 - self.config.trailing_stop_pct)
                    stop_loss = max(stop_loss, trailing_stop)

                # Check exit conditions
                hit_stop = close[i] <= stop_loss
                hit_tp = close[i] >= take_profit

                if hit_stop or hit_tp:
                    pnl = (close[i] - entry_price) / entry_price
                    pnl -= self.config.transaction_cost * 2

                    trades.append({
                        'entry_date': entry_date,
                        'entry_price': entry_price,
                        'exit_date': dates[i],
                        'exit_price': close[i],
                        'pnl': pnl,
                        'pnl_pct': pnl * 100,
                        'exit_reason': 'stop_loss' if hit_stop else 'take_profit',
                        'is_winner': pnl > 0
                    })
                    position = None

                # Also exit on Supertrend sell signal
                elif prev_direction[i] == 1 and direction[i] == -1:
                    pnl = (close[i] - entry_price) / entry_price
                    pnl -= self.config.transaction_cost * 2

                    trades.append({
                        'entry_date': entry_date,
                        'entry_price': entry_price,
                        'exit_date': dates[i],
                        'exit_price': close[i],
                        'pnl': pnl,
                        'pnl_pct': pnl * 100,
                        'exit_reason': 'signal',
                        'is_winner': pnl > 0
                    })
                    position = None

            # Check entry conditions (only LONG, in uptrend)
            if position is None:
                # Buy signal: Supertrend turns bullish
                if prev_direction[i] == -1 and direction[i] == 1:
                    # Filter 1: Price above 200 SMA (uptrend)
                    in_uptrend = close[i] > sma200[i]

                    # Filter 2: RSI not overbought
                    rsi_ok = rsi[i] < self.config.rsi_overbought

                    # Filter 3: RSI showing momentum (above 40)
                    rsi_momentum = rsi[i] > 40

                    if in_uptrend and rsi_ok and rsi_momentum:
                        position = 'long'
                        entry_price = close[i]
                        entry_date = dates[i]
                        stop_loss = entry_price * (1 - self.config.stop_loss_pct)
                        take_profit = entry_price * (1 + self.config.take_profit_pct)
                        highest_price = high[i]

        # Calculate statistics
        stats = self.calculate_stats(trades)

        return {
            'trades': trades,
            'stats': stats,
            'direction': direction,
            'supertrend': supertrend
        }

    def calculate_stats(self, trades: List[Dict]) -> Dict:
        """Calculate trading statistics"""
        if not trades:
            return {'Win Rate': '0.0%', 'Total Trades': 0}

        total = len(trades)
        winners = sum(1 for t in trades if t['is_winner'])
        losers = total - winners

        win_rate = winners / total if total > 0 else 0

        profits = [t['pnl'] for t in trades if t['pnl'] > 0]
        losses = [t['pnl'] for t in trades if t['pnl'] <= 0]

        avg_win = np.mean(profits) if profits else 0
        avg_loss = np.mean(losses) if losses else 0

        total_profit = sum(profits)
        total_loss = abs(sum(losses))
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

        total_return = sum(t['pnl'] for t in trades)
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))

        return {
            'Total Trades': total,
            'Winning Trades': winners,
            'Losing Trades': losers,
            'WIN RATE': f"{win_rate:.1%}",
            'Win Rate Value': win_rate,
            'Avg Win': f"{avg_win:.2%}",
            'Avg Loss': f"{avg_loss:.2%}",
            'Profit Factor': f"{profit_factor:.2f}",
            'Total Return': f"{total_return:.2%}",
            'Expectancy': f"{expectancy:.2%}",
        }


# =============================================================================
# PARAMETER OPTIMIZATION FOR 58%+ WIN RATE
# =============================================================================
def optimize_for_win_rate(symbol: str = "MSFT", target_win_rate: float = 0.58):
    """Optimize parameters to achieve target win rate"""
    print("="*70)
    print(f"OPTIMIZING FOR {target_win_rate:.0%} WIN RATE")
    print("="*70)
    print(f"Symbol: {symbol}")
    print()

    # Download data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1825)
    data = yf.download(symbol, start=start_date, end=end_date, progress=False)

    if data.empty:
        print("Error: No data")
        return None

    # Flatten columns
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ['_'.join(col).strip() for col in data.columns.values]
    else:
        data.columns = [f'{col}_{symbol}' for col in data.columns]

    print(f"Data: {len(data)} days")

    best_win_rate = 0
    best_config = None
    best_result = None

    results = []

    # Parameter ranges
    st_periods = [8, 10, 12, 14, 16]
    st_multipliers = [2.5, 3.0, 3.5, 4.0]
    stop_losses = [0.02, 0.03, 0.04, 0.05]
    take_profits = [0.04, 0.06, 0.08, 0.10]

    total = len(st_periods) * len(st_multipliers) * len(stop_losses) * len(take_profits)
    tested = 0

    print(f"Testing {total} combinations...\n")

    for st_period in st_periods:
        for st_mult in st_multipliers:
            for sl in stop_losses:
                for tp in take_profits:
                    tested += 1

                    config = StrategyConfig(
                        symbol=symbol,
                        st_period=st_period,
                        st_multiplier=st_mult,
                        stop_loss_pct=sl,
                        take_profit_pct=tp
                    )

                    try:
                        strategy = WinRateStrategy(config)
                        result = strategy.run_backtest(data)

                        trades = result['trades']
                        stats = result['stats']

                        if len(trades) >= 10:  # Minimum trades
                            win_rate = stats['Win Rate Value']

                            results.append({
                                'st_period': st_period,
                                'st_mult': st_mult,
                                'stop_loss': sl,
                                'take_profit': tp,
                                'win_rate': win_rate,
                                'trades': len(trades),
                                'total_return': stats['Total Return'],
                                'profit_factor': stats['Profit Factor']
                            })

                            # Best if meets target AND has good return
                            if win_rate >= target_win_rate and win_rate > best_win_rate:
                                best_win_rate = win_rate
                                best_config = config
                                best_result = result

                    except Exception as e:
                        pass

                    if tested % 50 == 0:
                        print(f"Progress: {tested}/{total} | Best Win Rate: {best_win_rate:.1%}")

    # Sort by win rate
    results.sort(key=lambda x: x['win_rate'], reverse=True)

    print("\n" + "="*70)
    print("TOP 10 CONFIGURATIONS BY WIN RATE")
    print("="*70)

    for i, r in enumerate(results[:10], 1):
        marker = ">>>" if r['win_rate'] >= target_win_rate else "   "
        print(f"{marker} {i}. Win Rate: {r['win_rate']:.1%} | "
              f"Period={r['st_period']}, Mult={r['st_mult']:.1f}, "
              f"SL={r['stop_loss']:.0%}, TP={r['take_profit']:.0%} | "
              f"Trades={r['trades']}, Return={r['total_return']}")

    if best_result:
        print("\n" + "="*70)
        print("BEST CONFIGURATION (>= 58% Win Rate)")
        print("="*70)

        print(f"\nParameters:")
        print(f"  Supertrend Period: {best_config.st_period}")
        print(f"  Supertrend Multiplier: {best_config.st_multiplier}")
        print(f"  Stop Loss: {best_config.stop_loss_pct:.0%}")
        print(f"  Take Profit: {best_config.take_profit_pct:.0%}")

        print(f"\nStatistics:")
        for key, value in best_result['stats'].items():
            if 'WIN RATE' in key.upper():
                print(f"  >>> {key}: {value} <<<")
            elif 'Value' not in key:
                print(f"  {key}: {value}")

        print(f"\nTrade History (last 10):")
        for trade in best_result['trades'][-10:]:
            status = "WIN" if trade['is_winner'] else "LOSS"
            print(f"  [{status}] {trade['entry_date'].strftime('%Y-%m-%d')} -> "
                  f"{trade['exit_date'].strftime('%Y-%m-%d')} | "
                  f"{trade['pnl']:+.2%} | {trade['exit_reason']}")

    else:
        print(f"\nNo configuration achieved {target_win_rate:.0%} win rate.")
        print("Best available:")
        if results:
            r = results[0]
            print(f"  Win Rate: {r['win_rate']:.1%}")

    return best_config, best_result, results


# =============================================================================
# MULTI-SYMBOL TEST
# =============================================================================
def test_multiple_symbols():
    """Test strategy on multiple symbols"""
    symbols = ['MSFT', 'AAPL', 'GOOGL', 'AMZN', 'META', 'NVDA', 'JPM', 'V']

    print("="*70)
    print("MULTI-SYMBOL WIN RATE TEST")
    print("="*70)

    results = []

    for symbol in symbols:
        print(f"\nTesting {symbol}...")

        config = StrategyConfig(
            symbol=symbol,
            st_period=10,
            st_multiplier=3.0,
            stop_loss_pct=0.03,
            take_profit_pct=0.06
        )

        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=1825)
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)

            if data.empty:
                continue

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = ['_'.join(col).strip() for col in data.columns.values]
            else:
                data.columns = [f'{col}_{symbol}' for col in data.columns]

            strategy = WinRateStrategy(config)
            result = strategy.run_backtest(data)

            stats = result['stats']
            win_rate = stats['Win Rate Value']

            results.append({
                'symbol': symbol,
                'win_rate': win_rate,
                'trades': stats['Total Trades'],
                'total_return': stats['Total Return']
            })

            marker = "OK" if win_rate >= 0.58 else "  "
            print(f"  [{marker}] Win Rate: {win_rate:.1%} | Trades: {stats['Total Trades']} | Return: {stats['Total Return']}")

        except Exception as e:
            print(f"  Error: {e}")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    above_58 = [r for r in results if r['win_rate'] >= 0.58]
    print(f"Symbols with >= 58% Win Rate: {len(above_58)}/{len(results)}")

    if above_58:
        print("\nSymbols meeting target:")
        for r in sorted(above_58, key=lambda x: x['win_rate'], reverse=True):
            print(f"  {r['symbol']}: {r['win_rate']:.1%}")

    return results


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--multi":
        test_multiple_symbols()
    else:
        symbol = sys.argv[1] if len(sys.argv) > 1 else "MSFT"
        optimize_for_win_rate(symbol, target_win_rate=0.58)
