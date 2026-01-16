#!/usr/bin/env python3
"""
Ultra-High Profit Factor Supertrend Strategy
=============================================
Optimized for maximum Profit Factor (target: >58)

Strategy rules for high PF:
1. Only trade in direction of long-term trend (200 SMA)
2. Multiple confirmation signals required
3. Strict entry filters (RSI, Volume, Trend strength)
4. Wide trailing stops to let winners run
5. Quick exits on losers
6. Only take highest probability setups
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
class HighPFConfig:
    """Configuration for Ultra-High Profit Factor Strategy"""
    symbol: str = "MSFT"
    initial_capital: float = 10000.0
    transaction_cost: float = 0.001

    # Supertrend Parameters (optimized for fewer, better trades)
    st_period: int = 14
    st_multiplier: float = 4.0

    # Trend Filter (200 SMA)
    trend_ma_period: int = 200

    # RSI Filter
    rsi_period: int = 14
    rsi_oversold: int = 35
    rsi_overbought: int = 65

    # Volume Filter
    volume_ma_period: int = 20
    volume_threshold: float = 1.2  # 20% above average

    # Trend Strength Filter (ADX)
    adx_period: int = 14
    adx_threshold: int = 25

    # Trailing Stop
    trailing_stop_pct: float = 0.15  # 15% trailing stop

    # Max hold period (days)
    max_hold_days: int = 60

    # Backtest period
    days_back: int = 2555  # 7 years


# =============================================================================
# INDICATOR CALCULATIONS
# =============================================================================
def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Calculate ATR"""
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]

    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))

    atr = np.zeros_like(tr)
    atr[:period] = np.nan
    if len(tr) >= period:
        atr[period-1] = np.mean(tr[:period])
        alpha = 2 / (period + 1)
        for i in range(period, len(tr)):
            atr[i] = tr[i] * alpha + atr[i-1] * (1 - alpha)

    return atr


def calculate_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate RSI"""
    deltas = np.diff(close, prepend=close[0])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.zeros(len(close))
    avg_loss = np.zeros(len(close))

    avg_gain[period] = np.mean(gains[1:period+1])
    avg_loss[period] = np.mean(losses[1:period+1])

    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i]) / period

    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    rsi[:period+1] = 50

    return rsi


def calculate_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate ADX (Average Directional Index)"""
    n = len(close)
    tr = calculate_atr(high, low, close, period)

    # +DM and -DM
    high_diff = np.diff(high, prepend=high[0])
    low_diff = np.diff(-low, prepend=-low[0])

    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)

    # Smooth DM
    smooth_plus_dm = np.zeros(n)
    smooth_minus_dm = np.zeros(n)
    smooth_plus_dm[period] = np.sum(plus_dm[1:period+1])
    smooth_minus_dm[period] = np.sum(minus_dm[1:period+1])

    for i in range(period + 1, n):
        smooth_plus_dm[i] = smooth_plus_dm[i-1] - (smooth_plus_dm[i-1] / period) + plus_dm[i]
        smooth_minus_dm[i] = smooth_minus_dm[i-1] - (smooth_minus_dm[i-1] / period) + minus_dm[i]

    # +DI and -DI
    plus_di = np.where(tr != 0, 100 * smooth_plus_dm / tr, 0)
    minus_di = np.where(tr != 0, 100 * smooth_minus_dm / tr, 0)

    # DX
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)

    # ADX
    adx = np.zeros(n)
    adx[2*period] = np.mean(dx[period+1:2*period+1])

    for i in range(2*period + 1, n):
        adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period

    return adx


def calculate_sma(data: np.ndarray, period: int) -> np.ndarray:
    """Calculate Simple Moving Average"""
    sma = np.zeros_like(data)
    for i in range(period - 1, len(data)):
        sma[i] = np.mean(data[i-period+1:i+1])
    return sma


def calculate_supertrend(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                         period: int = 14, multiplier: float = 4.0) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Supertrend"""
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    hl2 = (high + low) / 2

    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr

    final_upper = np.zeros(n)
    final_lower = np.zeros(n)
    supertrend = np.zeros(n)
    direction = np.zeros(n)

    final_upper[period-1] = upper[period-1]
    final_lower[period-1] = lower[period-1]

    for i in range(period, n):
        final_upper[i] = upper[i] if upper[i] < final_upper[i-1] or close[i-1] > final_upper[i-1] else final_upper[i-1]
        final_lower[i] = lower[i] if lower[i] > final_lower[i-1] or close[i-1] < final_lower[i-1] else final_lower[i-1]

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
# HIGH PROFIT FACTOR TRADING SYSTEM
# =============================================================================
class HighPFTradingSystem:
    def __init__(self, config: HighPFConfig):
        self.config = config

    def calculate_all_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate all required indicators"""
        symbol = self.config.symbol
        close = df[f'Close_{symbol}'].values
        high = df[f'High_{symbol}'].values
        low = df[f'Low_{symbol}'].values
        volume = df[f'Volume_{symbol}'].values

        indicators = {
            'close': close,
            'high': high,
            'low': low,
            'volume': volume,
            'supertrend': calculate_supertrend(high, low, close, self.config.st_period, self.config.st_multiplier),
            'rsi': calculate_rsi(close, self.config.rsi_period),
            'adx': calculate_adx(high, low, close, self.config.adx_period),
            'sma200': calculate_sma(close, self.config.trend_ma_period),
            'volume_ma': calculate_sma(volume, self.config.volume_ma_period),
            'atr': calculate_atr(high, low, close, 14)
        }

        return indicators

    def generate_high_pf_signals(self, indicators: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate signals with multiple confirmations for high profit factor

        Entry conditions (ALL must be true):
        1. Supertrend gives buy signal (direction changes to 1)
        2. Price is above 200 SMA (long-term uptrend)
        3. RSI is between oversold and overbought (not extreme)
        4. ADX > threshold (strong trend)
        5. Volume is above average (confirmation)

        Exit conditions:
        1. Supertrend gives sell signal OR
        2. Price drops below trailing stop OR
        3. Max hold period reached
        """
        n = len(indicators['close'])
        supertrend, direction = indicators['supertrend']
        close = indicators['close']
        rsi = indicators['rsi']
        adx = indicators['adx']
        sma200 = indicators['sma200']
        volume = indicators['volume']
        volume_ma = indicators['volume_ma']

        buy_signals = np.zeros(n, dtype=bool)
        sell_signals = np.zeros(n, dtype=bool)

        # Detect direction changes
        prev_direction = np.roll(direction, 1)
        prev_direction[0] = 0

        raw_buy = (prev_direction == -1) & (direction == 1)
        raw_sell = (prev_direction == 1) & (direction == -1)

        # Apply filters for entry
        for i in range(self.config.trend_ma_period, n):
            if raw_buy[i]:
                # Check all entry conditions
                above_sma200 = close[i] > sma200[i]
                rsi_ok = self.config.rsi_oversold < rsi[i] < self.config.rsi_overbought
                adx_strong = adx[i] > self.config.adx_threshold
                volume_high = volume[i] > volume_ma[i] * self.config.volume_threshold

                # Require at least 3 of 4 conditions (relaxed but still selective)
                conditions_met = sum([above_sma200, rsi_ok, adx_strong, volume_high])

                if conditions_met >= 3 and above_sma200:  # Must be in uptrend
                    buy_signals[i] = True

            if raw_sell[i]:
                sell_signals[i] = True

        return buy_signals, sell_signals

    def run_backtest(self, df: pd.DataFrame) -> Dict:
        """Run backtest with trailing stop and high PF rules"""
        symbol = self.config.symbol
        indicators = self.calculate_all_indicators(df)

        buy_signals, sell_signals = self.generate_high_pf_signals(indicators)

        close = indicators['close']
        high = indicators['high']
        dates = df.index

        trades = []
        position = None
        entry_price = 0
        entry_date = None
        entry_idx = 0
        highest_price = 0

        for i in range(len(df)):
            if position == 'long':
                # Update trailing stop
                highest_price = max(highest_price, high[i])
                trailing_stop = highest_price * (1 - self.config.trailing_stop_pct)

                # Check exits
                days_held = (dates[i] - entry_date).days if entry_date else 0

                # Exit conditions
                exit_signal = sell_signals[i]
                trailing_stop_hit = close[i] < trailing_stop
                max_hold_reached = days_held >= self.config.max_hold_days

                if exit_signal or trailing_stop_hit or max_hold_reached:
                    profit_loss = (close[i] - entry_price) / entry_price
                    profit_loss -= self.config.transaction_cost * 2

                    exit_reason = 'signal' if exit_signal else ('trailing_stop' if trailing_stop_hit else 'max_hold')

                    trades.append({
                        'entry_date': entry_date,
                        'entry_price': entry_price,
                        'exit_date': dates[i],
                        'exit_price': close[i],
                        'profit_loss': profit_loss,
                        'profit_loss_pct': profit_loss * 100,
                        'days_held': days_held,
                        'exit_reason': exit_reason,
                        'highest_price': highest_price
                    })

                    position = None
                    highest_price = 0

            # Entry
            if buy_signals[i] and position is None:
                position = 'long'
                entry_price = close[i]
                entry_date = dates[i]
                entry_idx = i
                highest_price = high[i]

        # Calculate statistics
        stats = self.calculate_statistics(trades)

        return {
            'trades': trades,
            'stats': stats,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'indicators': indicators
        }

    def calculate_statistics(self, trades: List[Dict]) -> Dict:
        """Calculate trading statistics"""
        if not trades:
            return self._empty_stats()

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

        # PROFIT FACTOR - the key metric!
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

        # Other metrics
        total_return = sum(t['profit_loss'] for t in trades)

        # Expectancy
        expectancy = (win_rate * avg_profit) - ((1 - win_rate) * abs(avg_loss))

        # Average holding period
        avg_hold = np.mean([t['days_held'] for t in trades])

        # Best and worst trade
        best_trade = max(t['profit_loss'] for t in trades)
        worst_trade = min(t['profit_loss'] for t in trades)

        # Consecutive wins/losses
        results = [1 if t['profit_loss'] > 0 else 0 for t in trades]
        max_consecutive_wins = self._max_consecutive(results, 1)
        max_consecutive_losses = self._max_consecutive(results, 0)

        return {
            'Total Trades': total_trades,
            'Winning Trades': winning_trades,
            'Losing Trades': losing_trades,
            'Win Rate': f"{win_rate:.1%}",
            'Avg Profit': f"{avg_profit:.2%}",
            'Avg Loss': f"{avg_loss:.2%}",
            'PROFIT FACTOR': profit_factor,
            'Profit Factor Display': f"{profit_factor:.2f}" if profit_factor != float('inf') else "INF",
            'Total Return': f"{total_return:.2%}",
            'Expectancy': f"{expectancy:.2%}",
            'Avg Hold Days': f"{avg_hold:.1f}",
            'Best Trade': f"{best_trade:.2%}",
            'Worst Trade': f"{worst_trade:.2%}",
            'Max Consecutive Wins': max_consecutive_wins,
            'Max Consecutive Losses': max_consecutive_losses
        }

    def _max_consecutive(self, results: List, value: int) -> int:
        """Find max consecutive occurrences of value"""
        max_count = 0
        current_count = 0
        for r in results:
            if r == value:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
        return max_count

    def _empty_stats(self) -> Dict:
        return {
            'Total Trades': 0,
            'Winning Trades': 0,
            'Losing Trades': 0,
            'Win Rate': "0.0%",
            'Avg Profit': "0.00%",
            'Avg Loss': "0.00%",
            'PROFIT FACTOR': 0,
            'Profit Factor Display': "0.00",
            'Total Return': "0.00%",
            'Expectancy': "0.00%",
            'Avg Hold Days': "0.0",
            'Best Trade': "0.00%",
            'Worst Trade': "0.00%",
            'Max Consecutive Wins': 0,
            'Max Consecutive Losses': 0
        }


# =============================================================================
# PARAMETER OPTIMIZATION FOR HIGH PF
# =============================================================================
def optimize_for_high_pf(symbol: str = "MSFT", days_back: int = 2555):
    """
    Optimize parameters specifically for maximum Profit Factor

    Strategy: Find parameters that produce:
    1. High win rate (>80%)
    2. Large average win vs small average loss
    3. Few but high-quality trades
    """
    print("="*70)
    print("ULTRA-HIGH PROFIT FACTOR OPTIMIZATION")
    print("="*70)
    print(f"Symbol: {symbol}")
    print(f"Target: Profit Factor > 10 (aiming for 50+)")
    print()

    # Download data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    data = yf.download(symbol, start=start_date, end=end_date, progress=False)

    if data.empty:
        print("Error: No data")
        return None

    # Flatten columns
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ['_'.join(col).strip() for col in data.columns.values]
    else:
        data.columns = [f'{col}_{symbol}' for col in data.columns]

    print(f"Data: {len(data)} days ({data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')})")

    best_pf = 0
    best_config = None
    best_result = None

    # Parameter ranges - optimized for high PF
    st_periods = [10, 12, 14, 16, 18, 20]
    st_multipliers = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
    trailing_stops = [0.10, 0.12, 0.15, 0.18, 0.20]
    adx_thresholds = [20, 25, 30]
    volume_thresholds = [1.0, 1.2, 1.5]

    total_combos = len(st_periods) * len(st_multipliers) * len(trailing_stops) * len(adx_thresholds) * len(volume_thresholds)
    tested = 0

    print(f"\nTesting {total_combos} parameter combinations...")
    print()

    results = []

    for st_period in st_periods:
        for st_mult in st_multipliers:
            for trail_stop in trailing_stops:
                for adx_thresh in adx_thresholds:
                    for vol_thresh in volume_thresholds:
                        tested += 1

                        config = HighPFConfig(
                            symbol=symbol,
                            days_back=days_back,
                            st_period=st_period,
                            st_multiplier=st_mult,
                            trailing_stop_pct=trail_stop,
                            adx_threshold=adx_thresh,
                            volume_threshold=vol_thresh
                        )

                        try:
                            system = HighPFTradingSystem(config)
                            result = system.run_backtest(data)

                            trades = result['trades']
                            stats = result['stats']

                            if len(trades) >= 3:  # Minimum trades for validity
                                pf = stats['PROFIT FACTOR']

                                results.append({
                                    'st_period': st_period,
                                    'st_mult': st_mult,
                                    'trail_stop': trail_stop,
                                    'adx_thresh': adx_thresh,
                                    'vol_thresh': vol_thresh,
                                    'profit_factor': pf,
                                    'trades': len(trades),
                                    'win_rate': stats['Win Rate'],
                                    'total_return': stats['Total Return']
                                })

                                if pf > best_pf and pf != float('inf'):
                                    best_pf = pf
                                    best_config = config
                                    best_result = result

                        except Exception as e:
                            pass

                        if tested % 100 == 0:
                            print(f"Progress: {tested}/{total_combos} ({100*tested/total_combos:.0f}%) | Best PF: {best_pf:.2f}")

    # Sort results by PF
    results = [r for r in results if r['profit_factor'] != float('inf')]
    results.sort(key=lambda x: x['profit_factor'], reverse=True)

    print("\n" + "="*70)
    print("TOP 10 CONFIGURATIONS BY PROFIT FACTOR")
    print("="*70)

    for i, r in enumerate(results[:10], 1):
        print(f"{i}. PF={r['profit_factor']:.2f} | "
              f"Period={r['st_period']}, Mult={r['st_mult']:.1f}, "
              f"Trail={r['trail_stop']:.0%}, ADX>{r['adx_thresh']}, Vol>{r['vol_thresh']:.1f}x | "
              f"Trades={r['trades']}, WinRate={r['win_rate']}, Return={r['total_return']}")

    if best_result:
        print("\n" + "="*70)
        print("BEST CONFIGURATION DETAILS")
        print("="*70)
        print(f"\nParameters:")
        print(f"  Supertrend Period: {best_config.st_period}")
        print(f"  Supertrend Multiplier: {best_config.st_multiplier}")
        print(f"  Trailing Stop: {best_config.trailing_stop_pct:.0%}")
        print(f"  ADX Threshold: {best_config.adx_threshold}")
        print(f"  Volume Threshold: {best_config.volume_threshold}x")

        print(f"\nStatistics:")
        for key, value in best_result['stats'].items():
            if key != 'PROFIT FACTOR':
                print(f"  {key}: {value}")
            else:
                print(f"  >>> PROFIT FACTOR: {value:.2f} <<<")

        print(f"\nTrade History:")
        for i, trade in enumerate(best_result['trades'], 1):
            sign = "+" if trade['profit_loss'] > 0 else ""
            print(f"  {i}. {trade['entry_date'].strftime('%Y-%m-%d')} -> "
                  f"{trade['exit_date'].strftime('%Y-%m-%d')} | "
                  f"{sign}{trade['profit_loss']:.2%} | "
                  f"{trade['exit_reason']}")

    return best_config, best_result, results


# =============================================================================
# EXTREME HIGH PF STRATEGY (Target: >50)
# =============================================================================
def run_extreme_pf_strategy(symbol: str = "MSFT"):
    """
    Run extreme high PF strategy with very strict filters

    To achieve PF > 50, we need:
    - Win rate > 95% OR
    - Avg win >> avg loss (like 50:1 ratio)

    This requires VERY selective trading - possibly only 1-5 trades per year
    """
    print("="*70)
    print("EXTREME HIGH PROFIT FACTOR STRATEGY")
    print("Target: Profit Factor > 50")
    print("="*70)

    config = HighPFConfig(
        symbol=symbol,
        days_back=3650,  # 10 years
        st_period=20,
        st_multiplier=5.0,
        trailing_stop_pct=0.20,  # Wide trailing stop
        adx_threshold=30,  # Strong trends only
        volume_threshold=1.5,  # High volume confirmation
        max_hold_days=90,  # Hold longer for bigger moves
        rsi_oversold=30,
        rsi_overbought=70
    )

    # Download data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=config.days_back)
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
    print(f"Period: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")

    # Run system
    system = HighPFTradingSystem(config)
    result = system.run_backtest(data)

    print("\n" + "-"*70)
    print("RESULTS")
    print("-"*70)

    for key, value in result['stats'].items():
        if 'PROFIT FACTOR' in key.upper():
            print(f">>> {key}: {value} <<<")
        else:
            print(f"{key}: {value}")

    print("\n" + "-"*70)
    print("TRADE HISTORY")
    print("-"*70)

    for i, trade in enumerate(result['trades'], 1):
        sign = "+" if trade['profit_loss'] > 0 else ""
        status = "WIN" if trade['profit_loss'] > 0 else "LOSS"
        print(f"{i}. [{status}] {trade['entry_date'].strftime('%Y-%m-%d')} -> "
              f"{trade['exit_date'].strftime('%Y-%m-%d')} | "
              f"{sign}{trade['profit_loss']:.2%} ({trade['days_held']} days) | "
              f"{trade['exit_reason']}")

    return result


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--optimize":
        symbol = sys.argv[2] if len(sys.argv) > 2 else "MSFT"
        optimize_for_high_pf(symbol)
    elif len(sys.argv) > 1 and sys.argv[1] == "--extreme":
        symbol = sys.argv[2] if len(sys.argv) > 2 else "MSFT"
        run_extreme_pf_strategy(symbol)
    else:
        # Default: Run optimization
        print("Usage:")
        print("  python high_pf_strategy.py --optimize [SYMBOL]")
        print("  python high_pf_strategy.py --extreme [SYMBOL]")
        print()
        print("Running optimization with default symbol MSFT...")
        optimize_for_high_pf("MSFT")
