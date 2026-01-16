#!/usr/bin/env python3
"""
Strategy Parameter Optimizer
=============================
Optimizes indicator parameters for each stock by backtesting.

Parameters to optimize:
- SUPERTREND: st_period (10-20), st_multiplier (2-5)
- TREND_FOLLOW: ema_fast (10-30), ema_slow (40-60)
- Trailing Stop: 5%-25%
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
import logging
from itertools import product

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

CATEGORIES_FILE = "stock_categories.json"


# =============================================================================
# INDICATORS (copied from ib_paper_trader for standalone use)
# =============================================================================
def calculate_atr(high, low, close, period=14):
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    true_range = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = np.zeros_like(true_range)
    atr[:period] = np.nan
    if len(true_range) >= period:
        atr[period-1] = np.mean(true_range[:period])
        multiplier = 2 / (period + 1)
        for i in range(period, len(true_range)):
            atr[i] = true_range[i] * multiplier + atr[i-1] * (1 - multiplier)
    return atr


def calculate_supertrend(df: pd.DataFrame, period: int = 15, multiplier: float = 4.0) -> pd.DataFrame:
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values

    atr = calculate_atr(high, low, close, period)
    hl2 = (high + low) / 2

    basic_upper = hl2 + (multiplier * atr)
    basic_lower = hl2 - (multiplier * atr)

    n = len(close)
    final_upper = np.zeros(n)
    final_lower = np.zeros(n)
    supertrend = np.zeros(n)
    direction = np.zeros(n)

    final_upper[0] = basic_upper[0]
    final_lower[0] = basic_lower[0]

    for i in range(1, n):
        if basic_upper[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]:
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = final_upper[i-1]

        if basic_lower[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]:
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = final_lower[i-1]

        if i < period:
            direction[i] = 1
            supertrend[i] = final_lower[i]
        else:
            if supertrend[i-1] == final_upper[i-1]:
                if close[i] > final_upper[i]:
                    direction[i] = 1
                    supertrend[i] = final_lower[i]
                else:
                    direction[i] = -1
                    supertrend[i] = final_upper[i]
            else:
                if close[i] < final_lower[i]:
                    direction[i] = -1
                    supertrend[i] = final_upper[i]
                else:
                    direction[i] = 1
                    supertrend[i] = final_lower[i]

    df = df.copy()
    df['direction'] = direction
    return df


def calculate_ema(close: np.ndarray, period: int) -> np.ndarray:
    n = len(close)
    ema = np.zeros(n)
    if n < period:
        ema[:] = np.nan
        return ema
    ema[:period] = np.nan
    ema[period-1] = np.mean(close[:period])
    multiplier = 2 / (period + 1)
    for i in range(period, n):
        ema[i] = close[i] * multiplier + ema[i-1] * (1 - multiplier)
    return ema


def calculate_ema_crossover(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.DataFrame:
    close = df['close'].values
    ema_fast = calculate_ema(close, fast)
    ema_slow = calculate_ema(close, slow)
    df = df.copy()
    df['direction'] = np.where(ema_fast > ema_slow, 1, -1)
    return df


# =============================================================================
# BACKTEST ENGINE
# =============================================================================
def backtest_strategy(df: pd.DataFrame, strategy: str, params: dict,
                      trailing_stop: float = 0.12) -> dict:
    """
    Backtest a strategy with given parameters on a single stock.
    Returns performance metrics.
    """
    if len(df) < 100:
        return None

    # Calculate signals
    if strategy == 'SUPERTREND':
        df = calculate_supertrend(df, params.get('period', 15), params.get('multiplier', 4.0))
    elif strategy == 'EMA':
        df = calculate_ema_crossover(df, params.get('fast', 20), params.get('slow', 50))
    else:
        return None

    close = df['close'].values
    direction = df['direction'].values

    # Simulate trading
    position = 0
    entry_price = 0
    high_price = 0
    cash = 10000
    trades = []
    equity_curve = [cash]

    for i in range(60, len(close)):
        price = close[i]

        if position == 0:
            # No position - check for BUY signal
            if direction[i] == 1:
                position = cash / price
                entry_price = price
                high_price = price
                cash = 0
        else:
            # In position
            high_price = max(high_price, price)

            # Check trailing stop
            stop_price = high_price * (1 - trailing_stop)

            # Check for SELL signal or stop
            if direction[i] == -1 or price <= stop_price:
                cash = position * price
                pnl = (price - entry_price) / entry_price
                trades.append(pnl)
                position = 0

        # Record equity
        equity = cash + position * price if position > 0 else cash
        equity_curve.append(equity)

    # Close final position
    if position > 0:
        cash = position * close[-1]
        pnl = (close[-1] - entry_price) / entry_price
        trades.append(pnl)

    if len(trades) == 0:
        return None

    # Calculate metrics
    equity_curve = np.array(equity_curve)
    returns = np.diff(equity_curve) / equity_curve[:-1]

    total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]

    # Max drawdown
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / peak
    max_dd = np.max(drawdown)

    # Sharpe ratio
    if np.std(returns) > 0:
        sharpe = np.sqrt(252) * np.mean(returns) / np.std(returns)
    else:
        sharpe = 0

    # Win rate
    winners = [t for t in trades if t > 0]
    win_rate = len(winners) / len(trades) if trades else 0

    # Buy & Hold comparison
    bh_return = (close[-1] - close[60]) / close[60]

    return {
        'total_return': total_return,
        'max_drawdown': max_dd,
        'sharpe': sharpe,
        'win_rate': win_rate,
        'num_trades': len(trades),
        'vs_bh': total_return - bh_return,
        'bh_return': bh_return
    }


def optimize_supertrend(df: pd.DataFrame, symbol: str) -> dict:
    """Find optimal Supertrend parameters for a stock."""
    best_result = None
    best_params = None
    best_score = -999

    # Parameter grid
    periods = [10, 12, 15, 18, 20]
    multipliers = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    trailing_stops = [0.08, 0.10, 0.12, 0.15, 0.18, 0.20]

    for period, mult, ts in product(periods, multipliers, trailing_stops):
        params = {'period': period, 'multiplier': mult}
        result = backtest_strategy(df, 'SUPERTREND', params, ts)

        if result is None:
            continue

        # Score: Sharpe ratio + bonus for beating B&H
        score = result['sharpe'] + (0.5 if result['vs_bh'] > 0 else 0)

        if score > best_score:
            best_score = score
            best_result = result
            best_params = {
                'st_period': period,
                'st_multiplier': mult,
                'trailing_stop_pct': ts
            }

    return {
        'symbol': symbol,
        'params': best_params,
        'result': best_result,
        'score': best_score
    }


def optimize_ema(df: pd.DataFrame, symbol: str) -> dict:
    """Find optimal EMA crossover parameters for a stock."""
    best_result = None
    best_params = None
    best_score = -999

    # Parameter grid
    fast_periods = [10, 15, 20, 25, 30]
    slow_periods = [40, 50, 60, 70]
    trailing_stops = [0.10, 0.12, 0.15, 0.18, 0.20]

    for fast, slow, ts in product(fast_periods, slow_periods, trailing_stops):
        if fast >= slow:
            continue

        params = {'fast': fast, 'slow': slow}
        result = backtest_strategy(df, 'EMA', params, ts)

        if result is None:
            continue

        score = result['sharpe'] + (0.5 if result['vs_bh'] > 0 else 0)

        if score > best_score:
            best_score = score
            best_result = result
            best_params = {
                'ema_fast': fast,
                'ema_slow': slow,
                'trailing_stop_pct': ts
            }

    return {
        'symbol': symbol,
        'params': best_params,
        'result': best_result,
        'score': best_score
    }


def fetch_data_ib(symbols: List[str]) -> Dict[str, pd.DataFrame]:
    """Fetch 1 year of data from IB."""
    try:
        from ib_insync import IB, Stock, util
        from ib_paper_trader import get_ticker_contract_params
    except ImportError:
        logger.error("ib_insync not available")
        return {}

    data = {}
    ib = IB()

    try:
        ib.connect('127.0.0.1', 7497, clientId=52)
        logger.info(f"Connected to IB. Fetching data for {len(symbols)} symbols...")

        for symbol in symbols:
            try:
                exchange, currency = get_ticker_contract_params(symbol)
                contract = Stock(symbol, exchange, currency)
                ib.qualifyContracts(contract)

                bars = ib.reqHistoricalData(
                    contract,
                    endDateTime='',
                    durationStr='1 Y',
                    barSizeSetting='1 day',
                    whatToShow='TRADES',
                    useRTH=True
                )

                if bars:
                    df = util.df(bars)
                    df.columns = [c.lower() for c in df.columns]
                    df.set_index('date', inplace=True)
                    data[symbol] = df
                    logger.info(f"  {symbol}: {len(df)} days")

                ib.sleep(0.3)

            except Exception as e:
                logger.warning(f"  {symbol}: Failed - {e}")

        ib.disconnect()

    except Exception as e:
        logger.error(f"Could not connect to IB: {e}")

    return data


def load_config():
    with open(CATEGORIES_FILE, 'r') as f:
        return json.load(f)


def save_config(config):
    config['_last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M')
    with open(CATEGORIES_FILE, 'w') as f:
        json.dump(config, f, indent=4)
    logger.info(f"Config saved to {CATEGORIES_FILE}")


def main():
    print("="*70)
    print("STRATEGY PARAMETER OPTIMIZER")
    print("="*70)

    config = load_config()

    # Get all tickers by strategy
    supertrend_tickers = config['strategies'].get('SUPERTREND', {}).get('tickers', [])
    german_tickers = config['strategies'].get('GERMAN', {}).get('tickers', [])
    trend_follow_tickers = config['strategies'].get('TREND_FOLLOW', {}).get('tickers', [])
    buyhold_tickers = config['strategies'].get('BUY_HOLD', {}).get('tickers', [])

    all_tickers = supertrend_tickers + german_tickers + trend_follow_tickers + buyhold_tickers

    print(f"\nOptimizing {len(all_tickers)} stocks:")
    print(f"  SUPERTREND: {len(supertrend_tickers)}")
    print(f"  GERMAN: {len(german_tickers)}")
    print(f"  TREND_FOLLOW: {len(trend_follow_tickers)}")
    print(f"  BUY_HOLD: {len(buyhold_tickers)}")

    # Fetch data
    print("\n" + "-"*70)
    print("FETCHING DATA FROM IB...")
    print("-"*70)

    data = fetch_data_ib(all_tickers)

    if len(data) == 0:
        print("No data available. Make sure TWS is running.")
        return

    print(f"\nLoaded data for {len(data)} symbols")

    # Optimize SUPERTREND stocks
    print("\n" + "-"*70)
    print("OPTIMIZING SUPERTREND PARAMETERS...")
    print("-"*70)

    st_results = []
    for symbol in supertrend_tickers + german_tickers:
        if symbol not in data:
            continue
        print(f"\n{symbol}:", end=" ")
        result = optimize_supertrend(data[symbol], symbol)
        if result['params']:
            st_results.append(result)
            print(f"Period={result['params']['st_period']}, "
                  f"Mult={result['params']['st_multiplier']}, "
                  f"TS={result['params']['trailing_stop_pct']*100:.0f}% "
                  f"-> Return={result['result']['total_return']*100:+.1f}%, "
                  f"vs B&H={result['result']['vs_bh']*100:+.1f}%")
        else:
            print("No valid parameters found")

    # Optimize TREND_FOLLOW stocks
    print("\n" + "-"*70)
    print("OPTIMIZING TREND_FOLLOW (EMA) PARAMETERS...")
    print("-"*70)

    ema_results = []
    for symbol in trend_follow_tickers:
        if symbol not in data:
            continue
        print(f"\n{symbol}:", end=" ")
        result = optimize_ema(data[symbol], symbol)
        if result['params']:
            ema_results.append(result)
            print(f"EMA {result['params']['ema_fast']}/{result['params']['ema_slow']}, "
                  f"TS={result['params']['trailing_stop_pct']*100:.0f}% "
                  f"-> Return={result['result']['total_return']*100:+.1f}%, "
                  f"vs B&H={result['result']['vs_bh']*100:+.1f}%")
        else:
            print("No valid parameters found")

    # Optimize BUY_HOLD trailing stops
    print("\n" + "-"*70)
    print("OPTIMIZING BUY_HOLD TRAILING STOPS...")
    print("-"*70)

    bh_results = []
    for symbol in buyhold_tickers:
        if symbol not in data:
            continue
        df = data[symbol]
        close = df['close'].values

        best_ts = 0.20
        best_return = -999

        for ts in [0.15, 0.18, 0.20, 0.22, 0.25]:
            # Simulate trailing stop
            position = 10000 / close[60]
            high_price = close[60]

            for i in range(61, len(close)):
                high_price = max(high_price, close[i])
                stop_price = high_price * (1 - ts)

                if close[i] <= stop_price:
                    # Stopped out - re-enter after 5 days
                    position = (position * close[i]) / close[min(i+5, len(close)-1)]
                    high_price = close[min(i+5, len(close)-1)]

            final_value = position * close[-1]
            total_return = (final_value - 10000) / 10000

            if total_return > best_return:
                best_return = total_return
                best_ts = ts

        bh_return = (close[-1] - close[60]) / close[60]
        bh_results.append({
            'symbol': symbol,
            'trailing_stop': best_ts,
            'return': best_return,
            'bh_return': bh_return
        })
        print(f"{symbol}: TS={best_ts*100:.0f}% -> Return={best_return*100:+.1f}% (B&H={bh_return*100:+.1f}%)")

    # Summary
    print("\n" + "="*70)
    print("OPTIMIZATION SUMMARY")
    print("="*70)

    # Average best parameters for SUPERTREND
    if st_results:
        avg_period = np.mean([r['params']['st_period'] for r in st_results])
        avg_mult = np.mean([r['params']['st_multiplier'] for r in st_results])
        avg_ts = np.mean([r['params']['trailing_stop_pct'] for r in st_results])

        print(f"\nSUPERTREND Optimal (average):")
        print(f"  Period: {avg_period:.0f}")
        print(f"  Multiplier: {avg_mult:.1f}")
        print(f"  Trailing Stop: {avg_ts*100:.0f}%")

        # Update config
        config['strategies']['SUPERTREND']['settings'] = {
            'st_period': int(round(avg_period)),
            'st_multiplier': round(avg_mult, 1),
            'trailing_stop_pct': round(avg_ts, 2)
        }
        if 'GERMAN' in config['strategies']:
            config['strategies']['GERMAN']['settings'] = config['strategies']['SUPERTREND']['settings'].copy()

    # Average best parameters for TREND_FOLLOW
    if ema_results:
        avg_fast = np.mean([r['params']['ema_fast'] for r in ema_results])
        avg_slow = np.mean([r['params']['ema_slow'] for r in ema_results])
        avg_ts = np.mean([r['params']['trailing_stop_pct'] for r in ema_results])

        print(f"\nTREND_FOLLOW Optimal (average):")
        print(f"  EMA Fast: {avg_fast:.0f}")
        print(f"  EMA Slow: {avg_slow:.0f}")
        print(f"  Trailing Stop: {avg_ts*100:.0f}%")

        config['strategies']['TREND_FOLLOW']['settings'] = {
            'ema_fast': int(round(avg_fast)),
            'ema_slow': int(round(avg_slow)),
            'trailing_stop_pct': round(avg_ts, 2)
        }

    # Average trailing stop for BUY_HOLD
    if bh_results:
        avg_ts = np.mean([r['trailing_stop'] for r in bh_results])
        print(f"\nBUY_HOLD Optimal Trailing Stop: {avg_ts*100:.0f}%")

        config['strategies']['BUY_HOLD']['settings']['trailing_stop_pct'] = round(avg_ts, 2)

    # Save updated config
    print("\n" + "-"*70)
    save_answer = input("Save optimized parameters to config? (y/n): ")
    if save_answer.lower() == 'y':
        save_config(config)
        print("Parameters saved!")
    else:
        print("Parameters not saved.")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
