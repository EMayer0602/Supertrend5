#!/usr/bin/env python3
"""
Stock Strategy Categorizer
===========================
Tests EVERY stock with EVERY strategy (SUPERTREND, BUY_HOLD, TREND_FOLLOW)
and assigns each stock to its BEST performing strategy.

Output:
- Performance comparison for each stock
- Optimal strategy assignment
- Updated stock_categories.json
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================
INITIAL_CAPITAL = 10000  # Per stock test
SIMULATION_DAYS = 252    # 1 year trading days
FEE_PER_TRADE = 1.0

CATEGORIES_FILE = "stock_categories.json"

# Strategy settings
STRATEGY_SETTINGS = {
    'SUPERTREND': {
        'st_period': 10,
        'st_multiplier': 2.0,
        'trailing_stop_pct': 0.12
    },
    'BUY_HOLD': {
        'trailing_stop_pct': 0.20,
        'reentry_after_days': 5
    },
    'TREND_FOLLOW': {
        'ema_fast': 18,
        'ema_slow': 55,
        'trailing_stop_pct': 0.10
    }
}


def get_all_tickers() -> List[str]:
    """Get all tickers from current config"""
    with open(CATEGORIES_FILE, 'r') as f:
        config = json.load(f)

    all_tickers = []
    for strat_name, strat_data in config.get('strategies', {}).items():
        if strat_name != 'EXCLUDED':
            all_tickers.extend(strat_data.get('tickers', []))

    return list(set(all_tickers))


def get_ticker_contract_params(symbol: str) -> Tuple[str, str]:
    """Get exchange and currency for a ticker"""
    with open(CATEGORIES_FILE, 'r') as f:
        config = json.load(f)

    ticker_settings = config.get('ticker_settings', {})
    if symbol in ticker_settings:
        return (
            ticker_settings[symbol].get('exchange', 'SMART'),
            ticker_settings[symbol].get('currency', 'USD')
        )
    return ('SMART', 'USD')


# =============================================================================
# INDICATORS
# =============================================================================
def calculate_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Supertrend indicator"""
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    n = len(close)

    # ATR calculation
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1]))

    atr = np.zeros(n)
    atr[:period] = np.nan
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period

    # Supertrend
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    supertrend = np.zeros(n)
    direction = np.ones(n)  # 1 = uptrend, -1 = downtrend

    for i in range(period, n):
        if close[i] > upper_band[i-1]:
            direction[i] = 1
        elif close[i] < lower_band[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            if direction[i] == 1 and lower_band[i] < lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
            if direction[i] == -1 and upper_band[i] > upper_band[i-1]:
                upper_band[i] = upper_band[i-1]

        supertrend[i] = lower_band[i] if direction[i] == 1 else upper_band[i]

    return supertrend, direction


def calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """Calculate EMA"""
    n = len(prices)
    ema = np.zeros(n)

    if n < period:
        ema[:] = np.nan
        return ema

    ema[:period] = np.nan
    ema[period-1] = np.mean(prices[:period])

    multiplier = 2 / (period + 1)
    for i in range(period, n):
        ema[i] = (prices[i] * multiplier) + (ema[i-1] * (1 - multiplier))

    return ema


# =============================================================================
# STRATEGY SIGNALS
# =============================================================================
def get_supertrend_signal(df: pd.DataFrame, settings: dict) -> str:
    """Get Supertrend signal"""
    if len(df) < 60:
        return "HOLD"

    period = settings.get('st_period', 10)
    multiplier = settings.get('st_multiplier', 2.0)

    supertrend, direction = calculate_supertrend(df, period, multiplier)

    if len(direction) < 2:
        return "HOLD"

    current_dir = direction[-1]
    prev_dir = direction[-2]

    if current_dir == 1 and prev_dir == -1:
        return "BUY"
    elif current_dir == -1 and prev_dir == 1:
        return "SELL"
    elif current_dir == 1:
        return "BUY"  # Uptrend - stay long
    else:
        return "SELL"  # Downtrend - stay out


def get_buyhold_signal(df: pd.DataFrame, settings: dict) -> str:
    """Buy and Hold - always BUY (trailing stop managed separately)"""
    return "BUY"


def get_trendfollow_signal(df: pd.DataFrame, settings: dict) -> str:
    """EMA crossover signal"""
    if len(df) < 60:
        return "HOLD"

    ema_fast = settings.get('ema_fast', 18)
    ema_slow = settings.get('ema_slow', 55)

    close = df['close'].values
    fast = calculate_ema(close, ema_fast)
    slow = calculate_ema(close, ema_slow)

    if np.isnan(fast[-1]) or np.isnan(slow[-1]):
        return "HOLD"

    if fast[-1] > slow[-1]:
        return "BUY"
    else:
        return "SELL"


def get_signal(df: pd.DataFrame, strategy: str, settings: dict) -> str:
    """Get signal for a specific strategy"""
    if strategy == 'SUPERTREND':
        return get_supertrend_signal(df, settings)
    elif strategy == 'BUY_HOLD':
        return get_buyhold_signal(df, settings)
    elif strategy == 'TREND_FOLLOW':
        return get_trendfollow_signal(df, settings)
    return "HOLD"


# =============================================================================
# BACKTESTER
# =============================================================================
def backtest_stock_strategy(df: pd.DataFrame, strategy: str, settings: dict) -> dict:
    """Backtest a single stock with a single strategy"""
    capital = INITIAL_CAPITAL
    position = 0
    entry_price = 0
    high_price = 0
    trades = []
    equity_curve = []

    trailing_stop_pct = settings.get('trailing_stop_pct', 0.15)
    reentry_cooldown = 0
    reentry_days = settings.get('reentry_after_days', 0)

    dates = df.index[-SIMULATION_DAYS:] if len(df) > SIMULATION_DAYS else df.index

    for i, date in enumerate(dates):
        idx = df.index.get_loc(date)
        if idx < 60:
            continue

        df_slice = df.iloc[:idx+1]
        price = df.loc[date, 'close']

        # Update equity
        if position > 0:
            current_value = capital + position * price
            high_price = max(high_price, price)

            # Check trailing stop
            stop_price = high_price * (1 - trailing_stop_pct)
            if price <= stop_price:
                # Hit trailing stop - sell
                capital += position * price - FEE_PER_TRADE
                pnl = (price - entry_price) * position - 2 * FEE_PER_TRADE
                trades.append({
                    'entry_price': entry_price,
                    'exit_price': price,
                    'pnl': pnl,
                    'pnl_pct': (price / entry_price - 1) * 100,
                    'reason': 'TRAILING_STOP'
                })
                position = 0
                entry_price = 0
                high_price = 0
                reentry_cooldown = reentry_days
        else:
            current_value = capital
            if reentry_cooldown > 0:
                reentry_cooldown -= 1

        # Get signal
        signal = get_signal(df_slice, strategy, settings)

        # Execute trades
        if signal == "BUY" and position == 0 and reentry_cooldown == 0:
            # Buy
            quantity = int((capital - FEE_PER_TRADE) / price)
            if quantity > 0:
                position = quantity
                entry_price = price
                high_price = price
                capital -= quantity * price + FEE_PER_TRADE

        elif signal == "SELL" and position > 0:
            # Sell
            capital += position * price - FEE_PER_TRADE
            pnl = (price - entry_price) * position - 2 * FEE_PER_TRADE
            trades.append({
                'entry_price': entry_price,
                'exit_price': price,
                'pnl': pnl,
                'pnl_pct': (price / entry_price - 1) * 100,
                'reason': 'SIGNAL'
            })
            position = 0
            entry_price = 0
            high_price = 0

        # Record equity
        equity = capital + position * price
        equity_curve.append(equity)

    # Close any open position at end
    if position > 0:
        final_price = df.iloc[-1]['close']
        capital += position * final_price
        pnl = (final_price - entry_price) * position - FEE_PER_TRADE
        trades.append({
            'entry_price': entry_price,
            'exit_price': final_price,
            'pnl': pnl,
            'pnl_pct': (final_price / entry_price - 1) * 100,
            'reason': 'END'
        })

    final_equity = capital

    # Calculate stats
    total_return = final_equity - INITIAL_CAPITAL
    total_return_pct = (final_equity / INITIAL_CAPITAL - 1) * 100

    # Max drawdown
    max_dd = 0
    peak = INITIAL_CAPITAL
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak
        max_dd = max(max_dd, dd)

    # Win rate
    winners = [t for t in trades if t['pnl'] > 0]
    win_rate = len(winners) / len(trades) * 100 if trades else 0

    # Sharpe
    if len(equity_curve) > 1:
        returns = pd.Series(equity_curve).pct_change().dropna()
        sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
    else:
        sharpe = 0

    return {
        'strategy': strategy,
        'final_equity': final_equity,
        'total_return': total_return,
        'total_return_pct': total_return_pct,
        'max_drawdown_pct': max_dd * 100,
        'sharpe': sharpe,
        'num_trades': len(trades),
        'win_rate': win_rate
    }


# =============================================================================
# DATA FETCHING
# =============================================================================
def fetch_data_ib(symbols: List[str]) -> Dict[str, pd.DataFrame]:
    """Fetch data from Interactive Brokers"""
    try:
        from ib_insync import IB, Stock, util
    except ImportError:
        logger.warning("ib_insync not installed")
        return {}

    data = {}
    ib = IB()

    try:
        ib.connect('127.0.0.1', 7497, clientId=53)
        logger.info(f"Connected to IB. Fetching {len(symbols)} symbols...")

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

                ib.sleep(0.5)
            except Exception as e:
                logger.warning(f"  {symbol}: {e}")

        ib.disconnect()
    except Exception as e:
        logger.error(f"IB connection failed: {e}")

    return data


def generate_synthetic_data(symbols: List[str], days: int = 400) -> Dict[str, pd.DataFrame]:
    """Generate synthetic data for testing"""
    np.random.seed(42)
    data = {}

    base_prices = {
        'NVDA': 500, 'AMD': 140, 'AVGO': 180, 'META': 520, 'TSLA': 250,
        'COIN': 250, 'MSTR': 450, 'PLTR': 70, 'SHOP': 100, 'UBER': 75,
        'CRWD': 350, 'MU': 100, 'JPM': 200, 'INOD': 180, 'QUBT': 15,
        'DRH': 10, 'MRNA': 50, 'MRK': 100, 'NFLX': 900, 'NKE': 75,
        'PFE': 25, 'PYPL': 85, 'PDYN': 40, 'QBTS': 8, 'TKMS': 30,
        'JNJ': 150, 'TGT': 130, 'UNH': 550, 'SPY': 580, 'QQQ': 500,
        'GOOGL': 175, 'AAPL': 190, 'AMZN': 185, 'MSFT': 420, 'CRM': 280,
        'ORCL': 130, 'NOW': 780, 'ADBE': 550, 'PANW': 320, 'SNOW': 180,
        'V': 280, 'MA': 480, 'LLY': 780, 'BABA': 85,
        'SMCI': 600, 'ARM': 140, 'RIVN': 15, 'LCID': 3, 'NIO': 5,
        'SOFI': 10, 'HOOD': 20, 'AFRM': 45, 'UPST': 35, 'DKNG': 40,
        'IWM': 210, 'DIA': 390, 'XLF': 42, 'XLK': 210, 'VTI': 270, 'VOO': 530
    }

    # Different behavior types for stocks
    # Trending stocks benefit from BUY_HOLD
    trending = ['NVDA', 'AVGO', 'META', 'PLTR', 'CRWD', 'GOOGL', 'AAPL', 'MSFT', 'LLY', 'NOW']
    # Volatile/sideways stocks benefit from SUPERTREND
    volatile = ['COIN', 'MSTR', 'QUBT', 'QBTS', 'MRNA', 'RIVN', 'LCID', 'NIO', 'UPST', 'AFRM']
    # ETFs and stable stocks benefit from TREND_FOLLOW
    stable = ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'XLF', 'XLK', 'JPM', 'JNJ']

    end_date = datetime.now()
    dates = pd.date_range(end=end_date, periods=days, freq='B')

    for symbol in symbols:
        start_price = base_prices.get(symbol, 100)

        # Set volatility and drift based on stock type
        if symbol in trending:
            vol = 0.025
            drift = 0.0008  # Strong uptrend
        elif symbol in volatile:
            vol = 0.05
            drift = 0.0001  # Sideways with high vol
        elif symbol in stable:
            vol = 0.012
            drift = 0.0004  # Moderate trend, low vol
        else:
            vol = 0.03
            drift = 0.0004

        returns = np.random.normal(drift, vol, days)
        prices = start_price * np.exp(np.cumsum(returns))

        df = pd.DataFrame(index=dates)
        df['close'] = prices
        df['high'] = df['close'] * (1 + np.abs(np.random.normal(0, vol*0.5, days)))
        df['low'] = df['close'] * (1 - np.abs(np.random.normal(0, vol*0.5, days)))
        df['open'] = df['close'].shift(1).fillna(start_price)
        df['high'] = df[['high', 'close', 'open']].max(axis=1)
        df['low'] = df[['low', 'close', 'open']].min(axis=1)
        df['volume'] = np.random.randint(1000000, 10000000, days)

        data[symbol] = df

    return data


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("="*80)
    print("         STOCK STRATEGY CATEGORIZER")
    print("         Test all stocks with all strategies")
    print("="*80)

    # Get all tickers
    tickers = get_all_tickers()
    print(f"\nTotal stocks to analyze: {len(tickers)}")

    # Separate German stocks (they stay in GERMAN category)
    german_stocks = ['TKMS']
    us_stocks = [t for t in tickers if t not in german_stocks]

    print(f"US stocks: {len(us_stocks)}")
    print(f"German stocks: {len(german_stocks)}")

    # Fetch data
    print("\n" + "-"*80)
    print("  FETCHING DATA...")
    print("-"*80)

    data = fetch_data_ib(us_stocks)

    if len(data) < 10:
        print("  Using synthetic data (IB not available)")
        data = generate_synthetic_data(us_stocks)

    print(f"\n  Loaded data for {len(data)} symbols")

    # Test each stock with each strategy
    print("\n" + "-"*80)
    print("  TESTING STRATEGIES...")
    print("-"*80)

    results = {}
    strategies = ['SUPERTREND', 'BUY_HOLD', 'TREND_FOLLOW']

    for symbol in data.keys():
        df = data[symbol]
        if len(df) < 100:
            continue

        results[symbol] = {}
        for strat in strategies:
            settings = STRATEGY_SETTINGS[strat]
            result = backtest_stock_strategy(df, strat, settings)
            results[symbol][strat] = result

        # Find best strategy
        best_strat = max(strategies, key=lambda s: results[symbol][s]['total_return_pct'])
        results[symbol]['best'] = best_strat

        # Print progress
        st_ret = results[symbol]['SUPERTREND']['total_return_pct']
        bh_ret = results[symbol]['BUY_HOLD']['total_return_pct']
        tf_ret = results[symbol]['TREND_FOLLOW']['total_return_pct']

        print(f"  {symbol:<8} ST:{st_ret:>+7.1f}%  BH:{bh_ret:>+7.1f}%  TF:{tf_ret:>+7.1f}%  -> {best_strat}")

    # Categorize stocks
    print("\n" + "="*80)
    print("  OPTIMAL STRATEGY ASSIGNMENT")
    print("="*80)

    categorized = {
        'SUPERTREND': [],
        'BUY_HOLD': [],
        'TREND_FOLLOW': [],
        'GERMAN': german_stocks.copy()
    }

    for symbol, res in results.items():
        best = res['best']
        categorized[best].append(symbol)

    # Print summary
    for strat, tickers in categorized.items():
        print(f"\n  {strat} ({len(tickers)} stocks):")
        for i in range(0, len(tickers), 10):
            chunk = tickers[i:i+10]
            print(f"    {', '.join(chunk)}")

    # Show detailed comparison
    print("\n" + "="*80)
    print("  DETAILED PERFORMANCE COMPARISON")
    print("="*80)
    print(f"\n  {'Symbol':<8} {'Best':<12} {'ST Return':>12} {'BH Return':>12} {'TF Return':>12} {'Diff':>10}")
    print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")

    # Sort by difference between best and worst
    sorted_results = sorted(results.items(),
                           key=lambda x: max(x[1][s]['total_return_pct'] for s in strategies) -
                                        min(x[1][s]['total_return_pct'] for s in strategies),
                           reverse=True)

    for symbol, res in sorted_results:
        st = res['SUPERTREND']['total_return_pct']
        bh = res['BUY_HOLD']['total_return_pct']
        tf = res['TREND_FOLLOW']['total_return_pct']
        best = res['best']
        diff = max(st, bh, tf) - min(st, bh, tf)
        print(f"  {symbol:<8} {best:<12} {st:>+11.1f}% {bh:>+11.1f}% {tf:>+11.1f}% {diff:>+9.1f}%")

    # Update config file
    print("\n" + "-"*80)
    print("  UPDATING stock_categories.json...")
    print("-"*80)

    with open(CATEGORIES_FILE, 'r') as f:
        config = json.load(f)

    # Update tickers in each strategy
    config['strategies']['SUPERTREND']['tickers'] = sorted(categorized['SUPERTREND'])
    config['strategies']['BUY_HOLD']['tickers'] = sorted(categorized['BUY_HOLD'])
    config['strategies']['TREND_FOLLOW']['tickers'] = sorted(categorized['TREND_FOLLOW'])
    config['strategies']['GERMAN']['tickers'] = sorted(categorized['GERMAN'])
    config['_last_updated'] = datetime.now().strftime('%Y-%m-%d')
    config['_comment'] = f"Auto-categorized {len(results)} stocks based on 1Y backtest"

    with open(CATEGORIES_FILE, 'w') as f:
        json.dump(config, f, indent=4)

    print(f"  Updated {CATEGORIES_FILE}")
    print(f"  - SUPERTREND: {len(categorized['SUPERTREND'])} stocks")
    print(f"  - BUY_HOLD: {len(categorized['BUY_HOLD'])} stocks")
    print(f"  - TREND_FOLLOW: {len(categorized['TREND_FOLLOW'])} stocks")
    print(f"  - GERMAN: {len(categorized['GERMAN'])} stocks")

    # Summary stats
    print("\n" + "="*80)
    print("  STRATEGY SUMMARY")
    print("="*80)

    for strat in strategies:
        strat_returns = [results[s][strat]['total_return_pct'] for s in results]
        avg_return = np.mean(strat_returns)
        print(f"\n  {strat}:")
        print(f"    Avg Return across all stocks: {avg_return:+.1f}%")
        print(f"    Stocks assigned: {len(categorized[strat])}")

        if categorized[strat]:
            assigned_returns = [results[s][strat]['total_return_pct'] for s in categorized[strat] if s in results]
            if assigned_returns:
                print(f"    Avg Return for assigned: {np.mean(assigned_returns):+.1f}%")

    print("\n" + "="*80)
    print("  DONE")
    print("="*80)


if __name__ == "__main__":
    main()
