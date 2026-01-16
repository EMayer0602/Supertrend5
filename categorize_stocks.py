#!/usr/bin/env python3
"""
Stock Strategy Categorizer
===========================
Tests EVERY stock with 6 strategies:
- SUPERTREND: Supertrend indicator
- KAMA: Kaufman Adaptive Moving Average crossover
- JMA: Jurik Moving Average crossover
- TREND_FOLLOW: EMA 18/55 crossover
- SMA: Simple Moving Average crossover
- EMA: EMA 12/26 crossover

Assigns each stock to its BEST performing strategy.
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
    'KAMA': {
        'kama_period': 10,
        'kama_fast': 2,
        'kama_slow': 30,
        'signal_period': 10,
        'trailing_stop_pct': 0.12
    },
    'JMA': {
        'jma_period': 7,
        'jma_phase': 50,
        'signal_period': 21,
        'trailing_stop_pct': 0.12
    },
    'TREND_FOLLOW': {
        'ema_fast': 18,
        'ema_slow': 55,
        'trailing_stop_pct': 0.10
    },
    'SMA': {
        'sma_fast': 20,
        'sma_slow': 50,
        'trailing_stop_pct': 0.12
    },
    'EMA': {
        'ema_fast': 12,
        'ema_slow': 26,
        'trailing_stop_pct': 0.12
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


def calculate_sma(prices: np.ndarray, period: int) -> np.ndarray:
    """Calculate Simple Moving Average"""
    n = len(prices)
    sma = np.zeros(n)

    if n < period:
        sma[:] = np.nan
        return sma

    sma[:period-1] = np.nan
    for i in range(period-1, n):
        sma[i] = np.mean(prices[i-period+1:i+1])

    return sma


def calculate_kama(prices: np.ndarray, period: int = 10, fast: int = 2, slow: int = 30) -> np.ndarray:
    """Calculate Kaufman Adaptive Moving Average (KAMA)"""
    n = len(prices)
    kama = np.zeros(n)

    if n < period + 1:
        kama[:] = np.nan
        return kama

    kama[:period] = np.nan

    # Smoothing constants
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)

    # Initialize KAMA
    kama[period] = prices[period]

    for i in range(period + 1, n):
        # Efficiency Ratio (ER)
        change = abs(prices[i] - prices[i - period])
        volatility = sum(abs(prices[j] - prices[j-1]) for j in range(i - period + 1, i + 1))

        if volatility != 0:
            er = change / volatility
        else:
            er = 0

        # Smoothing Constant (SC)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

        # KAMA
        kama[i] = kama[i-1] + sc * (prices[i] - kama[i-1])

    return kama


def calculate_jma(prices: np.ndarray, period: int = 7, phase: int = 50) -> np.ndarray:
    """Calculate Jurik Moving Average (JMA) - simplified approximation"""
    n = len(prices)
    jma = np.zeros(n)

    if n < period:
        jma[:] = np.nan
        return jma

    # Phase adjustment (-100 to +100)
    phase_ratio = phase / 100.0
    beta = 0.45 * (period - 1) / (0.45 * (period - 1) + 2)

    # Initialize
    jma[:period] = np.nan
    e0 = prices[period-1]
    e1 = 0
    e2 = 0

    for i in range(period, n):
        price = prices[i]

        # JMA calculation with phase
        e0 = (1 - beta) * price + beta * e0
        e1 = (price - e0) * (1 - beta) + beta * e1
        e2 = (e0 + phase_ratio * e1 - jma[i-1]) * (1 - beta) ** 2 + beta ** 2 * e2

        jma[i] = jma[i-1] + e2 if i > period else e0 + phase_ratio * e1

    return jma


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


def get_kama_signal(df: pd.DataFrame, settings: dict) -> str:
    """KAMA crossover signal - KAMA vs Signal line"""
    if len(df) < 60:
        return "HOLD"

    period = settings.get('kama_period', 10)
    fast = settings.get('kama_fast', 2)
    slow = settings.get('kama_slow', 30)
    signal_period = settings.get('signal_period', 10)

    close = df['close'].values
    kama = calculate_kama(close, period, fast, slow)
    signal = calculate_sma(kama, signal_period)

    if np.isnan(kama[-1]) or np.isnan(signal[-1]):
        return "HOLD"

    # KAMA above signal = bullish
    if kama[-1] > signal[-1]:
        return "BUY"
    else:
        return "SELL"


def get_jma_signal(df: pd.DataFrame, settings: dict) -> str:
    """JMA crossover signal - JMA vs Signal line"""
    if len(df) < 60:
        return "HOLD"

    period = settings.get('jma_period', 7)
    phase = settings.get('jma_phase', 50)
    signal_period = settings.get('signal_period', 21)

    close = df['close'].values
    jma = calculate_jma(close, period, phase)
    signal = calculate_sma(jma, signal_period)

    if np.isnan(jma[-1]) or np.isnan(signal[-1]):
        return "HOLD"

    # JMA above signal = bullish
    if jma[-1] > signal[-1]:
        return "BUY"
    else:
        return "SELL"


def get_trendfollow_signal(df: pd.DataFrame, settings: dict) -> str:
    """EMA 18/55 crossover signal"""
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


def get_sma_signal(df: pd.DataFrame, settings: dict) -> str:
    """SMA crossover signal"""
    if len(df) < 60:
        return "HOLD"

    sma_fast = settings.get('sma_fast', 20)
    sma_slow = settings.get('sma_slow', 50)

    close = df['close'].values
    fast = calculate_sma(close, sma_fast)
    slow = calculate_sma(close, sma_slow)

    if np.isnan(fast[-1]) or np.isnan(slow[-1]):
        return "HOLD"

    if fast[-1] > slow[-1]:
        return "BUY"
    else:
        return "SELL"


def get_ema_signal(df: pd.DataFrame, settings: dict) -> str:
    """EMA 12/26 crossover signal"""
    if len(df) < 60:
        return "HOLD"

    ema_fast = settings.get('ema_fast', 12)
    ema_slow = settings.get('ema_slow', 26)

    close = df['close'].values
    fast = calculate_ema(close, ema_fast)
    slow = calculate_ema(close, ema_slow)

    if np.isnan(fast[-1]) or np.isnan(slow[-1]):
        return "HOLD"

    if fast[-1] > slow[-1]:
        return "BUY"
    else:
        return "SELL"


def check_htf_filter(df: pd.DataFrame, period: int = 200) -> bool:
    """HTF Filter: Price above 200 SMA = bullish trend"""
    if len(df) < period:
        return True  # Not enough data, allow trades

    close = df['close'].values
    sma200 = calculate_sma(close, period)

    if np.isnan(sma200[-1]):
        return True

    # Price above SMA200 = bullish, allow buys
    return close[-1] > sma200[-1]


def get_signal(df: pd.DataFrame, strategy: str, settings: dict, use_htf: bool = False) -> str:
    """Get signal for a specific strategy, optionally with HTF filter"""
    # Get base signal
    base_strategy = strategy.replace('_HTF', '')

    if base_strategy == 'SUPERTREND':
        signal = get_supertrend_signal(df, settings)
    elif base_strategy == 'KAMA':
        signal = get_kama_signal(df, settings)
    elif base_strategy == 'JMA':
        signal = get_jma_signal(df, settings)
    elif base_strategy == 'TREND_FOLLOW':
        signal = get_trendfollow_signal(df, settings)
    elif base_strategy == 'SMA':
        signal = get_sma_signal(df, settings)
    elif base_strategy == 'EMA':
        signal = get_ema_signal(df, settings)
    else:
        signal = "HOLD"

    # Apply HTF filter if enabled
    if use_htf and signal == "BUY":
        if not check_htf_filter(df):
            return "HOLD"  # Block buy if below 200 SMA

    return signal


# =============================================================================
# BACKTESTER
# =============================================================================
def backtest_stock_strategy(df: pd.DataFrame, strategy: str, settings: dict, use_htf: bool = False) -> dict:
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

        # Get signal (with optional HTF filter)
        signal = get_signal(df_slice, strategy, settings, use_htf)

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
    print("="*100)
    print("         STOCK STRATEGY CATEGORIZER - 6 Strategies + HTF Filter")
    print("         Tests: SUPERTREND, KAMA, JMA, TREND_FOLLOW, SMA, EMA (each with/without HTF)")
    print("="*100)

    # Get all tickers
    tickers = get_all_tickers()
    print(f"\nTotal stocks to analyze: {len(tickers)}")

    # Separate German stocks (they stay in GERMAN category)
    german_stocks = ['TKMS']
    us_stocks = [t for t in tickers if t not in german_stocks]

    print(f"US stocks: {len(us_stocks)}")
    print(f"German stocks: {len(german_stocks)}")

    # Fetch data
    print("\n" + "-"*100)
    print("  FETCHING DATA...")
    print("-"*100)

    data = fetch_data_ib(us_stocks)

    if len(data) < 10:
        print("  Using synthetic data (IB not available)")
        data = generate_synthetic_data(us_stocks)

    print(f"\n  Loaded data for {len(data)} symbols")

    # Define all strategy combinations (6 base + 6 with HTF = 12 total)
    base_strategies = ['SUPERTREND', 'KAMA', 'JMA', 'TREND_FOLLOW', 'SMA', 'EMA']
    all_strategies = []
    for strat in base_strategies:
        all_strategies.append((strat, False))  # Without HTF
        all_strategies.append((strat + '_HTF', True))  # With HTF filter

    # Test each stock with each strategy
    print("\n" + "-"*100)
    print("  TESTING 12 STRATEGY COMBINATIONS...")
    print("-"*100)

    results = {}

    for symbol in data.keys():
        df = data[symbol]
        if len(df) < 100:
            continue

        results[symbol] = {}

        for strat_name, use_htf in all_strategies:
            base_strat = strat_name.replace('_HTF', '')
            settings = STRATEGY_SETTINGS.get(base_strat, STRATEGY_SETTINGS['SUPERTREND'])
            result = backtest_stock_strategy(df, base_strat, settings, use_htf)
            result['strategy'] = strat_name
            results[symbol][strat_name] = result

        # Find best strategy by final equity
        best_strat = max([s[0] for s in all_strategies],
                        key=lambda s: results[symbol][s]['final_equity'])
        results[symbol]['best'] = best_strat
        results[symbol]['best_return'] = results[symbol][best_strat]['total_return_pct']

        # Print progress (compact format)
        best_ret = results[symbol][best_strat]['total_return_pct']
        print(f"  {symbol:<8} Best: {best_strat:<15} Return: {best_ret:>+7.1f}%")

    # Categorize stocks by best strategy
    print("\n" + "="*100)
    print("  OPTIMAL STRATEGY ASSIGNMENT (by Final Equity)")
    print("="*100)

    categorized = {}
    for strat_name, _ in all_strategies:
        categorized[strat_name] = []
    categorized['GERMAN'] = german_stocks.copy()

    for symbol, res in results.items():
        best = res['best']
        categorized[best].append(symbol)

    # Print summary by strategy
    for strat_name in sorted(categorized.keys()):
        tickers_list = categorized[strat_name]
        if tickers_list:
            print(f"\n  {strat_name} ({len(tickers_list)} stocks):")
            for i in range(0, len(tickers_list), 8):
                chunk = tickers_list[i:i+8]
                print(f"    {', '.join(chunk)}")

    # Show detailed comparison table
    print("\n" + "="*100)
    print("  DETAILED PERFORMANCE COMPARISON (all 12 strategies)")
    print("="*100)

    # Header
    header = f"  {'Symbol':<8} {'BEST':<15}"
    for strat in base_strategies:
        header += f" {strat[:6]:>8}"
    header += "  HTF+"
    for strat in base_strategies:
        header += f" {strat[:6]:>8}"
    print(header)
    print("  " + "-"*96)

    # Sort by best return
    sorted_results = sorted(results.items(),
                           key=lambda x: x[1]['best_return'],
                           reverse=True)

    for symbol, res in sorted_results:
        line = f"  {symbol:<8} {res['best']:<15}"
        # Without HTF
        for strat in base_strategies:
            ret = res[strat]['total_return_pct']
            line += f" {ret:>+7.1f}%"
        line += "  |"
        # With HTF
        for strat in base_strategies:
            ret = res[strat + '_HTF']['total_return_pct']
            line += f" {ret:>+7.1f}%"
        print(line)

    # Strategy performance summary
    print("\n" + "="*100)
    print("  STRATEGY PERFORMANCE SUMMARY")
    print("="*100)
    print(f"\n  {'Strategy':<20} {'Avg Return':>12} {'Stocks':>10} {'Best For':>10}")
    print(f"  {'-'*20} {'-'*12} {'-'*10} {'-'*10}")

    for strat_name, _ in all_strategies:
        returns = [results[s][strat_name]['total_return_pct'] for s in results]
        avg_ret = np.mean(returns) if returns else 0
        count = len(categorized.get(strat_name, []))
        print(f"  {strat_name:<20} {avg_ret:>+11.1f}% {count:>10} {count:>10}")

    # Update config file with simplified categories
    print("\n" + "-"*100)
    print("  UPDATING stock_categories.json...")
    print("-"*100)

    with open(CATEGORIES_FILE, 'r') as f:
        config = json.load(f)

    # Create new strategy entries for all 12 strategies
    new_strategies = {}

    for strat_name, use_htf in all_strategies:
        base_strat = strat_name.replace('_HTF', '')
        base_settings = STRATEGY_SETTINGS.get(base_strat, {}).copy()
        if use_htf:
            base_settings['htf_filter'] = True
            base_settings['htf_period'] = 200

        new_strategies[strat_name] = {
            'description': f"{base_strat} {'with HTF filter (200 SMA)' if use_htf else 'without HTF filter'}",
            'settings': base_settings,
            'tickers': sorted(categorized.get(strat_name, []))
        }

    # Keep GERMAN separate
    new_strategies['GERMAN'] = {
        'description': 'German stocks (IBIS/EUR)',
        'settings': STRATEGY_SETTINGS['SUPERTREND'].copy(),
        'tickers': german_stocks
    }

    # Keep EXCLUDED
    new_strategies['EXCLUDED'] = config.get('strategies', {}).get('EXCLUDED', {'tickers': []})

    config['strategies'] = new_strategies
    config['_last_updated'] = datetime.now().strftime('%Y-%m-%d')
    config['_comment'] = f"Auto-categorized {len(results)} stocks with 12 strategies (6 base + 6 HTF)"

    with open(CATEGORIES_FILE, 'w') as f:
        json.dump(config, f, indent=4)

    print(f"  Updated {CATEGORIES_FILE}")

    # Count summary
    total_assigned = 0
    for strat_name, _ in all_strategies:
        count = len(categorized.get(strat_name, []))
        if count > 0:
            print(f"  - {strat_name}: {count} stocks")
            total_assigned += count
    print(f"  - GERMAN: {len(german_stocks)} stocks")
    print(f"\n  Total: {total_assigned + len(german_stocks)} stocks categorized")

    # Export results to CSV
    print("\n" + "-"*100)
    print("  EXPORTING RESULTS...")
    print("-"*100)

    export_data = []
    for symbol, res in results.items():
        row = {'Symbol': symbol, 'Best_Strategy': res['best']}
        for strat_name, _ in all_strategies:
            row[f'{strat_name}_Return'] = res[strat_name]['total_return_pct']
            row[f'{strat_name}_Equity'] = res[strat_name]['final_equity']
        export_data.append(row)

    df_export = pd.DataFrame(export_data)
    df_export = df_export.sort_values('Best_Strategy')
    export_file = 'strategy_comparison.csv'
    df_export.to_csv(export_file, index=False)
    print(f"  Results saved to: {export_file}")

    print("\n" + "="*100)
    print("  DONE - Run simulation_report.py to test the new categorization")
    print("="*100)


if __name__ == "__main__":
    main()
