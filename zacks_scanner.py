#!/usr/bin/env python3
"""
Zacks-Style Portfolio Scanner
Scans for stocks with high volatility and positive momentum.
Filters for Zacks Rank 1-2 stocks (pre-defined list).

Uses IB TWS for price data.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import argparse
from typing import List, Dict, Optional
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

def get_ib_connection(host: str = '127.0.0.1', port: int = 7497, client_id: int = 25) -> IB:
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


def download_history(symbol: str, days: int = 180) -> Optional[pd.DataFrame]:
    """Download historical data from TWS"""
    try:
        ib = get_ib_connection()

        contract = Stock(symbol, 'SMART', 'USD')
        ib.qualifyContracts(contract)

        bars = ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr=f'{days} D',
            barSizeSetting='1 day',
            whatToShow='TRADES',
            useRTH=True,
            formatDate=1
        )

        if not bars:
            return None

        df = util.df(bars)
        df.set_index('date', inplace=True)

        # Small delay to avoid pacing violations
        ib.sleep(0.3)

        return df
    except Exception as e:
        print(f"\nError downloading {symbol}: {e}")
        return None


# =============================================================================
# PRE-DEFINED ZACKS RANK 1-2 STOCKS
# Updated from https://www.zacks.com/stocks/zacks-rank
# =============================================================================

# Zacks Rank 1 = Strong Buy
ZACKS_RANK_1 = [
    # Tech - Strong Buy
    'NVDA', 'AVGO', 'AMD', 'MRVL', 'ON', 'ANET', 'CRWD', 'PANW', 'FTNT',
    # Semiconductors
    'KLAC', 'LRCX', 'AMAT', 'NXPI', 'ADI', 'MCHP',
    # Cloud/Software
    'NOW', 'SNOW', 'DDOG', 'NET', 'ZS',
    # E-commerce/Retail
    'AMZN', 'COST', 'ORLY', 'ULTA', 'DECK',
    # Healthcare/Biotech
    'LLY', 'VRTX', 'REGN', 'ISRG', 'DXCM',
    # Finance
    'V', 'MA', 'AXP', 'COIN',
    # Energy
    'FANG', 'DVN', 'EOG',
    # Other Growth
    'META', 'GOOGL', 'NFLX', 'BKNG', 'ABNB'
]

# Zacks Rank 2 = Buy
ZACKS_RANK_2 = [
    # Tech - Buy
    'AAPL', 'MSFT', 'CSCO', 'QCOM', 'TXN', 'MU', 'INTC',
    # Software
    'ADBE', 'CRM', 'INTU', 'ADSK', 'SNPS', 'CDNS',
    # Internet
    'GOOG', 'PYPL', 'SQ', 'SHOP', 'MELI',
    # Consumer
    'TSLA', 'NKE', 'SBUX', 'CMG',
    # Healthcare
    'PFE', 'ABBV', 'MRK', 'AMGN', 'GILD',
    # Industrial
    'HON', 'CAT', 'DE', 'GE',
    # Comm
    'TMUS', 'T', 'VZ'
]

# All high-volatility NASDAQ stocks (for scanning)
NASDAQ_VOLATILE = [
    'NVDA', 'AMD', 'TSLA', 'META', 'AMZN', 'GOOGL', 'NFLX', 'AVGO',
    'MRVL', 'ON', 'MU', 'AMAT', 'LRCX', 'KLAC', 'NXPI', 'ADI',
    'CRWD', 'PANW', 'ZS', 'FTNT', 'DDOG', 'NET', 'SNOW', 'NOW',
    'COIN', 'SQ', 'PYPL', 'SHOP', 'MELI', 'ABNB',
    'ENPH', 'SEDG', 'FSLR', 'PLUG', 'RIVN', 'LCID',
    'ROKU', 'ZM', 'DOCU', 'OKTA', 'PLTR', 'SOFI', 'HOOD', 'UPST',
    'SMCI', 'ARM', 'CELH', 'AXON', 'DECK', 'ULTA'
]


def calculate_metrics(df: pd.DataFrame) -> dict:
    """Calculate volatility metrics for a stock."""
    if df is None or len(df) < 20:
        return None

    try:
        high = df['high']
        low = df['low']
        close = df['close']

        # ATR%
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]

        current_price = close.iloc[-1]
        if current_price <= 0:
            return None

        atr_pct = (atr / current_price) * 100

        # Annual volatility
        returns = close.pct_change().dropna()
        annual_vol = returns.std() * np.sqrt(252) * 100

        # Momentum (3-month return)
        if len(close) >= 63:
            momentum = ((close.iloc[-1] / close.iloc[-63]) - 1) * 100
        else:
            momentum = ((close.iloc[-1] / close.iloc[0]) - 1) * 100

        # 20-day return (short-term momentum)
        if len(close) >= 20:
            momentum_20d = ((close.iloc[-1] / close.iloc[-20]) - 1) * 100
        else:
            momentum_20d = momentum

        return {
            'price': round(float(current_price), 2),
            'atr_pct': round(float(atr_pct), 2),
            'annual_vol': round(float(annual_vol), 1),
            'momentum_3m': round(float(momentum), 1),
            'momentum_20d': round(float(momentum_20d), 1)
        }
    except Exception as e:
        return None


def scan_portfolio(
    symbols: List[str] = None,
    min_atr_pct: float = 2.0,
    min_price: float = 10.0,
    max_price: float = 500.0,
    top_n: int = 30,
    min_momentum: float = -10.0,
    zacks_only: bool = True
) -> List[dict]:
    """
    Scan for high-volatility stocks with good momentum.

    Args:
        symbols: List of symbols to scan
        min_atr_pct: Minimum ATR%
        min_price: Minimum stock price
        max_price: Maximum stock price
        top_n: Number of top stocks to return
        min_momentum: Minimum 3-month momentum %
        zacks_only: Only include Zacks Rank 1-2 stocks

    Returns:
        List of qualifying stocks with metrics
    """
    if symbols is None:
        if zacks_only:
            symbols = list(set(ZACKS_RANK_1 + ZACKS_RANK_2))
            print(f"Using Zacks Rank 1-2 stocks: {len(symbols)} symbols")
        else:
            symbols = list(set(ZACKS_RANK_1 + ZACKS_RANK_2 + NASDAQ_VOLATILE))
            print(f"Using all high-volatility stocks: {len(symbols)} symbols")

    results = []
    total = len(symbols)

    print(f"\nScanning {total} symbols for ATR% >= {min_atr_pct} and Momentum >= {min_momentum}%...")
    print("-" * 60)

    for i, symbol in enumerate(symbols, 1):
        print(f"\r[{i}/{total}] Downloading {symbol}...", end='', flush=True)

        try:
            # Download historical data
            df = download_history(symbol, days=180)

            if df is None or len(df) < 20:
                continue

            # Calculate metrics
            metrics = calculate_metrics(df)
            if not metrics:
                continue

            price = metrics['price']

            # Price filter
            if price < min_price or price > max_price:
                continue

            # ATR% filter
            if metrics['atr_pct'] < min_atr_pct:
                continue

            # Momentum filter
            if metrics['momentum_3m'] < min_momentum:
                continue

            # Determine Zacks rank
            if symbol in ZACKS_RANK_1:
                rank = 1
                rank_label = 'Strong Buy'
            elif symbol in ZACKS_RANK_2:
                rank = 2
                rank_label = 'Buy'
            else:
                rank = 3
                rank_label = 'Hold'

            # Calculate score
            # Higher rank (Strong Buy) + higher volatility + positive momentum
            rank_score = 4 - rank  # 3 for Strong Buy, 2 for Buy, 1 for Hold
            vol_score = min(metrics['atr_pct'], 8) / 8  # Normalize 0-1
            mom_score = 1 + (metrics['momentum_3m'] / 100)

            combined_score = rank_score * vol_score * mom_score * 10

            results.append({
                'symbol': symbol,
                'price': price,
                'rank': rank,
                'rank_label': rank_label,
                'atr_pct': metrics['atr_pct'],
                'annual_vol': metrics['annual_vol'],
                'momentum_3m': metrics['momentum_3m'],
                'momentum_20d': metrics['momentum_20d'],
                'score': round(combined_score, 2)
            })

        except Exception as e:
            continue

    print(f"\n\nFound {len(results)} stocks matching criteria")

    # Sort by score
    results.sort(key=lambda x: x['score'], reverse=True)

    return results[:top_n]


def save_results(results: List[dict], filename: str = 'zacks_tickers.json'):
    """Save results to JSON file."""
    output = {
        'scan_date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'criteria': 'Zacks Rank 1-2 + High Volatility + Positive Momentum',
        'count': len(results),
        'tickers': [r['symbol'] for r in results],
        'details': results
    }

    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved {len(results)} tickers to {filename}")


def print_results(results: List[dict]):
    """Print results in a formatted table."""
    if not results:
        print("No stocks found matching criteria.")
        return

    print("\n" + "=" * 95)
    print(f"{'Symbol':<7} {'Price':>8} {'Rank':>12} {'ATR%':>7} {'Vol%':>7} {'Mom3M':>8} {'Mom20D':>8} {'Score':>7}")
    print("=" * 95)

    for r in results:
        print(f"{r['symbol']:<7} ${r['price']:>7.2f} {r['rank_label']:>12} {r['atr_pct']:>6.2f}% {r['annual_vol']:>6.1f}% {r['momentum_3m']:>7.1f}% {r['momentum_20d']:>7.1f}% {r['score']:>7.1f}")

    print("=" * 95)
    print(f"\nRank: 1=Strong Buy, 2=Buy | Score = Rank + ATR% + Momentum")


def main():
    parser = argparse.ArgumentParser(description='Zacks Portfolio Scanner (IB)')
    parser.add_argument('--min-atr', type=float, default=2.0, help='Minimum ATR%%')
    parser.add_argument('--min-price', type=float, default=10.0, help='Minimum stock price')
    parser.add_argument('--max-price', type=float, default=500.0, help='Maximum stock price')
    parser.add_argument('--top', type=int, default=30, help='Number of top stocks to return')
    parser.add_argument('--min-momentum', type=float, default=-10.0, help='Minimum 3-month momentum %%')
    parser.add_argument('--output', type=str, default='zacks_tickers.json', help='Output filename')
    parser.add_argument('--all', action='store_true', help='Include all volatile stocks, not just Zacks 1-2')
    parser.add_argument('--port', type=int, default=7497, help='TWS port (7497=Paper, 7496=Live)')

    args = parser.parse_args()

    try:
        # Run scan
        results = scan_portfolio(
            min_atr_pct=args.min_atr,
            min_price=args.min_price,
            max_price=args.max_price,
            top_n=args.top,
            min_momentum=args.min_momentum,
            zacks_only=not args.all
        )

        # Print and save results
        print_results(results)
        save_results(results, args.output)

        # Print usage hint
        print(f"\nNachste Schritte:")
        print(f"  1. python new5.py --long-short --tickers {args.output}")
        print(f"  2. python portfolio_simulation.py 90 --take-profit 0.30 --min-return 0.25")

    finally:
        disconnect_ib()


if __name__ == '__main__':
    main()
