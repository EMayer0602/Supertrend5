#!/usr/bin/env python3
"""
Analyst Rating Portfolio Scanner
Scans for stocks with Strong Buy/Buy analyst ratings from Nasdaq.
Combines ratings with volatility and momentum metrics.

Uses:
- Nasdaq API for analyst ratings (Buy/Hold/Sell counts)
- IB TWS for price data
"""

import json
import requests
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
# NASDAQ ANALYST API
# =============================================================================

def get_nasdaq_analyst_rating(symbol: str) -> Optional[dict]:
    """
    Get analyst ratings from Nasdaq API.
    Returns buy/hold/sell counts and target price.
    """
    try:
        url = f'https://api.nasdaq.com/api/analyst/{symbol}/targetprice'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json'
        }

        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            return None

        data = response.json()
        overview = data.get('data', {}).get('consensusOverview', {})

        if not overview:
            return None

        buy = overview.get('buy', 0)
        hold = overview.get('hold', 0)
        sell = overview.get('sell', 0)
        total = buy + hold + sell

        if total == 0:
            return None

        # Calculate rank based on analyst consensus
        buy_pct = buy / total * 100

        if buy_pct >= 80:
            rank = 1  # Strong Buy (80%+ Buy ratings)
            rank_label = 'Strong Buy'
        elif buy_pct >= 60:
            rank = 2  # Buy (60-80% Buy ratings)
            rank_label = 'Buy'
        elif buy_pct >= 40:
            rank = 3  # Hold
            rank_label = 'Hold'
        else:
            rank = 4  # Sell
            rank_label = 'Sell'

        return {
            'buy': buy,
            'hold': hold,
            'sell': sell,
            'total': total,
            'buy_pct': round(buy_pct, 1),
            'rank': rank,
            'rank_label': rank_label,
            'target_price': overview.get('priceTarget', 0),
            'target_low': overview.get('lowPriceTarget', 0),
            'target_high': overview.get('highPriceTarget', 0)
        }

    except Exception as e:
        return None


def scan_nasdaq_ratings(symbols: List[str], max_rank: int = 2, min_analysts: int = 5) -> Dict[str, dict]:
    """
    Scan multiple symbols for analyst ratings via Nasdaq API.

    Args:
        symbols: List of ticker symbols
        max_rank: Maximum rank to include (1=Strong Buy, 2=Buy)
        min_analysts: Minimum number of analysts required

    Returns:
        Dict of {symbol: rating_data} for qualifying stocks
    """
    results = {}
    total = len(symbols)

    print(f"\nFetching analyst ratings for {total} symbols from Nasdaq...")
    print("-" * 60)

    for i, symbol in enumerate(symbols, 1):
        print(f"\r[{i}/{total}] Checking {symbol}...", end='', flush=True)

        rating = get_nasdaq_analyst_rating(symbol)

        if rating and rating['total'] >= min_analysts and rating['rank'] <= max_rank:
            results[symbol] = rating

        time.sleep(0.1)  # Rate limiting

    print(f"\n\nFound {len(results)} stocks with Rank <= {max_rank}")
    return results


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
# SYMBOL LISTS
# =============================================================================

# NASDAQ-100 Components
NASDAQ_100 = [
    'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'GOOG', 'META', 'TSLA', 'AVGO', 'COST',
    'NFLX', 'AMD', 'ADBE', 'PEP', 'CSCO', 'INTC', 'CMCSA', 'TMUS', 'QCOM', 'TXN',
    'AMGN', 'INTU', 'AMAT', 'ISRG', 'HON', 'BKNG', 'LRCX', 'VRTX', 'MU', 'ADI',
    'REGN', 'SBUX', 'MDLZ', 'KLAC', 'GILD', 'PANW', 'SNPS', 'CDNS', 'ASML', 'MELI',
    'PYPL', 'CRWD', 'MAR', 'CTAS', 'ORLY', 'CSX', 'MNST', 'NXPI', 'MRVL', 'WDAY',
    'ADSK', 'PCAR', 'FTNT', 'ROST', 'DXCM', 'ADP', 'CHTR', 'KDP', 'AEP', 'PAYX',
    'MCHP', 'KHC', 'CPRT', 'MRNA', 'ODFL', 'EXC', 'LULU', 'IDXX', 'FAST', 'EA',
    'CTSH', 'CSGP', 'VRSK', 'XEL', 'GEHC', 'DDOG', 'ANSS', 'FANG', 'BKR', 'TEAM',
    'ZS', 'DLTR', 'WBD', 'TTWO', 'ILMN', 'WBA', 'ALGN', 'ENPH', 'ON', 'ARM'
]

# High-volatility growth stocks
GROWTH_STOCKS = [
    'COIN', 'SQ', 'SHOP', 'ABNB', 'RIVN', 'LCID', 'PLTR', 'SOFI', 'HOOD', 'UPST',
    'SMCI', 'ROKU', 'ZM', 'DOCU', 'NET', 'SNOW', 'MDB', 'OKTA', 'BILL', 'CELH',
    'AXON', 'DECK', 'ULTA', 'LLY', 'NOW', 'CRM', 'ORCL', 'IBM', 'ACN', 'DELL'
]

def load_symbols_from_file(filename: str) -> List[str]:
    """Load symbols from a TXT file (one per line)."""
    symbols = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Handle comma-separated format (TICKER,rank)
                    ticker = line.split(',')[0].strip().upper()
                    symbols.append(ticker)
        print(f"Loaded {len(symbols)} symbols from {filename}")
        return symbols
    except FileNotFoundError:
        print(f"File not found: {filename}")
        return []
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return []


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
    analyst_ratings: Dict[str, dict],
    min_atr_pct: float = 2.0,
    min_price: float = 10.0,
    max_price: float = 500.0,
    top_n: int = 30,
    min_momentum: float = -10.0
) -> List[dict]:
    """
    Scan stocks with analyst ratings for high volatility and momentum.

    Args:
        analyst_ratings: Dict of {symbol: rating_data} from Nasdaq API
        min_atr_pct: Minimum ATR%
        min_price: Minimum stock price
        max_price: Maximum stock price
        top_n: Number of top stocks to return
        min_momentum: Minimum 3-month momentum %

    Returns:
        List of qualifying stocks with metrics
    """
    if not analyst_ratings:
        print("No stocks with Strong Buy/Buy ratings found.")
        return []

    symbols = list(analyst_ratings.keys())
    total = len(symbols)

    print(f"\nDownloading price data for {total} Strong Buy/Buy stocks from IB...")
    print("-" * 60)

    results = []

    for i, symbol in enumerate(symbols, 1):
        print(f"\r[{i}/{total}] Downloading {symbol}...", end='', flush=True)

        try:
            # Download historical data from IB
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

            # Get analyst rating data
            rating = analyst_ratings[symbol]

            # Calculate upside to target
            target = rating.get('target_price', price)
            upside = ((target / price) - 1) * 100 if price > 0 and target > 0 else 0

            # Calculate score
            rank_score = max(0, 4 - rating['rank'])  # 3 for Strong Buy, 2 for Buy
            vol_score = min(metrics['atr_pct'], 8) / 8
            mom_score = 1 + (metrics['momentum_3m'] / 100)
            upside_score = 1 + (upside / 100) if upside > 0 else 0.5

            combined_score = rank_score * vol_score * mom_score * upside_score * 10

            results.append({
                'symbol': symbol,
                'price': price,
                'rank': rating['rank'],
                'rank_label': rating['rank_label'],
                'buy_pct': rating['buy_pct'],
                'analysts': rating['total'],
                'target': round(target, 2),
                'upside_pct': round(upside, 1),
                'atr_pct': metrics['atr_pct'],
                'annual_vol': metrics['annual_vol'],
                'momentum_3m': metrics['momentum_3m'],
                'score': round(combined_score, 2)
            })

        except Exception as e:
            continue

    print(f"\n\nFound {len(results)} stocks matching all criteria")

    # Sort by score
    results.sort(key=lambda x: x['score'], reverse=True)

    return results[:top_n]


def save_results(results: List[dict], filename: str = 'analyst_tickers.json'):
    """Save results to JSON file."""
    output = {
        'scan_date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'criteria': 'Analyst Strong Buy/Buy + High Volatility + Positive Momentum',
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

    print("\n" + "=" * 115)
    print(f"{'Symbol':<7} {'Price':>8} {'Rank':>12} {'Buy%':>6} {'#Anlst':>7} {'Target':>8} {'Upside':>8} {'ATR%':>6} {'Mom%':>7} {'Score':>7}")
    print("=" * 115)

    for r in results:
        print(f"{r['symbol']:<7} ${r['price']:>7.2f} {r['rank_label']:>12} {r['buy_pct']:>5.1f}% {r['analysts']:>7} ${r['target']:>7.2f} {r['upside_pct']:>7.1f}% {r['atr_pct']:>5.2f}% {r['momentum_3m']:>6.1f}% {r['score']:>7.1f}")

    print("=" * 115)
    print(f"\nRank: Strong Buy (80%+ Buy) | Buy (60-80%) | Score = Rank + Volatility + Momentum + Upside")


def main():
    parser = argparse.ArgumentParser(
        description='Analyst Rating Portfolio Scanner',
        epilog='''
Scannt NASDAQ-100 und Growth-Aktien via Nasdaq API fuer Analysten-Ratings.
Filtert nach Strong Buy/Buy und kombiniert mit Volatilitaet + Momentum.

Beispiel:
  python zacks_scanner.py --top 30 --min-atr 2.5
  python new5.py --long-short --tickers analyst_tickers.json
        '''
    )
    parser.add_argument('--symbols', type=str, help='Optional: File with symbols to scan (one per line)')
    parser.add_argument('--max-rank', type=int, default=2, help='Max rank (1=Strong Buy, 2=Buy)')
    parser.add_argument('--min-analysts', type=int, default=10, help='Minimum analyst coverage')
    parser.add_argument('--min-atr', type=float, default=2.0, help='Minimum ATR%%')
    parser.add_argument('--min-price', type=float, default=10.0, help='Minimum stock price')
    parser.add_argument('--max-price', type=float, default=500.0, help='Maximum stock price')
    parser.add_argument('--top', type=int, default=30, help='Number of top stocks to return')
    parser.add_argument('--min-momentum', type=float, default=-10.0, help='Minimum 3-month momentum %%')
    parser.add_argument('--output', type=str, default='analyst_tickers.json', help='Output filename')
    parser.add_argument('--port', type=int, default=7497, help='TWS port (7497=Paper, 7496=Live)')
    parser.add_argument('--include-growth', action='store_true', help='Include high-volatility growth stocks')

    args = parser.parse_args()

    # Get symbols to scan
    if args.symbols:
        symbols = load_symbols_from_file(args.symbols)
    else:
        symbols = NASDAQ_100.copy()
        if args.include_growth:
            symbols.extend([s for s in GROWTH_STOCKS if s not in symbols])
        print(f"Scanning {'NASDAQ-100 + Growth' if args.include_growth else 'NASDAQ-100'}: {len(symbols)} symbols")

    if not symbols:
        print("Keine Symbole zum Scannen.")
        return

    try:
        # Step 1: Get analyst ratings from Nasdaq API (no TWS needed)
        analyst_ratings = scan_nasdaq_ratings(
            symbols=symbols,
            max_rank=args.max_rank,
            min_analysts=args.min_analysts
        )

        if not analyst_ratings:
            print("\nKeine Aktien mit Strong Buy/Buy Rating gefunden.")
            return

        # Step 2: Get price data from IB and filter by volatility/momentum
        results = scan_portfolio(
            analyst_ratings=analyst_ratings,
            min_atr_pct=args.min_atr,
            min_price=args.min_price,
            max_price=args.max_price,
            top_n=args.top,
            min_momentum=args.min_momentum
        )

        # Print and save results
        print_results(results)

        if results:
            save_results(results, args.output)

            # Print usage hint
            print(f"\nNaechste Schritte:")
            print(f"  1. python new5.py --long-short --tickers {args.output}")
            print(f"  2. python portfolio_simulation.py 90 --take-profit 0.30 --min-return 0.25")

    finally:
        disconnect_ib()


if __name__ == '__main__':
    main()
