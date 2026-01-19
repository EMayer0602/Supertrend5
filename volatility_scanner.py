"""
Volatility Scanner for NASDAQ Stocks
=====================================
- Scans all tickers for volatility (ATR %)
- Filters top N most volatile stocks
- Saves filtered list for categorization
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple
import json

from new5 import download_from_tws, disconnect_ib, ALL_TICKERS


def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate Average True Range"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))

    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value doesn't have previous close

    # EMA of True Range
    atr = np.zeros_like(tr)
    atr[period-1] = np.mean(tr[:period])

    multiplier = 2 / (period + 1)
    for i in range(period, len(tr)):
        atr[i] = (tr[i] * multiplier) + (atr[i-1] * (1 - multiplier))

    return atr


def calculate_volatility_metrics(df: pd.DataFrame, symbol: str) -> Dict:
    """Calculate various volatility metrics for a symbol"""
    close_col = f'Close_{symbol}'
    high_col = f'High_{symbol}'
    low_col = f'Low_{symbol}'

    if close_col not in df.columns:
        return None

    close = df[close_col].values
    high = df[high_col].values
    low = df[low_col].values

    if len(close) < 30:
        return None

    # ATR (14-day)
    atr = calculate_atr(high, low, close, 14)
    current_atr = atr[-1]
    current_price = close[-1]

    # ATR as percentage of price
    atr_pct = (current_atr / current_price) * 100 if current_price > 0 else 0

    # Daily returns volatility (standard deviation)
    returns = np.diff(close) / close[:-1]
    daily_volatility = np.std(returns) * 100

    # Annualized volatility
    annual_volatility = daily_volatility * np.sqrt(252)

    # Average daily range
    daily_range = (high - low) / close * 100
    avg_daily_range = np.mean(daily_range[-30:])

    # Price momentum (30-day return)
    if len(close) >= 30:
        momentum_30d = (close[-1] / close[-30] - 1) * 100
    else:
        momentum_30d = 0

    return {
        'symbol': symbol,
        'price': current_price,
        'atr': current_atr,
        'atr_pct': atr_pct,
        'daily_vol': daily_volatility,
        'annual_vol': annual_volatility,
        'avg_range': avg_daily_range,
        'momentum_30d': momentum_30d
    }


def scan_volatility(tickers: List[str] = None, days: int = 100,
                    min_price: float = 10.0, max_price: float = 1000.0) -> List[Dict]:
    """
    Scan all tickers for volatility metrics.

    Args:
        tickers: List of tickers to scan (default: ALL_TICKERS)
        days: Days of history to download
        min_price: Minimum stock price filter
        max_price: Maximum stock price filter

    Returns:
        List of volatility metrics sorted by ATR%
    """
    if tickers is None:
        tickers = ALL_TICKERS

    print(f"\n{'='*60}")
    print("VOLATILITY SCANNER")
    print(f"{'='*60}")
    print(f"Scanning {len(tickers)} tickers...")
    print(f"Price range: ${min_price:.0f} - ${max_price:.0f}")
    print(f"History: {days} days")
    print()

    results = []

    for i, symbol in enumerate(tickers, 1):
        try:
            df = download_from_tws(symbol, days)
            if df.empty:
                continue

            metrics = calculate_volatility_metrics(df, symbol)
            if metrics is None:
                continue

            # Apply price filter
            if metrics['price'] < min_price or metrics['price'] > max_price:
                continue

            results.append(metrics)

            print(f"  [{i:3d}/{len(tickers)}] {symbol:5s}: "
                  f"ATR={metrics['atr_pct']:.1f}% "
                  f"Vol={metrics['annual_vol']:.0f}% "
                  f"Mom={metrics['momentum_30d']:+.1f}%")

        except Exception as e:
            print(f"  [{i:3d}/{len(tickers)}] {symbol:5s}: ERROR - {e}")

    # Sort by ATR percentage (highest first)
    results.sort(key=lambda x: x['atr_pct'], reverse=True)

    return results


def filter_top_volatile(results: List[Dict], top_n: int = 30,
                        min_atr_pct: float = 2.0,
                        min_momentum: float = -20.0) -> List[Dict]:
    """
    Filter top N most volatile stocks with additional criteria.

    Args:
        results: Volatility scan results
        top_n: Number of top volatile stocks to keep
        min_atr_pct: Minimum ATR% threshold
        min_momentum: Minimum 30-day momentum (filter out strong downtrends)
    """
    filtered = [
        r for r in results
        if r['atr_pct'] >= min_atr_pct and r['momentum_30d'] >= min_momentum
    ]

    return filtered[:top_n]


def save_volatile_tickers(results: List[Dict], filename: str = "volatile_tickers.json"):
    """Save filtered tickers to JSON file"""
    data = {
        'scan_date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'count': len(results),
        'tickers': [r['symbol'] for r in results],
        'metrics': results
    }

    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nSaved {len(results)} tickers to {filename}")


def print_volatility_report(results: List[Dict], top_n: int = 30):
    """Print formatted volatility report"""
    print(f"\n{'='*80}")
    print(f"TOP {top_n} MOST VOLATILE STOCKS")
    print(f"{'='*80}")
    print(f"{'Rank':>4} {'Symbol':>6} {'Price':>8} {'ATR%':>7} {'AnnVol':>7} {'Range':>7} {'Mom30d':>8}")
    print("-" * 80)

    for i, r in enumerate(results[:top_n], 1):
        print(f"{i:4d} {r['symbol']:>6} ${r['price']:>6.2f} "
              f"{r['atr_pct']:>6.1f}% {r['annual_vol']:>6.0f}% "
              f"{r['avg_range']:>6.1f}% {r['momentum_30d']:>+7.1f}%")

    print("-" * 80)

    # Summary statistics
    if results:
        avg_atr = np.mean([r['atr_pct'] for r in results[:top_n]])
        avg_vol = np.mean([r['annual_vol'] for r in results[:top_n]])
        print(f"\nAverage ATR%: {avg_atr:.1f}%")
        print(f"Average Annual Vol: {avg_vol:.0f}%")


def run_volatility_scan(top_n: int = 30, min_atr: float = 2.0,
                        min_price: float = 10.0, max_price: float = 500.0,
                        save: bool = True, tickers: List[str] = None):
    """
    Main function to run volatility scan and filter.

    Args:
        top_n: Number of top volatile stocks to select
        min_atr: Minimum ATR% threshold
        min_price: Minimum stock price
        max_price: Maximum stock price
        save: Whether to save results to file
        tickers: Custom ticker list (default: ALL_TICKERS)

    Returns:
        List of filtered volatile tickers
    """
    # Scan tickers
    results = scan_volatility(
        tickers=tickers,
        days=100,
        min_price=min_price,
        max_price=max_price
    )

    # Filter top volatile
    filtered = filter_top_volatile(
        results,
        top_n=top_n,
        min_atr_pct=min_atr,
        min_momentum=-30.0  # Allow some downtrend but not extreme
    )

    # Print report
    print_volatility_report(filtered, top_n)

    # Save to file
    if save:
        save_volatile_tickers(filtered)

    disconnect_ib()

    return filtered


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description='Volatility Scanner')
    parser.add_argument('--top', type=int, default=30, help='Top N volatile stocks (default: 30)')
    parser.add_argument('--min-atr', type=float, default=2.0, help='Min ATR%% (default: 2.0)')
    parser.add_argument('--min-price', type=float, default=10.0, help='Min price (default: $10)')
    parser.add_argument('--max-price', type=float, default=500.0, help='Max price (default: $500)')
    parser.add_argument('--no-save', action='store_true', help='Do not save results to file')
    parser.add_argument('--tickers', type=str, help='JSON file with tickers (e.g., analyst_tickers.json)')

    args = parser.parse_args()

    # Load tickers from file if provided
    custom_tickers = None
    if args.tickers:
        try:
            with open(args.tickers, 'r') as f:
                data = json.load(f)
                custom_tickers = data.get('tickers', [])
                print(f"Loaded {len(custom_tickers)} tickers from {args.tickers}")
        except Exception as e:
            print(f"Error loading tickers: {e}")

    results = run_volatility_scan(
        top_n=args.top,
        min_atr=args.min_atr,
        min_price=args.min_price,
        max_price=args.max_price,
        save=not args.no_save,
        tickers=custom_tickers
    )

    print(f"\n{'='*60}")
    print("NEXT STEPS")
    print(f"{'='*60}")
    print("Run categorization on volatile tickers:")
    print("  python new5.py --long-short --tickers volatile_tickers.json")
    print()
