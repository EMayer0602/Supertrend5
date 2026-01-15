"""
Stock Categorization Script
============================
Analyzes stocks and categorizes them by optimal strategy.

Usage:
    python categorize_stocks.py                    # Analyze and show recommendations
    python categorize_stocks.py --apply            # Apply recommendations to config
    python categorize_stocks.py --add AAPL GOOGL   # Add tickers to analyze
    python categorize_stocks.py --remove AAPL      # Remove ticker from all categories
    python categorize_stocks.py --list             # List current categories
    python categorize_stocks.py --move AAPL MOMENTUM  # Move ticker to category
"""

import json
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

CONFIG_FILE = "stock_categories.json"


def load_config() -> dict:
    """Load stock categories config"""
    try:
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Config file {CONFIG_FILE} not found. Creating default...")
        return create_default_config()


def save_config(config: dict):
    """Save stock categories config"""
    config['_last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M')
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Config saved to {CONFIG_FILE}")


def create_default_config() -> dict:
    """Create default config structure"""
    return {
        "_comment": "Stock categories for multi-strategy trading",
        "_last_updated": datetime.now().strftime('%Y-%m-%d'),
        "strategies": {
            "SUPERTREND": {"description": "Volatile stocks", "settings": {}, "tickers": []},
            "TREND_FOLLOW": {"description": "Stable uptrends", "settings": {}, "tickers": []},
            "MOMENTUM": {"description": "Bull-runs", "settings": {}, "tickers": []},
            "EXCLUDED": {"description": "Excluded stocks", "tickers": []}
        },
        "watchlist": {"description": "To analyze", "tickers": []}
    }


def analyze_stock(symbol: str, days: int = 365) -> Optional[Dict]:
    """Analyze a stock and determine its characteristics"""
    try:
        import yfinance as yf

        end = datetime.now()
        start = end - timedelta(days=days)
        df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)

        if df.empty or len(df) < 50:
            return None

        # Handle MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        volume = df['Volume'].values

        # Calculate metrics
        returns = np.diff(close) / close[:-1]

        # 1. Volatility (annualized)
        volatility = np.std(returns) * np.sqrt(252)

        # 2. Total return
        total_return = (close[-1] - close[0]) / close[0]

        # 3. Max Drawdown
        running_max = np.maximum.accumulate(close)
        drawdowns = (close - running_max) / running_max
        max_drawdown = abs(np.min(drawdowns))

        # 4. Trend Strength (R² of linear regression)
        x = np.arange(len(close))
        z = np.polyfit(x, close, 1)
        p = np.poly1d(z)
        ss_res = np.sum((close - p(x)) ** 2)
        ss_tot = np.sum((close - np.mean(close)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # 5. Trend direction
        trend_direction = "bullish" if z[0] > 0 else "bearish"

        # 6. Recent momentum (last 3 months vs previous 3 months)
        if len(close) > 126:
            recent_return = (close[-1] - close[-63]) / close[-63]
            prev_return = (close[-63] - close[-126]) / close[-126]
            momentum_acceleration = recent_return - prev_return
        else:
            recent_return = total_return
            momentum_acceleration = 0

        # 7. Average volume
        avg_volume = np.mean(volume[-20:])

        # Classify
        if volatility > 0.40:
            vol_class = "HIGH"
        elif volatility > 0.25:
            vol_class = "MEDIUM"
        else:
            vol_class = "LOW"

        if r_squared > 0.7:
            trend_class = "STRONG"
        elif r_squared > 0.4:
            trend_class = "MODERATE"
        else:
            trend_class = "WEAK"

        # Determine recommended strategy
        if vol_class == "LOW" and trend_class == "STRONG" and trend_direction == "bullish":
            recommended = "TREND_FOLLOW"
            reason = "Low volatility + strong uptrend"
        elif total_return > 1.0 and momentum_acceleration > 0.1:
            # More than 100% return and accelerating
            recommended = "MOMENTUM"
            reason = f"Strong momentum (+{total_return*100:.0f}% return, accelerating)"
        elif vol_class in ["HIGH", "MEDIUM"] or trend_class == "WEAK":
            recommended = "SUPERTREND"
            reason = f"Volatile ({vol_class}) or weak trend - Supertrend optimal"
        elif trend_direction == "bearish":
            recommended = "EXCLUDED"
            reason = "Bearish trend - avoid"
        else:
            recommended = "SUPERTREND"
            reason = "Default - use Supertrend"

        return {
            'symbol': symbol,
            'volatility': volatility,
            'vol_class': vol_class,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'r_squared': r_squared,
            'trend_class': trend_class,
            'trend_direction': trend_direction,
            'recent_momentum': recent_return,
            'momentum_acceleration': momentum_acceleration,
            'avg_volume': avg_volume,
            'recommended': recommended,
            'reason': reason
        }

    except Exception as e:
        print(f"Error analyzing {symbol}: {e}")
        return None


def get_all_tickers(config: dict) -> List[str]:
    """Get all tickers from config"""
    tickers = set()
    for strategy in config['strategies'].values():
        tickers.update(strategy.get('tickers', []))
    tickers.update(config.get('watchlist', {}).get('tickers', []))
    return sorted(list(tickers))


def find_ticker_category(config: dict, ticker: str) -> Optional[str]:
    """Find which category a ticker is in"""
    for cat_name, cat_data in config['strategies'].items():
        if ticker in cat_data.get('tickers', []):
            return cat_name
    if ticker in config.get('watchlist', {}).get('tickers', []):
        return 'WATCHLIST'
    return None


def remove_ticker_from_all(config: dict, ticker: str):
    """Remove ticker from all categories"""
    for cat_data in config['strategies'].values():
        if ticker in cat_data.get('tickers', []):
            cat_data['tickers'].remove(ticker)
    if ticker in config.get('watchlist', {}).get('tickers', []):
        config['watchlist']['tickers'].remove(ticker)


def add_ticker_to_category(config: dict, ticker: str, category: str):
    """Add ticker to a category"""
    remove_ticker_from_all(config, ticker)

    if category.upper() == 'WATCHLIST':
        if 'watchlist' not in config:
            config['watchlist'] = {'tickers': []}
        config['watchlist']['tickers'].append(ticker)
    elif category.upper() in config['strategies']:
        config['strategies'][category.upper()]['tickers'].append(ticker)
    else:
        print(f"Unknown category: {category}")


def list_categories(config: dict):
    """List all categories and their tickers"""
    print("\n" + "="*80)
    print("STOCK CATEGORIES")
    print("="*80)

    for cat_name, cat_data in config['strategies'].items():
        tickers = cat_data.get('tickers', [])
        desc = cat_data.get('description', '')
        print(f"\n{cat_name} ({len(tickers)} stocks) - {desc}")
        print("-"*60)
        if tickers:
            # Print in rows of 10
            for i in range(0, len(tickers), 10):
                print("  " + ", ".join(tickers[i:i+10]))
        else:
            print("  (empty)")

    watchlist = config.get('watchlist', {}).get('tickers', [])
    print(f"\nWATCHLIST ({len(watchlist)} stocks)")
    print("-"*60)
    if watchlist:
        print("  " + ", ".join(watchlist))
    else:
        print("  (empty)")


def analyze_and_recommend(config: dict, tickers: List[str] = None):
    """Analyze tickers and show recommendations"""
    if tickers is None:
        tickers = get_all_tickers(config)

    if not tickers:
        print("No tickers to analyze")
        return []

    print("\n" + "="*80)
    print("ANALYZING STOCKS FOR OPTIMAL STRATEGY")
    print("="*80)

    results = []
    changes = []

    for i, symbol in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] Analyzing {symbol}...", end=" ")

        analysis = analyze_stock(symbol)
        if analysis is None:
            print("Error/No data")
            continue

        current_cat = find_ticker_category(config, symbol)
        recommended = analysis['recommended']

        if current_cat != recommended:
            changes.append({
                'ticker': symbol,
                'from': current_cat,
                'to': recommended,
                'reason': analysis['reason']
            })
            print(f"-> {recommended} (was: {current_cat}) - {analysis['reason']}")
        else:
            print(f"OK ({recommended})")

        results.append(analysis)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    if changes:
        print(f"\n{len(changes)} recommended changes:")
        print("-"*60)
        for c in changes:
            print(f"  {c['ticker']}: {c['from'] or 'NEW'} -> {c['to']}")
            print(f"    Reason: {c['reason']}")
    else:
        print("\nNo changes recommended - all tickers in optimal categories")

    # Stats by category
    print("\n" + "-"*60)
    print("Recommendations by category:")
    cats = {}
    for r in results:
        cat = r['recommended']
        if cat not in cats:
            cats[cat] = []
        cats[cat].append(r['symbol'])

    for cat, syms in sorted(cats.items()):
        print(f"  {cat}: {len(syms)} stocks")
        print(f"    {', '.join(syms[:10])}" + ("..." if len(syms) > 10 else ""))

    return changes


def apply_changes(config: dict, changes: List[dict]):
    """Apply recommended changes to config"""
    if not changes:
        print("No changes to apply")
        return

    print(f"\nApplying {len(changes)} changes...")
    for c in changes:
        add_ticker_to_category(config, c['ticker'], c['to'])
        print(f"  Moved {c['ticker']} to {c['to']}")

    save_config(config)


def main():
    config = load_config()

    args = sys.argv[1:]

    if not args:
        # Default: analyze and show recommendations
        changes = analyze_and_recommend(config)
        if changes:
            print("\nRun with --apply to apply these changes")

    elif args[0] == '--list':
        list_categories(config)

    elif args[0] == '--apply':
        changes = analyze_and_recommend(config)
        if changes:
            apply_changes(config, changes)

    elif args[0] == '--add' and len(args) > 1:
        new_tickers = [t.upper() for t in args[1:]]
        print(f"Adding tickers to watchlist: {new_tickers}")
        for t in new_tickers:
            add_ticker_to_category(config, t, 'WATCHLIST')
        save_config(config)
        print("\nRun 'python categorize_stocks.py' to analyze and categorize them")

    elif args[0] == '--remove' and len(args) > 1:
        tickers = [t.upper() for t in args[1:]]
        print(f"Removing tickers: {tickers}")
        for t in tickers:
            remove_ticker_from_all(config, t)
        save_config(config)

    elif args[0] == '--move' and len(args) >= 3:
        ticker = args[1].upper()
        category = args[2].upper()
        print(f"Moving {ticker} to {category}")
        add_ticker_to_category(config, ticker, category)
        save_config(config)

    elif args[0] == '--analyze' and len(args) > 1:
        tickers = [t.upper() for t in args[1:]]
        analyze_and_recommend(config, tickers)

    else:
        print(__doc__)


if __name__ == "__main__":
    main()
