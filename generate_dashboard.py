#!/usr/bin/env python3
"""
Dashboard Generator for Supertrend Trading System
================================================
Connects to TWS, reads all positions, and generates an HTML dashboard.

Usage:
    python generate_dashboard.py              # Generate dashboard
    python generate_dashboard.py --live       # Auto-refresh every 60 seconds
"""

import json
import sys
from datetime import datetime
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import ib_insync
try:
    from ib_insync import IB, Stock, util
    IB_AVAILABLE = True
except ImportError:
    IB_AVAILABLE = False
    logger.warning("ib_insync not installed. Run: pip install ib_insync")

# Configuration
IB_HOST = "127.0.0.1"
IB_PORT = 7497  # Paper trading
IB_CLIENT_ID = 99  # Different from trader to avoid conflicts

CATEGORIES_FILE = "stock_categories.json"
OUTPUT_FILE = "dashboard.html"


def load_stock_categories() -> dict:
    """Load stock categories from config file"""
    try:
        with open(CATEGORIES_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def get_all_tickers() -> List[str]:
    """Get all active tickers from config"""
    config = load_stock_categories()
    if not config:
        return []

    tickers = []
    for strategy, data in config.get('strategies', {}).items():
        if strategy != 'EXCLUDED':
            tickers.extend(data.get('tickers', []))
    return list(set(tickers))


def get_ticker_strategy(ticker: str) -> str:
    """Get strategy for a ticker"""
    config = load_stock_categories()
    if config:
        for strategy, data in config.get('strategies', {}).items():
            if ticker in data.get('tickers', []):
                return strategy
    return "UNKNOWN"


def connect_to_ib() -> Optional[IB]:
    """Connect to Interactive Brokers TWS"""
    if not IB_AVAILABLE:
        logger.error("ib_insync not available")
        return None

    try:
        ib = IB()
        ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID)
        logger.info(f"Connected to IB at {IB_HOST}:{IB_PORT}")
        return ib
    except Exception as e:
        logger.error(f"Could not connect to IB: {e}")
        return None


def get_portfolio_data(ib: IB) -> Dict:
    """Get all portfolio data from IB"""
    data = {
        'account': {},
        'positions': [],
        'total_pnl': 0,
        'daily_pnl': 0,
        'timestamp': datetime.now().isoformat()
    }

    # Account info
    for av in ib.accountValues():
        if av.tag in ['NetLiquidation', 'AvailableFunds', 'BuyingPower', 'TotalCashValue', 'UnrealizedPnL', 'RealizedPnL']:
            data['account'][av.tag] = float(av.value)

    # Portfolio positions
    portfolio = ib.portfolio()

    for item in portfolio:
        symbol = item.contract.symbol
        quantity = item.position
        entry_price = item.averageCost
        market_value = item.marketValue
        unrealized_pnl = item.unrealizedPNL

        if quantity != 0:
            current_price = abs(market_value / quantity)
        else:
            current_price = entry_price

        pnl_pct = (unrealized_pnl / (entry_price * abs(quantity))) * 100 if entry_price * abs(quantity) != 0 else 0

        strategy = get_ticker_strategy(symbol)

        data['positions'].append({
            'symbol': symbol,
            'strategy': strategy,
            'quantity': int(quantity),
            'entry_price': round(entry_price, 2),
            'current_price': round(current_price, 2),
            'market_value': round(market_value, 2),
            'pnl_value': round(unrealized_pnl, 2),
            'pnl_pct': round(pnl_pct, 2)
        })

        data['total_pnl'] += unrealized_pnl

    # Get daily P&L
    try:
        account_id = ib.managedAccounts()[0] if ib.managedAccounts() else ''
        if account_id:
            ib.reqPnL(account_id, '')
            ib.sleep(1)
            pnl_list = ib.pnl()
            for pnl in pnl_list:
                if pnl.dailyPnL is not None:
                    import math
                    if not math.isnan(pnl.dailyPnL):
                        data['daily_pnl'] = pnl.dailyPnL
                        break
    except:
        pass

    # Sort positions by P&L
    data['positions'].sort(key=lambda x: x['pnl_value'], reverse=True)

    return data


def generate_html(data: Dict, auto_refresh: bool = False) -> str:
    """Generate HTML dashboard"""

    positions_html = ""
    for pos in data['positions']:
        pnl_class = "positive" if pos['pnl_value'] >= 0 else "negative"
        positions_html += f"""
        <tr class="{pnl_class}">
            <td><strong>{pos['symbol']}</strong></td>
            <td><span class="strategy-badge {pos['strategy'].lower()}">{pos['strategy']}</span></td>
            <td>{pos['quantity']}</td>
            <td>${pos['entry_price']:,.2f}</td>
            <td>${pos['current_price']:,.2f}</td>
            <td>${pos['market_value']:,.2f}</td>
            <td class="{pnl_class}">{pos['pnl_pct']:+.2f}%</td>
            <td class="{pnl_class}">${pos['pnl_value']:+,.2f}</td>
        </tr>
        """

    total_pnl_class = "positive" if data['total_pnl'] >= 0 else "negative"
    daily_pnl_class = "positive" if data['daily_pnl'] >= 0 else "negative"

    refresh_meta = '<meta http-equiv="refresh" content="60">' if auto_refresh else ''

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    {refresh_meta}
    <title>Supertrend Trading Dashboard</title>
    <style>
        :root {{
            --bg-dark: #1a1a2e;
            --bg-card: #16213e;
            --text-primary: #eee;
            --text-secondary: #aaa;
            --positive: #00d26a;
            --negative: #ff4757;
            --accent: #4da6ff;
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: var(--bg-dark);
            color: var(--text-primary);
            min-height: 100vh;
            padding: 20px;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}

        header {{
            text-align: center;
            padding: 20px;
            margin-bottom: 30px;
        }}

        h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            background: linear-gradient(90deg, var(--accent), #00d26a);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .timestamp {{
            color: var(--text-secondary);
            font-size: 0.9em;
        }}

        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}

        .card {{
            background: var(--bg-card);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }}

        .card-label {{
            color: var(--text-secondary);
            font-size: 0.85em;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }}

        .card-value {{
            font-size: 1.8em;
            font-weight: bold;
        }}

        .card-value.positive {{
            color: var(--positive);
        }}

        .card-value.negative {{
            color: var(--negative);
        }}

        .positions-table {{
            width: 100%;
            background: var(--bg-card);
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }}

        .positions-table table {{
            width: 100%;
            border-collapse: collapse;
        }}

        .positions-table th {{
            background: rgba(77, 166, 255, 0.2);
            padding: 15px;
            text-align: left;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85em;
            letter-spacing: 1px;
        }}

        .positions-table td {{
            padding: 12px 15px;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }}

        .positions-table tr:hover {{
            background: rgba(255,255,255,0.03);
        }}

        .positive {{
            color: var(--positive);
        }}

        .negative {{
            color: var(--negative);
        }}

        .strategy-badge {{
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.75em;
            font-weight: bold;
            text-transform: uppercase;
        }}

        .strategy-badge.supertrend {{
            background: rgba(255, 107, 107, 0.2);
            color: #ff6b6b;
        }}

        .strategy-badge.buy_hold {{
            background: rgba(0, 210, 106, 0.2);
            color: #00d26a;
        }}

        .strategy-badge.kama {{
            background: rgba(77, 166, 255, 0.2);
            color: #4da6ff;
        }}

        .strategy-badge.jma {{
            background: rgba(255, 193, 7, 0.2);
            color: #ffc107;
        }}

        .strategy-badge.trend_follow {{
            background: rgba(156, 39, 176, 0.2);
            color: #ce93d8;
        }}

        .strategy-badge.momentum {{
            background: rgba(255, 152, 0, 0.2);
            color: #ffb74d;
        }}

        .footer {{
            text-align: center;
            padding: 30px;
            color: var(--text-secondary);
            font-size: 0.85em;
        }}

        @media (max-width: 768px) {{
            .positions-table {{
                overflow-x: auto;
            }}

            h1 {{
                font-size: 1.8em;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Supertrend Trading Dashboard</h1>
            <p class="timestamp">Last Updated: {data['timestamp']}</p>
            {'<p style="color: var(--accent);">Auto-refresh: 60 seconds</p>' if auto_refresh else ''}
        </header>

        <div class="summary-cards">
            <div class="card">
                <div class="card-label">Portfolio Value</div>
                <div class="card-value">${data['account'].get('NetLiquidation', 0):,.2f}</div>
            </div>
            <div class="card">
                <div class="card-label">Cash Available</div>
                <div class="card-value">${data['account'].get('AvailableFunds', 0):,.2f}</div>
            </div>
            <div class="card">
                <div class="card-label">Open Positions</div>
                <div class="card-value">{len(data['positions'])}</div>
            </div>
            <div class="card">
                <div class="card-label">Total P&L</div>
                <div class="card-value {total_pnl_class}">${data['total_pnl']:+,.2f}</div>
            </div>
            <div class="card">
                <div class="card-label">Daily P&L</div>
                <div class="card-value {daily_pnl_class}">${data['daily_pnl']:+,.2f}</div>
            </div>
        </div>

        <div class="positions-table">
            <table>
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Strategy</th>
                        <th>Qty</th>
                        <th>Entry</th>
                        <th>Current</th>
                        <th>Value</th>
                        <th>P&L %</th>
                        <th>P&L $</th>
                    </tr>
                </thead>
                <tbody>
                    {positions_html if positions_html else '<tr><td colspan="8" style="text-align:center; padding:40px;">No open positions</td></tr>'}
                </tbody>
            </table>
        </div>

        <div class="footer">
            <p>Supertrend Multi-Strategy Trading System</p>
            <p>Connected to TWS Paper Trading @ {IB_HOST}:{IB_PORT}</p>
        </div>
    </div>
</body>
</html>
"""
    return html


def main():
    print("="*60)
    print("SUPERTREND DASHBOARD GENERATOR")
    print("="*60)

    auto_refresh = "--live" in sys.argv

    # Connect to IB
    ib = connect_to_ib()

    if ib is None:
        print("\nCould not connect to TWS. Make sure:")
        print("  1. TWS is running")
        print("  2. API is enabled (Edit -> Global Configuration -> API -> Settings)")
        print("  3. Port 7497 is configured for Paper Trading")
        print("\nGenerating empty dashboard...")

        data = {
            'account': {'NetLiquidation': 0, 'AvailableFunds': 0},
            'positions': [],
            'total_pnl': 0,
            'daily_pnl': 0,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    else:
        print("\nFetching portfolio data...")
        data = get_portfolio_data(ib)

        # Disconnect
        ib.disconnect()
        print("Disconnected from IB")

    # Generate HTML
    print(f"\nGenerating dashboard...")
    html = generate_html(data, auto_refresh)

    # Write to file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\nDashboard saved to: {OUTPUT_FILE}")
    print(f"Positions: {len(data['positions'])}")
    print(f"Total P&L: ${data['total_pnl']:+,.2f}")
    print(f"Daily P&L: ${data['daily_pnl']:+,.2f}")

    if auto_refresh:
        print("\nAuto-refresh enabled (60 seconds)")
        print("Open dashboard.html in your browser")

    # Try to open in browser
    try:
        import webbrowser
        import os
        webbrowser.open('file://' + os.path.realpath(OUTPUT_FILE))
        print("\nOpened dashboard in browser")
    except:
        print(f"\nOpen {OUTPUT_FILE} in your browser to view")


if __name__ == "__main__":
    main()
