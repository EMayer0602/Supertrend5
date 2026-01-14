#!/usr/bin/env python
"""
Run TWS Live Dashboard - updates every 30 seconds from TWS.

Usage:
    python run_dashboard.py

Press Ctrl+C to stop.
"""

from tws_connector import TWSConnector
from trade_monitor import TradeMonitor
from dashboard import TradeDashboard
import webbrowser
import os
import time
import math
from datetime import datetime

def update_dashboard(connector, monitor, dashboard, html_file):
    """Update monitor from TWS and regenerate dashboard."""
    # Clear old trades
    monitor.open_trades.clear()
    monitor.all_trades.clear()

    # Sync fresh data from TWS
    connector.sync_to_monitor(monitor)

    # Calculate total daily PnL from positions
    total_daily_pnl = 0.0
    for trade in monitor.open_trades:
        if trade.daily_pnl and not math.isnan(trade.daily_pnl):
            total_daily_pnl += trade.daily_pnl

    # Use TWS account daily PnL if available, otherwise use sum
    if monitor.tws_daily_pnl is not None:
        daily_pnl = monitor.tws_daily_pnl
    else:
        daily_pnl = total_daily_pnl

    # Print summary
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Update")
    print(f"  Gesamt Kapital: ${monitor.initial_capital + (monitor.tws_unrealized_pnl or 0) + (monitor.tws_realized_pnl or 0):,.2f}")
    print(f"  Daily PnL:      ${daily_pnl:+,.2f}")
    print(f"  Unrealized:     ${monitor.tws_unrealized_pnl or 0:+,.2f}")
    print(f"  Positions:      {len(monitor.open_trades)}")

    # Regenerate dashboard HTML
    dashboard.save(html_file, auto_refresh=30)


def main():
    print("=" * 60)
    print("TWS Live Dashboard")
    print("Updates every 30 seconds - Press Ctrl+C to stop")
    print("=" * 60)

    # Create monitor
    monitor = TradeMonitor(initial_capital=100000)

    # Try different ports
    ports = [7496, 7497, 4001, 4002]
    connector = None

    for port in ports:
        print(f"Trying TWS on port {port}...")
        connector = TWSConnector(host='127.0.0.1', port=port)
        if connector.connect():
            print(f"Connected to TWS on port {port}!")
            break
        connector = None

    if not connector:
        print("\nCould not connect to TWS. Make sure:")
        print("  1. TWS is running")
        print("  2. API is enabled in TWS settings")
        print("  3. Socket port is correct (7496 for TWS, 4001 for IB Gateway)")
        return

    # Create dashboard
    dashboard = TradeDashboard(monitor)
    html_file = 'tws_dashboard.html'

    # Initial update
    update_dashboard(connector, monitor, dashboard, html_file)

    # Open in browser
    abs_path = os.path.abspath(html_file)
    webbrowser.open(f'file://{abs_path}')
    print(f"\nDashboard opened: {abs_path}")
    print("Updating every 30 seconds...")

    try:
        while True:
            time.sleep(30)
            update_dashboard(connector, monitor, dashboard, html_file)
    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        connector.disconnect()
        print("Disconnected from TWS.")


if __name__ == '__main__':
    main()
