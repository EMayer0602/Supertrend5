#!/usr/bin/env python
"""
Run TWS Dashboard - execute this on your machine where TWS is running.

Usage:
    python run_dashboard.py

Make sure TWS API is enabled:
    1. TWS -> Edit -> Global Configuration -> API -> Settings
    2. Enable "Enable ActiveX and Socket Clients"
    3. Socket port: 7496 (TWS) or 4001 (IB Gateway)
"""

from tws_connector import TWSConnector
from trade_monitor import TradeMonitor
from dashboard import TradeDashboard
import webbrowser
import os

def main():
    # Create monitor with initial capital (will be updated from TWS)
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

    # Sync positions from TWS
    print("\nSyncing positions from TWS...")
    connector.sync_to_monitor(monitor)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Net Liquidation: ${monitor.initial_capital:,.2f}")
    print(f"Cash Balance:    ${monitor.current_capital:,.2f}")
    print(f"Daily PnL:       ${monitor.get_daily_pnl():+,.2f}")
    print(f"Unrealized PnL:  ${monitor.tws_unrealized_pnl or 0:+,.2f}")
    print(f"Realized PnL:    ${monitor.tws_realized_pnl or 0:+,.2f}")
    print(f"Open Positions:  {len(monitor.open_trades)}")
    print(f"{'='*60}")

    # Generate HTML dashboard
    dashboard = TradeDashboard(monitor)
    html_file = dashboard.generate_html('tws_dashboard.html')

    print(f"\nDashboard saved to: {html_file}")

    # Open in browser
    abs_path = os.path.abspath(html_file)
    webbrowser.open(f'file://{abs_path}')
    print(f"Opening in browser...")

    # Disconnect
    connector.disconnect()
    print("\nDisconnected from TWS.")

if __name__ == '__main__':
    main()
