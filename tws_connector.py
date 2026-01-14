"""
TWS (Interactive Brokers Trader Workstation) Connector

Connects to TWS/IB Gateway to:
- Fetch real-time positions
- Monitor account balance
- Track open orders
- Stream live prices

Requirements:
    pip install ib_insync

TWS Settings:
    1. Edit -> Global Configuration -> API -> Settings
    2. Enable "Enable ActiveX and Socket Clients"
    3. Set Socket port (default: 7497 for TWS, 4001 for IB Gateway)
    4. Disable "Read-Only API" if you want to trade
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
import threading
import time

try:
    from ib_insync import IB, Stock, Forex, Future, Option, Contract, Position, PortfolioItem
    IB_AVAILABLE = True
except ImportError:
    IB_AVAILABLE = False
    print("ib_insync not installed. Install with: pip install ib_insync")

from trade_monitor import TradeMonitor, Trade, TradeDirection, TradeStatus


@dataclass
class TWPositionInfo:
    """Position information from TWS."""
    symbol: str
    sec_type: str
    exchange: str
    currency: str
    position: float
    avg_cost: float
    market_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    account: str


@dataclass
class TWAccountInfo:
    """Account information from TWS."""
    account_id: str
    net_liquidation: float
    total_cash: float
    buying_power: float
    gross_position_value: float
    unrealized_pnl: float
    realized_pnl: float
    currency: str = "USD"


class TWSConnector:
    """
    Connector for Interactive Brokers TWS/Gateway.

    Usage:
        connector = TWSConnector()
        connector.connect()

        # Get positions
        positions = connector.get_positions()

        # Get account info
        account = connector.get_account_summary()

        # Stream updates to trade monitor
        connector.stream_to_monitor(monitor)
    """

    def __init__(self,
                 host: str = "127.0.0.1",
                 port: int = 7497,  # 7497 for TWS, 4001 for IB Gateway
                 client_id: int = 1):
        """
        Initialize TWS connector.

        Args:
            host: TWS host (usually localhost)
            port: TWS port (7497 for TWS paper, 7496 for TWS live, 4001/4002 for Gateway)
            client_id: Unique client ID for this connection
        """
        if not IB_AVAILABLE:
            raise ImportError("ib_insync not installed. Run: pip install ib_insync")

        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib = IB()

        self._connected = False
        self._positions: List[TWPositionInfo] = []
        self._account_info: Optional[TWAccountInfo] = None

        # Callbacks
        self._on_position_update: List[Callable] = []
        self._on_price_update: List[Callable] = []

        # Streaming
        self._streaming = False
        self._stream_thread: Optional[threading.Thread] = None

    def connect(self, timeout: int = 10) -> bool:
        """
        Connect to TWS.

        Args:
            timeout: Connection timeout in seconds

        Returns:
            True if connected successfully
        """
        try:
            self.ib.connect(self.host, self.port, clientId=self.client_id, timeout=timeout)
            self._connected = self.ib.isConnected()

            if self._connected:
                print(f"Connected to TWS at {self.host}:{self.port}")
                # Wait for account data to be ready
                self.ib.sleep(1)
            else:
                print("Failed to connect to TWS")

            return self._connected
        except Exception as e:
            print(f"Connection error: {e}")
            print("\nMake sure TWS is running and API is enabled:")
            print("  1. TWS -> Edit -> Global Configuration -> API -> Settings")
            print("  2. Enable 'Enable ActiveX and Socket Clients'")
            print(f"  3. Socket port should be {self.port}")
            return False

    def disconnect(self):
        """Disconnect from TWS."""
        self._streaming = False
        if self.ib.isConnected():
            self.ib.disconnect()
        self._connected = False
        print("Disconnected from TWS")

    def is_connected(self) -> bool:
        """Check if connected to TWS."""
        return self.ib.isConnected() if self.ib else False

    def get_positions(self) -> List[TWPositionInfo]:
        """
        Get all current positions.

        Returns:
            List of TWPositionInfo objects
        """
        if not self.is_connected():
            print("Not connected to TWS")
            return []

        positions = []

        # Get portfolio items (includes market values)
        portfolio = self.ib.portfolio()

        for item in portfolio:
            pos_info = TWPositionInfo(
                symbol=item.contract.symbol,
                sec_type=item.contract.secType,
                exchange=item.contract.exchange or item.contract.primaryExchange,
                currency=item.contract.currency,
                position=item.position,
                avg_cost=item.averageCost,
                market_price=item.marketPrice,
                market_value=item.marketValue,
                unrealized_pnl=item.unrealizedPNL,
                realized_pnl=item.realizedPNL,
                account=item.account
            )
            positions.append(pos_info)

        self._positions = positions
        return positions

    def get_account_summary(self) -> Optional[TWAccountInfo]:
        """
        Get account summary.

        Returns:
            TWAccountInfo object or None
        """
        if not self.is_connected():
            print("Not connected to TWS")
            return None

        # Get account values
        account_values = self.ib.accountValues()

        # Parse account values into dict
        values = {}
        account_id = ""

        for av in account_values:
            if av.tag in ['NetLiquidation', 'TotalCashValue', 'BuyingPower',
                          'GrossPositionValue', 'UnrealizedPnL', 'RealizedPnL']:
                if av.currency == 'USD' or av.currency == 'BASE':
                    values[av.tag] = float(av.value) if av.value else 0.0
                    account_id = av.account

        if values:
            self._account_info = TWAccountInfo(
                account_id=account_id,
                net_liquidation=values.get('NetLiquidation', 0),
                total_cash=values.get('TotalCashValue', 0),
                buying_power=values.get('BuyingPower', 0),
                gross_position_value=values.get('GrossPositionValue', 0),
                unrealized_pnl=values.get('UnrealizedPnL', 0),
                realized_pnl=values.get('RealizedPnL', 0)
            )
            return self._account_info

        return None

    def get_open_orders(self) -> List[Dict]:
        """Get all open orders."""
        if not self.is_connected():
            return []

        orders = self.ib.openOrders()
        return [
            {
                'order_id': o.orderId,
                'symbol': o.contract.symbol if hasattr(o, 'contract') else 'N/A',
                'action': o.action,
                'quantity': o.totalQuantity,
                'order_type': o.orderType,
                'limit_price': o.lmtPrice if hasattr(o, 'lmtPrice') else None,
                'status': o.status if hasattr(o, 'status') else 'Unknown'
            }
            for o in orders
        ]

    def get_executions(self) -> List[Dict]:
        """Get today's executions/fills."""
        if not self.is_connected():
            return []

        fills = self.ib.fills()
        return [
            {
                'time': f.time,
                'symbol': f.contract.symbol,
                'action': f.execution.side,
                'quantity': f.execution.shares,
                'price': f.execution.price,
                'commission': f.commissionReport.commission if f.commissionReport else 0
            }
            for f in fills
        ]

    def subscribe_market_data(self, symbol: str, sec_type: str = "STK",
                               exchange: str = "SMART", currency: str = "USD"):
        """
        Subscribe to market data for a symbol.

        Args:
            symbol: Ticker symbol
            sec_type: Security type (STK, FUT, OPT, CASH)
            exchange: Exchange
            currency: Currency
        """
        if not self.is_connected():
            return None

        if sec_type == "STK":
            contract = Stock(symbol, exchange, currency)
        elif sec_type == "CASH":
            contract = Forex(symbol)
        else:
            contract = Contract(symbol=symbol, secType=sec_type,
                              exchange=exchange, currency=currency)

        self.ib.qualifyContracts(contract)
        ticker = self.ib.reqMktData(contract)
        return ticker

    def sync_to_monitor(self, monitor: TradeMonitor):
        """
        Sync TWS positions to TradeMonitor.

        Args:
            monitor: TradeMonitor instance to update
        """
        positions = self.get_positions()
        account = self.get_account_summary()

        if account:
            monitor.initial_capital = account.net_liquidation
            monitor.current_capital = account.total_cash

        # Collect all prices for batch update
        prices_to_update = {}

        # Update positions
        for pos in positions:
            if pos.position != 0:
                # Check if we already have this position
                existing = [t for t in monitor.open_trades
                           if t.symbol == pos.symbol and t.status == TradeStatus.OPEN]

                if not existing:
                    # Add as open trade
                    direction = TradeDirection.LONG if pos.position > 0 else TradeDirection.SHORT
                    trade = Trade(
                        trade_id=f"TWS_{pos.symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                        symbol=pos.symbol,
                        direction=direction,
                        status=TradeStatus.OPEN,
                        entry_date=datetime.now(),
                        entry_price=pos.avg_cost,
                        entry_quantity=abs(int(pos.position)),
                        unrealized_pnl=pos.unrealized_pnl,
                        realized_pnl=pos.realized_pnl,
                        current_price=pos.market_price  # Set current price per trade
                    )
                    monitor.open_trades.append(trade)
                    monitor.all_trades.append(trade)
                else:
                    # Update existing trade
                    trade = existing[0]
                    trade.unrealized_pnl = pos.unrealized_pnl
                    trade.current_price = pos.market_price

                # Collect price for this symbol
                prices_to_update[pos.symbol] = pos.market_price

        # Update all prices at once
        if prices_to_update:
            monitor.update_prices(prices_to_update)

        return monitor

    def start_streaming(self, monitor: TradeMonitor, interval: float = 5.0):
        """
        Start streaming updates to monitor.

        Args:
            monitor: TradeMonitor to update
            interval: Update interval in seconds
        """
        self._streaming = True

        def stream_loop():
            while self._streaming and self.is_connected():
                try:
                    self.ib.sleep(0.1)  # Process IB events
                    self.sync_to_monitor(monitor)
                    time.sleep(interval)
                except Exception as e:
                    print(f"Streaming error: {e}")
                    time.sleep(interval)

        self._stream_thread = threading.Thread(target=stream_loop, daemon=True)
        self._stream_thread.start()
        print(f"Started streaming updates every {interval}s")

    def stop_streaming(self):
        """Stop streaming updates."""
        self._streaming = False
        if self._stream_thread:
            self._stream_thread.join(timeout=5)
        print("Stopped streaming")


def print_tws_status(connector: TWSConnector):
    """Print TWS connection status and positions."""
    print("\n" + "=" * 60)
    print("TWS CONNECTION STATUS")
    print("=" * 60)

    if not connector.is_connected():
        print("NOT CONNECTED to TWS")
        print("\nTo connect:")
        print("  1. Start TWS or IB Gateway")
        print("  2. Enable API: Edit -> Global Config -> API -> Settings")
        print("  3. Enable 'Enable ActiveX and Socket Clients'")
        print("  4. Set port to 7497 (TWS) or 4001 (Gateway)")
        return

    print("CONNECTED to TWS")

    # Account info
    account = connector.get_account_summary()
    if account:
        print(f"\n{'='*40}")
        print("ACCOUNT SUMMARY")
        print(f"{'='*40}")
        print(f"  Account:          {account.account_id}")
        print(f"  Net Liquidation:  ${account.net_liquidation:,.2f}")
        print(f"  Total Cash:       ${account.total_cash:,.2f}")
        print(f"  Buying Power:     ${account.buying_power:,.2f}")
        print(f"  Position Value:   ${account.gross_position_value:,.2f}")
        print(f"  Unrealized PnL:   ${account.unrealized_pnl:,.2f}")
        print(f"  Realized PnL:     ${account.realized_pnl:,.2f}")

    # Positions
    positions = connector.get_positions()
    if positions:
        print(f"\n{'='*40}")
        print(f"POSITIONS ({len(positions)})")
        print(f"{'='*40}")
        print(f"{'Symbol':<10} {'Qty':>8} {'AvgCost':>10} {'MktPrice':>10} {'MktVal':>12} {'UnrlzdPnL':>12}")
        print("-" * 70)
        for pos in positions:
            print(f"{pos.symbol:<10} {pos.position:>8.0f} ${pos.avg_cost:>9.2f} ${pos.market_price:>9.2f} ${pos.market_value:>11.2f} ${pos.unrealized_pnl:>11.2f}")
    else:
        print("\nNo open positions")

    # Open orders
    orders = connector.get_open_orders()
    if orders:
        print(f"\n{'='*40}")
        print(f"OPEN ORDERS ({len(orders)})")
        print(f"{'='*40}")
        for o in orders:
            print(f"  {o['symbol']} {o['action']} {o['quantity']} @ {o['order_type']}")

    print("\n" + "=" * 60)


# Example usage
if __name__ == "__main__":
    print("TWS Connector Demo")
    print("=" * 50)

    if not IB_AVAILABLE:
        print("\nib_insync not installed!")
        print("Install with: pip install ib_insync")
        exit(1)

    connector = TWSConnector(
        host="127.0.0.1",
        port=7497,  # Change to 4001 for IB Gateway
        client_id=1
    )

    if connector.connect():
        print_tws_status(connector)

        # Create monitor and sync
        from trade_monitor import TradeMonitor
        monitor = TradeMonitor(initial_capital=10000, symbol="Portfolio")
        connector.sync_to_monitor(monitor)
        monitor.print_summary()

        connector.disconnect()
    else:
        print("\nCould not connect to TWS")
        print("Make sure TWS is running with API enabled")
