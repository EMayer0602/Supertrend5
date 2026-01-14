"""
Equity Curve Calculator - Calculate equity curve from historical trade data.

Reads minute data for all symbols from entry to exit and calculates:
- Unrealized PnL over time for open positions
- Realized PnL when positions are closed
- Includes fees/commissions

This allows calculating the equity curve without running the dashboard continuously.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import json
import os

# TWS/IB for historical data
try:
    from ib_insync import IB, Stock, util
    IB_AVAILABLE = True
except ImportError:
    IB_AVAILABLE = False

# Fallback to yfinance if IB not available
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


@dataclass
class TradeRecord:
    """Record of a trade for equity calculation."""
    symbol: str
    direction: str  # 'LONG' or 'SHORT'
    entry_date: datetime
    entry_price: float
    quantity: int
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    entry_commission: float = 0.0  # Commission at entry (deducted immediately)
    exit_commission: float = 0.0   # Commission at exit (deducted at close)

    @property
    def is_closed(self) -> bool:
        return self.exit_date is not None


@dataclass
class EquityPoint:
    """A point on the equity curve."""
    timestamp: datetime
    unrealized_pnl: float
    realized_pnl: float
    total_pnl: float
    fees: float


class EquityCurveCalculator:
    """Calculate equity curve from historical trade data."""

    def __init__(self, trades_file: str = "trades_history.json", ib_port: int = 7497):
        self.trades_file = trades_file
        self.trades: List[TradeRecord] = []
        self.minute_data_cache: Dict[str, pd.DataFrame] = {}
        self.ib_port = ib_port
        self.ib: Optional[IB] = None
        self._load_trades()

    def connect_ib(self) -> bool:
        """Connect to TWS/IB Gateway."""
        if not IB_AVAILABLE:
            print("ib_insync not installed")
            return False

        if self.ib and self.ib.isConnected():
            return True

        try:
            self.ib = IB()
            self.ib.connect('127.0.0.1', self.ib_port, clientId=20)
            return True
        except Exception as e:
            print(f"TWS connection error: {e}")
            self.ib = None
            return False

    def disconnect_ib(self):
        """Disconnect from TWS."""
        if self.ib and self.ib.isConnected():
            self.ib.disconnect()
            self.ib = None

    def _load_trades(self):
        """Load trades from JSON file."""
        if os.path.exists(self.trades_file):
            try:
                with open(self.trades_file, 'r') as f:
                    data = json.load(f)
                    for t in data:
                        # Support old format (single commission) and new format (entry/exit)
                        if 'entry_commission' in t:
                            entry_comm = t['entry_commission']
                            exit_comm = t.get('exit_commission', 0.0)
                        else:
                            # Old format: split commission evenly or assign to entry
                            old_comm = t.get('commission', 0.0)
                            entry_comm = old_comm / 2 if t.get('exit_date') else old_comm
                            exit_comm = old_comm / 2 if t.get('exit_date') else 0.0

                        self.trades.append(TradeRecord(
                            symbol=t['symbol'],
                            direction=t['direction'],
                            entry_date=datetime.fromisoformat(t['entry_date']),
                            entry_price=t['entry_price'],
                            quantity=t['quantity'],
                            exit_date=datetime.fromisoformat(t['exit_date']) if t.get('exit_date') else None,
                            exit_price=t.get('exit_price'),
                            entry_commission=entry_comm,
                            exit_commission=exit_comm
                        ))
            except Exception as e:
                print(f"Error loading trades: {e}")

    def _save_trades(self):
        """Save trades to JSON file."""
        data = []
        for t in self.trades:
            data.append({
                'symbol': t.symbol,
                'direction': t.direction,
                'entry_date': t.entry_date.isoformat(),
                'entry_price': t.entry_price,
                'quantity': t.quantity,
                'exit_date': t.exit_date.isoformat() if t.exit_date else None,
                'exit_price': t.exit_price,
                'entry_commission': t.entry_commission,
                'exit_commission': t.exit_commission
            })
        with open(self.trades_file, 'w') as f:
            json.dump(data, f, indent=2)

    def add_trade(self, symbol: str, direction: str, entry_date: datetime,
                  entry_price: float, quantity: int, entry_commission: float = 0.0):
        """Add a new trade. Entry commission is deducted immediately from PnL."""
        trade = TradeRecord(
            symbol=symbol,
            direction=direction,
            entry_date=entry_date,
            entry_price=entry_price,
            quantity=quantity,
            entry_commission=entry_commission
        )
        self.trades.append(trade)
        self._save_trades()
        return trade

    def close_trade(self, symbol: str, exit_date: datetime, exit_price: float,
                    exit_commission: float = 0.0):
        """Close an open trade. Exit commission is deducted from realized PnL."""
        for trade in self.trades:
            if trade.symbol == symbol and not trade.is_closed:
                trade.exit_date = exit_date
                trade.exit_price = exit_price
                trade.exit_commission = exit_commission
                self._save_trades()
                return trade
        return None

    def import_from_tws(self, monitor):
        """Import current positions from TWS monitor."""
        for trade in monitor.open_trades:
            # Check if already exists
            exists = any(t.symbol == trade.symbol and not t.is_closed for t in self.trades)
            if not exists:
                self.add_trade(
                    symbol=trade.symbol,
                    direction=trade.direction.value,
                    entry_date=trade.entry_date,
                    entry_price=trade.entry_price,
                    quantity=trade.entry_quantity,
                    entry_commission=trade.commission  # Entry commission from TWS
                )
        self._save_trades()

    def get_minute_data(self, symbol: str, start_date: datetime,
                        end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Get minute/hourly data for a symbol from TWS or yfinance."""
        end_date = end_date or datetime.now()
        cache_key = f"{symbol}_{start_date.date()}_{end_date.date()}"

        if cache_key in self.minute_data_cache:
            return self.minute_data_cache[cache_key]

        # Try TWS first
        df = self._get_ib_data(symbol, start_date, end_date)

        # Fallback to yfinance
        if df.empty and YFINANCE_AVAILABLE:
            df = self._get_yfinance_data(symbol, start_date, end_date)

        if not df.empty:
            self.minute_data_cache[cache_key] = df

        return df

    def _get_ib_data(self, symbol: str, start_date: datetime,
                     end_date: datetime) -> pd.DataFrame:
        """Fetch historical data from TWS/IB."""
        if not self.connect_ib():
            return pd.DataFrame()

        try:
            contract = Stock(symbol, 'SMART', 'USD')
            self.ib.qualifyContracts(contract)

            days_diff = (end_date - start_date).days

            # IB duration string format
            if days_diff <= 1:
                duration = '1 D'
                bar_size = '1 min'
            elif days_diff <= 7:
                duration = f'{days_diff} D'
                bar_size = '5 mins'
            elif days_diff <= 30:
                duration = f'{days_diff} D'
                bar_size = '1 hour'
            else:
                duration = f'{min(days_diff, 365)} D'
                bar_size = '1 hour'

            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime=end_date,
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow='TRADES',
                useRTH=True,
                formatDate=1
            )

            if bars:
                df = util.df(bars)
                df.set_index('date', inplace=True)
                df.rename(columns={'open': 'Open', 'high': 'High',
                                   'low': 'Low', 'close': 'Close',
                                   'volume': 'Volume'}, inplace=True)
                return df

        except Exception as e:
            print(f"IB data error for {symbol}: {e}")

        return pd.DataFrame()

    def _get_yfinance_data(self, symbol: str, start_date: datetime,
                           end_date: datetime) -> pd.DataFrame:
        """Fallback: Fetch data from yfinance."""
        try:
            ticker = yf.Ticker(symbol)
            days_diff = (end_date - start_date).days

            if days_diff <= 7:
                df = ticker.history(start=start_date, end=end_date, interval='1m')
            else:
                df = ticker.history(start=start_date, end=end_date, interval='1h')

            return df
        except Exception as e:
            print(f"yfinance error for {symbol}: {e}")
            return pd.DataFrame()

    def calculate_trade_pnl_series(self, trade: TradeRecord) -> pd.Series:
        """
        Calculate PnL series for a single trade.
        Entry commission is deducted from the start (affects unrealized PnL).
        """
        end_date = trade.exit_date or datetime.now()
        df = self.get_minute_data(trade.symbol, trade.entry_date, end_date)

        if df.empty:
            return pd.Series()

        # Calculate PnL at each point
        if trade.direction == 'LONG':
            pnl = (df['Close'] - trade.entry_price) * trade.quantity
        else:  # SHORT
            pnl = (trade.entry_price - df['Close']) * trade.quantity

        # Deduct entry commission from start (affects all unrealized PnL points)
        pnl = pnl - trade.entry_commission

        return pnl

    def calculate_equity_curve(self, days: int = 30) -> pd.DataFrame:
        """
        Calculate the complete equity curve.

        Returns DataFrame with columns:
        - timestamp
        - unrealized_pnl
        - realized_pnl
        - total_pnl
        - fees
        """
        if not self.trades:
            return pd.DataFrame()

        start_date = datetime.now() - timedelta(days=days)

        # Collect all PnL series
        all_pnl = {}
        realized_events = []  # (timestamp, realized_pnl, fee)
        total_fees = 0.0

        for trade in self.trades:
            if trade.entry_date < start_date and trade.is_closed and trade.exit_date < start_date:
                continue  # Skip old closed trades

            pnl_series = self.calculate_trade_pnl_series(trade)
            if not pnl_series.empty:
                all_pnl[trade.symbol] = pnl_series

            # Track realized PnL events
            if trade.is_closed:
                if trade.direction == 'LONG':
                    realized = (trade.exit_price - trade.entry_price) * trade.quantity
                else:
                    realized = (trade.entry_price - trade.exit_price) * trade.quantity
                # Deduct both entry and exit commissions
                total_commission = trade.entry_commission + trade.exit_commission
                realized -= total_commission
                realized_events.append((trade.exit_date, realized, total_commission))
                total_fees += total_commission

        if not all_pnl:
            return pd.DataFrame()

        # Combine all PnL series
        combined = pd.DataFrame(all_pnl)
        combined['unrealized_pnl'] = combined.sum(axis=1)

        # Add realized PnL column (cumulative)
        combined['realized_pnl'] = 0.0
        combined['fees'] = 0.0

        cumulative_realized = 0.0
        cumulative_fees = 0.0
        for ts, realized, fee in sorted(realized_events):
            cumulative_realized += realized
            cumulative_fees += fee
            # Use > instead of >= to avoid double-counting at exit_date
            # At exit_date: unrealized still shows the trade
            # After exit_date: unrealized=0, realized takes over
            mask = combined.index > ts
            combined.loc[mask, 'realized_pnl'] = cumulative_realized
            combined.loc[mask, 'fees'] = cumulative_fees

        # Calculate total
        combined['total_pnl'] = combined['unrealized_pnl'] + combined['realized_pnl']

        # Keep only relevant columns
        result = combined[['unrealized_pnl', 'realized_pnl', 'total_pnl', 'fees']].copy()
        result.index.name = 'timestamp'

        return result

    def get_equity_curve_data(self, days: int = 30) -> List[Tuple[datetime, float]]:
        """Get equity curve data for plotting (timestamp, total_pnl)."""
        df = self.calculate_equity_curve(days)
        if df.empty:
            return []

        return [(idx.to_pydatetime(), row['total_pnl']) for idx, row in df.iterrows()]

    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price from IB or yfinance."""
        # Try IB first
        if self.connect_ib():
            try:
                contract = Stock(symbol, 'SMART', 'USD')
                self.ib.qualifyContracts(contract)
                ticker = self.ib.reqMktData(contract, '', False, False)
                self.ib.sleep(2)
                if ticker.last and ticker.last > 0:
                    self.ib.cancelMktData(contract)
                    return ticker.last
                if ticker.close and ticker.close > 0:
                    self.ib.cancelMktData(contract)
                    return ticker.close
                self.ib.cancelMktData(contract)
            except Exception as e:
                print(f"IB price error for {symbol}: {e}")

        # Fallback to yfinance
        if YFINANCE_AVAILABLE:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='1d')
                if not hist.empty:
                    return hist['Close'].iloc[-1]
            except Exception:
                pass

        return None

    def get_worst_performers(self, count: int = 3) -> List[Tuple[TradeRecord, float, float]]:
        """
        Get the worst performing open positions.

        Returns list of (trade, pnl_absolute, pnl_percent) sorted by pnl_percent ascending.
        """
        open_trades = [t for t in self.trades if not t.is_closed]
        if not open_trades:
            return []

        performances = []
        for trade in open_trades:
            try:
                current_price = self._get_current_price(trade.symbol)
                if current_price is None:
                    print(f"Could not get price for {trade.symbol}")
                    continue

                # Calculate PnL
                if trade.direction == 'LONG':
                    pnl_absolute = (current_price - trade.entry_price) * trade.quantity
                else:  # SHORT
                    pnl_absolute = (trade.entry_price - current_price) * trade.quantity

                # Calculate percentage return
                cost_basis = trade.entry_price * trade.quantity
                pnl_percent = (pnl_absolute / cost_basis) * 100 if cost_basis > 0 else 0

                performances.append((trade, pnl_absolute, pnl_percent, current_price))
            except Exception as e:
                print(f"Error getting price for {trade.symbol}: {e}")
                continue

        # Sort by percentage (ascending = worst first)
        performances.sort(key=lambda x: x[2])

        # Return top N worst
        return [(t, pnl, pct) for t, pnl, pct, _ in performances[:count]]

    def print_worst_performers(self, count: int = 3):
        """Print the worst performing positions."""
        worst = self.get_worst_performers(count)

        if not worst:
            print("No open positions to analyze")
            return

        print(f"\n{'='*60}")
        print(f"TOP {count} SCHLECHTESTE PERFORMER")
        print(f"{'='*60}")
        print(f"{'Symbol':<10} {'Richtung':<8} {'Einstieg':>10} {'PnL $':>12} {'PnL %':>10}")
        print("-" * 60)

        for trade, pnl_abs, pnl_pct in worst:
            print(f"{trade.symbol:<10} {trade.direction:<8} ${trade.entry_price:>9.2f} "
                  f"${pnl_abs:>+11.2f} {pnl_pct:>+9.2f}%")

        print("-" * 60)
        total_loss = sum(pnl for _, pnl, _ in worst)
        print(f"{'Gesamt Verlust:':<30} ${total_loss:>+11.2f}")

        return worst

    def suggest_sells(self, count: int = 3) -> List[str]:
        """
        Suggest symbols to sell (worst performers).
        Returns list of symbols.
        """
        worst = self.get_worst_performers(count)
        return [trade.symbol for trade, _, _ in worst]

    def print_summary(self):
        """Print summary of trades and equity."""
        print(f"\n{'='*60}")
        print("TRADE SUMMARY")
        print(f"{'='*60}")

        open_trades = [t for t in self.trades if not t.is_closed]
        closed_trades = [t for t in self.trades if t.is_closed]

        print(f"Open trades: {len(open_trades)}")
        print(f"Closed trades: {len(closed_trades)}")

        total_realized = 0.0
        total_fees = 0.0

        for t in closed_trades:
            if t.direction == 'LONG':
                pnl = (t.exit_price - t.entry_price) * t.quantity
            else:
                pnl = (t.entry_price - t.exit_price) * t.quantity
            total_commission = t.entry_commission + t.exit_commission
            pnl -= total_commission
            total_realized += pnl
            total_fees += total_commission

        print(f"\nRealized PnL: ${total_realized:+,.2f}")
        print(f"Total Fees: ${total_fees:,.2f}")

        if open_trades:
            print(f"\nOpen Positions:")
            for t in open_trades:
                print(f"  {t.symbol}: {t.quantity} @ ${t.entry_price:.2f} ({t.direction})")


def calculate_equity_from_trades(trades_file: str = "trades_history.json",
                                  days: int = 30) -> List[Tuple[datetime, float]]:
    """Convenience function to calculate equity curve from trades file."""
    calc = EquityCurveCalculator(trades_file)
    return calc.get_equity_curve_data(days)


def main():
    """CLI for equity calculator."""
    import argparse

    parser = argparse.ArgumentParser(description='Equity Curve Calculator')
    parser.add_argument('--import-tws', action='store_true',
                        help='Import open positions from TWS')
    parser.add_argument('--port', type=int, default=7497,
                        help='TWS port (default: 7497)')
    parser.add_argument('--add', nargs=5, metavar=('SYMBOL', 'DIR', 'PRICE', 'QTY', 'DATE'),
                        help='Add trade: SYMBOL LONG/SHORT PRICE QTY YYYY-MM-DD')
    parser.add_argument('--close', nargs=3, metavar=('SYMBOL', 'PRICE', 'DATE'),
                        help='Close trade: SYMBOL PRICE YYYY-MM-DD')
    parser.add_argument('--list', action='store_true', help='List all trades')
    parser.add_argument('--calc', action='store_true', help='Calculate equity curve')
    parser.add_argument('--worst', type=int, nargs='?', const=3, metavar='N',
                        help='Show N worst performers (default: 3)')
    parser.add_argument('--days', type=int, default=7, help='Days for calculation')
    args = parser.parse_args()

    calc = EquityCurveCalculator()

    if args.import_tws:
        try:
            from tws_connector import TWSConnector
            from trade_monitor import TradeMonitor

            print(f"Connecting to TWS on port {args.port}...")
            connector = TWSConnector(port=args.port)

            if connector.connect():
                monitor = TradeMonitor(initial_capital=100000)
                connector.sync_to_monitor(monitor)

                print(f"\nImporting {len(monitor.open_trades)} positions...")
                for trade in monitor.open_trades:
                    exists = any(t.symbol == trade.symbol and not t.is_closed
                                for t in calc.trades)
                    if not exists:
                        calc.add_trade(
                            symbol=trade.symbol,
                            direction=trade.direction.value,
                            entry_date=trade.entry_date,
                            entry_price=trade.entry_price,
                            quantity=trade.entry_quantity,
                            entry_commission=trade.commission
                        )
                        print(f"  Added: {trade.symbol} {trade.entry_quantity} @ ${trade.entry_price:.2f}")
                    else:
                        print(f"  Skip (exists): {trade.symbol}")

                connector.disconnect()
                print(f"\nTrades saved to: {calc.trades_file}")
            else:
                print("Could not connect to TWS")

        except ImportError:
            print("ib_insync not installed")

    elif args.add:
        symbol, direction, price, qty, date_str = args.add
        entry_date = datetime.strptime(date_str, '%Y-%m-%d')
        calc.add_trade(symbol, direction.upper(), entry_date,
                      float(price), int(qty))
        print(f"Added: {symbol} {direction} {qty} @ ${price}")

    elif args.close:
        symbol, price, date_str = args.close
        exit_date = datetime.strptime(date_str, '%Y-%m-%d')
        trade = calc.close_trade(symbol, exit_date, float(price))
        if trade:
            print(f"Closed: {symbol} @ ${price}")
        else:
            print(f"No open trade found for {symbol}")

    elif args.list:
        calc.print_summary()

    elif args.worst:
        calc.print_worst_performers(args.worst)

    elif args.calc:
        print(f"Calculating equity curve for {args.days} days...")
        data = calc.get_equity_curve_data(args.days)
        if data:
            print(f"\nEquity curve: {len(data)} data points")
            print(f"Start: {data[0][0]} -> ${data[0][1]:+,.2f}")
            print(f"End:   {data[-1][0]} -> ${data[-1][1]:+,.2f}")
        else:
            print("No data available")

    else:
        calc.print_summary()
        print("\nUsage:")
        print("  python equity_calculator.py --import-tws  # Import from TWS")
        print("  python equity_calculator.py --list        # List trades")
        print("  python equity_calculator.py --calc        # Calculate curve")
        print("  python equity_calculator.py --worst       # Show 3 worst performers")
        print("  python equity_calculator.py --worst 5     # Show 5 worst performers")


if __name__ == "__main__":
    main()
