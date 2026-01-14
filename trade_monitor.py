"""
Trade Monitor for Supertrend Trading System.

Provides real-time monitoring of:
- Open and closed trades with entry/exit dates, times, and prices
- Live PnL (Profit and Loss) tracking
- Comprehensive trading statistics
- Capital/equity curve visualization

Can be used with backtesting or extended for live trading with TWS (Interactive Brokers).

Author: Trading System Developer
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
from enum import Enum
import json
import os
import threading
import time

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class TradeStatus(Enum):
    """Trade status enumeration."""
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    PENDING = "PENDING"
    CANCELLED = "CANCELLED"


class TradeDirection(Enum):
    """Trade direction enumeration."""
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class Trade:
    """
    Represents a single trade with full tracking information.
    """
    trade_id: str
    symbol: str
    direction: TradeDirection
    status: TradeStatus

    # Entry information
    entry_date: datetime
    entry_price: float
    entry_quantity: int = 1

    # Exit information (populated when closed)
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None

    # PnL tracking
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    commission: float = 0.0

    # Risk management
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop_pct: Optional[float] = None
    highest_price_since_entry: Optional[float] = None

    # Metadata
    strategy_name: str = "Supertrend"
    notes: str = ""

    def calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized PnL at current price."""
        if self.status != TradeStatus.OPEN:
            return 0.0

        if self.direction == TradeDirection.LONG:
            self.unrealized_pnl = (current_price - self.entry_price) * self.entry_quantity
        else:  # SHORT
            self.unrealized_pnl = (self.entry_price - current_price) * self.entry_quantity

        return self.unrealized_pnl

    def calculate_unrealized_pnl_pct(self, current_price: float) -> float:
        """Calculate unrealized PnL as percentage."""
        if self.status != TradeStatus.OPEN:
            return 0.0

        if self.direction == TradeDirection.LONG:
            return (current_price - self.entry_price) / self.entry_price * 100
        else:
            return (self.entry_price - current_price) / self.entry_price * 100

    def close(self, exit_price: float, exit_date: datetime, exit_reason: str = "signal"):
        """Close the trade and calculate realized PnL."""
        self.exit_price = exit_price
        self.exit_date = exit_date
        self.exit_reason = exit_reason
        self.status = TradeStatus.CLOSED

        if self.direction == TradeDirection.LONG:
            self.realized_pnl = (exit_price - self.entry_price) * self.entry_quantity
        else:
            self.realized_pnl = (self.entry_price - exit_price) * self.entry_quantity

        self.realized_pnl -= self.commission
        self.unrealized_pnl = 0.0

    def duration(self) -> Optional[timedelta]:
        """Calculate trade duration."""
        if self.exit_date:
            return self.exit_date - self.entry_date
        return datetime.now() - self.entry_date

    def to_dict(self) -> Dict:
        """Convert trade to dictionary for serialization."""
        return {
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'direction': self.direction.value,
            'status': self.status.value,
            'entry_date': self.entry_date.isoformat(),
            'entry_price': self.entry_price,
            'entry_quantity': self.entry_quantity,
            'exit_date': self.exit_date.isoformat() if self.exit_date else None,
            'exit_price': self.exit_price,
            'exit_reason': self.exit_reason,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'commission': self.commission,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'strategy_name': self.strategy_name,
            'notes': self.notes
        }


@dataclass
class EquityPoint:
    """Single point on the equity curve."""
    timestamp: datetime
    equity: float
    cash: float
    positions_value: float
    unrealized_pnl: float
    realized_pnl: float


class TradeMonitor:
    """
    Comprehensive trade monitoring system.

    Tracks:
    - Open and closed trades
    - Entry/exit dates, times, prices
    - Real-time PnL
    - Trading statistics
    - Equity/capital curve

    Can be used with backtesting or extended for live trading.
    """

    def __init__(self,
                 initial_capital: float = 10000.0,
                 transaction_cost_pct: float = 0.001,
                 symbol: str = "UNKNOWN"):
        """
        Initialize the trade monitor.

        Args:
            initial_capital: Starting capital
            transaction_cost_pct: Transaction cost as percentage (e.g., 0.001 = 0.1%)
            symbol: Default trading symbol
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.transaction_cost_pct = transaction_cost_pct
        self.symbol = symbol

        # Trade tracking
        self.open_trades: List[Trade] = []
        self.closed_trades: List[Trade] = []
        self.all_trades: List[Trade] = []

        # Equity curve
        self.equity_curve: List[EquityPoint] = []

        # Statistics cache
        self._stats_cache: Optional[Dict] = None
        self._stats_dirty = True

        # Trade ID counter
        self._trade_counter = 0

        # Callbacks for real-time updates
        self._on_trade_open: List[Callable] = []
        self._on_trade_close: List[Callable] = []
        self._on_equity_update: List[Callable] = []

        # Price feed
        self.current_price: float = 0.0
        self.price_history: List[Tuple[datetime, float]] = []

        # Monitoring thread (for live trading)
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None

    def generate_trade_id(self) -> str:
        """Generate unique trade ID."""
        self._trade_counter += 1
        return f"{self.symbol}_{datetime.now().strftime('%Y%m%d')}_{self._trade_counter:04d}"

    def open_trade(self,
                   direction: TradeDirection,
                   entry_price: float,
                   entry_date: Optional[datetime] = None,
                   quantity: int = 1,
                   stop_loss: Optional[float] = None,
                   take_profit: Optional[float] = None,
                   trailing_stop_pct: Optional[float] = None,
                   strategy_name: str = "Supertrend",
                   notes: str = "") -> Trade:
        """
        Open a new trade.

        Args:
            direction: LONG or SHORT
            entry_price: Entry price
            entry_date: Entry datetime (defaults to now)
            quantity: Number of units
            stop_loss: Stop loss price
            take_profit: Take profit price
            trailing_stop_pct: Trailing stop percentage
            strategy_name: Name of the strategy
            notes: Additional notes

        Returns:
            The created Trade object
        """
        trade = Trade(
            trade_id=self.generate_trade_id(),
            symbol=self.symbol,
            direction=direction,
            status=TradeStatus.OPEN,
            entry_date=entry_date or datetime.now(),
            entry_price=entry_price,
            entry_quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop_pct=trailing_stop_pct,
            highest_price_since_entry=entry_price,
            commission=entry_price * quantity * self.transaction_cost_pct,
            strategy_name=strategy_name,
            notes=notes
        )

        self.open_trades.append(trade)
        self.all_trades.append(trade)
        self._stats_dirty = True

        # Trigger callbacks
        for callback in self._on_trade_open:
            callback(trade)

        return trade

    def close_trade(self,
                    trade: Trade,
                    exit_price: float,
                    exit_date: Optional[datetime] = None,
                    exit_reason: str = "signal") -> Trade:
        """
        Close an open trade.

        Args:
            trade: The trade to close
            exit_price: Exit price
            exit_date: Exit datetime (defaults to now)
            exit_reason: Reason for exit

        Returns:
            The closed Trade object
        """
        if trade.status != TradeStatus.OPEN:
            raise ValueError(f"Trade {trade.trade_id} is not open")

        # Add exit commission
        trade.commission += exit_price * trade.entry_quantity * self.transaction_cost_pct

        # Close the trade
        trade.close(exit_price, exit_date or datetime.now(), exit_reason)

        # Move from open to closed
        self.open_trades.remove(trade)
        self.closed_trades.append(trade)

        # Update capital
        self.current_capital += trade.realized_pnl

        self._stats_dirty = True

        # Trigger callbacks
        for callback in self._on_trade_close:
            callback(trade)

        return trade

    def close_trade_by_id(self, trade_id: str, exit_price: float,
                          exit_date: Optional[datetime] = None,
                          exit_reason: str = "signal") -> Optional[Trade]:
        """Close a trade by its ID."""
        for trade in self.open_trades:
            if trade.trade_id == trade_id:
                return self.close_trade(trade, exit_price, exit_date, exit_reason)
        return None

    def close_all_trades(self, exit_price: float,
                         exit_date: Optional[datetime] = None,
                         exit_reason: str = "close_all") -> List[Trade]:
        """Close all open trades."""
        closed = []
        for trade in list(self.open_trades):
            closed.append(self.close_trade(trade, exit_price, exit_date, exit_reason))
        return closed

    def update_price(self, price: float, timestamp: Optional[datetime] = None):
        """
        Update current price and recalculate unrealized PnL.

        Args:
            price: Current market price
            timestamp: Price timestamp
        """
        timestamp = timestamp or datetime.now()
        self.current_price = price
        self.price_history.append((timestamp, price))

        # Update unrealized PnL for all open trades
        total_unrealized = 0.0
        for trade in self.open_trades:
            trade.calculate_unrealized_pnl(price)
            total_unrealized += trade.unrealized_pnl

            # Update trailing stop tracking
            if trade.trailing_stop_pct and trade.direction == TradeDirection.LONG:
                if price > (trade.highest_price_since_entry or 0):
                    trade.highest_price_since_entry = price

        # Record equity point
        positions_value = sum(t.entry_price * t.entry_quantity for t in self.open_trades)
        realized_pnl = sum(t.realized_pnl for t in self.closed_trades)

        equity_point = EquityPoint(
            timestamp=timestamp,
            equity=self.current_capital + total_unrealized,
            cash=self.current_capital,
            positions_value=positions_value,
            unrealized_pnl=total_unrealized,
            realized_pnl=realized_pnl
        )
        self.equity_curve.append(equity_point)

        # Trigger callbacks
        for callback in self._on_equity_update:
            callback(equity_point)

    def check_stop_loss(self, price: float, timestamp: Optional[datetime] = None) -> List[Trade]:
        """Check and execute stop losses."""
        closed = []
        for trade in list(self.open_trades):
            if trade.stop_loss:
                if trade.direction == TradeDirection.LONG and price <= trade.stop_loss:
                    closed.append(self.close_trade(trade, price, timestamp, "stop_loss"))
                elif trade.direction == TradeDirection.SHORT and price >= trade.stop_loss:
                    closed.append(self.close_trade(trade, price, timestamp, "stop_loss"))
        return closed

    def check_trailing_stop(self, price: float, timestamp: Optional[datetime] = None) -> List[Trade]:
        """Check and execute trailing stops."""
        closed = []
        for trade in list(self.open_trades):
            if trade.trailing_stop_pct and trade.direction == TradeDirection.LONG:
                highest = trade.highest_price_since_entry or trade.entry_price
                trailing_stop_price = highest * (1 - trade.trailing_stop_pct)
                if price <= trailing_stop_price:
                    closed.append(self.close_trade(trade, price, timestamp, "trailing_stop"))
        return closed

    def check_take_profit(self, price: float, timestamp: Optional[datetime] = None) -> List[Trade]:
        """Check and execute take profits."""
        closed = []
        for trade in list(self.open_trades):
            if trade.take_profit:
                if trade.direction == TradeDirection.LONG and price >= trade.take_profit:
                    closed.append(self.close_trade(trade, price, timestamp, "take_profit"))
                elif trade.direction == TradeDirection.SHORT and price <= trade.take_profit:
                    closed.append(self.close_trade(trade, price, timestamp, "take_profit"))
        return closed

    def get_statistics(self, force_recalculate: bool = False) -> Dict:
        """
        Calculate comprehensive trading statistics.

        Returns:
            Dictionary with all trading statistics
        """
        if not self._stats_dirty and self._stats_cache and not force_recalculate:
            return self._stats_cache

        all_trades = self.closed_trades
        if not all_trades:
            self._stats_cache = self._empty_statistics()
            self._stats_dirty = False
            return self._stats_cache

        # Basic counts
        total_trades = len(all_trades)
        winning_trades = [t for t in all_trades if t.realized_pnl > 0]
        losing_trades = [t for t in all_trades if t.realized_pnl <= 0]

        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        win_rate = win_count / total_trades if total_trades > 0 else 0

        # Profit metrics
        gross_profit = sum(t.realized_pnl for t in winning_trades)
        gross_loss = abs(sum(t.realized_pnl for t in losing_trades))
        net_profit = gross_profit - gross_loss
        total_commission = sum(t.commission for t in all_trades)

        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Average trade metrics
        avg_profit = np.mean([t.realized_pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.realized_pnl for t in losing_trades]) if losing_trades else 0
        avg_trade = np.mean([t.realized_pnl for t in all_trades])

        # Trade duration
        durations = [t.duration().total_seconds() / 86400 for t in all_trades if t.duration()]  # In days
        avg_duration = np.mean(durations) if durations else 0
        max_duration = np.max(durations) if durations else 0
        min_duration = np.min(durations) if durations else 0

        # Equity curve analysis
        if self.equity_curve:
            equity_values = np.array([e.equity for e in self.equity_curve])
            returns = np.diff(equity_values) / equity_values[:-1]
            returns = returns[~np.isnan(returns)]

            # Max drawdown
            rolling_max = np.maximum.accumulate(equity_values)
            drawdowns = (equity_values - rolling_max) / rolling_max
            max_drawdown = abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0

            # Sharpe Ratio (annualized, assuming daily data)
            if len(returns) > 1 and np.std(returns) > 0:
                sharpe_ratio = np.sqrt(252) * (np.mean(returns) / np.std(returns))
            else:
                sharpe_ratio = 0

            # Sortino Ratio
            negative_returns = returns[returns < 0]
            if len(negative_returns) > 0 and np.std(negative_returns) > 0:
                sortino_ratio = np.sqrt(252) * (np.mean(returns) / np.std(negative_returns))
            else:
                sortino_ratio = 0

            # Calmar Ratio
            total_return = (equity_values[-1] - self.initial_capital) / self.initial_capital
            calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0
        else:
            max_drawdown = 0
            sharpe_ratio = 0
            sortino_ratio = 0
            calmar_ratio = 0
            total_return = (self.current_capital - self.initial_capital) / self.initial_capital

        # Expectancy
        expectancy = (win_rate * avg_profit) - ((1 - win_rate) * abs(avg_loss))

        # Consecutive wins/losses
        results = [1 if t.realized_pnl > 0 else -1 for t in all_trades]
        max_consecutive_wins = self._max_consecutive(results, 1)
        max_consecutive_losses = self._max_consecutive(results, -1)

        # Largest win/loss
        largest_win = max([t.realized_pnl for t in winning_trades]) if winning_trades else 0
        largest_loss = min([t.realized_pnl for t in losing_trades]) if losing_trades else 0

        self._stats_cache = {
            # Count metrics
            "Total Trades": total_trades,
            "Winning Trades": win_count,
            "Losing Trades": loss_count,
            "Win Rate": f"{win_rate:.1%}",
            "Open Trades": len(self.open_trades),

            # Profit metrics
            "Net Profit": f"${net_profit:,.2f}",
            "Gross Profit": f"${gross_profit:,.2f}",
            "Gross Loss": f"${gross_loss:,.2f}",
            "Total Commission": f"${total_commission:,.2f}",
            "Profit Factor": f"{profit_factor:.2f}" if profit_factor != float('inf') else "INF",

            # Trade averages
            "Avg Trade": f"${avg_trade:,.2f}",
            "Avg Profit": f"${avg_profit:,.2f}",
            "Avg Loss": f"${avg_loss:,.2f}",
            "Largest Win": f"${largest_win:,.2f}",
            "Largest Loss": f"${largest_loss:,.2f}",

            # Duration
            "Avg Duration": f"{avg_duration:.1f} days",
            "Max Duration": f"{max_duration:.1f} days",
            "Min Duration": f"{min_duration:.1f} days",

            # Risk metrics
            "Total Return": f"{total_return:.2%}",
            "Max Drawdown": f"{max_drawdown:.2%}",
            "Sharpe Ratio": f"{sharpe_ratio:.2f}",
            "Sortino Ratio": f"{sortino_ratio:.2f}",
            "Calmar Ratio": f"{calmar_ratio:.2f}",
            "Expectancy": f"${expectancy:,.2f}",

            # Streaks
            "Max Consecutive Wins": max_consecutive_wins,
            "Max Consecutive Losses": max_consecutive_losses,

            # Capital
            "Initial Capital": f"${self.initial_capital:,.2f}",
            "Current Capital": f"${self.current_capital:,.2f}",
            "Unrealized PnL": f"${sum(t.unrealized_pnl for t in self.open_trades):,.2f}",
        }

        self._stats_dirty = False
        return self._stats_cache

    def _max_consecutive(self, results: List[int], value: int) -> int:
        """Calculate max consecutive occurrences of value."""
        max_count = 0
        current_count = 0
        for r in results:
            if r == value:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
        return max_count

    def _empty_statistics(self) -> Dict:
        """Return empty statistics dictionary."""
        return {
            "Total Trades": 0,
            "Winning Trades": 0,
            "Losing Trades": 0,
            "Win Rate": "0.0%",
            "Open Trades": len(self.open_trades),
            "Net Profit": "$0.00",
            "Gross Profit": "$0.00",
            "Gross Loss": "$0.00",
            "Total Commission": "$0.00",
            "Profit Factor": "0.00",
            "Avg Trade": "$0.00",
            "Avg Profit": "$0.00",
            "Avg Loss": "$0.00",
            "Largest Win": "$0.00",
            "Largest Loss": "$0.00",
            "Avg Duration": "0.0 days",
            "Max Duration": "0.0 days",
            "Min Duration": "0.0 days",
            "Total Return": "0.00%",
            "Max Drawdown": "0.00%",
            "Sharpe Ratio": "0.00",
            "Sortino Ratio": "0.00",
            "Calmar Ratio": "0.00",
            "Expectancy": "$0.00",
            "Max Consecutive Wins": 0,
            "Max Consecutive Losses": 0,
            "Initial Capital": f"${self.initial_capital:,.2f}",
            "Current Capital": f"${self.current_capital:,.2f}",
            "Unrealized PnL": "$0.00",
        }

    def get_open_trades_summary(self) -> pd.DataFrame:
        """Get summary of open trades as DataFrame."""
        if not self.open_trades:
            return pd.DataFrame()

        data = []
        for trade in self.open_trades:
            unrealized_pnl_pct = trade.calculate_unrealized_pnl_pct(self.current_price)
            data.append({
                'Trade ID': trade.trade_id,
                'Direction': trade.direction.value,
                'Entry Date': trade.entry_date.strftime('%Y-%m-%d %H:%M'),
                'Entry Price': f"${trade.entry_price:.2f}",
                'Current Price': f"${self.current_price:.2f}",
                'Quantity': trade.entry_quantity,
                'Unrealized PnL': f"${trade.unrealized_pnl:,.2f}",
                'Unrealized PnL %': f"{unrealized_pnl_pct:.2f}%",
                'Duration': str(trade.duration()).split('.')[0] if trade.duration() else "N/A",
                'Stop Loss': f"${trade.stop_loss:.2f}" if trade.stop_loss else "None",
                'Take Profit': f"${trade.take_profit:.2f}" if trade.take_profit else "None",
            })

        return pd.DataFrame(data)

    def get_closed_trades_summary(self, last_n: Optional[int] = None) -> pd.DataFrame:
        """Get summary of closed trades as DataFrame."""
        trades = self.closed_trades[-last_n:] if last_n else self.closed_trades

        if not trades:
            return pd.DataFrame()

        data = []
        for trade in trades:
            pnl_pct = (trade.realized_pnl / (trade.entry_price * trade.entry_quantity)) * 100
            data.append({
                'Trade ID': trade.trade_id,
                'Direction': trade.direction.value,
                'Entry Date': trade.entry_date.strftime('%Y-%m-%d %H:%M'),
                'Entry Price': f"${trade.entry_price:.2f}",
                'Exit Date': trade.exit_date.strftime('%Y-%m-%d %H:%M') if trade.exit_date else "N/A",
                'Exit Price': f"${trade.exit_price:.2f}" if trade.exit_price else "N/A",
                'Quantity': trade.entry_quantity,
                'Realized PnL': f"${trade.realized_pnl:,.2f}",
                'PnL %': f"{pnl_pct:.2f}%",
                'Exit Reason': trade.exit_reason or "N/A",
                'Duration': str(trade.duration()).split('.')[0] if trade.duration() else "N/A",
                'Commission': f"${trade.commission:.2f}",
            })

        return pd.DataFrame(data)

    def get_equity_curve_df(self) -> pd.DataFrame:
        """Get equity curve as DataFrame."""
        if not self.equity_curve:
            return pd.DataFrame()

        data = []
        for point in self.equity_curve:
            data.append({
                'Timestamp': point.timestamp,
                'Equity': point.equity,
                'Cash': point.cash,
                'Positions Value': point.positions_value,
                'Unrealized PnL': point.unrealized_pnl,
                'Realized PnL': point.realized_pnl,
            })

        return pd.DataFrame(data).set_index('Timestamp')

    def print_summary(self):
        """Print comprehensive trading summary to console."""
        print("\n" + "=" * 80)
        print("TRADE MONITOR SUMMARY")
        print("=" * 80)

        stats = self.get_statistics()

        print(f"\n{'='*40}")
        print("CAPITAL & RETURNS")
        print(f"{'='*40}")
        print(f"  Initial Capital:    {stats['Initial Capital']}")
        print(f"  Current Capital:    {stats['Current Capital']}")
        print(f"  Total Return:       {stats['Total Return']}")
        print(f"  Unrealized PnL:     {stats['Unrealized PnL']}")

        print(f"\n{'='*40}")
        print("TRADE COUNTS")
        print(f"{'='*40}")
        print(f"  Total Trades:       {stats['Total Trades']}")
        print(f"  Winning Trades:     {stats['Winning Trades']}")
        print(f"  Losing Trades:      {stats['Losing Trades']}")
        print(f"  Open Trades:        {stats['Open Trades']}")
        print(f"  Win Rate:           {stats['Win Rate']}")

        print(f"\n{'='*40}")
        print("PROFIT METRICS")
        print(f"{'='*40}")
        print(f"  Net Profit:         {stats['Net Profit']}")
        print(f"  Gross Profit:       {stats['Gross Profit']}")
        print(f"  Gross Loss:         {stats['Gross Loss']}")
        print(f"  Profit Factor:      {stats['Profit Factor']}")
        print(f"  Expectancy:         {stats['Expectancy']}")

        print(f"\n{'='*40}")
        print("TRADE AVERAGES")
        print(f"{'='*40}")
        print(f"  Avg Trade:          {stats['Avg Trade']}")
        print(f"  Avg Profit:         {stats['Avg Profit']}")
        print(f"  Avg Loss:           {stats['Avg Loss']}")
        print(f"  Largest Win:        {stats['Largest Win']}")
        print(f"  Largest Loss:       {stats['Largest Loss']}")

        print(f"\n{'='*40}")
        print("RISK METRICS")
        print(f"{'='*40}")
        print(f"  Max Drawdown:       {stats['Max Drawdown']}")
        print(f"  Sharpe Ratio:       {stats['Sharpe Ratio']}")
        print(f"  Sortino Ratio:      {stats['Sortino Ratio']}")
        print(f"  Calmar Ratio:       {stats['Calmar Ratio']}")

        print(f"\n{'='*40}")
        print("DURATION & STREAKS")
        print(f"{'='*40}")
        print(f"  Avg Duration:       {stats['Avg Duration']}")
        print(f"  Max Consec. Wins:   {stats['Max Consecutive Wins']}")
        print(f"  Max Consec. Losses: {stats['Max Consecutive Losses']}")

        # Open trades
        if self.open_trades:
            print(f"\n{'='*40}")
            print("OPEN TRADES")
            print(f"{'='*40}")
            df = self.get_open_trades_summary()
            print(df.to_string(index=False))

        # Recent closed trades
        if self.closed_trades:
            print(f"\n{'='*40}")
            print("RECENT CLOSED TRADES (Last 5)")
            print(f"{'='*40}")
            df = self.get_closed_trades_summary(last_n=5)
            print(df.to_string(index=False))

        print("\n" + "=" * 80)

    def plot_equity_curve(self, show: bool = True, save_path: Optional[str] = None) -> Optional[go.Figure]:
        """
        Plot the equity curve with Plotly.

        Args:
            show: Whether to display the plot
            save_path: Path to save HTML file

        Returns:
            Plotly figure object
        """
        if not PLOTLY_AVAILABLE:
            print("Plotly not available. Install with: pip install plotly")
            return None

        if not self.equity_curve:
            print("No equity curve data available")
            return None

        df = self.get_equity_curve_df()

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(
                'Equity Curve',
                'Unrealized PnL',
                'Drawdown'
            ),
            row_heights=[0.5, 0.25, 0.25]
        )

        # Equity curve
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Equity'],
                mode='lines',
                name='Total Equity',
                line=dict(color='#2196F3', width=2)
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Cash'],
                mode='lines',
                name='Cash',
                line=dict(color='#4CAF50', width=1, dash='dash')
            ),
            row=1, col=1
        )

        # Initial capital reference line
        fig.add_hline(
            y=self.initial_capital,
            line_dash="dot",
            line_color="gray",
            annotation_text=f"Initial: ${self.initial_capital:,.0f}",
            row=1, col=1
        )

        # Unrealized PnL
        colors = ['#4CAF50' if x >= 0 else '#F44336' for x in df['Unrealized PnL']]
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['Unrealized PnL'],
                name='Unrealized PnL',
                marker_color=colors
            ),
            row=2, col=1
        )

        # Drawdown
        equity_values = df['Equity'].values
        rolling_max = np.maximum.accumulate(equity_values)
        drawdown = (equity_values - rolling_max) / rolling_max * 100

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=drawdown,
                mode='lines',
                name='Drawdown %',
                fill='tozeroy',
                line=dict(color='#F44336', width=1),
                fillcolor='rgba(244, 67, 54, 0.3)'
            ),
            row=3, col=1
        )

        # Layout
        fig.update_layout(
            title=dict(
                text=f'<b>{self.symbol} Trade Monitor - Equity Curve</b>',
                x=0.5,
                font=dict(size=20)
            ),
            height=800,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.02
            ),
            template='plotly_white',
            hovermode='x unified'
        )

        fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
        fig.update_yaxes(title_text="Unrealized PnL ($)", row=2, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=3, col=1)

        if save_path:
            fig.write_html(save_path)
            print(f"Chart saved to: {save_path}")

        if show:
            fig.show()

        return fig

    def export_trades(self, filepath: str, format: str = 'csv'):
        """
        Export trades to file.

        Args:
            filepath: Output file path
            format: 'csv' or 'json'
        """
        all_trades_data = [t.to_dict() for t in self.all_trades]

        if format == 'csv':
            df = pd.DataFrame(all_trades_data)
            df.to_csv(filepath, index=False)
        elif format == 'json':
            with open(filepath, 'w') as f:
                json.dump(all_trades_data, f, indent=2, default=str)

        print(f"Exported {len(all_trades_data)} trades to {filepath}")

    def import_from_backtest(self, long_trades: List[Dict], short_trades: List[Dict],
                              equity_curve: np.ndarray, dates: pd.DatetimeIndex):
        """
        Import trades from a backtest result.

        Args:
            long_trades: List of long trade dictionaries from backtest
            short_trades: List of short trade dictionaries from backtest
            equity_curve: Numpy array of equity values
            dates: DatetimeIndex for the equity curve
        """
        # Import long trades
        for trade in long_trades:
            t = Trade(
                trade_id=self.generate_trade_id(),
                symbol=self.symbol,
                direction=TradeDirection.LONG,
                status=TradeStatus.CLOSED,
                entry_date=trade['entry_date'],
                entry_price=trade['entry_price'],
                exit_date=trade['exit_date'],
                exit_price=trade['exit_price'],
                exit_reason=trade.get('exit_reason', 'signal'),
                realized_pnl=trade['profit_loss'] * trade['entry_price'],
                commission=trade['entry_price'] * 2 * self.transaction_cost_pct
            )
            self.closed_trades.append(t)
            self.all_trades.append(t)

        # Import short trades
        for trade in short_trades:
            t = Trade(
                trade_id=self.generate_trade_id(),
                symbol=self.symbol,
                direction=TradeDirection.SHORT,
                status=TradeStatus.CLOSED,
                entry_date=trade['entry_date'],
                entry_price=trade['entry_price'],
                exit_date=trade['exit_date'],
                exit_price=trade['exit_price'],
                exit_reason=trade.get('exit_reason', 'signal'),
                realized_pnl=trade['profit_loss'] * trade['entry_price'],
                commission=trade['entry_price'] * 2 * self.transaction_cost_pct
            )
            self.closed_trades.append(t)
            self.all_trades.append(t)

        # Import equity curve
        for i, (date, equity) in enumerate(zip(dates, equity_curve)):
            point = EquityPoint(
                timestamp=date,
                equity=equity,
                cash=equity,
                positions_value=0,
                unrealized_pnl=0,
                realized_pnl=equity - self.initial_capital
            )
            self.equity_curve.append(point)

        self.current_capital = equity_curve[-1] if len(equity_curve) > 0 else self.initial_capital
        self._stats_dirty = True

    # Callback registration
    def on_trade_open(self, callback: Callable[[Trade], None]):
        """Register callback for trade open events."""
        self._on_trade_open.append(callback)

    def on_trade_close(self, callback: Callable[[Trade], None]):
        """Register callback for trade close events."""
        self._on_trade_close.append(callback)

    def on_equity_update(self, callback: Callable[[EquityPoint], None]):
        """Register callback for equity update events."""
        self._on_equity_update.append(callback)


def create_monitor_from_backtest(symbol: str,
                                  long_trades: List[Dict],
                                  short_trades: List[Dict],
                                  equity_curve: np.ndarray,
                                  dates: pd.DatetimeIndex,
                                  initial_capital: float = 10000.0) -> TradeMonitor:
    """
    Create a TradeMonitor from backtest results.

    Args:
        symbol: Trading symbol
        long_trades: List of long trade dicts
        short_trades: List of short trade dicts
        equity_curve: Equity values array
        dates: Date index
        initial_capital: Initial capital

    Returns:
        Configured TradeMonitor instance
    """
    monitor = TradeMonitor(
        initial_capital=initial_capital,
        symbol=symbol
    )
    monitor.import_from_backtest(long_trades, short_trades, equity_curve, dates)
    return monitor


# Example usage and testing
if __name__ == "__main__":
    print("Trade Monitor Demo")
    print("=" * 50)

    # Create monitor
    monitor = TradeMonitor(
        initial_capital=10000.0,
        transaction_cost_pct=0.001,
        symbol="DEMO"
    )

    # Simulate some trades
    print("\nSimulating trades...")

    # Open a long trade
    trade1 = monitor.open_trade(
        direction=TradeDirection.LONG,
        entry_price=100.0,
        entry_date=datetime.now() - timedelta(days=10),
        quantity=50,
        trailing_stop_pct=0.05,
        notes="First demo trade"
    )
    print(f"Opened trade: {trade1.trade_id}")

    # Update price a few times
    for i, price in enumerate([101, 103, 105, 108, 106, 104]):
        monitor.update_price(price, datetime.now() - timedelta(days=9-i))

    # Close the trade
    monitor.close_trade(trade1, exit_price=104.0, exit_reason="signal")
    print(f"Closed trade: {trade1.trade_id}, PnL: ${trade1.realized_pnl:.2f}")

    # Open and close another trade
    trade2 = monitor.open_trade(
        direction=TradeDirection.LONG,
        entry_price=105.0,
        entry_date=datetime.now() - timedelta(days=5),
        quantity=50
    )
    monitor.update_price(103.0)
    monitor.close_trade(trade2, exit_price=103.0, exit_reason="stop_loss")
    print(f"Closed trade: {trade2.trade_id}, PnL: ${trade2.realized_pnl:.2f}")

    # Open a winning trade
    trade3 = monitor.open_trade(
        direction=TradeDirection.LONG,
        entry_price=102.0,
        entry_date=datetime.now() - timedelta(days=2),
        quantity=50
    )
    for price in [103, 106, 110, 115]:
        monitor.update_price(price)
    monitor.close_trade(trade3, exit_price=115.0, exit_reason="take_profit")
    print(f"Closed trade: {trade3.trade_id}, PnL: ${trade3.realized_pnl:.2f}")

    # Print summary
    monitor.print_summary()

    # Show closed trades
    print("\n\nClosed Trades DataFrame:")
    print(monitor.get_closed_trades_summary())

    # Export trades
    monitor.export_trades("demo_trades.csv", format='csv')

    # Plot equity curve if plotly available
    if PLOTLY_AVAILABLE:
        monitor.plot_equity_curve(show=False, save_path="demo_equity_curve.html")
