"""
Tests for the Trade Monitor system.
Tests trade tracking, PnL calculations, statistics, and equity curves.
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trade_monitor import (
    TradeMonitor,
    Trade,
    TradeStatus,
    TradeDirection,
    EquityPoint,
    create_monitor_from_backtest
)


class TestTrade:
    """Tests for the Trade dataclass."""

    def test_trade_creation(self):
        """Test basic trade creation."""
        trade = Trade(
            trade_id="TEST_001",
            symbol="AAPL",
            direction=TradeDirection.LONG,
            status=TradeStatus.OPEN,
            entry_date=datetime.now(),
            entry_price=150.0,
            entry_quantity=10
        )

        assert trade.trade_id == "TEST_001"
        assert trade.symbol == "AAPL"
        assert trade.direction == TradeDirection.LONG
        assert trade.status == TradeStatus.OPEN
        assert trade.entry_price == 150.0
        assert trade.entry_quantity == 10

    def test_calculate_unrealized_pnl_long(self):
        """Test unrealized PnL for long trade."""
        trade = Trade(
            trade_id="TEST_001",
            symbol="AAPL",
            direction=TradeDirection.LONG,
            status=TradeStatus.OPEN,
            entry_date=datetime.now(),
            entry_price=100.0,
            entry_quantity=10
        )

        # Price up 10%
        pnl = trade.calculate_unrealized_pnl(110.0)
        assert pnl == 100.0  # (110 - 100) * 10 = 100

        # Price down 5%
        pnl = trade.calculate_unrealized_pnl(95.0)
        assert pnl == -50.0  # (95 - 100) * 10 = -50

    def test_calculate_unrealized_pnl_short(self):
        """Test unrealized PnL for short trade."""
        trade = Trade(
            trade_id="TEST_001",
            symbol="AAPL",
            direction=TradeDirection.SHORT,
            status=TradeStatus.OPEN,
            entry_date=datetime.now(),
            entry_price=100.0,
            entry_quantity=10
        )

        # Price down 10% (profitable for short)
        pnl = trade.calculate_unrealized_pnl(90.0)
        assert pnl == 100.0  # (100 - 90) * 10 = 100

        # Price up 5% (loss for short)
        pnl = trade.calculate_unrealized_pnl(105.0)
        assert pnl == -50.0  # (100 - 105) * 10 = -50

    def test_close_trade_long(self):
        """Test closing a long trade."""
        trade = Trade(
            trade_id="TEST_001",
            symbol="AAPL",
            direction=TradeDirection.LONG,
            status=TradeStatus.OPEN,
            entry_date=datetime.now() - timedelta(days=5),
            entry_price=100.0,
            entry_quantity=10
        )

        exit_date = datetime.now()
        trade.close(exit_price=120.0, exit_date=exit_date, exit_reason="signal")

        assert trade.status == TradeStatus.CLOSED
        assert trade.exit_price == 120.0
        assert trade.exit_date == exit_date
        assert trade.exit_reason == "signal"
        assert trade.realized_pnl == 200.0  # (120 - 100) * 10

    def test_trade_duration(self):
        """Test trade duration calculation."""
        entry_date = datetime.now() - timedelta(days=5, hours=3)
        trade = Trade(
            trade_id="TEST_001",
            symbol="AAPL",
            direction=TradeDirection.LONG,
            status=TradeStatus.OPEN,
            entry_date=entry_date,
            entry_price=100.0
        )

        duration = trade.duration()
        assert duration.days >= 5

    def test_trade_to_dict(self):
        """Test trade serialization."""
        trade = Trade(
            trade_id="TEST_001",
            symbol="AAPL",
            direction=TradeDirection.LONG,
            status=TradeStatus.OPEN,
            entry_date=datetime.now(),
            entry_price=100.0
        )

        data = trade.to_dict()

        assert 'trade_id' in data
        assert 'symbol' in data
        assert 'direction' in data
        assert data['direction'] == "LONG"


class TestTradeMonitor:
    """Tests for the TradeMonitor class."""

    @pytest.fixture
    def monitor(self):
        """Create a fresh TradeMonitor for each test."""
        return TradeMonitor(
            initial_capital=10000.0,
            transaction_cost_pct=0.001,
            symbol="TEST"
        )

    def test_monitor_initialization(self, monitor):
        """Test monitor initializes correctly."""
        assert monitor.initial_capital == 10000.0
        assert monitor.current_capital == 10000.0
        assert monitor.symbol == "TEST"
        assert len(monitor.open_trades) == 0
        assert len(monitor.closed_trades) == 0

    def test_open_trade(self, monitor):
        """Test opening a trade."""
        trade = monitor.open_trade(
            direction=TradeDirection.LONG,
            entry_price=100.0,
            quantity=10
        )

        assert len(monitor.open_trades) == 1
        assert trade.status == TradeStatus.OPEN
        assert trade in monitor.all_trades

    def test_close_trade(self, monitor):
        """Test closing a trade."""
        trade = monitor.open_trade(
            direction=TradeDirection.LONG,
            entry_price=100.0,
            quantity=10
        )

        monitor.close_trade(trade, exit_price=110.0, exit_reason="signal")

        assert len(monitor.open_trades) == 0
        assert len(monitor.closed_trades) == 1
        assert trade.status == TradeStatus.CLOSED
        assert trade.realized_pnl > 0

    def test_capital_updates_on_close(self, monitor):
        """Test capital updates when trade is closed."""
        initial = monitor.current_capital

        trade = monitor.open_trade(
            direction=TradeDirection.LONG,
            entry_price=100.0,
            quantity=10
        )

        # Win $100 (10% on 10 shares at $100)
        monitor.close_trade(trade, exit_price=110.0)

        # Capital should increase (minus commission)
        assert monitor.current_capital > initial

    def test_update_price(self, monitor):
        """Test price updates."""
        trade = monitor.open_trade(
            direction=TradeDirection.LONG,
            entry_price=100.0,
            quantity=10
        )

        monitor.update_price(110.0, symbol=trade.symbol)

        assert monitor.current_prices.get(trade.symbol) == 110.0
        assert trade.current_price == 110.0
        assert trade.unrealized_pnl == 100.0
        assert len(monitor.equity_curve) == 1

    def test_stop_loss_trigger(self, monitor):
        """Test stop loss triggers correctly."""
        trade = monitor.open_trade(
            direction=TradeDirection.LONG,
            entry_price=100.0,
            quantity=10,
            stop_loss=95.0
        )

        # Price above stop - should not trigger
        closed = monitor.check_stop_loss(98.0)
        assert len(closed) == 0

        # Price at stop - should trigger
        closed = monitor.check_stop_loss(94.0)
        assert len(closed) == 1
        assert closed[0].exit_reason == "stop_loss"

    def test_trailing_stop(self, monitor):
        """Test trailing stop triggers correctly."""
        trade = monitor.open_trade(
            direction=TradeDirection.LONG,
            entry_price=100.0,
            quantity=10,
            trailing_stop_pct=0.05  # 5%
        )

        # Price goes up to 120, highest tracked
        monitor.update_price(120.0)
        assert trade.highest_price_since_entry == 120.0

        # Price drops but not to trailing stop (120 * 0.95 = 114)
        closed = monitor.check_trailing_stop(116.0)
        assert len(closed) == 0

        # Price drops to trailing stop level
        closed = monitor.check_trailing_stop(113.0)
        assert len(closed) == 1
        assert closed[0].exit_reason == "trailing_stop"

    def test_take_profit_trigger(self, monitor):
        """Test take profit triggers correctly."""
        trade = monitor.open_trade(
            direction=TradeDirection.LONG,
            entry_price=100.0,
            quantity=10,
            take_profit=120.0
        )

        # Price below target
        closed = monitor.check_take_profit(115.0)
        assert len(closed) == 0

        # Price at target
        closed = monitor.check_take_profit(121.0)
        assert len(closed) == 1
        assert closed[0].exit_reason == "take_profit"

    def test_close_all_trades(self, monitor):
        """Test closing all open trades."""
        monitor.open_trade(TradeDirection.LONG, 100.0, quantity=10)
        monitor.open_trade(TradeDirection.LONG, 105.0, quantity=5)
        monitor.open_trade(TradeDirection.SHORT, 110.0, quantity=8)

        assert len(monitor.open_trades) == 3

        closed = monitor.close_all_trades(exit_price=108.0)

        assert len(closed) == 3
        assert len(monitor.open_trades) == 0
        assert len(monitor.closed_trades) == 3


class TestTradeMonitorStatistics:
    """Tests for statistics calculation."""

    @pytest.fixture
    def monitor_with_trades(self):
        """Create monitor with sample trades."""
        monitor = TradeMonitor(initial_capital=10000.0, symbol="TEST")

        # Create some trades with varied results
        trades_data = [
            (100.0, 110.0, "signal"),  # Win 10%
            (108.0, 100.0, "stop_loss"),  # Loss ~7.4%
            (102.0, 115.0, "signal"),  # Win ~12.7%
            (112.0, 108.0, "signal"),  # Loss ~3.6%
            (105.0, 120.0, "take_profit"),  # Win ~14.3%
        ]

        for i, (entry, exit_p, reason) in enumerate(trades_data):
            trade = monitor.open_trade(
                TradeDirection.LONG,
                entry_price=entry,
                entry_date=datetime.now() - timedelta(days=20-i*3),
                quantity=10
            )
            monitor.update_price(exit_p)
            monitor.close_trade(
                trade,
                exit_price=exit_p,
                exit_date=datetime.now() - timedelta(days=18-i*3),
                exit_reason=reason
            )

        return monitor

    def test_statistics_structure(self, monitor_with_trades):
        """Test statistics returns all required fields."""
        stats = monitor_with_trades.get_statistics()

        required_fields = [
            'Total Trades', 'Winning Trades', 'Losing Trades', 'Win Rate',
            'Net Profit', 'Profit Factor', 'Max Drawdown', 'Sharpe Ratio'
        ]

        for field in required_fields:
            assert field in stats

    def test_win_rate_calculation(self, monitor_with_trades):
        """Test win rate is correctly calculated."""
        stats = monitor_with_trades.get_statistics()

        # 3 wins out of 5 trades = 60%
        assert "60" in stats['Win Rate']

    def test_trade_counts(self, monitor_with_trades):
        """Test trade counts are correct."""
        stats = monitor_with_trades.get_statistics()

        assert stats['Total Trades'] == 5
        assert stats['Winning Trades'] == 3
        assert stats['Losing Trades'] == 2

    def test_empty_trades_statistics(self):
        """Test statistics with no trades."""
        monitor = TradeMonitor(initial_capital=10000.0, symbol="TEST")
        stats = monitor.get_statistics()

        assert stats['Total Trades'] == 0
        assert stats['Win Rate'] == "0.0%"


class TestTradeMonitorDataFrames:
    """Tests for DataFrame outputs."""

    @pytest.fixture
    def monitor_with_data(self):
        """Create monitor with sample data."""
        monitor = TradeMonitor(initial_capital=10000.0, symbol="TEST")

        # Open trade
        monitor.open_trade(
            TradeDirection.LONG,
            entry_price=100.0,
            quantity=10
        )
        monitor.update_price(105.0, symbol="TEST")

        # Closed trade
        trade = monitor.open_trade(
            TradeDirection.LONG,
            entry_price=95.0,
            quantity=5,
            entry_date=datetime.now() - timedelta(days=5)
        )
        monitor.close_trade(trade, 102.0, exit_reason="signal")

        return monitor

    def test_open_trades_summary(self, monitor_with_data):
        """Test open trades summary DataFrame."""
        df = monitor_with_data.get_open_trades_summary()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert 'Trade ID' in df.columns
        assert 'Unrealized PnL' in df.columns
        assert 'Symbol' in df.columns  # New column

    def test_closed_trades_summary(self, monitor_with_data):
        """Test closed trades summary DataFrame."""
        df = monitor_with_data.get_closed_trades_summary()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert 'Realized PnL' in df.columns
        assert 'Exit Reason' in df.columns

    def test_equity_curve_df(self, monitor_with_data):
        """Test equity curve DataFrame."""
        df = monitor_with_data.get_equity_curve_df()

        assert isinstance(df, pd.DataFrame)
        assert 'Equity' in df.columns
        assert 'Cash' in df.columns


class TestTradeMonitorCallbacks:
    """Tests for callback functionality."""

    def test_on_trade_open_callback(self):
        """Test trade open callback is triggered."""
        monitor = TradeMonitor(initial_capital=10000.0, symbol="TEST")
        callback_triggered = []

        def on_open(trade):
            callback_triggered.append(trade.trade_id)

        monitor.on_trade_open(on_open)
        monitor.open_trade(TradeDirection.LONG, 100.0)

        assert len(callback_triggered) == 1

    def test_on_trade_close_callback(self):
        """Test trade close callback is triggered."""
        monitor = TradeMonitor(initial_capital=10000.0, symbol="TEST")
        callback_triggered = []

        def on_close(trade):
            callback_triggered.append(trade.realized_pnl)

        monitor.on_trade_close(on_close)

        trade = monitor.open_trade(TradeDirection.LONG, 100.0, quantity=10)
        monitor.close_trade(trade, 110.0)

        assert len(callback_triggered) == 1
        assert callback_triggered[0] > 0


class TestCreateMonitorFromBacktest:
    """Tests for backtest import functionality."""

    def test_import_from_backtest(self):
        """Test importing backtest results."""
        long_trades = [
            {
                'entry_date': datetime.now() - timedelta(days=30),
                'entry_price': 100.0,
                'exit_date': datetime.now() - timedelta(days=20),
                'exit_price': 110.0,
                'profit_loss': 0.10,
                'exit_reason': 'signal'
            }
        ]
        short_trades = []

        equity = np.linspace(10000, 11000, 30)
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')

        monitor = create_monitor_from_backtest(
            symbol="TEST",
            long_trades=long_trades,
            short_trades=short_trades,
            equity_curve=equity,
            dates=dates
        )

        assert len(monitor.closed_trades) == 1
        assert len(monitor.equity_curve) == 30
        assert monitor.current_capital == 11000.0
