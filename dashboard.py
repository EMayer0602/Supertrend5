"""
Trade Monitor HTML Dashboard

Creates a comprehensive HTML dashboard showing:
- Account summary and capital
- Open positions with real-time PnL
- Closed trades history
- Trading statistics
- Equity curve chart
- Performance metrics

Can connect to TWS for live data or use backtest results.

Usage:
    python dashboard.py              # Demo mode
    python dashboard.py --tws        # Connect to TWS
    python dashboard.py --port 4001  # Use IB Gateway port
"""

import os
import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import webbrowser
import argparse

import pandas as pd
import numpy as np

# PnL history for capital curve
try:
    from pnl_history import PnLHistory
    PNL_HISTORY_AVAILABLE = True
except ImportError:
    PNL_HISTORY_AVAILABLE = False

# Historical equity calculator
try:
    from equity_calculator import EquityCurveCalculator
    EQUITY_CALC_AVAILABLE = True
except ImportError:
    EQUITY_CALC_AVAILABLE = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from trade_monitor import (
    TradeMonitor, Trade, TradeDirection, TradeStatus,
    EquityPoint, create_monitor_from_backtest
)


class TradeDashboard:
    """
    HTML Dashboard for Trade Monitor.

    Features:
    - Real-time position tracking
    - PnL visualization
    - Trading statistics
    - Equity curve
    - Auto-refresh capability
    """

    def __init__(self, monitor: TradeMonitor, title: str = "Trade Monitor Dashboard",
                 ib_port: int = 7497, ib_connection=None):
        """
        Initialize dashboard.

        Args:
            monitor: TradeMonitor instance
            title: Dashboard title
            ib_port: TWS/IB Gateway port for historical data
            ib_connection: Existing IB connection to reuse
        """
        self.monitor = monitor
        self.title = title
        self.last_update = datetime.now()
        self.ib_port = ib_port
        self.ib_connection = ib_connection
        self._equity_calc = None

    def generate_html(self, auto_refresh: int = 0) -> str:
        """
        Generate complete HTML dashboard.

        Args:
            auto_refresh: Auto-refresh interval in seconds (0 = disabled)

        Returns:
            HTML string
        """
        stats = self.monitor.get_statistics()

        # Build HTML sections
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.title}</title>
    {"<meta http-equiv='refresh' content='" + str(auto_refresh) + "'>" if auto_refresh > 0 else ""}
    <style>
        {self._get_css()}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{self.title}</h1>
            <div class="update-time">Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        </header>

        {self._generate_summary_cards(stats)}

        <div class="row">
            <div class="col-6">
                {self._generate_capital_curve()}
            </div>
            <div class="col-6">
                {self._generate_daily_pnl_curve()}
            </div>
        </div>

        <div class="row">
            <div class="col-8">
                {self._generate_equity_chart()}
            </div>
            <div class="col-4">
                {self._generate_statistics_panel(stats)}
            </div>
        </div>

        <div class="row">
            <div class="col-12">
                {self._generate_open_positions_table()}
            </div>
        </div>

        <div class="row">
            <div class="col-12">
                {self._generate_closed_trades_table()}
            </div>
        </div>

        <div class="row">
            <div class="col-6">
                {self._generate_pnl_chart()}
            </div>
            <div class="col-6">
                {self._generate_win_loss_chart(stats)}
            </div>
        </div>

        <footer>
            <p>Trade Monitor Dashboard | Symbol: {self.monitor.symbol} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </footer>
    </div>

    <script>
        {self._get_javascript()}
    </script>
</body>
</html>
"""
        return html

    def _get_css(self) -> str:
        """Get CSS styles."""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e0e0e0;
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1800px;
            margin: 0 auto;
        }

        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 30px;
            padding-bottom: 15px;
            border-bottom: 2px solid #0f3460;
        }

        header h1 {
            color: #00d9ff;
            font-size: 2em;
            text-shadow: 0 0 10px rgba(0, 217, 255, 0.3);
        }

        .update-time {
            color: #888;
            font-size: 0.9em;
        }

        .row {
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
        }

        .col-4 { flex: 0 0 33.333%; }
        .col-6 { flex: 0 0 calc(50% - 10px); }
        .col-8 { flex: 0 0 66.666%; }
        .col-12 { flex: 0 0 100%; }

        .card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }

        .card h2 {
            color: #00d9ff;
            margin-bottom: 15px;
            font-size: 1.2em;
            border-bottom: 1px solid rgba(0, 217, 255, 0.3);
            padding-bottom: 10px;
        }

        .summary-cards {
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
        }

        .summary-card {
            flex: 1;
            background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .summary-card .label {
            font-size: 0.85em;
            color: #888;
            margin-bottom: 5px;
        }

        .summary-card .value {
            font-size: 1.8em;
            font-weight: bold;
        }

        .summary-card .value.positive { color: #00e676; }
        .summary-card .value.negative { color: #ff5252; }
        .summary-card .value.neutral { color: #00d9ff; }

        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }

        th, td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        th {
            background: rgba(0, 217, 255, 0.1);
            color: #00d9ff;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.8em;
            letter-spacing: 1px;
        }

        tr:hover {
            background: rgba(255, 255, 255, 0.05);
        }

        .pnl-positive { color: #00e676; }
        .pnl-negative { color: #ff5252; }

        .stat-row {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }

        .stat-row:last-child {
            border-bottom: none;
        }

        .stat-label {
            color: #888;
        }

        .stat-value {
            font-weight: 600;
            color: #e0e0e0;
        }

        .chart-container {
            height: 400px;
            margin-top: 15px;
        }

        .badge {
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.75em;
            font-weight: 600;
        }

        .badge-long {
            background: rgba(0, 230, 118, 0.2);
            color: #00e676;
        }

        .badge-short {
            background: rgba(255, 82, 82, 0.2);
            color: #ff5252;
        }

        .badge-open {
            background: rgba(0, 217, 255, 0.2);
            color: #00d9ff;
        }

        .badge-closed {
            background: rgba(136, 136, 136, 0.2);
            color: #888;
        }

        footer {
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.85em;
            margin-top: 30px;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
        }

        @media (max-width: 1200px) {
            .row { flex-wrap: wrap; }
            .col-4, .col-6, .col-8 { flex: 0 0 100%; }
        }
        """

    def _get_javascript(self) -> str:
        """Get JavaScript for interactivity."""
        return """
        // Auto-scroll to see latest trades
        document.addEventListener('DOMContentLoaded', function() {
            // Highlight positive/negative values
            document.querySelectorAll('td').forEach(function(td) {
                const text = td.textContent;
                if (text.includes('$')) {
                    const value = parseFloat(text.replace(/[$,]/g, ''));
                    if (value > 0 && !td.classList.contains('pnl-positive')) {
                        if (text.includes('+') || td.cellIndex > 5) {
                            td.classList.add('pnl-positive');
                        }
                    } else if (value < 0) {
                        td.classList.add('pnl-negative');
                    }
                }
            });
        });
        """

    def _generate_summary_cards(self, stats: Dict) -> str:
        """Generate summary cards HTML."""
        # Get values - prefer TWS values if available
        unrealized = self.monitor.tws_unrealized_pnl if self.monitor.tws_unrealized_pnl is not None else self.monitor.get_unrealized_pnl()
        realized = self.monitor.tws_realized_pnl if self.monitor.tws_realized_pnl is not None else self.monitor.get_realized_pnl()
        daily_pnl = self.monitor.get_daily_pnl()  # Realized + Unrealized from TWS

        # Gesamt Kapital = Initial Capital + Unrealized + Realized
        gesamt_kapital = self.monitor.initial_capital + unrealized + realized

        # Other stats
        open_trades = len(self.monitor.open_trades)

        # CSS classes
        daily_pnl_class = 'positive' if daily_pnl >= 0 else 'negative'
        unrealized_class = 'positive' if unrealized >= 0 else 'negative'
        realized_class = 'positive' if realized >= 0 else 'negative'

        return f"""
        <div class="summary-cards">
            <div class="summary-card">
                <div class="label">Gesamt Kapital</div>
                <div class="value neutral">${gesamt_kapital:,.2f}</div>
            </div>
            <div class="summary-card">
                <div class="label">Daily PnL</div>
                <div class="value {daily_pnl_class}">${daily_pnl:+,.2f}</div>
            </div>
            <div class="summary-card">
                <div class="label">Unrealized PnL</div>
                <div class="value {unrealized_class}">${unrealized:+,.2f}</div>
            </div>
            <div class="summary-card">
                <div class="label">Realized PnL</div>
                <div class="value {realized_class}">${realized:+,.2f}</div>
            </div>
            <div class="summary-card">
                <div class="label">Cash Balance</div>
                <div class="value neutral">${self.monitor.current_capital:,.2f}</div>
            </div>
            <div class="summary-card">
                <div class="label">Open Positions</div>
                <div class="value neutral">{open_trades}</div>
            </div>
        </div>
        """

    def _generate_statistics_panel(self, stats: Dict) -> str:
        """Generate statistics panel HTML."""
        stat_rows = ""
        display_stats = [
            ('Initial Capital', stats.get('Initial Capital', '-')),
            ('Current Capital', stats.get('Current Capital', '-')),
            ('Profit Factor', stats.get('Profit Factor', '-')),
            ('Sharpe Ratio', stats.get('Sharpe Ratio', '-')),
            ('Sortino Ratio', stats.get('Sortino Ratio', '-')),
            ('Max Drawdown', stats.get('Max Drawdown', '-')),
            ('Avg Trade', stats.get('Avg Trade', '-')),
            ('Avg Duration', stats.get('Avg Duration', '-')),
            ('Expectancy', stats.get('Expectancy', '-')),
            ('Max Consec. Wins', stats.get('Max Consecutive Wins', '-')),
            ('Max Consec. Losses', stats.get('Max Consecutive Losses', '-')),
        ]

        for label, value in display_stats:
            stat_rows += f"""
            <div class="stat-row">
                <span class="stat-label">{label}</span>
                <span class="stat-value">{value}</span>
            </div>
            """

        return f"""
        <div class="card">
            <h2>Performance Metrics</h2>
            {stat_rows}
        </div>
        """

    def _generate_open_positions_table(self) -> str:
        """Generate open positions table HTML."""
        if not self.monitor.open_trades:
            return """
            <div class="card">
                <h2>Open Positions</h2>
                <p style="color: #888; text-align: center; padding: 20px;">No open positions</p>
            </div>
            """

        rows = ""
        for trade in self.monitor.open_trades:
            direction_badge = 'badge-long' if trade.direction == TradeDirection.LONG else 'badge-short'
            pnl_class = 'pnl-positive' if trade.unrealized_pnl >= 0 else 'pnl-negative'
            daily_class = 'pnl-positive' if trade.daily_pnl >= 0 else 'pnl-negative'

            # Get current price for this specific symbol
            current_price = trade.current_price or self.monitor.current_prices.get(trade.symbol, 0)
            pnl_pct = trade.calculate_unrealized_pnl_pct(current_price) if current_price > 0 else 0
            current_price_str = f"${current_price:,.2f}" if current_price > 0 else "-"
            import math
            daily_pnl_str = f"${trade.daily_pnl:+,.0f}" if trade.daily_pnl and not math.isnan(trade.daily_pnl) else "-"

            rows += f"""
            <tr>
                <td>{trade.symbol}</td>
                <td><span class="badge {direction_badge}">{trade.direction.value}</span></td>
                <td>{trade.entry_quantity}</td>
                <td>${trade.entry_price:,.2f}</td>
                <td>{current_price_str}</td>
                <td class="{daily_class}">{daily_pnl_str}</td>
                <td class="{pnl_class}">${trade.unrealized_pnl:,.2f}</td>
                <td class="{pnl_class}">{pnl_pct:+.2f}%</td>
            </tr>
            """

        return f"""
        <div class="card">
            <h2>Open Positions ({len(self.monitor.open_trades)})</h2>
            <table>
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Direction</th>
                        <th>Qty</th>
                        <th>Avg Cost</th>
                        <th>Market Price</th>
                        <th>Daily PnL</th>
                        <th>Unrealized PnL</th>
                        <th>PnL %</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
        </div>
        """

    def _generate_closed_trades_table(self, limit: int = 20) -> str:
        """Generate closed trades table HTML."""
        trades = self.monitor.closed_trades[-limit:][::-1]  # Most recent first

        if not trades:
            return """
            <div class="card">
                <h2>Closed Trades</h2>
                <p style="color: #888; text-align: center; padding: 20px;">No closed trades</p>
            </div>
            """

        rows = ""
        for trade in trades:
            direction_badge = 'badge-long' if trade.direction == TradeDirection.LONG else 'badge-short'
            pnl_class = 'pnl-positive' if trade.realized_pnl >= 0 else 'pnl-negative'
            pnl_pct = (trade.realized_pnl / (trade.entry_price * trade.entry_quantity)) * 100 if trade.entry_price > 0 else 0
            duration = trade.duration()
            duration_str = f"{duration.days}d {duration.seconds//3600}h" if duration else "-"
            exit_price_str = f"${trade.exit_price:,.2f}" if trade.exit_price else "-"
            exit_date_str = trade.exit_date.strftime('%Y-%m-%d %H:%M') if trade.exit_date else '-'

            rows += f"""
            <tr>
                <td>{trade.trade_id}</td>
                <td>{trade.symbol}</td>
                <td><span class="badge {direction_badge}">{trade.direction.value}</span></td>
                <td>{trade.entry_date.strftime('%Y-%m-%d %H:%M')}</td>
                <td>${trade.entry_price:,.2f}</td>
                <td>{exit_date_str}</td>
                <td>{exit_price_str}</td>
                <td class="{pnl_class}">${trade.realized_pnl:,.2f}</td>
                <td class="{pnl_class}">{pnl_pct:+.2f}%</td>
                <td>{trade.exit_reason or '-'}</td>
                <td>{duration_str}</td>
            </tr>
            """

        return f"""
        <div class="card">
            <h2>Closed Trades (Last {len(trades)})</h2>
            <table>
                <thead>
                    <tr>
                        <th>Trade ID</th>
                        <th>Symbol</th>
                        <th>Direction</th>
                        <th>Entry Date</th>
                        <th>Entry Price</th>
                        <th>Exit Date</th>
                        <th>Exit Price</th>
                        <th>Realized PnL</th>
                        <th>PnL %</th>
                        <th>Exit Reason</th>
                        <th>Duration</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
        </div>
        """

    def _generate_equity_chart(self) -> str:
        """Generate unrealized PnL chart HTML using Plotly."""
        # Get current unrealized PnL from TWS or calculate
        unrealized_pnl = self.monitor.tws_unrealized_pnl if self.monitor.tws_unrealized_pnl is not None else self.monitor.get_unrealized_pnl()

        if not PLOTLY_AVAILABLE:
            return f"""
            <div class="card">
                <h2>Unrealized PnL</h2>
                <p style="color: {'#00e676' if unrealized_pnl >= 0 else '#ff5252'}; text-align: center; padding: 20px; font-size: 2em;">
                    ${unrealized_pnl:+,.2f}
                </p>
            </div>
            """

        # Calculate unrealized PnL per position for bar chart
        positions = []
        pnls = []
        for trade in self.monitor.open_trades:
            positions.append(trade.symbol)
            pnl = trade.unrealized_pnl if trade.unrealized_pnl else 0
            pnls.append(pnl)

        if not positions:
            return f"""
            <div class="card">
                <h2>Unrealized PnL</h2>
                <p style="color: #888; text-align: center; padding: 20px;">No open positions</p>
            </div>
            """

        # Sort by PnL
        sorted_data = sorted(zip(positions, pnls), key=lambda x: x[1], reverse=True)
        positions, pnls = zip(*sorted_data) if sorted_data else ([], [])

        colors = ['#00e676' if p >= 0 else '#ff5252' for p in pnls]

        fig = go.Figure()

        # Bar chart of unrealized PnL per position
        fig.add_trace(go.Bar(
            x=list(positions),
            y=list(pnls),
            marker_color=colors,
            text=[f'${p:+,.0f}' for p in pnls],
            textposition='outside'
        ))

        # Zero line
        fig.add_hline(y=0, line_dash="solid", line_color="#666", line_width=1)

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e0e0e0'),
            margin=dict(l=50, r=20, t=30, b=80),
            height=350,
            xaxis=dict(
                showgrid=False,
                title='',
                tickangle=-45
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)',
                title='Unrealized PnL ($)'
            ),
            showlegend=False
        )

        chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

        pnl_color = '#00e676' if unrealized_pnl >= 0 else '#ff5252'
        return f"""
        <div class="card">
            <h2>Unrealized PnL <span style="color: {pnl_color}; float: right;">${unrealized_pnl:+,.2f}</span></h2>
            <div class="chart-container">
                {chart_html}
            </div>
        </div>
        """

    def _sync_trades_to_equity_calc(self):
        """Sync trades from monitor to equity calculator."""
        if not EQUITY_CALC_AVAILABLE:
            return None

        if self._equity_calc is None:
            # Pass existing IB connection to avoid multiple connections
            self._equity_calc = EquityCurveCalculator(
                ib_port=self.ib_port,
                ib_connection=self.ib_connection
            )

        # Sync open trades from monitor
        for trade in self.monitor.open_trades:
            exists = any(t.symbol == trade.symbol and not t.is_closed
                        for t in self._equity_calc.trades)
            if not exists:
                self._equity_calc.add_trade(
                    symbol=trade.symbol,
                    direction=trade.direction.value,
                    entry_date=trade.entry_date,
                    entry_price=trade.entry_price,
                    quantity=trade.entry_quantity,
                    entry_commission=trade.commission
                )

        return self._equity_calc

    def _generate_capital_curve(self, hours: int = 8, days: int = 7) -> str:
        """Generate capital curve (Realized + Unrealized PnL over time).

        Syncs trades from TWS and calculates curve from IB historical data.
        """
        if not PLOTLY_AVAILABLE:
            return """
            <div class="card">
                <h2>Kapitalkurve (R+U PnL)</h2>
                <p style="color: #888; text-align: center; padding: 20px;">Plotly not available</p>
            </div>
            """

        data = []

        # Sync trades from monitor and calculate equity curve
        if EQUITY_CALC_AVAILABLE:
            try:
                calc = self._sync_trades_to_equity_calc()
                if calc and calc.trades:
                    data = calc.get_equity_curve_data(days=days)
            except Exception as e:
                print(f"Equity calc error: {e}")

        # Fall back to live PnL history
        if not data and PNL_HISTORY_AVAILABLE:
            try:
                history = PnLHistory()
                data = history.get_total_pnl_curve(hours=hours)
            except Exception:
                pass

        if len(data) < 2:
            return """
            <div class="card">
                <h2>Kapitalkurve (R+U PnL)</h2>
                <p style="color: #888; text-align: center; padding: 20px;">
                    Keine Trades im Monitor. Verbinde mit TWS.
                </p>
            </div>
            """

        try:
            timestamps = [d[0] for d in data]
            values = [d[1] for d in data]

            current_pnl = values[-1] if values else 0
            line_color = '#00d9ff'

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=timestamps,
                y=values,
                mode='lines',
                name='Total PnL',
                line=dict(color=line_color, width=2),
                fill='tozeroy',
                fillcolor='rgba(0, 217, 255, 0.1)'
            ))

            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e0e0e0'),
                margin=dict(l=50, r=20, t=30, b=50),
                height=300,
                xaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.1)',
                    title=''
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.1)',
                    title='Total PnL ($)'
                ),
                showlegend=False
            )

            chart_html = fig.to_html(full_html=False, include_plotlyjs=False)

            pnl_color = '#00e676' if current_pnl >= 0 else '#ff5252'
            return f"""
            <div class="card">
                <h2>Kapitalkurve (R+U) <span style="color: {pnl_color}; float: right;">${current_pnl:+,.2f}</span></h2>
                <div class="chart-container">
                    {chart_html}
                </div>
            </div>
            """
        except Exception as e:
            return f"""
            <div class="card">
                <h2>Kapitalkurve (R+U PnL)</h2>
                <p style="color: #888; text-align: center; padding: 20px;">Error: {e}</p>
            </div>
            """

    def _generate_daily_pnl_curve(self, hours: int = 8) -> str:
        """Generate Daily PnL curve over time."""
        if not PLOTLY_AVAILABLE or not PNL_HISTORY_AVAILABLE:
            return """
            <div class="card">
                <h2>Daily PnL</h2>
                <p style="color: #888; text-align: center; padding: 20px;">No history data available</p>
            </div>
            """

        try:
            history = PnLHistory()
            data = history.get_daily_pnl_curve(hours=hours)

            if len(data) < 2:
                return """
                <div class="card">
                    <h2>Daily PnL</h2>
                    <p style="color: #888; text-align: center; padding: 20px;">
                        Recording data...
                    </p>
                </div>
                """

            timestamps = [d[0] for d in data]
            values = [d[1] for d in data]

            current_pnl = values[-1] if values else 0
            line_color = '#00e676' if current_pnl >= 0 else '#ff5252'

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=timestamps,
                y=values,
                mode='lines',
                name='Daily PnL',
                line=dict(color=line_color, width=2),
                fill='tozeroy',
                fillcolor='rgba(0, 230, 118, 0.1)' if current_pnl >= 0 else 'rgba(255, 82, 82, 0.1)'
            ))

            fig.add_hline(y=0, line_dash="dash", line_color="#666", line_width=1)

            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e0e0e0'),
                margin=dict(l=50, r=20, t=30, b=50),
                height=300,
                xaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.1)',
                    title=''
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.1)',
                    title='Daily PnL ($)'
                ),
                showlegend=False
            )

            chart_html = fig.to_html(full_html=False, include_plotlyjs=False)

            pnl_color = '#00e676' if current_pnl >= 0 else '#ff5252'
            return f"""
            <div class="card">
                <h2>Daily PnL <span style="color: {pnl_color}; float: right;">${current_pnl:+,.2f}</span></h2>
                <div class="chart-container">
                    {chart_html}
                </div>
            </div>
            """
        except Exception as e:
            return f"""
            <div class="card">
                <h2>Daily PnL</h2>
                <p style="color: #888; text-align: center; padding: 20px;">Error: {e}</p>
            </div>
            """

    def _generate_pnl_chart(self) -> str:
        """Generate PnL distribution chart."""
        if not PLOTLY_AVAILABLE or not self.monitor.closed_trades:
            return """
            <div class="card">
                <h2>Trade PnL Distribution</h2>
                <p style="color: #888; text-align: center; padding: 20px;">No trade data available</p>
            </div>
            """

        pnls = [t.realized_pnl for t in self.monitor.closed_trades]
        colors = ['#00e676' if p >= 0 else '#ff5252' for p in pnls]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(range(1, len(pnls) + 1)),
            y=pnls,
            marker_color=colors,
            name='PnL'
        ))

        fig.add_hline(y=0, line_color='#666')

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e0e0e0'),
            margin=dict(l=50, r=20, t=30, b=50),
            height=300,
            xaxis=dict(
                showgrid=False,
                title='Trade #'
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)',
                title='PnL ($)'
            ),
            showlegend=False
        )

        chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

        return f"""
        <div class="card">
            <h2>Trade PnL Distribution</h2>
            <div class="chart-container" style="height: 300px;">
                {chart_html}
            </div>
        </div>
        """

    def _generate_win_loss_chart(self, stats: Dict) -> str:
        """Generate win/loss pie chart."""
        if not PLOTLY_AVAILABLE:
            return ""

        wins = stats.get('Winning Trades', 0)
        losses = stats.get('Losing Trades', 0)

        if wins == 0 and losses == 0:
            return """
            <div class="card">
                <h2>Win/Loss Ratio</h2>
                <p style="color: #888; text-align: center; padding: 20px;">No trade data available</p>
            </div>
            """

        fig = go.Figure()
        fig.add_trace(go.Pie(
            labels=['Wins', 'Losses'],
            values=[wins, losses],
            hole=0.6,
            marker=dict(colors=['#00e676', '#ff5252']),
            textinfo='label+percent',
            textfont=dict(color='#e0e0e0')
        ))

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e0e0e0'),
            margin=dict(l=20, r=20, t=30, b=30),
            height=300,
            showlegend=False,
            annotations=[dict(
                text=f'{wins}/{wins+losses}',
                x=0.5, y=0.5,
                font_size=20,
                showarrow=False,
                font=dict(color='#00d9ff')
            )]
        )

        chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

        return f"""
        <div class="card">
            <h2>Win/Loss Ratio</h2>
            <div class="chart-container" style="height: 300px;">
                {chart_html}
            </div>
        </div>
        """

    def _parse_currency(self, value: str) -> float:
        """Parse currency string to float."""
        try:
            return float(value.replace('$', '').replace(',', '').strip())
        except:
            return 0.0

    def save(self, filepath: str = "dashboard.html", auto_refresh: int = 0):
        """
        Save dashboard to HTML file.

        Args:
            filepath: Output file path
            auto_refresh: Auto-refresh interval in seconds
        """
        html = self.generate_html(auto_refresh=auto_refresh)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)

        print(f"Dashboard saved to: {filepath}")
        return filepath

    def open_in_browser(self, filepath: str = "dashboard.html", auto_refresh: int = 0):
        """Save and open dashboard in browser."""
        self.save(filepath, auto_refresh)
        webbrowser.open(f'file://{os.path.abspath(filepath)}')


def create_demo_monitor() -> TradeMonitor:
    """Create a demo monitor with sample trades for multiple symbols."""
    monitor = TradeMonitor(initial_capital=50000.0, symbol="Portfolio")

    # Demo closed trades with different symbols
    closed_trades_data = [
        ("AAPL", 150.0, 168.0, "signal", TradeDirection.LONG, 5, 30),
        ("MSFT", 320.0, 305.0, "stop_loss", TradeDirection.LONG, 3, 20),
        ("GOOGL", 140.0, 158.0, "take_profit", TradeDirection.LONG, 8, 25),
        ("TSLA", 250.0, 235.0, "signal", TradeDirection.SHORT, 4, 15),
        ("NVDA", 480.0, 520.0, "trailing_stop", TradeDirection.LONG, 6, 10),
        ("AMZN", 180.0, 175.0, "signal", TradeDirection.LONG, 2, 20),
        ("META", 500.0, 545.0, "signal", TradeDirection.LONG, 5, 8),
        ("AMD", 165.0, 158.0, "stop_loss", TradeDirection.LONG, 3, 25),
    ]

    base_date = datetime.now() - timedelta(days=60)

    for i, (symbol, entry, exit_p, reason, direction, days, qty) in enumerate(closed_trades_data):
        entry_date = base_date + timedelta(days=i*7)
        exit_date = entry_date + timedelta(days=days)

        trade = monitor.open_trade(
            direction=direction,
            entry_price=entry,
            entry_date=entry_date,
            quantity=qty,
            symbol=symbol
        )

        # Update price to simulate movement
        for d in range(days):
            price = entry + (exit_p - entry) * (d / days)
            monitor.update_price(price, symbol=symbol, timestamp=entry_date + timedelta(days=d))

        monitor.close_trade(trade, exit_p, exit_date, reason)

    # Add multiple open positions with different symbols and prices
    open_positions = [
        ("AAPL", 175.0, 182.50, 40, 168.0, 0.05),
        ("MSFT", 415.0, 428.75, 25, 400.0, 0.04),
        ("NVDA", 875.0, 892.30, 12, 840.0, 0.06),
        ("GOOGL", 175.0, 171.25, 30, 165.0, None),  # Losing position
    ]

    for symbol, entry, current, qty, stop, trailing in open_positions:
        trade = monitor.open_trade(
            direction=TradeDirection.LONG,
            entry_price=entry,
            quantity=qty,
            stop_loss=stop,
            trailing_stop_pct=trailing,
            symbol=symbol
        )
        # Update each symbol with its own current price
        monitor.update_price(current, symbol=symbol)

    return monitor


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Trade Monitor Dashboard')
    parser.add_argument('--tws', action='store_true', help='Connect to TWS')
    parser.add_argument('--host', default='127.0.0.1', help='TWS host')
    parser.add_argument('--port', type=int, default=7497, help='TWS port (7497=TWS, 4001=Gateway)')
    parser.add_argument('--refresh', type=int, default=30, help='Auto-refresh interval (0=disabled)')
    parser.add_argument('--output', default='dashboard.html', help='Output file')
    args = parser.parse_args()

    if args.tws:
        # Try to connect to TWS
        try:
            from tws_connector import TWSConnector, print_tws_status
            from pnl_history import PnLHistory, record_pnl
            import time

            print(f"Connecting to TWS at {args.host}:{args.port}...")
            connector = TWSConnector(host=args.host, port=args.port)

            if connector.connect():
                print_tws_status(connector)

                # Create monitor from TWS
                monitor = TradeMonitor(initial_capital=10000.0, symbol="TWS Portfolio")

                # Create PnL history tracker
                pnl_history = PnLHistory()

                # Initial sync
                connector.sync_to_monitor(monitor)

                # Record initial PnL
                record_pnl(monitor)

                # Create dashboard
                dashboard = TradeDashboard(monitor, title="TWS Trade Monitor")
                dashboard.open_in_browser(args.output, auto_refresh=args.refresh)

                if args.refresh > 0:
                    print(f"\nLive mode: updating every {args.refresh} seconds")
                    print("Recording PnL history for capital curve...")
                    print("Press Ctrl+C to stop...")
                    try:
                        while True:
                            time.sleep(args.refresh)
                            # Clear and resync
                            monitor.open_trades.clear()
                            monitor.all_trades.clear()
                            connector.sync_to_monitor(monitor)

                            # Record PnL to history
                            record_pnl(monitor)

                            dashboard.save(args.output, auto_refresh=args.refresh)
                            print(f"  Updated: {len(monitor.open_trades)} positions, Daily PnL: ${monitor.tws_daily_pnl or 0:+,.2f}")
                    except KeyboardInterrupt:
                        print("\nStopping...")
                else:
                    input("Press Enter to disconnect...")

                connector.disconnect()
            else:
                print("Could not connect to TWS. Using demo data...")
                monitor = create_demo_monitor()
                dashboard = TradeDashboard(monitor, title="Trade Monitor (Demo)")
                dashboard.open_in_browser(args.output, auto_refresh=0)

        except ImportError:
            print("ib_insync not installed. Using demo data...")
            print("Install with: pip install ib_insync")
            monitor = create_demo_monitor()
            dashboard = TradeDashboard(monitor, title="Trade Monitor (Demo)")
            dashboard.open_in_browser(args.output, auto_refresh=0)
    else:
        # Demo mode
        print("Creating demo dashboard...")
        monitor = create_demo_monitor()
        dashboard = TradeDashboard(monitor, title="Trade Monitor Dashboard")
        dashboard.open_in_browser(args.output, auto_refresh=0)
        print("\nRun with --tws flag to connect to TWS")


if __name__ == "__main__":
    main()
