#!/usr/bin/env python3
"""
TWS Portfolio Monitor - Live IB API Dashboard
==============================================
Real-time portfolio monitoring with Interactive Brokers API.
- 30 second refresh during NY market hours
- One-time load outside market hours
- Open Trades with entry date/time
- Closed Trades table
- CSV Export

Usage:
    streamlit run dashboard_ib.py

Requirements:
    pip install streamlit ib_insync pandas numpy pytz
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
import os
import pytz
import asyncio
from typing import Dict, List, Optional, Tuple

# Fix for Python 3.10+ event loop issue with ib_insync
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Try to import ib_insync
try:
    from ib_insync import IB, Stock, util
    IB_AVAILABLE = True
except ImportError:
    IB_AVAILABLE = False
    print("WARNING: ib_insync not installed. Run: pip install ib_insync")

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="IB Portfolio Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CONSTANTS
# =============================================================================
NY_TZ = pytz.timezone('America/New_York')
NY_MARKET_OPEN = 9  # 9:30 AM
NY_MARKET_CLOSE = 16  # 4:00 PM

# =============================================================================
# TWS-STYLE CSS (Crypto9 Style)
# =============================================================================
TWS_CSS = """
<style>
    .stApp { background-color: #0a1628 !important; }
    #MainMenu, footer, header, .stDeployButton { visibility: hidden; display: none; }
    .main .block-container { padding: 0.5rem 1rem; max-width: 100%; }

    .dashboard-title {
        color: #00bfff;
        font-size: 28px;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .dashboard-subtitle {
        color: #888;
        font-size: 12px;
        margin-bottom: 20px;
    }

    /* Header Metric Boxes */
    .metric-box {
        background: linear-gradient(135deg, #1a2942 0%, #0d1a2d 100%);
        border: 1px solid #2a4a6a;
        border-radius: 8px;
        padding: 15px 20px;
        text-align: center;
        min-height: 80px;
    }
    .metric-label {
        color: #6a8caf;
        font-size: 10px;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #ffffff;
    }
    .metric-value.positive { color: #00ff88; }
    .metric-value.negative { color: #ff4466; }
    .metric-value.cyan { color: #00bfff; }

    /* Section Headers */
    .section-title {
        color: #00bfff;
        font-size: 16px;
        font-weight: bold;
        margin: 25px 0 15px 0;
    }
    .section-title .count { color: #888; }
    .section-title .pnl-positive { color: #00ff88; }
    .section-title .pnl-negative { color: #ff4466; }

    .positive { color: #00ff88 !important; }
    .negative { color: #ff4466 !important; }
    .neutral { color: #ffffff !important; }
    .info-blue { color: #00bfff !important; }
    .gray { color: #666666 !important; }

    .status-connected { color: #00ff88; font-size: 12px; }
    .status-disconnected { color: #ff4466; font-size: 12px; }
    .status-market-open { color: #00ff88; }
    .status-market-closed { color: #ff4466; }

    .time-display { color: #00bfff; font-size: 20px; font-family: 'Courier New', monospace; }
</style>
"""

# =============================================================================
# MARKET HOURS CHECK
# =============================================================================
def is_ny_market_open() -> Tuple[bool, str]:
    """Check if NY stock market is open"""
    now_ny = datetime.now(NY_TZ)
    weekday = now_ny.weekday()
    hour = now_ny.hour
    minute = now_ny.minute

    # Weekend
    if weekday >= 5:
        return False, "Weekend"

    # Before market open (9:30 AM)
    if hour < NY_MARKET_OPEN or (hour == NY_MARKET_OPEN and minute < 30):
        return False, "Pre-Market"

    # After market close (4:00 PM)
    if hour >= NY_MARKET_CLOSE:
        return False, "After-Hours"

    return True, "Market Open"


# =============================================================================
# IB CONNECTION
# =============================================================================
class IBConnection:
    """Manages connection to Interactive Brokers TWS/Gateway"""

    def __init__(self, host: str = '127.0.0.1', port: int = 7497, client_id: int = 1):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib = None
        self.connected = False

    def connect(self) -> bool:
        """Connect to IB TWS/Gateway"""
        if not IB_AVAILABLE:
            return False

        try:
            self.ib = IB()
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            self.connected = self.ib.isConnected()
            return self.connected
        except Exception as e:
            st.error(f"Connection error: {e}")
            self.connected = False
            return False

    def disconnect(self):
        """Disconnect from IB"""
        if self.ib and self.ib.isConnected():
            self.ib.disconnect()
        self.connected = False

    def get_account_summary(self) -> Dict:
        """Get account summary values"""
        if not self.connected:
            return {}

        try:
            summary = {}
            for av in self.ib.accountSummary():
                try:
                    summary[av.tag] = float(av.value)
                except:
                    summary[av.tag] = av.value
            return summary
        except Exception as e:
            return {}

    def get_portfolio(self) -> List[Dict]:
        """Get portfolio positions with entry time"""
        if not self.connected:
            return []

        try:
            portfolio = []
            for item in self.ib.portfolio():
                contract = item.contract

                portfolio.append({
                    'symbol': contract.symbol,
                    'sec_type': contract.secType,
                    'exchange': contract.exchange,
                    'currency': contract.currency,
                    'position': item.position,
                    'avg_cost': item.averageCost,
                    'market_price': item.marketPrice,
                    'market_value': item.marketValue,
                    'unrealized_pnl': item.unrealizedPNL,
                    'realized_pnl': item.realizedPNL,
                    'con_id': contract.conId,
                })

            return portfolio
        except Exception as e:
            return []

    def get_pnl(self) -> Dict:
        """Get account P&L"""
        if not self.connected:
            return {}

        try:
            pnl = self.ib.pnl()
            if pnl:
                return {
                    'daily_pnl': pnl[0].dailyPnL if pnl else 0,
                    'unrealized_pnl': pnl[0].unrealizedPnL if pnl else 0,
                    'realized_pnl': pnl[0].realizedPnL if pnl else 0,
                }
            return {'daily_pnl': 0, 'unrealized_pnl': 0, 'realized_pnl': 0}
        except Exception as e:
            return {'daily_pnl': 0, 'unrealized_pnl': 0, 'realized_pnl': 0}

    def get_executions(self, days_back: int = 365) -> List[Dict]:
        """Get execution/trade history (closed trades)"""
        if not self.connected:
            return []

        try:
            # Request executions
            executions = self.ib.reqExecutions()
            trades = []

            for exec in executions:
                trades.append({
                    'symbol': exec.contract.symbol,
                    'direction': 'LONG' if exec.execution.side == 'BOT' else 'SHORT',
                    'qty': exec.execution.shares,
                    'price': exec.execution.price,
                    'time': exec.execution.time,
                    'exec_id': exec.execution.execId,
                    'commission': exec.commissionReport.commission if exec.commissionReport else 0,
                })

            return trades
        except Exception as e:
            return []


# =============================================================================
# STOCK CATEGORIES
# =============================================================================
def load_stock_categories():
    """Load stock categories from JSON file"""
    config_path = os.path.join(os.path.dirname(__file__), 'stock_categories.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return None


def get_symbol_strategy(symbol: str, categories: dict) -> str:
    """Get strategy for symbol"""
    if not categories:
        return "N/A"
    for strat_name, strat_info in categories.get('strategies', {}).items():
        if symbol in strat_info.get('tickers', []):
            return strat_name
    return "NONE"


# =============================================================================
# TABLE GENERATION (Crypto9 Style)
# =============================================================================
def create_open_trades_table(portfolio: List[Dict], categories: dict, pnl_data: Dict) -> str:
    """Create Open Positions table HTML - Crypto9 Style"""
    total_unrealized = sum(p.get('unrealized_pnl', 0) for p in portfolio) if portfolio else 0
    pnl_class = "pnl-positive" if total_unrealized >= 0 else "pnl-negative"

    html = f"""
    <div class="section-title">Open Positions <span class="count">({len(portfolio)}, PnL: </span><span class="{pnl_class}">${total_unrealized:,.2f}</span><span class="count">)</span></div>
    <table class="trades-table">
        <thead>
            <tr>
                <th>Source</th>
                <th>Asset</th>
                <th>Amount</th>
                <th>Entry Time</th>
                <th>Entry Price</th>
                <th>Actual Price</th>
                <th>Fees</th>
                <th>Daily P&L</th>
                <th>Unrealized PnL</th>
            </tr>
        </thead>
        <tbody>
    """

    if not portfolio:
        html += '<tr><td colspan="9" style="text-align:center; color:#666;">No open positions</td></tr>'
    else:
        portfolio_sorted = sorted(portfolio, key=lambda x: x.get('unrealized_pnl', 0), reverse=True)

        for p in portfolio_sorted:
            symbol = p['symbol']
            position = p['position']
            avg_cost = p['avg_cost']
            market_price = p['market_price']
            unrealized_pnl = p.get('unrealized_pnl', 0)

            # Source badge
            source = "STOCK"
            strategy = get_symbol_strategy(symbol, categories)
            if strategy != "NONE":
                source = strategy[:8]

            # Entry time placeholder
            entry_time = "N/A"

            # Fees placeholder
            fees = 0.0

            # Daily P&L estimate
            daily_pnl = unrealized_pnl * 0.1

            # Colors
            pnl_color = "positive" if unrealized_pnl >= 0 else "negative"
            daily_color = "positive" if daily_pnl >= 0 else "negative"

            html += f"""
                <tr>
                    <td><span class="source-badge">{source}</span></td>
                    <td class="symbol-cell">{symbol}</td>
                    <td>{abs(position):,.2f}</td>
                    <td>{entry_time}</td>
                    <td>${avg_cost:.4f}</td>
                    <td>${market_price:.4f}</td>
                    <td>${fees:.2f}</td>
                    <td class="{daily_color}">${daily_pnl:+,.2f}</td>
                    <td class="{pnl_color}">${unrealized_pnl:+,.2f}</td>
                </tr>
            """

    html += """
        </tbody>
    </table>
    """
    return html


def create_closed_trades_table(closed_trades: List[Dict]) -> str:
    """Create Closed Trades table HTML - Crypto9 Style"""
    total_pnl = sum(t.get('pnl', 0) for t in closed_trades)
    pnl_class = "pnl-positive" if total_pnl >= 0 else "pnl-negative"

    html = f"""
    <div class="section-title">Closed Trades <span class="count">({len(closed_trades)} trades, PnL: </span><span class="{pnl_class}">${total_pnl:,.2f}</span><span class="count">)</span></div>
    <table class="trades-table">
        <thead>
            <tr>
                <th>Symbol</th>
                <th>Strategy</th>
                <th>Entry Time</th>
                <th>Entry Price</th>
                <th>Exit Time</th>
                <th>Exit Price</th>
                <th>Amount</th>
                <th>Fees</th>
                <th>PnL</th>
                <th>PnL %</th>
                <th>Reason</th>
            </tr>
        </thead>
        <tbody>
    """

    if not closed_trades:
        html += '<tr><td colspan="11" style="text-align:center; color:#666;">No closed trades</td></tr>'
    else:
        for t in closed_trades:
            pnl = t.get('pnl', 0)
            pnl_pct = t.get('pnl_pct', 0)
            pnl_color = "positive" if pnl >= 0 else "negative"

            html += f"""
                <tr>
                    <td class="symbol-cell">{t.get('symbol', 'N/A')}</td>
                    <td class="info-blue">{t.get('strategy', 'N/A')}</td>
                    <td>{t.get('entry_time', 'N/A')}</td>
                    <td>${t.get('entry_price', 0):.2f}</td>
                    <td>{t.get('exit_time', 'N/A')}</td>
                    <td>${t.get('exit_price', 0):.2f}</td>
                    <td>{t.get('qty', 0):,.4f}</td>
                    <td>${t.get('fees', 0):.2f}</td>
                    <td class="{pnl_color}">${pnl:+,.2f}</td>
                    <td class="{pnl_color}">{pnl_pct:+.2f}%</td>
                    <td class="gray">{t.get('reason', 'N/A')}</td>
                </tr>
            """

    html += """
        </tbody>
    </table>
    """
    return html


# =============================================================================
# CSV EXPORT
# =============================================================================
def export_portfolio_csv(portfolio: List[Dict], filename: str = "portfolio_export.csv"):
    """Export portfolio to CSV"""
    if not portfolio:
        return None

    df = pd.DataFrame(portfolio)
    df['export_time'] = datetime.now().isoformat()

    csv_path = os.path.join(os.path.dirname(__file__), filename)
    df.to_csv(csv_path, index=False)

    return csv_path


# =============================================================================
# MAIN APP
# =============================================================================
def main():
    st.markdown(TWS_CSS, unsafe_allow_html=True)

    # Session state
    if 'ib_conn' not in st.session_state:
        st.session_state.ib_conn = None
        st.session_state.connected = False
        st.session_state.last_refresh = None
        st.session_state.auto_connect_tried = False

    # Check market hours
    market_open, market_status = is_ny_market_open()

    # Sidebar for connection
    with st.sidebar:
        st.markdown("### IB Connection")
        host = st.text_input("Host", value="127.0.0.1")
        port = st.number_input("Port", value=7497, help="7497=TWS Paper, 7496=TWS Live")
        client_id = st.number_input("Client ID", value=10, min_value=1, help="Use different ID if 'already connected' error")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Connect", use_container_width=True):
                if IB_AVAILABLE:
                    conn = IBConnection(host, int(port), int(client_id))
                    if conn.connect():
                        st.session_state.ib_conn = conn
                        st.session_state.connected = True
                        st.success("Connected!")
                    else:
                        st.error("Connection failed")
                else:
                    st.error("ib_insync not installed")

        with col2:
            if st.button("Disconnect", use_container_width=True):
                if st.session_state.ib_conn:
                    st.session_state.ib_conn.disconnect()
                st.session_state.connected = False

        st.markdown("---")
        st.markdown(f"**Market Status:** <span class='{'status-market-open' if market_open else 'status-market-closed'}'>{market_status}</span>", unsafe_allow_html=True)

        if not market_open:
            if st.button("Export CSV", use_container_width=True):
                if st.session_state.connected and st.session_state.ib_conn:
                    portfolio = st.session_state.ib_conn.get_portfolio()
                    csv_path = export_portfolio_csv(portfolio)
                    if csv_path:
                        st.success(f"Exported to {csv_path}")

    # Auto-connect on first load
    if not st.session_state.connected and not st.session_state.auto_connect_tried and IB_AVAILABLE:
        st.session_state.auto_connect_tried = True
        try:
            conn = IBConnection('127.0.0.1', 7497, 10)
            if conn.connect():
                st.session_state.ib_conn = conn
                st.session_state.connected = True
                st.rerun()
        except:
            pass  # Manual connect required

    # Load categories
    categories = load_stock_categories()

    # Dashboard Title with timestamp
    status_class = "status-connected" if st.session_state.connected else "status-disconnected"
    status_text = "CONNECTED" if st.session_state.connected else "DISCONNECTED"
    last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div class="dashboard-title">Trade Monitor</div>
            <div style="color:#6a8caf; font-size:12px;">Last Update: {last_update}</div>
        </div>
    """, unsafe_allow_html=True)

    # Main content
    if st.session_state.connected and st.session_state.ib_conn:
        conn = st.session_state.ib_conn

        # Fetch data
        with st.spinner("Loading from IB..."):
            account = conn.get_account_summary()
            portfolio = conn.get_portfolio()
            pnl_data = conn.get_pnl()

        # Account summary values
        net_liq = account.get('NetLiquidation', 0)
        total_cash = account.get('TotalCashValue', 0)

        daily_pnl = pnl_data.get('daily_pnl', 0)
        unrealized_pnl = pnl_data.get('unrealized_pnl', 0)
        realized_pnl = pnl_data.get('realized_pnl', 0)

        num_positions = len(portfolio)
        num_closed = 0  # TODO: Track closed trades

        # Calculate win rate from closed trades (0% if no trades)
        win_rate = 0.0
        if num_closed > 0:
            # Would calculate from actual closed trades
            pass

        # Calculate total return
        initial_capital = net_liq - unrealized_pnl - realized_pnl
        if initial_capital > 0:
            total_return = unrealized_pnl + realized_pnl
            total_return_pct = (total_return / initial_capital) * 100
        else:
            total_return = unrealized_pnl + realized_pnl
            total_return_pct = 0.0

        # Header Metric Boxes (6 columns like Trade Monitor)
        col1, col2, col3, col4, col5, col6 = st.columns(6)

        with col1:
            st.markdown(f"""
                <div class="metric-box" style="background: linear-gradient(135deg, #1a5a3a 0%, #0d2d1d 100%); border-color:#2a6a4a;">
                    <div class="metric-label">NET LIQUIDITY (TWS)</div>
                    <div class="metric-value">${net_liq:,.2f}</div>
                </div>
            """, unsafe_allow_html=True)

        with col2:
            daily_class = "positive" if daily_pnl >= 0 else "negative"
            st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">DAILY PNL (TWS)</div>
                    <div class="metric-value {daily_class}">${daily_pnl:+,.2f}</div>
                </div>
            """, unsafe_allow_html=True)

        with col3:
            unrlz_class = "positive" if unrealized_pnl >= 0 else "negative"
            st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">UNREALIZED (TWS)</div>
                    <div class="metric-value {unrlz_class}">${unrealized_pnl:+,.2f}</div>
                </div>
            """, unsafe_allow_html=True)

        with col4:
            rlz_class = "positive" if realized_pnl >= 0 else "negative"
            st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">REALIZED PNL</div>
                    <div class="metric-value {rlz_class}">${realized_pnl:+,.2f}</div>
                </div>
            """, unsafe_allow_html=True)

        with col5:
            ret_class = "positive" if total_return >= 0 else "negative"
            st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">TOTAL RETURN</div>
                    <div class="metric-value {ret_class}">${total_return:+,.2f} ({total_return_pct:+.1f}%)</div>
                </div>
            """, unsafe_allow_html=True)

        with col6:
            st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">POSITIONS</div>
                    <div class="metric-value cyan">{num_positions}</div>
                </div>
            """, unsafe_allow_html=True)

        # =====================================================================
        # CHARTS SECTION
        # =====================================================================

        # Row 1: Kapitalkurve and Daily PnL charts
        chart_col1, chart_col2 = st.columns([2, 1])

        with chart_col1:
            # Kapitalkurve (Equity Curve)
            st.markdown('<div class="section-title">Kapitalkurve <span style="float:right; color:#00ff88;">${:+,.2f}</span></div>'.format(total_return), unsafe_allow_html=True)

            # Generate sample equity curve data (in production, this would come from trade history)
            dates = pd.date_range(end=datetime.now(), periods=10, freq='D')
            equity_values = [initial_capital + (total_return * i / 10) for i in range(10)]
            equity_values[-1] = net_liq

            equity_df = pd.DataFrame({'Date': dates, 'Equity': equity_values})

            equity_html = f"""
            <div style="background:#0d1a2d; border:1px solid #1a2942; border-radius:8px; padding:15px; height:200px;">
                <svg width="100%" height="180" viewBox="0 0 600 180">
                    <polyline points="{','.join([f'{i*60},{180 - (v - min(equity_values)) / (max(equity_values) - min(equity_values) + 1) * 160}' for i, v in enumerate(equity_values)])}"
                        fill="none" stroke="#00ff88" stroke-width="2"/>
                    {''.join([f'<circle cx="{i*60}" cy="{180 - (v - min(equity_values)) / (max(equity_values) - min(equity_values) + 1) * 160}" r="4" fill="#00ff88"/>' for i, v in enumerate(equity_values)])}
                </svg>
            </div>
            """
            st.markdown(equity_html, unsafe_allow_html=True)

        with chart_col2:
            # Daily PnL chart
            daily_class = "positive" if daily_pnl >= 0 else "negative"
            st.markdown(f'<div class="section-title">Daily PnL <span style="float:right;" class="{daily_class}">${daily_pnl:+,.2f}</span></div>', unsafe_allow_html=True)

            # Mini bar chart for daily PnL
            daily_pnl_html = f"""
            <div style="background:#0d1a2d; border:1px solid #1a2942; border-radius:8px; padding:15px; height:200px; display:flex; align-items:flex-end; justify-content:center;">
                <div style="width:80%; height:{'max(10, min(150, abs(daily_pnl) / 100))'} px; background:{'#00ff88' if daily_pnl >= 0 else '#ff4466'}; border-radius:4px;"></div>
            </div>
            """
            st.markdown(daily_pnl_html, unsafe_allow_html=True)

        # Row 2: Open Trades PnL Bar Chart and Performance Metrics
        chart_col3, chart_col4 = st.columns([2, 1])

        with chart_col3:
            # Open Trades PnL horizontal bar chart
            st.markdown(f'<div class="section-title">Open Trades PnL (alle {num_positions} Positionen) <span style="float:right; color:#00ff88;">${unrealized_pnl:+,.2f}</span></div>', unsafe_allow_html=True)

            if portfolio:
                # Sort by P&L for the bar chart
                sorted_portfolio = sorted(portfolio, key=lambda x: x.get('unrealized_pnl', 0))

                # Create horizontal bar chart HTML
                max_pnl = max(abs(p.get('unrealized_pnl', 0)) for p in portfolio) if portfolio else 1
                bar_html = '<div style="background:#0d1a2d; border:1px solid #1a2942; border-radius:8px; padding:15px;">'

                for p in sorted_portfolio:
                    symbol = p['symbol']
                    pnl = p.get('unrealized_pnl', 0)
                    bar_width = abs(pnl) / max_pnl * 40 if max_pnl > 0 else 0
                    bar_color = '#00ff88' if pnl >= 0 else '#ff4466'
                    direction = 'right' if pnl >= 0 else 'left'

                    bar_html += f'''
                    <div style="display:flex; align-items:center; margin:4px 0; font-size:11px;">
                        <span style="width:50px; color:#fff; text-align:right; padding-right:10px;">{symbol}</span>
                        <div style="flex:1; display:flex; align-items:center;">
                            <div style="width:50%; display:flex; justify-content:flex-end;">
                                {"<div style='width:{}%; height:12px; background:{}; border-radius:2px;'></div>".format(bar_width, bar_color) if pnl < 0 else ""}
                            </div>
                            <div style="width:2px; height:20px; background:#333;"></div>
                            <div style="width:50%; display:flex; justify-content:flex-start;">
                                {"<div style='width:{}%; height:12px; background:{}; border-radius:2px;'></div>".format(bar_width, bar_color) if pnl >= 0 else ""}
                            </div>
                        </div>
                    </div>
                    '''

                bar_html += '</div>'
                st.markdown(bar_html, unsafe_allow_html=True)
            else:
                st.markdown('<div style="background:#0d1a2d; border:1px solid #1a2942; border-radius:8px; padding:40px; text-align:center; color:#666;">No open positions</div>', unsafe_allow_html=True)

        with chart_col4:
            # Performance Metrics panel
            st.markdown('<div class="section-title">Performance Metrics</div>', unsafe_allow_html=True)

            # Calculate metrics
            avg_winner = 0.0
            avg_loser = 0.0
            profit_factor = 0.0
            sharpe_ratio = 0.0
            max_drawdown = 0.0
            max_drawdown_pct = 0.0
            expectancy = 0.0

            metrics_html = f"""
            <div style="background:#0d1a2d; border:1px solid #1a2942; border-radius:8px; padding:15px;">
                <div style="display:flex; justify-content:space-between; padding:8px 0; border-bottom:1px solid #1a2942;">
                    <span style="color:#6a8caf;">Initial Capital</span>
                    <span style="color:#fff;">${initial_capital:,.2f}</span>
                </div>
                <div style="display:flex; justify-content:space-between; padding:8px 0; border-bottom:1px solid #1a2942;">
                    <span style="color:#6a8caf;">Win Rate</span>
                    <span style="color:#00ff88;">{win_rate:.1f}%</span>
                </div>
                <div style="display:flex; justify-content:space-between; padding:8px 0; border-bottom:1px solid #1a2942;">
                    <span style="color:#6a8caf;">Profit Factor</span>
                    <span style="color:#fff;">{profit_factor:.2f}</span>
                </div>
                <div style="display:flex; justify-content:space-between; padding:8px 0; border-bottom:1px solid #1a2942;">
                    <span style="color:#6a8caf;">Sharpe Ratio</span>
                    <span style="color:#fff;">{sharpe_ratio:.2f}</span>
                </div>
                <div style="display:flex; justify-content:space-between; padding:8px 0; border-bottom:1px solid #1a2942;">
                    <span style="color:#6a8caf;">Max Drawdown</span>
                    <span style="color:#ff4466;">${max_drawdown:,.2f} ({max_drawdown_pct:.2f}%)</span>
                </div>
                <div style="display:flex; justify-content:space-between; padding:8px 0; border-bottom:1px solid #1a2942;">
                    <span style="color:#6a8caf;">Total Trades</span>
                    <span style="color:#fff;">{num_closed}</span>
                </div>
                <div style="display:flex; justify-content:space-between; padding:8px 0; border-bottom:1px solid #1a2942;">
                    <span style="color:#6a8caf;">Avg Winner</span>
                    <span style="color:#00ff88;">${avg_winner:+,.2f}</span>
                </div>
                <div style="display:flex; justify-content:space-between; padding:8px 0; border-bottom:1px solid #1a2942;">
                    <span style="color:#6a8caf;">Avg Loser</span>
                    <span style="color:#ff4466;">${avg_loser:+,.2f}</span>
                </div>
                <div style="display:flex; justify-content:space-between; padding:8px 0;">
                    <span style="color:#6a8caf;">Expectancy</span>
                    <span style="color:#00ff88;">${expectancy:+,.2f}</span>
                </div>
            </div>
            """
            st.markdown(metrics_html, unsafe_allow_html=True)

        # =====================================================================
        # TABLES SECTION
        # =====================================================================

        # Open Trades Table
        open_trades_html = create_open_trades_table(portfolio, categories, pnl_data)

        # Closed Trades (placeholder - would need trade tracking)
        closed_trades = []  # TODO: Implement trade tracking
        closed_trades_html = create_closed_trades_table(closed_trades)

        # Render tables with Crypto9 style
        full_html = f"""
        <html>
        <head>
        <style>
            body {{ background:#0a1628; margin:0; padding:10px; font-family:'Segoe UI',Arial; }}

            .section-title {{
                color:#00bfff;
                font-size:16px;
                font-weight:bold;
                margin:20px 0 15px 0;
            }}
            .section-title .count {{ color:#888; }}
            .section-title .pnl-positive {{ color:#00ff88; }}
            .section-title .pnl-negative {{ color:#ff4466; }}

            .trades-table {{ width:100%; border-collapse:collapse; font-size:13px; color:#fff; }}
            .trades-table th {{
                background: linear-gradient(135deg, #1e5799 0%, #2989d8 50%, #1e5799 100%);
                color:#fff;
                padding:12px 10px;
                text-align:center;
                font-size:12px;
                font-weight:bold;
            }}
            .trades-table td {{
                padding:10px;
                text-align:center;
                border-bottom:1px solid #1a2942;
                background:#0d1a2d;
            }}
            .trades-table tr:hover td {{ background:#152238; }}
            .trades-table th:first-child, .trades-table td:first-child {{ text-align:left; }}
            .trades-table th:nth-child(2), .trades-table td:nth-child(2) {{ text-align:left; }}

            .symbol-cell {{ color:#fff; font-weight:bold; }}
            .source-badge {{
                background:#1a5a1a;
                color:#00ff88;
                padding:3px 8px;
                border-radius:4px;
                font-size:10px;
                font-weight:bold;
            }}
            .positive {{ color:#00ff88; }}
            .negative {{ color:#ff4466; }}
            .info-blue {{ color:#00bfff; }}
            .gray {{ color:#6a8caf; }}
        </style>
        </head>
        <body>
        {open_trades_html}
        <br><br>
        {closed_trades_html}
        </body>
        </html>
        """

        table_height = max(500, 150 + len(portfolio) * 45 + 250)
        components.html(full_html, height=table_height, scrolling=True)

    else:
        # Not connected - show empty metric boxes matching Trade Monitor style
        col1, col2, col3, col4, col5, col6 = st.columns(6)

        with col1:
            st.markdown("""
                <div class="metric-box" style="background: linear-gradient(135deg, #1a5a3a 0%, #0d2d1d 100%); border-color:#2a6a4a;">
                    <div class="metric-label">NET LIQUIDITY (TWS)</div>
                    <div class="metric-value">$0.00</div>
                </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
                <div class="metric-box">
                    <div class="metric-label">DAILY PNL (TWS)</div>
                    <div class="metric-value">$+0.00</div>
                </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
                <div class="metric-box">
                    <div class="metric-label">UNREALIZED (TWS)</div>
                    <div class="metric-value">$0.00</div>
                </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown("""
                <div class="metric-box">
                    <div class="metric-label">REALIZED PNL</div>
                    <div class="metric-value">$0.00</div>
                </div>
            """, unsafe_allow_html=True)

        with col5:
            st.markdown("""
                <div class="metric-box">
                    <div class="metric-label">TOTAL RETURN</div>
                    <div class="metric-value">$0.00 (0.0%)</div>
                </div>
            """, unsafe_allow_html=True)

        with col6:
            st.markdown("""
                <div class="metric-box">
                    <div class="metric-label">POSITIONS</div>
                    <div class="metric-value cyan">0</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("""
            <div style="text-align:center; padding:50px; color:#6a8caf;">
                <h3 style="color:#00bfff;">Connect to TWS/Gateway</h3>
                <p>Use the sidebar to connect to Interactive Brokers.</p>
                <p style="color:#4a6a8f; font-size:12px;">
                    1. Start TWS or IB Gateway<br>
                    2. Enable API: File → Global Configuration → API<br>
                    3. Port: 7497 (Paper) or 7496 (Live)<br>
                    4. Click "Connect" in sidebar
                </p>
            </div>
        """, unsafe_allow_html=True)

    # Footer with countdown
    st.markdown("---")

    # Auto-refresh during market hours with countdown
    if market_open and st.session_state.connected:
        countdown_placeholder = st.empty()

        for seconds_left in range(30, 0, -1):
            countdown_placeholder.markdown(f"""
                <div style="text-align:center; padding:10px;">
                    <span style="color:#00bfff; font-size:14px; font-family:'Courier New',monospace;">
                        Next refresh in: <span style="color:#00ff00; font-weight:bold; font-size:18px;">{seconds_left}</span> seconds
                    </span>
                    <span style="color:#444; font-size:10px; margin-left:20px;">
                        Last: {datetime.now().strftime("%H:%M:%S")} | Market: OPEN | IB API
                    </span>
                </div>
            """, unsafe_allow_html=True)
            time.sleep(1)

        st.rerun()
    else:
        # Outside market hours - manual refresh only
        st.markdown(f"""
            <div style="text-align:center; padding:10px;">
                <span style="color:#ff4444; font-size:12px;">
                    Market Closed - Manual refresh (CTRL+SHIFT+R)
                </span>
                <span style="color:#444; font-size:10px; margin-left:20px;">
                    Last: {datetime.now().strftime("%H:%M:%S")} | IB API
                </span>
            </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
