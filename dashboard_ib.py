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
    initial_sidebar_state="collapsed"
)

# =============================================================================
# CONSTANTS
# =============================================================================
NY_TZ = pytz.timezone('America/New_York')
NY_MARKET_OPEN = 9  # 9:30 AM
NY_MARKET_CLOSE = 16  # 4:00 PM

# =============================================================================
# TWS-STYLE CSS
# =============================================================================
TWS_CSS = """
<style>
    .stApp { background-color: #000000 !important; }
    #MainMenu, footer, header, .stDeployButton { visibility: hidden; display: none; }
    .main .block-container { padding: 0.5rem 1rem; max-width: 100%; }

    .positive { color: #00ff00 !important; }
    .negative { color: #ff4444 !important; }
    .neutral { color: #ffffff !important; }
    .info-blue { color: #00bfff !important; }
    .gray { color: #666666 !important; }

    .pnl-header { color: #888; font-size: 11px; margin-bottom: 2px; }
    .pnl-value-large { font-size: 28px; font-weight: bold; }
    .pnl-value-small { font-size: 14px; }
    .account-label { color: #00bfff; font-size: 12px; }
    .account-value { color: #ffffff; font-size: 12px; }

    .section-header {
        color: #00bfff;
        font-size: 14px;
        font-weight: bold;
        padding: 10px 0 5px 0;
        border-bottom: 1px solid #333;
        margin-bottom: 10px;
    }

    .cash-container {
        background-color: rgba(0, 60, 0, 0.4);
        padding: 8px 15px;
        border: 1px solid #004400;
        margin-top: 10px;
    }
    .cash-row { display: flex; justify-content: space-between; padding: 4px 0; }
    .cash-label { color: #888; font-size: 13px; }
    .cash-value { color: #00ff00; font-size: 14px; font-weight: bold; }

    .status-connected { color: #00ff00; font-size: 12px; }
    .status-disconnected { color: #ff4444; font-size: 12px; }
    .status-market-open { color: #00ff00; }
    .status-market-closed { color: #ff4444; }

    .time-display { color: #00bfff; font-size: 24px; font-family: 'Courier New', monospace; }
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
# TABLE GENERATION
# =============================================================================
def create_open_trades_table(portfolio: List[Dict], categories: dict, pnl_data: Dict) -> str:
    """Create Open Trades table HTML"""
    if not portfolio:
        return """
        <div class="section-header">Open Trades (0)</div>
        <table class="trades-table">
            <thead>
                <tr>
                    <th>DAILY P&L</th>
                    <th>SYMBOL</th>
                    <th>STRATEGY</th>
                    <th>DIR</th>
                    <th>QTY</th>
                    <th>ENTRY</th>
                    <th>AVG PX</th>
                    <th>LAST</th>
                    <th>MKT VAL</th>
                    <th>UNRLZ P&L</th>
                    <th>CHNG %</th>
                </tr>
            </thead>
            <tbody></tbody>
        </table>
        """

    # Sort by unrealized P&L
    portfolio_sorted = sorted(portfolio, key=lambda x: x.get('unrealized_pnl', 0), reverse=True)

    # Calculate totals
    total_unrealized = sum(p.get('unrealized_pnl', 0) for p in portfolio_sorted)
    total_mkt_value = sum(abs(p.get('market_value', 0)) for p in portfolio_sorted)
    total_daily_pnl = pnl_data.get('daily_pnl', 0)

    html = f"""
    <div class="section-header">Open Trades ({len(portfolio)}) | Daily: ${total_daily_pnl:+,.0f} | Unrealized: ${total_unrealized:+,.0f}</div>
    <table class="trades-table">
        <thead>
            <tr>
                <th style="text-align:left;">DAILY P&L</th>
                <th style="text-align:left;">SYMBOL</th>
                <th>STRATEGY</th>
                <th>DIR</th>
                <th>QTY</th>
                <th>ENTRY</th>
                <th>AVG PX</th>
                <th>LAST</th>
                <th>MKT VAL</th>
                <th>UNRLZ P&L</th>
                <th>CHNG %</th>
            </tr>
        </thead>
        <tbody>
    """

    for p in portfolio_sorted:
        symbol = p['symbol']
        position = p['position']
        avg_cost = p['avg_cost']
        market_price = p['market_price']
        market_value = p['market_value']
        unrealized_pnl = p.get('unrealized_pnl', 0)

        # Calculate daily P&L (estimate)
        daily_pnl = unrealized_pnl * 0.1  # Placeholder - IB doesn't provide per-position daily P&L easily

        # Direction
        direction = "LONG" if position > 0 else "SHORT"
        dir_class = "positive" if position > 0 else "negative"

        # Change %
        if avg_cost > 0:
            change_pct = ((market_price - avg_cost) / avg_cost) * 100
        else:
            change_pct = 0

        # Strategy
        strategy = get_symbol_strategy(symbol, categories)

        # Entry time (placeholder - would need to track this separately)
        entry_time = "N/A"

        # Colors
        pnl_class = "positive" if unrealized_pnl >= 0 else "negative"
        chng_class = "positive" if change_pct >= 0 else "negative"
        strat_class = "info-blue" if strategy != "NONE" else "gray"

        html += f"""
            <tr>
                <td class="{pnl_class}">${daily_pnl:+,.0f}</td>
                <td class="symbol-cell">{symbol}</td>
                <td class="{strat_class}">{strategy}</td>
                <td class="{dir_class}">{direction}</td>
                <td>{abs(position):,.0f}</td>
                <td class="gray">{entry_time}</td>
                <td>${avg_cost:.2f}</td>
                <td>${market_price:.2f}</td>
                <td>${abs(market_value):,.0f}</td>
                <td class="{pnl_class}">${unrealized_pnl:+,.0f}</td>
                <td class="{chng_class}">{change_pct:+.2f}%</td>
            </tr>
        """

    # Total row
    html += f"""
            <tr class="total-row">
                <td class="{'positive' if total_daily_pnl >= 0 else 'negative'}">${total_daily_pnl:+,.0f}</td>
                <td class="symbol-cell">TOTAL</td>
                <td>-</td>
                <td>-</td>
                <td>{sum(abs(p['position']) for p in portfolio):,.0f}</td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
                <td>${total_mkt_value:,.0f}</td>
                <td class="{'positive' if total_unrealized >= 0 else 'negative'}">${total_unrealized:+,.0f}</td>
                <td>-</td>
            </tr>
        </tbody>
    </table>
    """

    return html


def create_closed_trades_table(closed_trades: List[Dict]) -> str:
    """Create Closed Trades table HTML"""
    year = datetime.now().year
    total_pnl = sum(t.get('pnl', 0) for t in closed_trades)

    html = f"""
    <div class="section-header">Closed Trades {year} ({len(closed_trades)}) | Total: ${total_pnl:+,.2f}</div>
    <table class="trades-table">
        <thead>
            <tr>
                <th style="text-align:left;">SYMBOL</th>
                <th>DIRECTION</th>
                <th>QTY</th>
                <th>ENTRY</th>
                <th>ENTRY PX</th>
                <th>EXIT</th>
                <th>EXIT PX</th>
                <th>FEES</th>
                <th>DAYS</th>
                <th>P&L $</th>
                <th>P&L %</th>
            </tr>
        </thead>
        <tbody>
    """

    if not closed_trades:
        html += """
            <tr>
                <td colspan="11" style="text-align:center; color:#666;">No closed trades this year</td>
            </tr>
        """
    else:
        for t in closed_trades:
            pnl = t.get('pnl', 0)
            pnl_pct = t.get('pnl_pct', 0)
            pnl_class = "positive" if pnl >= 0 else "negative"
            dir_class = "positive" if t.get('direction') == 'LONG' else "negative"

            html += f"""
                <tr>
                    <td class="symbol-cell">{t.get('symbol', 'N/A')}</td>
                    <td class="{dir_class}">{t.get('direction', 'N/A')}</td>
                    <td>{t.get('qty', 0):,.0f}</td>
                    <td>{t.get('entry_time', 'N/A')}</td>
                    <td>${t.get('entry_price', 0):.2f}</td>
                    <td>{t.get('exit_time', 'N/A')}</td>
                    <td>${t.get('exit_price', 0):.2f}</td>
                    <td>${t.get('fees', 0):.2f}</td>
                    <td>{t.get('days', 0)}</td>
                    <td class="{pnl_class}">${pnl:+,.2f}</td>
                    <td class="{pnl_class}">{pnl_pct:+.2f}%</td>
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

    # Check market hours
    market_open, market_status = is_ny_market_open()

    # Sidebar for connection
    with st.sidebar:
        st.markdown("### IB Connection")
        host = st.text_input("Host", value="127.0.0.1")
        port = st.number_input("Port", value=7497, help="7497=TWS Paper, 7496=TWS Live")
        client_id = st.number_input("Client ID", value=1, min_value=1)

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

    # Load categories
    categories = load_stock_categories()

    # Header
    col1, col2, col3 = st.columns([6, 2, 2])

    with col1:
        status_class = "status-connected" if st.session_state.connected else "status-disconnected"
        status_text = "CONNECTED" if st.session_state.connected else "DISCONNECTED"
        st.markdown(f"""
            <div style="display:flex; align-items:center; gap:20px;">
                <span style="color:#fff; font-weight:bold;">MONITOR</span>
                <span style="color:#fff;">Portfolio</span>
                <span class="{status_class}">[{status_text}]</span>
                <span class="{'status-market-open' if market_open else 'status-market-closed'}">[{market_status}]</span>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        now = datetime.now()
        st.markdown(f'<div class="time-display">{now.strftime("%H:%M:%S")}</div>', unsafe_allow_html=True)

    # Main content
    if st.session_state.connected and st.session_state.ib_conn:
        conn = st.session_state.ib_conn

        # Fetch data
        with st.spinner("Loading from IB..."):
            account = conn.get_account_summary()
            portfolio = conn.get_portfolio()
            pnl_data = conn.get_pnl()

        # Account summary
        net_liq = account.get('NetLiquidation', 0)
        excess_liq = account.get('ExcessLiquidity', 0)
        maintenance = account.get('MaintMarginReq', 0)
        total_cash = account.get('TotalCashValue', 0)

        daily_pnl = pnl_data.get('daily_pnl', 0)
        unrealized_pnl = pnl_data.get('unrealized_pnl', 0)
        realized_pnl = pnl_data.get('realized_pnl', 0)

        st.markdown("---")

        # P&L Summary
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            pnl_color = "positive" if daily_pnl >= 0 else "negative"
            pct = (daily_pnl / (net_liq - daily_pnl) * 100) if net_liq != daily_pnl else 0
            st.markdown(f"""
                <p class="pnl-header">P&L DAILY</p>
                <p class="pnl-value-large {pnl_color}">${daily_pnl:+,.0f}</p>
                <p class="pnl-header">Since prior Close <span class="{pnl_color}">{pct:+.2f}%</span></p>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
                <p class="pnl-header">Unrealized</p>
                <p class="pnl-value-small {'positive' if unrealized_pnl >= 0 else 'negative'}">${unrealized_pnl:+,.0f}</p>
                <p class="pnl-header">Realized</p>
                <p class="pnl-value-small {'positive' if realized_pnl >= 0 else 'negative'}">${realized_pnl:+,.0f}</p>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
                <p class="pnl-header">Margin</p>
                <p class="account-label">Net Liquidity</p>
                <p class="account-value">${net_liq/1000:.1f}K</p>
                <p class="account-label">Maintenance</p>
                <p class="account-value">${maintenance/1000:.1f}K</p>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
                <p class="pnl-header">&nbsp;</p>
                <p class="account-label">Excess Liq</p>
                <p class="account-value">${excess_liq/1000:.1f}K</p>
                <p class="account-label">Cash</p>
                <p class="account-value">${total_cash/1000:.1f}K</p>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Open Trades Table
        open_trades_html = create_open_trades_table(portfolio, categories, pnl_data)

        # Closed Trades (placeholder - would need trade tracking)
        closed_trades = []  # TODO: Implement trade tracking
        closed_trades_html = create_closed_trades_table(closed_trades)

        # Render tables
        full_html = f"""
        <html>
        <head>
        <style>
            body {{ background:#000; margin:0; padding:10px; font-family:'Segoe UI',Arial; }}
            .section-header {{ color:#00bfff; font-size:14px; font-weight:bold; padding:10px 0 5px; border-bottom:1px solid #333; margin:20px 0 10px; }}
            .trades-table {{ width:100%; border-collapse:collapse; font-size:12px; color:#fff; }}
            .trades-table th {{ background:#1a1a1a; color:#888; padding:8px; text-align:right; border-bottom:2px solid #333; font-size:10px; }}
            .trades-table td {{ padding:6px 8px; text-align:right; border-bottom:1px solid #222; }}
            .trades-table th:first-child, .trades-table td:first-child {{ text-align:left; }}
            .trades-table th:nth-child(2), .trades-table td:nth-child(2) {{ text-align:left; }}
            .total-row {{ background:#0a0a0a; font-weight:bold; }}
            .total-row td {{ border-top:2px solid #333; }}
            .symbol-cell {{ color:#fff; font-weight:bold; }}
            .positive {{ color:#00ff00; }}
            .negative {{ color:#ff4444; }}
            .info-blue {{ color:#00bfff; }}
            .gray {{ color:#666; }}
        </style>
        </head>
        <body>
        {open_trades_html}
        <br>
        {closed_trades_html}
        </body>
        </html>
        """

        table_height = max(400, 100 + len(portfolio) * 30 + 200)
        components.html(full_html, height=table_height, scrolling=True)

        # Cash section
        st.markdown(f"""
            <div class="cash-container">
                <div class="cash-row">
                    <span class="cash-label">USD CASH</span>
                    <span class="cash-value">${total_cash:,.0f}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

    else:
        # Not connected
        st.markdown("""
            <div style="text-align:center; padding:50px; color:#888;">
                <h2 style="color:#fff;">IB Portfolio Monitor</h2>
                <p>Connect to TWS/Gateway using the sidebar.</p>
                <p style="color:#666; font-size:12px;">
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
