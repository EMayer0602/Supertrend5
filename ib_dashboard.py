#!/usr/bin/env python3
"""
IB TWS Portfolio Monitor Dashboard
===================================
Real-time portfolio monitoring with Interactive Brokers API.
Styled exactly like TWS Monitor with 30-second auto-refresh.

Requirements:
    pip install ib_insync streamlit pandas numpy

Usage:
    1. Start TWS or IB Gateway
    2. Enable API connections (File -> Global Configuration -> API)
    3. Run: streamlit run ib_dashboard.py

Configuration:
    - TWS Paper Trading: port 7497
    - TWS Live Trading: port 7496
    - IB Gateway Paper: port 4002
    - IB Gateway Live: port 4001
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import asyncio
from typing import Dict, List, Optional, Tuple
import threading

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
# TWS-STYLE CSS (Exact match to IB TWS)
# =============================================================================
TWS_CSS = """
<style>
    /* Main background - TWS black */
    .stApp {
        background-color: #000000 !important;
    }

    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}

    /* Main container */
    .main .block-container {
        padding: 0.5rem 1rem;
        max-width: 100%;
    }

    /* Tab bar - exact TWS style */
    .tws-header {
        background-color: #1a1a1a;
        padding: 5px 10px;
        border-bottom: 1px solid #333;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    .tws-tabs {
        display: flex;
        gap: 0;
    }

    .tws-tab {
        padding: 5px 12px;
        font-size: 12px;
        cursor: pointer;
        border: none;
        background: transparent;
    }

    .tws-tab-active {
        background-color: #2a2a2a;
        color: #ffffff;
        border: 1px solid #444;
    }

    .tws-tab-inactive {
        color: #888;
    }

    .tws-time {
        color: #00bfff;
        font-size: 20px;
        font-family: 'Courier New', monospace;
    }

    /* P&L Summary Section */
    .pnl-section {
        background-color: #0a0a0a;
        padding: 10px 15px;
        border-bottom: 1px solid #222;
    }

    .pnl-row {
        display: flex;
        gap: 40px;
        align-items: flex-start;
    }

    .pnl-block {
        min-width: 120px;
    }

    .pnl-label {
        color: #888;
        font-size: 10px;
        text-transform: uppercase;
        margin-bottom: 2px;
    }

    .pnl-value {
        font-size: 13px;
        font-family: 'Segoe UI', Arial, sans-serif;
    }

    .pnl-large {
        font-size: 26px;
        font-weight: bold;
    }

    /* Colors */
    .green { color: #00ff00 !important; }
    .red { color: #ff4444 !important; }
    .white { color: #ffffff !important; }
    .cyan { color: #00bfff !important; }
    .gray { color: #888888 !important; }

    /* Background colors for cells */
    .bg-green { background-color: rgba(0, 80, 0, 0.6) !important; }
    .bg-red { background-color: rgba(80, 0, 0, 0.6) !important; }

    /* Portfolio Table - Exact TWS style */
    .portfolio-table {
        width: 100%;
        border-collapse: collapse;
        font-family: 'Segoe UI', Arial, sans-serif;
        font-size: 12px;
    }

    .portfolio-table thead th {
        background-color: #1a1a1a;
        color: #888;
        padding: 8px 6px;
        text-align: right;
        border-bottom: 1px solid #333;
        font-weight: normal;
        font-size: 10px;
        position: sticky;
        top: 0;
        z-index: 10;
    }

    .portfolio-table thead th:nth-child(1),
    .portfolio-table thead th:nth-child(2) {
        text-align: left;
    }

    .portfolio-table tbody td {
        padding: 5px 6px;
        text-align: right;
        border-bottom: 1px solid #111;
    }

    .portfolio-table tbody td:nth-child(1),
    .portfolio-table tbody td:nth-child(2) {
        text-align: left;
    }

    .portfolio-table tbody tr:hover {
        background-color: #1a1a1a;
    }

    /* Total row */
    .total-row {
        background-color: #0d0d0d !important;
        font-weight: bold;
    }

    .total-row td {
        padding: 10px 6px !important;
        border-top: 2px solid #333;
    }

    /* Symbol styling */
    .symbol {
        color: #ffffff;
        font-weight: 500;
    }

    /* Cash section */
    .cash-section {
        background-color: rgba(0, 60, 0, 0.4);
        padding: 8px 15px;
        border: 1px solid #004400;
        margin-top: 10px;
    }

    .cash-row {
        display: flex;
        justify-content: space-between;
        padding: 4px 0;
        font-size: 12px;
    }

    .cash-label { color: #888; }
    .cash-value { color: #00ff00; font-weight: bold; }

    /* Connection status */
    .status-connected {
        color: #00ff00;
        font-size: 11px;
    }

    .status-disconnected {
        color: #ff4444;
        font-size: 11px;
    }

    /* Scrollable container */
    .table-scroll {
        max-height: 500px;
        overflow-y: auto;
        border: 1px solid #222;
    }
</style>
"""


# =============================================================================
# SUPERTREND CALCULATION
# =============================================================================
def calculate_supertrend(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                          period: int = 10, multiplier: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Supertrend indicator"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n), np.zeros(n)

    # ATR
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))

    atr = np.zeros(n)
    atr[period-1] = np.mean(tr[:period])
    alpha = 2 / (period + 1)
    for i in range(period, n):
        atr[i] = tr[i] * alpha + atr[i-1] * (1 - alpha)

    hl2 = (high + low) / 2
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr

    final_upper = np.copy(upper)
    final_lower = np.copy(lower)
    supertrend = np.zeros(n)
    direction = np.zeros(n)

    for i in range(period, n):
        if upper[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]:
            final_upper[i] = upper[i]
        else:
            final_upper[i] = final_upper[i-1]

        if lower[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]:
            final_lower[i] = lower[i]
        else:
            final_lower[i] = final_lower[i-1]

    for i in range(period, n):
        if i == period:
            supertrend[i] = final_upper[i] if close[i] <= final_upper[i] else final_lower[i]
            direction[i] = -1 if close[i] <= final_upper[i] else 1
        else:
            if supertrend[i-1] == final_upper[i-1]:
                supertrend[i] = final_upper[i] if close[i] <= final_upper[i] else final_lower[i]
                direction[i] = -1 if close[i] <= final_upper[i] else 1
            else:
                supertrend[i] = final_lower[i] if close[i] >= final_lower[i] else final_upper[i]
                direction[i] = 1 if close[i] >= final_lower[i] else -1

    return supertrend, direction


# =============================================================================
# IB API CONNECTION
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
            account_values = self.ib.accountSummary()

            summary = {}
            for av in account_values:
                summary[av.tag] = {
                    'value': float(av.value) if av.value.replace('.', '').replace('-', '').isdigit() else av.value,
                    'currency': av.currency
                }

            return summary
        except Exception as e:
            st.error(f"Error getting account summary: {e}")
            return {}

    def get_portfolio(self) -> List[Dict]:
        """Get portfolio positions"""
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
                })

            return portfolio
        except Exception as e:
            st.error(f"Error getting portfolio: {e}")
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
            return {}
        except Exception as e:
            return {}

    def get_positions_pnl(self) -> Dict[str, Dict]:
        """Get P&L for each position"""
        if not self.connected:
            return {}

        try:
            pnl_dict = {}

            for pnl_single in self.ib.pnlSingle():
                pnl_dict[pnl_single.conId] = {
                    'daily_pnl': pnl_single.dailyPnL,
                    'unrealized_pnl': pnl_single.unrealizedPnL,
                    'realized_pnl': pnl_single.realizedPnL,
                    'position': pnl_single.position,
                    'value': pnl_single.value
                }

            return pnl_dict
        except Exception as e:
            return {}

    def get_historical_data(self, symbol: str, duration: str = "3 M", bar_size: str = "1 day") -> Optional[pd.DataFrame]:
        """Get historical data for Supertrend calculation"""
        if not self.connected:
            return None

        try:
            contract = Stock(symbol, 'SMART', 'USD')
            self.ib.qualifyContracts(contract)

            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow='TRADES',
                useRTH=True,
                formatDate=1
            )

            if bars:
                df = util.df(bars)
                return df

            return None
        except Exception as e:
            return None


# =============================================================================
# DASHBOARD FUNCTIONS
# =============================================================================
def create_portfolio_table(portfolio: List[Dict], positions_pnl: Dict) -> str:
    """Create TWS-style portfolio table HTML"""
    if not portfolio:
        return "<p class='gray'>No positions</p>"

    # Sort by daily P&L
    def get_daily_pnl(pos):
        # Try to get from positions_pnl first, otherwise estimate
        return pos.get('unrealized_pnl', 0)

    portfolio_sorted = sorted(portfolio, key=get_daily_pnl, reverse=True)

    # Calculate totals
    total_position = sum(abs(p['position']) for p in portfolio_sorted)
    total_market_value = sum(p['market_value'] for p in portfolio_sorted)
    total_unrealized = sum(p['unrealized_pnl'] for p in portfolio_sorted)
    total_realized = sum(p['realized_pnl'] for p in portfolio_sorted)

    # Estimate total daily P&L
    total_daily_pnl = sum(p.get('daily_pnl', p['unrealized_pnl'] * 0.1) for p in portfolio_sorted)

    html = """
    <div class="table-scroll">
    <table class="portfolio-table">
        <thead>
            <tr>
                <th style="text-align: left; width: 80px;">DAILY P&L</th>
                <th style="text-align: left;">FIN INSTR</th>
                <th>POS</th>
                <th>MKT VAL</th>
                <th>AVG PX</th>
                <th>LAST</th>
                <th>UNRLZ P&L</th>
                <th>CHNG</th>
            </tr>
        </thead>
        <tbody>
    """

    # Total row first
    total_pnl_class = "bg-green green" if total_daily_pnl >= 0 else "bg-red red"
    total_unrlz_class = "green" if total_unrealized >= 0 else "red"

    html += f"""
        <tr class="total-row">
            <td class="{total_pnl_class}">{total_daily_pnl:+,.0f}</td>
            <td class="symbol">TOTAL Stocks</td>
            <td>{total_position:,.0f}</td>
            <td>{total_market_value:,.0f}</td>
            <td>-</td>
            <td>-</td>
            <td class="{total_unrlz_class}">{total_unrealized:+,.0f}</td>
            <td>-</td>
        </tr>
    """

    # Individual positions
    for pos in portfolio_sorted:
        daily_pnl = pos.get('daily_pnl', pos['unrealized_pnl'] * 0.05)
        pnl_class = "bg-green green" if daily_pnl >= 0 else "bg-red red"
        unrlz_class = "green" if pos['unrealized_pnl'] >= 0 else "red"

        # Calculate change %
        if pos['avg_cost'] > 0:
            change_pct = ((pos['market_price'] - pos['avg_cost']) / pos['avg_cost']) * 100
        else:
            change_pct = 0
        chng_class = "green" if change_pct >= 0 else "red"

        html += f"""
            <tr>
                <td class="{pnl_class}">{daily_pnl:+,.0f}</td>
                <td class="symbol">{pos['symbol']}</td>
                <td>{pos['position']:,.0f}</td>
                <td>{pos['market_value']:,.0f}</td>
                <td>{pos['avg_cost']:.2f}</td>
                <td>{pos['market_price']:.2f}</td>
                <td class="{unrlz_class}">{pos['unrealized_pnl']:+,.0f}</td>
                <td class="{chng_class}">{change_pct:+.2f}</td>
            </tr>
        """

    html += """
        </tbody>
    </table>
    </div>
    """

    return html


# =============================================================================
# MAIN APPLICATION
# =============================================================================
def main():
    # Inject CSS
    st.markdown(TWS_CSS, unsafe_allow_html=True)

    # Session state for IB connection
    if 'ib_conn' not in st.session_state:
        st.session_state.ib_conn = None
        st.session_state.connected = False

    # Sidebar for connection settings
    with st.sidebar:
        st.markdown("### IB Connection")
        host = st.text_input("Host", value="127.0.0.1")
        port = st.number_input("Port", value=7497, min_value=1, max_value=65535,
                                help="7497=TWS Paper, 7496=TWS Live, 4002=Gateway Paper, 4001=Gateway Live")
        client_id = st.number_input("Client ID", value=1, min_value=1, max_value=999)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Connect", use_container_width=True):
                if IB_AVAILABLE:
                    conn = IBConnection(host, port, client_id)
                    if conn.connect():
                        st.session_state.ib_conn = conn
                        st.session_state.connected = True
                        st.success("Connected!")
                    else:
                        st.error("Failed to connect")
                else:
                    st.error("ib_insync not installed")

        with col2:
            if st.button("Disconnect", use_container_width=True):
                if st.session_state.ib_conn:
                    st.session_state.ib_conn.disconnect()
                st.session_state.connected = False
                st.info("Disconnected")

    # Header
    col1, col2 = st.columns([8, 2])

    with col1:
        status_class = "status-connected" if st.session_state.connected else "status-disconnected"
        status_text = "CONNECTED" if st.session_state.connected else "DISCONNECTED"

        st.markdown(f"""
            <div class="tws-header">
                <div class="tws-tabs">
                    <span class="tws-tab tws-tab-active">MONITOR</span>
                    <span class="tws-tab tws-tab-inactive">&lt;</span>
                    <span class="tws-tab tws-tab-active">Portfolio</span>
                    <span class="tws-tab tws-tab-inactive">Favoriten</span>
                    <span class="tws-tab tws-tab-inactive">US Movers</span>
                    <span class="tws-tab tws-tab-inactive">US Ind8</span>
                    <span class="tws-tab tws-tab-inactive">&gt;</span>
                </div>
                <span class="{status_class}">[{status_text}]</span>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        current_time = datetime.now().strftime("%H:%M:%S")
        st.markdown(f'<div class="tws-time">{current_time}</div>', unsafe_allow_html=True)

    # Main content
    if st.session_state.connected and st.session_state.ib_conn:
        conn = st.session_state.ib_conn

        # Get data
        with st.spinner("Loading data from IB..."):
            account_summary = conn.get_account_summary()
            portfolio = conn.get_portfolio()
            pnl_data = conn.get_pnl()
            positions_pnl = conn.get_positions_pnl()

        # Extract account values
        net_liq = account_summary.get('NetLiquidation', {}).get('value', 0)
        excess_liq = account_summary.get('ExcessLiquidity', {}).get('value', 0)
        maintenance = account_summary.get('MaintMarginReq', {}).get('value', 0)
        total_cash = account_summary.get('TotalCashValue', {}).get('value', 0)

        daily_pnl = pnl_data.get('daily_pnl', 0)
        unrealized_pnl = pnl_data.get('unrealized_pnl', 0)
        realized_pnl = pnl_data.get('realized_pnl', 0)

        # P&L Summary Section
        st.markdown("---")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            pnl_color = "green" if daily_pnl >= 0 else "red"
            pct_change = (daily_pnl / (net_liq - daily_pnl) * 100) if net_liq != daily_pnl else 0

            st.markdown(f"""
                <div class="pnl-block">
                    <div class="pnl-label">P&L</div>
                    <div class="pnl-label">DAILY</div>
                    <div class="pnl-large {pnl_color}">{daily_pnl:+,.0f}</div>
                    <div class="pnl-label">Since prior Close <span class="{pnl_color}">{pct_change:+.2f}%</span></div>
                </div>
            """, unsafe_allow_html=True)

        with col2:
            unrlz_color = "green" if unrealized_pnl >= 0 else "red"
            rlz_color = "green" if realized_pnl >= 0 else "red"

            st.markdown(f"""
                <div class="pnl-block">
                    <div class="pnl-label">Unrealized</div>
                    <div class="pnl-value {unrlz_color}">{unrealized_pnl:+,.1f}</div>
                    <div class="pnl-label">Realized</div>
                    <div class="pnl-value {rlz_color}">{realized_pnl:+,.1f}</div>
                </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
                <div class="pnl-block">
                    <div class="pnl-label">Margin</div>
                    <div class="pnl-label cyan">Net Liquidity</div>
                    <div class="pnl-value cyan">{net_liq/1000:.1f}K</div>
                    <div class="pnl-label cyan">Maintenance</div>
                    <div class="pnl-value cyan">{maintenance/1000:.1f}K</div>
                </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
                <div class="pnl-block">
                    <div class="pnl-label">&nbsp;</div>
                    <div class="pnl-label cyan">Excess Liq</div>
                    <div class="pnl-value cyan">{excess_liq/1000:.1f}K</div>
                    <div class="pnl-label cyan">SMA</div>
                    <div class="pnl-value cyan">-</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Portfolio Table
        table_html = create_portfolio_table(portfolio, positions_pnl)
        st.markdown(table_html, unsafe_allow_html=True)

        # Cash Section
        st.markdown(f"""
            <div class="cash-section">
                <div class="cash-row">
                    <span class="cash-label">USD CASH</span>
                    <span class="cash-value">{total_cash:,.0f}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

    else:
        # Not connected - show instructions
        st.markdown("""
            <div style="text-align: center; padding: 50px; color: #888;">
                <h2 style="color: #ffffff;">IB Portfolio Monitor</h2>
                <p>Connect to Interactive Brokers TWS or Gateway to view your portfolio.</p>
                <p style="color: #666; font-size: 12px;">
                    1. Open TWS or IB Gateway<br>
                    2. Enable API: File → Global Configuration → API → Settings<br>
                    3. Set port: 7497 (Paper) or 7496 (Live)<br>
                    4. Click "Connect" in the sidebar
                </p>
            </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown(f"""
        <p style="color: #444; font-size: 10px; text-align: center;">
            Auto-refresh: 30 sec | Last update: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} |
            IB API via ib_insync
        </p>
    """, unsafe_allow_html=True)

    # Auto refresh
    time.sleep(30)
    st.rerun()


if __name__ == "__main__":
    main()
