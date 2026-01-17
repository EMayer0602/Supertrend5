#!/usr/bin/env python3
"""
TWS-Style Portfolio Monitor Dashboard v2.0
==========================================
A real-time portfolio monitoring dashboard styled like Interactive Brokers TWS.
Auto-refreshes every 30 seconds with live market data.

Usage:
    streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import json
import time
import os

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Portfolio Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# TWS-STYLE CSS
# =============================================================================
TWS_CSS = """
<style>
    /* Main background */
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
        padding: 1rem 2rem;
        max-width: 100%;
    }

    /* Tab bar styling */
    .tab-container {
        background-color: #1a1a1a;
        padding: 8px 15px;
        border-bottom: 1px solid #333;
        display: flex;
        align-items: center;
        gap: 5px;
    }

    .tab-active {
        background-color: #2a2a2a;
        color: #ffffff;
        padding: 6px 15px;
        border: 1px solid #444;
        font-size: 13px;
        cursor: pointer;
    }

    .tab-inactive {
        background-color: transparent;
        color: #888;
        padding: 6px 15px;
        border: none;
        font-size: 13px;
        cursor: pointer;
    }

    /* P&L Box */
    .pnl-container {
        background-color: #0a0a0a;
        border: 1px solid #333;
        padding: 15px 20px;
        margin: 10px 0;
    }

    .pnl-header {
        color: #888;
        font-size: 11px;
        margin-bottom: 5px;
    }

    .pnl-value-large {
        font-size: 32px;
        font-weight: bold;
        font-family: 'Segoe UI', Arial, sans-serif;
    }

    .pnl-value-small {
        font-size: 14px;
        font-family: 'Segoe UI', Arial, sans-serif;
    }

    .positive { color: #00ff00 !important; }
    .negative { color: #ff4444 !important; }
    .neutral { color: #ffffff !important; }
    .info-blue { color: #00bfff !important; }
    .gray { color: #666666 !important; }

    /* Account Info */
    .account-row {
        display: flex;
        justify-content: space-between;
        padding: 3px 0;
    }

    .account-label {
        color: #00bfff;
        font-size: 12px;
    }

    .account-value {
        color: #ffffff;
        font-size: 12px;
    }

    /* Portfolio Table */
    .portfolio-table {
        width: 100%;
        border-collapse: collapse;
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 13px;
        margin-top: 10px;
    }

    .portfolio-table th {
        background-color: #1a1a1a;
        color: #888;
        padding: 10px 8px;
        text-align: right;
        border-bottom: 2px solid #333;
        font-weight: normal;
        font-size: 11px;
        position: sticky;
        top: 0;
    }

    .portfolio-table th:nth-child(1),
    .portfolio-table th:nth-child(2) {
        text-align: left;
    }

    .portfolio-table td {
        padding: 6px 8px;
        text-align: right;
        border-bottom: 1px solid #1a1a1a;
        white-space: nowrap;
    }

    .portfolio-table td:nth-child(1),
    .portfolio-table td:nth-child(2) {
        text-align: left;
    }

    .portfolio-table tr:hover {
        background-color: #1a1a1a;
    }

    .portfolio-table .total-row {
        background-color: #0a0a0a;
        font-weight: bold;
        border-top: 2px solid #333;
    }

    .portfolio-table .total-row td {
        padding: 12px 8px;
    }

    /* Cell backgrounds for P&L */
    .cell-positive {
        background-color: rgba(0, 100, 0, 0.4);
        color: #00ff00;
    }

    .cell-negative {
        background-color: rgba(100, 0, 0, 0.4);
        color: #ff4444;
    }

    /* Symbol column */
    .symbol-cell {
        color: #ffffff;
        font-weight: bold;
    }

    /* Cash section */
    .cash-container {
        background-color: rgba(0, 80, 0, 0.3);
        border: 1px solid #004400;
        padding: 10px 15px;
        margin-top: 15px;
    }

    .cash-row {
        display: flex;
        justify-content: space-between;
        padding: 5px 0;
    }

    .cash-label {
        color: #888;
        font-size: 13px;
    }

    .cash-value {
        color: #00ff00;
        font-size: 14px;
        font-weight: bold;
    }

    /* Time display */
    .time-display {
        color: #00bfff;
        font-size: 28px;
        font-family: 'Courier New', monospace;
        text-align: right;
        padding: 10px;
    }

    /* Scrollable table container */
    .table-container {
        max-height: 600px;
        overflow-y: auto;
        border: 1px solid #333;
    }

    /* Streamlit metric override */
    [data-testid="stMetric"] {
        background-color: #1a1a1a;
        padding: 10px;
        border: 1px solid #333;
    }

    [data-testid="stMetricLabel"] {
        color: #888 !important;
    }
</style>
"""

# =============================================================================
# SUPERTREND CALCULATION
# =============================================================================
def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n), np.zeros(n)

    # ATR calculation
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
# DATA LOADING
# =============================================================================
def load_portfolio_config():
    """Load portfolio configuration from JSON file"""
    config_path = os.path.join(os.path.dirname(__file__), 'portfolio_config.json')

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        # Default portfolio
        return {
            "account": {"eur_cash": 20000, "usd_cash": 31228},
            "positions": [
                {"symbol": "MSFT", "qty": 50, "avg_price": 380.00},
                {"symbol": "AAPL", "qty": 100, "avg_price": 185.00},
            ]
        }


def load_stock_categories():
    """Load stock categories/strategies from JSON file"""
    config_path = os.path.join(os.path.dirname(__file__), 'stock_categories.json')

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return None


def get_symbol_strategy(symbol: str, categories: dict) -> str:
    """Get the assigned strategy for a symbol"""
    if not categories:
        return "N/A"

    for strat_name, strat_info in categories.get('strategies', {}).items():
        if symbol in strat_info.get('tickers', []):
            return strat_name

    return "NONE"


@st.cache_data(ttl=30)  # Cache for 30 seconds
def fetch_market_data(symbols):
    """Fetch current market data for all symbols"""
    data = {}

    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d", interval="1d")

            if hist.empty:
                continue

            current = hist.iloc[-1]
            prev = hist.iloc[-2] if len(hist) > 1 else hist.iloc[-1]

            # Get more history for Supertrend
            hist_long = ticker.history(period="3mo", interval="1d")

            signal = "N/A"
            if len(hist_long) >= 20:
                _, direction = calculate_supertrend(
                    hist_long['High'].values,
                    hist_long['Low'].values,
                    hist_long['Close'].values,
                    period=10, multiplier=3.0
                )
                signal = "LONG" if direction[-1] == 1 else "SHORT"

            data[symbol] = {
                'last': current['Close'],
                'prev_close': prev['Close'],
                'high': current['High'],
                'low': current['Low'],
                'volume': current['Volume'],
                'signal': signal
            }

        except Exception as e:
            st.warning(f"Error fetching {symbol}: {e}")

    return data


# =============================================================================
# CALCULATIONS
# =============================================================================
def calculate_portfolio_values(positions, market_data):
    """Calculate all portfolio values"""
    portfolio = []

    for pos in positions:
        symbol = pos['symbol']
        qty = pos['qty']
        avg_price = pos['avg_price']

        if symbol not in market_data:
            continue

        md = market_data[symbol]
        last_price = md['last']
        prev_close = md['prev_close']

        # Calculate values
        mkt_value = qty * last_price
        daily_change = last_price - prev_close
        daily_change_pct = (daily_change / prev_close) * 100 if prev_close != 0 else 0
        daily_pnl = qty * daily_change

        # Unrealized P&L
        unrealized_pnl = qty * (last_price - avg_price)

        portfolio.append({
            'symbol': symbol,
            'qty': qty,
            'avg_price': avg_price,
            'last_price': last_price,
            'mkt_value': mkt_value,
            'daily_pnl': daily_pnl,
            'daily_change_pct': daily_change_pct,
            'unrealized_pnl': unrealized_pnl,
            'signal': md['signal']
        })

    return portfolio


# =============================================================================
# HTML TABLE GENERATION
# =============================================================================
def create_portfolio_table_html(portfolio, categories=None):
    """Create TWS-style HTML table with strategy column"""
    if not portfolio:
        return "<p style='color: #888;'>No positions</p>"

    # Sort by daily P&L (descending)
    portfolio_sorted = sorted(portfolio, key=lambda x: x['daily_pnl'], reverse=True)

    # Calculate totals
    total_daily_pnl = sum(p['daily_pnl'] for p in portfolio_sorted)
    total_mkt_value = sum(abs(p['mkt_value']) for p in portfolio_sorted)
    total_unrealized = sum(p['unrealized_pnl'] for p in portfolio_sorted)
    total_qty = sum(abs(p['qty']) for p in portfolio_sorted)

    html = """
    <div class="table-container">
    <table class="portfolio-table">
        <thead>
            <tr>
                <th style="text-align: left;">DAILY P&L</th>
                <th style="text-align: left;">FIN INSTR</th>
                <th>STRATEGY</th>
                <th>POS</th>
                <th>MKT VAL</th>
                <th>AVG PX</th>
                <th>LAST</th>
                <th>UNRLZ P&L</th>
                <th>CHNG</th>
                <th>SIGNAL</th>
            </tr>
        </thead>
        <tbody>
    """

    # Individual positions FIRST
    for p in portfolio_sorted:
        pnl_class = "cell-positive" if p['daily_pnl'] >= 0 else "cell-negative"
        unrlz_class = "cell-positive" if p['unrealized_pnl'] >= 0 else "cell-negative"
        chng_class = "positive" if p['daily_change_pct'] >= 0 else "negative"

        # Get strategy for symbol
        strategy = get_symbol_strategy(p['symbol'], categories) if categories else "N/A"

        # Signal color
        signal = p.get('signal', 'N/A')
        signal_class = "positive" if signal == "LONG" else "negative" if signal == "SHORT" else "neutral"

        # Format numbers
        daily_pnl_str = f"{p['daily_pnl']:+,.0f}" if p['daily_pnl'] != 0 else "0"
        unrlz_str = f"{p['unrealized_pnl']:+,.0f}" if p['unrealized_pnl'] != 0 else "0"
        chng_str = f"{p['daily_change_pct']:+.2f}"
        qty_str = f"{p['qty']:,}" if p['qty'] >= 0 else f"{p['qty']:,}"

        # Strategy color (cyan for assigned, gray for none)
        strat_class = "info-blue" if strategy != "NONE" else "gray"

        html += f"""
            <tr>
                <td class="{pnl_class}">{daily_pnl_str}</td>
                <td class="symbol-cell">{p['symbol']}</td>
                <td class="{strat_class}">{strategy}</td>
                <td>{qty_str}</td>
                <td>{abs(p['mkt_value']):,.0f}</td>
                <td>{p['avg_price']:.2f}</td>
                <td>{p['last_price']:.2f}</td>
                <td class="{unrlz_class}">{unrlz_str}</td>
                <td class="{chng_class}">{chng_str}</td>
                <td class="{signal_class}">{signal}</td>
            </tr>
        """

    # Total row LAST (at bottom)
    total_pnl_class = "cell-positive" if total_daily_pnl >= 0 else "cell-negative"
    total_unrlz_class = "cell-positive" if total_unrealized >= 0 else "cell-negative"
    total_pnl_str = f"{total_daily_pnl:+,.0f}" if total_daily_pnl != 0 else "0"
    total_unrlz_str = f"{total_unrealized:+,.0f}" if total_unrealized != 0 else "0"

    html += f"""
        <tr class="total-row">
            <td class="{total_pnl_class}">{total_pnl_str}</td>
            <td class="symbol-cell">TOTAL Stocks</td>
            <td>-</td>
            <td>{total_qty:,.0f}</td>
            <td>{total_mkt_value:,.0f}</td>
            <td>-</td>
            <td>-</td>
            <td class="{total_unrlz_class}">{total_unrlz_str}</td>
            <td>-</td>
            <td>-</td>
        </tr>
    """

    html += """
        </tbody>
    </table>
    </div>
    """

    return html


# =============================================================================
# MAIN APP
# =============================================================================
def main():
    # Inject CSS
    st.markdown(TWS_CSS, unsafe_allow_html=True)

    # Load portfolio and categories
    config = load_portfolio_config()
    positions = config['positions']
    account = config['account']
    categories = load_stock_categories()

    symbols = [p['symbol'] for p in positions]

    # Header with tabs and time
    col1, col2 = st.columns([8, 2])

    with col1:
        st.markdown("""
            <div class="tab-container">
                <span class="tab-active">MONITOR</span>
                <span class="tab-inactive">&lt;</span>
                <span class="tab-active">Portfolio</span>
                <span class="tab-inactive">Favoriten</span>
                <span class="tab-inactive">US Movers</span>
                <span class="tab-inactive">US Ind8</span>
                <span class="tab-inactive">&gt;</span>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        current_time = datetime.now().strftime("%H:%M:%S")
        st.markdown(f'<div class="time-display">{current_time}</div>', unsafe_allow_html=True)

    # Fetch market data
    with st.spinner('Loading market data...'):
        market_data = fetch_market_data(symbols)

    if not market_data:
        st.error("Could not fetch market data")
        time.sleep(30)
        st.rerun()
        return

    # Calculate portfolio values
    portfolio = calculate_portfolio_values(positions, market_data)

    # Calculate summary values
    total_daily_pnl = sum(p['daily_pnl'] for p in portfolio)
    total_unrealized = sum(p['unrealized_pnl'] for p in portfolio)
    total_mkt_value = sum(abs(p['mkt_value']) for p in portfolio)

    # Account values - use from config if available, otherwise calculate
    cash_total = account.get('usd_cash', 0) + account.get('eur_cash', 0)
    net_liquidity = account.get('net_liquidity', total_mkt_value + cash_total)
    excess_liq = account.get('excess_liquidity', net_liquidity * 0.85)
    maintenance = account.get('maintenance', net_liquidity * 0.15)

    # P&L Summary Row
    st.markdown("---")

    col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 2])

    with col1:
        st.markdown('<p class="pnl-header">P&L</p>', unsafe_allow_html=True)
        pnl_color = "positive" if total_daily_pnl >= 0 else "negative"
        pnl_sign = "+" if total_daily_pnl >= 0 else ""
        st.markdown(f'''
            <p class="pnl-header">DAILY</p>
            <p class="pnl-value-large {pnl_color}">{pnl_sign}{total_daily_pnl:,.0f}</p>
        ''', unsafe_allow_html=True)

        prev_mkt = total_mkt_value - total_daily_pnl
        pct_change = (total_daily_pnl / prev_mkt * 100) if prev_mkt != 0 else 0
        st.markdown(f'''
            <p class="pnl-header">Since prior Close <span class="{pnl_color}">{pct_change:+.2f}%</span></p>
        ''', unsafe_allow_html=True)

    with col2:
        unrlz_color = "positive" if total_unrealized >= 0 else "negative"
        st.markdown(f'''
            <p class="pnl-header">Unrealized</p>
            <p class="pnl-value-small {unrlz_color}">{total_unrealized:+,.1f}</p>
            <p class="pnl-header">Realized</p>
            <p class="pnl-value-small negative">-3.4K</p>
        ''', unsafe_allow_html=True)

    with col3:
        st.markdown(f'''
            <p class="pnl-header">Margin</p>
            <p class="account-label">Net Liquidity</p>
            <p class="account-value info-blue">{net_liquidity/1000:.1f}K</p>
            <p class="account-label">Maintenance</p>
            <p class="account-value info-blue">{maintenance/1000:.1f}K</p>
        ''', unsafe_allow_html=True)

    with col4:
        st.markdown(f'''
            <p class="pnl-header">&nbsp;</p>
            <p class="account-label">Excess Liq</p>
            <p class="account-value info-blue">{excess_liq/1000:.1f}K</p>
            <p class="account-label">SMA</p>
            <p class="account-value info-blue">-</p>
        ''', unsafe_allow_html=True)

    with col5:
        st.markdown(f'''
            <p class="pnl-header">Signals</p>
            <p class="account-label">Long</p>
            <p class="account-value positive">{sum(1 for p in portfolio if p['signal'] == 'LONG')}</p>
            <p class="account-label">Short</p>
            <p class="account-value negative">{sum(1 for p in portfolio if p['signal'] == 'SHORT')}</p>
        ''', unsafe_allow_html=True)

    st.markdown("---")

    # Portfolio Table (with strategy from stock_categories.json)
    table_html = create_portfolio_table_html(portfolio, categories)
    st.markdown(table_html, unsafe_allow_html=True)

    # Cash Section
    st.markdown(f'''
        <div class="cash-container">
            <div class="cash-row">
                <span class="cash-label">EUR CASH</span>
                <span class="cash-value">{account.get("eur_cash", 0):,.0f}</span>
            </div>
            <div class="cash-row">
                <span class="cash-label">USD CASH</span>
                <span class="cash-value">{account.get("usd_cash", 0):,.0f}</span>
            </div>
        </div>
    ''', unsafe_allow_html=True)

    # Footer with refresh info
    st.markdown("---")
    st.markdown(f'''
        <p style="color: #444; font-size: 11px; text-align: center;">
            Auto-refresh: 30 sec | Last update: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} |
            Data: yfinance
        </p>
    ''', unsafe_allow_html=True)

    # Auto refresh
    time.sleep(30)
    st.rerun()


if __name__ == "__main__":
    main()
