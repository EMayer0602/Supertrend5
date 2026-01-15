#!/usr/bin/env python3
"""
TWS Dashboard Generator - Full Featured
========================================
Connects to TWS, reads positions & account data, generates HTML dashboard.

Features:
- Real-time portfolio data from TWS
- Equity curve tracking
- Daily P&L chart
- Per-position P&L breakdown
- Performance metrics (Sharpe, Sortino, Drawdown, etc.)
- Auto-refresh capability

Usage:
    python generate_dashboard.py                    # Generate dashboard
    python generate_dashboard.py --live             # Auto-refresh every 60s
    python generate_dashboard.py --port 7496        # Use live trading port
"""

import json
import sys
import os
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import ib_insync
try:
    from ib_insync import IB, Stock, util
    IB_AVAILABLE = True
except ImportError:
    IB_AVAILABLE = False
    logger.warning("ib_insync not installed. Run: pip install ib_insync")

# Configuration
IB_HOST = "127.0.0.1"
IB_PORT = 7497  # Default: Paper trading (7496 for live)
IB_CLIENT_ID = 99

CATEGORIES_FILE = "stock_categories.json"
OUTPUT_FILE = "dashboard.html"
HISTORY_FILE = "portfolio_history.json"


def load_stock_categories() -> dict:
    """Load stock categories from config file"""
    try:
        with open(CATEGORIES_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def get_ticker_strategy(ticker: str) -> str:
    """Get strategy for a ticker"""
    config = load_stock_categories()
    if config:
        for strategy, data in config.get('strategies', {}).items():
            if ticker in data.get('tickers', []):
                return strategy
    return "MANUAL"


def load_history() -> Dict:
    """Load portfolio history from file"""
    try:
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {'equity': [], 'daily_pnl': [], 'trades': []}


def save_history(history: Dict):
    """Save portfolio history to file"""
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)


def connect_to_ib(port: int = IB_PORT) -> Optional[IB]:
    """Connect to Interactive Brokers TWS"""
    if not IB_AVAILABLE:
        logger.error("ib_insync not available")
        return None

    try:
        ib = IB()
        ib.connect(IB_HOST, port, clientId=IB_CLIENT_ID)
        logger.info(f"Connected to IB at {IB_HOST}:{port}")
        return ib
    except Exception as e:
        logger.error(f"Could not connect to IB: {e}")
        return None


def get_executions(ib: IB, days_back: int = 7) -> List[Dict]:
    """Get recent trade executions from TWS"""
    from ib_insync import ExecutionFilter

    executions = []
    try:
        # Request executions
        exec_filter = ExecutionFilter()
        fills = ib.reqExecutions(exec_filter)
        ib.sleep(2)

        for fill in fills:
            exec_info = fill.execution
            contract = fill.contract

            executions.append({
                'symbol': contract.symbol,
                'side': exec_info.side,  # BOT or SLD
                'quantity': int(exec_info.shares),
                'price': exec_info.price,
                'time': exec_info.time,
                'exec_id': exec_info.execId,
                'order_id': exec_info.orderId,
                'commission': fill.commissionReport.commission if fill.commissionReport else 0,
                'realized_pnl': fill.commissionReport.realizedPNL if fill.commissionReport else 0
            })

    except Exception as e:
        logger.warning(f"Could not get executions: {e}")

    return executions


def get_closed_trades(executions: List[Dict]) -> List[Dict]:
    """Calculate closed trades from executions"""
    # Group by symbol
    from collections import defaultdict
    trades_by_symbol = defaultdict(list)

    for ex in executions:
        trades_by_symbol[ex['symbol']].append(ex)

    closed_trades = []

    for symbol, execs in trades_by_symbol.items():
        # Sort by time
        execs.sort(key=lambda x: x['time'])

        position = 0
        entry_price = 0
        entry_time = None

        for ex in execs:
            qty = ex['quantity']
            if ex['side'] == 'SLD':
                qty = -qty

            if position == 0:
                # Opening trade
                position = qty
                entry_price = ex['price']
                entry_time = ex['time']
            elif (position > 0 and qty < 0) or (position < 0 and qty > 0):
                # Closing trade
                exit_price = ex['price']
                exit_time = ex['time']

                if position > 0:
                    pnl = (exit_price - entry_price) * min(abs(position), abs(qty))
                else:
                    pnl = (entry_price - exit_price) * min(abs(position), abs(qty))

                # Calculate duration
                try:
                    from datetime import datetime
                    entry_dt = datetime.fromisoformat(entry_time.replace('Z', '+00:00')) if isinstance(entry_time, str) else entry_time
                    exit_dt = datetime.fromisoformat(exit_time.replace('Z', '+00:00')) if isinstance(exit_time, str) else exit_time
                    duration = (exit_dt - entry_dt).days
                except:
                    duration = 0

                closed_trades.append({
                    'symbol': symbol,
                    'direction': 'LONG' if position > 0 else 'SHORT',
                    'quantity': min(abs(position), abs(qty)),
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'entry_time': str(entry_time),
                    'exit_time': str(exit_time),
                    'pnl': round(pnl, 2),
                    'pnl_pct': round((pnl / (entry_price * min(abs(position), abs(qty)))) * 100, 2),
                    'duration': duration
                })

                # Update position
                position += qty
                if position != 0:
                    entry_price = ex['price']
                    entry_time = ex['time']
            else:
                # Adding to position
                total_cost = position * entry_price + qty * ex['price']
                position += qty
                if position != 0:
                    entry_price = total_cost / position

    return closed_trades


def get_portfolio_data(ib: IB) -> Dict:
    """Get all portfolio data from IB"""
    data = {
        'account': {},
        'positions': [],
        'closed_trades': [],
        'executions': [],
        'total_unrealized_pnl': 0,
        'total_realized_pnl': 0,
        'daily_pnl': 0,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # Account info
    account_tags = [
        'NetLiquidation', 'AvailableFunds', 'BuyingPower',
        'TotalCashValue', 'UnrealizedPnL', 'RealizedPnL',
        'GrossPositionValue', 'InitMarginReq', 'MaintMarginReq'
    ]

    for av in ib.accountValues():
        if av.tag in account_tags and av.currency == 'USD':
            try:
                data['account'][av.tag] = float(av.value)
            except:
                pass

    # Portfolio positions
    portfolio = ib.portfolio()

    for item in portfolio:
        symbol = item.contract.symbol
        quantity = item.position
        avg_cost = item.averageCost
        market_value = item.marketValue
        unrealized_pnl = item.unrealizedPNL

        if quantity == 0:
            continue

        # Current price
        if quantity != 0:
            current_price = abs(market_value / quantity)
        else:
            current_price = avg_cost

        # P&L percentage
        cost_basis = avg_cost * abs(quantity)
        pnl_pct = (unrealized_pnl / cost_basis) * 100 if cost_basis != 0 else 0

        # Direction
        direction = "LONG" if quantity > 0 else "SHORT"

        # Strategy
        strategy = get_ticker_strategy(symbol)

        data['positions'].append({
            'symbol': symbol,
            'direction': direction,
            'strategy': strategy,
            'quantity': int(abs(quantity)),
            'avg_cost': round(avg_cost, 2),
            'current_price': round(current_price, 2),
            'market_value': round(market_value, 2),
            'unrealized_pnl': round(unrealized_pnl, 2),
            'pnl_pct': round(pnl_pct, 2),
            'daily_pnl': 0  # Will be updated if available
        })

        data['total_unrealized_pnl'] += unrealized_pnl

    # Get realized P&L from account
    data['total_realized_pnl'] = data['account'].get('RealizedPnL', 0)

    # Get daily P&L
    try:
        accounts = ib.managedAccounts()
        if accounts:
            account_id = accounts[0]
            ib.reqPnL(account_id, '')
            ib.sleep(1)
            for pnl in ib.pnl():
                if pnl.dailyPnL is not None and not math.isnan(pnl.dailyPnL):
                    data['daily_pnl'] = pnl.dailyPnL
                    break
    except Exception as e:
        logger.warning(f"Could not get daily PnL: {e}")

    # Sort positions by unrealized P&L (best first)
    data['positions'].sort(key=lambda x: x['unrealized_pnl'], reverse=True)

    # Get executions and closed trades
    try:
        executions = get_executions(ib)
        data['executions'] = executions
        data['closed_trades'] = get_closed_trades(executions)
    except Exception as e:
        logger.warning(f"Could not get closed trades: {e}")

    return data


def calculate_performance_metrics(history: Dict, current_data: Dict) -> Dict:
    """Calculate performance metrics from history"""
    metrics = {
        'initial_capital': 50000.00,  # Default, can be configured
        'current_capital': current_data['account'].get('NetLiquidation', 0),
        'profit_factor': 0,
        'sharpe_ratio': 0,
        'sortino_ratio': 0,
        'max_drawdown': 0,
        'avg_trade': 0,
        'avg_duration': 0,
        'expectancy': 0,
        'max_consec_wins': 0,
        'max_consec_losses': 0,
        'total_trades': 0,
        'win_rate': 0
    }

    equity_history = history.get('equity', [])
    trades = history.get('trades', [])

    # Calculate from equity history
    if len(equity_history) >= 2:
        equities = [e['value'] for e in equity_history]
        returns = []
        for i in range(1, len(equities)):
            if equities[i-1] != 0:
                ret = (equities[i] - equities[i-1]) / equities[i-1]
                returns.append(ret)

        if returns:
            import numpy as np
            returns = np.array(returns)

            # Sharpe Ratio (annualized, assuming daily data)
            if np.std(returns) > 0:
                metrics['sharpe_ratio'] = round(np.sqrt(252) * np.mean(returns) / np.std(returns), 2)

            # Sortino Ratio
            negative_returns = returns[returns < 0]
            if len(negative_returns) > 0 and np.std(negative_returns) > 0:
                metrics['sortino_ratio'] = round(np.sqrt(252) * np.mean(returns) / np.std(negative_returns), 2)

            # Max Drawdown
            peak = equities[0]
            max_dd = 0
            for eq in equities:
                if eq > peak:
                    peak = eq
                dd = (peak - eq) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)
            metrics['max_drawdown'] = round(max_dd * 100, 2)

        metrics['initial_capital'] = equities[0] if equities else 50000

    # Calculate from trades
    if trades:
        profits = [t['pnl'] for t in trades if t['pnl'] > 0]
        losses = [t['pnl'] for t in trades if t['pnl'] <= 0]

        total_profit = sum(profits) if profits else 0
        total_loss = abs(sum(losses)) if losses else 0

        metrics['total_trades'] = len(trades)
        metrics['win_rate'] = round(len(profits) / len(trades) * 100, 1) if trades else 0
        metrics['profit_factor'] = round(total_profit / total_loss, 2) if total_loss > 0 else total_profit
        metrics['avg_trade'] = round(sum(t['pnl'] for t in trades) / len(trades), 2) if trades else 0

        # Avg duration
        durations = [t.get('duration', 0) for t in trades if t.get('duration')]
        metrics['avg_duration'] = round(sum(durations) / len(durations), 1) if durations else 0

        # Expectancy
        win_rate = len(profits) / len(trades) if trades else 0
        avg_win = sum(profits) / len(profits) if profits else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        metrics['expectancy'] = round(win_rate * avg_win - (1 - win_rate) * abs(avg_loss), 2)

        # Consecutive wins/losses
        max_wins = max_losses = current_wins = current_losses = 0
        for t in trades:
            if t['pnl'] > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)

        metrics['max_consec_wins'] = max_wins
        metrics['max_consec_losses'] = max_losses

    return metrics


def update_history(history: Dict, data: Dict) -> Dict:
    """Update history with current data point"""
    today = datetime.now().strftime('%Y-%m-%d')

    # Update equity curve (one entry per day)
    equity_value = data['account'].get('NetLiquidation', 0)

    if history['equity']:
        last_date = history['equity'][-1].get('date', '')
        if last_date != today:
            history['equity'].append({
                'date': today,
                'value': equity_value,
                'timestamp': data['timestamp']
            })
        else:
            # Update today's entry
            history['equity'][-1] = {
                'date': today,
                'value': equity_value,
                'timestamp': data['timestamp']
            }
    else:
        history['equity'].append({
            'date': today,
            'value': equity_value,
            'timestamp': data['timestamp']
        })

    # Update daily P&L
    daily_pnl = data.get('daily_pnl', 0)
    if history['daily_pnl']:
        last_date = history['daily_pnl'][-1].get('date', '')
        if last_date != today:
            history['daily_pnl'].append({
                'date': today,
                'value': daily_pnl
            })
        else:
            history['daily_pnl'][-1] = {
                'date': today,
                'value': daily_pnl
            }
    else:
        history['daily_pnl'].append({
            'date': today,
            'value': daily_pnl
        })

    # Update trades from closed trades
    new_trades = data.get('closed_trades', [])
    existing_trade_keys = set()
    for t in history.get('trades', []):
        key = f"{t['symbol']}_{t.get('exit_time', '')}_{t['pnl']}"
        existing_trade_keys.add(key)

    for trade in new_trades:
        key = f"{trade['symbol']}_{trade.get('exit_time', '')}_{trade['pnl']}"
        if key not in existing_trade_keys:
            history['trades'].append(trade)
            existing_trade_keys.add(key)

    # Keep only last 90 days of data
    history['equity'] = history['equity'][-90:]
    history['daily_pnl'] = history['daily_pnl'][-90:]
    history['trades'] = history.get('trades', [])[-100:]  # Keep last 100 trades

    return history


def generate_html(data: Dict, history: Dict, metrics: Dict, auto_refresh: bool = False) -> str:
    """Generate HTML dashboard with charts"""

    # Prepare chart data
    equity_labels = json.dumps([e['date'] for e in history.get('equity', [])][-30:])
    equity_values = json.dumps([e['value'] for e in history.get('equity', [])][-30:])

    daily_pnl_labels = json.dumps([p['date'] for p in history.get('daily_pnl', [])][-14:])
    daily_pnl_values = json.dumps([p['value'] for p in history.get('daily_pnl', [])][-14:])

    # Position P&L for bar chart
    pos_symbols = json.dumps([p['symbol'] for p in data['positions'][:10]])
    pos_pnl = json.dumps([p['unrealized_pnl'] for p in data['positions'][:10]])
    pos_colors = json.dumps(['#00d26a' if p['unrealized_pnl'] >= 0 else '#ff4757' for p in data['positions'][:10]])

    # Positions table rows
    positions_html = ""
    for pos in data['positions']:
        pnl_class = "positive" if pos['unrealized_pnl'] >= 0 else "negative"
        daily_pnl_display = f"${pos['daily_pnl']:+,.2f}" if pos['daily_pnl'] != 0 else "-"

        positions_html += f"""
        <tr>
            <td><strong>{pos['symbol']}</strong></td>
            <td><span class="badge {pos['direction'].lower()}">{pos['direction']}</span></td>
            <td>{pos['quantity']}</td>
            <td>${pos['avg_cost']:,.2f}</td>
            <td>${pos['current_price']:,.2f}</td>
            <td>{daily_pnl_display}</td>
            <td class="{pnl_class}">${pos['unrealized_pnl']:+,.2f}</td>
            <td class="{pnl_class}">{pos['pnl_pct']:+.2f}%</td>
        </tr>
        """

    # Closed trades table rows
    closed_trades = data.get('closed_trades', []) + history.get('trades', [])
    # Remove duplicates and sort by exit time
    seen = set()
    unique_trades = []
    for t in closed_trades:
        key = f"{t['symbol']}_{t.get('exit_time', '')}_{t['pnl']}"
        if key not in seen:
            seen.add(key)
            unique_trades.append(t)
    closed_trades = sorted(unique_trades, key=lambda x: x.get('exit_time', ''), reverse=True)[:20]

    closed_trades_html = ""
    total_closed_pnl = 0
    for trade in closed_trades:
        pnl_class = "positive" if trade['pnl'] >= 0 else "negative"
        total_closed_pnl += trade['pnl']

        closed_trades_html += f"""
        <tr>
            <td><strong>{trade['symbol']}</strong></td>
            <td><span class="badge {trade['direction'].lower()}">{trade['direction']}</span></td>
            <td>{trade['quantity']}</td>
            <td>${trade['entry_price']:,.2f}</td>
            <td>${trade['exit_price']:,.2f}</td>
            <td>{trade.get('duration', 0)}d</td>
            <td class="{pnl_class}">${trade['pnl']:+,.2f}</td>
            <td class="{pnl_class}">{trade['pnl_pct']:+.2f}%</td>
        </tr>
        """

    num_closed = len(closed_trades)
    closed_pnl_class = "positive" if total_closed_pnl >= 0 else "negative"

    # Calculate totals
    net_liq = data['account'].get('NetLiquidation', 0)
    cash = data['account'].get('TotalCashValue', 0)
    unrealized = data['total_unrealized_pnl']
    realized = data['total_realized_pnl']
    daily = data['daily_pnl']
    num_positions = len(data['positions'])

    # Color classes
    daily_class = "positive" if daily >= 0 else "negative"
    unrealized_class = "positive" if unrealized >= 0 else "negative"
    realized_class = "positive" if realized >= 0 else "negative"

    # Equity change
    equity_change = 0
    if history.get('equity') and len(history['equity']) >= 2:
        equity_change = history['equity'][-1]['value'] - history['equity'][0]['value']
    equity_change_class = "positive" if equity_change >= 0 else "negative"

    refresh_meta = '<meta http-equiv="refresh" content="60">' if auto_refresh else ''

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    {refresh_meta}
    <title>Trade Monitor</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --bg-dark: #0f1419;
            --bg-card: #1a2332;
            --bg-card-alt: #212d3b;
            --text-primary: #e6e9ed;
            --text-secondary: #8899a6;
            --positive: #00d26a;
            --negative: #ff4757;
            --accent: #4da6ff;
            --border: #2d3e50;
        }}

        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: var(--bg-dark);
            color: var(--text-primary);
            min-height: 100vh;
            padding: 20px;
        }}

        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }}

        .header h1 {{
            font-size: 1.5em;
            color: var(--accent);
        }}

        .timestamp {{
            color: var(--text-secondary);
            font-size: 0.85em;
        }}

        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(6, 1fr);
            gap: 15px;
            margin-bottom: 20px;
        }}

        .summary-card {{
            background: var(--bg-card);
            border-radius: 8px;
            padding: 15px;
            text-align: center;
            border: 1px solid var(--border);
        }}

        .summary-card.highlight {{
            background: linear-gradient(135deg, #1e5a3a 0%, #1a2332 100%);
            border-color: var(--positive);
        }}

        .summary-label {{
            color: var(--text-secondary);
            font-size: 0.75em;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }}

        .summary-value {{
            font-size: 1.4em;
            font-weight: 600;
        }}

        .summary-value.positive {{ color: var(--positive); }}
        .summary-value.negative {{ color: var(--negative); }}

        .charts-row {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }}

        .chart-card {{
            background: var(--bg-card);
            border-radius: 8px;
            padding: 20px;
            border: 1px solid var(--border);
        }}

        .chart-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}

        .chart-title {{
            color: var(--accent);
            font-size: 1em;
            font-weight: 600;
        }}

        .chart-value {{
            font-size: 1.1em;
            font-weight: 600;
        }}

        .chart-container {{
            height: 200px;
        }}

        .bottom-row {{
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }}

        .metrics-card {{
            background: var(--bg-card);
            border-radius: 8px;
            padding: 20px;
            border: 1px solid var(--border);
        }}

        .metrics-title {{
            color: var(--accent);
            font-size: 1em;
            font-weight: 600;
            margin-bottom: 15px;
        }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }}

        .metric-row {{
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid var(--border);
        }}

        .metric-label {{
            color: var(--text-secondary);
            font-size: 0.85em;
        }}

        .metric-value {{
            font-weight: 600;
            font-size: 0.9em;
        }}

        .positions-card {{
            background: var(--bg-card);
            border-radius: 8px;
            padding: 20px;
            border: 1px solid var(--border);
        }}

        .positions-title {{
            color: var(--accent);
            font-size: 1em;
            font-weight: 600;
            margin-bottom: 15px;
        }}

        .positions-table {{
            width: 100%;
            border-collapse: collapse;
        }}

        .positions-table th {{
            text-align: left;
            padding: 10px;
            color: var(--text-secondary);
            font-size: 0.75em;
            text-transform: uppercase;
            letter-spacing: 1px;
            border-bottom: 1px solid var(--border);
        }}

        .positions-table td {{
            padding: 12px 10px;
            border-bottom: 1px solid var(--border);
            font-size: 0.9em;
        }}

        .positions-table tr:hover {{
            background: var(--bg-card-alt);
        }}

        .positive {{ color: var(--positive); }}
        .negative {{ color: var(--negative); }}

        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.7em;
            font-weight: 600;
            text-transform: uppercase;
        }}

        .badge.long {{
            background: rgba(0, 210, 106, 0.2);
            color: var(--positive);
        }}

        .badge.short {{
            background: rgba(255, 71, 87, 0.2);
            color: var(--negative);
        }}

        @media (max-width: 1200px) {{
            .summary-grid {{ grid-template-columns: repeat(3, 1fr); }}
            .charts-row {{ grid-template-columns: 1fr; }}
            .bottom-row {{ grid-template-columns: 1fr; }}
        }}

        @media (max-width: 768px) {{
            .summary-grid {{ grid-template-columns: repeat(2, 1fr); }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Trade Monitor</h1>
        <span class="timestamp">Last Update: {data['timestamp']}</span>
    </div>

    <div class="summary-grid">
        <div class="summary-card highlight">
            <div class="summary-label">Gesamt Kapital</div>
            <div class="summary-value">${net_liq:,.2f}</div>
        </div>
        <div class="summary-card">
            <div class="summary-label">Daily PnL</div>
            <div class="summary-value {daily_class}">${daily:+,.2f}</div>
        </div>
        <div class="summary-card">
            <div class="summary-label">Unrealized PnL</div>
            <div class="summary-value {unrealized_class}">${unrealized:+,.2f}</div>
        </div>
        <div class="summary-card">
            <div class="summary-label">Realized PnL</div>
            <div class="summary-value {realized_class}">${realized:+,.2f}</div>
        </div>
        <div class="summary-card">
            <div class="summary-label">Cash Balance</div>
            <div class="summary-value">${cash:,.2f}</div>
        </div>
        <div class="summary-card">
            <div class="summary-label">Open Positions</div>
            <div class="summary-value">{num_positions}</div>
        </div>
    </div>

    <div class="charts-row">
        <div class="chart-card">
            <div class="chart-header">
                <span class="chart-title">Kapitalkurve (R+U)</span>
                <span class="chart-value {equity_change_class}">${equity_change:+,.2f}</span>
            </div>
            <div class="chart-container">
                <canvas id="equityChart"></canvas>
            </div>
        </div>
        <div class="chart-card">
            <div class="chart-header">
                <span class="chart-title">Daily PnL</span>
                <span class="chart-value">Recording data...</span>
            </div>
            <div class="chart-container">
                <canvas id="dailyPnlChart"></canvas>
            </div>
        </div>
    </div>

    <div class="bottom-row">
        <div class="chart-card">
            <div class="chart-header">
                <span class="chart-title">Unrealized PnL</span>
                <span class="chart-value {unrealized_class}">${unrealized:+,.2f}</span>
            </div>
            <div class="chart-container">
                <canvas id="positionPnlChart"></canvas>
            </div>
        </div>
        <div class="metrics-card">
            <div class="metrics-title">Performance Metrics</div>
            <div class="metric-row">
                <span class="metric-label">Initial Capital</span>
                <span class="metric-value">${metrics['initial_capital']:,.2f}</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Current Capital</span>
                <span class="metric-value">${metrics['current_capital']:,.2f}</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Profit Factor</span>
                <span class="metric-value">{metrics['profit_factor']}</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Sharpe Ratio</span>
                <span class="metric-value">{metrics['sharpe_ratio']}</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Sortino Ratio</span>
                <span class="metric-value">{metrics['sortino_ratio']}</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Max Drawdown</span>
                <span class="metric-value">{metrics['max_drawdown']}%</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Avg Trade</span>
                <span class="metric-value">${metrics['avg_trade']}</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Avg Duration</span>
                <span class="metric-value">{metrics['avg_duration']} days</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Expectancy</span>
                <span class="metric-value">${metrics['expectancy']}</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Max Consec. Wins</span>
                <span class="metric-value">{metrics['max_consec_wins']}</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Max Consec. Losses</span>
                <span class="metric-value">{metrics['max_consec_losses']}</span>
            </div>
        </div>
    </div>

    <div class="positions-card">
        <div class="positions-title">Open Positions ({num_positions})</div>
        <table class="positions-table">
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
                {positions_html if positions_html else '<tr><td colspan="8" style="text-align:center; padding:40px; color:var(--text-secondary);">No open positions</td></tr>'}
            </tbody>
        </table>
    </div>

    <div class="positions-card" style="margin-top: 20px;">
        <div class="positions-title">Closed Trades ({num_closed}) <span class="{closed_pnl_class}" style="float:right;">Total: ${total_closed_pnl:+,.2f}</span></div>
        <table class="positions-table">
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th>Direction</th>
                    <th>Qty</th>
                    <th>Entry</th>
                    <th>Exit</th>
                    <th>Duration</th>
                    <th>P&L $</th>
                    <th>P&L %</th>
                </tr>
            </thead>
            <tbody>
                {closed_trades_html if closed_trades_html else '<tr><td colspan="8" style="text-align:center; padding:40px; color:var(--text-secondary);">No closed trades</td></tr>'}
            </tbody>
        </table>
    </div>

    <script>
        // Chart.js configuration
        Chart.defaults.color = '#8899a6';
        Chart.defaults.borderColor = '#2d3e50';

        // Equity Chart
        new Chart(document.getElementById('equityChart'), {{
            type: 'line',
            data: {{
                labels: {equity_labels},
                datasets: [{{
                    data: {equity_values},
                    borderColor: '#00d26a',
                    backgroundColor: 'rgba(0, 210, 106, 0.1)',
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0,
                    borderWidth: 2
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ display: false }} }},
                scales: {{
                    x: {{ display: false }},
                    y: {{
                        grid: {{ color: '#2d3e50' }},
                        ticks: {{ callback: v => '$' + v.toLocaleString() }}
                    }}
                }}
            }}
        }});

        // Daily P&L Chart
        const dailyPnlData = {daily_pnl_values};
        const dailyPnlColors = dailyPnlData.map(v => v >= 0 ? '#00d26a' : '#ff4757');

        new Chart(document.getElementById('dailyPnlChart'), {{
            type: 'bar',
            data: {{
                labels: {daily_pnl_labels},
                datasets: [{{
                    data: dailyPnlData,
                    backgroundColor: dailyPnlColors,
                    borderRadius: 4
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ display: false }} }},
                scales: {{
                    x: {{ display: false }},
                    y: {{
                        grid: {{ color: '#2d3e50' }},
                        ticks: {{ callback: v => '$' + v.toLocaleString() }}
                    }}
                }}
            }}
        }});

        // Position P&L Chart
        new Chart(document.getElementById('positionPnlChart'), {{
            type: 'bar',
            data: {{
                labels: {pos_symbols},
                datasets: [{{
                    data: {pos_pnl},
                    backgroundColor: {pos_colors},
                    borderRadius: 4
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: 'y',
                plugins: {{
                    legend: {{ display: false }},
                    tooltip: {{
                        callbacks: {{
                            label: ctx => '$' + ctx.raw.toLocaleString()
                        }}
                    }}
                }},
                scales: {{
                    x: {{
                        grid: {{ color: '#2d3e50' }},
                        ticks: {{ callback: v => '$' + v.toLocaleString() }}
                    }},
                    y: {{ grid: {{ display: false }} }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
    return html


def main():
    print("="*60)
    print("TWS DASHBOARD GENERATOR")
    print("="*60)

    # Parse arguments
    auto_refresh = "--live" in sys.argv
    port = IB_PORT

    for i, arg in enumerate(sys.argv):
        if arg == "--port" and i + 1 < len(sys.argv):
            try:
                port = int(sys.argv[i + 1])
            except:
                pass

    print(f"\nConnecting to TWS on port {port}...")

    # Load history
    history = load_history()

    # Connect to IB
    ib = connect_to_ib(port)

    if ib is None:
        print("\n⚠ Could not connect to TWS!")
        print("Make sure TWS is running with API enabled.")
        print("\nGenerating dashboard with cached data...")

        data = {
            'account': {'NetLiquidation': 0, 'TotalCashValue': 0},
            'positions': [],
            'total_unrealized_pnl': 0,
            'total_realized_pnl': 0,
            'daily_pnl': 0,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    else:
        print("Fetching portfolio data...")
        data = get_portfolio_data(ib)

        # Update history
        history = update_history(history, data)
        save_history(history)

        ib.disconnect()
        print("Disconnected from TWS")

    # Calculate metrics
    metrics = calculate_performance_metrics(history, data)

    # Generate HTML
    print(f"\nGenerating dashboard...")
    html = generate_html(data, history, metrics, auto_refresh)

    # Write to file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\n✓ Dashboard saved to: {OUTPUT_FILE}")
    print(f"  Positions: {len(data['positions'])}")
    print(f"  Unrealized P&L: ${data['total_unrealized_pnl']:+,.2f}")
    print(f"  Daily P&L: ${data['daily_pnl']:+,.2f}")

    if auto_refresh:
        print("\n  Auto-refresh: 60 seconds")

    # Try to open in browser
    try:
        import webbrowser
        webbrowser.open('file://' + os.path.realpath(OUTPUT_FILE))
        print("\n  Opened in browser")
    except:
        print(f"\n  Open {OUTPUT_FILE} in your browser")


if __name__ == "__main__":
    main()
