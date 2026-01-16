#!/usr/bin/env python3
"""
HTML Report Generator
=====================
Generates a comprehensive HTML report from simulation results
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
import json
import os
import webbrowser

from ib_paper_trader import (
    get_signal_for_strategy, get_ticker_strategy,
    get_stocks_by_strategy, load_stock_categories,
    get_ticker_contract_params
)

# =============================================================================
# CONFIGURATION
# =============================================================================
INITIAL_CAPITAL = 20000
MAX_POSITIONS = 30
SIMULATION_DAYS = 252
FEE_PER_TRADE = 1.0
CATEGORIES_FILE = "stock_categories.json"
OUTPUT_FILE = "simulation_report.html"


def get_all_tickers() -> List[str]:
    stocks = get_stocks_by_strategy()
    all_tickers = []
    for tickers in stocks.values():
        all_tickers.extend(tickers)
    return list(set(all_tickers))


def fetch_data_ib(symbols: List[str]) -> Dict[str, pd.DataFrame]:
    """Fetch data from IB"""
    try:
        from ib_insync import IB, Stock, util
    except ImportError:
        print("ERROR: ib_insync not installed!")
        return {}

    data = {}
    ib = IB()

    try:
        ib.connect('127.0.0.1', 7497, clientId=54)
        print(f"Connected to IB. Fetching {len(symbols)} symbols...")

        for symbol in symbols:
            try:
                exchange, currency = get_ticker_contract_params(symbol)
                contract = Stock(symbol, exchange, currency)
                ib.qualifyContracts(contract)

                bars = ib.reqHistoricalData(
                    contract,
                    endDateTime='',
                    durationStr='1 Y',
                    barSizeSetting='1 day',
                    whatToShow='TRADES',
                    useRTH=True
                )

                if bars:
                    df = util.df(bars)
                    df.columns = [c.lower() for c in df.columns]
                    df.set_index('date', inplace=True)
                    data[symbol] = df
                    print(f"  {symbol}: {len(df)} days")

                ib.sleep(0.5)
            except Exception as e:
                print(f"  {symbol}: {e}")

        ib.disconnect()
    except Exception as e:
        print(f"IB connection failed: {e}")

    return data


def calculate_supertrend(df, period=10, multiplier=2.0):
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    n = len(close)

    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))

    atr = np.zeros(n)
    atr[:period] = np.nan
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period

    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    direction = np.ones(n)
    for i in range(period, n):
        if close[i] > upper_band[i-1]:
            direction[i] = 1
        elif close[i] < lower_band[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]

    return direction


def run_simulation(data: Dict[str, pd.DataFrame]):
    """Run the simulation and return results"""
    results = {
        'trades': [],
        'equity_curve': [],
        'stock_stats': {},
        'strategy_stats': {},
        'final_stats': {}
    }

    cash = INITIAL_CAPITAL
    positions = {}
    equity_curve = []
    trades = []

    # Get common dates
    all_dates = None
    for symbol, df in data.items():
        dates = set(df.index)
        all_dates = dates if all_dates is None else all_dates.intersection(dates)

    if not all_dates:
        return results

    sorted_dates = sorted(all_dates)[-SIMULATION_DAYS:]
    ticker_strategies = {symbol: get_ticker_strategy(symbol) for symbol in data.keys()}

    # Load strategy settings
    categories = load_stock_categories()
    strategy_settings = {}
    for strat_name, strat_data in categories.get('strategies', {}).items():
        strategy_settings[strat_name] = strat_data.get('settings', {})

    for date in sorted_dates:
        date_str = str(date)[:10]
        current_prices = {}

        for symbol, df in data.items():
            if date in df.index:
                current_prices[symbol] = df.loc[date, 'close']

        # Update positions and check trailing stops
        for symbol in list(positions.keys()):
            if symbol in current_prices:
                pos = positions[symbol]
                price = current_prices[symbol]
                pos['current_price'] = price
                pos['high_price'] = max(pos['high_price'], price)

                # Check trailing stop
                strat = pos['strategy']
                settings = strategy_settings.get(strat, {})
                trailing_stop = settings.get('trailing_stop_pct', 0.12)
                stop_price = pos['high_price'] * (1 - trailing_stop)

                if price <= stop_price:
                    pnl = (price - pos['entry_price']) * pos['quantity'] - 2 * FEE_PER_TRADE
                    trades.append({
                        'symbol': symbol,
                        'strategy': strat,
                        'entry_date': pos['entry_date'],
                        'entry_price': pos['entry_price'],
                        'exit_date': date_str,
                        'exit_price': price,
                        'quantity': pos['quantity'],
                        'pnl': pnl,
                        'pnl_pct': (price / pos['entry_price'] - 1) * 100,
                        'reason': 'TRAILING_STOP'
                    })
                    cash += pos['quantity'] * price - FEE_PER_TRADE
                    del positions[symbol]

        # Check signals
        for symbol in data.keys():
            if symbol not in data or date not in data[symbol].index:
                continue

            df_slice = data[symbol].loc[:date]
            if len(df_slice) < 60:
                continue

            strat = ticker_strategies.get(symbol, "SUPERTREND")
            signal = get_signal_for_strategy(df_slice, strat)
            price = current_prices.get(symbol)

            if price is None:
                continue

            # BUY
            if signal == "BUY" and symbol not in positions and len(positions) < MAX_POSITIONS:
                equity = cash + sum(p['quantity'] * p['current_price'] for p in positions.values())
                target_value = equity / MAX_POSITIONS
                quantity = int(target_value / price)

                if quantity > 0 and cash >= quantity * price + FEE_PER_TRADE:
                    cash -= quantity * price + FEE_PER_TRADE
                    positions[symbol] = {
                        'entry_price': price,
                        'current_price': price,
                        'high_price': price,
                        'quantity': quantity,
                        'entry_date': date_str,
                        'strategy': strat
                    }

            # SELL
            elif signal == "SELL" and symbol in positions:
                pos = positions[symbol]
                pnl = (price - pos['entry_price']) * pos['quantity'] - 2 * FEE_PER_TRADE
                trades.append({
                    'symbol': symbol,
                    'strategy': strat,
                    'entry_date': pos['entry_date'],
                    'entry_price': pos['entry_price'],
                    'exit_date': date_str,
                    'exit_price': price,
                    'quantity': pos['quantity'],
                    'pnl': pnl,
                    'pnl_pct': (price / pos['entry_price'] - 1) * 100,
                    'reason': 'SIGNAL'
                })
                cash += pos['quantity'] * price - FEE_PER_TRADE
                del positions[symbol]

        # Record equity
        equity = cash + sum(p['quantity'] * p['current_price'] for p in positions.values())
        equity_curve.append({'date': date_str, 'equity': equity, 'positions': len(positions)})

    # Calculate final stats
    final_equity = equity_curve[-1]['equity'] if equity_curve else INITIAL_CAPITAL
    total_return = final_equity - INITIAL_CAPITAL
    total_return_pct = (total_return / INITIAL_CAPITAL) * 100

    # Max drawdown
    peak = INITIAL_CAPITAL
    max_dd = 0
    for e in equity_curve:
        if e['equity'] > peak:
            peak = e['equity']
        dd = (peak - e['equity']) / peak
        max_dd = max(max_dd, dd)

    # Trade stats
    closed_trades = [t for t in trades if 'exit_date' in t]
    winners = [t for t in closed_trades if t['pnl'] > 0]
    losers = [t for t in closed_trades if t['pnl'] <= 0]

    results['trades'] = trades
    results['equity_curve'] = equity_curve
    results['open_positions'] = positions
    results['final_stats'] = {
        'initial_capital': INITIAL_CAPITAL,
        'final_equity': final_equity,
        'total_return': total_return,
        'total_return_pct': total_return_pct,
        'max_drawdown_pct': max_dd * 100,
        'total_trades': len(closed_trades),
        'winners': len(winners),
        'losers': len(losers),
        'win_rate': len(winners) / len(closed_trades) * 100 if closed_trades else 0,
        'total_profit': sum(t['pnl'] for t in winners),
        'total_loss': sum(t['pnl'] for t in losers),
        'avg_win': np.mean([t['pnl'] for t in winners]) if winners else 0,
        'avg_loss': np.mean([t['pnl'] for t in losers]) if losers else 0
    }

    # Strategy stats
    for trade in closed_trades:
        strat = trade['strategy']
        if strat not in results['strategy_stats']:
            results['strategy_stats'][strat] = {'trades': 0, 'pnl': 0, 'winners': 0}
        results['strategy_stats'][strat]['trades'] += 1
        results['strategy_stats'][strat]['pnl'] += trade['pnl']
        if trade['pnl'] > 0:
            results['strategy_stats'][strat]['winners'] += 1

    # Stock stats
    for trade in closed_trades:
        symbol = trade['symbol']
        if symbol not in results['stock_stats']:
            results['stock_stats'][symbol] = {'trades': 0, 'pnl': 0, 'winners': 0, 'strategy': trade['strategy']}
        results['stock_stats'][symbol]['trades'] += 1
        results['stock_stats'][symbol]['pnl'] += trade['pnl']
        if trade['pnl'] > 0:
            results['stock_stats'][symbol]['winners'] += 1

    return results


def generate_html(results: dict) -> str:
    """Generate HTML report"""
    stats = results['final_stats']
    trades = results['trades']
    equity_curve = results['equity_curve']
    strategy_stats = results['strategy_stats']
    stock_stats = results['stock_stats']
    open_positions = results.get('open_positions', {})

    # Equity curve data for chart
    eq_dates = [e['date'] for e in equity_curve]
    eq_values = [e['equity'] for e in equity_curve]

    html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Trading Simulation Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #1a1a2e; color: #eee; }}
        h1, h2, h3 {{ color: #00d4ff; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        .card {{ background: #16213e; border-radius: 10px; padding: 20px; margin: 20px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; }}
        .stat-box {{ background: #0f3460; padding: 15px; border-radius: 8px; text-align: center; }}
        .stat-value {{ font-size: 24px; font-weight: bold; color: #00d4ff; }}
        .stat-label {{ font-size: 12px; color: #888; margin-top: 5px; }}
        .positive {{ color: #00ff88; }}
        .negative {{ color: #ff4444; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #333; }}
        th {{ background: #0f3460; color: #00d4ff; }}
        tr:hover {{ background: #1f4068; }}
        .chart-container {{ height: 400px; }}
        .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Trading Simulation Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Period: {SIMULATION_DAYS} Trading Days</p>

        <div class="card">
            <h2>Performance Summary</h2>
            <div class="stats-grid">
                <div class="stat-box">
                    <div class="stat-value">${stats['initial_capital']:,.0f}</div>
                    <div class="stat-label">Start Capital</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">${stats['final_equity']:,.0f}</div>
                    <div class="stat-label">End Capital</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value {'positive' if stats['total_return'] >= 0 else 'negative'}">${stats['total_return']:+,.0f}</div>
                    <div class="stat-label">Total P&L</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value {'positive' if stats['total_return_pct'] >= 0 else 'negative'}">{stats['total_return_pct']:+.1f}%</div>
                    <div class="stat-label">Return %</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{stats['total_trades']}</div>
                    <div class="stat-label">Total Trades</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{stats['win_rate']:.1f}%</div>
                    <div class="stat-label">Win Rate</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value negative">{stats['max_drawdown_pct']:.1f}%</div>
                    <div class="stat-label">Max Drawdown</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{stats['winners']}/{stats['losers']}</div>
                    <div class="stat-label">Winners/Losers</div>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>Equity Curve</h2>
            <div class="chart-container">
                <canvas id="equityChart"></canvas>
            </div>
        </div>

        <div class="two-col">
            <div class="card">
                <h2>Strategy Performance</h2>
                <table>
                    <tr><th>Strategy</th><th>Trades</th><th>Win Rate</th><th>P&L</th></tr>
'''

    for strat, data in sorted(strategy_stats.items(), key=lambda x: x[1]['pnl'], reverse=True):
        win_rate = data['winners'] / data['trades'] * 100 if data['trades'] > 0 else 0
        pnl_class = 'positive' if data['pnl'] >= 0 else 'negative'
        html += f'''                    <tr>
                        <td>{strat}</td>
                        <td>{data['trades']}</td>
                        <td>{win_rate:.1f}%</td>
                        <td class="{pnl_class}">${data['pnl']:+,.0f}</td>
                    </tr>
'''

    html += '''                </table>
            </div>

            <div class="card">
                <h2>Top 10 Stocks</h2>
                <table>
                    <tr><th>Symbol</th><th>Strategy</th><th>Trades</th><th>P&L</th></tr>
'''

    sorted_stocks = sorted(stock_stats.items(), key=lambda x: x[1]['pnl'], reverse=True)[:10]
    for symbol, data in sorted_stocks:
        pnl_class = 'positive' if data['pnl'] >= 0 else 'negative'
        html += f'''                    <tr>
                        <td>{symbol}</td>
                        <td>{data['strategy']}</td>
                        <td>{data['trades']}</td>
                        <td class="{pnl_class}">${data['pnl']:+,.0f}</td>
                    </tr>
'''

    html += '''                </table>
            </div>
        </div>

        <div class="card">
            <h2>Open Positions</h2>
            <table>
                <tr><th>Symbol</th><th>Strategy</th><th>Entry Date</th><th>Entry Price</th><th>Current</th><th>Qty</th><th>Unrealized P&L</th></tr>
'''

    for symbol, pos in sorted(open_positions.items(), key=lambda x: (x[1]['current_price'] - x[1]['entry_price']) * x[1]['quantity'], reverse=True):
        unrealized = (pos['current_price'] - pos['entry_price']) * pos['quantity']
        pnl_class = 'positive' if unrealized >= 0 else 'negative'
        html += f'''                <tr>
                    <td>{symbol}</td>
                    <td>{pos['strategy']}</td>
                    <td>{pos['entry_date']}</td>
                    <td>${pos['entry_price']:.2f}</td>
                    <td>${pos['current_price']:.2f}</td>
                    <td>{pos['quantity']}</td>
                    <td class="{pnl_class}">${unrealized:+,.0f}</td>
                </tr>
'''

    html += '''            </table>
        </div>

        <div class="card">
            <h2>Trade History (Last 50)</h2>
            <table>
                <tr><th>Symbol</th><th>Strategy</th><th>Entry</th><th>Exit</th><th>Entry $</th><th>Exit $</th><th>P&L</th><th>%</th><th>Reason</th></tr>
'''

    for trade in sorted(trades, key=lambda x: x.get('exit_date', ''), reverse=True)[:50]:
        if 'exit_date' not in trade:
            continue
        pnl_class = 'positive' if trade['pnl'] >= 0 else 'negative'
        html += f'''                <tr>
                    <td>{trade['symbol']}</td>
                    <td>{trade['strategy']}</td>
                    <td>{trade['entry_date']}</td>
                    <td>{trade['exit_date']}</td>
                    <td>${trade['entry_price']:.2f}</td>
                    <td>${trade['exit_price']:.2f}</td>
                    <td class="{pnl_class}">${trade['pnl']:+,.0f}</td>
                    <td class="{pnl_class}">{trade['pnl_pct']:+.1f}%</td>
                    <td>{trade['reason']}</td>
                </tr>
'''

    html += f'''            </table>
        </div>
    </div>

    <script>
        const ctx = document.getElementById('equityChart').getContext('2d');
        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: {eq_dates},
                datasets: [{{
                    label: 'Equity',
                    data: {eq_values},
                    borderColor: '#00d4ff',
                    backgroundColor: 'rgba(0, 212, 255, 0.1)',
                    fill: true,
                    tension: 0.1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        beginAtZero: false,
                        grid: {{ color: '#333' }},
                        ticks: {{ color: '#888' }}
                    }},
                    x: {{
                        grid: {{ color: '#333' }},
                        ticks: {{ color: '#888', maxTicksLimit: 12 }}
                    }}
                }},
                plugins: {{
                    legend: {{ labels: {{ color: '#eee' }} }}
                }}
            }}
        }});
    </script>
</body>
</html>'''

    return html


def main():
    print("="*80)
    print("         HTML REPORT GENERATOR")
    print("="*80)

    tickers = get_all_tickers()
    print(f"\nLoading data for {len(tickers)} stocks...")

    data = fetch_data_ib(tickers)

    if len(data) < 10:
        print("\nERROR: Konnte keine IB Daten laden!")
        print("Bitte TWS starten und API aktivieren (Port 7497)")
        return

    print(f"\nLoaded {len(data)} symbols. Running simulation...")

    results = run_simulation(data)

    print("\nGenerating HTML report...")
    html = generate_html(results)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\nReport saved to: {OUTPUT_FILE}")
    print("Opening in browser...")

    # Open in browser
    webbrowser.open('file://' + os.path.realpath(OUTPUT_FILE))

    print("\nDone!")


if __name__ == "__main__":
    main()
