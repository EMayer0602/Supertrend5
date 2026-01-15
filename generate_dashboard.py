#!/usr/bin/env python3
"""
TWS Dashboard Generator - Mit FlexQuery und korrekter Kapitalberechnung
========================================================================

KAPITALBERECHNUNG:
    Startkapital = 50000
    Kapital_pro_Trade = Kapital / 30
    Amount = round(Kapital_pro_Trade / entry_price)

    Beim Kauf:      PnL = -entry_fee
    Jeden Tag:      PnL = amount * (current_price - entry_price)  [LONG]
                    PnL = amount * (entry_price - current_price)  [SHORT]
    Beim Verkauf:   PnL = amount * (exit_price - entry_price) - exit_fee

    Kapital[t] = Kapital[t-1] + Σ(PnL aller Positionen)

FLEXQUERY:
    Historische Trades werden von IB FlexQuery API geholt.
    Aktivierung: Account Management -> Reports -> Flex Queries

Usage:
    python generate_dashboard.py                    # Generate dashboard
    python generate_dashboard.py --live             # Auto-refresh every 30s
    python generate_dashboard.py --port 7496        # Use live trading port
    python generate_dashboard.py --setup-flex       # Setup FlexQuery
"""

import json
import sys
import os
import math
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
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

# =============================================================================
# CONFIGURATION
# =============================================================================
IB_HOST = "127.0.0.1"
IB_PORT = 7497  # Paper: 7497, Live: 7496
IB_CLIENT_ID = 99

# Kapital Konfiguration
INITIAL_CAPITAL = 50000.00
NUM_POSITIONS = 30
ENTRY_FEE = 1.00  # USD pro Trade
EXIT_FEE = 1.00   # USD pro Trade

# FlexQuery Konfiguration (wird aus config geladen)
FLEX_TOKEN = ""  # Dein FlexQuery Token
FLEX_QUERY_ID = ""  # Deine Query ID

# Files
CONFIG_FILE = "dashboard_config.json"
CATEGORIES_FILE = "stock_categories.json"
OUTPUT_FILE = "dashboard.html"
HISTORY_FILE = "portfolio_history.json"

# FlexQuery API URLs
FLEX_REQUEST_URL = "https://gdcdyn.interactivebrokers.com/Universal/servlet/FlexStatementService.SendRequest"
FLEX_STATEMENT_URL = "https://gdcdyn.interactivebrokers.com/Universal/servlet/FlexStatementService.GetStatement"


# =============================================================================
# CONFIG MANAGEMENT
# =============================================================================
def load_config() -> Dict:
    """Load dashboard configuration"""
    default_config = {
        'initial_capital': INITIAL_CAPITAL,
        'num_positions': NUM_POSITIONS,
        'entry_fee': ENTRY_FEE,
        'exit_fee': EXIT_FEE,
        'flex_token': '',
        'flex_query_id': '',
        'ib_port': IB_PORT
    }
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
            # Merge with defaults
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value
            return config
    except FileNotFoundError:
        save_config(default_config)
        return default_config


def save_config(config: Dict):
    """Save dashboard configuration"""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)


def load_history() -> Dict:
    """Load portfolio history"""
    try:
        with open(HISTORY_FILE, 'r') as f:
            data = json.load(f)
            for key in ['equity', 'daily_pnl', 'trades', 'open_positions']:
                if key not in data:
                    data[key] = []
            return data
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            'equity': [],
            'daily_pnl': [],
            'trades': [],
            'open_positions': [],
            'initial_capital': INITIAL_CAPITAL
        }


def save_history(history: Dict):
    """Save portfolio history"""
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)


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


# =============================================================================
# FLEXQUERY API
# =============================================================================
def setup_flexquery():
    """Interactive setup for FlexQuery"""
    print("\n" + "="*60)
    print("FLEXQUERY SETUP")
    print("="*60)
    print("""
Um FlexQuery zu aktivieren:

1. Gehe zu: https://www.interactivebrokers.com/sso/Login
2. Login -> Account Management
3. Reports -> Flex Queries -> Create
4. Erstelle eine neue "Trade" Query mit diesen Feldern:
   - Trade Date/Time
   - Symbol
   - Buy/Sell
   - Quantity
   - Price
   - Commission
   - Realized P&L
   - Account ID

5. Speichere und notiere die "Query ID"
6. Gehe zu: Settings -> Flex Web Service
7. Aktiviere es und kopiere den "Token"
""")

    config = load_config()

    token = input("\nFlexQuery Token eingeben (oder Enter zum Überspringen): ").strip()
    if token:
        config['flex_token'] = token

    query_id = input("FlexQuery ID eingeben (oder Enter zum Überspringen): ").strip()
    if query_id:
        config['flex_query_id'] = query_id

    if token and query_id:
        save_config(config)
        print("\n✓ FlexQuery Konfiguration gespeichert!")
        print(f"  Token: {token[:10]}...")
        print(f"  Query ID: {query_id}")

        # Test connection
        print("\nTeste Verbindung...")
        trades = fetch_flexquery_trades(token, query_id)
        if trades:
            print(f"✓ Erfolgreich! {len(trades)} Trades gefunden.")
        else:
            print("⚠ Keine Trades gefunden oder Fehler. Prüfe Token/Query ID.")
    else:
        print("\n⚠ Setup unvollständig. Führe --setup-flex erneut aus.")


def fetch_flexquery_trades(token: str, query_id: str, max_retries: int = 3) -> List[Dict]:
    """Fetch trades from IB FlexQuery API"""
    if not token or not query_id:
        logger.warning("FlexQuery Token oder Query ID nicht konfiguriert")
        return []

    trades = []

    try:
        # Step 1: Request the report
        logger.info("Requesting FlexQuery report...")
        request_params = {
            't': token,
            'q': query_id,
            'v': '3'
        }

        response = requests.get(FLEX_REQUEST_URL, params=request_params, timeout=30)

        if response.status_code != 200:
            logger.error(f"FlexQuery request failed: {response.status_code}")
            return []

        # Parse response to get reference code
        root = ET.fromstring(response.text)

        status = root.find('.//Status')
        if status is not None and status.text != 'Success':
            error = root.find('.//ErrorMessage')
            error_msg = error.text if error is not None else 'Unknown error'
            logger.error(f"FlexQuery error: {error_msg}")
            return []

        ref_code = root.find('.//ReferenceCode')
        if ref_code is None:
            logger.error("No reference code in response")
            return []

        reference_code = ref_code.text
        logger.info(f"Got reference code: {reference_code}")

        # Step 2: Wait and fetch the statement
        import time
        for attempt in range(max_retries):
            time.sleep(2)  # Wait before fetching

            statement_params = {
                't': token,
                'q': reference_code,
                'v': '3'
            }

            statement_response = requests.get(FLEX_STATEMENT_URL, params=statement_params, timeout=60)

            if statement_response.status_code == 200:
                # Check if still processing
                if 'Statement generation in progress' in statement_response.text:
                    logger.info(f"Statement still processing, attempt {attempt + 1}/{max_retries}")
                    continue

                # Parse trades from XML
                trades = parse_flexquery_trades(statement_response.text)
                break

        logger.info(f"Fetched {len(trades)} trades from FlexQuery")

    except requests.exceptions.RequestException as e:
        logger.error(f"FlexQuery request error: {e}")
    except ET.ParseError as e:
        logger.error(f"FlexQuery XML parse error: {e}")
    except Exception as e:
        logger.error(f"FlexQuery error: {e}")

    return trades


def parse_flexquery_trades(xml_text: str) -> List[Dict]:
    """Parse trades from FlexQuery XML response"""
    trades = []

    try:
        root = ET.fromstring(xml_text)

        # Find all Trade elements
        for trade in root.findall('.//Trade'):
            try:
                symbol = trade.get('symbol', '')
                if not symbol:
                    continue

                # Try different attribute names for time
                trade_time = (trade.get('tradeTime') or
                             trade.get('tradeTimestamp') or
                             trade.get('dateTime', '')[-8:] if trade.get('dateTime') else '' or
                             trade.get('executionTime') or
                             trade.get('time') or '')

                # Parse trade data
                trade_data = {
                    'symbol': symbol,
                    'trade_date': trade.get('tradeDate', ''),
                    'trade_time': trade_time,
                    'datetime': f"{trade.get('tradeDate', '')} {trade_time}",
                    'side': trade.get('buySell', ''),  # BUY or SELL
                    'quantity': abs(int(float(trade.get('quantity', 0)))),
                    'price': float(trade.get('tradePrice', 0)),
                    'commission': abs(float(trade.get('ibCommission', 0))),
                    'realized_pnl': float(trade.get('fifoPnlRealized', 0)),
                    'currency': trade.get('currency', 'USD'),
                    'account': trade.get('accountId', ''),
                    'order_id': trade.get('ibOrderID', ''),
                    'exec_id': trade.get('ibExecID', '')
                }

                # Debug: print all attributes of first trade
                if len(trades) == 0:
                    logger.info(f"Trade attributes: {list(trade.attrib.keys())}")

                trades.append(trade_data)

            except (ValueError, TypeError) as e:
                logger.warning(f"Could not parse trade: {e}")
                continue

        # Sort by datetime
        trades.sort(key=lambda x: x.get('datetime', ''))

        # Debug: show first trade to verify data
        if trades:
            first = trades[0]
            logger.info(f"Sample trade: {first['symbol']} date={first['trade_date']} time={first['trade_time']}")

    except ET.ParseError as e:
        logger.error(f"XML parse error: {e}")

    return trades


def process_trades_to_positions(trades: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Process raw trades into open positions and closed trades
    Returns: (open_positions, closed_trades)
    """
    from collections import defaultdict

    # Group trades by symbol
    trades_by_symbol = defaultdict(list)
    for trade in trades:
        trades_by_symbol[trade['symbol']].append(trade)

    open_positions = []
    closed_trades = []

    for symbol, symbol_trades in trades_by_symbol.items():
        # Sort by datetime
        symbol_trades.sort(key=lambda x: x.get('datetime', ''))

        position = 0
        entry_price = 0
        entry_date = None
        entry_time = None
        entry_commission = 0

        for trade in symbol_trades:
            qty = trade['quantity']
            if trade['side'] == 'SELL':
                qty = -qty

            if position == 0:
                # Opening position
                position = qty
                entry_price = trade['price']
                entry_date = trade['trade_date']
                entry_time = trade['trade_time']
                entry_commission = trade['commission']

            elif (position > 0 and qty < 0) or (position < 0 and qty > 0):
                # Closing or reducing position
                close_qty = min(abs(position), abs(qty))
                exit_price = trade['price']
                exit_date = trade['trade_date']
                exit_time = trade['trade_time']
                exit_commission = trade['commission']

                # Calculate P&L
                if position > 0:  # Long position closed
                    pnl = close_qty * (exit_price - entry_price) - entry_commission - exit_commission
                else:  # Short position closed
                    pnl = close_qty * (entry_price - exit_price) - entry_commission - exit_commission

                pnl_pct = (pnl / (entry_price * close_qty)) * 100 if entry_price * close_qty != 0 else 0

                # Calculate duration
                try:
                    entry_dt = datetime.strptime(entry_date, '%Y%m%d')
                    exit_dt = datetime.strptime(exit_date, '%Y%m%d')
                    duration = (exit_dt - entry_dt).days
                except:
                    duration = 0

                closed_trades.append({
                    'symbol': symbol,
                    'direction': 'LONG' if position > 0 else 'SHORT',
                    'quantity': close_qty,
                    'entry_price': round(entry_price, 2),
                    'exit_price': round(exit_price, 2),
                    'entry_date': entry_date,
                    'entry_time': entry_time,
                    'exit_date': exit_date,
                    'exit_time': exit_time,
                    'pnl': round(pnl, 2),
                    'pnl_pct': round(pnl_pct, 2),
                    'duration': duration,
                    'entry_fee': entry_commission,
                    'exit_fee': exit_commission
                })

                # Update position
                position += qty
                if position != 0:
                    # Still have remaining position
                    entry_price = trade['price']
                    entry_date = trade['trade_date']
                    entry_time = trade['trade_time']
                    entry_commission = trade['commission']
            else:
                # Adding to position - average price
                total_value = position * entry_price + qty * trade['price']
                position += qty
                if position != 0:
                    entry_price = total_value / position
                entry_commission += trade['commission']

        # Remaining position is open
        if position != 0:
            open_positions.append({
                'symbol': symbol,
                'direction': 'LONG' if position > 0 else 'SHORT',
                'quantity': abs(position),
                'entry_price': round(entry_price, 2),
                'entry_date': entry_date,
                'entry_time': entry_time,
                'entry_fee': entry_commission,
                'current_price': 0,  # Will be updated from TWS
                'unrealized_pnl': 0,
                'pnl_pct': 0
            })

    return open_positions, closed_trades


# =============================================================================
# TWS CONNECTION
# =============================================================================
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


def get_current_prices(ib: IB, symbols: List[str]) -> Dict[str, float]:
    """Get current prices for symbols from TWS"""
    prices = {}

    for symbol in symbols:
        try:
            contract = Stock(symbol, 'SMART', 'USD')
            ib.qualifyContracts(contract)

            ticker = ib.reqMktData(contract, '', False, False)
            ib.sleep(1)

            if ticker.last and not math.isnan(ticker.last):
                prices[symbol] = ticker.last
            elif ticker.close and not math.isnan(ticker.close):
                prices[symbol] = ticker.close

            ib.cancelMktData(contract)

        except Exception as e:
            logger.warning(f"Could not get price for {symbol}: {e}")

    return prices


def get_portfolio_from_tws(ib: IB) -> Dict:
    """Get portfolio data directly from TWS"""
    data = {
        'account': {},
        'positions': [],
        'total_unrealized_pnl': 0,
        'total_realized_pnl': 0,
        'daily_pnl': 0,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # Account values
    for av in ib.accountValues():
        if av.currency in ['USD', 'BASE', '']:
            try:
                data['account'][av.tag] = float(av.value)
            except:
                pass

    data['total_unrealized_pnl'] = data['account'].get('UnrealizedPnL', 0)
    data['total_realized_pnl'] = data['account'].get('RealizedPnL', 0)

    # Portfolio positions
    for item in ib.portfolio():
        if item.position == 0:
            continue

        symbol = item.contract.symbol
        quantity = item.position
        avg_cost = item.averageCost
        market_value = item.marketValue
        unrealized_pnl = item.unrealizedPNL

        current_price = abs(market_value / quantity) if quantity != 0 else avg_cost
        cost_basis = abs(avg_cost * quantity)
        pnl_pct = (unrealized_pnl / cost_basis) * 100 if cost_basis != 0 else 0

        data['positions'].append({
            'symbol': symbol,
            'direction': 'LONG' if quantity > 0 else 'SHORT',
            'quantity': int(abs(quantity)),
            'entry_price': round(avg_cost, 2),
            'current_price': round(current_price, 2),
            'market_value': round(market_value, 2),
            'unrealized_pnl': round(unrealized_pnl, 2),
            'pnl_pct': round(pnl_pct, 2),
            'strategy': get_ticker_strategy(symbol)
        })

    # Daily PnL
    try:
        accounts = ib.managedAccounts()
        if accounts:
            ib.reqPnL(accounts[0], '')
            ib.sleep(2)
            for pnl in ib.pnl():
                if pnl.dailyPnL and not math.isnan(pnl.dailyPnL):
                    data['daily_pnl'] = round(pnl.dailyPnL, 2)
                    break
            ib.cancelPnL(accounts[0])
    except:
        pass

    data['positions'].sort(key=lambda x: x['unrealized_pnl'], reverse=True)

    return data


# =============================================================================
# KAPITALBERECHNUNG
# =============================================================================
def calculate_equity_curve(history: Dict, current_positions: List[Dict],
                          closed_trades: List[Dict], config: Dict) -> List[Dict]:
    """
    Berechne Kapitalkurve basierend auf Trades

    Formel:
        Startkapital = config['initial_capital']
        Beim Kauf:      PnL = -entry_fee
        Jeden Tag:      PnL = amount * (current_price - entry_price)  [LONG]
        Beim Verkauf:   PnL = amount * (exit_price - entry_price) - exit_fee
        Kapital[t] = Kapital[t-1] + Σ(PnL)
    """
    initial_capital = config.get('initial_capital', INITIAL_CAPITAL)
    entry_fee = config.get('entry_fee', ENTRY_FEE)
    exit_fee = config.get('exit_fee', EXIT_FEE)

    # Sammle alle Events (Einstieg, Ausstieg) mit Datum
    events = []

    # Geschlossene Trades
    for trade in closed_trades:
        # Entry event
        entry_date = trade.get('entry_date', '')
        if len(entry_date) == 8:  # YYYYMMDD format
            entry_date = f"{entry_date[:4]}-{entry_date[4:6]}-{entry_date[6:8]}"

        events.append({
            'date': entry_date,
            'type': 'ENTRY',
            'symbol': trade['symbol'],
            'direction': trade['direction'],
            'quantity': trade['quantity'],
            'price': trade['entry_price'],
            'fee': trade.get('entry_fee', entry_fee)
        })

        # Exit event
        exit_date = trade.get('exit_date', '')
        if len(exit_date) == 8:
            exit_date = f"{exit_date[:4]}-{exit_date[4:6]}-{exit_date[6:8]}"

        events.append({
            'date': exit_date,
            'type': 'EXIT',
            'symbol': trade['symbol'],
            'direction': trade['direction'],
            'quantity': trade['quantity'],
            'entry_price': trade['entry_price'],
            'exit_price': trade['exit_price'],
            'pnl': trade['pnl'],
            'fee': trade.get('exit_fee', exit_fee)
        })

    # Offene Positionen (nur Entry)
    for pos in current_positions:
        entry_date = pos.get('entry_date', '')
        if len(entry_date) == 8:
            entry_date = f"{entry_date[:4]}-{entry_date[4:6]}-{entry_date[6:8]}"

        events.append({
            'date': entry_date,
            'type': 'ENTRY',
            'symbol': pos['symbol'],
            'direction': pos['direction'],
            'quantity': pos['quantity'],
            'price': pos['entry_price'],
            'fee': pos.get('entry_fee', entry_fee)
        })

    # Sortiere Events nach Datum
    events.sort(key=lambda x: x.get('date', ''))

    # Berechne Kapitalkurve
    equity_curve = []
    capital = initial_capital

    # Gruppiere Events nach Datum
    from collections import defaultdict
    events_by_date = defaultdict(list)
    for event in events:
        if event.get('date'):
            events_by_date[event['date']].append(event)

    # Verarbeite jeden Tag
    for date in sorted(events_by_date.keys()):
        day_pnl = 0

        for event in events_by_date[date]:
            if event['type'] == 'ENTRY':
                # Beim Einstieg: nur Fee abziehen
                day_pnl -= event['fee']
            elif event['type'] == 'EXIT':
                # Beim Ausstieg: realisierter P&L
                day_pnl += event['pnl']

        capital += day_pnl

        equity_curve.append({
            'date': date,
            'value': round(capital, 2),
            'daily_pnl': round(day_pnl, 2)
        })

    # Füge heute hinzu mit unrealized P&L
    today = datetime.now().strftime('%Y-%m-%d')
    if equity_curve and equity_curve[-1]['date'] != today:
        # Berechne aktuellen unrealized P&L
        unrealized = sum(p.get('unrealized_pnl', 0) for p in current_positions)
        current_capital = capital + unrealized

        equity_curve.append({
            'date': today,
            'value': round(current_capital, 2),
            'daily_pnl': round(unrealized, 2)
        })

    return equity_curve


def calculate_performance_metrics(equity_curve: List[Dict], closed_trades: List[Dict],
                                  config: Dict) -> Dict:
    """Berechne Performance Metriken"""
    initial_capital = config.get('initial_capital', INITIAL_CAPITAL)
    current_capital = equity_curve[-1]['value'] if equity_curve else initial_capital

    metrics = {
        'initial_capital': initial_capital,
        'current_capital': current_capital,
        'total_return': current_capital - initial_capital,
        'total_return_pct': ((current_capital - initial_capital) / initial_capital) * 100 if initial_capital > 0 else 0,
        'profit_factor': 0,
        'sharpe_ratio': 0,
        'sortino_ratio': 0,
        'max_drawdown': 0,
        'max_drawdown_pct': 0,
        'win_rate': 0,
        'avg_trade': 0,
        'avg_winner': 0,
        'avg_loser': 0,
        'total_trades': len(closed_trades),
        'winning_trades': 0,
        'losing_trades': 0,
        'expectancy': 0
    }

    # Trade-basierte Metriken
    if closed_trades:
        winners = [t for t in closed_trades if t.get('pnl', 0) > 0]
        losers = [t for t in closed_trades if t.get('pnl', 0) <= 0]

        metrics['winning_trades'] = len(winners)
        metrics['losing_trades'] = len(losers)
        metrics['win_rate'] = round(len(winners) / len(closed_trades) * 100, 1)

        total_profit = sum(t['pnl'] for t in winners) if winners else 0
        total_loss = abs(sum(t['pnl'] for t in losers)) if losers else 0

        metrics['profit_factor'] = round(total_profit / total_loss, 2) if total_loss > 0 else total_profit
        metrics['avg_trade'] = round(sum(t['pnl'] for t in closed_trades) / len(closed_trades), 2)
        metrics['avg_winner'] = round(total_profit / len(winners), 2) if winners else 0
        metrics['avg_loser'] = round(-total_loss / len(losers), 2) if losers else 0

        # Expectancy
        win_rate = len(winners) / len(closed_trades)
        metrics['expectancy'] = round(
            win_rate * metrics['avg_winner'] + (1 - win_rate) * metrics['avg_loser'], 2
        )

    # Equity-basierte Metriken
    if len(equity_curve) >= 2:
        values = [e['value'] for e in equity_curve]

        # Daily returns
        returns = []
        for i in range(1, len(values)):
            if values[i-1] != 0:
                ret = (values[i] - values[i-1]) / values[i-1]
                returns.append(ret)

        if returns:
            import numpy as np
            returns = np.array(returns)

            # Sharpe Ratio (annualized)
            if np.std(returns) > 0:
                metrics['sharpe_ratio'] = round(np.sqrt(252) * np.mean(returns) / np.std(returns), 2)

            # Sortino Ratio
            neg_returns = returns[returns < 0]
            if len(neg_returns) > 0 and np.std(neg_returns) > 0:
                metrics['sortino_ratio'] = round(np.sqrt(252) * np.mean(returns) / np.std(neg_returns), 2)

        # Max Drawdown
        peak = values[0]
        max_dd = 0
        max_dd_pct = 0
        for val in values:
            if val > peak:
                peak = val
            dd = peak - val
            dd_pct = dd / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
                max_dd_pct = dd_pct

        metrics['max_drawdown'] = round(max_dd, 2)
        metrics['max_drawdown_pct'] = round(max_dd_pct * 100, 2)

    return metrics


# =============================================================================
# HTML GENERATION
# =============================================================================
def generate_html(data: Dict, equity_curve: List[Dict], closed_trades: List[Dict],
                  metrics: Dict, config: Dict, auto_refresh: bool = False) -> str:
    """Generate HTML dashboard"""

    # Chart data
    equity_labels = json.dumps([e['date'][-5:] for e in equity_curve[-30:]])
    equity_values = json.dumps([e['value'] for e in equity_curve[-30:]])

    daily_pnl_labels = json.dumps([e['date'][-5:] for e in equity_curve[-14:]])
    daily_pnl_values = json.dumps([e.get('daily_pnl', 0) for e in equity_curve[-14:]])

    # Position chart data
    positions = data.get('positions', [])
    pos_symbols = json.dumps([p['symbol'] for p in positions[:10]])
    pos_pnl = json.dumps([p['unrealized_pnl'] for p in positions[:10]])
    pos_colors = json.dumps(['#00d26a' if p['unrealized_pnl'] >= 0 else '#ff4757' for p in positions[:10]])

    # Positions table
    positions_html = ""
    for pos in positions:
        pnl_class = "positive" if pos['unrealized_pnl'] >= 0 else "negative"
        positions_html += f"""
        <tr>
            <td><strong>{pos['symbol']}</strong></td>
            <td><span class="badge {pos['direction'].lower()}">{pos['direction']}</span></td>
            <td>{pos['quantity']}</td>
            <td>${pos['entry_price']:,.2f}</td>
            <td>${pos['current_price']:,.2f}</td>
            <td>${pos.get('market_value', 0):,.2f}</td>
            <td class="{pnl_class}">${pos['unrealized_pnl']:+,.2f}</td>
            <td class="{pnl_class}">{pos['pnl_pct']:+.2f}%</td>
        </tr>
        """

    # Closed trades table
    closed_trades_sorted = sorted(closed_trades, key=lambda x: x.get('exit_date', ''), reverse=True)[:20]
    closed_trades_html = ""
    total_closed_pnl = sum(t.get('pnl', 0) for t in closed_trades)

    for trade in closed_trades_sorted:
        pnl_class = "positive" if trade.get('pnl', 0) >= 0 else "negative"

        # Entry date/time formatting
        entry_date = trade.get('entry_date', 'N/A')
        entry_time = trade.get('entry_time', '')
        # Handle YYYYMMDD format
        if len(entry_date) == 8 and entry_date.isdigit():
            entry_date_fmt = f"{entry_date[4:6]}-{entry_date[6:8]}"
        # Handle YYYY-MM-DD format
        elif len(entry_date) == 10 and '-' in entry_date:
            entry_date_fmt = entry_date[5:]  # MM-DD
        else:
            entry_date_fmt = entry_date
        # Add time if available
        if entry_time and len(entry_time) >= 5:
            entry_datetime = f"{entry_date_fmt} {entry_time[:5]}"
        elif entry_time and len(entry_time) >= 4:
            entry_datetime = f"{entry_date_fmt} {entry_time[:2]}:{entry_time[2:4]}"
        else:
            entry_datetime = entry_date_fmt

        # Exit date/time formatting
        exit_date = trade.get('exit_date', 'N/A')
        exit_time = trade.get('exit_time', '')
        # Handle YYYYMMDD format
        if len(exit_date) == 8 and exit_date.isdigit():
            exit_date_fmt = f"{exit_date[4:6]}-{exit_date[6:8]}"
        # Handle YYYY-MM-DD format
        elif len(exit_date) == 10 and '-' in exit_date:
            exit_date_fmt = exit_date[5:]  # MM-DD
        else:
            exit_date_fmt = exit_date
        # Add time if available
        if exit_time and len(exit_time) >= 5:
            exit_datetime = f"{exit_date_fmt} {exit_time[:5]}"
        elif exit_time and len(exit_time) >= 4:
            exit_datetime = f"{exit_date_fmt} {exit_time[:2]}:{exit_time[2:4]}"
        else:
            exit_datetime = exit_date_fmt

        closed_trades_html += f"""
        <tr>
            <td><strong>{trade.get('symbol', 'N/A')}</strong></td>
            <td><span class="badge {trade.get('direction', 'LONG').lower()}">{trade.get('direction', 'LONG')}</span></td>
            <td>{trade.get('quantity', 0)}</td>
            <td>{entry_datetime}</td>
            <td>${trade.get('entry_price', 0):,.2f}</td>
            <td>{exit_datetime}</td>
            <td>${trade.get('exit_price', 0):,.2f}</td>
            <td>{trade.get('duration', 0)}d</td>
            <td class="{pnl_class}">${trade.get('pnl', 0):+,.2f}</td>
            <td class="{pnl_class}">{trade.get('pnl_pct', 0):+.2f}%</td>
        </tr>
        """

    # Summary values
    net_liq = data['account'].get('NetLiquidation', metrics['current_capital'])
    cash = data['account'].get('TotalCashValue', 0)
    unrealized = data.get('total_unrealized_pnl', sum(p['unrealized_pnl'] for p in positions))
    realized = data.get('total_realized_pnl', total_closed_pnl)
    daily = data.get('daily_pnl', 0)
    num_positions = len(positions)

    # Classes
    daily_class = "positive" if daily >= 0 else "negative"
    unrealized_class = "positive" if unrealized >= 0 else "negative"
    realized_class = "positive" if realized >= 0 else "negative"
    total_return_class = "positive" if metrics['total_return'] >= 0 else "negative"
    closed_pnl_class = "positive" if total_closed_pnl >= 0 else "negative"

    # Metric classes
    winrate_class = "positive" if metrics['win_rate'] >= 50 else "negative"
    pf_class = "positive" if metrics['profit_factor'] >= 1 else "negative"
    sharpe_class = "positive" if metrics['sharpe_ratio'] >= 0 else "negative"
    expectancy_class = "positive" if metrics['expectancy'] >= 0 else "negative"

    refresh_meta = '<meta http-equiv="refresh" content="30">' if auto_refresh else ''

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
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
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
        .header h1 {{ font-size: 1.5em; color: var(--accent); }}
        .timestamp {{ color: var(--text-secondary); font-size: 0.85em; }}
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
        .summary-value {{ font-size: 1.4em; font-weight: 600; }}
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
        .chart-title {{ color: var(--accent); font-size: 1em; font-weight: 600; }}
        .chart-value {{ font-size: 1.1em; font-weight: 600; }}
        .chart-container {{ height: 200px; }}
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
        .metric-row {{
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid var(--border);
        }}
        .metric-label {{ color: var(--text-secondary); font-size: 0.85em; }}
        .metric-value {{ font-weight: 600; font-size: 0.9em; }}
        .positions-card {{
            background: var(--bg-card);
            border-radius: 8px;
            padding: 20px;
            border: 1px solid var(--border);
            margin-bottom: 20px;
        }}
        .positions-title {{
            color: var(--accent);
            font-size: 1em;
            font-weight: 600;
            margin-bottom: 15px;
        }}
        .positions-table {{ width: 100%; border-collapse: collapse; }}
        .positions-table th {{
            text-align: left;
            padding: 10px;
            color: var(--text-secondary);
            font-size: 0.75em;
            text-transform: uppercase;
            border-bottom: 1px solid var(--border);
        }}
        .positions-table td {{
            padding: 12px 10px;
            border-bottom: 1px solid var(--border);
            font-size: 0.9em;
        }}
        .positions-table tr:hover {{ background: var(--bg-card-alt); }}
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
        .badge.long {{ background: rgba(0,210,106,0.2); color: var(--positive); }}
        .badge.short {{ background: rgba(255,71,87,0.2); color: var(--negative); }}
        .config-note {{
            background: var(--bg-card-alt);
            border-left: 3px solid var(--accent);
            padding: 10px 15px;
            margin: 20px 0;
            font-size: 0.85em;
            color: var(--text-secondary);
        }}
        @media (max-width: 1200px) {{
            .summary-grid {{ grid-template-columns: repeat(3, 1fr); }}
            .charts-row, .bottom-row {{ grid-template-columns: 1fr; }}
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
            <div class="summary-label">Kapital</div>
            <div class="summary-value">${metrics['current_capital']:,.2f}</div>
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
            <div class="summary-value {closed_pnl_class}">${total_closed_pnl:+,.2f}</div>
        </div>
        <div class="summary-card">
            <div class="summary-label">Total Return</div>
            <div class="summary-value {total_return_class}">${metrics['total_return']:+,.2f} ({metrics['total_return_pct']:+.1f}%)</div>
        </div>
        <div class="summary-card">
            <div class="summary-label">Positions</div>
            <div class="summary-value">{num_positions}</div>
        </div>
    </div>

    <div class="charts-row">
        <div class="chart-card">
            <div class="chart-header">
                <span class="chart-title">Kapitalkurve</span>
                <span class="chart-value {total_return_class}">${metrics['total_return']:+,.2f}</span>
            </div>
            <div class="chart-container">
                <canvas id="equityChart"></canvas>
            </div>
        </div>
        <div class="chart-card">
            <div class="chart-header">
                <span class="chart-title">Daily PnL</span>
                <span class="chart-value {daily_class}">${daily:+,.2f}</span>
            </div>
            <div class="chart-container">
                <canvas id="dailyPnlChart"></canvas>
            </div>
        </div>
    </div>

    <div class="bottom-row">
        <div class="chart-card">
            <div class="chart-header">
                <span class="chart-title">Unrealized PnL by Position</span>
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
                <span class="metric-label">Win Rate</span>
                <span class="metric-value {winrate_class}">{metrics['win_rate']}%</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Profit Factor</span>
                <span class="metric-value {pf_class}">{metrics['profit_factor']}</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Sharpe Ratio</span>
                <span class="metric-value {sharpe_class}">{metrics['sharpe_ratio']}</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Max Drawdown</span>
                <span class="metric-value negative">-${metrics['max_drawdown']:,.2f} ({metrics['max_drawdown_pct']}%)</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Total Trades</span>
                <span class="metric-value">{metrics['total_trades']}</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Avg Winner</span>
                <span class="metric-value positive">${metrics['avg_winner']:+,.2f}</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Avg Loser</span>
                <span class="metric-value negative">${metrics['avg_loser']:,.2f}</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Expectancy</span>
                <span class="metric-value {expectancy_class}">${metrics['expectancy']:+,.2f}</span>
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
                    <th>Entry Price</th>
                    <th>Current Price</th>
                    <th>Market Value</th>
                    <th>Unrealized PnL</th>
                    <th>PnL %</th>
                </tr>
            </thead>
            <tbody>
                {positions_html if positions_html else '<tr><td colspan="8" style="text-align:center;padding:40px;color:var(--text-secondary);">No open positions</td></tr>'}
            </tbody>
        </table>
    </div>

    <div class="positions-card">
        <div class="positions-title">Closed Trades ({len(closed_trades)}) <span class="{closed_pnl_class}" style="float:right;">Total: ${total_closed_pnl:+,.2f}</span></div>
        <table class="positions-table">
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th>Direction</th>
                    <th>Qty</th>
                    <th>Entry</th>
                    <th>Entry Price</th>
                    <th>Exit</th>
                    <th>Exit Price</th>
                    <th>Days</th>
                    <th>P&L $</th>
                    <th>P&L %</th>
                </tr>
            </thead>
            <tbody>
                {closed_trades_html if closed_trades_html else '<tr><td colspan="10" style="text-align:center;padding:40px;color:var(--text-secondary);">No closed trades yet</td></tr>'}
            </tbody>
        </table>
    </div>

    <div class="config-note">
        <strong>Kapitalberechnung:</strong> Startkapital ${config.get('initial_capital', INITIAL_CAPITAL):,.0f} |
        Entry Fee ${config.get('entry_fee', ENTRY_FEE):.2f} | Exit Fee ${config.get('exit_fee', EXIT_FEE):.2f} |
        FlexQuery: {'✓ Aktiv' if config.get('flex_token') else '✗ Nicht konfiguriert (--setup-flex)'}
    </div>

    <script>
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
                    backgroundColor: 'rgba(0,210,106,0.1)',
                    fill: true,
                    tension: 0.4,
                    pointRadius: 3,
                    borderWidth: 2
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ display: false }} }},
                scales: {{
                    x: {{ grid: {{ color: '#2d3e50' }} }},
                    y: {{ grid: {{ color: '#2d3e50' }}, ticks: {{ callback: v => '$' + v.toLocaleString() }} }}
                }}
            }}
        }});

        // Daily PnL Chart
        const dailyData = {daily_pnl_values};
        new Chart(document.getElementById('dailyPnlChart'), {{
            type: 'bar',
            data: {{
                labels: {daily_pnl_labels},
                datasets: [{{
                    data: dailyData,
                    backgroundColor: dailyData.map(v => v >= 0 ? '#00d26a' : '#ff4757'),
                    borderRadius: 4
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ display: false }} }},
                scales: {{
                    x: {{ grid: {{ display: false }} }},
                    y: {{ grid: {{ color: '#2d3e50' }}, ticks: {{ callback: v => '$' + v.toLocaleString() }} }}
                }}
            }}
        }});

        // Position PnL Chart
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
                plugins: {{ legend: {{ display: false }} }},
                scales: {{
                    x: {{ grid: {{ color: '#2d3e50' }}, ticks: {{ callback: v => '$' + v.toLocaleString() }} }},
                    y: {{ grid: {{ display: false }} }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
    return html


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("="*60)
    print("TWS DASHBOARD GENERATOR")
    print("="*60)

    # Check for setup mode
    if "--setup-flex" in sys.argv:
        setup_flexquery()
        return

    # Load config
    config = load_config()

    # Parse arguments
    auto_refresh = "--live" in sys.argv
    port = config.get('ib_port', IB_PORT)

    for i, arg in enumerate(sys.argv):
        if arg == "--port" and i + 1 < len(sys.argv):
            try:
                port = int(sys.argv[i + 1])
            except:
                pass

    # Load history
    history = load_history()

    print(f"\nConfig: Startkapital ${config['initial_capital']:,.0f}, {config['num_positions']} Positionen")
    print(f"FlexQuery: {'Konfiguriert' if config.get('flex_token') else 'Nicht konfiguriert'}")

    # Fetch trades from FlexQuery
    flex_trades = []
    if config.get('flex_token') and config.get('flex_query_id'):
        print("\nFetching trades from FlexQuery...")
        flex_trades = fetch_flexquery_trades(config['flex_token'], config['flex_query_id'])
        print(f"  Found {len(flex_trades)} trades")

    # Process trades
    flex_open_positions, flex_closed_trades = process_trades_to_positions(flex_trades)

    # Debug: show first closed trade data
    if flex_closed_trades:
        t = flex_closed_trades[0]
        logger.info(f"First closed trade: {t.get('symbol')} entry_date={t.get('entry_date')} entry_time={t.get('entry_time')} exit_date={t.get('exit_date')} exit_time={t.get('exit_time')}")

    # Connect to TWS for current prices
    print(f"\nConnecting to TWS on port {port}...")
    ib = connect_to_ib(port)

    if ib:
        print("Fetching portfolio data...")
        data = get_portfolio_from_tws(ib)

        # Update FlexQuery positions with current prices
        if flex_open_positions:
            symbols = [p['symbol'] for p in flex_open_positions]
            prices = get_current_prices(ib, symbols)

            for pos in flex_open_positions:
                if pos['symbol'] in prices:
                    current = prices[pos['symbol']]
                    pos['current_price'] = round(current, 2)

                    if pos['direction'] == 'LONG':
                        pos['unrealized_pnl'] = round(pos['quantity'] * (current - pos['entry_price']), 2)
                    else:
                        pos['unrealized_pnl'] = round(pos['quantity'] * (pos['entry_price'] - current), 2)

                    cost = pos['entry_price'] * pos['quantity']
                    pos['pnl_pct'] = round((pos['unrealized_pnl'] / cost) * 100, 2) if cost != 0 else 0

        ib.disconnect()
        print("Disconnected from TWS")
    else:
        print("\n⚠ Could not connect to TWS")
        data = {
            'account': {'NetLiquidation': config['initial_capital']},
            'positions': flex_open_positions,
            'total_unrealized_pnl': sum(p.get('unrealized_pnl', 0) for p in flex_open_positions),
            'total_realized_pnl': sum(t.get('pnl', 0) for t in flex_closed_trades),
            'daily_pnl': 0,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

    # Use FlexQuery data if available, otherwise TWS data
    positions = data.get('positions', []) if data.get('positions') else flex_open_positions
    closed_trades = flex_closed_trades if flex_closed_trades else history.get('trades', [])

    # Calculate equity curve from trades
    print("\nCalculating equity curve...")
    equity_curve = calculate_equity_curve(history, positions, closed_trades, config)

    # Calculate metrics
    metrics = calculate_performance_metrics(equity_curve, closed_trades, config)

    # Update data with calculated values
    if equity_curve:
        data['account']['NetLiquidation'] = equity_curve[-1]['value']

    # Save history
    history['trades'] = closed_trades
    history['equity'] = equity_curve
    history['open_positions'] = positions
    save_history(history)

    # Generate HTML
    print("\nGenerating dashboard...")
    html = generate_html(data, equity_curve, closed_trades, metrics, config, auto_refresh)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(html)

    # Summary
    print(f"\n{'='*60}")
    print(f"✓ Dashboard saved to: {OUTPUT_FILE}")
    print(f"{'='*60}")
    print(f"  Kapital:           ${metrics['current_capital']:,.2f}")
    print(f"  Total Return:      ${metrics['total_return']:+,.2f} ({metrics['total_return_pct']:+.1f}%)")
    print(f"  Open Positions:    {len(positions)}")
    print(f"  Closed Trades:     {len(closed_trades)}")
    print(f"  Win Rate:          {metrics['win_rate']}%")
    print(f"  Profit Factor:     {metrics['profit_factor']}")

    if not config.get('flex_token'):
        print(f"\n  ⚠ FlexQuery nicht konfiguriert!")
        print(f"    Führe aus: python generate_dashboard.py --setup-flex")

    # Open in browser
    try:
        import webbrowser
        webbrowser.open('file://' + os.path.realpath(OUTPUT_FILE))
        print("\n  Opened in browser")
    except:
        pass


if __name__ == "__main__":
    main()
