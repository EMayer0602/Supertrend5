"""
IB Paper Trader with Live Dashboard
====================================
- Live trading with Interactive Brokers TWS/IB Gateway
- Uses strategy signals from long_short_categorized.json
- Trailing stop management
- Real-time position monitoring
- HTML dashboard with live updates
- Separate LONG/SHORT optimization support

Usage:
    python ib_paper_trader.py              # Normal trading
    python ib_paper_trader.py --force      # Override market hours check
    python ib_paper_trader.py --dashboard  # Generate dashboard only
    python ib_paper_trader.py --long-only  # Only trade LONG positions
"""

import asyncio
import json
import logging
import sys
import time
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
import numpy as np
import pandas as pd

# Fix for Python 3.10+ event loop issue
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

from ib_insync import IB, Stock, MarketOrder, LimitOrder, util

# Import strategy calculations from new5.py
from new5 import (
    calculate_supertrend_vectorized,
    calculate_jma, calculate_kama, calculate_sma, calculate_ema,
    ALL_TICKERS
)

# =============================================================================
# CONFIGURATION
# =============================================================================
STATE_FILE = "ib_paper_trader_state.json"
LOG_FILE = "ib_paper_trader.log"
DASHBOARD_FILE = "ib_dashboard.html"

# Trading parameters
MAX_POSITIONS = 30
POSITION_STAKE = 25000  # $ per position
TRAILING_STOP_PCT = 0.20  # 20% trailing stop for B&H
MIN_PNL_THRESHOLD = 0.15  # 15% minimum expected return

# IB connection
IB_HOST = "127.0.0.1"
IB_PORT = 7497  # TWS paper trading (7496 for live)
IB_CLIENT_ID = 1

# Market hours (US Eastern)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0

# =============================================================================
# LOGGING SETUP
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================
@dataclass
class PositionInfo:
    """Position information"""
    symbol: str
    direction: str  # 'LONG' or 'SHORT'
    quantity: int
    avg_cost: float
    highest_price: float  # For trailing stop
    lowest_price: float   # For short trailing stop
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    market_value: float = 0.0
    strategy: str = ""
    entry_date: str = ""

    def update_price(self, price: float):
        """Update current price and calculate PnL"""
        self.current_price = price
        self.market_value = abs(self.quantity) * price

        if self.direction == 'LONG':
            self.unrealized_pnl = self.quantity * (price - self.avg_cost)
            if price > self.highest_price:
                self.highest_price = price
        else:  # SHORT
            self.unrealized_pnl = abs(self.quantity) * (self.avg_cost - price)
            if self.lowest_price == 0 or price < self.lowest_price:
                self.lowest_price = price

    def check_trailing_stop(self) -> bool:
        """Check if trailing stop is hit"""
        if self.direction == 'LONG':
            stop_price = self.highest_price * (1 - TRAILING_STOP_PCT)
            return self.current_price <= stop_price
        else:  # SHORT
            if self.lowest_price == 0:
                return False
            stop_price = self.lowest_price * (1 + TRAILING_STOP_PCT)
            return self.current_price >= stop_price

    def pnl_percent(self) -> float:
        """Calculate PnL percentage"""
        cost = abs(self.quantity) * self.avg_cost
        if cost == 0:
            return 0
        return self.unrealized_pnl / cost


@dataclass
class TradeLog:
    """Trade log entry"""
    timestamp: str
    symbol: str
    action: str  # BUY, SELL, SHORT, COVER
    quantity: int
    price: float = 0.0
    order_type: str = "MKT"
    status: str = "Pending"
    order_id: int = 0
    realized_pnl: float = 0.0


# =============================================================================
# IB PAPER TRADER
# =============================================================================
class IBPaperTrader:
    """
    Live paper trading with Interactive Brokers
    """

    def __init__(self,
                 host: str = IB_HOST,
                 port: int = IB_PORT,
                 client_id: int = IB_CLIENT_ID,
                 long_only: bool = False):

        self.host = host
        self.port = port
        self.client_id = client_id
        self.long_only = long_only

        self.ib = IB()
        self.connected = False

        # State
        self.positions: Dict[str, PositionInfo] = {}
        self.signals: Dict[str, str] = {}  # symbol -> current signal
        self.trade_log: List[TradeLog] = []
        self.pending_orders: Dict[int, TradeLog] = {}

        # Strategy assignments
        self.long_assignments: Dict = {}
        self.short_assignments: Dict = {}

        # Price data cache
        self.price_data: Dict[str, pd.DataFrame] = {}
        self.last_prices: Dict[str, float] = {}

        # Account info
        self.account_value = 0.0
        self.buying_power = 0.0
        self.daily_pnl = 0.0

        # Load state
        self.load_state()
        self.load_strategy_assignments()

    def load_strategy_assignments(self):
        """Load strategy assignments from categorization file"""
        try:
            with open('long_short_categorized.json', 'r') as f:
                data = json.load(f)

                # Load LONG assignments
                long_cats = data.get('long_categories', {})
                for cat_name, items in long_cats.items():
                    if 'UNDERPERFORM' in cat_name:
                        continue
                    for item in items:
                        symbol = item['symbol']
                        base = cat_name.replace('_LONG_HTF', '').replace('_LONG_NOHTF', '')
                        has_htf = '_HTF' in cat_name
                        self.long_assignments[symbol] = {
                            'strategy': f"{base}{'_HTF' if has_htf else ''}",
                            'params': item.get('params', {}),
                            'return': item.get('return', 0)
                        }

                # Load SHORT assignments
                short_cats = data.get('short_categories', {})
                for cat_name, items in short_cats.items():
                    if 'UNDERPERFORM' in cat_name:
                        continue
                    for item in items:
                        symbol = item['symbol']
                        base = cat_name.replace('_SHORT_HTF', '').replace('_SHORT_NOHTF', '')
                        has_htf = '_HTF' in cat_name
                        self.short_assignments[symbol] = {
                            'strategy': f"{base}{'_HTF' if has_htf else ''}",
                            'params': item.get('params', {}),
                            'return': item.get('return', 0)
                        }

                logger.info(f"Loaded LONG: {len(self.long_assignments)}, SHORT: {len(self.short_assignments)} assignments")

        except FileNotFoundError:
            logger.warning("long_short_categorized.json not found, using defaults")
        except Exception as e:
            logger.error(f"Error loading strategy assignments: {e}")

    def load_state(self):
        """Load state from file (with backward compatibility)"""
        try:
            with open(STATE_FILE, 'r') as f:
                data = json.load(f)

                # Load positions (handle old and new format)
                for symbol, pos_data in data.get('positions', {}).items():
                    # Detect old format (no 'direction' field)
                    if 'direction' not in pos_data:
                        # Old format - infer direction from highest_price vs avg_cost
                        # or default to LONG
                        direction = 'LONG'
                        quantity = 0  # Will be synced from IB
                    else:
                        direction = pos_data.get('direction', 'LONG')
                        quantity = pos_data.get('quantity', 0)

                    self.positions[symbol] = PositionInfo(
                        symbol=symbol,
                        direction=direction,
                        quantity=quantity,
                        avg_cost=pos_data.get('avg_cost', 0),
                        highest_price=pos_data.get('highest_price', 0),
                        lowest_price=pos_data.get('lowest_price', 0),
                        strategy=pos_data.get('strategy', ''),
                        entry_date=pos_data.get('entry_date', '')
                    )

                # Load signals
                self.signals = data.get('signals', {})

                # Load trade log (handle old format without all fields)
                for log_data in data.get('trade_log', [])[-100:]:  # Keep last 100
                    try:
                        # Add missing fields with defaults
                        log_data.setdefault('price', 0.0)
                        log_data.setdefault('order_id', 0)
                        log_data.setdefault('realized_pnl', 0.0)
                        self.trade_log.append(TradeLog(**log_data))
                    except Exception:
                        pass  # Skip malformed entries

                logger.info(f"Loaded state: {len(self.positions)} positions, {len(self.signals)} signals")

        except FileNotFoundError:
            logger.info("No existing state file, starting fresh")
        except Exception as e:
            logger.error(f"Error loading state: {e}")

    def save_state(self):
        """Save state to file"""
        try:
            data = {
                'positions': {
                    symbol: {
                        'direction': pos.direction,
                        'quantity': pos.quantity,
                        'avg_cost': pos.avg_cost,
                        'highest_price': pos.highest_price,
                        'lowest_price': pos.lowest_price,
                        'strategy': pos.strategy,
                        'entry_date': pos.entry_date
                    }
                    for symbol, pos in self.positions.items()
                },
                'signals': self.signals,
                'trade_log': [asdict(log) for log in self.trade_log[-100:]],
                'last_update': datetime.now().isoformat()
            }

            with open(STATE_FILE, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving state: {e}")

    def connect(self) -> bool:
        """Connect to IB TWS"""
        try:
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            self.connected = True

            # Subscribe to events
            self.ib.positionEvent += self.on_position
            self.ib.orderStatusEvent += self.on_order_status
            self.ib.pnlEvent += self.on_pnl
            self.ib.updatePortfolioEvent += self.on_portfolio_update
            self.ib.errorEvent += self.on_error

            logger.info(f"Connected to IB at {self.host}:{self.port}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False

    def disconnect(self):
        """Disconnect from IB"""
        if self.connected:
            self.ib.disconnect()
            self.connected = False
            logger.info("Disconnected from IB")

    def on_position(self, position):
        """Handle position updates from IB"""
        logger.info(f"position: {position}")

        symbol = position.contract.symbol
        qty = int(position.position)
        avg_cost = position.avgCost

        if qty != 0:
            direction = 'LONG' if qty > 0 else 'SHORT'

            if symbol not in self.positions:
                self.positions[symbol] = PositionInfo(
                    symbol=symbol,
                    direction=direction,
                    quantity=abs(qty),
                    avg_cost=avg_cost,
                    highest_price=avg_cost if direction == 'LONG' else 0,
                    lowest_price=avg_cost if direction == 'SHORT' else 0,
                    entry_date=datetime.now().strftime('%Y-%m-%d')
                )
            else:
                pos = self.positions[symbol]
                pos.quantity = abs(qty)
                pos.avg_cost = avg_cost
                pos.direction = direction
        else:
            # Position closed
            if symbol in self.positions:
                del self.positions[symbol]

    def on_portfolio_update(self, item):
        """Handle portfolio updates"""
        logger.info(f"updatePortfolio: {item}")

        symbol = item.contract.symbol

        if symbol in self.positions:
            pos = self.positions[symbol]
            pos.current_price = item.marketPrice
            pos.market_value = abs(item.marketValue)
            pos.unrealized_pnl = item.unrealizedPNL

            # Update highest/lowest
            if pos.direction == 'LONG' and item.marketPrice > pos.highest_price:
                pos.highest_price = item.marketPrice
            elif pos.direction == 'SHORT':
                if pos.lowest_price == 0 or item.marketPrice < pos.lowest_price:
                    pos.lowest_price = item.marketPrice

    def on_order_status(self, trade):
        """Handle order status updates"""
        order = trade.order
        status = trade.orderStatus.status

        logger.info(f"Order {order.orderId}: {status}")

        if order.orderId in self.pending_orders:
            self.pending_orders[order.orderId].status = status

            if status in ['Filled', 'Cancelled', 'Error']:
                log = self.pending_orders.pop(order.orderId)
                if status == 'Filled':
                    log.price = trade.orderStatus.avgFillPrice
                    log.status = 'Filled'
                self.trade_log.append(log)

    def on_pnl(self, entry):
        """Handle PnL updates"""
        self.daily_pnl = entry.dailyPnL if entry.dailyPnL else 0

    def on_error(self, reqId, errorCode, errorString, contract):
        """Handle errors"""
        logger.error(f"[{errorCode}] {errorString}")

    def sync_positions(self):
        """Synchronize positions with IB account"""
        logger.info("Synchronizing positions...")

        # Request positions
        self.ib.reqPositions()
        self.ib.sleep(2)

        # Request portfolio
        accounts = self.ib.managedAccounts()
        if accounts:
            self.ib.reqAccountSummary()
            self.ib.sleep(1)

            # Subscribe to PnL
            for symbol in self.positions:
                contract = Stock(symbol, 'SMART', 'USD')
                self.ib.qualifyContracts(contract)

        logger.info(f"Synchronized {len(self.positions)} positions")

    def download_price_data(self, symbol: str, days: int = 100) -> Optional[pd.DataFrame]:
        """Download historical price data"""
        try:
            contract = Stock(symbol, 'SMART', 'USD')
            self.ib.qualifyContracts(contract)

            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr=f'{days} D',
                barSizeSetting='1 day',
                whatToShow='TRADES',
                useRTH=True,
                formatDate=1
            )

            if bars:
                df = util.df(bars)
                df.set_index('date', inplace=True)
                df.rename(columns={
                    'open': f'Open_{symbol}',
                    'high': f'High_{symbol}',
                    'low': f'Low_{symbol}',
                    'close': f'Close_{symbol}',
                    'volume': f'Volume_{symbol}'
                }, inplace=True)

                self.ib.sleep(0.3)  # Avoid pacing violations
                return df

        except Exception as e:
            logger.error(f"Error downloading {symbol}: {e}")

        return None

    def generate_signal(self, symbol: str) -> Optional[str]:
        """Generate trading signal for symbol"""
        if symbol not in self.price_data:
            df = self.download_price_data(symbol)
            if df is not None:
                self.price_data[symbol] = df
            else:
                return None

        df = self.price_data[symbol]
        if len(df) < 50:
            return None

        close_col = f'Close_{symbol}'
        high_col = f'High_{symbol}'
        low_col = f'Low_{symbol}'

        if close_col not in df.columns:
            return None

        close = df[close_col].values
        high = df[high_col].values
        low = df[low_col].values
        close_series = df[close_col]

        # Get strategy assignments
        long_assign = self.long_assignments.get(symbol, {})
        short_assign = self.short_assignments.get(symbol, {})

        current_pos = self.positions.get(symbol)
        has_long = current_pos and current_pos.direction == 'LONG'
        has_short = current_pos and current_pos.direction == 'SHORT'

        try:
            # LONG signal calculation
            long_signal = None
            long_strategy = long_assign.get('strategy', 'SUPERTREND')
            long_params = long_assign.get('params', {})
            long_base = long_strategy.replace('_HTF', '').replace('_NOHTF', '')

            if long_base == 'SUPERTREND':
                period = long_params.get('period', 10)
                mult = long_params.get('multiplier', 3.0)
                _, direction, _ = calculate_supertrend_vectorized(high, low, close, period, mult)
                if len(direction) >= 2:
                    if direction[-1] == 1 and direction[-2] == -1:
                        long_signal = 'BUY'
                    elif direction[-1] == -1 and direction[-2] == 1:
                        long_signal = 'SELL'
            elif long_base in ['JMA', 'KAMA', 'EMA', 'SMA']:
                fast = long_params.get('fast', 10)
                slow = long_params.get('slow', 30)
                if long_base == 'JMA':
                    ma_fast = calculate_jma(close_series, fast).values
                    ma_slow = calculate_jma(close_series, slow).values
                elif long_base == 'KAMA':
                    period = long_params.get('period', 10)
                    signal = long_params.get('signal', 14)
                    ma_fast = calculate_kama(close_series, period).values
                    ma_slow = calculate_sma(close_series, signal).values
                elif long_base == 'EMA':
                    ma_fast = calculate_ema(close_series, fast).values
                    ma_slow = calculate_ema(close_series, slow).values
                else:
                    ma_fast = calculate_sma(close_series, fast).values
                    ma_slow = calculate_sma(close_series, slow).values

                if len(ma_fast) >= 2 and len(ma_slow) >= 2:
                    if ma_fast[-1] > ma_slow[-1] and ma_fast[-2] <= ma_slow[-2]:
                        long_signal = 'BUY'
                    elif ma_fast[-1] < ma_slow[-1] and ma_fast[-2] >= ma_slow[-2]:
                        long_signal = 'SELL'

            # SHORT signal calculation (skip if long_only)
            short_signal = None
            if not self.long_only:
                short_strategy = short_assign.get('strategy', 'SUPERTREND')
                short_params = short_assign.get('params', {})
                short_base = short_strategy.replace('_HTF', '').replace('_NOHTF', '')

                if short_base == 'SUPERTREND':
                    period = short_params.get('period', 7)
                    mult = short_params.get('multiplier', 2.0)
                    _, direction, _ = calculate_supertrend_vectorized(high, low, close, period, mult)
                    if len(direction) >= 2:
                        if direction[-1] == -1 and direction[-2] == 1:
                            short_signal = 'SHORT'
                        elif direction[-1] == 1 and direction[-2] == -1:
                            short_signal = 'COVER'
                elif short_base in ['JMA', 'KAMA', 'EMA', 'SMA']:
                    fast = short_params.get('fast', 5)
                    slow = short_params.get('slow', 20)
                    if short_base == 'JMA':
                        ma_fast = calculate_jma(close_series, fast).values
                        ma_slow = calculate_jma(close_series, slow).values
                    elif short_base == 'KAMA':
                        period = short_params.get('period', 7)
                        signal = short_params.get('signal', 10)
                        ma_fast = calculate_kama(close_series, period).values
                        ma_slow = calculate_sma(close_series, signal).values
                    elif short_base == 'EMA':
                        ma_fast = calculate_ema(close_series, fast).values
                        ma_slow = calculate_ema(close_series, slow).values
                    else:
                        ma_fast = calculate_sma(close_series, fast).values
                        ma_slow = calculate_sma(close_series, slow).values

                    if len(ma_fast) >= 2 and len(ma_slow) >= 2:
                        if ma_fast[-1] < ma_slow[-1] and ma_fast[-2] >= ma_slow[-2]:
                            short_signal = 'SHORT'
                        elif ma_fast[-1] > ma_slow[-1] and ma_fast[-2] <= ma_slow[-2]:
                            short_signal = 'COVER'

            # Determine final signal based on current position
            if has_short and short_signal == 'COVER':
                return 'COVER'
            if has_long and long_signal == 'SELL':
                return 'SELL'
            if not has_long and not has_short:
                if long_signal == 'BUY':
                    return 'BUY'
                elif short_signal == 'SHORT':
                    return 'SHORT'

            # Return current hold signal
            if has_long:
                return 'HOLD_LONG'
            elif has_short:
                return 'HOLD_SHORT'

            return 'NEUTRAL'

        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return None

    def calculate_quantity(self, price: float) -> int:
        """Calculate position size"""
        return max(1, int(POSITION_STAKE / price))

    def place_order(self, symbol: str, action: str, quantity: int) -> Optional[int]:
        """Place an order with IB"""
        try:
            contract = Stock(symbol, 'SMART', 'USD')
            self.ib.qualifyContracts(contract)

            # Get current price for limit order
            ticker = self.ib.reqMktData(contract, '', False, False)
            self.ib.sleep(1)

            price = ticker.last if ticker.last > 0 else ticker.close
            if price <= 0:
                logger.error(f"No price available for {symbol}")
                return None

            # Use limit order slightly away from current price
            if action in ['BUY', 'COVER']:
                limit_price = round(price * 1.001, 2)  # 0.1% above
            else:  # SELL, SHORT
                limit_price = round(price * 0.999, 2)  # 0.1% below

            order = LimitOrder(action, quantity, limit_price)
            order.tif = 'DAY'

            trade = self.ib.placeOrder(contract, order)

            # Log the order
            log = TradeLog(
                timestamp=datetime.now().isoformat(),
                symbol=symbol,
                action=action,
                quantity=quantity,
                price=limit_price,
                order_type='LMT',
                status='Submitted',
                order_id=trade.order.orderId
            )

            self.pending_orders[trade.order.orderId] = log

            logger.info(f"Placed {action} order for {quantity} {symbol} @ ${limit_price:.2f}")

            return trade.order.orderId

        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None

    def check_market_hours(self) -> bool:
        """Check if market is open"""
        now = datetime.now()

        # Skip weekends
        if now.weekday() >= 5:
            return False

        market_open = now.replace(hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MINUTE, second=0)
        market_close = now.replace(hour=MARKET_CLOSE_HOUR, minute=MARKET_CLOSE_MINUTE, second=0)

        return market_open <= now <= market_close

    def check_data_subscription(self) -> bool:
        """Check if we have market data subscription"""
        try:
            # Test with SPY
            contract = Stock('SPY', 'SMART', 'USD')
            self.ib.qualifyContracts(contract)

            ticker = self.ib.reqMktData(contract, '', False, False)
            self.ib.sleep(2)

            if ticker.last > 0 or ticker.close > 0:
                logger.info("[OK] Market Data: LIVE (real-time)")
                return True
            else:
                logger.warning("[!] Market Data: DELAYED or unavailable")
                return False

        except Exception as e:
            logger.error(f"Market data check failed: {e}")
            return False

    def run_trading_loop(self, force: bool = False):
        """Main trading loop"""
        if not self.connected:
            if not self.connect():
                return

        try:
            # Sync positions
            self.sync_positions()

            # Check market data
            self.check_data_subscription()

            # Subscribe to PnL
            accounts = self.ib.managedAccounts()
            if accounts:
                for symbol in list(self.positions.keys()):
                    contract = Stock(symbol, 'SMART', 'USD')
                    self.ib.qualifyContracts(contract)

                logger.info(f"Subscribed to PnL updates for {len(self.positions)} positions")

            # Trading loop
            while True:
                try:
                    # Check market hours
                    if not force and not self.check_market_hours():
                        logger.info("Market closed. Waiting... (use --force to override)")
                        self.ib.sleep(300)  # Wait 5 minutes
                        continue

                    # Get symbols to check
                    symbols = list(set(
                        list(self.long_assignments.keys()) +
                        list(self.short_assignments.keys()) +
                        list(self.positions.keys())
                    ))

                    # Generate and process signals
                    for symbol in symbols:
                        try:
                            signal = self.generate_signal(symbol)
                            if signal:
                                self.signals[symbol] = signal

                                # Execute trades based on signals
                                if signal == 'BUY' and symbol not in self.positions:
                                    if len(self.positions) < MAX_POSITIONS:
                                        # Get current price
                                        if symbol in self.price_data:
                                            price = self.price_data[symbol][f'Close_{symbol}'].iloc[-1]
                                            qty = self.calculate_quantity(price)
                                            self.place_order(symbol, 'BUY', qty)

                                elif signal == 'SELL' and symbol in self.positions:
                                    pos = self.positions[symbol]
                                    if pos.direction == 'LONG':
                                        self.place_order(symbol, 'SELL', pos.quantity)

                                elif signal == 'SHORT' and symbol not in self.positions:
                                    if len(self.positions) < MAX_POSITIONS and not self.long_only:
                                        if symbol in self.price_data:
                                            price = self.price_data[symbol][f'Close_{symbol}'].iloc[-1]
                                            qty = self.calculate_quantity(price)
                                            self.place_order(symbol, 'SELL', qty)  # Short = SELL

                                elif signal == 'COVER' and symbol in self.positions:
                                    pos = self.positions[symbol]
                                    if pos.direction == 'SHORT':
                                        self.place_order(symbol, 'BUY', pos.quantity)  # Cover = BUY

                            self.ib.sleep(0.5)

                        except Exception as e:
                            logger.error(f"Error processing {symbol}: {e}")

                    # Check trailing stops
                    for symbol, pos in list(self.positions.items()):
                        if pos.check_trailing_stop():
                            logger.info(f"Trailing stop triggered for {symbol}")
                            if pos.direction == 'LONG':
                                self.place_order(symbol, 'SELL', pos.quantity)
                            else:
                                self.place_order(symbol, 'BUY', pos.quantity)

                    # Save state and generate dashboard
                    self.save_state()
                    self.generate_dashboard()

                    # Wait before next cycle
                    self.ib.sleep(60)  # 1 minute

                except KeyboardInterrupt:
                    logger.info("Interrupted by user")
                    break
                except Exception as e:
                    logger.error(f"Trading loop error: {e}")
                    self.ib.sleep(60)

        finally:
            self.save_state()
            self.disconnect()

    def generate_dashboard(self):
        """Generate HTML dashboard"""
        # Calculate statistics
        total_unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        total_market_value = sum(p.market_value for p in self.positions.values())

        long_positions = [p for p in self.positions.values() if p.direction == 'LONG']
        short_positions = [p for p in self.positions.values() if p.direction == 'SHORT']

        # Sort positions by unrealized PnL
        sorted_positions = sorted(self.positions.values(), key=lambda p: p.unrealized_pnl, reverse=True)

        # Recent trades
        recent_trades = sorted(self.trade_log, key=lambda t: t.timestamp, reverse=True)[:20]

        # Generate position rows
        position_rows = ""
        for pos in sorted_positions:
            pnl_class = 'positive' if pos.unrealized_pnl >= 0 else 'negative'
            pnl_pct = pos.pnl_percent() * 100
            direction_class = 'long' if pos.direction == 'LONG' else 'short'

            # Trailing stop price
            if pos.direction == 'LONG':
                stop_price = pos.highest_price * (1 - TRAILING_STOP_PCT)
            else:
                stop_price = pos.lowest_price * (1 + TRAILING_STOP_PCT) if pos.lowest_price > 0 else 0

            position_rows += f'''
            <tr>
                <td><span class="symbol">{pos.symbol}</span></td>
                <td><span class="direction {direction_class}">{pos.direction}</span></td>
                <td>{pos.quantity}</td>
                <td>${pos.avg_cost:.2f}</td>
                <td>${pos.current_price:.2f}</td>
                <td>${pos.market_value:,.2f}</td>
                <td class="{pnl_class}">${pos.unrealized_pnl:,.2f}</td>
                <td class="{pnl_class}">{pnl_pct:+.2f}%</td>
                <td>${stop_price:.2f}</td>
                <td>{pos.strategy}</td>
            </tr>
            '''

        # Generate trade log rows
        trade_rows = ""
        for trade in recent_trades:
            action_class = 'buy' if trade.action in ['BUY', 'COVER'] else 'sell'
            trade_rows += f'''
            <tr>
                <td>{trade.timestamp[:19]}</td>
                <td>{trade.symbol}</td>
                <td><span class="action {action_class}">{trade.action}</span></td>
                <td>{trade.quantity}</td>
                <td>${trade.price:.2f}</td>
                <td>{trade.status}</td>
            </tr>
            '''

        # Position PnL data for chart
        pnl_data = [
            {"symbol": p.symbol, "pnl": p.unrealized_pnl, "direction": p.direction}
            for p in sorted_positions
        ]

        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="refresh" content="60">
    <title>IB Paper Trader Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0a1628 0%, #1a2744 100%);
            color: #e0e0e0;
            min-height: 100vh;
        }}
        .header {{
            background: rgba(26, 39, 68, 0.95);
            padding: 20px 30px;
            border-bottom: 2px solid #4ecdc4;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .header h1 {{
            color: #4ecdc4;
            font-size: 28px;
        }}
        .header .status {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .header .status .dot {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #27ae60;
            animation: pulse 2s infinite;
        }}
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.5; }}
        }}
        .header .timestamp {{
            color: #8899a6;
            font-size: 14px;
        }}
        .stats-bar {{
            display: flex;
            gap: 20px;
            padding: 20px 30px;
            background: rgba(20, 35, 60, 0.8);
            flex-wrap: wrap;
        }}
        .stat-card {{
            background: rgba(30, 50, 80, 0.8);
            border-radius: 10px;
            padding: 15px 25px;
            min-width: 180px;
            border: 1px solid #2a3f5f;
        }}
        .stat-card.highlight {{
            background: linear-gradient(135deg, #1e8449 0%, #145a32 100%);
            border-color: #27ae60;
        }}
        .stat-card.warning {{
            background: linear-gradient(135deg, #b03a2e 0%, #78281f 100%);
            border-color: #e74c3c;
        }}
        .stat-card .label {{
            font-size: 11px;
            color: #8899a6;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .stat-card .value {{
            font-size: 26px;
            font-weight: bold;
            margin-top: 5px;
        }}
        .stat-card .value.positive {{ color: #4ecdc4; }}
        .stat-card .value.negative {{ color: #ff6b6b; }}
        .container {{
            padding: 20px 30px;
        }}
        .section {{
            background: rgba(20, 35, 60, 0.8);
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid #2a3f5f;
        }}
        .section h2 {{
            color: #4ecdc4;
            margin-bottom: 15px;
            font-size: 18px;
            border-bottom: 1px solid #2a3f5f;
            padding-bottom: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #2a3f5f;
        }}
        th {{
            background: rgba(30, 50, 80, 0.6);
            color: #4ecdc4;
            font-weight: 600;
            font-size: 12px;
            text-transform: uppercase;
        }}
        tr:hover {{
            background: rgba(78, 205, 196, 0.1);
        }}
        .symbol {{
            font-weight: bold;
            color: #fff;
        }}
        .direction {{
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 11px;
            font-weight: bold;
        }}
        .direction.long {{
            background: #1e8449;
            color: #fff;
        }}
        .direction.short {{
            background: #b03a2e;
            color: #fff;
        }}
        .action {{
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 11px;
            font-weight: bold;
        }}
        .action.buy {{
            background: #1e8449;
            color: #fff;
        }}
        .action.sell {{
            background: #b03a2e;
            color: #fff;
        }}
        .positive {{ color: #4ecdc4; }}
        .negative {{ color: #ff6b6b; }}
        .chart-container {{
            height: 400px;
            margin-top: 20px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        @media (max-width: 1200px) {{
            .grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>IB Paper Trader Dashboard</h1>
        <div class="status">
            <div class="dot"></div>
            <span>LIVE</span>
            <span class="timestamp">Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span>
        </div>
    </div>

    <div class="stats-bar">
        <div class="stat-card {'highlight' if total_unrealized >= 0 else 'warning'}">
            <div class="label">Unrealized P&L</div>
            <div class="value {'positive' if total_unrealized >= 0 else 'negative'}">${total_unrealized:+,.2f}</div>
        </div>
        <div class="stat-card">
            <div class="label">Market Value</div>
            <div class="value">${total_market_value:,.2f}</div>
        </div>
        <div class="stat-card">
            <div class="label">Daily P&L</div>
            <div class="value {'positive' if self.daily_pnl >= 0 else 'negative'}">${self.daily_pnl:+,.2f}</div>
        </div>
        <div class="stat-card">
            <div class="label">Positions</div>
            <div class="value">{len(self.positions)}</div>
        </div>
        <div class="stat-card">
            <div class="label">Long / Short</div>
            <div class="value">{len(long_positions)} / {len(short_positions)}</div>
        </div>
        <div class="stat-card">
            <div class="label">Mode</div>
            <div class="value">{'LONG ONLY' if self.long_only else 'LONG+SHORT'}</div>
        </div>
    </div>

    <div class="container">
        <div class="section">
            <h2>Open Positions ({len(self.positions)})</h2>
            <table>
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Direction</th>
                        <th>Qty</th>
                        <th>Avg Cost</th>
                        <th>Current</th>
                        <th>Mkt Value</th>
                        <th>Unr. P&L</th>
                        <th>P&L %</th>
                        <th>Stop</th>
                        <th>Strategy</th>
                    </tr>
                </thead>
                <tbody>
                    {position_rows}
                </tbody>
            </table>
        </div>

        <div class="grid">
            <div class="section">
                <h2>Position P&L Chart</h2>
                <div id="pnlChart" class="chart-container"></div>
            </div>

            <div class="section">
                <h2>Recent Trades</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Time</th>
                            <th>Symbol</th>
                            <th>Action</th>
                            <th>Qty</th>
                            <th>Price</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        {trade_rows}
                    </tbody>
                </table>
            </div>
        </div>

        <div class="section">
            <h2>Signal Summary</h2>
            <p style="color: #8899a6; margin-bottom: 10px;">
                Long Signals: {len(self.long_assignments)} |
                Short Signals: {len(self.short_assignments)} |
                Active Signals: {len([s for s in self.signals.values() if s not in ['NEUTRAL', 'HOLD_LONG', 'HOLD_SHORT']])}
            </p>
        </div>
    </div>

    <script>
        // Position PnL Chart
        var pnlData = {json.dumps(pnl_data)};

        var colors = pnlData.map(d => d.pnl >= 0 ? '#4ecdc4' : '#ff6b6b');

        var trace = {{
            x: pnlData.map(d => d.symbol),
            y: pnlData.map(d => d.pnl),
            type: 'bar',
            marker: {{
                color: colors
            }},
            text: pnlData.map(d => '$' + d.pnl.toFixed(2)),
            textposition: 'auto',
        }};

        var layout = {{
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{ color: '#e0e0e0' }},
            margin: {{ t: 20, b: 80, l: 60, r: 20 }},
            xaxis: {{
                tickangle: -45,
                gridcolor: '#2a3f5f'
            }},
            yaxis: {{
                title: 'P&L ($)',
                gridcolor: '#2a3f5f',
                zerolinecolor: '#4a5568'
            }}
        }};

        Plotly.newPlot('pnlChart', [trace], layout, {{responsive: true}});
    </script>
</body>
</html>
'''

        try:
            with open(DASHBOARD_FILE, 'w') as f:
                f.write(html)
            logger.info(f"Dashboard generated: {DASHBOARD_FILE}")
        except Exception as e:
            logger.error(f"Error generating dashboard: {e}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='IB Paper Trader')
    parser.add_argument('--force', action='store_true', help='Override market hours check')
    parser.add_argument('--dashboard', action='store_true', help='Generate dashboard only (no IB connection)')
    parser.add_argument('--sync', action='store_true', help='Sync positions with IB and generate dashboard')
    parser.add_argument('--long-only', action='store_true', help='Only trade LONG positions')
    parser.add_argument('--port', type=int, default=7497, help='TWS port (default: 7497)')
    args = parser.parse_args()

    trader = IBPaperTrader(port=args.port, long_only=args.long_only)

    if args.dashboard:
        # Just generate dashboard from saved state (no IB connection)
        trader.generate_dashboard()
        print(f"Dashboard generated: {DASHBOARD_FILE}")
    elif args.sync:
        # Connect to IB, sync positions, and generate dashboard
        if trader.connect():
            try:
                trader.sync_positions()
                trader.check_data_subscription()
                trader.save_state()
                trader.generate_dashboard()
                print(f"Synced {len(trader.positions)} positions")
                print(f"Dashboard generated: {DASHBOARD_FILE}")
            finally:
                trader.disconnect()
        else:
            print("Failed to connect to IB")
    else:
        # Run trading loop
        trader.run_trading_loop(force=args.force)


if __name__ == '__main__':
    main()
