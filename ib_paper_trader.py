"""
Interactive Brokers Paper Trading System for Supertrend Strategy
================================================================
Trades the top 30 stocks identified by the Supertrend screener.

Requirements:
- ib_insync: pip install ib_insync
- TWS or IB Gateway running with Paper Trading account
- API connections enabled in TWS/Gateway

Usage:
    python ib_paper_trader.py              # Run paper trader
    python ib_paper_trader.py --dry-run    # Simulation mode (no orders)
    python ib_paper_trader.py --status     # Show current positions
    python ib_paper_trader.py --force      # Run even outside market hours
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import json
import time
import logging
import sys

# Timezone handling
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except ImportError:
    from backports.zoneinfo import ZoneInfo  # pip install backports.zoneinfo

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ib_paper_trader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Try to import ib_insync
try:
    from ib_insync import IB, Stock, MarketOrder, LimitOrder, util
    IB_AVAILABLE = True
except ImportError:
    IB_AVAILABLE = False
    logger.warning("ib_insync not installed. Run: pip install ib_insync")


# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass
class TradingConfig:
    """Paper trading configuration"""
    # IB Connection
    host: str = "127.0.0.1"
    port: int = 7497  # 7497 for TWS Paper, 4002 for Gateway Paper
    client_id: int = 1

    # Portfolio Settings
    initial_capital: float = 100000.0  # $100k paper trading account
    max_position_pct: float = 0.05     # Max 5% per position
    max_positions: int = 20            # Max 20 concurrent positions

    # Supertrend Parameters (optimized from backtests)
    st_period: int = 15
    st_multiplier: float = 4.0

    # Risk Management
    stop_loss_pct: float = 0.08        # 8% stop loss
    trailing_stop_pct: float = 0.12    # 12% trailing stop

    # Trading Rules
    trade_only_market_hours: bool = True
    min_volume: int = 100000           # Minimum daily volume
    min_price: float = 5.0             # Minimum stock price

    # Update Frequency
    signal_check_interval: int = 300   # Check signals every 5 minutes


# Top 30 Supertrend Candidates (from screener results)
TOP_30_STOCKS = [
    # EXCELLENT - High Volatility Winners
    "NFLX",   # +218% outperformance
    "COIN",   # +204% outperformance
    "SHOP",   # +156% outperformance
    "META",   # +143% outperformance
    "DKNG",   # +118% outperformance
    "ARKK",   # +113% outperformance (ETF)
    "MRNA",   # +81% outperformance
    "ROKU",   # +69% outperformance
    "PYPL",   # +62% outperformance
    "SNOW",   # +60% outperformance
    "TSLA",   # +9% outperformance
    "AMD",    # +16% outperformance
    "RBLX",   # +22% outperformance

    # GOOD - Medium Volatility Winners
    "ADBE",   # +72% outperformance
    "BA",     # +59% outperformance
    "DIS",    # +58% outperformance
    "CRM",    # +52% outperformance
    "NKE",    # +52% outperformance
    "TGT",    # +42% outperformance
    "PFE",    # +26% outperformance
    "UNH",    # +22% outperformance
    "LOW",    # +22% outperformance
    "SBUX",   # +22% outperformance

    # Additional volatile stocks for diversification
    "SQ",     # Square/Block - high volatility fintech
    "UBER",   # Uber - volatile tech
    "SNAP",   # Snapchat - high volatility social
    "PINS",   # Pinterest - volatile social
    "DOCU",   # DocuSign - volatile SaaS
    "ZM",     # Zoom - high volatility
    "CRWD",   # CrowdStrike - volatile cybersecurity
]


# =============================================================================
# SUPERTREND CALCULATION
# =============================================================================
def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Calculate Average True Range"""
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]

    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)

    true_range = np.maximum(np.maximum(tr1, tr2), tr3)

    atr = np.zeros_like(true_range)
    atr[:period] = np.nan
    atr[period-1] = np.mean(true_range[:period])

    multiplier = 2 / (period + 1)
    for i in range(period, len(true_range)):
        atr[i] = true_range[i] * multiplier + atr[i-1] * (1 - multiplier)

    return atr


def calculate_supertrend(df: pd.DataFrame, period: int = 15, multiplier: float = 4.0) -> pd.DataFrame:
    """Calculate Supertrend indicator"""
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values

    atr = calculate_atr(high, low, close, period)
    hl2 = (high + low) / 2

    basic_upper = hl2 + (multiplier * atr)
    basic_lower = hl2 - (multiplier * atr)

    n = len(close)
    final_upper = np.zeros(n)
    final_lower = np.zeros(n)
    supertrend = np.zeros(n)
    direction = np.zeros(n)

    final_upper[0] = basic_upper[0]
    final_lower[0] = basic_lower[0]

    for i in range(1, n):
        # Upper band
        if basic_upper[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]:
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = final_upper[i-1]

        # Lower band
        if basic_lower[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]:
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = final_lower[i-1]

        # Direction and Supertrend
        if i < period:
            direction[i] = 1
            supertrend[i] = final_lower[i]
        else:
            if supertrend[i-1] == final_upper[i-1]:
                if close[i] > final_upper[i]:
                    direction[i] = 1
                    supertrend[i] = final_lower[i]
                else:
                    direction[i] = -1
                    supertrend[i] = final_upper[i]
            else:
                if close[i] < final_lower[i]:
                    direction[i] = -1
                    supertrend[i] = final_upper[i]
                else:
                    direction[i] = 1
                    supertrend[i] = final_lower[i]

    df['supertrend'] = supertrend
    df['st_direction'] = direction
    df['atr'] = atr

    return df


def get_signal(df: pd.DataFrame) -> str:
    """Get current trading signal from Supertrend"""
    if len(df) < 2:
        return "HOLD"

    current_dir = df['st_direction'].iloc[-1]
    prev_dir = df['st_direction'].iloc[-2]

    # Signal on direction change
    if current_dir == 1 and prev_dir == -1:
        return "BUY"
    elif current_dir == -1 and prev_dir == 1:
        return "SELL"
    elif current_dir == 1:
        return "HOLD_LONG"
    else:
        return "HOLD_SHORT"


# =============================================================================
# IB PAPER TRADER
# =============================================================================
class IBPaperTrader:
    """Interactive Brokers Paper Trading System"""

    def __init__(self, config: TradingConfig, dry_run: bool = False):
        self.config = config
        self.dry_run = dry_run
        self.ib: Optional[IB] = None
        self.positions: Dict[str, dict] = {}
        self.signals: Dict[str, str] = {}
        self.historical_data: Dict[str, pd.DataFrame] = {}
        self.trade_log: List[dict] = []

        # Load existing state
        self._load_state()

    def connect(self) -> bool:
        """Connect to IB TWS/Gateway"""
        if not IB_AVAILABLE:
            logger.error("ib_insync not available. Install with: pip install ib_insync")
            return False

        if self.dry_run:
            logger.info("DRY RUN MODE - No actual orders will be placed")
            return True

        try:
            self.ib = IB()
            self.ib.connect(
                self.config.host,
                self.config.port,
                clientId=self.config.client_id
            )
            logger.info(f"Connected to IB at {self.config.host}:{self.config.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to IB: {e}")
            return False

    def disconnect(self):
        """Disconnect from IB"""
        if self.ib and self.ib.isConnected():
            self.ib.disconnect()
            logger.info("Disconnected from IB")

    def get_account_info(self) -> dict:
        """Get account information"""
        if self.dry_run:
            return {
                'NetLiquidation': self.config.initial_capital,
                'AvailableFunds': self.config.initial_capital,
                'BuyingPower': self.config.initial_capital * 4
            }

        if not self.ib:
            return {}

        account_values = self.ib.accountValues()
        info = {}
        for av in account_values:
            if av.tag in ['NetLiquidation', 'AvailableFunds', 'BuyingPower', 'TotalCashValue']:
                info[av.tag] = float(av.value)
        return info

    def get_current_positions(self) -> Dict[str, dict]:
        """Get current positions from IB"""
        if self.dry_run:
            return self.positions

        if not self.ib:
            return {}

        positions = {}
        for pos in self.ib.positions():
            symbol = pos.contract.symbol
            positions[symbol] = {
                'quantity': pos.position,
                'avg_cost': pos.avgCost,
                'market_value': pos.position * pos.avgCost
            }
        return positions

    def fetch_historical_data(self, symbol: str, days: int = 60) -> Optional[pd.DataFrame]:
        """Fetch historical data for a symbol"""
        if self.dry_run:
            # Use yfinance for dry run
            try:
                import yfinance as yf
                end = datetime.now()
                start = end - timedelta(days=days)
                df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
                if df.empty:
                    return None

                # Handle MultiIndex columns (new yfinance format)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0].lower() for col in df.columns]
                else:
                    df.columns = [c.lower() for c in df.columns]

                # Remove adj close if present
                if 'adj close' in df.columns:
                    df = df.drop('adj close', axis=1)
                return df
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
                return None

        if not self.ib:
            return None

        try:
            contract = Stock(symbol, 'SMART', 'USD')
            self.ib.qualifyContracts(contract)

            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr=f'{days} D',
                barSizeSetting='1 day',
                whatToShow='TRADES',
                useRTH=True
            )

            if not bars:
                return None

            df = util.df(bars)
            df.columns = [c.lower() for c in df.columns]
            df.set_index('date', inplace=True)
            return df

        except Exception as e:
            logger.error(f"Error fetching IB data for {symbol}: {e}")
            return None

    def calculate_position_size(self, symbol: str, price: float) -> int:
        """Calculate position size based on risk management rules"""
        account = self.get_account_info()
        equity = account.get('NetLiquidation', self.config.initial_capital)

        # Max position value
        max_position_value = equity * self.config.max_position_pct

        # Calculate shares
        shares = int(max_position_value / price)

        # Ensure minimum of 1 share
        return max(1, shares)

    def place_order(self, symbol: str, action: str, quantity: int, order_type: str = "MKT") -> bool:
        """Place an order"""
        if self.dry_run:
            logger.info(f"[DRY RUN] Would place {action} order for {quantity} shares of {symbol}")

            # Simulate order execution
            if action == "BUY":
                df = self.historical_data.get(symbol)
                if df is not None and len(df) > 0:
                    price = df['close'].iloc[-1]
                    self.positions[symbol] = {
                        'quantity': quantity,
                        'avg_cost': price,
                        'entry_date': datetime.now().isoformat(),
                        'highest_price': price
                    }
            elif action == "SELL" and symbol in self.positions:
                del self.positions[symbol]

            self._save_state()
            return True

        if not self.ib:
            return False

        try:
            contract = Stock(symbol, 'SMART', 'USD')
            self.ib.qualifyContracts(contract)

            if order_type == "MKT":
                order = MarketOrder(action, quantity)
            else:
                # Get current price for limit order
                ticker = self.ib.reqMktData(contract)
                self.ib.sleep(1)
                price = ticker.last if ticker.last else ticker.close
                order = LimitOrder(action, quantity, price)

            trade = self.ib.placeOrder(contract, order)
            self.ib.sleep(2)  # Wait for order to process

            logger.info(f"Placed {action} order for {quantity} shares of {symbol}")

            # Log trade
            self.trade_log.append({
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'order_type': order_type,
                'status': trade.orderStatus.status
            })

            self._save_state()
            return True

        except Exception as e:
            logger.error(f"Error placing order for {symbol}: {e}")
            return False

    def check_stop_loss(self, symbol: str, current_price: float) -> bool:
        """Check if stop loss is triggered"""
        if symbol not in self.positions:
            return False

        pos = self.positions[symbol]
        entry_price = pos['avg_cost']
        highest_price = pos.get('highest_price', entry_price)

        # Update highest price
        if current_price > highest_price:
            self.positions[symbol]['highest_price'] = current_price
            highest_price = current_price

        # Check trailing stop
        trailing_stop_price = highest_price * (1 - self.config.trailing_stop_pct)
        if current_price <= trailing_stop_price:
            logger.warning(f"TRAILING STOP triggered for {symbol}: {current_price:.2f} <= {trailing_stop_price:.2f}")
            return True

        # Check fixed stop loss
        stop_loss_price = entry_price * (1 - self.config.stop_loss_pct)
        if current_price <= stop_loss_price:
            logger.warning(f"STOP LOSS triggered for {symbol}: {current_price:.2f} <= {stop_loss_price:.2f}")
            return True

        return False

    def update_signals(self):
        """Update trading signals for all stocks"""
        logger.info("Updating signals for all stocks...")

        for symbol in TOP_30_STOCKS:
            try:
                # Fetch data
                df = self.fetch_historical_data(symbol, days=60)
                if df is None or len(df) < 20:
                    logger.warning(f"Insufficient data for {symbol}")
                    continue

                # Store data
                self.historical_data[symbol] = df

                # Calculate Supertrend
                df = calculate_supertrend(df, self.config.st_period, self.config.st_multiplier)

                # Get signal
                signal = get_signal(df)
                self.signals[symbol] = signal

                current_price = df['close'].iloc[-1]
                st_value = df['supertrend'].iloc[-1]

                logger.info(f"{symbol}: Price={current_price:.2f}, ST={st_value:.2f}, Signal={signal}")

            except Exception as e:
                logger.error(f"Error updating signal for {symbol}: {e}")

        self._save_state()

    def execute_signals(self):
        """Execute trading signals"""
        logger.info("Executing signals...")

        current_positions = self.get_current_positions()
        num_positions = len(current_positions)

        for symbol, signal in self.signals.items():
            df = self.historical_data.get(symbol)
            if df is None or len(df) == 0:
                continue

            current_price = df['close'].iloc[-1]

            # Check if we have a position
            has_position = symbol in current_positions

            # BUY signal
            if signal == "BUY" and not has_position:
                if num_positions >= self.config.max_positions:
                    logger.info(f"Max positions reached, skipping BUY for {symbol}")
                    continue

                quantity = self.calculate_position_size(symbol, current_price)
                logger.info(f"BUY SIGNAL: {symbol} - {quantity} shares @ ${current_price:.2f}")

                if self.place_order(symbol, "BUY", quantity):
                    num_positions += 1

            # SELL signal or stop loss
            elif has_position:
                should_sell = False

                if signal == "SELL":
                    logger.info(f"SELL SIGNAL: {symbol}")
                    should_sell = True
                elif self.check_stop_loss(symbol, current_price):
                    should_sell = True

                if should_sell:
                    quantity = int(current_positions[symbol]['quantity'])
                    self.place_order(symbol, "SELL", quantity)

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for a symbol from IB"""
        if not self.ib or not self.ib.isConnected():
            # Fallback to historical data
            df = self.historical_data.get(symbol)
            if df is not None and len(df) > 0:
                return float(df['close'].iloc[-1])
            return None

        try:
            contract = Stock(symbol, 'SMART', 'USD')
            self.ib.qualifyContracts(contract)
            ticker = self.ib.reqMktData(contract, '', False, False)
            self.ib.sleep(1)  # Wait for data

            # Try different price fields
            if ticker.last and ticker.last > 0:
                price = ticker.last
            elif ticker.close and ticker.close > 0:
                price = ticker.close
            elif ticker.bid and ticker.ask:
                price = (ticker.bid + ticker.ask) / 2
            else:
                price = None

            self.ib.cancelMktData(contract)
            return price

        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return None

    def show_status(self):
        """Show current portfolio status"""
        print("\n" + "="*80)
        print("SUPERTREND PAPER TRADING STATUS")
        print("="*80)

        # Account info
        account = self.get_account_info()
        print(f"\nAccount Value: ${account.get('NetLiquidation', 0):,.2f}")
        print(f"Available Funds: ${account.get('AvailableFunds', 0):,.2f}")

        # Positions
        positions = self.get_current_positions()
        print(f"\nOpen Positions: {len(positions)}/{self.config.max_positions}")
        print("-"*60)

        total_pnl = 0
        if positions:
            for symbol, pos in positions.items():
                # Get current price from IB
                current_price = self.get_current_price(symbol)
                if current_price is None:
                    current_price = pos['avg_cost']

                entry_price = pos['avg_cost']
                quantity = pos['quantity']

                # Calculate P/L correctly for long and short positions
                if quantity > 0:
                    # Long position
                    pnl_pct = (current_price - entry_price) / entry_price * 100
                    pnl_value = (current_price - entry_price) * quantity
                    pos_type = "LONG"
                else:
                    # Short position (negative quantity)
                    pnl_pct = (entry_price - current_price) / entry_price * 100
                    pnl_value = (entry_price - current_price) * abs(quantity)
                    pos_type = "SHORT"

                total_pnl += pnl_value

                print(f"  {symbol:<6} | {pos_type:<5} | Qty: {quantity:>6.0f} | Entry: ${entry_price:>8.2f} | "
                      f"Current: ${current_price:>8.2f} | P/L: {pnl_pct:>+6.1f}% (${pnl_value:>+,.0f})")

            print("-"*60)
            print(f"  {'TOTAL P/L:':<52} ${total_pnl:>+,.0f}")
        else:
            print("  No open positions")

        # Signals
        print(f"\nCurrent Signals:")
        print("-"*60)

        buy_signals = [s for s, sig in self.signals.items() if sig == "BUY"]
        sell_signals = [s for s, sig in self.signals.items() if sig == "SELL"]

        if buy_signals:
            print(f"  BUY:  {', '.join(buy_signals)}")
        if sell_signals:
            print(f"  SELL: {', '.join(sell_signals)}")

        print("="*80 + "\n")

    def is_market_open(self) -> Tuple[bool, str]:
        """
        Check if US stock market (NYSE/NASDAQ) is open.
        Regular Trading Hours: 9:30 AM - 4:00 PM Eastern Time, Mon-Fri
        Returns: (is_open, status_message)
        """
        try:
            et = ZoneInfo("America/New_York")
            berlin = ZoneInfo("Europe/Berlin")
        except Exception:
            # Fallback if timezone not available
            logger.warning("Could not load timezone, assuming market is open")
            return True, "Timezone unavailable"

        now_et = datetime.now(et)
        now_berlin = datetime.now(berlin)
        weekday = now_et.weekday()  # 0=Monday, 6=Sunday

        # Weekend check
        if weekday >= 5:
            next_open_et = now_et + timedelta(days=(7 - weekday))
            next_open_et = next_open_et.replace(hour=9, minute=30, second=0, microsecond=0)
            next_open_berlin = next_open_et.astimezone(berlin)
            return False, f"Weekend - Market opens {next_open_berlin.strftime('%A %H:%M Berlin')} ({next_open_et.strftime('%H:%M ET')})"

        market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)

        # Convert to Berlin time for display
        market_open_berlin = market_open.astimezone(berlin)
        market_close_berlin = market_close.astimezone(berlin)

        current_time = now_et.time()
        open_time = market_open.time()
        close_time = market_close.time()

        if current_time < open_time:
            mins_until = int((market_open - now_et).total_seconds() / 60)
            return False, f"Pre-market - Opens in {mins_until} min @ {market_open_berlin.strftime('%H:%M Berlin')} ({market_open.strftime('%H:%M ET')})"
        elif current_time >= close_time:
            next_open_et = now_et + timedelta(days=1)
            if next_open_et.weekday() >= 5:
                next_open_et += timedelta(days=(7 - next_open_et.weekday()))
            next_open_et = next_open_et.replace(hour=9, minute=30, second=0, microsecond=0)
            next_open_berlin = next_open_et.astimezone(berlin)
            return False, f"After-hours - Opens {next_open_berlin.strftime('%A %H:%M Berlin')} ({next_open_et.strftime('%H:%M ET')})"
        else:
            mins_until_close = int((market_close - now_et).total_seconds() / 60)
            return True, f"Market OPEN - Closes in {mins_until_close} min @ {market_close_berlin.strftime('%H:%M Berlin')} ({market_close.strftime('%H:%M ET')})"

    def run(self, force: bool = False):
        """Main trading loop"""
        logger.info("Starting Supertrend Paper Trader...")

        if not self.connect():
            logger.error("Could not connect. Exiting.")
            return

        try:
            while True:
                try:
                    # Check market hours
                    is_open, status = self.is_market_open()
                    print(f"\n>>> {status}")

                    if not is_open and not force:
                        logger.info(f"Market closed. Waiting... (use --force to override)")
                        # Show status but don't trade
                        self.show_status()
                        # Wait longer when market is closed (5 minutes)
                        time.sleep(300)
                        continue

                    # Update signals
                    self.update_signals()

                    # Execute signals
                    self.execute_signals()

                    # Show status
                    self.show_status()

                    # Wait for next update
                    logger.info(f"Waiting {self.config.signal_check_interval} seconds for next update...")
                    time.sleep(self.config.signal_check_interval)

                except KeyboardInterrupt:
                    logger.info("Interrupted by user")
                    break
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    time.sleep(60)

        finally:
            self.disconnect()
            self._save_state()

    def _save_state(self):
        """Save current state to file"""
        state = {
            'positions': self.positions,
            'signals': self.signals,
            'trade_log': self.trade_log,
            'last_update': datetime.now().isoformat()
        }
        with open('ib_paper_trader_state.json', 'w') as f:
            json.dump(state, f, indent=2)

    def _load_state(self):
        """Load state from file"""
        try:
            with open('ib_paper_trader_state.json', 'r') as f:
                state = json.load(f)
                self.positions = state.get('positions', {})
                self.signals = state.get('signals', {})
                self.trade_log = state.get('trade_log', [])
        except FileNotFoundError:
            pass


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("="*80)
    print("SUPERTREND IB PAPER TRADING SYSTEM")
    print("="*80)
    print(f"\nTracking {len(TOP_30_STOCKS)} stocks:")
    print(", ".join(TOP_30_STOCKS[:15]))
    print(", ".join(TOP_30_STOCKS[15:]))

    # Parse arguments
    dry_run = "--dry-run" in sys.argv
    status_only = "--status" in sys.argv
    force = "--force" in sys.argv

    if dry_run:
        print("\n*** DRY RUN MODE - No real orders will be placed ***")

    if force:
        print("\n*** FORCE MODE - Trading even outside market hours ***")

    config = TradingConfig()
    trader = IBPaperTrader(config, dry_run=dry_run)

    if status_only:
        trader.connect()
        is_open, status = trader.is_market_open()
        print(f"\n>>> {status}")
        trader.update_signals()
        trader.show_status()
        trader.disconnect()
    else:
        print(f"\nSettings:")
        print(f"  Initial Capital: ${config.initial_capital:,.0f}")
        print(f"  Max Position: {config.max_position_pct*100:.0f}%")
        print(f"  Max Positions: {config.max_positions}")
        print(f"  Stop Loss: {config.stop_loss_pct*100:.0f}%")
        print(f"  Trailing Stop: {config.trailing_stop_pct*100:.0f}%")
        print(f"  Supertrend: Period={config.st_period}, Mult={config.st_multiplier}")
        print(f"  Market Hours: NYSE/NASDAQ 9:30-16:00 ET (Mon-Fri)")
        print(f"\nStarting trader...")

        trader.run(force=force)


if __name__ == "__main__":
    main()
