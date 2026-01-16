#!/usr/bin/env python3
"""
Jahres-Simulation mit detaillierten Trade-Listen und Statistiken
================================================================
Simuliert 1 Jahr Trading mit allen 60 Aktien und erstellt:
- Komplette Trade-Liste (Entry/Exit)
- Per-Stock Performance
- Strategy Performance
- Gesamtstatistiken
- CSV Export
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import logging
import os

# Import indicator functions from ib_paper_trader
from ib_paper_trader import (
    get_signal_for_strategy, get_ticker_strategy,
    get_stocks_by_strategy, load_stock_categories,
    get_ticker_contract_params
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================
INITIAL_CAPITAL = 20000
MAX_POSITIONS = 30
SIMULATION_DAYS = 252  # Trading days in a year
FEE_PER_TRADE = 1.0

CATEGORIES_FILE = "stock_categories.json"
OUTPUT_DIR = "simulation_results"


def get_all_tickers() -> List[str]:
    """Get all active tickers"""
    stocks = get_stocks_by_strategy()
    all_tickers = []
    for tickers in stocks.values():
        all_tickers.extend(tickers)
    return list(set(all_tickers))


def fetch_historical_data_ib(symbols: List[str], days: int = 365) -> Dict[str, pd.DataFrame]:
    """Fetch historical data from Interactive Brokers"""
    try:
        from ib_insync import IB, Stock, util
    except ImportError:
        logger.error("ib_insync not installed, using synthetic data")
        return generate_synthetic_data(symbols, days)

    data = {}
    ib = IB()

    try:
        ib.connect('127.0.0.1', 7497, clientId=52)
        logger.info(f"Connected to IB. Fetching data for {len(symbols)} symbols...")

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
                    logger.info(f"  {symbol}: {len(df)} days")

                ib.sleep(0.5)

            except Exception as e:
                logger.warning(f"  {symbol}: Failed - {e}")

        ib.disconnect()

    except Exception as e:
        logger.error(f"Could not connect to IB: {e}")
        logger.info("Using synthetic data for backtesting...")
        return generate_synthetic_data(symbols, days)

    return data


def generate_synthetic_data(symbols: List[str], days: int = 400) -> Dict[str, pd.DataFrame]:
    """Generate realistic synthetic stock data for backtesting"""
    np.random.seed(42)
    data = {}

    # Extended base prices for all 60 stocks
    base_prices = {
        'NVDA': 500, 'AMD': 140, 'AVGO': 180, 'META': 520, 'TSLA': 250,
        'COIN': 250, 'MSTR': 450, 'PLTR': 70, 'SHOP': 100, 'UBER': 75,
        'CRWD': 350, 'MU': 100, 'JPM': 200, 'INOD': 180, 'QUBT': 15,
        'DRH': 10, 'MRNA': 50, 'MRK': 100, 'NFLX': 900, 'NKE': 75,
        'PFE': 25, 'PYPL': 85, 'PDYN': 40, 'QBTS': 8, 'TKMS': 30,
        'JNJ': 150, 'TGT': 130, 'UNH': 550, 'SPY': 580, 'QQQ': 500,
        'GOOGL': 175, 'AAPL': 190, 'AMZN': 185, 'MSFT': 420, 'CRM': 280,
        'ORCL': 130, 'NOW': 780, 'ADBE': 550, 'PANW': 320, 'SNOW': 180,
        'V': 280, 'MA': 480, 'LLY': 780, 'BABA': 85,
        'SMCI': 600, 'ARM': 140, 'RIVN': 15, 'LCID': 3, 'NIO': 5,
        'SOFI': 10, 'HOOD': 20, 'AFRM': 45, 'UPST': 35, 'DKNG': 40,
        'IWM': 210, 'DIA': 390, 'XLF': 42, 'XLK': 210, 'VTI': 270, 'VOO': 530
    }

    volatility = {
        'NVDA': 0.035, 'AMD': 0.035, 'AVGO': 0.025, 'META': 0.03, 'TSLA': 0.045,
        'COIN': 0.05, 'MSTR': 0.06, 'PLTR': 0.04, 'SHOP': 0.035, 'UBER': 0.03,
        'CRWD': 0.035, 'MU': 0.035, 'JPM': 0.02, 'INOD': 0.04, 'QUBT': 0.08,
        'DRH': 0.025, 'MRNA': 0.045, 'MRK': 0.02, 'NFLX': 0.03, 'NKE': 0.025,
        'PFE': 0.02, 'PYPL': 0.035, 'PDYN': 0.04, 'QBTS': 0.08, 'TKMS': 0.03,
        'JNJ': 0.015, 'TGT': 0.025, 'UNH': 0.02, 'SPY': 0.012, 'QQQ': 0.015,
        'GOOGL': 0.025, 'AAPL': 0.022, 'AMZN': 0.028, 'MSFT': 0.022, 'CRM': 0.03,
        'ORCL': 0.025, 'NOW': 0.03, 'ADBE': 0.028, 'PANW': 0.035, 'SNOW': 0.045,
        'V': 0.018, 'MA': 0.018, 'LLY': 0.025, 'BABA': 0.04,
        'SMCI': 0.06, 'ARM': 0.045, 'RIVN': 0.07, 'LCID': 0.08, 'NIO': 0.065,
        'SOFI': 0.055, 'HOOD': 0.055, 'AFRM': 0.06, 'UPST': 0.065, 'DKNG': 0.045,
        'IWM': 0.015, 'DIA': 0.012, 'XLF': 0.018, 'XLK': 0.016, 'VTI': 0.013, 'VOO': 0.012
    }

    drift = {
        'NVDA': 0.001, 'AMD': 0.0008, 'AVGO': 0.0007, 'META': 0.0006, 'TSLA': 0.0005,
        'COIN': 0.0003, 'MSTR': 0.0002, 'PLTR': 0.0008, 'SHOP': 0.0004, 'UBER': 0.0005,
        'CRWD': 0.0006, 'MU': 0.0005, 'JPM': 0.0004, 'INOD': 0.0007, 'QUBT': 0.001,
        'DRH': 0.0003, 'MRNA': -0.0002, 'MRK': 0.0002, 'NFLX': 0.0005, 'NKE': 0.0001,
        'PFE': -0.0001, 'PYPL': 0.0003, 'PDYN': 0.0004, 'QBTS': 0.0008, 'TKMS': 0.0003,
        'JNJ': 0.0002, 'TGT': 0.0002, 'UNH': 0.0003, 'SPY': 0.0004, 'QQQ': 0.0005,
        'GOOGL': 0.0005, 'AAPL': 0.0004, 'AMZN': 0.0005, 'MSFT': 0.0005, 'CRM': 0.0004,
        'ORCL': 0.0004, 'NOW': 0.0006, 'ADBE': 0.0004, 'PANW': 0.0005, 'SNOW': 0.0003,
        'V': 0.0003, 'MA': 0.0003, 'LLY': 0.0006, 'BABA': 0.0001,
        'SMCI': 0.0008, 'ARM': 0.0007, 'RIVN': -0.0002, 'LCID': -0.0003, 'NIO': -0.0002,
        'SOFI': 0.0004, 'HOOD': 0.0005, 'AFRM': 0.0003, 'UPST': 0.0002, 'DKNG': 0.0004,
        'IWM': 0.0003, 'DIA': 0.0003, 'XLF': 0.0003, 'XLK': 0.0004, 'VTI': 0.0004, 'VOO': 0.0004
    }

    end_date = datetime.now()
    dates = pd.date_range(end=end_date, periods=days, freq='B')

    logger.info(f"Generating synthetic data for {len(symbols)} symbols...")

    for symbol in symbols:
        start_price = base_prices.get(symbol, 100)
        vol = volatility.get(symbol, 0.03)
        mu = drift.get(symbol, 0.0003)

        returns = np.random.normal(mu, vol, days)
        prices = start_price * np.exp(np.cumsum(returns))

        df = pd.DataFrame(index=dates)
        df['close'] = prices

        daily_range = vol * 0.5
        df['high'] = df['close'] * (1 + np.abs(np.random.normal(0, daily_range, days)))
        df['low'] = df['close'] * (1 - np.abs(np.random.normal(0, daily_range, days)))
        df['open'] = df['close'].shift(1).fillna(start_price) * (1 + np.random.normal(0, vol*0.3, days))

        df['high'] = df[['high', 'close', 'open']].max(axis=1)
        df['low'] = df[['low', 'close', 'open']].min(axis=1)
        df['volume'] = np.random.randint(1000000, 10000000, days)

        data[symbol] = df

    logger.info(f"Generated synthetic data for {len(data)} symbols")
    return data


# =============================================================================
# TRADE TRACKING
# =============================================================================
class Trade:
    """Represents a complete round-trip trade"""
    def __init__(self, symbol: str, strategy: str, entry_date: str, entry_price: float, quantity: int):
        self.symbol = symbol
        self.strategy = strategy
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.quantity = quantity
        self.exit_date = None
        self.exit_price = None
        self.exit_reason = None
        self.high_price = entry_price

    def close(self, exit_date: str, exit_price: float, reason: str = "SIGNAL"):
        self.exit_date = exit_date
        self.exit_price = exit_price
        self.exit_reason = reason

    @property
    def is_closed(self) -> bool:
        return self.exit_date is not None

    @property
    def pnl(self) -> float:
        if not self.is_closed:
            return 0
        return (self.exit_price - self.entry_price) * self.quantity - 2 * FEE_PER_TRADE

    @property
    def pnl_pct(self) -> float:
        if not self.is_closed or self.entry_price == 0:
            return 0
        return ((self.exit_price - self.entry_price) / self.entry_price) * 100

    @property
    def holding_days(self) -> int:
        if not self.is_closed:
            return 0
        entry = pd.to_datetime(self.entry_date)
        exit = pd.to_datetime(self.exit_date)
        return (exit - entry).days

    def to_dict(self) -> dict:
        return {
            'Symbol': self.symbol,
            'Strategy': self.strategy,
            'Entry Date': self.entry_date,
            'Entry Price': f"${self.entry_price:.2f}",
            'Quantity': self.quantity,
            'Exit Date': self.exit_date or '-',
            'Exit Price': f"${self.exit_price:.2f}" if self.exit_price else '-',
            'Exit Reason': self.exit_reason or '-',
            'P&L': f"${self.pnl:+.2f}" if self.is_closed else '-',
            'P&L %': f"{self.pnl_pct:+.1f}%" if self.is_closed else '-',
            'Days': self.holding_days if self.is_closed else '-'
        }


class Position:
    def __init__(self, symbol: str, entry_price: float, quantity: int, entry_date: str, strategy: str):
        self.symbol = symbol
        self.entry_price = entry_price
        self.quantity = quantity
        self.entry_date = entry_date
        self.strategy = strategy
        self.current_price = entry_price
        self.high_price = entry_price

    def update_price(self, price: float):
        self.current_price = price
        self.high_price = max(self.high_price, price)

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def cost_basis(self) -> float:
        return self.quantity * self.entry_price

    @property
    def unrealized_pnl(self) -> float:
        return self.market_value - self.cost_basis


class SimulationEngine:
    def __init__(self, initial_capital: float, max_positions: int):
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve = []
        self.daily_returns = []
        self.total_fees = 0

    @property
    def equity(self) -> float:
        return self.cash + sum(p.market_value for p in self.positions.values())

    def buy(self, symbol: str, price: float, date: str, strategy: str) -> bool:
        if symbol in self.positions or len(self.positions) >= self.max_positions:
            return False

        target_value = self.equity / self.max_positions
        quantity = int(target_value / price)
        if quantity <= 0:
            return False

        cost = quantity * price + FEE_PER_TRADE
        if cost > self.cash:
            quantity = int((self.cash - FEE_PER_TRADE) / price)
            if quantity <= 0:
                return False
            cost = quantity * price + FEE_PER_TRADE

        self.cash -= cost
        self.positions[symbol] = Position(symbol, price, quantity, date, strategy)
        self.total_fees += FEE_PER_TRADE

        # Create trade record
        trade = Trade(symbol, strategy, date, price, quantity)
        self.trades.append(trade)

        return True

    def sell(self, symbol: str, price: float, date: str, reason: str = "SIGNAL") -> Optional[float]:
        if symbol not in self.positions:
            return None

        pos = self.positions[symbol]
        proceeds = pos.quantity * price - FEE_PER_TRADE
        pnl = proceeds - pos.cost_basis

        self.cash += proceeds
        self.total_fees += FEE_PER_TRADE

        # Close the trade
        for trade in reversed(self.trades):
            if trade.symbol == symbol and not trade.is_closed:
                trade.high_price = pos.high_price
                trade.close(date, price, reason)
                break

        del self.positions[symbol]
        return pnl

    def update_prices(self, prices: Dict[str, float]):
        for symbol, pos in self.positions.items():
            if symbol in prices:
                pos.update_price(prices[symbol])

    def record_equity(self, date: str):
        self.equity_curve.append({
            'date': date,
            'equity': self.equity,
            'cash': self.cash,
            'num_positions': len(self.positions)
        })

        if len(self.equity_curve) > 1:
            prev_eq = self.equity_curve[-2]['equity']
            curr_eq = self.equity
            daily_ret = (curr_eq - prev_eq) / prev_eq if prev_eq > 0 else 0
            self.daily_returns.append(daily_ret)


def run_simulation(data: Dict[str, pd.DataFrame]) -> SimulationEngine:
    """Run full simulation following signals per stock strategy"""
    engine = SimulationEngine(INITIAL_CAPITAL, MAX_POSITIONS)

    # Load strategy settings for trailing stops
    categories = load_stock_categories()
    strategy_settings = {}
    for strat_name, strat_data in categories.get('strategies', {}).items():
        strategy_settings[strat_name] = strat_data.get('settings', {})

    # Get common dates
    all_dates = None
    for symbol, df in data.items():
        dates = set(df.index)
        all_dates = dates if all_dates is None else all_dates.intersection(dates)

    if not all_dates:
        logger.error("No common dates found")
        return engine

    sorted_dates = sorted(all_dates)
    if len(sorted_dates) > SIMULATION_DAYS:
        sorted_dates = sorted_dates[-SIMULATION_DAYS:]

    # Get strategy per ticker
    ticker_strategies = {symbol: get_ticker_strategy(symbol) for symbol in data.keys()}

    logger.info(f"Running simulation: {len(sorted_dates)} trading days")

    for i, date in enumerate(sorted_dates):
        date_str = str(date)[:10]

        # Get current prices
        current_prices = {}
        for symbol, df in data.items():
            if date in df.index:
                current_prices[symbol] = df.loc[date, 'close']

        # Update position prices
        engine.update_prices(current_prices)

        # Check trailing stops first
        positions_to_close = []
        for symbol, pos in engine.positions.items():
            strat = pos.strategy
            settings = strategy_settings.get(strat, {})
            trailing_stop_pct = settings.get('trailing_stop_pct', 0.12)

            if pos.high_price > 0:
                stop_price = pos.high_price * (1 - trailing_stop_pct)
                if pos.current_price <= stop_price:
                    positions_to_close.append((symbol, "TRAILING_STOP"))

        for symbol, reason in positions_to_close:
            price = current_prices.get(symbol, engine.positions[symbol].current_price)
            engine.sell(symbol, price, date_str, reason)

        # Check signals for each stock
        for symbol in data.keys():
            if symbol not in data or date not in data[symbol].index:
                continue

            df_slice = data[symbol].loc[:date].copy()
            if len(df_slice) < 60:
                continue

            strat = ticker_strategies.get(symbol, "SUPERTREND")
            signal = get_signal_for_strategy(df_slice, strat)
            price = current_prices.get(symbol)

            if price is None:
                continue

            has_position = symbol in engine.positions

            # BUY signal and no position -> Buy
            if signal == "BUY" and not has_position:
                engine.buy(symbol, price, date_str, strat)

            # SELL signal and has position -> Sell
            elif signal == "SELL" and has_position:
                engine.sell(symbol, price, date_str, "SIGNAL")

        engine.record_equity(date_str)

    return engine


# =============================================================================
# STATISTICS
# =============================================================================
def calculate_statistics(engine: SimulationEngine) -> dict:
    """Calculate comprehensive statistics"""
    stats = {}

    # Basic stats
    stats['initial_capital'] = engine.initial_capital
    stats['final_equity'] = engine.equity
    stats['total_return'] = engine.equity - engine.initial_capital
    stats['total_return_pct'] = ((engine.equity / engine.initial_capital) - 1) * 100
    stats['total_fees'] = engine.total_fees

    # Trade stats
    closed_trades = [t for t in engine.trades if t.is_closed]
    stats['total_trades'] = len(closed_trades)

    if closed_trades:
        winners = [t for t in closed_trades if t.pnl > 0]
        losers = [t for t in closed_trades if t.pnl <= 0]

        stats['winning_trades'] = len(winners)
        stats['losing_trades'] = len(losers)
        stats['win_rate'] = (len(winners) / len(closed_trades)) * 100

        stats['total_profit'] = sum(t.pnl for t in winners)
        stats['total_loss'] = sum(t.pnl for t in losers)

        stats['avg_win'] = np.mean([t.pnl for t in winners]) if winners else 0
        stats['avg_loss'] = np.mean([t.pnl for t in losers]) if losers else 0
        stats['avg_win_pct'] = np.mean([t.pnl_pct for t in winners]) if winners else 0
        stats['avg_loss_pct'] = np.mean([t.pnl_pct for t in losers]) if losers else 0

        stats['largest_win'] = max(t.pnl for t in winners) if winners else 0
        stats['largest_loss'] = min(t.pnl for t in losers) if losers else 0

        stats['avg_holding_days'] = np.mean([t.holding_days for t in closed_trades])

        # Profit factor
        if stats['total_loss'] < 0:
            stats['profit_factor'] = abs(stats['total_profit'] / stats['total_loss'])
        else:
            stats['profit_factor'] = float('inf') if stats['total_profit'] > 0 else 0
    else:
        stats['winning_trades'] = 0
        stats['losing_trades'] = 0
        stats['win_rate'] = 0
        stats['total_profit'] = 0
        stats['total_loss'] = 0
        stats['avg_win'] = 0
        stats['avg_loss'] = 0
        stats['profit_factor'] = 0
        stats['avg_holding_days'] = 0

    # Risk metrics
    if engine.equity_curve:
        equity_values = [e['equity'] for e in engine.equity_curve]

        # Max drawdown
        peak = equity_values[0]
        max_dd = 0
        max_dd_start = 0
        max_dd_end = 0
        dd_start = 0

        for i, val in enumerate(equity_values):
            if val > peak:
                peak = val
                dd_start = i
            dd = (peak - val) / peak
            if dd > max_dd:
                max_dd = dd
                max_dd_start = dd_start
                max_dd_end = i

        stats['max_drawdown_pct'] = max_dd * 100

        # Sharpe ratio (annualized)
        if engine.daily_returns:
            returns = np.array(engine.daily_returns)
            if returns.std() > 0:
                stats['sharpe_ratio'] = np.sqrt(252) * returns.mean() / returns.std()
            else:
                stats['sharpe_ratio'] = 0

            # Sortino ratio (downside deviation)
            downside = returns[returns < 0]
            if len(downside) > 0 and downside.std() > 0:
                stats['sortino_ratio'] = np.sqrt(252) * returns.mean() / downside.std()
            else:
                stats['sortino_ratio'] = 0
        else:
            stats['sharpe_ratio'] = 0
            stats['sortino_ratio'] = 0
    else:
        stats['max_drawdown_pct'] = 0
        stats['sharpe_ratio'] = 0
        stats['sortino_ratio'] = 0

    # Open positions
    stats['open_positions'] = len(engine.positions)
    stats['unrealized_pnl'] = sum(p.unrealized_pnl for p in engine.positions.values())

    return stats


def calculate_strategy_stats(engine: SimulationEngine) -> Dict[str, dict]:
    """Calculate statistics per strategy"""
    strategy_stats = {}

    for trade in engine.trades:
        strat = trade.strategy
        if strat not in strategy_stats:
            strategy_stats[strat] = {
                'trades': [], 'pnl': 0, 'winners': 0, 'losers': 0
            }

        if trade.is_closed:
            strategy_stats[strat]['trades'].append(trade)
            strategy_stats[strat]['pnl'] += trade.pnl
            if trade.pnl > 0:
                strategy_stats[strat]['winners'] += 1
            else:
                strategy_stats[strat]['losers'] += 1

    # Calculate win rates
    for strat, data in strategy_stats.items():
        total = data['winners'] + data['losers']
        data['win_rate'] = (data['winners'] / total * 100) if total > 0 else 0
        data['total_trades'] = total

    return strategy_stats


def calculate_stock_stats(engine: SimulationEngine) -> Dict[str, dict]:
    """Calculate statistics per stock"""
    stock_stats = {}

    for trade in engine.trades:
        symbol = trade.symbol
        if symbol not in stock_stats:
            stock_stats[symbol] = {
                'strategy': trade.strategy,
                'trades': 0, 'pnl': 0, 'winners': 0, 'losers': 0
            }

        if trade.is_closed:
            stock_stats[symbol]['trades'] += 1
            stock_stats[symbol]['pnl'] += trade.pnl
            if trade.pnl > 0:
                stock_stats[symbol]['winners'] += 1
            else:
                stock_stats[symbol]['losers'] += 1

    # Calculate win rates
    for symbol, data in stock_stats.items():
        total = data['winners'] + data['losers']
        data['win_rate'] = (data['winners'] / total * 100) if total > 0 else 0

    return stock_stats


# =============================================================================
# REPORTING
# =============================================================================
def print_report(engine: SimulationEngine, stats: dict, strategy_stats: dict, stock_stats: dict):
    """Print comprehensive report"""
    print("\n" + "="*80)
    print("                    JAHRES-SIMULATION REPORT")
    print("="*80)
    print(f"  Zeitraum: {SIMULATION_DAYS} Handelstage (~1 Jahr)")
    print(f"  Datum: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # OVERALL PERFORMANCE
    print("\n" + "-"*80)
    print("  GESAMTPERFORMANCE")
    print("-"*80)
    print(f"  Startkapital:        ${stats['initial_capital']:>12,.2f}")
    print(f"  Endkapital:          ${stats['final_equity']:>12,.2f}")
    print(f"  Gewinn/Verlust:      ${stats['total_return']:>+12,.2f}  ({stats['total_return_pct']:+.1f}%)")
    print(f"  Gebühren:            ${stats['total_fees']:>12,.2f}")

    # TRADE STATISTICS
    print("\n" + "-"*80)
    print("  TRADE STATISTIKEN")
    print("-"*80)
    print(f"  Abgeschlossene Trades:  {stats['total_trades']:>8}")
    print(f"  Gewinner:               {stats['winning_trades']:>8}  ({stats['win_rate']:.1f}%)")
    print(f"  Verlierer:              {stats['losing_trades']:>8}  ({100-stats['win_rate']:.1f}%)")
    print(f"  ")
    print(f"  Gesamtgewinn:        ${stats['total_profit']:>12,.2f}")
    print(f"  Gesamtverlust:       ${stats['total_loss']:>12,.2f}")
    print(f"  Profit Factor:          {stats['profit_factor']:>8.2f}")
    print(f"  ")
    print(f"  Ø Gewinn pro Trade:  ${stats['avg_win']:>12,.2f}  ({stats.get('avg_win_pct', 0):+.1f}%)")
    print(f"  Ø Verlust pro Trade: ${stats['avg_loss']:>12,.2f}  ({stats.get('avg_loss_pct', 0):+.1f}%)")
    print(f"  Größter Gewinn:      ${stats.get('largest_win', 0):>12,.2f}")
    print(f"  Größter Verlust:     ${stats.get('largest_loss', 0):>12,.2f}")
    print(f"  Ø Haltedauer:           {stats['avg_holding_days']:>8.1f} Tage")

    # RISK METRICS
    print("\n" + "-"*80)
    print("  RISIKO-KENNZAHLEN")
    print("-"*80)
    print(f"  Max Drawdown:           {stats['max_drawdown_pct']:>8.1f}%")
    print(f"  Sharpe Ratio:           {stats['sharpe_ratio']:>8.2f}")
    print(f"  Sortino Ratio:          {stats['sortino_ratio']:>8.2f}")

    # STRATEGY PERFORMANCE
    print("\n" + "-"*80)
    print("  PERFORMANCE NACH STRATEGIE")
    print("-"*80)
    print(f"  {'Strategie':<15} {'Trades':>8} {'Gewinner':>10} {'Win Rate':>10} {'P&L':>14}")
    print(f"  {'-'*15} {'-'*8} {'-'*10} {'-'*10} {'-'*14}")
    for strat, data in sorted(strategy_stats.items(), key=lambda x: x[1]['pnl'], reverse=True):
        print(f"  {strat:<15} {data['total_trades']:>8} {data['winners']:>10} {data['win_rate']:>9.1f}% ${data['pnl']:>+12,.2f}")

    # TOP/BOTTOM STOCKS
    print("\n" + "-"*80)
    print("  TOP 10 AKTIEN (nach P&L)")
    print("-"*80)
    sorted_stocks = sorted(stock_stats.items(), key=lambda x: x[1]['pnl'], reverse=True)
    print(f"  {'Symbol':<8} {'Strategie':<15} {'Trades':>7} {'Win%':>7} {'P&L':>14}")
    print(f"  {'-'*8} {'-'*15} {'-'*7} {'-'*7} {'-'*14}")
    for symbol, data in sorted_stocks[:10]:
        print(f"  {symbol:<8} {data['strategy']:<15} {data['trades']:>7} {data['win_rate']:>6.0f}% ${data['pnl']:>+12,.2f}")

    print("\n  BOTTOM 10 AKTIEN (nach P&L)")
    print(f"  {'-'*8} {'-'*15} {'-'*7} {'-'*7} {'-'*14}")
    for symbol, data in sorted_stocks[-10:]:
        print(f"  {symbol:<8} {data['strategy']:<15} {data['trades']:>7} {data['win_rate']:>6.0f}% ${data['pnl']:>+12,.2f}")

    # OPEN POSITIONS
    if engine.positions:
        print("\n" + "-"*80)
        print("  OFFENE POSITIONEN")
        print("-"*80)
        print(f"  {'Symbol':<8} {'Strategie':<15} {'Entry':>10} {'Current':>10} {'P&L':>12} {'P&L%':>8}")
        print(f"  {'-'*8} {'-'*15} {'-'*10} {'-'*10} {'-'*12} {'-'*8}")
        for symbol, pos in sorted(engine.positions.items(), key=lambda x: x[1].unrealized_pnl, reverse=True):
            pnl_pct = ((pos.current_price / pos.entry_price) - 1) * 100
            print(f"  {symbol:<8} {pos.strategy:<15} ${pos.entry_price:>9.2f} ${pos.current_price:>9.2f} ${pos.unrealized_pnl:>+10,.2f} {pnl_pct:>+7.1f}%")
        print(f"\n  Unrealisierter P&L: ${stats['unrealized_pnl']:>+,.2f}")

    print("\n" + "="*80)


def export_to_csv(engine: SimulationEngine, stats: dict, strategy_stats: dict, stock_stats: dict):
    """Export results to CSV files"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 1. Trade List
    if engine.trades:
        trades_data = []
        for t in engine.trades:
            trades_data.append({
                'Symbol': t.symbol,
                'Strategy': t.strategy,
                'Entry_Date': t.entry_date,
                'Entry_Price': t.entry_price,
                'Quantity': t.quantity,
                'Exit_Date': t.exit_date or '',
                'Exit_Price': t.exit_price or '',
                'Exit_Reason': t.exit_reason or '',
                'PnL': t.pnl if t.is_closed else '',
                'PnL_Pct': t.pnl_pct if t.is_closed else '',
                'Holding_Days': t.holding_days if t.is_closed else ''
            })
        df_trades = pd.DataFrame(trades_data)
        trades_file = f"{OUTPUT_DIR}/trades_{timestamp}.csv"
        df_trades.to_csv(trades_file, index=False)
        print(f"  Trade-Liste gespeichert: {trades_file}")

    # 2. Equity Curve
    if engine.equity_curve:
        df_equity = pd.DataFrame(engine.equity_curve)
        equity_file = f"{OUTPUT_DIR}/equity_curve_{timestamp}.csv"
        df_equity.to_csv(equity_file, index=False)
        print(f"  Equity-Kurve gespeichert: {equity_file}")

    # 3. Stock Performance
    stock_data = []
    for symbol, data in stock_stats.items():
        stock_data.append({
            'Symbol': symbol,
            'Strategy': data['strategy'],
            'Trades': data['trades'],
            'Winners': data['winners'],
            'Losers': data['losers'],
            'Win_Rate': data['win_rate'],
            'PnL': data['pnl']
        })
    df_stocks = pd.DataFrame(stock_data)
    df_stocks = df_stocks.sort_values('PnL', ascending=False)
    stocks_file = f"{OUTPUT_DIR}/stock_performance_{timestamp}.csv"
    df_stocks.to_csv(stocks_file, index=False)
    print(f"  Aktien-Performance gespeichert: {stocks_file}")

    # 4. Summary
    summary_file = f"{OUTPUT_DIR}/summary_{timestamp}.txt"
    with open(summary_file, 'w') as f:
        f.write("SIMULATION SUMMARY\n")
        f.write("="*50 + "\n\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    print(f"  Zusammenfassung gespeichert: {summary_file}")


def print_trade_list(engine: SimulationEngine, limit: int = 50):
    """Print detailed trade list"""
    closed_trades = [t for t in engine.trades if t.is_closed]

    print("\n" + "="*100)
    print("  TRADE-LISTE (letzte {} Trades)".format(min(limit, len(closed_trades))))
    print("="*100)
    print(f"  {'#':>3} {'Symbol':<8} {'Strategie':<12} {'Entry':<12} {'Exit':<12} {'Entry$':>9} {'Exit$':>9} {'P&L':>10} {'%':>7} {'Reason':<10}")
    print(f"  {'-'*3} {'-'*8} {'-'*12} {'-'*12} {'-'*12} {'-'*9} {'-'*9} {'-'*10} {'-'*7} {'-'*10}")

    # Sort by exit date
    sorted_trades = sorted(closed_trades, key=lambda t: t.exit_date or '', reverse=True)

    for i, t in enumerate(sorted_trades[:limit], 1):
        pnl_color = "+" if t.pnl > 0 else ""
        print(f"  {i:>3} {t.symbol:<8} {t.strategy:<12} {t.entry_date:<12} {t.exit_date:<12} ${t.entry_price:>8.2f} ${t.exit_price:>8.2f} ${t.pnl:>+9.2f} {t.pnl_pct:>+6.1f}% {t.exit_reason:<10}")

    print("="*100)


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("="*80)
    print("         JAHRES-SIMULATION MIT TRADE-LISTEN UND STATISTIKEN")
    print("="*80)

    # Show strategy assignments
    stocks_by_strat = get_stocks_by_strategy()
    print("\nStrategien:")
    total_stocks = 0
    for strat, tickers in stocks_by_strat.items():
        total_stocks += len(tickers)
        print(f"  {strat} ({len(tickers)}): {', '.join(tickers[:8])}{'...' if len(tickers) > 8 else ''}")

    print(f"\nKonfiguration:")
    print(f"  Startkapital:    ${INITIAL_CAPITAL:,}")
    print(f"  Max Positionen:  {MAX_POSITIONS}")
    print(f"  Handelstage:     {SIMULATION_DAYS} (~1 Jahr)")
    print(f"  Aktien-Universum: {total_stocks} Symbole")

    tickers = get_all_tickers()

    print("\n" + "-"*80)
    print("  LOADING HISTORICAL DATA...")
    print("-"*80)

    data = fetch_historical_data_ib(tickers, days=400)

    if len(data) < 10:
        print("  Using synthetic data (IB not connected)")
        data = generate_synthetic_data(tickers, days=400)

    print(f"\n  Loaded data for {len(data)} symbols")

    print("\n" + "-"*80)
    print("  RUNNING SIMULATION...")
    print("-"*80)

    engine = run_simulation(data)

    # Calculate statistics
    stats = calculate_statistics(engine)
    strategy_stats = calculate_strategy_stats(engine)
    stock_stats = calculate_stock_stats(engine)

    # Print report
    print_report(engine, stats, strategy_stats, stock_stats)

    # Print trade list
    print_trade_list(engine)

    # Export to CSV
    print("\n" + "-"*80)
    print("  EXPORTING RESULTS...")
    print("-"*80)
    export_to_csv(engine, stats, strategy_stats, stock_stats)

    print("\n" + "="*80)
    print("  SIMULATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
