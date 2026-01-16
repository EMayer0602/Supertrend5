#!/usr/bin/env python3
"""
Rotation Strategy Backtester
============================
Simulates a daily rotation strategy over 1 year:
- Start with 30 positions
- Daily: Sell 3 worst performers, buy 3 new with best signals
- Compare: Worst by Daily PnL vs Worst by Total PnL

Uses historical data from IB or Yahoo Finance.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================
INITIAL_CAPITAL = 20000
MAX_POSITIONS = 30
DAILY_ROTATION = 3  # Sell 3 worst, buy 3 new
SIMULATION_DAYS = 252  # 1 trading year
FEE_PER_TRADE = 1.0  # $1 per trade

CATEGORIES_FILE = "stock_categories.json"


# =============================================================================
# DATA LOADING
# =============================================================================
def load_stock_categories() -> Dict[str, List[str]]:
    """Load stocks from categories file"""
    try:
        with open(CATEGORIES_FILE, 'r') as f:
            config = json.load(f)

        stocks = {}
        for strategy, data in config.get('strategies', {}).items():
            if strategy != 'EXCLUDED':
                stocks[strategy] = data.get('tickers', [])
        return stocks
    except FileNotFoundError:
        logger.error(f"{CATEGORIES_FILE} not found")
        return {}


def get_all_tickers() -> List[str]:
    """Get all active tickers"""
    categories = load_stock_categories()
    all_tickers = []
    for tickers in categories.values():
        all_tickers.extend(tickers)
    return list(set(all_tickers))


def fetch_historical_data_yf(symbols: List[str], days: int = 400) -> Dict[str, pd.DataFrame]:
    """Fetch historical data from Yahoo Finance"""
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed. Run: pip install yfinance")
        return {}

    data = {}
    end = datetime.now()
    start = end - timedelta(days=days)

    logger.info(f"Fetching data for {len(symbols)} symbols from Yahoo Finance...")

    for symbol in symbols:
        try:
            df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0].lower() for col in df.columns]
                else:
                    df.columns = [c.lower() for c in df.columns]
                data[symbol] = df
                logger.info(f"  {symbol}: {len(df)} days")
        except Exception as e:
            logger.warning(f"  {symbol}: Failed - {e}")

    return data


def generate_synthetic_data(symbols: List[str], days: int = 400) -> Dict[str, pd.DataFrame]:
    """Generate realistic synthetic stock data for backtesting"""
    np.random.seed(42)  # For reproducibility
    data = {}

    # Realistic starting prices for different stock types
    base_prices = {
        'NVDA': 500, 'AMD': 140, 'AVGO': 180, 'META': 520, 'TSLA': 250,
        'COIN': 250, 'MSTR': 450, 'PLTR': 70, 'SHOP': 100, 'UBER': 75,
        'CRWD': 350, 'MU': 100, 'JPM': 200, 'INOD': 180, 'QUBT': 15,
        'DRH': 10, 'MRNA': 50, 'MRK': 100, 'NFLX': 900, 'NKE': 75,
        'PFE': 25, 'PYPL': 85, 'PDYN': 40, 'QBTS': 8, 'TKMS': 30,
        'JNJ': 150, 'TGT': 130, 'UNH': 550, 'SPY': 580, 'QQQ': 500
    }

    # Daily volatility estimates (higher = more volatile)
    volatility = {
        'NVDA': 0.035, 'AMD': 0.035, 'AVGO': 0.025, 'META': 0.03, 'TSLA': 0.045,
        'COIN': 0.05, 'MSTR': 0.06, 'PLTR': 0.04, 'SHOP': 0.035, 'UBER': 0.03,
        'CRWD': 0.035, 'MU': 0.035, 'JPM': 0.02, 'INOD': 0.04, 'QUBT': 0.08,
        'DRH': 0.025, 'MRNA': 0.045, 'MRK': 0.02, 'NFLX': 0.03, 'NKE': 0.025,
        'PFE': 0.02, 'PYPL': 0.035, 'PDYN': 0.04, 'QBTS': 0.08, 'TKMS': 0.03,
        'JNJ': 0.015, 'TGT': 0.025, 'UNH': 0.02, 'SPY': 0.012, 'QQQ': 0.015
    }

    # Daily drift (expected return, positive = bullish)
    drift = {
        'NVDA': 0.001, 'AMD': 0.0008, 'AVGO': 0.0007, 'META': 0.0006, 'TSLA': 0.0005,
        'COIN': 0.0003, 'MSTR': 0.0002, 'PLTR': 0.0008, 'SHOP': 0.0004, 'UBER': 0.0005,
        'CRWD': 0.0006, 'MU': 0.0005, 'JPM': 0.0004, 'INOD': 0.0007, 'QUBT': 0.001,
        'DRH': 0.0003, 'MRNA': -0.0002, 'MRK': 0.0002, 'NFLX': 0.0005, 'NKE': 0.0001,
        'PFE': -0.0001, 'PYPL': 0.0003, 'PDYN': 0.0004, 'QBTS': 0.0008, 'TKMS': 0.0003,
        'JNJ': 0.0002, 'TGT': 0.0002, 'UNH': 0.0003, 'SPY': 0.0004, 'QQQ': 0.0005
    }

    end_date = datetime.now()
    dates = pd.date_range(end=end_date, periods=days, freq='B')  # Business days

    logger.info(f"Generating synthetic data for {len(symbols)} symbols...")

    for symbol in symbols:
        start_price = base_prices.get(symbol, 100)
        vol = volatility.get(symbol, 0.03)
        mu = drift.get(symbol, 0.0003)

        # Generate price path using geometric Brownian motion
        returns = np.random.normal(mu, vol, days)
        prices = start_price * np.exp(np.cumsum(returns))

        # Generate OHLC from close prices
        df = pd.DataFrame(index=dates)
        df['close'] = prices

        # Generate realistic OHLC
        daily_range = vol * 0.5
        df['high'] = df['close'] * (1 + np.abs(np.random.normal(0, daily_range, days)))
        df['low'] = df['close'] * (1 - np.abs(np.random.normal(0, daily_range, days)))
        df['open'] = df['close'].shift(1).fillna(start_price) * (1 + np.random.normal(0, vol*0.3, days))

        # Ensure high >= close and low <= close
        df['high'] = df[['high', 'close', 'open']].max(axis=1)
        df['low'] = df[['low', 'close', 'open']].min(axis=1)

        df['volume'] = np.random.randint(1000000, 10000000, days)

        data[symbol] = df
        logger.info(f"  {symbol}: {len(df)} days (synthetic)")

    return data


def fetch_historical_data_ib(symbols: List[str], days: int = 400) -> Dict[str, pd.DataFrame]:
    """Fetch historical data from Interactive Brokers"""
    try:
        from ib_insync import IB, Stock, util
    except ImportError:
        logger.error("ib_insync not installed, using synthetic data")
        return generate_synthetic_data(symbols, days)

    data = {}
    ib = IB()

    try:
        ib.connect('127.0.0.1', 7497, clientId=50)
        logger.info(f"Connected to IB. Fetching data for {len(symbols)} symbols...")

        for symbol in symbols:
            try:
                # Skip EUR stocks for simplicity
                contract = Stock(symbol, 'SMART', 'USD')
                ib.qualifyContracts(contract)

                bars = ib.reqHistoricalData(
                    contract,
                    endDateTime='',
                    durationStr=f'{days} D',
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

                ib.sleep(0.5)  # Rate limiting

            except Exception as e:
                logger.warning(f"  {symbol}: Failed - {e}")

        ib.disconnect()

    except Exception as e:
        logger.error(f"Could not connect to IB: {e}")
        logger.info("Using synthetic data for backtesting...")
        return generate_synthetic_data(symbols, days)

    return data


# =============================================================================
# SUPERTREND INDICATOR
# =============================================================================
def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]

    atr = np.zeros(len(tr))
    atr[period-1] = np.mean(tr[:period])

    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period

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
        if basic_upper[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]:
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = final_upper[i-1]

        if basic_lower[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]:
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = final_lower[i-1]

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

    df = df.copy()
    df['supertrend'] = supertrend
    df['st_direction'] = direction
    df['signal'] = np.where(direction == 1, 'BUY', 'SELL')

    return df


# =============================================================================
# BACKTEST ENGINE
# =============================================================================
class Position:
    """Represents an open position"""
    def __init__(self, symbol: str, entry_price: float, quantity: int, entry_date: str):
        self.symbol = symbol
        self.entry_price = entry_price
        self.quantity = quantity
        self.entry_date = entry_date
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

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.cost_basis == 0:
            return 0
        return (self.unrealized_pnl / self.cost_basis) * 100


class BacktestEngine:
    """Backtesting engine for rotation strategy"""

    def __init__(self, initial_capital: float, max_positions: int, daily_rotation: int):
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.daily_rotation = daily_rotation

        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trade_history = []
        self.equity_curve = []

        self.total_fees = 0
        self.total_trades = 0

    @property
    def equity(self) -> float:
        positions_value = sum(p.market_value for p in self.positions.values())
        return self.cash + positions_value

    @property
    def unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl for p in self.positions.values())

    def buy(self, symbol: str, price: float, date: str) -> bool:
        """Open a new position"""
        if symbol in self.positions:
            return False

        if len(self.positions) >= self.max_positions:
            return False

        # Calculate position size: Kapital / 30 (dynamic)
        target_value = self.equity / self.max_positions
        quantity = int(target_value / price)

        if quantity <= 0:
            return False

        cost = quantity * price + FEE_PER_TRADE

        if cost > self.cash:
            # Reduce quantity if not enough cash
            quantity = int((self.cash - FEE_PER_TRADE) / price)
            if quantity <= 0:
                return False
            cost = quantity * price + FEE_PER_TRADE

        self.cash -= cost
        self.positions[symbol] = Position(symbol, price, quantity, date)
        self.total_fees += FEE_PER_TRADE
        self.total_trades += 1

        self.trade_history.append({
            'date': date,
            'symbol': symbol,
            'action': 'BUY',
            'price': price,
            'quantity': quantity,
            'value': quantity * price,
            'fee': FEE_PER_TRADE,
            'position_size_pct': (quantity * price / self.equity) * 100
        })

        return True

    def sell(self, symbol: str, price: float, date: str) -> Optional[float]:
        """Close a position"""
        if symbol not in self.positions:
            return None

        pos = self.positions[symbol]
        proceeds = pos.quantity * price - FEE_PER_TRADE
        pnl = proceeds - pos.cost_basis

        self.cash += proceeds
        self.total_fees += FEE_PER_TRADE
        self.total_trades += 1

        self.trade_history.append({
            'date': date,
            'symbol': symbol,
            'action': 'SELL',
            'price': price,
            'quantity': pos.quantity,
            'value': pos.quantity * price,
            'pnl': pnl,
            'fee': FEE_PER_TRADE,
            'hold_days': (pd.Timestamp(date) - pd.Timestamp(pos.entry_date)).days
        })

        del self.positions[symbol]
        return pnl

    def update_prices(self, prices: Dict[str, float]):
        """Update all position prices"""
        for symbol, pos in self.positions.items():
            if symbol in prices:
                pos.update_price(prices[symbol])

    def get_worst_positions_by_daily_pnl(self, prev_prices: Dict[str, float], n: int) -> List[str]:
        """Get n worst performers by daily PnL"""
        daily_pnl = {}
        for symbol, pos in self.positions.items():
            if symbol in prev_prices:
                prev = prev_prices[symbol]
                daily_change = (pos.current_price - prev) * pos.quantity
                daily_pnl[symbol] = daily_change

        sorted_positions = sorted(daily_pnl.items(), key=lambda x: x[1])
        return [symbol for symbol, _ in sorted_positions[:n]]

    def get_worst_positions_by_total_pnl(self, n: int) -> List[str]:
        """Get n worst performers by total PnL"""
        total_pnl = {symbol: pos.unrealized_pnl for symbol, pos in self.positions.items()}
        sorted_positions = sorted(total_pnl.items(), key=lambda x: x[1])
        return [symbol for symbol, _ in sorted_positions[:n]]

    def record_equity(self, date: str):
        """Record current equity for equity curve"""
        self.equity_curve.append({
            'date': date,
            'equity': self.equity,
            'cash': self.cash,
            'positions_value': self.equity - self.cash,
            'num_positions': len(self.positions),
            'unrealized_pnl': self.unrealized_pnl
        })


def run_backtest(
    data: Dict[str, pd.DataFrame],
    strategy: str = 'total_pnl',  # 'daily_pnl' or 'total_pnl'
    initial_capital: float = INITIAL_CAPITAL,
    max_positions: int = MAX_POSITIONS,
    daily_rotation: int = DAILY_ROTATION
) -> BacktestEngine:
    """
    Run backtest with rotation strategy

    Args:
        data: Historical price data per symbol
        strategy: 'daily_pnl' or 'total_pnl' for worst performer selection
        initial_capital: Starting capital
        max_positions: Maximum positions
        daily_rotation: Number to rotate daily

    Returns:
        BacktestEngine with results
    """
    engine = BacktestEngine(initial_capital, max_positions, daily_rotation)

    # Get common dates across all symbols
    all_dates = None
    for symbol, df in data.items():
        dates = set(df.index)
        if all_dates is None:
            all_dates = dates
        else:
            all_dates = all_dates.intersection(dates)

    if not all_dates:
        logger.error("No common dates found")
        return engine

    sorted_dates = sorted(all_dates)

    # Use last SIMULATION_DAYS
    if len(sorted_dates) > SIMULATION_DAYS:
        sorted_dates = sorted_dates[-SIMULATION_DAYS:]

    logger.info(f"Running backtest: {len(sorted_dates)} days, strategy={strategy}")

    # Calculate signals for all stocks
    signals = {}
    for symbol, df in data.items():
        try:
            df_with_st = calculate_supertrend(df.copy())
            signals[symbol] = df_with_st
        except Exception as e:
            logger.warning(f"Could not calculate Supertrend for {symbol}: {e}")

    prev_prices = {}

    for i, date in enumerate(sorted_dates):
        date_str = str(date)[:10]

        # Get current prices
        current_prices = {}
        for symbol, df in data.items():
            if date in df.index:
                current_prices[symbol] = df.loc[date, 'close']

        # Update position prices
        engine.update_prices(current_prices)

        if i == 0:
            # Day 1: Open initial positions with BUY signals
            buy_candidates = []
            for symbol in signals:
                if symbol in signals and date in signals[symbol].index:
                    sig = signals[symbol].loc[date, 'signal']
                    if sig == 'BUY' and symbol in current_prices:
                        buy_candidates.append(symbol)

            # Buy up to max_positions
            for symbol in buy_candidates[:max_positions]:
                engine.buy(symbol, current_prices[symbol], date_str)

        else:
            # Daily rotation
            if len(engine.positions) >= daily_rotation:
                # Get worst performers
                if strategy == 'daily_pnl':
                    worst = engine.get_worst_positions_by_daily_pnl(prev_prices, daily_rotation)
                else:
                    worst = engine.get_worst_positions_by_total_pnl(daily_rotation)

                # Sell worst
                for symbol in worst:
                    if symbol in current_prices:
                        engine.sell(symbol, current_prices[symbol], date_str)

                # Find new buy candidates (not already held, with BUY signal)
                buy_candidates = []
                for symbol in signals:
                    if symbol not in engine.positions:
                        if symbol in signals and date in signals[symbol].index:
                            sig = signals[symbol].loc[date, 'signal']
                            if sig == 'BUY' and symbol in current_prices:
                                buy_candidates.append(symbol)

                # Buy new positions
                for symbol in buy_candidates[:daily_rotation]:
                    engine.buy(symbol, current_prices[symbol], date_str)

        # Record equity
        engine.record_equity(date_str)

        # Store prices for next day's daily PnL calculation
        prev_prices = current_prices.copy()

    return engine


def print_results(engine: BacktestEngine, strategy_name: str):
    """Print backtest results"""
    print(f"\n{'='*60}")
    print(f"BACKTEST RESULTS: {strategy_name}")
    print(f"{'='*60}")

    if not engine.equity_curve:
        print("No data")
        return

    start_equity = engine.initial_capital
    end_equity = engine.equity
    total_return = end_equity - start_equity
    total_return_pct = (total_return / start_equity) * 100

    # Calculate metrics
    equity_values = [e['equity'] for e in engine.equity_curve]

    # Max drawdown
    peak = equity_values[0]
    max_dd = 0
    for val in equity_values:
        if val > peak:
            peak = val
        dd = (peak - val) / peak
        max_dd = max(max_dd, dd)

    # Sharpe ratio (simplified)
    returns = pd.Series(equity_values).pct_change().dropna()
    if len(returns) > 0 and returns.std() > 0:
        sharpe = np.sqrt(252) * returns.mean() / returns.std()
    else:
        sharpe = 0

    # Win rate
    closed_trades = [t for t in engine.trade_history if t['action'] == 'SELL']
    if closed_trades:
        winners = [t for t in closed_trades if t.get('pnl', 0) > 0]
        win_rate = len(winners) / len(closed_trades) * 100
        avg_win = np.mean([t['pnl'] for t in winners]) if winners else 0
        losers = [t for t in closed_trades if t.get('pnl', 0) <= 0]
        avg_loss = np.mean([t['pnl'] for t in losers]) if losers else 0
    else:
        win_rate = 0
        avg_win = 0
        avg_loss = 0

    print(f"\n  Startkapital:      ${start_equity:,.2f}")
    print(f"  Endkapital:        ${end_equity:,.2f}")
    print(f"  Total Return:      ${total_return:+,.2f} ({total_return_pct:+.1f}%)")
    print(f"")
    print(f"  Total Trades:      {engine.total_trades}")
    print(f"  Total Fees:        ${engine.total_fees:,.2f}")
    print(f"  Win Rate:          {win_rate:.1f}%")
    print(f"  Avg Winner:        ${avg_win:+,.2f}")
    print(f"  Avg Loser:         ${avg_loss:,.2f}")
    print(f"")
    print(f"  Max Drawdown:      {max_dd*100:.1f}%")
    print(f"  Sharpe Ratio:      {sharpe:.2f}")
    print(f"")
    print(f"  Final Positions:   {len(engine.positions)}")

    return {
        'strategy': strategy_name,
        'start_equity': start_equity,
        'end_equity': end_equity,
        'total_return': total_return,
        'total_return_pct': total_return_pct,
        'total_trades': engine.total_trades,
        'total_fees': engine.total_fees,
        'win_rate': win_rate,
        'max_drawdown': max_dd * 100,
        'sharpe': sharpe
    }


def main():
    print("="*60)
    print("ROTATION STRATEGY BACKTESTER")
    print("="*60)
    print(f"\nKonfiguration:")
    print(f"  Startkapital:  ${INITIAL_CAPITAL:,}")
    print(f"  Max Positions: {MAX_POSITIONS}")
    print(f"  Daily Rotation: {DAILY_ROTATION} (sell worst, buy new)")
    print(f"  Simulation:    {SIMULATION_DAYS} Tage (~1 Jahr)")

    # Get all tickers
    tickers = get_all_tickers()
    print(f"\n  Aktien:        {len(tickers)} Symbole")

    # Fetch historical data
    print("\n" + "-"*60)
    print("LOADING HISTORICAL DATA...")
    print("-"*60)

    # Try IB first, fall back to Yahoo
    data = fetch_historical_data_ib(tickers, days=400)

    if len(data) < 10:
        logger.info("Not enough data, using synthetic data...")
        data = generate_synthetic_data(tickers, days=400)

    if len(data) < 10:
        logger.error("Not enough data to run backtest")
        return

    print(f"\nLoaded data for {len(data)} symbols")

    # Run backtest with DAILY PnL strategy
    print("\n" + "-"*60)
    print("RUNNING BACKTEST: DAILY PNL STRATEGY")
    print("-"*60)
    engine_daily = run_backtest(data, strategy='daily_pnl')
    results_daily = print_results(engine_daily, "Sell 3 Worst by DAILY PnL")

    # Run backtest with TOTAL PnL strategy
    print("\n" + "-"*60)
    print("RUNNING BACKTEST: TOTAL PNL STRATEGY")
    print("-"*60)
    engine_total = run_backtest(data, strategy='total_pnl')
    results_total = print_results(engine_total, "Sell 3 Worst by TOTAL PnL")

    # Comparison
    print("\n" + "="*60)
    print("VERGLEICH")
    print("="*60)
    print(f"\n{'Metrik':<20} {'Daily PnL':>15} {'Total PnL':>15} {'Besser':>12}")
    print("-"*62)

    if results_daily and results_total:
        metrics = [
            ('Total Return', 'total_return', '${:+,.0f}'),
            ('Return %', 'total_return_pct', '{:+.1f}%'),
            ('Win Rate', 'win_rate', '{:.1f}%'),
            ('Max Drawdown', 'max_drawdown', '{:.1f}%'),
            ('Sharpe Ratio', 'sharpe', '{:.2f}'),
        ]

        for name, key, fmt in metrics:
            d = results_daily[key]
            t = results_total[key]

            if key == 'max_drawdown':
                better = 'Daily' if d < t else 'Total'
            else:
                better = 'Daily' if d > t else 'Total'

            print(f"{name:<20} {fmt.format(d):>15} {fmt.format(t):>15} {better:>12}")

    print("\n" + "="*60)
    print("EMPFEHLUNG:")
    if results_daily and results_total:
        if results_daily['total_return'] > results_total['total_return']:
            print("  → Verwende DAILY PNL Strategie (bessere Returns)")
        else:
            print("  → Verwende TOTAL PNL Strategie (bessere Returns)")
    print("="*60)


if __name__ == "__main__":
    main()
