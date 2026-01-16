#!/usr/bin/env python3
"""
Strategy Backtester - Follows indicator signals per stock category
==================================================================
Each stock uses its assigned strategy from stock_categories.json:
- SUPERTREND: Supertrend indicator
- KAMA: Kaufman Adaptive Moving Average crossover
- JMA: Jurik Moving Average crossover
- BUY_HOLD: Always long (trailing stop for protection)
- TREND_FOLLOW: KAMA crossover for ETFs

Buy when signal = BUY, Sell when signal = SELL. No rotation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import logging

# Import indicator functions from ib_paper_trader
from ib_paper_trader import (
    get_signal_for_strategy, get_ticker_strategy,
    get_stocks_by_strategy, load_stock_categories
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================
INITIAL_CAPITAL = 20000
MAX_POSITIONS = 30
SIMULATION_DAYS = 365  # Full calendar year
FEE_PER_TRADE = 1.0

CATEGORIES_FILE = "stock_categories.json"


def get_all_tickers() -> List[str]:
    """Get all active tickers"""
    stocks = get_stocks_by_strategy()
    all_tickers = []
    for tickers in stocks.values():
        all_tickers.extend(tickers)
    return list(set(all_tickers))


def generate_synthetic_data(symbols: List[str], days: int = 400) -> Dict[str, pd.DataFrame]:
    """Generate realistic synthetic stock data for backtesting"""
    np.random.seed(42)
    data = {}

    base_prices = {
        'NVDA': 500, 'AMD': 140, 'AVGO': 180, 'META': 520, 'TSLA': 250,
        'COIN': 250, 'MSTR': 450, 'PLTR': 70, 'SHOP': 100, 'UBER': 75,
        'CRWD': 350, 'MU': 100, 'JPM': 200, 'INOD': 180, 'QUBT': 15,
        'DRH': 10, 'MRNA': 50, 'MRK': 100, 'NFLX': 900, 'NKE': 75,
        'PFE': 25, 'PYPL': 85, 'PDYN': 40, 'QBTS': 8, 'TKMS': 30,
        'JNJ': 150, 'TGT': 130, 'UNH': 550, 'SPY': 580, 'QQQ': 500
    }

    volatility = {
        'NVDA': 0.035, 'AMD': 0.035, 'AVGO': 0.025, 'META': 0.03, 'TSLA': 0.045,
        'COIN': 0.05, 'MSTR': 0.06, 'PLTR': 0.04, 'SHOP': 0.035, 'UBER': 0.03,
        'CRWD': 0.035, 'MU': 0.035, 'JPM': 0.02, 'INOD': 0.04, 'QUBT': 0.08,
        'DRH': 0.025, 'MRNA': 0.045, 'MRK': 0.02, 'NFLX': 0.03, 'NKE': 0.025,
        'PFE': 0.02, 'PYPL': 0.035, 'PDYN': 0.04, 'QBTS': 0.08, 'TKMS': 0.03,
        'JNJ': 0.015, 'TGT': 0.025, 'UNH': 0.02, 'SPY': 0.012, 'QQQ': 0.015
    }

    drift = {
        'NVDA': 0.001, 'AMD': 0.0008, 'AVGO': 0.0007, 'META': 0.0006, 'TSLA': 0.0005,
        'COIN': 0.0003, 'MSTR': 0.0002, 'PLTR': 0.0008, 'SHOP': 0.0004, 'UBER': 0.0005,
        'CRWD': 0.0006, 'MU': 0.0005, 'JPM': 0.0004, 'INOD': 0.0007, 'QUBT': 0.001,
        'DRH': 0.0003, 'MRNA': -0.0002, 'MRK': 0.0002, 'NFLX': 0.0005, 'NKE': 0.0001,
        'PFE': -0.0001, 'PYPL': 0.0003, 'PDYN': 0.0004, 'QBTS': 0.0008, 'TKMS': 0.0003,
        'JNJ': 0.0002, 'TGT': 0.0002, 'UNH': 0.0003, 'SPY': 0.0004, 'QQQ': 0.0005
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
        logger.info(f"  {symbol}: {len(df)} days (synthetic)")

    return data


def fetch_historical_data_ib(symbols: List[str], days: int = 365) -> Dict[str, pd.DataFrame]:
    """Fetch historical data from Interactive Brokers"""
    try:
        from ib_insync import IB, Stock, util
        from ib_paper_trader import get_ticker_contract_params
    except ImportError:
        logger.error("ib_insync not installed, using synthetic data")
        return generate_synthetic_data(symbols, days)

    data = {}
    ib = IB()

    try:
        ib.connect('127.0.0.1', 7497, clientId=51)
        logger.info(f"Connected to IB. Fetching data for {len(symbols)} symbols...")

        for symbol in symbols:
            try:
                # Get correct exchange/currency from config
                exchange, currency = get_ticker_contract_params(symbol)
                contract = Stock(symbol, exchange, currency)
                ib.qualifyContracts(contract)

                bars = ib.reqHistoricalData(
                    contract,
                    endDateTime='',
                    durationStr='1 Y',  # Use 1 year instead of 400 D
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


# =============================================================================
# BACKTEST ENGINE - Signal Following (No Rotation)
# =============================================================================
class Position:
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


class BacktestEngine:
    def __init__(self, initial_capital: float, max_positions: int):
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trade_history = []
        self.equity_curve = []
        self.total_fees = 0
        self.total_trades = 0

    @property
    def equity(self) -> float:
        return self.cash + sum(p.market_value for p in self.positions.values())

    def buy(self, symbol: str, price: float, date: str) -> bool:
        if symbol in self.positions or len(self.positions) >= self.max_positions:
            return False

        # Position size: Equity / max_positions
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
        self.positions[symbol] = Position(symbol, price, quantity, date)
        self.total_fees += FEE_PER_TRADE
        self.total_trades += 1

        self.trade_history.append({
            'date': date, 'symbol': symbol, 'action': 'BUY',
            'price': price, 'quantity': quantity, 'fee': FEE_PER_TRADE
        })
        return True

    def sell(self, symbol: str, price: float, date: str) -> Optional[float]:
        if symbol not in self.positions:
            return None

        pos = self.positions[symbol]
        proceeds = pos.quantity * price - FEE_PER_TRADE
        pnl = proceeds - pos.cost_basis

        self.cash += proceeds
        self.total_fees += FEE_PER_TRADE
        self.total_trades += 1

        self.trade_history.append({
            'date': date, 'symbol': symbol, 'action': 'SELL',
            'price': price, 'quantity': pos.quantity, 'pnl': pnl, 'fee': FEE_PER_TRADE
        })

        del self.positions[symbol]
        return pnl

    def update_prices(self, prices: Dict[str, float]):
        for symbol, pos in self.positions.items():
            if symbol in prices:
                pos.update_price(prices[symbol])

    def record_equity(self, date: str):
        self.equity_curve.append({
            'date': date, 'equity': self.equity,
            'num_positions': len(self.positions)
        })


def run_backtest(data: Dict[str, pd.DataFrame]) -> BacktestEngine:
    """Run backtest following signals per stock strategy"""
    engine = BacktestEngine(INITIAL_CAPITAL, MAX_POSITIONS)

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

    logger.info(f"Running backtest: {len(sorted_dates)} days")
    logger.info(f"Strategies: {ticker_strategies}")

    for i, date in enumerate(sorted_dates):
        date_str = str(date)[:10]

        # Get current prices
        current_prices = {}
        for symbol, df in data.items():
            if date in df.index:
                current_prices[symbol] = df.loc[date, 'close']

        # Update position prices
        engine.update_prices(current_prices)

        # Check signals for each stock
        for symbol in data.keys():
            if symbol not in data or date not in data[symbol].index:
                continue

            df_slice = data[symbol].loc[:date].copy()
            if len(df_slice) < 60:  # Need 60 for EMA 50 + buffer
                continue

            strat = ticker_strategies.get(symbol, "SUPERTREND")
            signal = get_signal_for_strategy(df_slice, strat)
            price = current_prices.get(symbol)

            if price is None:
                continue

            has_position = symbol in engine.positions

            # BUY signal and no position -> Buy
            if signal == "BUY" and not has_position:
                engine.buy(symbol, price, date_str)

            # SELL signal and has position -> Sell
            elif signal == "SELL" and has_position:
                engine.sell(symbol, price, date_str)

        engine.record_equity(date_str)

    return engine


def print_results(engine: BacktestEngine):
    """Print backtest results"""
    print("\n" + "="*60)
    print("BACKTEST RESULTS: Strategy per Stock (No Rotation)")
    print("="*60)

    if not engine.equity_curve:
        print("No data")
        return

    start_equity = engine.initial_capital
    end_equity = engine.equity
    total_return = end_equity - start_equity
    total_return_pct = (total_return / start_equity) * 100

    # Max drawdown
    equity_values = [e['equity'] for e in engine.equity_curve]
    peak = equity_values[0]
    max_dd = 0
    for val in equity_values:
        if val > peak:
            peak = val
        dd = (peak - val) / peak
        max_dd = max(max_dd, dd)

    # Sharpe ratio
    returns = pd.Series(equity_values).pct_change().dropna()
    sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0

    # Win rate
    closed_trades = [t for t in engine.trade_history if t['action'] == 'SELL']
    if closed_trades:
        winners = [t for t in closed_trades if t.get('pnl', 0) > 0]
        win_rate = len(winners) / len(closed_trades) * 100
    else:
        win_rate = 0

    print(f"\n  Startkapital:      ${start_equity:,.2f}")
    print(f"  Endkapital:        ${end_equity:,.2f}")
    print(f"  Total Return:      ${total_return:+,.2f} ({total_return_pct:+.1f}%)")
    print(f"")
    print(f"  Total Trades:      {engine.total_trades}")
    print(f"  Total Fees:        ${engine.total_fees:,.2f}")
    print(f"  Win Rate:          {win_rate:.1f}%")
    print(f"")
    print(f"  Max Drawdown:      {max_dd*100:.1f}%")
    print(f"  Sharpe Ratio:      {sharpe:.2f}")
    print(f"")
    print(f"  Final Positions:   {len(engine.positions)}")

    # Show positions by strategy
    print("\n" + "-"*60)
    print("Positionen nach Strategie:")
    stocks_by_strat = get_stocks_by_strategy()
    for strat, tickers in stocks_by_strat.items():
        in_position = [s for s in tickers if s in engine.positions]
        print(f"  {strat}: {len(in_position)}/{len(tickers)} aktiv")

    print("="*60)


def main():
    print("="*60)
    print("STRATEGY BACKTESTER (Signal Following)")
    print("="*60)

    # Show strategy assignments
    stocks_by_strat = get_stocks_by_strategy()
    print("\nStrategien:")
    for strat, tickers in stocks_by_strat.items():
        print(f"  {strat} ({len(tickers)}): {', '.join(tickers[:5])}{'...' if len(tickers) > 5 else ''}")

    print(f"\nKonfiguration:")
    print(f"  Startkapital:  ${INITIAL_CAPITAL:,}")
    print(f"  Max Positions: {MAX_POSITIONS}")
    print(f"  Simulation:    {SIMULATION_DAYS} Tage (~1 Jahr)")

    tickers = get_all_tickers()
    print(f"  Aktien:        {len(tickers)} Symbole")

    print("\n" + "-"*60)
    print("LOADING HISTORICAL DATA...")
    print("-"*60)

    data = fetch_historical_data_ib(tickers, days=400)

    if len(data) < 10:
        data = generate_synthetic_data(tickers, days=400)

    print(f"\nLoaded data for {len(data)} symbols")

    print("\n" + "-"*60)
    print("RUNNING BACKTEST...")
    print("-"*60)

    engine = run_backtest(data)
    print_results(engine)


if __name__ == "__main__":
    main()
