"""
Portfolio Simulation with Multi-Strategy Trading
=================================================
- $20,000 initial capital, max 30 positions
- 10 positions: Buy & Hold (best 10, 20% trailing stop, rebalancing)
- 20 positions: Strategy signals (Long/Short)
- Daily PnL calculation
- HTML report with trade lists, charts, statistics
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import json

# Import from new5.py
from new5 import (
    download_from_tws, disconnect_ib, ALL_TICKERS,
    calculate_supertrend_vectorized, generate_signals_vectorized,
    calculate_jma, calculate_kama, calculate_sma, calculate_ema,
    get_ma_crossover_signals, get_htf_trend
)


@dataclass
class Position:
    """Represents a single position"""
    symbol: str
    direction: str  # 'LONG' or 'SHORT'
    quantity: int
    entry_price: float
    entry_date: datetime
    entry_fee: float
    strategy: str  # 'BUYHOLD', 'SUPERTREND', 'JMA', etc.

    # For trailing stop (B&H)
    highest_price: float = 0.0
    trailing_stop_pct: float = 0.20

    # Current state
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    # Exit info (when closed)
    exit_price: float = 0.0
    exit_date: Optional[datetime] = None
    exit_fee: float = 0.0
    exit_reason: str = ""

    def update_price(self, price: float):
        """Update current price and calculate unrealized PnL"""
        self.current_price = price
        if self.direction == 'LONG':
            self.unrealized_pnl = self.quantity * (price - self.entry_price)
            if price > self.highest_price:
                self.highest_price = price
        else:  # SHORT
            self.unrealized_pnl = self.quantity * (self.entry_price - price)

    def check_trailing_stop(self) -> bool:
        """Check if trailing stop is hit (for B&H positions)"""
        if self.strategy != 'BUYHOLD' or self.highest_price == 0:
            return False
        stop_price = self.highest_price * (1 - self.trailing_stop_pct)
        return self.current_price <= stop_price

    def get_market_value(self) -> float:
        """Get current market value"""
        return self.quantity * self.current_price

    def get_daily_pnl(self, prev_close: float) -> float:
        """Calculate daily PnL"""
        if self.direction == 'LONG':
            return self.quantity * (self.current_price - prev_close)
        else:
            return self.quantity * (prev_close - self.current_price)

    def close(self, exit_price: float, exit_date: datetime, fee_rate: float, reason: str = ""):
        """Close the position"""
        self.exit_price = exit_price
        self.exit_date = exit_date
        self.exit_fee = self.quantity * exit_price * fee_rate
        self.exit_reason = reason

        if self.direction == 'LONG':
            self.realized_pnl = self.quantity * (exit_price - self.entry_price) - self.entry_fee - self.exit_fee
        else:
            self.realized_pnl = self.quantity * (self.entry_price - exit_price) - self.entry_fee - self.exit_fee

        self.unrealized_pnl = 0

    def days_held(self) -> int:
        """Calculate days held"""
        end_date = self.exit_date if self.exit_date else datetime.now()
        return (end_date - self.entry_date).days

    def pnl_percent(self) -> float:
        """Calculate PnL percentage"""
        cost = self.quantity * self.entry_price
        if cost == 0:
            return 0
        pnl = self.realized_pnl if self.exit_date else self.unrealized_pnl
        return pnl / cost


class PortfolioSimulator:
    """
    Portfolio Simulation Engine
    - Manages 30 positions (10 B&H + 20 Strategy)
    - Daily PnL tracking
    - Position rebalancing
    """

    def __init__(self,
                 initial_capital: float = 20000.0,
                 max_positions: int = 30,
                 bh_positions: int = 10,
                 strategy_positions: int = 20,
                 fee_rate: float = 0.001,  # 0.1% fee
                 trailing_stop_pct: float = 0.20):

        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.max_positions = max_positions
        self.bh_positions = bh_positions
        self.strategy_positions = strategy_positions
        self.fee_rate = fee_rate
        self.trailing_stop_pct = trailing_stop_pct

        self.open_positions: Dict[str, Position] = {}  # symbol -> Position
        self.closed_positions: List[Position] = []

        # Daily tracking
        self.equity_curve: List[Dict] = []
        self.daily_pnl_history: List[Dict] = []

        # Strategy assignments (loaded from categorization)
        self.strategy_assignments: Dict = {}

        # Price data cache
        self.price_data: Dict[str, pd.DataFrame] = {}

    def get_position_stake(self) -> float:
        """Calculate stake per position"""
        return self.capital / self.max_positions

    def calculate_quantity(self, price: float) -> int:
        """Calculate quantity based on stake"""
        stake = self.get_position_stake()
        return max(1, int(stake / price))

    def open_position(self, symbol: str, direction: str, price: float,
                      date: datetime, strategy: str) -> Optional[Position]:
        """Open a new position"""
        if symbol in self.open_positions:
            return None  # Already have position

        if len(self.open_positions) >= self.max_positions:
            return None  # Max positions reached

        quantity = self.calculate_quantity(price)
        entry_fee = quantity * price * self.fee_rate

        position = Position(
            symbol=symbol,
            direction=direction,
            quantity=quantity,
            entry_price=price,
            entry_date=date,
            entry_fee=entry_fee,
            strategy=strategy,
            highest_price=price if direction == 'LONG' else 0,
            trailing_stop_pct=self.trailing_stop_pct
        )
        position.current_price = price

        self.open_positions[symbol] = position
        self.capital -= entry_fee  # Deduct entry fee

        return position

    def close_position(self, symbol: str, price: float, date: datetime, reason: str = "") -> Optional[Position]:
        """Close an existing position"""
        if symbol not in self.open_positions:
            return None

        position = self.open_positions.pop(symbol)
        position.close(price, date, self.fee_rate, reason)

        self.capital -= position.exit_fee  # Deduct exit fee
        self.capital += position.realized_pnl + position.entry_fee + position.exit_fee  # Add realized PnL

        self.closed_positions.append(position)
        return position

    def get_best_bh_candidates(self, date: datetime, lookback_days: int = 30) -> List[Tuple[str, float]]:
        """
        Find best 10 Buy & Hold candidates based on recent performance.
        Returns list of (symbol, return) sorted by return descending.
        """
        candidates = []

        for symbol in ALL_TICKERS:
            if symbol not in self.price_data:
                continue

            df = self.price_data[symbol]

            # Get data up to current date
            df_slice = df[df.index <= date]
            if len(df_slice) < lookback_days:
                continue

            close_col = f'Close_{symbol}'
            if close_col not in df_slice.columns:
                continue

            # Calculate return over lookback period
            recent = df_slice.tail(lookback_days)
            if len(recent) < 2:
                continue

            start_price = recent[close_col].iloc[0]
            end_price = recent[close_col].iloc[-1]

            if start_price > 0:
                ret = (end_price - start_price) / start_price
                candidates.append((symbol, ret))

        # Sort by return descending and return top 10
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:self.bh_positions]

    def rebalance_bh_positions(self, date: datetime, prices: Dict[str, float]):
        """
        Rebalance Buy & Hold positions.
        - Check if current top 10 changed
        - Sell positions no longer in top 10
        - Buy new positions in top 10
        """
        best_candidates = self.get_best_bh_candidates(date)
        best_symbols = set(sym for sym, _ in best_candidates)

        # Current B&H positions
        current_bh = {sym: pos for sym, pos in self.open_positions.items()
                      if pos.strategy == 'BUYHOLD'}
        current_bh_symbols = set(current_bh.keys())

        # Positions to close (no longer in top 10)
        to_close = current_bh_symbols - best_symbols
        for symbol in to_close:
            if symbol in prices:
                self.close_position(symbol, prices[symbol], date, "BH_REBALANCE")

        # Positions to open (new in top 10)
        to_open = best_symbols - current_bh_symbols
        for symbol in to_open:
            if symbol in prices and len(self.open_positions) < self.max_positions:
                self.open_position(symbol, 'LONG', prices[symbol], date, 'BUYHOLD')

    def check_trailing_stops(self, date: datetime, prices: Dict[str, float]):
        """Check and execute trailing stops for B&H positions"""
        to_close = []

        for symbol, position in self.open_positions.items():
            if position.strategy == 'BUYHOLD' and symbol in prices:
                position.update_price(prices[symbol])
                if position.check_trailing_stop():
                    to_close.append(symbol)

        for symbol in to_close:
            self.close_position(symbol, prices[symbol], date, "TRAILING_STOP")

    def process_strategy_signals(self, date: datetime, prices: Dict[str, float],
                                  signals: Dict[str, Dict]):
        """
        Process strategy signals for the day.
        signals format: {symbol: {'action': 'BUY'/'SELL'/'SHORT'/'COVER', 'strategy': 'SUPERTREND', ...}}
        """
        for symbol, signal in signals.items():
            if symbol not in prices:
                continue

            price = prices[symbol]
            action = signal.get('action')
            strategy = signal.get('strategy', 'STRATEGY')

            if action == 'BUY':
                # Open long position
                if symbol not in self.open_positions:
                    strategy_count = sum(1 for p in self.open_positions.values()
                                        if p.strategy != 'BUYHOLD')
                    if strategy_count < self.strategy_positions:
                        self.open_position(symbol, 'LONG', price, date, strategy)

            elif action == 'SELL':
                # Close long position
                if symbol in self.open_positions:
                    pos = self.open_positions[symbol]
                    if pos.direction == 'LONG' and pos.strategy != 'BUYHOLD':
                        self.close_position(symbol, price, date, "SIGNAL_SELL")

            elif action == 'SHORT':
                # Open short position
                if symbol not in self.open_positions:
                    strategy_count = sum(1 for p in self.open_positions.values()
                                        if p.strategy != 'BUYHOLD')
                    if strategy_count < self.strategy_positions:
                        self.open_position(symbol, 'SHORT', price, date, strategy)

            elif action == 'COVER':
                # Close short position
                if symbol in self.open_positions:
                    pos = self.open_positions[symbol]
                    if pos.direction == 'SHORT':
                        self.close_position(symbol, price, date, "SIGNAL_COVER")

    def update_daily(self, date: datetime, prices: Dict[str, float], prev_prices: Dict[str, float]):
        """Update all positions and calculate daily PnL"""
        daily_pnl = 0.0
        unrealized_total = 0.0

        for symbol, position in self.open_positions.items():
            if symbol in prices:
                prev_close = prev_prices.get(symbol, position.current_price)
                position.update_price(prices[symbol])
                daily_pnl += position.get_daily_pnl(prev_close)
                unrealized_total += position.unrealized_pnl

        self.capital += daily_pnl

        # Record daily state
        realized_total = sum(p.realized_pnl for p in self.closed_positions)

        self.equity_curve.append({
            'date': date,
            'capital': self.capital,
            'unrealized': unrealized_total,
            'realized': realized_total,
            'positions': len(self.open_positions),
            'daily_pnl': daily_pnl
        })

        self.daily_pnl_history.append({
            'date': date,
            'pnl': daily_pnl
        })

    def load_price_data(self, symbols: List[str], days_back: int = 180):
        """Load price data for all symbols"""
        print(f"Loading price data for {len(symbols)} symbols...")

        for i, symbol in enumerate(symbols, 1):
            try:
                df = download_from_tws(symbol, days_back)
                if not df.empty:
                    self.price_data[symbol] = df
                    print(f"  [{i}/{len(symbols)}] {symbol}: {len(df)} days")
            except Exception as e:
                print(f"  [{i}/{len(symbols)}] {symbol}: ERROR - {e}")

        print(f"Loaded data for {len(self.price_data)} symbols")

    def generate_strategy_signals(self, symbol: str, date: datetime) -> Optional[Dict]:
        """Generate trading signal for a symbol based on its assigned strategy"""
        if symbol not in self.price_data:
            return None

        df = self.price_data[symbol]
        df_slice = df[df.index <= date]

        if len(df_slice) < 50:
            return None

        close_col = f'Close_{symbol}'
        high_col = f'High_{symbol}'
        low_col = f'Low_{symbol}'

        if close_col not in df_slice.columns:
            return None

        close = df_slice[close_col].values
        high = df_slice[high_col].values
        low = df_slice[low_col].values
        close_series = df_slice[close_col]

        # Get strategy assignment
        assignment = self.strategy_assignments.get(symbol, {})
        strategy = assignment.get('strategy', 'SUPERTREND')
        params = assignment.get('params', {})
        use_htf = '_HTF' in strategy
        base_strategy = strategy.replace('_HTF', '').replace('_NOHTF', '')

        try:
            # Generate signals based on strategy
            if base_strategy == 'SUPERTREND':
                period = params.get('period', 10)
                mult = params.get('multiplier', 3.0)
                supertrend, direction, _ = calculate_supertrend_vectorized(high, low, close, period, mult)

                # Check for signal on last bar
                if len(direction) >= 2:
                    if direction[-1] == 1 and direction[-2] == -1:
                        return {'action': 'BUY', 'strategy': strategy}
                    elif direction[-1] == -1 and direction[-2] == 1:
                        return {'action': 'SELL', 'strategy': strategy}

            elif base_strategy == 'JMA':
                fast = params.get('fast', 10)
                slow = params.get('slow', 30)
                jma_fast = calculate_jma(close_series, fast).values
                jma_slow = calculate_jma(close_series, slow).values

                if len(jma_fast) >= 2 and len(jma_slow) >= 2:
                    cross_up = jma_fast[-1] > jma_slow[-1] and jma_fast[-2] <= jma_slow[-2]
                    cross_down = jma_fast[-1] < jma_slow[-1] and jma_fast[-2] >= jma_slow[-2]

                    if cross_up:
                        return {'action': 'BUY', 'strategy': strategy}
                    elif cross_down:
                        return {'action': 'SELL', 'strategy': strategy}

            elif base_strategy == 'KAMA':
                period = params.get('period', 10)
                signal = params.get('signal', 14)
                kama = calculate_kama(close_series, period).values
                signal_line = calculate_sma(close_series, signal).values

                if len(kama) >= 2 and len(signal_line) >= 2:
                    cross_up = kama[-1] > signal_line[-1] and kama[-2] <= signal_line[-2]
                    cross_down = kama[-1] < signal_line[-1] and kama[-2] >= signal_line[-2]

                    if cross_up:
                        return {'action': 'BUY', 'strategy': strategy}
                    elif cross_down:
                        return {'action': 'SELL', 'strategy': strategy}

            elif base_strategy == 'EMA':
                fast = params.get('fast', 12)
                slow = params.get('slow', 26)
                ema_fast = calculate_ema(close_series, fast).values
                ema_slow = calculate_ema(close_series, slow).values

                if len(ema_fast) >= 2 and len(ema_slow) >= 2:
                    cross_up = ema_fast[-1] > ema_slow[-1] and ema_fast[-2] <= ema_slow[-2]
                    cross_down = ema_fast[-1] < ema_slow[-1] and ema_fast[-2] >= ema_slow[-2]

                    if cross_up:
                        return {'action': 'BUY', 'strategy': strategy}
                    elif cross_down:
                        return {'action': 'SELL', 'strategy': strategy}

            elif base_strategy == 'SMA':
                fast = params.get('fast', 20)
                slow = params.get('slow', 50)
                sma_fast = calculate_sma(close_series, fast).values
                sma_slow = calculate_sma(close_series, slow).values

                if len(sma_fast) >= 2 and len(sma_slow) >= 2:
                    cross_up = sma_fast[-1] > sma_slow[-1] and sma_fast[-2] <= sma_slow[-2]
                    cross_down = sma_fast[-1] < sma_slow[-1] and sma_fast[-2] >= sma_slow[-2]

                    if cross_up:
                        return {'action': 'BUY', 'strategy': strategy}
                    elif cross_down:
                        return {'action': 'SELL', 'strategy': strategy}

        except Exception as e:
            pass

        return None

    def run_simulation(self, days_back: int = 180, rebalance_freq: int = 5):
        """
        Run the full portfolio simulation.

        Args:
            days_back: Number of days to simulate
            rebalance_freq: Days between B&H rebalancing checks
        """
        print("\n" + "="*80)
        print("PORTFOLIO SIMULATION")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Max Positions: {self.max_positions} (B&H: {self.bh_positions}, Strategy: {self.strategy_positions})")
        print(f"Period: {days_back} days")
        print("="*80)

        # Load strategy assignments
        try:
            with open('htf_categorized_results.json', 'r') as f:
                data = json.load(f)
                categories = data.get('categories', {})

                # Build assignment dict
                for cat_name, items in categories.items():
                    if cat_name in ['UNDERPERFORM', 'BUYHOLD']:
                        continue
                    for item in items:
                        symbol = item['symbol']
                        self.strategy_assignments[symbol] = {
                            'strategy': cat_name,
                            'params': item.get('params', {}),
                            'return': item.get('return', 0)
                        }
                print(f"Loaded {len(self.strategy_assignments)} strategy assignments")
        except Exception as e:
            print(f"Warning: Could not load strategy assignments: {e}")

        # Load price data
        self.load_price_data(ALL_TICKERS, days_back + 50)

        if not self.price_data:
            print("ERROR: No price data available")
            return

        # Get common date range
        all_dates = set()
        for symbol, df in self.price_data.items():
            all_dates.update(df.index.tolist())

        all_dates = sorted(list(all_dates))
        simulation_dates = all_dates[-days_back:] if len(all_dates) > days_back else all_dates

        print(f"\nSimulating {len(simulation_dates)} trading days...")
        print(f"Start: {simulation_dates[0].strftime('%Y-%m-%d')}")
        print(f"End: {simulation_dates[-1].strftime('%Y-%m-%d')}")

        prev_prices = {}

        for day_num, date in enumerate(simulation_dates):
            # Get prices for this day
            prices = {}
            for symbol, df in self.price_data.items():
                if date in df.index:
                    close_col = f'Close_{symbol}'
                    if close_col in df.columns:
                        prices[symbol] = df.loc[date, close_col]

            if not prices:
                continue

            # Day 1: Initialize B&H positions
            if day_num == 0:
                self.rebalance_bh_positions(date, prices)

            # Periodic B&H rebalancing
            elif day_num % rebalance_freq == 0:
                self.rebalance_bh_positions(date, prices)

            # Check trailing stops
            self.check_trailing_stops(date, prices)

            # Generate and process strategy signals
            signals = {}
            for symbol in self.strategy_assignments.keys():
                signal = self.generate_strategy_signals(symbol, date)
                if signal:
                    signals[symbol] = signal

            self.process_strategy_signals(date, prices, signals)

            # Update daily PnL
            self.update_daily(date, prices, prev_prices)

            prev_prices = prices.copy()

            # Progress update
            if (day_num + 1) % 20 == 0:
                print(f"  Day {day_num + 1}/{len(simulation_dates)}: "
                      f"Capital=${self.capital:,.2f}, Positions={len(self.open_positions)}")

        print("\n" + "="*80)
        print("SIMULATION COMPLETE")
        print("="*80)
        self.print_summary()

    def print_summary(self):
        """Print simulation summary"""
        total_return = (self.capital - self.initial_capital) / self.initial_capital

        realized_pnl = sum(p.realized_pnl for p in self.closed_positions)
        unrealized_pnl = sum(p.unrealized_pnl for p in self.open_positions.values())

        # Trade statistics
        total_trades = len(self.closed_positions)
        winning_trades = sum(1 for p in self.closed_positions if p.realized_pnl > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        avg_winner = np.mean([p.realized_pnl for p in self.closed_positions if p.realized_pnl > 0]) if winning_trades > 0 else 0
        avg_loser = np.mean([p.realized_pnl for p in self.closed_positions if p.realized_pnl <= 0]) if total_trades - winning_trades > 0 else 0

        # Profit factor
        gross_profit = sum(p.realized_pnl for p in self.closed_positions if p.realized_pnl > 0)
        gross_loss = abs(sum(p.realized_pnl for p in self.closed_positions if p.realized_pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Max drawdown
        equity_values = [e['capital'] for e in self.equity_curve]
        if equity_values:
            peak = equity_values[0]
            max_dd = 0
            for value in equity_values:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak
                if dd > max_dd:
                    max_dd = dd
        else:
            max_dd = 0

        # Sharpe ratio (simplified)
        if self.daily_pnl_history:
            daily_returns = [d['pnl'] / self.initial_capital for d in self.daily_pnl_history]
            if len(daily_returns) > 1 and np.std(daily_returns) > 0:
                sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
            else:
                sharpe = 0
        else:
            sharpe = 0

        print(f"\nInitial Capital:    ${self.initial_capital:,.2f}")
        print(f"Final Capital:      ${self.capital:,.2f}")
        print(f"Total Return:       {total_return:+.2%}")
        print(f"\nRealized PnL:       ${realized_pnl:+,.2f}")
        print(f"Unrealized PnL:     ${unrealized_pnl:+,.2f}")
        print(f"\nOpen Positions:     {len(self.open_positions)}")
        print(f"Closed Trades:      {total_trades}")
        print(f"Win Rate:           {win_rate:.1%}")
        print(f"Profit Factor:      {profit_factor:.2f}")
        print(f"Sharpe Ratio:       {sharpe:.2f}")
        print(f"Max Drawdown:       {max_dd:.2%}")
        print(f"\nAvg Winner:         ${avg_winner:+,.2f}")
        print(f"Avg Loser:          ${avg_loser:+,.2f}")

    def get_statistics(self) -> Dict:
        """Get simulation statistics as dict"""
        total_return = (self.capital - self.initial_capital) / self.initial_capital
        realized_pnl = sum(p.realized_pnl for p in self.closed_positions)
        unrealized_pnl = sum(p.unrealized_pnl for p in self.open_positions.values())

        total_trades = len(self.closed_positions)
        winning_trades = sum(1 for p in self.closed_positions if p.realized_pnl > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        avg_winner = np.mean([p.realized_pnl for p in self.closed_positions if p.realized_pnl > 0]) if winning_trades > 0 else 0
        avg_loser = np.mean([p.realized_pnl for p in self.closed_positions if p.realized_pnl <= 0]) if total_trades - winning_trades > 0 else 0

        gross_profit = sum(p.realized_pnl for p in self.closed_positions if p.realized_pnl > 0)
        gross_loss = abs(sum(p.realized_pnl for p in self.closed_positions if p.realized_pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        equity_values = [e['capital'] for e in self.equity_curve]
        if equity_values:
            peak = equity_values[0]
            max_dd = 0
            max_dd_value = 0
            for value in equity_values:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak
                dd_value = peak - value
                if dd > max_dd:
                    max_dd = dd
                    max_dd_value = dd_value
        else:
            max_dd = 0
            max_dd_value = 0

        if self.daily_pnl_history:
            daily_returns = [d['pnl'] / self.initial_capital for d in self.daily_pnl_history]
            if len(daily_returns) > 1 and np.std(daily_returns) > 0:
                sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
            else:
                sharpe = 0
            daily_pnl_today = self.daily_pnl_history[-1]['pnl'] if self.daily_pnl_history else 0
        else:
            sharpe = 0
            daily_pnl_today = 0

        expectancy = avg_winner * win_rate + avg_loser * (1 - win_rate)

        return {
            'initial_capital': self.initial_capital,
            'final_capital': self.capital,
            'total_return': total_return,
            'total_return_value': self.capital - self.initial_capital,
            'realized_pnl': realized_pnl,
            'unrealized_pnl': unrealized_pnl,
            'daily_pnl': daily_pnl_today,
            'open_positions': len(self.open_positions),
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'max_drawdown_value': max_dd_value,
            'avg_winner': avg_winner,
            'avg_loser': avg_loser,
            'expectancy': expectancy
        }

    def generate_html_report(self, filename: str = "portfolio_report.html"):
        """Generate comprehensive HTML report"""
        stats = self.get_statistics()

        # Separate positions
        long_open = [p for p in self.open_positions.values() if p.direction == 'LONG']
        short_open = [p for p in self.open_positions.values() if p.direction == 'SHORT']
        long_closed = [p for p in self.closed_positions if p.direction == 'LONG']
        short_closed = [p for p in self.closed_positions if p.direction == 'SHORT']

        # Sort by daily PnL for open, by realized PnL for closed
        long_open.sort(key=lambda p: p.get_daily_pnl(p.entry_price), reverse=True)
        short_open.sort(key=lambda p: p.get_daily_pnl(p.entry_price), reverse=True)
        long_closed.sort(key=lambda p: p.exit_date or datetime.now(), reverse=True)
        short_closed.sort(key=lambda p: p.exit_date or datetime.now(), reverse=True)

        # Equity curve data for chart
        equity_dates = [e['date'].strftime('%Y-%m-%d') for e in self.equity_curve]
        equity_values = [e['capital'] for e in self.equity_curve]
        daily_pnl_values = [e['daily_pnl'] for e in self.equity_curve]

        # Calculate position PnL for bar chart
        position_pnl = []
        for pos in self.open_positions.values():
            position_pnl.append({
                'symbol': pos.symbol,
                'pnl': pos.unrealized_pnl,
                'direction': pos.direction
            })
        position_pnl.sort(key=lambda x: x['pnl'])

        html = f'''<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Portfolio Simulation Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #0a1628;
            color: #e0e0e0;
            min-height: 100vh;
        }}
        .header {{
            background: linear-gradient(135deg, #1a2744 0%, #0d1929 100%);
            padding: 20px;
            border-bottom: 1px solid #2a3f5f;
        }}
        .header h1 {{
            color: #4ecdc4;
            margin-bottom: 10px;
        }}
        .header-stats {{
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            margin-top: 15px;
        }}
        .stat-card {{
            background: rgba(30, 50, 80, 0.6);
            border-radius: 8px;
            padding: 15px 25px;
            min-width: 150px;
            text-align: center;
            border: 1px solid #2a3f5f;
        }}
        .stat-card.highlight {{
            background: linear-gradient(135deg, #1e8449 0%, #145a32 100%);
            border-color: #27ae60;
        }}
        .stat-card .label {{
            font-size: 11px;
            color: #8899a6;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .stat-card .value {{
            font-size: 24px;
            font-weight: bold;
            margin-top: 5px;
        }}
        .stat-card .value.positive {{
            color: #4ecdc4;
        }}
        .stat-card .value.negative {{
            color: #ff6b6b;
        }}
        .container {{
            max-width: 1800px;
            margin: 0 auto;
            padding: 20px;
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
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .section h2 .total {{
            color: #4ecdc4;
            font-size: 16px;
        }}
        .section h2 .total.negative {{
            color: #ff6b6b;
        }}
        .charts-row {{
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }}
        .positions-row {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
        }}
        .metric {{
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #2a3f5f;
        }}
        .metric .label {{
            color: #8899a6;
        }}
        .metric .value {{
            font-weight: bold;
        }}
        .metric .value.positive {{
            color: #4ecdc4;
        }}
        .metric .value.negative {{
            color: #ff6b6b;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}
        th {{
            background: rgba(42, 63, 95, 0.5);
            padding: 12px 8px;
            text-align: left;
            font-weight: 600;
            color: #8899a6;
            text-transform: uppercase;
            font-size: 11px;
            letter-spacing: 0.5px;
        }}
        td {{
            padding: 10px 8px;
            border-bottom: 1px solid #1a2744;
        }}
        tr:hover {{
            background: rgba(42, 63, 95, 0.3);
        }}
        .direction {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: bold;
        }}
        .direction.long {{
            background: #1e8449;
            color: #fff;
        }}
        .direction.short {{
            background: #922b21;
            color: #fff;
        }}
        .pnl-bar {{
            height: 100%;
            min-height: 20px;
        }}
        .timestamp {{
            color: #8899a6;
            font-size: 12px;
        }}
        .chart-container {{
            height: 300px;
        }}
        .bar-chart-container {{
            height: 400px;
        }}
        @media (max-width: 1200px) {{
            .charts-row, .positions-row {{
                grid-template-columns: 1fr;
            }}
            .metrics-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Trade Monitor</h1>
        <span class="timestamp">Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span>
        <div class="header-stats">
            <div class="stat-card">
                <div class="label">Net Liquidity</div>
                <div class="value {'positive' if stats['final_capital'] > stats['initial_capital'] else 'negative'}">${stats['final_capital']:,.2f}</div>
            </div>
            <div class="stat-card">
                <div class="label">Daily PnL</div>
                <div class="value {'positive' if stats['daily_pnl'] >= 0 else 'negative'}">${stats['daily_pnl']:+,.2f}</div>
            </div>
            <div class="stat-card">
                <div class="label">Unrealized</div>
                <div class="value {'positive' if stats['unrealized_pnl'] >= 0 else 'negative'}">${stats['unrealized_pnl']:+,.2f}</div>
            </div>
            <div class="stat-card">
                <div class="label">Realized PnL</div>
                <div class="value {'positive' if stats['realized_pnl'] >= 0 else 'negative'}">${stats['realized_pnl']:+,.2f}</div>
            </div>
            <div class="stat-card highlight">
                <div class="label">Total Return</div>
                <div class="value {'positive' if stats['total_return'] >= 0 else 'negative'}">${stats['total_return_value']:+,.2f} ({stats['total_return']:+.1%})</div>
            </div>
            <div class="stat-card highlight">
                <div class="label">Positions</div>
                <div class="value">{stats['open_positions']}</div>
            </div>
        </div>
    </div>

    <div class="container">
        <div class="charts-row">
            <div class="section">
                <h2>Kapitalkurve <span class="total {'negative' if stats['total_return'] < 0 else ''}">${stats['final_capital']:,.2f}</span></h2>
                <div id="equity-chart" class="chart-container"></div>
            </div>
            <div class="section">
                <h2>Daily PnL <span class="total {'negative' if stats['daily_pnl'] < 0 else ''}">${stats['daily_pnl']:+,.2f}</span></h2>
                <div id="daily-pnl-chart" class="chart-container"></div>
            </div>
        </div>

        <div class="positions-row">
            <div class="section">
                <h2>Open Trades PnL (alle {len(self.open_positions)} Positionen) <span class="total {'negative' if stats['unrealized_pnl'] < 0 else ''}">${stats['unrealized_pnl']:+,.2f}</span></h2>
                <div id="positions-bar-chart" class="bar-chart-container"></div>
            </div>
            <div class="section">
                <h2>Performance Metrics</h2>
                <div class="metrics-grid">
                    <div class="metric">
                        <span class="label">Initial Capital</span>
                        <span class="value">${stats['initial_capital']:,.2f}</span>
                    </div>
                    <div class="metric">
                        <span class="label">Win Rate</span>
                        <span class="value">{stats['win_rate']:.1%}</span>
                    </div>
                    <div class="metric">
                        <span class="label">Profit Factor</span>
                        <span class="value">{stats['profit_factor']:.2f}</span>
                    </div>
                    <div class="metric">
                        <span class="label">Sharpe Ratio</span>
                        <span class="value">{stats['sharpe_ratio']:.2f}</span>
                    </div>
                    <div class="metric">
                        <span class="label">Max Drawdown</span>
                        <span class="value negative">${stats['max_drawdown_value']:,.2f} ({stats['max_drawdown']:.2%})</span>
                    </div>
                    <div class="metric">
                        <span class="label">Total Trades</span>
                        <span class="value">{stats['total_trades']}</span>
                    </div>
                    <div class="metric">
                        <span class="label">Avg Winner</span>
                        <span class="value positive">${stats['avg_winner']:+,.2f}</span>
                    </div>
                    <div class="metric">
                        <span class="label">Avg Loser</span>
                        <span class="value negative">${stats['avg_loser']:+,.2f}</span>
                    </div>
                    <div class="metric">
                        <span class="label">Expectancy</span>
                        <span class="value {'positive' if stats['expectancy'] > 0 else 'negative'}">${stats['expectancy']:+,.2f}</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- Closed Trades Long -->
        <div class="section">
            <h2>Closed Trades - LONG ({len(long_closed)}) <span class="total {'negative' if sum(p.realized_pnl for p in long_closed) < 0 else ''}">Total: ${sum(p.realized_pnl for p in long_closed):+,.2f}</span></h2>
            <table>
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Direction</th>
                        <th>QTY</th>
                        <th>Entry</th>
                        <th>Entry Price</th>
                        <th>Exit</th>
                        <th>Exit Price</th>
                        <th>Fees</th>
                        <th>Days</th>
                        <th>P&L $</th>
                        <th>P&L %</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(self._generate_closed_trade_row(p) for p in long_closed[:50])}
                </tbody>
            </table>
            {f'<p style="color: #8899a6; margin-top: 10px;">... and {len(long_closed) - 50} more trades</p>' if len(long_closed) > 50 else ''}
        </div>

        <!-- Closed Trades Short -->
        <div class="section">
            <h2>Closed Trades - SHORT ({len(short_closed)}) <span class="total {'negative' if sum(p.realized_pnl for p in short_closed) < 0 else ''}">Total: ${sum(p.realized_pnl for p in short_closed):+,.2f}</span></h2>
            <table>
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Direction</th>
                        <th>QTY</th>
                        <th>Entry</th>
                        <th>Entry Price</th>
                        <th>Exit</th>
                        <th>Exit Price</th>
                        <th>Fees</th>
                        <th>Days</th>
                        <th>P&L $</th>
                        <th>P&L %</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(self._generate_closed_trade_row(p) for p in short_closed[:50])}
                </tbody>
            </table>
            {f'<p style="color: #8899a6; margin-top: 10px;">... and {len(short_closed) - 50} more trades</p>' if len(short_closed) > 50 else ''}
        </div>

        <!-- Open Positions Long -->
        <div class="section">
            <h2>Open Positions - LONG ({len(long_open)}) <span class="total {'negative' if sum(p.unrealized_pnl for p in long_open) < 0 else ''}">Total: ${sum(p.unrealized_pnl for p in long_open):+,.2f}</span></h2>
            <table>
                <thead>
                    <tr>
                        <th>DLY</th>
                        <th>Symbol</th>
                        <th>POS</th>
                        <th>MKT VAL</th>
                        <th>AVG PX</th>
                        <th>LAST</th>
                        <th>ENTRY</th>
                        <th>UNRLZD</th>
                        <th>Strategy</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(self._generate_open_trade_row(p) for p in long_open)}
                </tbody>
            </table>
        </div>

        <!-- Open Positions Short -->
        <div class="section">
            <h2>Open Positions - SHORT ({len(short_open)}) <span class="total {'negative' if sum(p.unrealized_pnl for p in short_open) < 0 else ''}">Total: ${sum(p.unrealized_pnl for p in short_open):+,.2f}</span></h2>
            <table>
                <thead>
                    <tr>
                        <th>DLY</th>
                        <th>Symbol</th>
                        <th>POS</th>
                        <th>MKT VAL</th>
                        <th>AVG PX</th>
                        <th>LAST</th>
                        <th>ENTRY</th>
                        <th>UNRLZD</th>
                        <th>Strategy</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(self._generate_open_trade_row(p) for p in short_open)}
                </tbody>
            </table>
        </div>
    </div>

    <script>
        // Equity Curve Chart
        var equityTrace = {{
            x: {json.dumps(equity_dates)},
            y: {json.dumps(equity_values)},
            type: 'scatter',
            mode: 'lines',
            fill: 'tozeroy',
            line: {{color: '#4ecdc4', width: 2}},
            fillcolor: 'rgba(78, 205, 196, 0.1)'
        }};

        var equityLayout = {{
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            margin: {{l: 60, r: 20, t: 20, b: 40}},
            xaxis: {{
                gridcolor: '#2a3f5f',
                tickfont: {{color: '#8899a6'}}
            }},
            yaxis: {{
                gridcolor: '#2a3f5f',
                tickfont: {{color: '#8899a6'}},
                tickprefix: '$'
            }}
        }};

        Plotly.newPlot('equity-chart', [equityTrace], equityLayout, {{responsive: true}});

        // Daily PnL Chart
        var dailyPnlColors = {json.dumps(daily_pnl_values)}.map(v => v >= 0 ? '#4ecdc4' : '#ff6b6b');
        var dailyPnlTrace = {{
            x: {json.dumps(equity_dates)},
            y: {json.dumps(daily_pnl_values)},
            type: 'bar',
            marker: {{color: dailyPnlColors}}
        }};

        var dailyPnlLayout = {{
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            margin: {{l: 60, r: 20, t: 20, b: 40}},
            xaxis: {{
                gridcolor: '#2a3f5f',
                tickfont: {{color: '#8899a6'}}
            }},
            yaxis: {{
                gridcolor: '#2a3f5f',
                tickfont: {{color: '#8899a6'}},
                tickprefix: '$'
            }}
        }};

        Plotly.newPlot('daily-pnl-chart', [dailyPnlTrace], dailyPnlLayout, {{responsive: true}});

        // Positions Bar Chart
        var positionSymbols = {json.dumps([p['symbol'] for p in position_pnl])};
        var positionPnLs = {json.dumps([p['pnl'] for p in position_pnl])};
        var positionColors = positionPnLs.map(v => v >= 0 ? '#4ecdc4' : '#ff6b6b');

        var positionsTrace = {{
            y: positionSymbols,
            x: positionPnLs,
            type: 'bar',
            orientation: 'h',
            marker: {{color: positionColors}}
        }};

        var positionsLayout = {{
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            margin: {{l: 60, r: 20, t: 20, b: 40}},
            xaxis: {{
                gridcolor: '#2a3f5f',
                tickfont: {{color: '#8899a6'}},
                tickprefix: '$'
            }},
            yaxis: {{
                gridcolor: '#2a3f5f',
                tickfont: {{color: '#8899a6', size: 10}}
            }}
        }};

        Plotly.newPlot('positions-bar-chart', [positionsTrace], positionsLayout, {{responsive: true}});
    </script>
</body>
</html>
'''

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html)

        print(f"\nHTML report saved to: {filename}")
        return filename

    def _generate_closed_trade_row(self, pos: Position) -> str:
        """Generate HTML table row for closed trade"""
        pnl_class = 'positive' if pos.realized_pnl >= 0 else 'negative'
        pnl_pct = pos.pnl_percent()
        total_fees = pos.entry_fee + pos.exit_fee

        return f'''
            <tr>
                <td><strong>{pos.symbol}</strong></td>
                <td><span class="direction {'long' if pos.direction == 'LONG' else 'short'}">{pos.direction}</span></td>
                <td>{pos.quantity:,}</td>
                <td>{pos.entry_date.strftime('%Y-%m-%d %H:%M') if pos.entry_date else ''}</td>
                <td>${pos.entry_price:.2f}</td>
                <td>{pos.exit_date.strftime('%Y-%m-%d %H:%M') if pos.exit_date else ''}</td>
                <td>${pos.exit_price:.2f}</td>
                <td>${total_fees:.2f}</td>
                <td>{pos.days_held()}d</td>
                <td class="{pnl_class}">${pos.realized_pnl:+,.2f}</td>
                <td class="{pnl_class}">{pnl_pct:+.2%}</td>
            </tr>
        '''

    def _generate_open_trade_row(self, pos: Position) -> str:
        """Generate HTML table row for open position"""
        daily_pnl = pos.get_daily_pnl(pos.entry_price)  # Simplified
        daily_class = 'positive' if daily_pnl >= 0 else 'negative'
        unrlzd_class = 'positive' if pos.unrealized_pnl >= 0 else 'negative'

        return f'''
            <tr>
                <td class="{daily_class}">${daily_pnl:+,.2f}</td>
                <td><strong>{pos.symbol}</strong></td>
                <td>{pos.quantity:,}</td>
                <td>${pos.get_market_value():,.2f}</td>
                <td>${pos.entry_price:.2f}</td>
                <td>${pos.current_price:.2f}</td>
                <td>{pos.entry_date.strftime('%m/%d %H:%M') if pos.entry_date else ''}</td>
                <td class="{unrlzd_class}">${pos.unrealized_pnl:+,.2f}</td>
                <td>{pos.strategy}</td>
            </tr>
        '''


def run_portfolio_simulation(days_back: int = 180):
    """Main function to run portfolio simulation"""
    simulator = PortfolioSimulator(
        initial_capital=20000.0,
        max_positions=30,
        bh_positions=10,
        strategy_positions=20,
        fee_rate=0.001,  # 0.1%
        trailing_stop_pct=0.20  # 20% trailing stop for B&H
    )

    simulator.run_simulation(days_back=days_back, rebalance_freq=5)
    simulator.generate_html_report("portfolio_report.html")

    disconnect_ib()

    return simulator


if __name__ == "__main__":
    import sys

    days = 180
    if len(sys.argv) > 1:
        try:
            days = int(sys.argv[1])
        except:
            pass

    run_portfolio_simulation(days_back=days)
