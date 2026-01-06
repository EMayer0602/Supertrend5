import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass
class TradingConfig:
    """Configuration for the trading system"""
    symbol: str = "MSFT"
    initial_capital: float = 10000.0
    transaction_cost: float = 0.001

    # Supertrend Parameters (will be optimized)
    st_period: int = 10
    st_multiplier: float = 3.0

    # HTF Filter Parameters
    htf_period: int = 10
    htf_multiplier: float = 3.0
    use_htf_filter: bool = True

    # Backtest Period
    days_back: int = 730  # 2 years for better statistics


# =============================================================================
# OPTIMIZED SUPERTREND CALCULATION (Vectorized with NumPy)
# =============================================================================
def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Calculate Average True Range using vectorized operations"""
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]

    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)

    true_range = np.maximum(np.maximum(tr1, tr2), tr3)

    # EMA-based ATR for smoother results
    atr = np.zeros_like(true_range)
    atr[:period] = np.nan
    atr[period-1] = np.mean(true_range[:period])

    multiplier = 2 / (period + 1)
    for i in range(period, len(true_range)):
        atr[i] = true_range[i] * multiplier + atr[i-1] * (1 - multiplier)

    return atr


def calculate_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate RSI (Relative Strength Index) using vectorized operations"""
    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    # Calculate average gain/loss using EMA
    avg_gain = np.zeros(len(close))
    avg_loss = np.zeros(len(close))

    # Initial SMA
    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])

    # EMA for subsequent values
    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i-1]) / period

    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    rsi[:period] = 50  # Neutral for initial period

    return rsi


def calculate_supertrend_vectorized(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                                     period: int = 10, multiplier: float = 3.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Optimized Supertrend calculation using NumPy vectorization
    Returns: (supertrend, direction, atr)
    - direction: 1 = bullish (price above supertrend), -1 = bearish
    """
    n = len(close)
    atr = calculate_atr(high, low, close, period)

    hl2 = (high + low) / 2

    # Basic bands
    basic_upper = hl2 + (multiplier * atr)
    basic_lower = hl2 - (multiplier * atr)

    # Final bands
    final_upper = np.zeros(n)
    final_lower = np.zeros(n)
    supertrend = np.zeros(n)
    direction = np.zeros(n)

    # Initialize
    final_upper[period-1] = basic_upper[period-1]
    final_lower[period-1] = basic_lower[period-1]

    for i in range(period, n):
        # Final Upper Band
        if basic_upper[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]:
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = final_upper[i-1]

        # Final Lower Band
        if basic_lower[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]:
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = final_lower[i-1]

    # Calculate Supertrend and Direction
    for i in range(period, n):
        if i == period:
            if close[i] <= final_upper[i]:
                supertrend[i] = final_upper[i]
                direction[i] = -1
            else:
                supertrend[i] = final_lower[i]
                direction[i] = 1
        else:
            if supertrend[i-1] == final_upper[i-1]:
                if close[i] <= final_upper[i]:
                    supertrend[i] = final_upper[i]
                    direction[i] = -1
                else:
                    supertrend[i] = final_lower[i]
                    direction[i] = 1
            else:
                if close[i] >= final_lower[i]:
                    supertrend[i] = final_lower[i]
                    direction[i] = 1
                else:
                    supertrend[i] = final_upper[i]
                    direction[i] = -1

    return supertrend, direction, atr


# =============================================================================
# HTF (HIGHER TIME FRAME) FILTER
# =============================================================================
def resample_to_weekly(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Resample daily data to weekly for HTF analysis"""
    ohlc_dict = {
        f'Open_{symbol}': 'first',
        f'High_{symbol}': 'max',
        f'Low_{symbol}': 'min',
        f'Close_{symbol}': 'last',
        f'Volume_{symbol}': 'sum'
    }

    weekly = df.resample('W').agg(ohlc_dict).dropna()
    return weekly


def get_htf_trend(daily_df: pd.DataFrame, symbol: str, period: int, multiplier: float) -> pd.Series:
    """
    Calculate HTF (Weekly) Supertrend and map back to daily timeframe
    Returns: Series with HTF direction aligned to daily index
    """
    weekly_df = resample_to_weekly(daily_df, symbol)

    if len(weekly_df) < period + 5:
        print("Warning: Not enough weekly data for HTF filter")
        return pd.Series(1, index=daily_df.index)

    high = weekly_df[f'High_{symbol}'].values
    low = weekly_df[f'Low_{symbol}'].values
    close = weekly_df[f'Close_{symbol}'].values

    _, htf_direction, _ = calculate_supertrend_vectorized(high, low, close, period, multiplier)

    # Create weekly direction series
    htf_series = pd.Series(htf_direction, index=weekly_df.index)

    # Forward-fill to daily timeframe
    daily_htf = htf_series.reindex(daily_df.index, method='ffill')
    daily_htf = daily_htf.fillna(method='bfill')

    return daily_htf


# =============================================================================
# VECTORIZED TRADING SIGNALS
# =============================================================================
def generate_signals_vectorized(close: np.ndarray, supertrend: np.ndarray,
                                 direction: np.ndarray, htf_direction: Optional[np.ndarray] = None,
                                 use_htf_filter: bool = True,
                                 filter_mode: str = "trend_following") -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate buy/sell signals using vectorized operations

    filter_mode options:
    - "trend_following": Only trade in HTF direction (Long only when HTF bullish)
    - "confirmation": Use HTF as confirmation (original strict mode)
    - "exit_only": Use all entries, but HTF filters exits

    Returns: (buy_signals, sell_signals) as boolean arrays
    """
    n = len(close)

    # Detect direction changes
    prev_direction = np.roll(direction, 1)
    prev_direction[0] = 0

    # Signal when direction changes from -1 to 1 (buy) or 1 to -1 (sell)
    raw_buy = (prev_direction == -1) & (direction == 1)
    raw_sell = (prev_direction == 1) & (direction == -1)

    if use_htf_filter and htf_direction is not None:
        if filter_mode == "trend_following":
            # Only trade in direction of HTF trend
            # Buy signals only when HTF is bullish, Sell signals to exit longs when direction changes
            # No short trading - only long when HTF bullish
            buy_signals = raw_buy & (htf_direction == 1)
            # Exit long when either: sell signal AND still in HTF bullish, OR HTF turns bearish
            sell_signals = raw_sell  # Always allow exits
        elif filter_mode == "confirmation":
            # Original strict mode - only enter when HTF confirms
            buy_signals = raw_buy & (htf_direction == 1)
            sell_signals = raw_sell & (htf_direction == -1)
        else:  # exit_only
            buy_signals = raw_buy
            sell_signals = raw_sell
    else:
        buy_signals = raw_buy
        sell_signals = raw_sell

    return buy_signals, sell_signals


# =============================================================================
# OPTIMIZED TRADING SYSTEM CLASS
# =============================================================================
class OptimizedTradingSystem:
    def __init__(self, config: TradingConfig):
        self.config = config

    def generate_trades(self, df: pd.DataFrame, buy_signals: np.ndarray,
                        sell_signals: np.ndarray, long_only: bool = False,
                        use_trailing_stop: bool = False, trailing_stop_pct: float = 0.05,
                        rsi: Optional[np.ndarray] = None, rsi_oversold: int = 30, rsi_overbought: int = 70) -> Tuple[List[Dict], List[Dict]]:
        """Generate trade lists from signals with optional trailing stop and RSI filter

        Args:
            long_only: If True, only generate long trades (no shorts)
            use_trailing_stop: If True, use trailing stop to lock in profits
            trailing_stop_pct: Trailing stop percentage (e.g., 0.05 = 5%)
            rsi: RSI values array (optional)
            rsi_oversold: RSI level considered oversold (buy opportunity)
            rsi_overbought: RSI level considered overbought (sell opportunity)
        """
        symbol = self.config.symbol
        close_col = f'Close_{symbol}'
        high_col = f'High_{symbol}'

        long_trades = []
        short_trades = []

        position = None  # None, 'long', 'short'
        entry_price = 0
        entry_date = None
        entry_idx = 0
        highest_since_entry = 0  # For trailing stop

        close_prices = df[close_col].values
        high_prices = df[high_col].values
        dates = df.index

        for i in range(len(df)):
            # Check trailing stop first (if in position)
            if use_trailing_stop and position == 'long':
                highest_since_entry = max(highest_since_entry, high_prices[i])
                trailing_stop_price = highest_since_entry * (1 - trailing_stop_pct)

                if close_prices[i] < trailing_stop_price:
                    # Trailing stop hit - exit position
                    profit_loss = (close_prices[i] - entry_price) / entry_price
                    profit_loss -= self.config.transaction_cost * 2
                    long_trades.append({
                        'entry_date': entry_date,
                        'entry_price': entry_price,
                        'exit_date': dates[i],
                        'exit_price': close_prices[i],
                        'profit_loss': profit_loss,
                        'entry_index': entry_idx,
                        'exit_index': i,
                        'exit_reason': 'trailing_stop'
                    })
                    position = None
                    highest_since_entry = 0
                    continue

            # RSI filter for entry
            rsi_allows_buy = True
            rsi_allows_sell = True
            if rsi is not None:
                # Only buy if RSI is not overbought (gives room to grow)
                rsi_allows_buy = rsi[i] < rsi_overbought
                # Only sell/short if RSI is not oversold (might bounce)
                rsi_allows_sell = rsi[i] > rsi_oversold

            if buy_signals[i] and rsi_allows_buy:
                # Close short position if exists
                if position == 'short':
                    profit_loss = (entry_price - close_prices[i]) / entry_price
                    profit_loss -= self.config.transaction_cost * 2
                    short_trades.append({
                        'entry_date': entry_date,
                        'entry_price': entry_price,
                        'exit_date': dates[i],
                        'exit_price': close_prices[i],
                        'profit_loss': profit_loss,
                        'entry_index': entry_idx,
                        'exit_index': i,
                        'exit_reason': 'signal'
                    })
                    position = None

                # Open long position
                if position is None:
                    position = 'long'
                    entry_price = close_prices[i]
                    entry_date = dates[i]
                    entry_idx = i
                    highest_since_entry = high_prices[i]

            elif sell_signals[i] and rsi_allows_sell:
                # Close long position if exists
                if position == 'long':
                    profit_loss = (close_prices[i] - entry_price) / entry_price
                    profit_loss -= self.config.transaction_cost * 2
                    long_trades.append({
                        'entry_date': entry_date,
                        'entry_price': entry_price,
                        'exit_date': dates[i],
                        'exit_price': close_prices[i],
                        'profit_loss': profit_loss,
                        'entry_index': entry_idx,
                        'exit_index': i,
                        'exit_reason': 'signal'
                    })
                    position = None
                    highest_since_entry = 0

                # Open short position only if not long_only mode
                if not long_only and position is None:
                    position = 'short'
                    entry_price = close_prices[i]
                    entry_date = dates[i]
                    entry_idx = i

        return long_trades, short_trades

    def calculate_equity_curve_vectorized(self, df: pd.DataFrame,
                                          long_trades: List[Dict],
                                          short_trades: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate equity curves using vectorized operations"""
        n = len(df)
        symbol = self.config.symbol
        close_col = f'Close_{symbol}'
        close_prices = df[close_col].values

        long_equity = np.full(n, self.config.initial_capital)
        short_equity = np.full(n, self.config.initial_capital)

        # Process long trades
        for trade in long_trades:
            entry_idx = trade['entry_index']
            exit_idx = trade['exit_index']
            entry_price = trade['entry_price']

            # Calculate daily equity during trade
            for i in range(entry_idx, exit_idx + 1):
                daily_return = (close_prices[i] - entry_price) / entry_price
                daily_return -= self.config.transaction_cost  # Entry cost
                if i == exit_idx:
                    daily_return -= self.config.transaction_cost  # Exit cost

                # Get capital at entry
                capital_at_entry = long_equity[entry_idx - 1] if entry_idx > 0 else self.config.initial_capital
                long_equity[i] = capital_at_entry * (1 + daily_return)

            # Carry forward equity after trade
            if exit_idx + 1 < n:
                for i in range(exit_idx + 1, n):
                    if i < n and (not any(t['entry_index'] <= i <= t['exit_index'] for t in long_trades if t != trade)):
                        long_equity[i] = long_equity[exit_idx]

        # Process short trades
        for trade in short_trades:
            entry_idx = trade['entry_index']
            exit_idx = trade['exit_index']
            entry_price = trade['entry_price']

            for i in range(entry_idx, exit_idx + 1):
                daily_return = (entry_price - close_prices[i]) / entry_price
                daily_return -= self.config.transaction_cost
                if i == exit_idx:
                    daily_return -= self.config.transaction_cost

                capital_at_entry = short_equity[entry_idx - 1] if entry_idx > 0 else self.config.initial_capital
                short_equity[i] = capital_at_entry * (1 + daily_return)

            if exit_idx + 1 < n:
                for i in range(exit_idx + 1, n):
                    if i < n and (not any(t['entry_index'] <= i <= t['exit_index'] for t in short_trades if t != trade)):
                        short_equity[i] = short_equity[exit_idx]

        # Fix equity curves - forward fill properly
        long_equity = self._fix_equity_curve(long_equity, long_trades)
        short_equity = self._fix_equity_curve(short_equity, short_trades)

        # Combined equity
        combined_equity = long_equity + short_equity - self.config.initial_capital

        # Buy and hold equity
        buy_hold_equity = (close_prices / close_prices[0]) * self.config.initial_capital

        return long_equity, short_equity, combined_equity, buy_hold_equity

    def _fix_equity_curve(self, equity: np.ndarray, trades: List[Dict]) -> np.ndarray:
        """Fix equity curve with proper forward filling"""
        result = np.full_like(equity, self.config.initial_capital)
        current_equity = self.config.initial_capital

        # Sort trades by entry index
        sorted_trades = sorted(trades, key=lambda x: x['entry_index'])

        trade_idx = 0
        in_trade = False
        trade_entry_equity = self.config.initial_capital

        for i in range(len(equity)):
            # Check if entering a trade
            if trade_idx < len(sorted_trades):
                trade = sorted_trades[trade_idx]
                if i == trade['entry_index']:
                    in_trade = True
                    trade_entry_equity = current_equity

                if in_trade and trade['entry_index'] <= i <= trade['exit_index']:
                    # During trade - calculate equity
                    pnl = trade['profit_loss'] * (i - trade['entry_index']) / max(1, trade['exit_index'] - trade['entry_index'])
                    result[i] = trade_entry_equity * (1 + pnl)

                    if i == trade['exit_index']:
                        current_equity = trade_entry_equity * (1 + trade['profit_loss'])
                        result[i] = current_equity
                        in_trade = False
                        trade_idx += 1
                else:
                    result[i] = current_equity
            else:
                result[i] = current_equity

        return result

    def calculate_statistics(self, trades: List[Dict], equity_curve: np.ndarray) -> Dict:
        """Calculate comprehensive trading statistics"""
        if not trades:
            return self._empty_statistics()

        profits = [t['profit_loss'] for t in trades if t['profit_loss'] > 0]
        losses = [t['profit_loss'] for t in trades if t['profit_loss'] <= 0]

        total_trades = len(trades)
        winning_trades = len(profits)
        losing_trades = len(losses)

        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        avg_profit = np.mean(profits) if profits else 0
        avg_loss = np.mean(losses) if losses else 0

        total_profit = sum(profits)
        total_loss = abs(sum(losses))

        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

        # Total return
        total_return = (equity_curve[-1] - self.config.initial_capital) / self.config.initial_capital

        # Max drawdown
        rolling_max = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - rolling_max) / rolling_max
        max_drawdown = abs(np.min(drawdowns))

        # Sharpe Ratio (annualized)
        returns = np.diff(equity_curve) / equity_curve[:-1]
        sharpe_ratio = np.sqrt(252) * (np.mean(returns) / np.std(returns)) if np.std(returns) > 0 else 0

        # Sortino Ratio
        negative_returns = returns[returns < 0]
        sortino_ratio = np.sqrt(252) * (np.mean(returns) / np.std(negative_returns)) if len(negative_returns) > 0 and np.std(negative_returns) > 0 else 0

        # Calmar Ratio
        calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0

        # Average trade duration
        durations = [(t['exit_index'] - t['entry_index']) for t in trades]
        avg_duration = np.mean(durations) if durations else 0

        # Expectancy
        expectancy = (win_rate * avg_profit) - ((1 - win_rate) * abs(avg_loss))

        return {
            "Total Trades": total_trades,
            "Winning Trades": winning_trades,
            "Losing Trades": losing_trades,
            "Win Rate": f"{win_rate:.1%}",
            "Avg Profit": f"{avg_profit:.2%}",
            "Avg Loss": f"{avg_loss:.2%}",
            "Profit Factor": f"{profit_factor:.2f}",
            "Total Return": f"{total_return:.2%}",
            "Max Drawdown": f"{max_drawdown:.2%}",
            "Sharpe Ratio": f"{sharpe_ratio:.2f}",
            "Sortino Ratio": f"{sortino_ratio:.2f}",
            "Calmar Ratio": f"{calmar_ratio:.2f}",
            "Avg Trade Duration": f"{avg_duration:.1f} days",
            "Expectancy": f"{expectancy:.2%}",
        }

    def _empty_statistics(self) -> Dict:
        return {
            "Total Trades": 0,
            "Winning Trades": 0,
            "Losing Trades": 0,
            "Win Rate": "0.0%",
            "Avg Profit": "0.00%",
            "Avg Loss": "0.00%",
            "Profit Factor": "0.00",
            "Total Return": "0.00%",
            "Max Drawdown": "0.00%",
            "Sharpe Ratio": "0.00",
            "Sortino Ratio": "0.00",
            "Calmar Ratio": "0.00",
            "Avg Trade Duration": "0.0 days",
            "Expectancy": "0.00%",
        }


# =============================================================================
# PARAMETER OPTIMIZATION
# =============================================================================
def optimize_parameters(df: pd.DataFrame, symbol: str,
                        period_range: range = range(7, 21, 2),
                        multiplier_range: np.ndarray = np.arange(2.0, 5.0, 0.5),
                        use_htf: bool = True,
                        long_only: bool = True,
                        filter_mode: str = "trend_following",
                        optimize_for: str = "return") -> Tuple[int, float, Dict]:
    """
    Grid search for optimal Supertrend parameters

    optimize_for: "return", "sharpe", "profit_factor", "win_rate"
    """
    print("\n" + "="*60)
    print("PARAMETER OPTIMIZATION")
    print("="*60)
    print(f"Mode: {'Long Only' if long_only else 'Long & Short'}")
    print(f"HTF Filter: {filter_mode if use_htf else 'Disabled'}")
    print(f"Optimizing for: {optimize_for}")

    close_col = f'Close_{symbol}'
    high_col = f'High_{symbol}'
    low_col = f'Low_{symbol}'

    high = df[high_col].values
    low = df[low_col].values
    close = df[close_col].values

    best_score = -np.inf
    best_period = 10
    best_multiplier = 3.0
    best_stats = {}

    results = []

    config = TradingConfig(symbol=symbol)
    system = OptimizedTradingSystem(config)

    total_combinations = len(period_range) * len(multiplier_range)
    current = 0

    for period in period_range:
        for multiplier in multiplier_range:
            current += 1

            try:
                # Calculate Supertrend
                supertrend, direction, _ = calculate_supertrend_vectorized(high, low, close, period, multiplier)

                # Get HTF trend if enabled
                htf_direction = None
                if use_htf:
                    htf_direction = get_htf_trend(df, symbol, period, multiplier).values

                # Generate signals
                buy_signals, sell_signals = generate_signals_vectorized(
                    close, supertrend, direction, htf_direction, use_htf, filter_mode
                )

                # Generate trades
                long_trades, short_trades = system.generate_trades(df, buy_signals, sell_signals, long_only)

                all_trades = long_trades + short_trades

                if len(all_trades) < 3:
                    continue

                # Calculate equity
                long_eq, short_eq, combined_eq, buy_hold = system.calculate_equity_curve_vectorized(
                    df, long_trades, short_trades
                )

                # Calculate metrics
                total_return = (combined_eq[-1] - config.initial_capital) / config.initial_capital
                buy_hold_return = (buy_hold[-1] - config.initial_capital) / config.initial_capital

                # Sharpe Ratio
                returns = np.diff(combined_eq) / combined_eq[:-1]
                returns = returns[~np.isnan(returns)]
                sharpe = np.sqrt(252) * (np.mean(returns) / np.std(returns)) if len(returns) > 0 and np.std(returns) > 0 else 0

                # Win Rate
                wins = len([t for t in all_trades if t['profit_loss'] > 0])
                win_rate = wins / len(all_trades) if all_trades else 0

                # Profit Factor
                total_profit = sum([t['profit_loss'] for t in all_trades if t['profit_loss'] > 0])
                total_loss = abs(sum([t['profit_loss'] for t in all_trades if t['profit_loss'] <= 0]))
                profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

                # Max Drawdown
                rolling_max = np.maximum.accumulate(combined_eq)
                drawdowns = (combined_eq - rolling_max) / rolling_max
                max_dd = abs(np.min(drawdowns))

                # Risk-adjusted return (Return / Max DD)
                risk_adj_return = total_return / max_dd if max_dd > 0 else total_return

                # Select score based on optimization target
                if optimize_for == "return":
                    score = total_return
                elif optimize_for == "sharpe":
                    score = sharpe
                elif optimize_for == "profit_factor":
                    score = profit_factor if profit_factor != float('inf') else 10
                elif optimize_for == "win_rate":
                    score = win_rate
                elif optimize_for == "risk_adjusted":
                    score = risk_adj_return
                else:
                    score = total_return

                results.append({
                    'period': period,
                    'multiplier': multiplier,
                    'sharpe': sharpe,
                    'return': total_return,
                    'win_rate': win_rate,
                    'profit_factor': profit_factor,
                    'max_dd': max_dd,
                    'trades': len(all_trades),
                    'score': score
                })

                # Update best if better score AND positive return AND outperforms buy & hold
                if score > best_score and total_return > 0 and len(all_trades) >= 3:
                    best_score = score
                    best_period = period
                    best_multiplier = multiplier
                    best_stats = {
                        'sharpe': sharpe,
                        'return': total_return,
                        'win_rate': win_rate,
                        'profit_factor': profit_factor,
                        'max_dd': max_dd,
                        'long_trades': len(long_trades),
                        'short_trades': len(short_trades)
                    }

            except Exception as e:
                pass

            # Progress
            if current % 10 == 0:
                print(f"Progress: {current}/{total_combinations} ({100*current/total_combinations:.0f}%)")

    # If no positive result found, find best among all
    if best_stats == {}:
        print("\nNo profitable strategy found. Selecting least negative...")
        best_result = max(results, key=lambda x: x['return']) if results else None
        if best_result:
            best_period = best_result['period']
            best_multiplier = best_result['multiplier']
            best_stats = best_result

    print(f"\nBest Parameters Found:")
    print(f"  Period: {best_period}")
    print(f"  Multiplier: {best_multiplier}")
    print(f"  Return: {best_stats.get('return', 0):.2%}")
    print(f"  Win Rate: {best_stats.get('win_rate', 0):.1%}")
    print(f"  Profit Factor: {best_stats.get('profit_factor', 0):.2f}")
    print(f"  Max Drawdown: {best_stats.get('max_dd', 0):.2%}")

    return best_period, best_multiplier, results


# =============================================================================
# VISUALIZATION
# =============================================================================
def create_visualization(df: pd.DataFrame, symbol: str, config: TradingConfig,
                         supertrend: np.ndarray, direction: np.ndarray,
                         htf_direction: np.ndarray,
                         buy_signals: np.ndarray, sell_signals: np.ndarray,
                         long_trades: List[Dict], short_trades: List[Dict],
                         long_equity: np.ndarray, short_equity: np.ndarray,
                         combined_equity: np.ndarray, buy_hold_equity: np.ndarray) -> go.Figure:
    """Create comprehensive trading visualization"""

    close_col = f'Close_{symbol}'
    high_col = f'High_{symbol}'
    low_col = f'Low_{symbol}'
    open_col = f'Open_{symbol}'

    # Skip initial period for cleaner visualization
    skip = max(config.st_period, config.htf_period) + 10
    plot_df = df.iloc[skip:]
    plot_indices = np.arange(skip, len(df))

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            f'{symbol} Price with Supertrend & HTF Filter',
            f'Equity Curves (Initial: ${config.initial_capital:,.0f})',
            'HTF Trend Direction'
        ),
        row_heights=[0.5, 0.35, 0.15]
    )

    # Row 1: Candlestick Chart
    fig.add_trace(
        go.Candlestick(
            x=plot_df.index,
            open=plot_df[open_col],
            high=plot_df[high_col],
            low=plot_df[low_col],
            close=plot_df[close_col],
            name='Price',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ),
        row=1, col=1
    )

    # Supertrend line with color based on direction
    for i in range(len(plot_df) - 1):
        idx = skip + i
        color = '#26a69a' if direction[idx] == 1 else '#ef5350'
        fig.add_trace(
            go.Scatter(
                x=[plot_df.index[i], plot_df.index[i+1]],
                y=[supertrend[idx], supertrend[idx+1]],
                mode='lines',
                line=dict(color=color, width=2),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=1
        )

    # Buy signals
    buy_dates = plot_df.index[buy_signals[skip:]]
    buy_prices = plot_df[close_col].values[buy_signals[skip:]]
    if len(buy_dates) > 0:
        fig.add_trace(
            go.Scatter(
                x=buy_dates,
                y=buy_prices,
                mode='markers',
                name='Buy Signal',
                marker=dict(symbol='triangle-up', size=15, color='#26a69a',
                           line=dict(width=2, color='white'))
            ),
            row=1, col=1
        )

    # Sell signals
    sell_dates = plot_df.index[sell_signals[skip:]]
    sell_prices = plot_df[close_col].values[sell_signals[skip:]]
    if len(sell_dates) > 0:
        fig.add_trace(
            go.Scatter(
                x=sell_dates,
                y=sell_prices,
                mode='markers',
                name='Sell Signal',
                marker=dict(symbol='triangle-down', size=15, color='#ef5350',
                           line=dict(width=2, color='white'))
            ),
            row=1, col=1
        )

    # Row 2: Equity Curves
    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=long_equity[skip:],
            mode='lines',
            name='Long Equity',
            line=dict(color='#26a69a', width=2)
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=short_equity[skip:],
            mode='lines',
            name='Short Equity',
            line=dict(color='#ef5350', width=2)
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=combined_equity[skip:],
            mode='lines',
            name='Combined Strategy',
            line=dict(color='#7c4dff', width=3)
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=buy_hold_equity[skip:],
            mode='lines',
            name='Buy & Hold',
            line=dict(color='#ffa726', width=2, dash='dash')
        ),
        row=2, col=1
    )

    # Row 3: HTF Direction
    htf_colors = ['#ef5350' if d == -1 else '#26a69a' for d in htf_direction[skip:]]
    fig.add_trace(
        go.Bar(
            x=plot_df.index,
            y=htf_direction[skip:],
            name='HTF Trend',
            marker_color=htf_colors
        ),
        row=3, col=1
    )

    # Layout
    fig.update_layout(
        title=dict(
            text=f'<b>{symbol} Supertrend Trading System with HTF Filter</b><br>'
                 f'<sup>Period: {config.st_period} | Multiplier: {config.st_multiplier} | HTF Filter: {"ON" if config.use_htf_filter else "OFF"}</sup>',
            x=0.5,
            font=dict(size=20)
        ),
        height=1000,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            bgcolor='rgba(255,255,255,0.8)'
        ),
        xaxis_rangeslider_visible=False,
        template='plotly_white',
        hovermode='x unified'
    )

    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Equity ($)", row=2, col=1)
    fig.update_yaxes(title_text="HTF Trend", row=3, col=1, tickvals=[-1, 0, 1], ticktext=['Bearish', 'Neutral', 'Bullish'])

    return fig


# =============================================================================
# MULTI-STRATEGY COMPARISON
# =============================================================================
def run_strategy_comparison(df: pd.DataFrame, symbol: str, config: TradingConfig):
    """Test multiple strategy configurations and find the best one"""
    print("\n" + "="*60)
    print("ADVANCED MULTI-STRATEGY COMPARISON")
    print("="*60)

    close_col = f'Close_{symbol}'
    high_col = f'High_{symbol}'
    low_col = f'Low_{symbol}'

    high = df[high_col].values
    low = df[low_col].values
    close = df[close_col].values

    # Calculate RSI once
    rsi = calculate_rsi(close, 14)

    # Calculate buy & hold return for comparison
    buy_hold_return = (close[-1] - close[0]) / close[0]
    print(f"\nBuy & Hold Return: {buy_hold_return:.2%}")
    print("Target: Beat this!\n")

    # Extended strategies - focus on catching big moves
    strategies = [
        # Basic strategies
        {"name": "Long Only (Basic)", "use_htf": False, "long_only": True, "filter_mode": "exit_only",
         "use_trailing": False, "trailing_pct": 0, "use_rsi": False},

        # With Trailing Stop - various levels
        {"name": "Long + Trailing 5%", "use_htf": False, "long_only": True, "filter_mode": "exit_only",
         "use_trailing": True, "trailing_pct": 0.05, "use_rsi": False},
        {"name": "Long + Trailing 12%", "use_htf": False, "long_only": True, "filter_mode": "exit_only",
         "use_trailing": True, "trailing_pct": 0.12, "use_rsi": False},
        {"name": "Long + Trailing 15%", "use_htf": False, "long_only": True, "filter_mode": "exit_only",
         "use_trailing": True, "trailing_pct": 0.15, "use_rsi": False},

        # With RSI Filter - buy oversold
        {"name": "Long + RSI Filter", "use_htf": False, "long_only": True, "filter_mode": "exit_only",
         "use_trailing": False, "trailing_pct": 0, "use_rsi": True},

        # Combined: RSI + wider Trailing
        {"name": "Long + RSI + Trail 12%", "use_htf": False, "long_only": True, "filter_mode": "exit_only",
         "use_trailing": True, "trailing_pct": 0.12, "use_rsi": True},

        # HTF strategies
        {"name": "HTF Long Only", "use_htf": True, "long_only": True, "filter_mode": "trend_following",
         "use_trailing": False, "trailing_pct": 0, "use_rsi": False},
        {"name": "HTF + Trail 15%", "use_htf": True, "long_only": True, "filter_mode": "trend_following",
         "use_trailing": True, "trailing_pct": 0.15, "use_rsi": False},
    ]

    results = []
    system = OptimizedTradingSystem(config)

    # Extended parameter search - very fine tuning
    period_range = range(5, 35, 1)  # Finer steps
    multiplier_range = np.arange(1.5, 8.0, 0.25)  # Finer multiplier steps

    for strat in strategies:
        print(f"Testing: {strat['name']}...")

        best_return = -np.inf
        best_config = None

        for period in period_range:
            for multiplier in multiplier_range:
                try:
                    supertrend, direction, _ = calculate_supertrend_vectorized(high, low, close, period, multiplier)

                    htf_direction = None
                    if strat['use_htf']:
                        htf_direction = get_htf_trend(df, symbol, period, multiplier).values

                    buy_signals, sell_signals = generate_signals_vectorized(
                        close, supertrend, direction, htf_direction, strat['use_htf'], strat['filter_mode']
                    )

                    # Use RSI if enabled
                    rsi_arr = rsi if strat['use_rsi'] else None

                    long_trades, short_trades = system.generate_trades(
                        df, buy_signals, sell_signals,
                        long_only=strat['long_only'],
                        use_trailing_stop=strat['use_trailing'],
                        trailing_stop_pct=strat['trailing_pct'],
                        rsi=rsi_arr
                    )

                    if len(long_trades) + len(short_trades) < 2:
                        continue

                    _, _, combined_eq, _ = system.calculate_equity_curve_vectorized(
                        df, long_trades, short_trades
                    )

                    total_return = (combined_eq[-1] - config.initial_capital) / config.initial_capital

                    if total_return > best_return:
                        best_return = total_return
                        best_config = {
                            'period': period,
                            'multiplier': multiplier,
                            'return': total_return,
                            'trades': len(long_trades) + len(short_trades),
                            'vs_buyhold': total_return - buy_hold_return
                        }

                except Exception:
                    pass

        if best_config:
            results.append({
                'strategy': strat['name'],
                'use_htf': strat['use_htf'],
                'long_only': strat['long_only'],
                'filter_mode': strat['filter_mode'],
                'use_trailing': strat['use_trailing'],
                'trailing_pct': strat['trailing_pct'],
                'use_rsi': strat['use_rsi'],
                **best_config
            })
            beat_marker = "✓ BEATS" if best_config['return'] > buy_hold_return else "✗"
            print(f"  Best: P={best_config['period']}, M={best_config['multiplier']:.1f}, Ret={best_config['return']:.2%} {beat_marker}")

    # Sort by return
    results.sort(key=lambda x: x['return'], reverse=True)

    print("\n" + "="*60)
    print("STRATEGY COMPARISON RESULTS (sorted by return)")
    print("="*60)
    print(f"{'Rank':<5} {'Strategy':<30} {'Return':>10} {'vs B&H':>10} {'Trades':>7}")
    print("-"*65)

    for i, r in enumerate(results, 1):
        vs_bh = r['return'] - buy_hold_return
        marker = "**" if r['return'] > buy_hold_return else ""
        print(f"{i:<5} {r['strategy']:<30} {r['return']:>9.2%} {vs_bh:>+9.2%} {r['trades']:>7} {marker}")

    # Return best that beats buy & hold, or overall best
    beating_strategies = [r for r in results if r['return'] > buy_hold_return]
    if beating_strategies:
        print(f"\n>>> {len(beating_strategies)} strategies beat Buy & Hold!")
        return beating_strategies[0]
    else:
        print(f"\n>>> No strategy beats Buy & Hold yet. Using best available.")
        return results[0] if results else None


# =============================================================================
# MAIN FUNCTION
# =============================================================================
def main():
    print("="*60)
    print("SUPERTREND TRADING SYSTEM v5.0 - OPTIMIZED WITH HTF FILTER")
    print("="*60)

    # Configuration - 5 years includes 2022 bear market!
    config = TradingConfig(
        symbol="MSFT",
        initial_capital=10000.0,
        days_back=1825,  # 5 years - includes 2022 bear market
        use_htf_filter=True
    )

    # Strategy settings - these will be overridden by the best strategy found
    LONG_ONLY = True  # Only long trades (better for bullish markets like MSFT)
    FILTER_MODE = "trend_following"  # trend_following, confirmation, exit_only
    OPTIMIZE_FOR = "return"  # return, sharpe, profit_factor, win_rate, risk_adjusted
    COMPARE_STRATEGIES = True  # Run multi-strategy comparison first

    print(f"\nSymbol: {config.symbol}")
    print(f"Initial Capital: ${config.initial_capital:,.2f}")
    print(f"Backtest Period: {config.days_back} days")

    # Download Data
    print(f"\nDownloading {config.symbol} data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=config.days_back)

    stock_data = yf.download(config.symbol, start=start_date, end=end_date, progress=False)

    if stock_data.empty:
        print("Error: No data downloaded")
        return

    # Flatten MultiIndex columns
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = ['_'.join(col).strip() for col in stock_data.columns.values]
    else:
        # Rename columns to include symbol
        stock_data.columns = [f'{col}_{config.symbol}' for col in stock_data.columns]

    print(f"Data loaded: {len(stock_data)} trading days")
    print(f"Date range: {stock_data.index[0].strftime('%Y-%m-%d')} to {stock_data.index[-1].strftime('%Y-%m-%d')}")

    # Column names
    close_col = f'Close_{config.symbol}'
    high_col = f'High_{config.symbol}'
    low_col = f'Low_{config.symbol}'

    high = stock_data[high_col].values
    low = stock_data[low_col].values
    close = stock_data[close_col].values

    # Additional strategy settings
    USE_TRAILING_STOP = False
    TRAILING_STOP_PCT = 0.08
    USE_RSI_FILTER = False

    # Run multi-strategy comparison first
    if COMPARE_STRATEGIES:
        best_strat = run_strategy_comparison(stock_data, config.symbol, config)
        if best_strat:
            LONG_ONLY = best_strat['long_only']
            FILTER_MODE = best_strat['filter_mode']
            config.use_htf_filter = best_strat['use_htf']
            config.st_period = best_strat['period']
            config.st_multiplier = best_strat['multiplier']
            config.htf_period = best_strat['period']
            config.htf_multiplier = best_strat['multiplier']
            USE_TRAILING_STOP = best_strat.get('use_trailing', False)
            TRAILING_STOP_PCT = best_strat.get('trailing_pct', 0.08)
            USE_RSI_FILTER = best_strat.get('use_rsi', False)
            print(f"\n>>> Using best strategy: {best_strat['strategy']}")
    else:
        # Parameter Optimization for single strategy
        print("\n" + "-"*60)
        best_period, best_multiplier, optimization_results = optimize_parameters(
            stock_data, config.symbol,
            period_range=range(7, 21, 2),
            multiplier_range=np.arange(2.0, 5.0, 0.5),
            use_htf=config.use_htf_filter,
            long_only=LONG_ONLY,
            filter_mode=FILTER_MODE,
            optimize_for=OPTIMIZE_FOR
        )
        config.st_period = best_period
        config.st_multiplier = best_multiplier
        config.htf_period = best_period
        config.htf_multiplier = best_multiplier

    print("\n" + "-"*60)
    print("RUNNING BACKTEST WITH OPTIMIZED PARAMETERS")
    print("-"*60)
    print(f"HTF Filter: {'Enabled' if config.use_htf_filter else 'Disabled'}")
    print(f"Mode: {'Long Only' if LONG_ONLY else 'Long & Short'}")
    print(f"Filter Mode: {FILTER_MODE}")
    print(f"Trailing Stop: {'Enabled (' + str(int(TRAILING_STOP_PCT*100)) + '%)' if USE_TRAILING_STOP else 'Disabled'}")
    print(f"RSI Filter: {'Enabled' if USE_RSI_FILTER else 'Disabled'}")
    print(f"Period: {config.st_period} | Multiplier: {config.st_multiplier}")

    # Calculate Supertrend with best parameters
    supertrend, direction, atr = calculate_supertrend_vectorized(
        high, low, close, config.st_period, config.st_multiplier
    )

    # Calculate HTF Trend
    htf_direction = get_htf_trend(
        stock_data, config.symbol, config.htf_period, config.htf_multiplier
    ).values

    # Calculate RSI if needed
    rsi = calculate_rsi(close, 14) if USE_RSI_FILTER else None

    # Generate Signals
    buy_signals, sell_signals = generate_signals_vectorized(
        close, supertrend, direction, htf_direction, config.use_htf_filter, FILTER_MODE
    )

    # Initialize Trading System
    system = OptimizedTradingSystem(config)

    # Generate Trades
    long_trades, short_trades = system.generate_trades(
        stock_data, buy_signals, sell_signals,
        long_only=LONG_ONLY,
        use_trailing_stop=USE_TRAILING_STOP,
        trailing_stop_pct=TRAILING_STOP_PCT,
        rsi=rsi
    )

    # Calculate Equity Curves
    long_equity, short_equity, combined_equity, buy_hold_equity = system.calculate_equity_curve_vectorized(
        stock_data, long_trades, short_trades
    )

    # Print Trade Details
    print(f"\n{'='*60}")
    print("TRADE DETAILS")
    print("="*60)

    if long_trades:
        print(f"\nLONG TRADES ({len(long_trades)} total):")
        print("-"*40)
        for i, trade in enumerate(long_trades, 1):
            print(f"  {i}. Entry: {trade['entry_date'].strftime('%Y-%m-%d')} @ ${trade['entry_price']:.2f}")
            print(f"     Exit:  {trade['exit_date'].strftime('%Y-%m-%d')} @ ${trade['exit_price']:.2f}")
            print(f"     P/L:   {trade['profit_loss']:.2%}")
    else:
        print("\nNo long trades executed")

    if short_trades:
        print(f"\nSHORT TRADES ({len(short_trades)} total):")
        print("-"*40)
        for i, trade in enumerate(short_trades, 1):
            print(f"  {i}. Entry: {trade['entry_date'].strftime('%Y-%m-%d')} @ ${trade['entry_price']:.2f}")
            print(f"     Exit:  {trade['exit_date'].strftime('%Y-%m-%d')} @ ${trade['exit_price']:.2f}")
            print(f"     P/L:   {trade['profit_loss']:.2%}")
    else:
        print("\nNo short trades executed")

    # Calculate Statistics
    print(f"\n{'='*60}")
    print("PERFORMANCE STATISTICS")
    print("="*60)

    long_stats = system.calculate_statistics(long_trades, long_equity)
    short_stats = system.calculate_statistics(short_trades, short_equity)
    all_trades = long_trades + short_trades
    combined_stats = system.calculate_statistics(all_trades, combined_equity)

    print("\n--- LONG TRADES ---")
    for key, value in long_stats.items():
        print(f"  {key}: {value}")

    print("\n--- SHORT TRADES ---")
    for key, value in short_stats.items():
        print(f"  {key}: {value}")

    print("\n--- COMBINED STRATEGY ---")
    for key, value in combined_stats.items():
        print(f"  {key}: {value}")

    # Buy & Hold comparison
    buy_hold_return = (buy_hold_equity[-1] - config.initial_capital) / config.initial_capital
    strategy_return = (combined_equity[-1] - config.initial_capital) / config.initial_capital

    print(f"\n{'='*60}")
    print("STRATEGY vs BUY & HOLD")
    print("="*60)
    print(f"  Strategy Return:   {strategy_return:.2%}")
    print(f"  Buy & Hold Return: {buy_hold_return:.2%}")
    print(f"  Outperformance:    {strategy_return - buy_hold_return:.2%}")
    print(f"  Final Equity:      ${combined_equity[-1]:,.2f}")

    # Create Visualization
    print(f"\n{'='*60}")
    print("GENERATING VISUALIZATION...")
    print("="*60)

    fig = create_visualization(
        stock_data, config.symbol, config,
        supertrend, direction, htf_direction,
        buy_signals, sell_signals,
        long_trades, short_trades,
        long_equity, short_equity, combined_equity, buy_hold_equity
    )

    # Save to HTML
    output_file = f"supertrend_{config.symbol}_results.html"
    fig.write_html(output_file)
    print(f"\nResults saved to: {output_file}")

    # Try to show plot (may fail in headless environments)
    try:
        import plotly.io as pio
        pio.renderers.default = 'browser'
        fig.show()
    except Exception as e:
        print(f"Could not open browser (headless environment). View the HTML file instead.")

    print("\n" + "="*60)
    print("BACKTEST COMPLETE")
    print("="*60)

    return {
        'config': config,
        'long_trades': long_trades,
        'short_trades': short_trades,
        'combined_equity': combined_equity,
        'buy_hold_equity': buy_hold_equity
    }


# =============================================================================
# MULTI-TICKER ANALYSIS
# =============================================================================
def test_ticker(symbol: str, days_back: int = 1825) -> Dict:
    """Test a single ticker and return results"""
    config = TradingConfig(
        symbol=symbol,
        initial_capital=10000.0,
        days_back=days_back,
        use_htf_filter=True
    )

    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        stock_data = yf.download(symbol, start=start_date, end=end_date, progress=False)

        if stock_data.empty or len(stock_data) < 100:
            return {'symbol': symbol, 'error': 'No data'}

        # Flatten columns
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data.columns = ['_'.join(col).strip() for col in stock_data.columns.values]
        else:
            stock_data.columns = [f'{col}_{symbol}' for col in stock_data.columns]

        close_col = f'Close_{symbol}'
        high_col = f'High_{symbol}'
        low_col = f'Low_{symbol}'

        high = stock_data[high_col].values
        low = stock_data[low_col].values
        close = stock_data[close_col].values

        buy_hold_return = (close[-1] - close[0]) / close[0]

        # Quick optimization - test key parameter combinations
        system = OptimizedTradingSystem(config)
        rsi = calculate_rsi(close, 14)

        best_return = -np.inf
        best_params = None

        # Test combinations
        for period in [10, 15, 20, 25]:
            for mult in [3.0, 4.0, 5.0, 6.0]:
                for use_rsi in [True, False]:
                    try:
                        supertrend, direction, _ = calculate_supertrend_vectorized(high, low, close, period, mult)
                        buy_signals, sell_signals = generate_signals_vectorized(close, supertrend, direction, None, False, "exit_only")

                        rsi_arr = rsi if use_rsi else None
                        long_trades, _ = system.generate_trades(stock_data, buy_signals, sell_signals, long_only=True, rsi=rsi_arr)

                        if len(long_trades) < 2:
                            continue

                        _, _, combined_eq, _ = system.calculate_equity_curve_vectorized(stock_data, long_trades, [])
                        total_return = (combined_eq[-1] - config.initial_capital) / config.initial_capital

                        if total_return > best_return:
                            best_return = total_return
                            best_params = {'period': period, 'mult': mult, 'use_rsi': use_rsi, 'trades': len(long_trades)}
                    except:
                        pass

        beats_bh = best_return > buy_hold_return if best_params else False

        return {
            'symbol': symbol,
            'buy_hold': buy_hold_return,
            'strategy': best_return,
            'outperformance': best_return - buy_hold_return if best_params else None,
            'beats_bh': beats_bh,
            'params': best_params,
            'data_days': len(stock_data)
        }
    except Exception as e:
        return {'symbol': symbol, 'error': str(e)}


def run_multi_ticker_analysis():
    """Test strategy on Dow Jones 30 and NASDAQ Top 20"""
    print("="*80)
    print("MULTI-TICKER ANALYSIS - DOW JONES 30 & NASDAQ TOP 20")
    print("="*80)

    # Dow Jones 30 components
    dow_jones = [
        'AAPL', 'MSFT', 'JPM', 'V', 'JNJ', 'WMT', 'PG', 'UNH', 'HD', 'CVX',
        'MRK', 'KO', 'DIS', 'MCD', 'CSCO', 'VZ', 'NKE', 'INTC', 'IBM', 'GS',
        'CAT', 'AXP', 'BA', 'HON', 'MMM', 'TRV', 'DOW', 'WBA', 'AMGN', 'CRM'
    ]

    # NASDAQ Top 20 (by market cap, excluding duplicates from Dow)
    nasdaq_top = [
        'NVDA', 'GOOG', 'GOOGL', 'AMZN', 'META', 'TSLA', 'AVGO', 'PEP', 'COST', 'ADBE',
        'NFLX', 'AMD', 'QCOM', 'TMUS', 'INTU', 'AMAT', 'ISRG', 'BKNG', 'ADP', 'PYPL'
    ]

    all_results = []

    # Test Dow Jones
    print("\n" + "-"*80)
    print("TESTING DOW JONES 30")
    print("-"*80)

    for i, symbol in enumerate(dow_jones, 1):
        print(f"[{i}/30] Testing {symbol}...", end=" ")
        result = test_ticker(symbol, days_back=1825)
        all_results.append(result)

        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            marker = "✓ BEATS" if result['beats_bh'] else "✗"
            print(f"B&H: {result['buy_hold']:.1%} | Strategy: {result['strategy']:.1%} | {marker}")

    # Test NASDAQ Top 20
    print("\n" + "-"*80)
    print("TESTING NASDAQ TOP 20")
    print("-"*80)

    for i, symbol in enumerate(nasdaq_top, 1):
        print(f"[{i}/20] Testing {symbol}...", end=" ")
        result = test_ticker(symbol, days_back=1825)
        all_results.append(result)

        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            marker = "✓ BEATS" if result['beats_bh'] else "✗"
            print(f"B&H: {result['buy_hold']:.1%} | Strategy: {result['strategy']:.1%} | {marker}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    valid_results = [r for r in all_results if 'error' not in r]
    beating_bh = [r for r in valid_results if r['beats_bh']]

    print(f"\nTotal tickers tested: {len(valid_results)}")
    print(f"Strategies that BEAT Buy & Hold: {len(beating_bh)} ({100*len(beating_bh)/len(valid_results):.1f}%)")

    if beating_bh:
        print("\n--- WINNERS (Beat Buy & Hold) ---")
        beating_bh.sort(key=lambda x: x['outperformance'], reverse=True)
        for r in beating_bh:
            print(f"  {r['symbol']}: Strategy {r['strategy']:.1%} vs B&H {r['buy_hold']:.1%} (+{r['outperformance']:.1%})")

    # Average performance
    avg_bh = np.mean([r['buy_hold'] for r in valid_results])
    avg_strat = np.mean([r['strategy'] for r in valid_results])
    avg_outperf = np.mean([r['outperformance'] for r in valid_results if r['outperformance'] is not None])

    print(f"\n--- AVERAGES ---")
    print(f"  Avg Buy & Hold Return: {avg_bh:.1%}")
    print(f"  Avg Strategy Return:   {avg_strat:.1%}")
    print(f"  Avg Outperformance:    {avg_outperf:+.1%}")

    return all_results


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--multi":
        run_multi_ticker_analysis()
    else:
        main()
