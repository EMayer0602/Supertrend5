import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

pio.renderers.default = 'browser'

# Function to flatten dataframe
def flatten_dataframe(df):
    flattened_df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        flattened_df.columns = ['_'.join(col).strip() for col in df.columns.values]
    print(f"NaN values in flattened DataFrame: {flattened_df.isna().sum().sum()}")
    return flattened_df

# Function to calculate the Supertrend indicator
def get_supertrend(high, low, close, period, multiplier):
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    basic_upper = (high + low) / 2 + (multiplier * atr)
    basic_lower = (high + low) / 2 - (multiplier * atr)
    final_upper = pd.Series(0.0, index=close.index)
    final_lower = pd.Series(0.0, index=close.index)
    supertrend = pd.Series(0.0, index=close.index)
    for i in range(period, len(close)):
        if basic_upper.iloc[i] < final_upper.iloc[i-1] or close.iloc[i-1] > final_upper.iloc[i-1]:
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i-1]
        if basic_lower.iloc[i] > final_lower.iloc[i-1] or close.iloc[i-1] < final_lower.iloc[i-1]:
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i-1]
    for i in range(period, len(close)):
        if supertrend.iloc[i-1] == final_upper.iloc[i-1] and close.iloc[i] <= final_upper.iloc[i]:
            supertrend.iloc[i] = final_upper.iloc[i]
        elif supertrend.iloc[i-1] == final_upper.iloc[i-1] and close.iloc[i] > final_upper.iloc[i]:
            supertrend.iloc[i] = final_lower.iloc[i]
        elif supertrend.iloc[i-1] == final_lower.iloc[i-1] and close.iloc[i] >= final_lower.iloc[i]:
            supertrend.iloc[i] = final_lower.iloc[i]
        elif supertrend.iloc[i-1] == final_lower.iloc[i-1] and close.iloc[i] < final_lower.iloc[i]:
            supertrend.iloc[i] = final_upper.iloc[i]
        else:
            supertrend.iloc[i] = 0.0
    upt = []
    dt = []
    Close = close.iloc[period:]
    for i in range(len(Close)):
        if Close.iloc[i] > supertrend.iloc[period+i]:
            upt.append(supertrend.iloc[period+i])
            dt.append(np.nan)
        elif Close.iloc[i] < supertrend.iloc[period+i]:
            upt.append(np.nan)
            dt.append(supertrend.iloc[period+i])
        else:
            upt.append(np.nan)
            dt.append(np.nan)
    st = pd.Series(supertrend.iloc[period:].values, index=Close.index)
    upt = pd.Series(upt, index=Close.index)
    dt = pd.Series(dt, index=Close.index)
    return st, upt, dt

# Trading strategy implementation
def implement_st_strategy(prices, st):
    buy_price = [np.nan]  # First element has no previous data
    sell_price = [np.nan]
    st_signal = [0]
    signal = 0
    for i in range(1, len(st)):  # Start at 1 to avoid index -1
        if st.iloc[i-1] > prices.iloc[i-1] and st.iloc[i] < prices.iloc[i]:
            if signal != 1:
                buy_price.append(prices.iloc[i])
                sell_price.append(np.nan)
                signal = 1
                st_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                st_signal.append(0)
        elif st.iloc[i-1] < prices.iloc[i-1] and st.iloc[i] > prices.iloc[i]:
            if signal != -1:
                buy_price.append(np.nan)
                sell_price.append(prices.iloc[i])
                signal = -1
                st_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                st_signal.append(0)
        else:
            buy_price.append(np.nan)
            sell_price.append(np.nan)
            st_signal.append(0)
    return buy_price, sell_price, st_signal

# Define the TradingSystem class
class TradingSystem:
    def __init__(self, initial_capital=10000, position_size=2.0, stop_loss_pct=0.92, transaction_cost=0.001):
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.stop_loss_pct = stop_loss_pct
        self.transaction_cost = transaction_cost

    def generate_trading_lists(self, df, symbol):
        long_trades = []
        short_trades = []
        current_position = None
        entry_price = None
        entry_index = None
        entry_date = None
        for i, row in df.iterrows():
            trend_up = row['TrendUp']
            trend_down = row.get('TrendDown', 0)
            if trend_up and current_position is None:
                current_position = 'Long'
                entry_price = row[f'Close_{symbol}']
                entry_index = i
                entry_date = row.name
            elif trend_down and current_position is None:
                current_position = 'Short'
                entry_price = row[f'Close_{symbol}']
                entry_index = i
                entry_date = row.name
            elif trend_down and current_position == 'Long':
                exit_price = row[f'Close_{symbol}']
                exit_index = i
                exit_date = row.name
                profit_loss = (exit_price - entry_price) / entry_price
                long_trades.append({
                    'entry_date': entry_date,
                    'entry_price': entry_price,
                    'exit_date': exit_date,
                    'exit_price': exit_price,
                    'profit_loss': profit_loss,
                    'entry_index': entry_index,
                    'exit_index': exit_index,
                    'symbol': symbol
                })
                current_position = None
            elif trend_up and current_position == 'Short':
                exit_price = row[f'Close_{symbol}']
                exit_index = i
                exit_date = row.name
                profit_loss = (entry_price - exit_price) / entry_price
                short_trades.append({
                    'entry_date': entry_date,
                    'entry_price': entry_price,
                    'exit_date': exit_date,
                    'exit_price': exit_price,
                    'profit_loss': profit_loss,
                    'entry_index': entry_index,
                    'exit_index': exit_index,
                    'symbol': symbol
                })
                current_position = None
        return long_trades, short_trades
    
    def calculate_equity_curve(self, df, trades):
        if not trades:
            return pd.Series(self.initial_capital, index=df.index)
        equity_curve = pd.Series(index=df.index, dtype=float)
        equity_curve.iloc[0] = self.initial_capital
        current_capital = self.initial_capital
        current_position = None
        for date in df.index:
            if current_position:
                daily_return = df.loc[date, f'Close_{current_position["symbol"]}'] / df.loc[current_position["entry_date"], f'Close_{current_position["symbol"]}'] - 1
                current_capital = current_position["entry_capital"] * (1 + daily_return)
                equity_curve.loc[date] = current_capital
            else:
                equity_curve.loc[date] = current_capital
            for trade in trades:
                if pd.Timestamp(date) == pd.Timestamp(trade['entry_date']):
                    current_position = {
                        "symbol": trade["symbol"],
                        "entry_date": trade["entry_date"],
                        "entry_capital": current_capital,
                    }
                elif pd.Timestamp(date) == pd.Timestamp(trade['exit_date']) and current_position:
                    current_position = None
        return equity_curve.ffill().bfill()

    def _add_equity_curve(self, fig, equity_curve, name, color, row, col):
        fig.add_trace(
            go.Scatter(
                x=equity_curve.index,
                y=equity_curve.values,
                name=name,
                line=dict(color=color)
            ),
            row=row, col=col
        )
    
    def calculate_trade_statistics(self, trades, equity_curve):
        if not trades:
            return {
                "Total Trades": 0,
                "Winning Trades": 0,
                "Losing Trades": 0,
                "Win Rate": 0.0,
                "Average Profit": 0.0,
                "Average Loss": 0.0,
                "Profit Factor": 0.0,
                "Total Return": 0.0,
                "Max Drawdown": 0.0,
                "Sharpe Ratio": 0.0
            }
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t['profit_loss'] > 0])
        losing_trades = len([t for t in trades if t['profit_loss'] <= 0])
        profits = [t['profit_loss'] for t in trades if t['profit_loss'] > 0]
        losses = [t['profit_loss'] for t in trades if t['profit_loss'] <= 0]
        avg_profit = np.mean(profits) if profits else 0
        avg_loss = np.mean(losses) if losses else 0
        total_profit = sum(profits)
        total_loss = sum(losses)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        profit_factor = abs(total_profit / total_loss) if total_loss != 0 else float('inf')
        total_return = (equity_curve.iloc[-1] - self.initial_capital) / self.initial_capital
        max_drawdown = abs((equity_curve - equity_curve.expanding().max()).min())
        pct_changes = equity_curve.pct_change().dropna()
        if len(pct_changes) > 1 and pct_changes.std() != 0:
            sharpe_ratio = np.sqrt(252) * (pct_changes.mean() / pct_changes.std())
        else:
            sharpe_ratio = 0.0
        return {
            "Total Trades": total_trades,
            "Winning Trades": winning_trades,
            "Losing Trades": losing_trades,
            "Profits": profits,
            "Losses": losses,
            "Avg Profit": avg_profit,
            "Avg Loss": avg_loss,
            "Total Profit": total_profit,
            "Total Loss": total_loss,
            "Win Rate": win_rate,
            "Profit Factor": profit_factor,
            "Total Return": total_return,
            "Max Drawdown": max_drawdown,
            "Sharpe Ratio": sharpe_ratio,
        }

    def plot_results(self, df, long_trades, short_trades, long_equity, short_equity, buy_and_hold_equity, symbol):
        plot_df = df.iloc[15:]
        plot_long_equity = long_equity.iloc[15:]
        plot_short_equity = short_equity.iloc[15:]
        plot_buy_and_hold_equity = buy_and_hold_equity.iloc[15:]
        combined_equity = plot_long_equity + plot_short_equity - self.initial_capital
        combined_equity = combined_equity.ffill().bfill()
    
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(f'{symbol} Price and Supertrend Signals', f'{symbol} Equity Curves'),
            row_heights=[0.6, 0.4]
        )
        fig.add_trace(
            go.Candlestick(
                x=plot_df.index,
                open=plot_df[f'Open_{symbol}'],
                high=plot_df[f'High_{symbol}'],
                low=plot_df[f'Low_{symbol}'],
                close=plot_df[f'Close_{symbol}'],
                name='Candlestick'
            ),
            row=1, col=1
        )
        
        # Add Supertrend to the plot with changing colors
        for start, end in zip(df.index[:-1], df.index[1:]):
            color = 'green' if df[f'Close_{symbol}'][start] > df['Supertrend'][start] else 'red'
            fig.add_trace(
                go.Scatter(
                    x=[start, end],
                    y=[df['Supertrend'][start], df['Supertrend'][end]],
                    mode='lines',
                    line=dict(color=color, width=2),
                    showlegend=False
                ),
                row=1, col=1
            )
    
        # Add buy and sell signals to the plot
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Buy_Signal_Price'],
                mode='markers',
                name='Buy Signal',
                marker=dict(color='green', symbol='triangle-up', size=10)
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Sell_Signal_Price'],
                mode='markers',
                name='Sell Signal',
                marker=dict(color='red', symbol='triangle-down', size=10)
            ),
            row=1, col=1
        )
    
        # Add long trade markers
        if long_trades:
            long_entries = [trade['entry_date'] for trade in long_trades if trade['entry_date'] >= plot_df.index[0]]
            long_entry_prices = [trade['entry_price'] for trade in long_trades if trade['entry_date'] >= plot_df.index[0]]
            long_exits = [trade['exit_date'] for trade in long_trades if trade['exit_date'] >= plot_df.index[0]]
            long_exit_prices = [trade['exit_price'] for trade in long_trades if trade['exit_date'] >= plot_df.index[0]]
            fig.add_trace(
                go.Scatter(
                    x=long_entries,
                    y=[price + 0.5 for price in long_entry_prices],
                    mode='markers',
                    name='Long Entry',
                    marker=dict(symbol='triangle-up', size=10, color='green')
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=long_exits,
                    y=[price + 0.5 for price in long_exit_prices],
                    mode='markers',
                    name='Long Exit',
                    marker=dict(symbol='triangle-down', size=10, color='red')
                ),
                row=1, col=1
            )
    
        # Add short trade markers
        if short_trades:
            short_entries = [trade['entry_date'] for trade in short_trades if trade['entry_date'] >= plot_df.index[0]]
            short_entry_prices = [trade['entry_price'] for trade in short_trades if trade['entry_date'] >= plot_df.index[0]]
            short_exits = [trade['exit_date'] for trade in short_trades if trade['exit_date'] >= plot_df.index[0]]
            short_exit_prices = [trade['exit_price'] for trade in short_trades if trade['exit_date'] >= plot_df.index[0]]
            fig.add_trace(
                go.Scatter(
                    x=short_entries,
                    y=[price - 0.5 for price in short_entry_prices],
                    mode='markers',
                    name='Short Entry',
                    marker=dict(symbol='triangle-down', size=10, color='blue')
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=short_exits,
                    y=[price - 0.5 for price in short_exit_prices],
                    mode='markers',
                    name='Short Exit',
                    marker=dict(symbol='triangle-up', size=10, color='black')
                ),
                row=1, col=1
            )
    
        # Add equity curves to the plot
        fig.add_trace(
            go.Scatter(
                x=plot_long_equity.index,
                y=plot_long_equity.values,
                mode='lines',
                name='Long Equity',
                line=dict(color='green', width=2)
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=plot_short_equity.index,
                y=plot_short_equity.values,
                mode='lines',
                name='Short Equity',
                line=dict(color='red', width=2)
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=combined_equity.index,
                y=combined_equity.values,
                mode='lines',
                name='Combined Equity',
                line=dict(color='blue', width=2)
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=plot_buy_and_hold_equity.index,
                y=plot_buy_and_hold_equity.values,
                mode='lines',
                name='Buy and Hold Equity',
                line=dict(color='orange', width=2, dash='dash')
            ),
            row=2, col=1
        )
    
        fig.update_layout(
            title=f'Trading System Results for {symbol}',
            xaxis=dict(rangeslider=dict(visible=False)),
            yaxis_title='Price',
            yaxis2_title='Equity',
            height=1000,
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        price_min = plot_df[f'Low_{symbol}'].min()
        price_max = plot_df[f'High_{symbol}'].max()
        equity_min = min(plot_long_equity.min(), plot_short_equity.min(), combined_equity.min(), plot_buy_and_hold_equity.min())
        equity_max = max(plot_long_equity.max(), plot_short_equity.max(), combined_equity.max(), plot_buy_and_hold_equity.max())
        fig.update_yaxes(range=[price_min * 0.95, price_max * 1.05], row=1, col=1)
        fig.update_yaxes(range=[equity_min * 1.1 if equity_min < 0 else equity_min * 0.9, equity_max * 1.1], row=2, col=1)  # Adjusted to cover negative values
    
        return fig


    def print_statistics(self, stats, trade_type=""):
        print(f"\n{trade_type} Trading Statistics:")
        print("=" * 50)
        excluded_keys = ["Daily Returns", "Rolling Max", "Drawdown"]
        for key, value in stats.items():
            if key in excluded_keys:
                continue
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            elif isinstance(value, list):
                formatted_list = ', '.join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in value)
                print(f"{key}: {formatted_list}")
            elif isinstance(value, pd.Series):
                formatted_series = value.to_string()
                print(f"{key}:\n{formatted_series}")
            else:
                print(f"{key}: {value}")
    
# Configuration
CONFIG = {
    'symbols': [
        "HON",      # Honeywell
        "AAPL",     # Apple
        "MSFT",     # Microsoft
        "GOOGL",    # Alphabet
        "AMZN",     # Amazon
        "NVDA",     # NVIDIA
        "META",     # Meta
        "TSLA",     # Tesla
        "JPM",      # JPMorgan
        "V",        # Visa
    ],
    'supertrend_period': 7,
    'supertrend_multiplier': 3,
    'short_ma_period': 20,
    'long_ma_period': 50,
    'lookback_days': 365,
    'initial_capital': 10000,
}


def analyze_symbol(stock_symbol, system, config):
    """Analyze a single stock symbol and return results."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {stock_symbol}")
    print('='*60)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=config['lookback_days'])

    try:
        stock_data = yf.download(stock_symbol, start=start_date, end=end_date, progress=False)
        if stock_data.empty:
            print(f"No data available for {stock_symbol}")
            return None
    except Exception as e:
        print(f"Error downloading {stock_symbol}: {e}")
        return None

    # Flatten the DataFrame to handle MultiIndex columns
    stock_data = flatten_dataframe(stock_data)

    close_col = f'Close_{stock_symbol}'
    high_col = f'High_{stock_symbol}'
    low_col = f'Low_{stock_symbol}'

    # Check if required columns exist
    if close_col not in stock_data.columns:
        print(f"Required column {close_col} not found for {stock_symbol}")
        return None

    # Calculate Moving Averages for TrendUp/TrendDown
    stock_data['ShortMA'] = stock_data[close_col].rolling(window=config['short_ma_period']).mean()
    stock_data['LongMA'] = stock_data[close_col].rolling(window=config['long_ma_period']).mean()
    stock_data['TrendUp'] = (stock_data['ShortMA'] > stock_data['LongMA']).fillna(False)
    stock_data['TrendDown'] = ~stock_data['TrendUp']

    # Calculate Supertrend
    st, s_upt, st_dt = get_supertrend(
        stock_data[high_col],
        stock_data[low_col],
        stock_data[close_col],
        config['supertrend_period'],
        config['supertrend_multiplier']
    )
    stock_data['Supertrend'] = st
    stock_data['SupertrendUp'] = s_upt
    stock_data['SupertrendDown'] = st_dt

    # Implement the Supertrend trading strategy
    buy_price, sell_price, st_signal = implement_st_strategy(stock_data[close_col], stock_data['Supertrend'])
    stock_data['Buy_Signal_Price'] = buy_price
    stock_data['Sell_Signal_Price'] = sell_price
    stock_data['ST_Signal'] = st_signal

    long_trades, short_trades = system.generate_trading_lists(stock_data, stock_symbol)

    # Calculate equity curves based on Supertrend trades
    long_equity = system.calculate_equity_curve(stock_data, long_trades)
    short_equity = system.calculate_equity_curve(stock_data, short_trades)

    # Calculate buy and hold equity
    buy_and_hold_equity = (stock_data[close_col] / stock_data[close_col].iloc[0]) * system.initial_capital

    # Print trade summary
    print(f"\nLong Trades: {len(long_trades)}")
    print(f"Short Trades: {len(short_trades)}")

    # Calculate statistics
    long_stats = system.calculate_trade_statistics(long_trades, long_equity)
    short_stats = system.calculate_trade_statistics(short_trades, short_equity)
    system.print_statistics(long_stats, f"{stock_symbol} Long")
    system.print_statistics(short_stats, f"{stock_symbol} Short")

    return {
        'symbol': stock_symbol,
        'data': stock_data,
        'long_trades': long_trades,
        'short_trades': short_trades,
        'long_equity': long_equity,
        'short_equity': short_equity,
        'buy_and_hold_equity': buy_and_hold_equity,
        'long_stats': long_stats,
        'short_stats': short_stats
    }


def main():
    system = TradingSystem(initial_capital=CONFIG['initial_capital'])

    # Analyze all configured symbols
    results = []
    for symbol in CONFIG['symbols']:
        result = analyze_symbol(symbol, system, CONFIG)
        if result:
            results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    print(f"Total symbols analyzed: {len(results)}/{len(CONFIG['symbols'])}")

    # Print best performers
    if results:
        print("\nBest Long Performers (by Total Return):")
        sorted_long = sorted(results, key=lambda x: x['long_stats']['Total Return'], reverse=True)
        for i, r in enumerate(sorted_long[:3], 1):
            print(f"  {i}. {r['symbol']}: {r['long_stats']['Total Return']*100:.2f}%")

        print("\nBest Short Performers (by Total Return):")
        sorted_short = sorted(results, key=lambda x: x['short_stats']['Total Return'], reverse=True)
        for i, r in enumerate(sorted_short[:3], 1):
            print(f"  {i}. {r['symbol']}: {r['short_stats']['Total Return']*100:.2f}%")

    # Plot first symbol as example (or all if desired)
    if results:
        print(f"\nShowing chart for: {results[0]['symbol']}")
        r = results[0]
        fig = system.plot_results(
            r['data'],
            r['long_trades'],
            r['short_trades'],
            r['long_equity'],
            r['short_equity'],
            r['buy_and_hold_equity'],
            r['symbol']
        )
        fig.show()


if __name__ == "__main__":
    main()

