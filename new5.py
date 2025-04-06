import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Function to flatten dataframe
def flatten_dataframe(df):
    flattened_df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        flattened_df.columns = ['_'.join(col).strip() for col in df.columns.values]
    print(f"NaN values in flattened DataFrame: {flattened_df.isna().sum().sum()}")
    return flattened_df

# Function to calculate the Supertrend indicator
def get_supertrend(high, low, close, period, multiplier):
    # Calculate ATR
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    
    # Calculate basic upper and lower bands
    basic_upper = (high + low) / 2 + (multiplier * atr)
    basic_lower = (high + low) / 2 - (multiplier * atr)
    
    # Initialize Supertrend columns
    final_upper = pd.Series(0.0, index=close.index)
    final_lower = pd.Series(0.0, index=close.index)
    supertrend = pd.Series(0.0, index=close.index)
    
    # Calculate final upper and lower bands
    for i in range(period, len(close)):
        if basic_upper.iloc[i] < final_upper.iloc[i-1] or close.iloc[i-1] > final_upper.iloc[i-1]:
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i-1]
            
        if basic_lower.iloc[i] > final_lower.iloc[i-1] or close.iloc[i-1] < final_lower.iloc[i-1]:
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i-1]
    
    # Calculate Supertrend
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
    
    # Calculate uptrend and downtrend indicators
    upt = []
    dt = []
    Close = close.iloc[period:]  # Start from period to match supertrend values
    
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
    
    # Convert to Series with proper indexing
    st = pd.Series(supertrend.iloc[period:].values, index=Close.index)
    upt = pd.Series(upt, index=Close.index)
    dt = pd.Series(dt, index=Close.index)
    
    return st, upt, dt

# Define the TradingSystem class
class TradingSystem:
    def __init__(self, initial_capital=10000, position_size=1.0, stop_loss_pct=0.02, transaction_cost=0.001):
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
                    'exit_index': exit_index
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
                    'exit_index': exit_index
                })
                current_position = None
        return long_trades, short_trades

    def calculate_equity_curve(self, df, trades):
        if not trades:
            return pd.Series(self.initial_capital, index=df.index)
        equity_curve = pd.Series(index=df.index, dtype=float)
        equity_curve.iloc[0] = self.initial_capital
        sorted_trades = sorted(trades, key=lambda x: x['entry_date'])
        current_capital = self.initial_capital
        for date in df.index:
            daily_pnl = 0.0
            for trade in sorted_trades:
                entry_date = trade['entry_date']
                exit_date = trade['exit_date']
                if date == exit_date:
                    daily_pnl += float(trade['profit_loss']) * self.initial_capital
            current_capital += daily_pnl
            equity_curve[date] = current_capital
        equity_curve = equity_curve.ffill()
        return equity_curve

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
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min())
        daily_returns = equity_curve.pct_change().dropna()
        sharpe_ratio = np.sqrt(252) * (daily_returns.mean() / daily_returns.std()) if len(daily_returns) > 1 else 0
        return {
            "Total Trades": total_trades,
            "Winning Trades": winning_trades,
            "Losing Trades": losing_trades,
            "Win Rate": win_rate,
            "Average Profit": avg_profit,
            "Average Loss": avg_loss,
            "Profit Factor": profit_factor,
            "Total Return": total_return,
            "Max Drawdown": max_drawdown,
            "Sharpe Ratio": sharpe_ratio
        }

    def plot_results(self, df, long_trades, short_trades, long_equity, short_equity, symbol):
        plot_df = df.iloc[15:]
        plot_long_equity = long_equity.iloc[15:]
        plot_short_equity = short_equity.iloc[15:]
        print("Plot DataFrame (after index 15):")
        print(plot_df[[f'Open_{symbol}', f'High_{symbol}', f'Low_{symbol}', f'Close_{symbol}']].dropna().head())
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(f'{symbol} Price and Trades', f'{symbol} Equity Curves'),
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
        offset = 0.5
        if long_trades:
            long_entries = [trade['entry_date'] for trade in long_trades if trade['entry_date'] >= plot_df.index[0]]
            long_entry_prices = [trade['entry_price'] for trade in long_trades if trade['entry_date'] >= plot_df.index[0]]
            long_exits = [trade['exit_date'] for trade in long_trades if trade['exit_date'] >= plot_df.index[0]]
            long_exit_prices = [trade['exit_price'] for trade in long_trades if trade['exit_date'] >= plot_df.index[0]]
            fig.add_trace(
                go.Scatter(
                    x=long_entries,
                    y=[price + offset for price in long_entry_prices],
                    mode='markers',
                    name='Long Entry',
                    marker=dict(symbol='triangle-up', size=10, color='green')
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=long_exits,
                    y=[price + offset for price in long_exit_prices],
                    mode='markers',
                    name='Long Exit',
                    marker=dict(symbol='triangle-down', size=10, color='red')
                ),
                row=1, col=1
            )
        if short_trades:
            short_entries = [trade['entry_date'] for trade in short_trades if trade['entry_date'] >= plot_df.index[0]]
            short_entry_prices = [trade['entry_price'] for trade in short_trades if trade['entry_date'] >= plot_df.index[0]]
            short_exits = [trade['exit_date'] for trade in short_trades if trade['exit_date'] >= plot_df.index[0]]
            short_exit_prices = [trade['exit_price'] for trade in short_trades if trade['exit_date'] >= plot_df.index[0]]
            fig.add_trace(
                go.Scatter(
                    x=short_entries,
                    y=[price - offset for price in short_entry_prices],
                    mode='markers',
                    name='Short Entry',
                    marker=dict(symbol='triangle-down', size=10, color='blue')
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=short_exits,
                    y=[price - offset for price in short_exit_prices],
                    mode='markers',
                    name='Short Exit',
                    marker=dict(symbol='triangle-up', size=10, color='black')
                ),
                row=1, col=1
            )
        if len(plot_long_equity) > 0:
            self._add_equity_curve(fig, plot_long_equity, 'Long Equity', 'green', 2, 1)
        if len(plot_short_equity) > 0:
            self._add_equity_curve(fig, plot_short_equity, 'Short Equity', 'red', 2, 1)
        combined_equity = plot_long_equity + plot_short_equity - self.initial_capital
        combined_equity = combined_equity.ffill().bfill()
        self._add_equity_curve(fig, combined_equity, 'Combined Equity', 'blue', 2, 1)
        fig.update_layout(
            title=f'Trading System Results for {symbol}',
            xaxis=dict(rangeslider=dict(visible=False)),  # Make the rangeslider for chart1 invisible
            xaxis2=dict(rangeslider=dict(visible=True, thickness=0.05)),  # Ensure the rangeslider for chart2 is visible and as small as possible
            yaxis_title='Price',
            yaxis2_title='Equity',
            height=1000,
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        price_min = plot_df[f'Low_{symbol}'].min()
        price_max = plot_df[f'High_{symbol}'].max()
        equity_min = min(plot_long_equity.min(), plot_short_equity.min(), combined_equity.min())
        equity_max = max(plot_long_equity.max(), plot_short_equity.max(), combined_equity.max())
        fig.update_yaxes(range=[price_min * 0.95, price_max * 1.05], row=1, col=1)
        fig.update_yaxes(range=[equity_min * 1.1 if equity_min < 0 else equity_min * 0.9, equity_max * 1.1], row=2, col=1)  # Adjusted to cover negative values
        
        # Add Supertrend to the plot
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Supertrend'],
                mode='lines',
                name='Supertrend',
                line=dict(color='orange', width=2)
            ),
            row=1, col=1
        )

        fig.show()
        return fig

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

    def print_statistics(self, stats, trade_type=""):
        print(f"\n{trade_type} Trading Statistics:")
        print("=" * 50)
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")

def main():
    stock_symbol = "MSFT"
    system = TradingSystem()
    print(stock_symbol)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    btc_data = yf.download(stock_symbol, start=start_date, end=end_date)
    print("Available columns in btc_data:")
    print(btc_data.columns)
    
    # Flatten the DataFrame to handle MultiIndex columns
    btc_data = flatten_dataframe(btc_data)
    
    if 'TrendUp' not in btc_data.columns:
        try:
            close_col = f'Close_{stock_symbol}'
            btc_data['ShortMA'] = btc_data[close_col].rolling(window=20).mean()
            btc_data['LongMA'] = btc_data[close_col].rolling(window=50).mean()
            btc_data['TrendUp'] = btc_data['ShortMA'] > btc_data['LongMA']
            btc_data['TrendUp'] = btc_data['TrendUp'].fillna(False)
            print("TrendUp column created successfully")
        except Exception as e:
            print(f"Error creating TrendUp column: {e}")
            print("Available columns:", btc_data.columns)
            
    if 'TrendDown' not in btc_data.columns:
        btc_data['TrendDown'] = ~btc_data['TrendUp']

    # Calculate Supertrend
    st, s_upt, st_dt = get_supertrend(btc_data['High_MSFT'], btc_data['Low_MSFT'], btc_data['Close_MSFT'], 7, 3)
    btc_data['Supertrend'] = st
    btc_data['SupertrendUp'] = s_upt
    btc_data['SupertrendDown'] = st_dt
    
    long_trades, short_trades = system.generate_trading_lists(btc_data, stock_symbol)
    
    # Improved format for trade lists
    long_trades_df = pd.DataFrame(long_trades)
    short_trades_df = pd.DataFrame(short_trades)
    
    print("Long Trades:")
    print(long_trades_df.to_string(index=False))
    print("\nShort Trades:")
    print(short_trades_df.to_string(index=False))
    
    long_equity = system.calculate_equity_curve(btc_data, long_trades)
    short_equity = system.calculate_equity_curve(btc_data, short_trades)
    print('Long equity')
    print(long_equity.tail())
    print('Short equity')
    print(short_equity.tail())
    long_stats = system.calculate_trade_statistics(long_trades, long_equity)
    short_stats = system.calculate_trade_statistics(short_trades, short_equity)
    system.print_statistics(long_stats, "Long")
    system.print_statistics(short_stats, "Short")
    print("Candlestick Data for Plot:")
    btc_data_flat = flatten_dataframe(btc_data)  # Ensure DataFrame is flattened for plotting
    fig = system.plot_results(btc_data_flat, long_trades, short_trades, long_equity, short_equity, stock_symbol)
    
    # Add Supertrend to the plot with changing colors
    for start, end in zip(btc_data.index[:-1], btc_data.index[1:]):
        color = 'green' if btc_data['Close_MSFT'][start] > btc_data['Supertrend'][start] else 'red'
        fig.add_trace(
            go.Scatter(
                x=[start, end],
                y=[btc_data['Supertrend'][start], btc_data['Supertrend'][end]],
                mode='lines',
                line=dict(color=color, width=2),
                showlegend=False
            ),
            row=1, col=1
        )
    
    fig.show()
    print("BTC Data Null Values:", btc_data.isnull().sum())
    print("Short Equity Null Values:", short_equity.isnull().sum())
    print("Candlestick Chart Data:")
    try:
        print(btc_data[[f'Open_{stock_symbol}', f'High_{stock_symbol}', f'Low_{stock_symbol}', f'Close_{stock_symbol}']].dropna().head())
    except KeyError:
        try:
            print(btc_data[['Open_BTC-EUR', 'High_BTC-EUR', 'Low_BTC-EUR', 'Close_BTC-EUR']].dropna().head())
        except KeyError:
            print("Could not access OHLC columns - see available columns above")

import plotly.io as pio
pio.renderers.default = 'browser'

if __name__ == "__main__":
    main()
	
