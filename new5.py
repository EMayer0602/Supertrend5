	#Import necessary libraries
	import pandas as pd
	import yfinance as yf
	from datetime import datetime, timedelta
	import numpy as np
	import plotly.graph_objects as go
	from plotly.subplots import make_subplots
	
	# Assuming this is a custom module you've imported elsewhere
	# import system
	
	# Get date range for last 365 days
	end_date = datetime.now()
	start_date = end_date - timedelta(days=365)
	
	# Function to calculate technical indicators and trend signals
	def calculate_indicators_and_trends(df, period=14):
		df = df.copy()
		
		# Calculate technical indicators
		high_low = df['High'] - df['Low']
		high_close = abs(df['High'] - df['Close'].shift(1))
		low_close = abs(df['Low'] - df['Close'].shift(1))
		
		df['TR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
		df['UpMove'] = df['High'] - df['High'].shift(1)
		df['DownMove'] = df['Low'].shift(1) - df['Low']
		df['+DM'] = np.where((df['UpMove'] > df['DownMove']) & (df['UpMove'] > 0), df['UpMove'], 0)
		df['-DM'] = np.where((df['DownMove'] > df['UpMove']) & (df['DownMove'] > 0), df['DownMove'], 0)
		
		df['TR14'] = df['TR'].rolling(window=period).sum()
		df['+DM14'] = df['+DM'].rolling(window=period).sum()
		df['-DM14'] = df['-DM'].rolling(window=period).sum()
		df['+DI'] = 100 * (df['+DM14'] / df['TR14'])
		df['-DI'] = 100 * (df['-DM14'] / df['TR14'])
		df['DX'] = 100 * abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])
		df['ADX'] = df['DX'].rolling(window=period).mean()
		
		# Initialize trend signals
		df['TrendUp'] = 0
		df['TrendDown'] = 0
		
		for i in range(1, len(df)):
			prev_plus_di = df['+DI'].iloc[i-1]
			prev_minus_di = df['-DI'].iloc[i-1]
			curr_plus_di = df['+DI'].iloc[i]
			curr_minus_di = df['-DI'].iloc[i]
			
			# Trend logic
			if prev_plus_di <= prev_minus_di and curr_plus_di > curr_minus_di:
				df.at[df.index[i], 'TrendUp'] = 1
				df.at[df.index[i], 'TrendDown'] = 0
			elif prev_plus_di >= prev_minus_di and curr_plus_di < curr_minus_di:
				df.at[df.index[i], 'TrendDown'] = 1
				df.at[df.index[i], 'TrendUp'] = 0
	
		return df	
class TradingSystem:
    def __init__(self, stop_loss_pct=0.02, transaction_cost=0.001):
	# Your other methods follow here with proper indentation
        self.stop_loss_pct = stop_loss_pct
        self.transaction_cost = transaction_cost

	def generate_trading_lists(self, df):
		long_trades = []
		short_trades = []
		current_position = None
		entry_price = None
		entry_index = None
		entry_date = None

		for i in range(len(df)):
			row = df.iloc[i]
			
			# Close existing positions if trend changes
			if current_position == 'Long' and row['TrendDown'].item() == 1:
				long_trades.append({
					"Type": "Long",
					"Entry Date": entry_date,
					"Exit Date": row.name,
					"Entry Price": entry_price,
					"Exit Price": row['Close'],
					"PnL": (row['Close'] - entry_price) * self.position_size - 
						   (entry_price + row['Close']) * self.transaction_cost,
					"Duration": i - entry_index
				})
				current_position = None
						
			elif current_position == 'Short' and row['TrendUp'].item() == 1:
				short_trades.append({
					 "Type": "Short",
					 "Entry Date": entry_date,
					 "Exit Date": row.name,
					 "Entry Price": entry_price,
					 "Exit Price": row['Close'],
					 "PnL": (entry_price - row['Close']) * self.position_size - 
							(entry_price + row['Close']) * self.transaction_cost,
					"Duration": i - entry_index
				})
				current_position = None
				
			# Open new positions
			if row['TrendUp'].any() == 1 and current_position is None:
				current_position = 'Long'
				entry_price = row['Close']
				entry_index = i
				entry_date = row.name
				
			elif row['TrendDown'].any() == 1 and current_position is None:
				current_position = 'Short'
				entry_price = row['Close']
				entry_index = i
				entry_date = row.name
		
		# Close any open position at the end of the period
		if current_position == 'Long':
			long_trades.append({
				"Type": "Long",
				"Entry Date": entry_date,
				"Exit Date": df.index[-1],
				"Entry Price": entry_price,
				"Exit Price": df['Close'].iloc[-1],
				"PnL": (df['Close'].iloc[-1] - entry_price) * self.position_size - 
					   (entry_price + df['Close'].iloc[-1]) * self.transaction_cost,
				"Duration": len(df) - entry_index
			})
		elif current_position == 'Short':
			short_trades.append({
				"Type": "Short",
				"Entry Date": entry_date,
				"Exit Date": df.index[-1],
				"Entry Price": entry_price,
				"Exit Price": df['Close'].iloc[-1],
				"PnL": (entry_price - df['Close'].iloc[-1]) * self.position_size - 
					   (entry_price + df['Close'].iloc[-1]) * self.transaction_cost,
				"Duration": len(df) - entry_index
			})
		return long_trades, short_trades

	def calculate_equity_curve(self, df, trades):
		if not trades:  # If no trades, return initial capital as flat equity curve
			return pd.Series(self.initial_capital, index=df.index)
	
		# Initialize equity curve with initial capital
		equity_curve = pd.Series(index=df.index, dtype=float)
		equity_curve.iloc[0] = self.initial_capital
	
		# Sort trades by entry date
		sorted_trades = sorted(trades, key=lambda x: x['Entry Date'])
	
		# Track current capital
		current_capital = self.initial_capital
	
		# For each day in the dataset
		for date in df.index:
			daily_pnl = 0.0  # Reset daily PnL to float
			for trade in sorted_trades:
				entry_date = trade['Entry Date']
				exit_date = trade['Exit Date']
				if date == exit_date:  # Match exit date with current date
					daily_pnl += float(trade['PnL'])  # Ensure PnL is scalar
	
			# Update current capital
			current_capital += daily_pnl
			equity_curve[date] = current_capital
	
		# Forward fill any NaN values
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
	
		# Basic trade statistics
		total_trades = len(trades)
		winning_trades = len([t for t in trades if float(t['PnL']) > 0])
		losing_trades = len([t for t in trades if float(t['PnL']) <= 0])
		
		profits = [float(t['PnL']) for t in trades if float(t['PnL']) > 0]
		losses = [float(t['PnL']) for t in trades if float(t['PnL']) <= 0]
	
		avg_profit = np.mean(profits) if profits else 0
		avg_loss = np.mean(losses) if losses else 0
		total_profit = sum(profits)
		total_loss = sum(losses) 
		
		win_rate = winning_trades / total_trades if total_trades > 0 else 0
		profit_factor = abs(sum(profits) / sum(losses)) if losses and sum(losses) != 0 else float('inf')
		
		# Calculate returns and drawdown
		total_return = (equity_curve.iloc[-1] - self.initial_capital) / self.initial_capital
		
		# Calculate drawdown
		rolling_max = equity_curve.expanding().max()
		drawdown = (equity_curve - rolling_max) / rolling_max
		max_drawdown = abs(drawdown.min())
		
		# Calculate Sharpe Ratio (assuming risk-free rate = 0)
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

	def plot_results(self, df, long_trades, short_trades, long_equity, short_equity):
		# Start plotting from index 15
		plot_df = df.iloc[15:]  # Adjusting the DataFrame
		plot_long_equity = long_equity.iloc[15:]
		plot_short_equity = short_equity.iloc[15:]
	
		# Debugging: Print the DataFrame to ensure data is available
		print("Plot DataFrame (after index 15):")
		print(plot_df[['Open', 'High', 'Low', 'Close']].dropna().head())
	
		# Create subplots with 2 rows
		fig = make_subplots(
			rows=2, cols=1,
			shared_xaxes=True,
			vertical_spacing=0.05,
			subplot_titles=('Price and Trades', 'Equity Curves'),
			row_heights=[0.6, 0.4]
		)
	
		# Add candlestick chart
		fig.add_trace(
			go.Candlestick(
				x=plot_df.index,
				open=plot_df['Open'],
				high=plot_df['High'],
				low=plot_df['Low'],
				close=plot_df['Close'],
				name='Candlestick'
			),
			row=1, col=1
		)
	
		# Add long trade entry and exit points with an offset
		offset = 0.5  # Adjust this value as needed
		if long_trades:
			long_entries = [trade['Entry Date'] for trade in long_trades if trade['Entry Date'] >= plot_df.index[0]]
			long_entry_prices = [trade['Entry Price'] for trade in long_trades if trade['Entry Date'] >= plot_df.index[0]]
			long_exits = [trade['Exit Date'] for trade in long_trades if trade['Exit Date'] >= plot_df.index[0]]
			long_exit_prices = [trade['Exit Price'] for trade in long_trades if trade['Exit Date'] >= plot_df.index[0]]
	
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
	
		# Add short trade entry and exit points with an offset
		if short_trades:
			short_entries = [trade['Entry Date'] for trade in short_trades if trade['Entry Date'] >= plot_df.index[0]]
			short_entry_prices = [trade['Entry Price'] for trade in short_trades if trade['Entry Date'] >= plot_df.index[0]]
			short_exits = [trade['Exit Date'] for trade in short_trades if trade['Exit Date'] >= plot_df.index[0]]
			short_exit_prices = [trade['Exit Price'] for trade in short_trades if trade['Exit Date'] >= plot_df.index[0]]
	
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
	
		# Add equity curves
		if len(plot_long_equity) > 0:
			self._add_equity_curve(fig, plot_long_equity, 'Long Equity', 'green', 2, 1)
		if len(plot_short_equity) > 0:
			self._add_equity_curve(fig, plot_short_equity, 'Short Equity', 'red', 2, 1)
	
		# Add combined equity curve
		combined_equity = plot_long_equity + plot_short_equity - self.initial_capital
		combined_equity = combined_equity.ffill().bfill()
		self._add_equity_curve(fig, combined_equity, 'Combined Equity', 'blue', 2, 1)
	
		# Update layout
		fig.update_layout(
			title='Trading System Results',
			xaxis_title='Date',
			yaxis_title='Price',
			yaxis2_title='Equity',
			height=1000,
			showlegend=True,
			legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
		)
	
		# Explicitly set y-axes ranges based on data
		price_min = plot_df['Low'].min()
		price_max = plot_df['High'].max()
		equity_min = min(plot_long_equity.min(), plot_short_equity.min(), combined_equity.min())
		equity_max = max(plot_long_equity.max(), plot_short_equity.max(), combined_equity.max())
		fig.update_yaxes(range=[price_min * 0.95, price_max * 1.05], row=1, col=1)
		fig.update_yaxes(range=[equity_min * 1.05, equity_max * 1.05], row=2, col=1)
	
		fig.show()
	
		return fig

	def _add_indicator(self, fig, df, signal_column, name, color, row, col):
		fig.add_trace(
			go.Scatter(
				x=df.index,
				y=df[signal_column],
				name=name,
				line=dict(color=color, dash='dot')
			),
			row=row, col=col
		)
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
	
symbol = "AAPL"  # Global variable

def main():  # Adding a function definition since this appears to be part of a function
    stock_symbol = "AAPL"  # Changed 'ol' to 'stock_symbol' to match usage below
    print(stock_symbol)
    # Download and process data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    btc_data = yf.download(stock_symbol, start=start_date, end=end_date)
    
    # Calculate indicators and generate trades
    btc_data = calculate_indicators_and_trends(btc_data)
    btc_data.ffill(inplace=True)
    
    # Debugging: Print the dataframe to ensure data is available
    print("BTC Data:")
    print(btc_data[['Open', 'High', 'Low', 'Close']].tail())
    
    long_trades, short_trades = system.generate_trading_lists(btc_data)
    
    # Debugging: Print trades to ensure they are generated
    print("Long Trades:", long_trades)
    print("Short Trades:", short_trades)
    
    # Calculate equity curves
    long_equity = system.calculate_equity_curve(btc_data, long_trades)
    short_equity = system.calculate_equity_curve(btc_data, short_trades)
    
    # Debugging: Print equity curves to ensure they are generated
    print('Long equity')
    print(long_equity.tail())
    print('Short equity')
    print(short_equity.tail())
     
    # Calculate statistics
    long_stats = system.calculate_trade_statistics(long_trades, long_equity)
    short_stats = system.calculate_trade_statistics(short_trades, short_equity)
    
    # Print statistics
    system.print_statistics(long_stats, "Long")
    system.print_statistics(short_stats, "Short")
    
    # Plot results
    print("Candlestick Data for Plot:")
    fig = system.plot_results(btc_data, long_trades, short_trades, long_equity, short_equity)
    fig.show()
    
    print("BTC Data Null Values:", btc_data.isnull().sum())
    print("Short Equity Null Values:", short_equity.isnull().sum())
    print("Candlestick Chart Data:")
    print(btc_data[['Open', 'High', 'Low', 'Close']].dropna().head())
    import plotly.io as pio
    pio.renderers.default = 'browser'  # or 'browser'

if __name__ == "__main__":
    main()
