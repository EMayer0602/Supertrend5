# Supertrend5 - Multi-Strategy Trading System

A comprehensive trading system with strategy optimization, backtesting, and portfolio simulation using Interactive Brokers TWS.

## Features

- **Multi-Strategy Support**: SUPERTREND, JMA, KAMA, EMA, SMA
- **HTF Filter**: Higher Time Frame filter for trend confirmation
- **Walk-Forward Analysis**: Out-of-sample validation
- **Portfolio Simulation**: 30 positions (10 B&H + 20 Strategy)
- **Long & Short Trading**: Full directional trading support
- **HTML Reports**: Interactive charts and trade lists

## Requirements

- Python 3.8+
- Interactive Brokers TWS/Gateway running
- Required packages: `ib_insync`, `pandas`, `numpy`, `plotly`

```bash
pip install ib_insync pandas numpy plotly
```

## Quick Start

### 1. Strategy Optimization (Walk-Forward)

Optimize strategies on 9 months of historical data, reserving last 3 months for testing:

```bash
python new5.py --walk-forward
```

### 2. Portfolio Simulation

Run portfolio simulation on the test period:

```bash
python portfolio_simulation.py 90
```

## CLI Options (new5.py)

| Option | Description |
|--------|-------------|
| `--walk-forward` | Walk-forward analysis (9M optimize, 3M test) |
| `--htf-full` | Full HTF analysis with portfolio simulation |
| `--htf-compare` | HTF comparison and categorization only |
| `--all` | Optimize all tickers (365 days) |
| `--multi` | Multi-ticker analysis |
| `--screen` | Screen for Supertrend stocks |

## Portfolio Simulation

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Initial Capital | $20,000 | Starting capital |
| Max Positions | 30 | Maximum concurrent positions |
| B&H Positions | 10 | Buy & Hold positions (best performers) |
| Strategy Positions | 20 | Active trading positions |
| Fee Rate | 0.1% | Transaction fee |
| Trailing Stop | 20% | For B&H positions |

### Position Sizing

```
Stake per Position = Capital / 30
Quantity = round(Stake / Entry Price)
Entry Fee = Quantity × Entry Price × 0.001
Daily PnL = Quantity × (Close - Previous Close)
```

### Trading Logic

**Buy & Hold (10 positions)**:
- Top 10 performers from last 30 days
- Rebalanced every 5 days
- 20% trailing stop loss

**Strategy Positions (20 positions)**:
- Long/Short based on strategy signals
- Signal flow:
  - Bullish crossover → COVER (if short) → BUY
  - Bearish crossover → SELL (if long) → SHORT

## Strategies

### SUPERTREND
- Parameters: `period` (10, 14, 20), `multiplier` (2.0, 3.0, 4.0)
- Signal: Direction change from -1 to 1 (buy) or 1 to -1 (sell)

### JMA (Jurik Moving Average)
- Parameters: `fast` (7, 10, 14), `slow` (21, 30, 50)
- Signal: Fast/Slow crossover

### KAMA (Kaufman Adaptive MA)
- Parameters: `period` (10, 14, 20), `signal` (10, 14, 21)
- Signal: KAMA/Signal line crossover

### EMA (Exponential MA)
- Parameters: `fast` (8, 12, 20), `slow` (21, 26, 50)
- Signal: Fast/Slow crossover

### SMA (Simple MA)
- Parameters: `fast` (10, 20, 30), `slow` (50, 100, 200)
- Signal: Fast/Slow crossover

## HTF Filter

Higher Time Frame filter uses weekly Supertrend to filter daily signals:
- `_HTF` suffix: Strategy with HTF filter enabled
- `_NOHTF` suffix: Strategy without HTF filter

Only buy signals when weekly trend is bullish.

## Output Files

| File | Description |
|------|-------------|
| `htf_categorized_results.json` | Strategy assignments per symbol |
| `multi_portfolio_results.json` | Portfolio performance by category |
| `ticker_assignments.json` | All strategy results |
| `portfolio_report.html` | Interactive HTML report |

## HTML Report Contents

- **Header**: Net Liquidity, Daily PnL, Unrealized, Realized, Total Return
- **Charts**: Equity curve, Daily PnL bars, Position PnL
- **Trade Lists**:
  - Closed Trades - LONG
  - Closed Trades - SHORT
  - Open Positions - LONG
  - Open Positions - SHORT
- **Metrics**: Win Rate, Profit Factor, Sharpe Ratio, Max Drawdown

## Walk-Forward Analysis

```
|<-------- 12 Months Data -------->|
|                                  |
|<-- 9M Optimization -->|<- 3M Test ->|
|   (Training Data)     | (Out-of-Sample)|
```

1. **Optimization Phase**: Find best strategy/parameters on historical data
2. **Test Phase**: Validate on unseen data (last 3 months)

## Symbol Universe

- **DOW 30**: 30 blue-chip stocks
- **NASDAQ 100**: 100 tech/growth stocks
- **Combined**: 124 unique symbols

## Usage Scenarios

### Scenario 1: Walk-Forward Analysis (Recommended)

Best practice for validating strategies on out-of-sample data:

```bash
# 1. Start TWS/Gateway

# 2. Optimize on 9 months, test on last 3 months
python new5.py --walk-forward

# 3. Run portfolio simulation on test period (90 days)
python portfolio_simulation.py 90

# 4. View results
open portfolio_report.html
```

### Scenario 2: Quick Strategy Scan

Find best strategies for all symbols:

```bash
# Full HTF analysis (6 months)
python new5.py --htf-full
```

### Scenario 3: Single Symbol Analysis

Test a single symbol with all strategies:

```bash
python new5.py AAPL
```

### Scenario 4: Custom Simulation Period

```bash
# 6-month simulation
python portfolio_simulation.py 180

# 1-year simulation
python portfolio_simulation.py 365
```

### Scenario 5: Strategy Screening

Find stocks currently in Supertrend uptrend:

```bash
python new5.py --screen
```

### Scenario 6: Full Year Optimization

Optimize all strategies over 1 year:

```bash
python new5.py --all
```

## Complete Workflow Example

```bash
# Step 1: Ensure TWS is running on port 7497

# Step 2: Walk-forward optimization
# - Trains on data from 12-3 months ago
# - Saves best strategies to htf_categorized_results.json
python new5.py --walk-forward

# Step 3: Portfolio simulation on unseen data
# - Uses strategies from Step 2
# - Simulates last 90 days with $20,000
# - Generates portfolio_report.html
python portfolio_simulation.py 90

# Step 4: Review results
# - Check portfolio_report.html for:
#   - Equity curve
#   - Trade statistics
#   - Open/Closed positions (Long & Short)
#   - Win rate, Sharpe ratio, Max drawdown
```

## Interpreting Results

### Portfolio Report (HTML)

**Header Metrics:**
- **Net Liquidity**: Current portfolio value
- **Daily PnL**: Today's profit/loss
- **Unrealized**: Open position P&L
- **Realized PnL**: Closed trades P&L
- **Total Return**: Overall performance

**Performance Metrics:**
- **Win Rate**: % of profitable trades
- **Profit Factor**: Gross profit / Gross loss (>1 = profitable)
- **Sharpe Ratio**: Risk-adjusted return (>1 = good)
- **Max Drawdown**: Largest peak-to-trough decline

### JSON Output Files

**htf_categorized_results.json:**
```json
{
  "categories": {
    "SUPERTREND_HTF": [
      {"symbol": "AAPL", "return": 0.45, "params": {"period": 10, "multiplier": 3.0}}
    ],
    "JMA_NOHTF": [...],
    "UNDERPERFORM": [...]
  }
}
```

**portfolio_report.html:**
- Interactive Plotly charts
- Sortable trade tables
- Color-coded P&L (green/red)

## Categories (PnL >= 30%)

| Category | Description |
|----------|-------------|
| `SUPERTREND_HTF` | Supertrend with HTF filter |
| `SUPERTREND_NOHTF` | Supertrend without HTF |
| `JMA_HTF` / `JMA_NOHTF` | JMA crossover |
| `KAMA_HTF` / `KAMA_NOHTF` | KAMA crossover |
| `EMA_HTF` / `EMA_NOHTF` | EMA crossover |
| `SMA_HTF` / `SMA_NOHTF` | SMA crossover |
| `BUYHOLD` | Buy & Hold is best |
| `UNDERPERFORM` | Best PnL < 30% |

## License

Private use only.

## Author

Trading System v5.0
