# Supertrend5 - Multi-Strategy Trading System

A comprehensive trading system with strategy optimization, backtesting, and portfolio simulation using Interactive Brokers TWS.

## Features

- **Multi-Strategy Support**: SUPERTREND, JMA, KAMA, EMA, SMA
- **HTF Filter**: Higher Time Frame filter for trend confirmation
- **Walk-Forward Analysis**: Out-of-sample validation
- **Separate LONG/SHORT Optimization**: Different parameters for each direction
- **Portfolio Simulation**: 30 positions (10 B&H + 20 Strategy)
- **Long & Short Trading**: Full directional trading support
- **Long-Only Mode**: Option to disable short trading
- **HTML Reports**: Interactive charts and trade lists

## Requirements

- Python 3.8+
- Interactive Brokers TWS/Gateway running
- Required packages: `ib_insync`, `pandas`, `numpy`, `plotly`

```bash
pip install ib_insync pandas numpy plotly
```

## Quick Start

### 1. Strategy Optimization (Separate LONG/SHORT)

Optimize strategies separately for LONG and SHORT with different parameters:

```bash
python new5.py --long-short
```

### 2. Portfolio Simulation

Run portfolio simulation on the test period:

```bash
# Long & Short trading
python portfolio_simulation.py 90

# Long only (no shorts)
python portfolio_simulation.py 90 --long-only
```

## CLI Options (new5.py)

| Option | Description |
|--------|-------------|
| `--long-short` | **Separate LONG/SHORT optimization** (recommended) |
| `--walk-forward` | Walk-forward analysis (9M optimize, 3M test) |
| `--htf-full` | Full HTF analysis with portfolio simulation |
| `--htf-compare` | HTF comparison and categorization only |
| `--all` | Optimize all tickers (365 days) |
| `--multi` | Multi-ticker analysis |
| `--screen` | Screen for Supertrend stocks |

## CLI Options (portfolio_simulation.py)

```bash
python portfolio_simulation.py [days] [--long-only]
```

| Option | Description |
|--------|-------------|
| `days` | Number of days to simulate (default: 180) |
| `--long-only` | Only trade LONG positions (no shorts) |

## Separate LONG/SHORT Optimization

Short strategies often need different parameters than long strategies because markets fall faster than they rise.

### Parameter Differences

| Parameter | LONG | SHORT |
|-----------|------|-------|
| **SUPERTREND Period** | 10, 14, 20 | 7, 10, 14 (shorter) |
| **SUPERTREND Multiplier** | 2.0, 3.0, 4.0 | 1.5, 2.0, 2.5, 3.0 (tighter) |
| **MA Fast** | 7-30 | 5-20 (faster) |
| **MA Slow** | 21-200 | 14-50 (shorter) |
| **Min PnL Threshold** | 15% | 15% |

### Categories (Separate)

**LONG Categories:**
- `SUPERTREND_LONG_HTF`, `SUPERTREND_LONG_NOHTF`
- `JMA_LONG_HTF`, `JMA_LONG_NOHTF`
- `KAMA_LONG_HTF`, `KAMA_LONG_NOHTF`
- `EMA_LONG_HTF`, `EMA_LONG_NOHTF`
- `SMA_LONG_HTF`, `SMA_LONG_NOHTF`

**SHORT Categories:**
- `SUPERTREND_SHORT_HTF`, `SUPERTREND_SHORT_NOHTF`
- `JMA_SHORT_HTF`, `JMA_SHORT_NOHTF`
- `KAMA_SHORT_HTF`, `KAMA_SHORT_NOHTF`
- `EMA_SHORT_HTF`, `EMA_SHORT_NOHTF`
- `SMA_SHORT_HTF`, `SMA_SHORT_NOHTF`

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
- Uses **separate optimized parameters** for LONG vs SHORT
- Signal flow:
  - Bullish crossover → COVER (if short) → BUY
  - Bearish crossover → SELL (if long) → SHORT

## Strategies

### SUPERTREND
- **LONG**: `period` (10, 14, 20), `multiplier` (2.0, 3.0, 4.0)
- **SHORT**: `period` (7, 10, 14), `multiplier` (1.5, 2.0, 2.5, 3.0)
- Signal: Direction change

### JMA (Jurik Moving Average)
- **LONG**: `fast` (7, 10, 14), `slow` (21, 30, 50)
- **SHORT**: `fast` (5, 7, 10), `slow` (14, 21, 30)
- Signal: Fast/Slow crossover

### KAMA (Kaufman Adaptive MA)
- **LONG**: `period` (10, 14, 20), `signal` (10, 14, 21)
- **SHORT**: `period` (7, 10, 14), `signal` (7, 10, 14)
- Signal: KAMA/Signal line crossover

### EMA (Exponential MA)
- **LONG**: `fast` (8, 12, 20), `slow` (21, 26, 50)
- **SHORT**: `fast` (5, 8, 12), `slow` (13, 21, 26)
- Signal: Fast/Slow crossover

### SMA (Simple MA)
- **LONG**: `fast` (10, 20, 30), `slow` (50, 100, 200)
- **SHORT**: `fast` (5, 10, 20), `slow` (20, 30, 50)
- Signal: Fast/Slow crossover

## HTF Filter

Higher Time Frame filter uses weekly Supertrend to filter daily signals:
- `_HTF` suffix: Strategy with HTF filter enabled
- `_NOHTF` suffix: Strategy without HTF filter
- LONG: Only buy when weekly trend is bullish
- SHORT: Only short when weekly trend is bearish

## Output Files

| File | Description |
|------|-------------|
| `long_short_categorized.json` | **Separate LONG/SHORT assignments** |
| `htf_categorized_results.json` | Combined strategy assignments |
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

### Scenario 1: Separate LONG/SHORT Optimization (Recommended)

Best results with direction-specific parameters:

```bash
# 1. Start TWS/Gateway

# 2. Optimize LONG and SHORT separately
python new5.py --long-short

# 3. Run portfolio simulation (Long & Short)
python portfolio_simulation.py 90

# 4. View results
open portfolio_report.html
```

### Scenario 2: Long-Only Trading

If short strategies underperform:

```bash
# 1. Optimize (or use existing)
python new5.py --long-short

# 2. Run simulation with LONG only
python portfolio_simulation.py 90 --long-only
```

### Scenario 3: Walk-Forward Analysis

```bash
# Optimize on 9 months, test on last 3 months
python new5.py --walk-forward

# Run simulation
python portfolio_simulation.py 90
```

### Scenario 4: Quick Strategy Scan

Find best strategies for all symbols:

```bash
python new5.py --htf-full
```

### Scenario 5: Custom Simulation Period

```bash
# 6-month simulation
python portfolio_simulation.py 180

# 1-year simulation, long only
python portfolio_simulation.py 365 --long-only
```

## Complete Workflow Example

```bash
# Step 1: Ensure TWS is running on port 7497

# Step 2: Separate LONG/SHORT optimization
# - Different parameters for each direction
# - Saves to long_short_categorized.json
python new5.py --long-short

# Step 3: Portfolio simulation
# - Uses direction-specific parameters
# - Simulates last 90 days with $20,000
# - Generates portfolio_report.html
python portfolio_simulation.py 90

# Alternative: Long-only if shorts underperform
python portfolio_simulation.py 90 --long-only

# Step 4: Review results
open portfolio_report.html
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

**long_short_categorized.json (NEW):**
```json
{
  "long_categories": {
    "SUPERTREND_LONG_HTF": [
      {"symbol": "AAPL", "return": 0.45, "params": {"period": 10, "multiplier": 3.0}}
    ]
  },
  "short_categories": {
    "SUPERTREND_SHORT_HTF": [
      {"symbol": "TSLA", "return": 0.25, "params": {"period": 7, "multiplier": 2.0}}
    ]
  }
}
```

## License

Private use only.

## Author

Trading System v5.0
