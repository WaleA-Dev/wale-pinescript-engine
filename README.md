# Wale PineScript to Python Engine

An engine that converts TradingView PineScript strategies to Python, enabling local backtesting with exact trade-by-trade validation.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

---

## Project Structure

```
wale-pinescript-engine/
├── backtest_engine.py      # Main CLI entry point
├── requirements.txt        # Python dependencies
├── src/
│   ├── __init__.py
│   ├── indicators.py       # EMA, RMA, ATR, ADX, RSI, MACD, etc.
│   ├── parser.py           # PineScript parameter extraction
│   ├── backtest.py         # Core backtest engine
│   └── validator.py        # TradingView comparison
├── tests/
│   ├── __init__.py
│   └── test_indicators.py  # Indicator unit tests
└── docs/
    ├── 01-engine-architecture.md
    ├── 02-accuracy-testing.md
    └── 03-cli-reference.md
```

---

## What This Engine Does

1. **Parses PineScript** - Extracts strategy parameters, indicator logic, and entry/exit rules from your `.pine` files
2. **Rebuilds in Python** - Implements equivalent indicator calculations (EMA, ATR, ADX, etc.) matching PineScript's math exactly
3. **Runs Backtests** - Executes the strategy on OHLC data with proper fill timing, commission, and position sizing
4. **Validates Output** - Compares trade-by-trade results against TradingView's Excel export to ensure accuracy
5. **Feeds Analysis** - Produces trade lists for Monte Carlo stress testing and robustness analysis

---

## Why Build This?

TradingView's backtester is great for visualization but has limitations:

- **No programmatic access** - You cannot run thousands of parameter combinations
- **No custom robustness tests** - Walk-forward, CPCV, permutation tests are impossible
- **Limited export** - Getting raw trade data requires manual Excel downloads

This engine solves all of that while maintaining TradingView-equivalent accuracy.

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/WaleA-Dev/wale-pinescript-engine.git
cd wale-pinescript-engine

# Install dependencies
pip install -r requirements.txt

# Or install individually
pip install numpy pandas scipy openpyxl matplotlib
```

### Basic Usage

```bash
python backtest_engine.py \
    --csv your_ohlc_data.csv \
    --pine your_strategy.pine \
    --run_step1 true
```

### With TradingView Validation

```bash
python backtest_engine.py \
    --csv your_ohlc_data.csv \
    --pine your_strategy.pine \
    --excel tradingview_export.xlsx \
    --run_step1 true
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [Engine Architecture](docs/01-engine-architecture.md) | How the engine works |
| [Accuracy Testing](docs/02-accuracy-testing.md) | Validating your implementation |
| [CLI Reference](docs/03-cli-reference.md) | Command-line documentation |

---

## Architecture Overview

```
[PineScript File] --> [Parser] --> [StrategyParams]
                                         |
[OHLC CSV Data] --> [Indicator Engine] --+--> [Backtest Engine] --> [Trade List]
                                                                          |
[TradingView Excel Export] --> [Validator] <------------------------------+
                                    |
                              [Validation Report]
```

### Core Components

| Component | Purpose |
|-----------|---------|
| Parameter Parser | Extracts inputs from PineScript |
| Indicator Library | EMA, RMA, ATR, ADX implementations |
| Signal Logic | Entry/exit condition evaluation |
| Backtest Loop | Position management, fill execution |
| Trade Validator | Compare vs TradingView exports |

---

## Indicator Implementations

The engine implements PineScript indicators with exact mathematical equivalence:

### Exponential Moving Average (EMA)

```python
def ema(series: np.ndarray, length: int) -> np.ndarray:
    alpha = 2.0 / (length + 1)
    # Initialize with SMA of first 'length' values
    # Then apply recursive: out[i] = alpha * series[i] + (1 - alpha) * out[i-1]
```

**Key detail:** PineScript initializes EMA with an SMA seed. Many Python implementations skip this, causing divergence on early bars.

### Running Moving Average (RMA)

```python
def rma(series: np.ndarray, length: int) -> np.ndarray:
    alpha = 1.0 / length  # Different from EMA!
    # Same recursive structure as EMA but with alpha = 1/length
```

**Key detail:** RMA uses `alpha = 1/length`, not `2/(length+1)`. This is used internally by ATR and ADX.

### Average True Range (ATR)

```python
def atr(high, low, close, length):
    tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
    return rma(tr, length)  # NOT ema!
```

**Key detail:** TradingView's ATR uses RMA, not EMA. Using EMA will cause drift over time.

---

## Fill Timing Model

The engine matches TradingView's default execution model:

| Event | When Evaluated | When Filled |
|-------|----------------|-------------|
| Entry Signal | Bar N close | Bar N+1 open |
| Exit Signal | Bar N close | Bar N+1 open |
| Stop Loss | Bar N | Bar N (intrabar at stop price) |
| Trailing Stop | Bar N | Bar N (intrabar at trail price) |

**Important:** Signals are computed on bar close but executed on next bar open. This is PineScript's `process_orders_on_close=false` behavior.

---

## Validation Against TradingView

The engine performs trade-by-trade validation:

### What Gets Compared

| Field | Tolerance | Notes |
|-------|-----------|-------|
| Entry Time | Exact | Must match to the bar |
| Exit Time | Exact | Must match to the bar |
| Entry Price | 0.01 | Allows for rounding |
| Exit Price | 0.01 | Allows for rounding |
| Exit Signal | Exact | SL, Trail, PT, OB, etc. |
| PnL | 2% | Accounts for commission differences |

### Open Trade Handling

If validation fails only on the final trade and that trade is still open:

```python
if is_last_trade and excel_exit_signal == "Open" and our_exit_signal == "Open":
    if excel_exit_time > last_csv_time:
        # Ignore this trade - dataset ended before trade closed
        return PASS
```

---

## Common Accuracy Issues

| Issue | Symptom | Fix |
|-------|---------|-----|
| EMA Drift | First trades differ | Initialize with SMA seed |
| ATR Wrong | Stop losses trigger wrong | Use RMA not EMA |
| Fill Timing | Trades one bar off | Execute on next bar open |
| Position Sizing | PnL accumulates error | Use floor() for quantity |

See [Accuracy Testing Guide](docs/02-accuracy-testing.md) for detailed debugging steps.

---

## Output Files

```
backtest/out/
├── validation_report.txt       # TradingView comparison
├── research_outputs/
│   └── step1/
│       ├── trade_list.csv      # All trades
│       ├── equity_curve.csv    # Bar-by-bar equity
│       └── step1_report.txt    # Performance summary
└── plots/
    ├── step1_equity_drawdown.png
    └── step1_monthly_returns.png
```

---

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_indicators.py -v
```

---

## Programmatic Usage

```python
from src.indicators import ema, atr, adx, rsi
from src.parser import PineScriptParser, StrategyParams
from src.backtest import BacktestEngine, BacktestConfig
from src.validator import TradingViewValidator

# Load and parse strategy
parser = PineScriptParser(pine_path='strategy.pine')
params = parser.parse_params()

# Configure backtest
config = BacktestConfig(
    initial_capital=100000,
    commission_pct=0.1,
)

# Run backtest
engine = BacktestEngine(config=config, params=params)
result = engine.run(df)  # df is your OHLC DataFrame

# Access results
print(f"Total trades: {result.total_trades}")
print(f"Win rate: {result.win_rate:.1f}%")
print(f"Profit factor: {result.profit_factor:.2f}")

# Validate against TradingView
validator = TradingViewValidator(excel_path='tv_export.xlsx')
validation = validator.validate(result.trades)
print(validation.message)
```

---

## License

CC BY-NC-SA 4.0. You can use, modify, and share this code, but NOT for commercial purposes or sale. If you build on it, share your improvements under the same license. No warranty. Use at your own risk.

---

## Related Projects

- [Wale Monte Carlo Engine](https://github.com/WaleA-Dev/wale-montecarlo-engine) - Stress test your validated trades with 200K+ simulations
