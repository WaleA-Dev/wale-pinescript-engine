# Engine Architecture

How the PineScript to Python backtesting engine works under the hood.

---

## Overview

The engine has four main stages:

1. **Parse** - Extract parameters from PineScript
2. **Compute** - Calculate indicators on OHLC data
3. **Execute** - Run backtest with position management
4. **Validate** - Compare results to TradingView

---

## Stage 1: PineScript Parsing

The parser extracts strategy parameters from your `.pine` file using regex patterns.

### What Gets Extracted

```python
@dataclass
class StrategyParams:
    stop_loss_pct: float
    trailing_pct: float
    profit_target_pct: float
    use_ob_exit: bool
    use_ema_filter: bool
    entry_threshold: float
    use_consolidation_filter: bool
    ema_slope_lookback: int
    ema_slope_threshold: float
    range_lookback: int
    range_threshold: float
    use_momentum_confirm: bool
    momentum_ema_fast: int
    momentum_ema_slow: int
    use_adx_filter: bool
    adx_length: int
    adx_threshold: float
```

### Parsing Logic

The parser handles:

1. **Direct inputs** - `input.float()`, `input.bool()`, `input.int()`
2. **Presets** - Ternary expressions that map preset names to values
3. **Default values** - Falls back to defaults when parsing fails

```python
def _parse_input_value(text: str, name: str, kind: str, default):
    pattern = rf"{re.escape(name)}\s*=\s*input\.{kind}\(\s*([^\s,]+)"
    match = re.search(pattern, text)
    if not match:
        return default
    # ... parse and return value
```

---

## Stage 2: Indicator Computation

Indicators are computed once on the full OHLC dataset before the backtest loop.

### Indicator Library

| Function | PineScript Equivalent | Notes |
|----------|----------------------|-------|
| `ema(series, length)` | `ta.ema()` | Uses SMA seed |
| `rma(series, length)` | `ta.rma()` | Alpha = 1/length |
| `atr(h, l, c, length)` | `ta.atr()` | Uses RMA smoothing |
| `adx(h, l, c, length)` | `ta.adx()` | Full DMI calculation |

### EMA Implementation

```python
def ema(series: np.ndarray, length: int) -> np.ndarray:
    alpha = 2.0 / (length + 1)
    out = np.full(len(series), np.nan, dtype=float)
    
    # Find first valid window for SMA seed
    start_idx = None
    for i in range(length - 1, len(series)):
        window = series[i - length + 1 : i + 1]
        if np.isnan(window).any():
            continue
        out[i] = window.mean()  # SMA seed
        start_idx = i + 1
        break
    
    if start_idx is None:
        return out
    
    # Recursive EMA calculation
    for i in range(start_idx, len(series)):
        if np.isnan(series[i]) or np.isnan(out[i - 1]):
            out[i] = np.nan
        else:
            out[i] = alpha * series[i] + (1 - alpha) * out[i - 1]
    
    return out
```

### Why SMA Seed Matters

PineScript initializes EMA with the simple moving average of the first `length` values. If you start from bar 0 with no seed:

- Your EMA will diverge from TradingView on early bars
- This affects all dependent indicators
- Trades in the first few months may be wrong

---

## Stage 3: Backtest Execution

The backtest loop processes bars sequentially, managing position state.

### Order Flow

```
Bar N:
  1. Execute pending exit (if any) at open price
  2. Execute pending entry (if any) at open price
  3. Check stop loss against bar low
  4. Check trailing stop against bar low
  5. Check profit target against bar high
  6. Check exit signals (OB cross, etc.)
  7. Check entry signals
  8. Queue any new orders for next bar

Bar N+1:
  ... repeat ...
```

### Position State

```python
@dataclass
class Trade:
    trade_id: int
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_signal: Optional[str] = None
    qty: int = 0
    entry_value: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    max_high: float = 0.0
    min_low: float = 0.0
    trailing_active: bool = False
    trail_stop: Optional[float] = None
    bars_in_trade: int = 0
```

### Fill Timing

| Order Type | When Queued | When Filled | Fill Price |
|------------|-------------|-------------|------------|
| Market Entry | Bar N close | Bar N+1 open | Open price |
| Market Exit | Bar N close | Bar N+1 open | Open price |
| Stop Loss | - | Intrabar | Stop price (or open if gap) |
| Trailing Stop | - | Intrabar | Trail price (or open if gap) |

---

## Stage 4: Validation

The validator compares your trades against TradingView's Excel export.

### Comparison Logic

```python
def compare_trades(excel_trades, our_trades, last_csv_time):
    for idx, (excel, ours) in enumerate(zip(excel_trades, our_trades)):
        diffs = []
        
        if excel["entry_time"] != ours["entry_time"]:
            diffs.append("entry time")
        if excel["exit_time"] != ours["exit_time"]:
            diffs.append("exit time")
        if abs(excel["entry_price"] - ours["entry_price"]) > 0.01:
            diffs.append("entry price")
        # ... more checks ...
        
        if diffs:
            # Handle open trade exception
            if is_last and is_open_trade:
                if excel_exit_time > last_csv_time:
                    return (True, "All closed trades matched.", ...)
            
            return (False, f"Trade {idx} mismatch: {diffs}", ...)
    
    return (True, "All trades matched.", ...)
```

### Sharpe Ratio Matching

TradingView's Sharpe calculation is proprietary. The engine tries multiple methods:

1. Daily equity returns (simple)
2. Daily equity returns (log)
3. Bar-by-bar returns
4. Trade-by-trade returns
5. Excess returns

It selects the variant closest to the TradingView target and reports which was used.

---

## Data Flow Diagram

```
                    +----------------+
                    |  PineScript    |
                    |  (.pine file)  |
                    +-------+--------+
                            |
                            v
                    +-------+--------+
                    |     Parser     |
                    | (regex-based)  |
                    +-------+--------+
                            |
                            v
+----------------+  +-------+--------+
|   OHLC CSV     |->|   Indicator    |
|   (price data) |  |   Computation  |
+----------------+  +-------+--------+
                            |
                            v
                    +-------+--------+
                    |    Backtest    |
                    |     Loop       |
                    +-------+--------+
                            |
        +-------------------+-------------------+
        |                                       |
        v                                       v
+-------+--------+                      +-------+--------+
|   Trade List   |                      |  Equity Curve  |
|   (CSV/JSON)   |                      |    (Series)    |
+-------+--------+                      +-------+--------+
        |                                       |
        v                                       v
+-------+--------+                      +-------+--------+
|   Validator    |<---------------------|    Metrics     |
| (vs TradingView)|                     | (PF, DD, etc.) |
+----------------+                      +----------------+
```

---

## Configuration

### BacktestConfig

```python
@dataclass
class BacktestConfig:
    initial_capital: float = 100000.0
    commission_pct: float = 0.1
    order_size_pct: float = 100.0
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--csv` | Required | OHLC data file |
| `--pine` | Required | PineScript file |
| `--excel` | None | TradingView export |
| `--initial_capital` | 100000 | Starting capital |
| `--commission_pct` | 0.1 | Commission rate |
| `--holdout_months` | 6 | OOS holdout period |

---

## Error Handling

The engine provides detailed error messages for common issues:

| Error | Cause | Solution |
|-------|-------|----------|
| "Trade count mismatch" | Different number of trades | Check data alignment |
| "Entry time mismatch" | Wrong bar for entry | Check signal timing |
| "Exit price mismatch" | Wrong fill price | Check stop/trail logic |
| "PnL mismatch" | Wrong profit calculation | Check commission model |

Each error includes a hint suggesting the most likely cause.
