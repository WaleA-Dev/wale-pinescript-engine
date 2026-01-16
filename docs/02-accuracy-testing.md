# Accuracy Testing Guide

How to verify your PineScript to Python conversion produces identical results to TradingView.

---

## The Testing Philosophy

**Goal:** Every closed trade should match TradingView exactly. If it does not, something is wrong.

We do not aim for "close enough" - we aim for exact replication. Small differences compound over time and can completely change analysis results.

---

## Pre-Testing Checklist

Before running any tests, verify:

### 1. Data Alignment

```
[ ] CSV start date matches TradingView chart start
[ ] CSV end date matches TradingView chart end
[ ] Bar count matches (check for missing weekends/holidays)
[ ] Timezone is consistent (UTC vs exchange local)
[ ] OHLC values match TradingView to 2 decimal places
```

### 2. Strategy Settings

```
[ ] All input parameters match between Pine and Python
[ ] Commission rate matches
[ ] Initial capital matches
[ ] Order size percentage matches
[ ] Pyramiding setting matches (usually disabled)
```

### 3. Data Quality

```
[ ] No NaN values in OHLC columns
[ ] High >= Low for all bars
[ ] High >= Open and High >= Close for all bars
[ ] Low <= Open and Low <= Close for all bars
[ ] Timestamps are monotonically increasing
```

---

## The Testing Process

### Phase 1: Indicator Verification

Before testing the full strategy, verify individual indicators:

**Test 1: EMA Verification**
```python
# Compare your EMA output to TradingView's
# Export EMA values from TradingView using plot() and data export
for i in range(100):
    assert abs(our_ema[i] - tv_ema[i]) < 0.0001, f"EMA mismatch at bar {i}"
```

**Test 2: ATR Verification**
```python
# ATR is critical for stop loss calculations
# Verify it uses RMA not EMA
for i in range(100):
    assert abs(our_atr[i] - tv_atr[i]) < 0.0001, f"ATR mismatch at bar {i}"
```

### Phase 2: Signal Verification

Before testing fills, verify the signals fire on the same bars:

```python
# Export entry bars from TradingView
# Compare to your signal generation
our_entries = [i for i in range(len(df)) if entry_signal(df, i, params)]
tv_entries = [...]  # From TradingView export

assert our_entries == tv_entries, f"Entry signal mismatch"
```

### Phase 3: Full Trade Comparison

Run the full backtest and compare:

```bash
python backtest_engine.py \
    --csv data.csv \
    --pine strategy.pine \
    --excel tradingview_export.xlsx \
    --run_step1 true
```

Check `validation_report.txt` for results.

---

## Common Failure Modes

### Failure Mode 1: Off-by-One Bar

**Symptom:** Every trade entry/exit is exactly one bar early or late.

**Causes:**
- Signal evaluated on wrong bar (close vs open)
- Fill executed on wrong bar (same bar vs next bar)
- Index off-by-one in loop

**Debug:**
```python
print(f"Signal on bar {signal_bar}, time={df.loc[signal_bar, 'time']}")
print(f"Fill on bar {fill_bar}, time={df.loc[fill_bar, 'time']}")
```

### Failure Mode 2: First Trade Mismatch

**Symptom:** First trade differs, rest are fine.

**Causes:**
- Indicator warmup period insufficient
- EMA not initialized with SMA seed
- Strategy requires N bars of history before first signal

**Debug:**
```python
print(f"First non-NaN EMA: bar {np.argmax(~np.isnan(df['ema200']))}")
```

### Failure Mode 3: Stop Loss Trigger Price

**Symptom:** Exit price differs on stop loss trades.

**Causes:**
- Gap handling (open beyond stop vs fill at stop)
- ATR calculation mismatch
- Trailing stop activation logic

**Debug:**
```python
print(f"Entry: {trade.entry_price}")
print(f"ATR at entry: {atr_at_entry}")
print(f"Stop level: {stop_level}")
print(f"Bar that triggered stop: low={bar_low}")
```

### Failure Mode 4: Trailing Stop Activation

**Symptom:** Some trades exit via trailing stop when they should not.

**Causes:**
- Trailing activation threshold differs
- Trail offset calculation differs
- Trail updates on close vs high

**Debug:**
```python
for bar in trade_bars:
    print(f"Bar {bar}: high={high}, trail_active={trail_active}, trail={trail}")
```

### Failure Mode 5: Cumulative Drift

**Symptom:** Trades match early but diverge later.

**Causes:**
- Floating point accumulation in indicators
- Position size depends on equity which diverges
- Small PnL differences compound

**Debug:**
```python
checkpoints = [100, 500, 1000, 5000]
for cp in checkpoints:
    print(f"Bar {cp}: Our equity={our_equity[cp]}, TV={tv_equity[cp]}")
```

---

## Diagnostic Tools

### Trade Diff Report

```python
def generate_trade_diff(our_trades, tv_trades):
    for i, (ours, theirs) in enumerate(zip(our_trades, tv_trades)):
        diffs = []
        if ours['entry_time'] != theirs['entry_time']:
            diffs.append(f"entry: {ours['entry_time']} vs {theirs['entry_time']}")
        if abs(ours['pnl'] - theirs['pnl']) > 1.0:
            diffs.append(f"pnl: {ours['pnl']:.2f} vs {theirs['pnl']:.2f}")
        
        if diffs:
            print(f"\nTrade {i+1} MISMATCH:")
            for d in diffs:
                print(f"  {d}")
```

### Indicator Snapshot

```python
def indicator_snapshot(df, bar_idx):
    print(f"=== Bar {bar_idx}: {df.loc[bar_idx, 'time']} ===")
    print(f"OHLC: {df.loc[bar_idx, 'open']:.2f} / {df.loc[bar_idx, 'high']:.2f} / "
          f"{df.loc[bar_idx, 'low']:.2f} / {df.loc[bar_idx, 'close']:.2f}")
    print(f"EMA21: {df.loc[bar_idx, 'pivot']:.4f}")
    print(f"ATR14: {df.loc[bar_idx, 'atr14']:.4f}")
```

---

## Acceptance Criteria

### Minimum Acceptance

- All closed trades match entry/exit times exactly
- Entry/exit prices within 0.01
- PnL within 2% per trade
- Total PnL within 1%

### Strict Acceptance

- All of the above PLUS
- Sharpe ratio within 5%
- Max drawdown within 1%
- Monthly returns match within 2%

### Acceptable Differences

Some differences are acceptable if documented:

1. **Open trade at dataset end** - TradingView may show a trade not yet closed
2. **Sharpe methodology** - TradingView's exact formula is proprietary
3. **Sub-penny rounding** - Floating point differences on tiny amounts

---

## Regression Testing

Once you achieve validation, prevent regressions:

### Save Golden Output

```bash
cp backtest/out/research_outputs/step1/trade_list.csv tests/golden_trades.csv
```

### Automated Test

```python
def test_regression():
    result = run_backtest(df, params, config)
    trades = export_trades(result)
    golden = pd.read_csv("tests/golden_trades.csv")
    
    assert len(trades) == len(golden), "Trade count changed"
    
    for i, (ours, gold) in enumerate(zip(trades, golden)):
        assert ours['entry_time'] == gold['entry_time']
        assert abs(ours['pnl'] - gold['pnl']) < 1.0
```

---

## Summary

Accuracy testing is the foundation of trust in your backtest. Without it, any analysis is meaningless.

The process:
1. Verify data alignment first
2. Test indicators in isolation
3. Test signals in isolation
4. Test full trades
5. Document any accepted differences
6. Set up regression tests

Do not skip these steps.
