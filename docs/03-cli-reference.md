# CLI Reference

Command-line interface documentation for the PineScript to Python engine.

---

## Backtest Engine

The main backtest runner.

```bash
python backtest_engine.py [OPTIONS]
```

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--csv` | Path to OHLC CSV data file |
| `--pine` | Path to PineScript strategy file |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--excel` | None | TradingView Excel export for validation |
| `--run_step1` | false | Generate Step 1 research outputs |
| `--run_step2` | false | Run robustness tests |
| `--holdout_months` | 6 | Out-of-sample holdout period |
| `--initial_capital` | 100000 | Starting capital |
| `--commission_pct` | 0.1 | Commission as percent of trade value |

---

## Examples

### Basic Backtest

```bash
python backtest_engine.py \
    --csv jan_2_data_to_now.csv \
    --pine strategy.pine
```

### With Validation

```bash
python backtest_engine.py \
    --csv jan_2_data_to_now.csv \
    --pine strategy.pine \
    --excel tradingview_export.xlsx \
    --run_step1 true
```

### Full Research Run

```bash
python backtest_engine.py \
    --csv jan_2_data_to_now.csv \
    --pine strategy.pine \
    --excel tradingview_export.xlsx \
    --run_step1 true \
    --run_step2 true \
    --holdout_months 6
```

---

## Output Locations

```
backtest/out/
├── validation_report.txt       # TradingView comparison
├── research_outputs/
│   ├── step1/
│   │   ├── trade_list.csv      # All trades
│   │   ├── equity_curve.csv    # Bar-by-bar equity
│   │   └── step1_report.txt    # Performance summary
│   └── step2/
│       └── ...                 # Robustness results
└── plots/
    ├── step1_equity_drawdown.png
    ├── step1_monthly_returns.png
    └── step1_trade_pnl_hist.png
```

---

## Environment Setup

### Required Packages

```bash
pip install numpy pandas scipy openpyxl matplotlib
```

### Virtual Environment (Recommended)

```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

---

## Troubleshooting

### Module Not Found

```bash
# Ensure correct directory
cd "path/to/your/project"

# Ensure venv activated
.\venv\Scripts\activate
```

### Permission Denied (Windows)

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Data Not Found

Ensure your CSV path is correct:
```bash
# Relative path
--csv data/prices.csv

# Absolute path
--csv "C:\Users\you\data\prices.csv"
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Basic backtest | `python backtest_engine.py --csv data.csv --pine strat.pine` |
| With validation | Add `--excel export.xlsx` |
| Generate reports | Add `--run_step1 true` |
| Run robustness | Add `--run_step2 true` |
