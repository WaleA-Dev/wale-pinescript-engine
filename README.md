# Wale Backtest Engine

A local backtesting platform that translates PineScript strategies into Python code, runs them against real market data, and validates results with a 4-step statistical pipeline. Ships as both a Flask web app and a standalone Windows EXE.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

---

## Screenshots

### Web Backtester (Flask)

![Web Backtester](docs/screenshots/web_dashboard.png)

The dashboard you get when you launch the desktop app (or run the web app). Pick your data source and symbol in the left sidebar, choose a strategy (or paste PineScript and hit Translate), tweak commission and validation settings, then run the backtest. The right panel shows Summary (trades, profit factor, win rate, drawdown), plus Trades, Charts, and Validation tabs. Everything stays local at `127.0.0.1:5000`.

### Standalone Desktop App (Windows EXE)

Double-click **WaleBacktest.exe** and the full dashboard opens in its own native
window — its own taskbar icon, resizable, no browser involved (rendered with the
Edge WebView2 runtime that ships with Windows 10/11). No Python or terminal
required; closing the window shuts everything down. If WebView2 is missing on a
machine, the app falls back to opening the dashboard in the default browser.
Same engine under the hood whether you run the EXE or `python web_app.py`.

---

## What It Does

1. **Translates PineScript to Python.** Paste a TradingView strategy, hit Translate, and the engine parses indicators, conditions, and entries/exits into a working Python class.

2. **Runs backtests that match TradingView.** Signals are generated on bar close and filled at the next bar's open price, matching TV's default execution model. Entry and exit prices use actual open values, not close.

3. **Validates strategy robustness.** The 4-step pipeline runs in-sample optimization, permutation testing (Monte Carlo p-value), walk-forward out-of-sample testing, and walk-forward permutation testing. A strategy either passes or it doesn't.

4. **Lets you write strategies directly in Python.** The Python editor validates your code by actually running it against synthetic data before saving. If `generate_signals()` crashes, the file gets deleted and you get the error back.

---

## Quick Start

```bash
git clone https://github.com/WaleA-Dev/wale-pinescript-engine.git
cd wale-pinescript-engine
pip install -r requirements.txt

# Launch the web app
python web_app.py
# Opens http://127.0.0.1:5000 in your browser
```

### Build Standalone EXE

```bash
pip install pyinstaller
pyinstaller WaleBacktest.spec
# Run the launcher from dist
./dist/WaleBacktest.exe
```

---

## How to Use the Web App

### 1. Load Data

Pick a source from the sidebar:

- **Alpaca (live, free)** - The primary source. Enter your free Alpaca API key once
  (alpaca.markets — a paper account is enough) and fetch live/historical OHLCV for any
  US stock/ETF from 1-minute to daily bars, free via the IEX feed. Keys are validated
  against Alpaca when you save them, stored locally in `~/.wale_backtest/config.json`,
  and never leave your machine. Both classic header auth and the newer OAuth2
  client-credentials keys work — the app detects which kind you have.
- **Yahoo Finance** - Type a ticker (QQQ, NVDA, BTC-USD), pick an interval (1H, 1D, 1W), click Fetch.
- **Dukascopy** - For forex tick data. Enter a pair (EUR-USD), date range, and resampling period.
- **CSV Upload** - Drop in any OHLCV CSV file.

**Free trial:** without an Alpaca key you get 5 free data downloads (served via Yahoo).
After that, add your own free Alpaca key for unlimited live data. CSV upload is always free.

### 2. Pick or Write a Strategy

The engine ships with 12 built-in strategies (Donchian, EMA Cross, MACD, RSI, NDX Trader, etc). Select one from the dropdown.

Or write your own:

**Pine Editor** - Paste PineScript, click Translate. The engine parses it, generates a Python file, and registers it in the dropdown.

**Python Editor** - Write a `BaseStrategy` subclass directly. The editor pre-fills a working SMA crossover template. Click Save & Register. The backend validates syntax, checks for a `BaseStrategy` subclass, imports the module, and test-runs `generate_signals()` on 100 bars of synthetic data. If anything fails, you get the error and the file is cleaned up.

### 3. Run Backtest

Click "Run Backtest". The Summary tab shows key metrics (Profit Factor, Sharpe, Win Rate, Max Drawdown, Total Return). The Trades tab lists every trade with entry/exit dates, prices, bars held, and P&L. The Charts tab draws equity curve and drawdown with axis labels, date ticks, and a hover crosshair.

### 4. Validate

Click "4-Step Validation" to run the full statistical pipeline:

| Step | What It Does |
|------|-------------|
| 1. In-Sample Optimization | Grid search for best parameters on training data |
| 2. IS Permutation Test | Shuffles returns 200+ times to get a Monte Carlo p-value |
| 3. Walk-Forward OOS | Tests optimized params on unseen data |
| 4. WF Permutation Test | Confirms OOS results aren't just luck |

Final verdict: **VALIDATED**, **OVERFIT**, or **POOR**.

### 5. Export

Every tab has an Export button:

- Summary tab: metrics as CSV
- Trades tab: full trade list as CSV (with dates, prices, bars held, P&L)
- Charts tab: equity curve and drawdown as CSV
- Validation tab: full results as JSON

---

## Execution Model

The engine matches TradingView's default broker emulator:

| Event | Evaluated | Filled |
|-------|-----------|--------|
| `strategy.entry` / `strategy.close` | Bar N close | Bar N+1 open |
| `strategy.exit` stop/limit | placed bar N | checked intrabar from bar N+1 |
| Stop hit | intrabar | at the stop price (or open on gap-through) |
| Limit hit | intrabar | at the limit price (or open on gap-through) |

When both stop and limit are hit in one bar, TradingView's documented bar-path
assumption decides: if the open is closer to the high, price is assumed to travel
open→high→low→close; if closer to the low, open→low→high→close.

Trade prices in the trade list are the actual fill prices (open fills for market
orders, stop/limit levels for bracket exits), not close prices.

---

## Pine Transpiler

The transpiler is a real parser (lexer → expression AST → indentation-aware statement
parser), not regex matching. It compiles Pine strategies into event-driven Python that
runs against a TradingView-style broker emulator:

- **Vectorized where possible** — pure indicator math (`ta.*`, arithmetic, crossovers)
  is hoisted into numpy precomputation, including `ta.*` sub-expressions buried inside
  stateful conditions.
- **Stateful where necessary** — `var` declarations, `:=` mutation, `if`/`else if`/`else`
  blocks, ternaries, series history (`x[1]`), and `strategy.position_size` guards are
  compiled into a per-bar `on_bar()` loop with persistent state.
- **Real order semantics** — `strategy.entry` fills at next bar open;
  `strategy.exit(stop=, limit=, trail_points=, trail_offset=)` become standing orders
  checked intrabar with TradingView's bar-path heuristic (green bar: open→low→high→close);
  `strategy.close` exits at next bar open. Reversals, pyramiding=1, percent-of-equity
  sizing, whole-share quantities, and percent commission all come from the `strategy()`
  declaration.

Supported `ta.*`: ema, sma, rma, wma, hma, vwma, swma, alma, atr, tr, rsi, macd, bb,
stoch, cci, mfi, wpr, obv, stdev, variance, dev, highest, lowest, mom, roc, change,
sum, cum, avg, crossover, crossunder, cross, rising, falling, barssince, valuewhen,
pivothigh, pivotlow, supertrend.

Every construct the transpiler cannot honor produces an explicit warning in the UI —
nothing silently degrades. Not supported: `request.*`, `array.*`, `for`/`while` loops,
user-defined functions, `switch`.

Each translation is also smoke-tested automatically: the generated class is executed
on 300 bars of synthetic data before it's registered, and any per-bar error fails the
translation with the exact exception.

---

## Project Structure

```
wale-pinescript-engine/
  web_app.py                  Flask backend (web UI entry point)
  launcher.py                 Desktop app entry point (native WebView2 window)
  backtest_engine.py          CLI entry point
  templates/
    converge.html             Web UI (single-page dashboard)
  src/
    bar_returns.py            Bar-level return computation with next-bar-open fills
    data_loader.py            Yahoo, Dukascopy, CSV loading
    optimization.py           Grid search
    permutation.py            Monte Carlo permutation engine
    strategies/
      base.py                 BaseStrategy ABC with indicator helpers
      donchian.py             Donchian breakout
      ema_crossover.py        EMA crossover
      ndx_trader.py           NDX trend + RSI pullback
      ...                     12 strategies total
    pine_translator/
      parser.py               PineScript AST extraction
      translator.py           Pine to Python code generation
      pipeline.py             End-to-end translate + save + register
      validator.py            Syntax and structure validation
    validation/
      full_validation.py      4-step validation pipeline
      in_sample_permutation.py
      walk_forward.py
  tests/                      58 unit tests
  data/
    cache/                    Downloaded data cache
    uploads/                  CSV uploads
  docs/
    screenshots/
```

---

## Writing a Python Strategy

Every strategy inherits from `BaseStrategy` and implements `generate_signals()`:

```python
import numpy as np
import pandas as pd
from src.strategies.base import BaseStrategy

class MyStrategy(BaseStrategy):
    def __init__(self, **params):
        super().__init__(**params)
        self.params.setdefault('fast', 10)
        self.params.setdefault('slow', 30)

    def generate_signals(self, df):
        fast = df['close'].rolling(self.params['fast']).mean()
        slow = df['close'].rolling(self.params['slow']).mean()
        signal = pd.Series(0, index=df.index)
        signal[fast > slow] = 1
        signal[fast < slow] = -1
        return signal

    def param_grid(self):
        return {'fast': [5, 10, 20], 'slow': [20, 30, 50]}
```

Signal values: `1` = long, `-1` = short, `0` = flat.

The `param_grid()` method is optional but needed for optimization and validation.

---

## Indicator Helpers

`BaseStrategy` includes these static methods so you don't have to rewrite them:

```python
self.calc_ema(series, span)      # Exponential moving average
self.calc_sma(series, window)    # Simple moving average
self.calc_rsi(series, period)    # RSI with Wilder's RMA (alpha=1/period)
self.calc_atr(df, period)        # Average true range using RMA
self.calc_macd(series, fast, slow, signal)  # Returns (macd, signal_line, histogram)
self.crossover(a, b)             # True when a crosses above b
self.crossunder(a, b)            # True when a crosses below b
```

---

## Running Tests

```bash
python -m pytest tests/ -v

# 58 tests covering bar_returns, strategies, translator,
# parser, permutation, walk-forward, data loader, validation
```

---

## Data Sources

### Alpaca (primary — live, free)
Free live and historical US equities data via Alpaca's IEX feed. Sign up free at
[alpaca.markets](https://alpaca.markets) (paper account works), create API keys, and
paste them into the Data Source panel once. Supports 1min/5min/15min/30min/1H/1D bars,
split-adjusted, paginated to any range (IEX history is thinner before ~2017). Auth
works with classic `APCA-API-KEY-ID`/`APCA-API-SECRET-KEY` headers or OAuth2
client-credentials tokens from `authx.alpaca.markets` — auto-detected. Without keys,
the app allows 5 free trial downloads, then requires a key.

### Yahoo Finance
Type any ticker Yahoo supports. Daily data goes back decades. Hourly data is limited to the last 730 days by Yahoo's API.

### Dukascopy
Forex tick data from Dukascopy's free archive. Specify a date range and resampling period (1min to 1D). Requires `duka-dl` package (`pip install duka-dl`).

### CSV Upload
Any CSV with `open`, `high`, `low`, `close` columns. The first column should be a timestamp. Volume is optional.

---

## Requirements

- Python 3.10+
- Windows 10/11 (for EXE builds)
- Dependencies: Flask, pandas, numpy, scipy, matplotlib, requests, yfinance

Optional:
- duka-dl (for Dukascopy forex data)

---

## License

CC BY-NC-SA 4.0. Free to use and modify for personal and research purposes. Not for commercial use or resale. If you build on it, share your work under the same license. No warranty.

---

## Related Projects

- [Wale Monte Carlo Engine](https://github.com/WaleA-Dev/wale-montecarlo-engine) - Stress test validated trades with 200K+ simulations
