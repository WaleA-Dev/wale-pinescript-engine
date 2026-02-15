#!/usr/bin/env python3
"""Quick sanity check for a strategy on one dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bar_returns import compute_bar_returns, compute_metrics
from src.strategies import get_strategy_class


def parse_args():
    p = argparse.ArgumentParser(description="Quick strategy sanity validation")
    p.add_argument("strategy", help="Strategy name")
    p.add_argument("data", help="CSV path with OHLC")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.data)
    cols = {c.lower().strip(): c for c in df.columns}
    for c in ("open", "high", "low", "close"):
        if c not in cols:
            raise ValueError(f"Missing {c} column")
    run = pd.DataFrame({c: pd.to_numeric(df[cols[c]], errors="coerce") for c in ("open", "high", "low", "close")})
    run = run.dropna().reset_index(drop=True)

    strategy_cls = get_strategy_class(args.strategy)
    strategy = strategy_cls()
    signals = pd.Series(strategy.generate_signals(run), index=run.index)
    run["signal"] = signals
    run = compute_bar_returns(run)
    m = compute_metrics(run["strategy_return"])

    print(f"\nQUICK VALIDATION: {args.strategy}")
    print(f"Data rows: {len(run)}")
    print(f"Signals: long={int(np.sum(signals == 1))}, short={int(np.sum(signals == -1))}, flat={int(np.sum(signals == 0))}")
    print(f"Profit Factor: {m['profit_factor']:.4f}")
    print(f"Sharpe Ratio: {m['sharpe_ratio']:.4f}")
    print(f"Total Return: {m['total_return']:.2%}")
    print(f"Max Drawdown: {m['max_drawdown']:.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
