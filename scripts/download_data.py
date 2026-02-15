#!/usr/bin/env python3
"""Download free market data from Yahoo Finance."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import download_yahoo


DEFAULT_SYMBOLS = ["QQQ", "SPY", "BTC-USD"]


def parse_args():
    p = argparse.ArgumentParser(description="Download market data")
    p.add_argument("symbols", nargs="*", default=DEFAULT_SYMBOLS, help="Symbols to download")
    p.add_argument("--interval", default="1d", help="Data interval (default: 1d)")
    p.add_argument("--start", default="2014-01-01", help="Start date")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    for symbol in args.symbols:
        try:
            df = download_yahoo(symbol, interval=args.interval, start=args.start)
            print(f"  {symbol}: {len(df)} bars downloaded\n")
        except Exception as e:
            print(f"  {symbol}: ERROR - {e}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
