"""
Universal data loader.
Sources: Local CSV files, Yahoo Finance (yfinance).
No paid APIs. No Databento. No API keys required for default usage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
OHLC_COLUMNS = ["open", "high", "low", "close"]

# Column name variations to auto-detect
_COL_ALIASES = {
    "open": ["open", "o", "open_price", "Open", "OPEN"],
    "high": ["high", "h", "high_price", "High", "HIGH"],
    "low": ["low", "l", "low_price", "Low", "LOW"],
    "close": ["close", "c", "close_price", "Close", "CLOSE", "adj close", "adj_close"],
    "volume": ["volume", "v", "vol", "Volume", "VOLUME", "Vol"],
    "timestamp": [
        "timestamp", "date", "datetime", "time", "Date", "Datetime",
        "Timestamp", "DATE", "TIME", "DATETIME",
    ],
}

# Yahoo Finance symbol mapping for index proxies
SYMBOL_MAP = {
    "NDX": "QQQ",
    "NASDAQ:NDX": "QQQ",
    "SPX": "SPY",
    "SP500": "SPY",
    "BTC": "BTC-USD",
    "BTCUSD": "BTC-USD",
}

CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache"


def _resolve_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map varied column names to standardized lowercase names."""
    col_map = {}
    lower_cols = {c.lower().strip(): c for c in df.columns}

    for standard, aliases in _COL_ALIASES.items():
        for alias in aliases:
            key = alias.lower().strip()
            if key in lower_cols and standard not in col_map:
                col_map[standard] = lower_cols[key]
                break

    if not all(c in col_map for c in OHLC_COLUMNS):
        missing = [c for c in OHLC_COLUMNS if c not in col_map]
        raise ValueError(f"Cannot find OHLC columns. Missing: {missing}. Available: {list(df.columns)}")

    result = pd.DataFrame()
    for std_name, orig_name in col_map.items():
        if std_name == "timestamp":
            result[std_name] = pd.to_datetime(df[orig_name], errors="coerce", utc=True)
        else:
            result[std_name] = pd.to_numeric(df[orig_name], errors="coerce")

    # Fill missing optional columns
    if "volume" not in result.columns:
        result["volume"] = 0
    if "timestamp" not in result.columns:
        result["timestamp"] = pd.RangeIndex(len(result))

    return result


def load_csv(filepath: str | Path) -> pd.DataFrame:
    """Load any CSV/TSV with OHLC data. Auto-detect delimiter and columns."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    # Auto-detect delimiter
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        sample = f.read(2048)

    if "\t" in sample and sample.count("\t") > sample.count(","):
        sep = "\t"
    elif ";" in sample and sample.count(";") > sample.count(","):
        sep = ";"
    else:
        sep = ","

    df = pd.read_csv(path, sep=sep)
    result = _resolve_columns(df)

    # Drop rows with NaN in OHLC
    result = result.dropna(subset=OHLC_COLUMNS).reset_index(drop=True)

    # Set DatetimeIndex if timestamp is datetime
    if pd.api.types.is_datetime64_any_dtype(result["timestamp"]):
        result = result.set_index("timestamp").sort_index()
    else:
        result = result.set_index("timestamp")

    return result


def lookup_yahoo(symbol: str) -> dict:
    """Validate a ticker and return its info + max available date range."""
    import yfinance as yf

    mapped = SYMBOL_MAP.get(symbol.upper(), symbol)
    ticker = yf.Ticker(mapped)
    try:
        info = ticker.info or {}
    except Exception:
        info = {}

    name = info.get("shortName") or info.get("longName") or mapped
    asset_type = info.get("quoteType", "unknown")

    # Fetch a small history to find the earliest date
    try:
        hist = ticker.history(period="max", interval="1d")
        if hist.empty:
            return {"valid": False, "error": f"No data found for '{symbol}'", "symbol": mapped}
        start = str(hist.index[0].date())
        end = str(hist.index[-1].date())
        bars = len(hist)
    except Exception as e:
        return {"valid": False, "error": str(e), "symbol": mapped}

    return {
        "valid": True, "symbol": mapped, "name": name,
        "type": asset_type, "start": start, "end": end, "bars": bars,
    }


def download_yahoo(
    symbol: str,
    interval: str = "1d",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Download from Yahoo Finance. Cache to data/cache/.
    If start is None or empty, fetches maximum available data (period='max').
    """
    import yfinance as yf

    mapped = SYMBOL_MAP.get(symbol.upper(), symbol)
    cache_file = CACHE_DIR / f"{mapped}_{interval}.csv"

    print(f"Downloading {mapped} ({interval}) from Yahoo Finance...")
    ticker = yf.Ticker(mapped)

    if start:
        kwargs = {"interval": interval, "start": start}
        if end:
            kwargs["end"] = end
        hist = ticker.history(**kwargs)
    else:
        # No start date: fetch max available data
        hist = ticker.history(period="max", interval=interval)

    if hist.empty:
        raise ValueError(f"No data returned for {mapped}. Check the ticker symbol is valid.")

    # Standardize columns
    hist = hist.reset_index()
    col_map = {}
    for c in hist.columns:
        cl = c.lower().strip()
        if cl in ("date", "datetime"):
            col_map[c] = "timestamp"
        elif cl in ("open", "high", "low", "close", "volume"):
            col_map[c] = cl

    hist = hist.rename(columns=col_map)
    keep = [c for c in OHLCV_COLUMNS if c in hist.columns]
    hist = hist[keep]

    for c in OHLC_COLUMNS:
        hist[c] = pd.to_numeric(hist[c], errors="coerce")
    if "volume" in hist.columns:
        hist["volume"] = pd.to_numeric(hist["volume"], errors="coerce").fillna(0).astype(int)

    hist = hist.dropna(subset=OHLC_COLUMNS).reset_index(drop=True)

    # Cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    hist.to_csv(cache_file, index=False)
    print(f"Cached {len(hist)} bars to {cache_file}")

    if "timestamp" in hist.columns:
        hist["timestamp"] = pd.to_datetime(hist["timestamp"], errors="coerce")
        hist = hist.set_index("timestamp").sort_index()

    return hist


def load_data(source: str, **kwargs) -> pd.DataFrame:
    """Universal entry point. Auto-detect if source is filepath or symbol."""
    path = Path(source)
    if path.exists() and path.suffix.lower() in (".csv", ".tsv", ".txt"):
        return load_csv(source)

    # Check cache first
    interval = kwargs.get("interval", "1d")
    mapped = SYMBOL_MAP.get(source.upper(), source)
    cache_file = CACHE_DIR / f"{mapped}_{interval}.csv"
    if cache_file.exists():
        print(f"Loading from cache: {cache_file}")
        return load_csv(cache_file)

    # Download from Yahoo
    return download_yahoo(source, **kwargs)
