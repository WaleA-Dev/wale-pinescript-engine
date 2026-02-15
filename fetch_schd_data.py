#!/usr/bin/env python3
"""Fetch SCHD 1H data from Databento for testing."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from datetime import datetime
from data_providers.databento_provider import DatabentoProvider

API_KEY = os.environ.get('DATABENTO_API_KEY', 'db-Pg3tLFWEWsyPBqm6dYELTUdPQ7bWv')

provider = DatabentoProvider(api_key=API_KEY)

# SCHD is on NYSE ARCA (AMEX) - auto-detect should pick NYSE
detected = DatabentoProvider.detect_exchange('SCHD')
print(f"Auto-detected exchange for SCHD: {detected}")

print(f"\nFetching SCHD 1H data from Databento (NYSE/AMEX)...")
df = provider.fetch_ohlcv(
    symbol='SCHD',
    start_date=datetime(2014, 1, 1),  # Will be clamped to 2018-05-01
    end_date=datetime(2026, 2, 7),
    timeframe='1H',
    dataset='AUTO',
    rth_only=True,
)

if df.empty:
    print("No data returned!")
    sys.exit(1)

print(f"\nLoaded {len(df)} bars (raw)")
print(f"Date range: {df['time'].min()} to {df['time'].max()}")
print(f"Price range: ${df['close'].min():.2f} to ${df['close'].max():.2f}")

# Apply SCHD 3:1 stock split adjustment (effective October 10, 2024)
# TradingView shows split-adjusted prices, so we divide pre-split prices by 3
import pandas as pd
SPLIT_DATE = pd.Timestamp('2024-10-11', tz='UTC')  # Split effective after Oct 10 close
pre_split = df['time'] < SPLIT_DATE
n_pre = pre_split.sum()
if n_pre > 0:
    for col in ['open', 'high', 'low', 'close']:
        df.loc[pre_split, col] = df.loc[pre_split, col] / 3.0
    print(f"\nApplied 3:1 split adjustment to {n_pre} bars (pre {SPLIT_DATE.date()})")

print(f"Adjusted price range: ${df['close'].min():.2f} to ${df['close'].max():.2f}")
print(f"Avg bars/day: {df.groupby(df['time'].dt.date).size().mean():.1f}")

# Save
os.makedirs('data', exist_ok=True)
df.to_csv('data/SCHD_1H_databento.csv', index=False)
print(f"\nSaved to data/SCHD_1H_databento.csv")
