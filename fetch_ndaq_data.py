#!/usr/bin/env python3
"""
Fetch NDAQ 1H data from Databento and align to TradingView's bar structure.

Pipeline:
1. Fetch 1-minute data from Databento (XNAS.ITCH)
2. Apply 3-for-1 stock split adjustment (pre Aug 29, 2022)
3. Re-aggregate to 1H bars at :30 alignment (matching TradingView)
4. Filter to session hours 12:30-21:30 UTC (matching TV's chart session)
5. Save as CSV for backtesting

Usage:
    python fetch_ndaq_data.py
    python fetch_ndaq_data.py --api-key YOUR_KEY --start 2019 --end 2026
"""
import sys
import os
import argparse
import databento as db
import pandas as pd
from pathlib import Path


# NDAQ 3-for-1 stock split effective date
SPLIT_DATE = pd.Timestamp('2022-08-29')
SPLIT_RATIO = 3.0

# TradingView session hours (UTC): pre-market through market close
SESSION_START_HOUR = 12  # 12:30 UTC = ~7:30 AM ET
SESSION_END_HOUR = 21    # 21:30 UTC = ~4:30 PM ET


def fetch_minute_data(client, symbol, start, end):
    """Fetch 1-minute OHLCV from Databento."""
    data = client.timeseries.get_range(
        dataset='XNAS.ITCH',
        symbols=[symbol],
        schema='ohlcv-1m',
        start=start,
        end=end,
    )
    df = data.to_df().reset_index()
    if df.empty:
        return pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume'])

    if 'ts_event' in df.columns:
        df['time'] = pd.to_datetime(df['ts_event'], unit='ns', utc=True).dt.tz_localize(None)
    elif 'index' in df.columns:
        df['time'] = pd.to_datetime(df['index'])

    return df[['time', 'open', 'high', 'low', 'close', 'volume']].sort_values('time').reset_index(drop=True)


def apply_split_adjustment(df, split_date=SPLIT_DATE, ratio=SPLIT_RATIO):
    """Divide pre-split prices by split ratio to match TradingView's adjusted data."""
    pre_split = df['time'] < split_date
    n = pre_split.sum()
    if n > 0:
        for col in ['open', 'high', 'low', 'close']:
            df.loc[pre_split, col] = df.loc[pre_split, col] / ratio
        print(f"  Split-adjusted {n} bars (pre {split_date.date()}, ratio 1:{int(ratio)})")
    return df


def aggregate_to_1h_aligned(df):
    """Aggregate minute bars to 1H bars starting at :30 past the hour (matching TradingView)."""
    df['bar_time'] = (df['time'] - pd.Timedelta(minutes=30)).dt.floor('h') + pd.Timedelta(minutes=30)
    agg = df.groupby('bar_time').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }).reset_index().rename(columns={'bar_time': 'time'})
    return agg.sort_values('time').reset_index(drop=True)


def filter_session(df, start_hour=SESSION_START_HOUR, end_hour=SESSION_END_HOUR):
    """Filter to TradingView's chart session hours."""
    hour = df['time'].dt.hour
    return df[(hour >= start_hour) & (hour <= end_hour)].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description='Fetch NDAQ data from Databento')
    parser.add_argument('--api-key', default=None, help='Databento API key (or set DATABENTO_API_KEY env var)')
    parser.add_argument('--symbol', default='NDAQ')
    parser.add_argument('--start', type=int, default=2019, help='Start year')
    parser.add_argument('--end', type=int, default=2026, help='End year')
    parser.add_argument('--output', default='data/NDAQ_1H_TV_aligned.csv')
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get('DATABENTO_API_KEY', '')
    if not api_key:
        print('Error: Provide --api-key or set DATABENTO_API_KEY env var')
        return 1
    client = db.Historical(api_key)
    Path('data').mkdir(exist_ok=True)

    # Fetch 1-minute data in yearly chunks
    all_frames = []
    for year in range(args.start, args.end + 1):
        start = f'{year}-01-01'
        end = f'{year + 1}-01-01' if year < args.end else '2026-02-07'
        print(f'Fetching 1m {args.symbol} {year}...')
        try:
            df = fetch_minute_data(client, args.symbol, start, end)
            if not df.empty:
                print(f'  Got {len(df)} bars')
                all_frames.append(df)
            else:
                print(f'  No data')
        except Exception as e:
            print(f'  Error: {e}')

    if not all_frames:
        print('No data fetched!')
        return 1

    combined = pd.concat(all_frames, ignore_index=True).sort_values('time').reset_index(drop=True)
    print(f'\nTotal 1-minute bars: {len(combined)}')

    # Apply split adjustment
    combined = apply_split_adjustment(combined)

    # Aggregate to 1H :30-aligned bars
    hourly = aggregate_to_1h_aligned(combined)
    print(f'Aggregated to {len(hourly)} 1H bars')

    # Filter to session hours
    session = filter_session(hourly)
    print(f'Session-filtered: {len(session)} bars ({SESSION_START_HOUR}:30-{SESSION_END_HOUR}:30 UTC)')

    print(f'\nDate range: {session["time"].min()} to {session["time"].max()}')
    print(f'Price range: ${session["close"].min():.2f} to ${session["close"].max():.2f}')
    print(f'Avg bars/day: {session.groupby(session["time"].dt.date).size().mean():.1f}')

    # Save
    session.to_csv(args.output, index=False)
    print(f'\nSaved to {args.output}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
