"""Tests for robust timestamp parsing."""

import pandas as pd

from src.time_utils import parse_mixed_timestamp_series


def test_parse_epoch_seconds():
    s = pd.Series([1581712200, 1581715800])  # 2020-02-14 20:30 / 21:30 UTC
    parsed = parse_mixed_timestamp_series(s)
    assert parsed.iloc[0].year == 2020
    assert parsed.iloc[0].month == 2
    assert parsed.iloc[0].day == 14


def test_parse_epoch_milliseconds():
    s = pd.Series([1581712200000, 1581715800000])
    parsed = parse_mixed_timestamp_series(s)
    assert parsed.iloc[0].year == 2020
    assert parsed.iloc[0].minute == 30


def test_parse_datetime_strings():
    s = pd.Series(["2026-02-06 20:30:00", "2026-02-06 21:30:00"])
    parsed = parse_mixed_timestamp_series(s)
    assert parsed.iloc[0].year == 2026
    assert parsed.iloc[1].hour == 21

