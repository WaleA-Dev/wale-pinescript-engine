"""
Timestamp parsing helpers.

These utilities normalize mixed timestamp inputs from CSV files:
- epoch seconds / milliseconds / microseconds / nanoseconds
- datetime strings
- timezone-aware timestamps

All outputs are naive UTC datetimes for consistent engine behavior.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


EPOCH_UNITS = ("s", "ms", "us", "ns")


def _score_epoch_parse(parsed: pd.Series) -> float:
    """Score a parsed datetime series; higher is better."""
    valid = parsed.notna()
    if not valid.any():
        return -1.0

    years = parsed.loc[valid].dt.year
    in_range_ratio = ((years >= 1980) & (years <= 2100)).mean()
    valid_ratio = valid.mean()

    score = (in_range_ratio * 100.0) + (valid_ratio * 10.0)

    # Prefer candidates with temporal span (not a single repeated timestamp).
    span = parsed.loc[valid].max() - parsed.loc[valid].min()
    if span.total_seconds() > 0:
        score += 1.0

    return score


def parse_numeric_epoch_series(numeric_series: pd.Series) -> pd.Series:
    """
    Parse numeric epoch-like values with automatic unit detection.

    Tries s/ms/us/ns and chooses the most plausible interpretation.
    Returns tz-aware UTC timestamps.
    """
    best: Optional[pd.Series] = None
    best_score = -1.0

    for unit in EPOCH_UNITS:
        candidate = pd.to_datetime(numeric_series, unit=unit, errors="coerce", utc=True)
        score = _score_epoch_parse(candidate)
        if score > best_score:
            best = candidate
            best_score = score

    if best is None:
        return pd.to_datetime(numeric_series, errors="coerce", utc=True)
    return best


def parse_mixed_timestamp_series(series: pd.Series) -> pd.Series:
    """
    Parse mixed timestamp columns robustly.

    For mostly numeric series, epoch parsing is preferred.
    For mostly textual series, datetime parsing is preferred.
    Any NaT gaps are backfilled from the alternate parser.
    Returns naive UTC datetimes.
    """
    raw = series.copy()

    numeric = pd.to_numeric(raw, errors="coerce")
    numeric_ratio = float(numeric.notna().mean()) if len(raw) else 0.0

    parsed_numeric = parse_numeric_epoch_series(numeric)
    parsed_text = pd.to_datetime(raw, errors="coerce", utc=True)

    if numeric_ratio >= 0.8:
        parsed = parsed_numeric.where(parsed_numeric.notna(), parsed_text)
    else:
        parsed = parsed_text.where(parsed_text.notna(), parsed_numeric)

    # Return naive UTC for compatibility with existing engine code.
    return parsed.dt.tz_convert("UTC").dt.tz_localize(None)

