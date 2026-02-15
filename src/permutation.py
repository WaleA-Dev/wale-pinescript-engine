"""
Bar-component permutation algorithm for statistical validation.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

OHLC_COLUMNS: Tuple[str, str, str, str] = ("open", "high", "low", "close")


def permute_bars(df: pd.DataFrame, start_index: int = 0) -> pd.DataFrame:
    """
    Permute OHLC bars while preserving bar geometry and return distribution.

    Args:
        df: DataFrame with `open/high/low/close`.
        start_index: Permute from this index onward.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a DataFrame")

    missing = [c for c in OHLC_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLC columns: {missing}")

    if not isinstance(start_index, (int, np.integer)):
        raise TypeError("start_index must be an integer")

    n = len(df)
    if n == 0:
        if int(start_index) != 0:
            raise ValueError("start_index must be 0 when df is empty")
        return df.copy()

    start = int(start_index)
    if start < 0 or start > n:
        raise ValueError(f"start_index must be in [0, {n}]")
    if start == n:
        return df.copy()

    ohlc = df.loc[:, OHLC_COLUMNS].to_numpy(dtype=np.float64, copy=True)
    if np.any(ohlc <= 0) or not np.isfinite(ohlc).all():
        raise ValueError("OHLC values must be finite and strictly positive")

    log_ohlc = np.log(ohlc)
    seg = log_ohlc[start:, :]
    m = seg.shape[0]
    if m <= 1:
        return df.copy()

    rel_high = seg[:, 1] - seg[:, 0]
    rel_low = seg[:, 2] - seg[:, 0]
    rel_close = seg[:, 3] - seg[:, 0]
    gaps = seg[1:, 0] - seg[:-1, 3]

    intrabar_idx = np.random.permutation(m)
    gap_idx = np.random.permutation(m - 1)

    rel_high_p = rel_high[intrabar_idx]
    rel_low_p = rel_low[intrabar_idx]
    rel_close_p = rel_close[intrabar_idx]
    gaps_p = gaps[gap_idx]

    open_log = np.empty(m, dtype=np.float64)
    open_log[0] = seg[0, 0]
    open_log[1:] = open_log[0] + np.cumsum(rel_close_p[:-1] + gaps_p)

    perm_seg = np.empty((m, 4), dtype=np.float64)
    perm_seg[:, 0] = open_log
    perm_seg[:, 1] = open_log + rel_high_p
    perm_seg[:, 2] = open_log + rel_low_p
    perm_seg[:, 3] = open_log + rel_close_p

    # Preserve final close anchor and repair final bar geometry.
    perm_seg[-1, 3] = log_ohlc[-1, 3]
    perm_seg[-1, 1] = max(perm_seg[-1, 1], perm_seg[-1, 0], perm_seg[-1, 3])
    perm_seg[-1, 2] = min(perm_seg[-1, 2], perm_seg[-1, 0], perm_seg[-1, 3])

    out = df.copy()
    out_ohlc = ohlc.copy()
    out_ohlc[start:, :] = np.exp(perm_seg)

    # Preserve global anchors exactly.
    out_ohlc[0, 0] = ohlc[0, 0]
    out_ohlc[-1, 3] = ohlc[-1, 3]

    # Safety clamp for any numerical edge cases.
    out_ohlc[:, 1] = np.maximum(out_ohlc[:, 1], np.maximum(out_ohlc[:, 0], out_ohlc[:, 3]))
    out_ohlc[:, 2] = np.minimum(out_ohlc[:, 2], np.minimum(out_ohlc[:, 0], out_ohlc[:, 3]))

    out.loc[:, OHLC_COLUMNS] = out_ohlc
    return out
