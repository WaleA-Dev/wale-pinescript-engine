"""
Vectorized Pine `ta.*` / `math.*` function library.

All functions take/return numpy float arrays and mirror TradingView semantics
(warm-up bars are NaN). Generated strategies import this as `ta`.
"""

from __future__ import annotations

import numpy as np

from src.indicators import (  # TV-exact implementations
    sma as _sma, rma as _rma, tr as _tr, atr as _atr,
    rsi as _rsi, bollinger_bands as _bb,
    highest as _highest, lowest as _lowest,
    crossover as _crossover, crossunder as _crossunder,
)


def _arr(x, n: int | None = None) -> np.ndarray:
    """Coerce scalars/series to float ndarray (broadcast scalars to length n)."""
    if np.isscalar(x) or x is None:
        if n is None:
            raise ValueError("Scalar passed where series expected")
        return np.full(n, np.nan if x is None else float(x))
    return np.asarray(x, dtype=float)


def _int(x) -> int:
    return int(np.round(float(x)))


def shift(series, k: int = 1) -> np.ndarray:
    """Pine history access: series[k]."""
    a = np.asarray(series, dtype=float)
    k = _int(k)
    if k == 0:
        return a
    out = np.full_like(a, np.nan)
    if k > 0:
        out[k:] = a[:-k]
    else:
        out[:k] = a[-k:]
    return out


# ── Moving averages ──────────────────────────────────────────────────────────

def sma(series, length): return _sma(_arr(series), _int(length))
def rma(series, length): return _rma(_arr(series), _int(length))


def ema(series, length):
    """Doc-exact ta.ema: seeded with the first non-na source value.

    Official reference pseudocode:
        alpha = 2 / (length + 1)
        sum := na(sum[1]) ? src : alpha * src + (1 - alpha) * nz(sum[1])
    (na source values are ignored — the previous value carries forward.)
    """
    a = _arr(series)
    alpha = 2.0 / (_int(length) + 1)
    out = np.full_like(a, np.nan)
    prev = np.nan
    for i in range(len(a)):
        x = a[i]
        if np.isnan(x):
            out[i] = prev
            continue
        prev = x if np.isnan(prev) else alpha * x + (1.0 - alpha) * prev
        out[i] = prev
    return out


def wma(series, length):
    a = _arr(series)
    n = _int(length)
    w = np.arange(1, n + 1, dtype=float)
    out = np.full_like(a, np.nan)
    if len(a) >= n:
        windows = np.lib.stride_tricks.sliding_window_view(a, n)
        out[n - 1:] = windows @ w / w.sum()
    return out


def hma(series, length):
    n = _int(length)
    half = max(1, n // 2)
    sq = max(1, int(np.sqrt(n)))
    return wma(2 * wma(series, half) - wma(series, n), sq)


def vwma(series, volume, length):
    a, v = _arr(series), _arr(volume)
    n = _int(length)
    pv = a * v
    return _sma(pv, n) / np.where(_sma(v, n) == 0, np.nan, _sma(v, n))


def swma(series):
    a = _arr(series)
    w = np.array([1, 2, 2, 1], dtype=float) / 6.0
    out = np.full_like(a, np.nan)
    if len(a) >= 4:
        windows = np.lib.stride_tricks.sliding_window_view(a, 4)
        out[3:] = windows @ w[::-1]
    return out


def alma(series, length, offset=0.85, sigma=6.0):
    a = _arr(series)
    n = _int(length)
    m = offset * (n - 1)
    s = n / sigma
    w = np.exp(-((np.arange(n) - m) ** 2) / (2 * s * s))
    w /= w.sum()
    out = np.full_like(a, np.nan)
    if len(a) >= n:
        windows = np.lib.stride_tricks.sliding_window_view(a, n)
        out[n - 1:] = windows @ w
    return out


# ── Volatility / range ───────────────────────────────────────────────────────

def tr(high, low, close, handle_na: bool = True):
    return _tr(_arr(high), _arr(low), _arr(close))


def atr(high, low, close, length):
    return _atr(_arr(high), _arr(low), _arr(close), _int(length))


def stdev(series, length):
    a = _arr(series)
    n = _int(length)
    out = np.full_like(a, np.nan)
    if len(a) >= n:
        windows = np.lib.stride_tricks.sliding_window_view(a, n)
        out[n - 1:] = windows.std(axis=1)  # population stdev, matches Pine
    return out


def variance(series, length):
    s = stdev(series, length)
    return s * s


def dev(series, length):
    a = _arr(series)
    n = _int(length)
    out = np.full_like(a, np.nan)
    if len(a) >= n:
        windows = np.lib.stride_tricks.sliding_window_view(a, n)
        means = windows.mean(axis=1, keepdims=True)
        out[n - 1:] = np.abs(windows - means).mean(axis=1)
    return out


# ── Oscillators ──────────────────────────────────────────────────────────────

def rsi(series, length): return _rsi(_arr(series), _int(length))


def macd(series, fast=12, slow=26, signal=9):
    """ta.macd built on the doc-exact (src-seeded) ema."""
    line = ema(series, fast) - ema(series, slow)
    sig = ema(line, signal)
    return line, sig, line - sig


def bb(series, length, mult):
    """ta.bb returns [basis, upper, lower] like Pine."""
    upper, middle, lower = _bb(_arr(series), _int(length), float(mult))
    return middle, upper, lower


def stoch(close, high, low, length):
    """Pine ta.stoch(source, high, low, length) -> raw %K."""
    c, h, l = _arr(close), _arr(high), _arr(low)
    n = _int(length)
    hh = _highest(h, n)
    ll = _lowest(l, n)
    rng = hh - ll
    return np.where(rng == 0, 0.0, 100.0 * (c - ll) / np.where(rng == 0, np.nan, rng))


def cci(source, length):
    """Pine ta.cci(source, length) — TV passes the price source directly
    (typically hlc3); it is NOT derived from high/low/close here."""
    tp = _arr(source)
    ma = _sma(tp, _int(length))
    md = dev(tp, _int(length))
    return (tp - ma) / (0.015 * np.where(md == 0, np.nan, md))


def mfi(high, low, close, volume, length):
    tp = (_arr(high) + _arr(low) + _arr(close)) / 3.0
    v = _arr(volume)
    n = _int(length)
    delta = np.diff(tp, prepend=np.nan)
    pos = np.where(delta > 0, tp * v, 0.0)
    neg = np.where(delta < 0, tp * v, 0.0)
    pos_sum = _rolling_sum(pos, n)
    neg_sum = _rolling_sum(neg, n)
    ratio = pos_sum / np.where(neg_sum == 0, np.nan, neg_sum)
    return 100.0 - 100.0 / (1.0 + ratio)


def wpr(high, low, close, length):
    h, l, c = _arr(high), _arr(low), _arr(close)
    n = _int(length)
    hh = _highest(h, n)
    ll = _lowest(l, n)
    rng = hh - ll
    return -100.0 * (hh - c) / np.where(rng == 0, np.nan, rng)


def obv(close, volume):
    c, v = _arr(close), _arr(volume)
    sign = np.sign(np.diff(c, prepend=c[0]))
    return np.cumsum(sign * v)


def mom(series, length):
    return _arr(series) - shift(series, _int(length))


def roc(series, length):
    prev = shift(series, _int(length))
    return 100.0 * (_arr(series) - prev) / np.where(prev == 0, np.nan, prev)


def change(series, length=1):
    return _arr(series) - shift(series, _int(length))


# ── Extremes / windows ───────────────────────────────────────────────────────

def highest(series, length): return _highest(_arr(series), _int(length))
def lowest(series, length): return _lowest(_arr(series), _int(length))


def _rolling_sum(a: np.ndarray, n: int) -> np.ndarray:
    out = np.full_like(a, np.nan)
    if len(a) >= n:
        c = np.cumsum(np.nan_to_num(a))
        out[n - 1:] = c[n - 1:] - np.concatenate(([0.0], c[:-n]))
    return out


def sum(series, length):  # noqa: A001 — Pine name
    return _rolling_sum(_arr(series), _int(length))


def cum(series):
    return np.cumsum(np.nan_to_num(_arr(series)))


def avg(*args):
    stacked = np.vstack([_arr(a, n=len(_arr(args[0]))) if np.isscalar(a) else _arr(a)
                         for a in args])
    return stacked.mean(axis=0)


# ── Crosses / direction ──────────────────────────────────────────────────────

def _pair(a, b):
    if np.isscalar(a) and not np.isscalar(b):
        a = np.full(len(np.asarray(b)), float(a))
    if np.isscalar(b) and not np.isscalar(a):
        b = np.full(len(np.asarray(a)), float(b))
    return _arr(a), _arr(b)


def crossover(a, b):
    a, b = _pair(a, b)
    return _crossover(a, b).astype(bool)


def crossunder(a, b):
    a, b = _pair(a, b)
    return _crossunder(a, b).astype(bool)


def cross(a, b):
    return crossover(a, b) | crossunder(a, b)


def rising(series, length):
    a = _arr(series)
    n = _int(length)
    ok = np.ones(len(a), dtype=bool)
    for k in range(1, n + 1):
        ok &= shift(a, k - 1) > shift(a, k)
    ok[:n] = False
    return ok


def falling(series, length):
    a = _arr(series)
    n = _int(length)
    ok = np.ones(len(a), dtype=bool)
    for k in range(1, n + 1):
        ok &= shift(a, k - 1) < shift(a, k)
    ok[:n] = False
    return ok


# ── Event helpers ────────────────────────────────────────────────────────────

def barssince(condition):
    cond = np.asarray(condition, dtype=bool)
    out = np.full(len(cond), np.nan)
    last = -1
    for i in range(len(cond)):
        if cond[i]:
            last = i
        if last >= 0:
            out[i] = i - last
    return out


def valuewhen(condition, source, occurrence=0):
    cond = np.asarray(condition, dtype=bool)
    src = _arr(source)
    occ = _int(occurrence)
    out = np.full(len(cond), np.nan)
    hits: list[float] = []
    for i in range(len(cond)):
        if cond[i]:
            hits.append(src[i])
        if len(hits) > occ:
            out[i] = hits[-1 - occ]
    return out


def pivothigh(source, leftbars, rightbars):
    src = _arr(source)
    lb, rb = _int(leftbars), _int(rightbars)
    n = len(src)
    out = np.full(n, np.nan)
    for i in range(lb, n - rb):
        window = src[i - lb: i + rb + 1]
        if np.isnan(window).any():
            continue
        if src[i] == window.max() and (window == src[i]).sum() == 1:
            out[i + rb] = src[i]  # confirmed rb bars later, like Pine
    return out


def pivotlow(source, leftbars, rightbars):
    src = _arr(source)
    lb, rb = _int(leftbars), _int(rightbars)
    n = len(src)
    out = np.full(n, np.nan)
    for i in range(lb, n - rb):
        window = src[i - lb: i + rb + 1]
        if np.isnan(window).any():
            continue
        if src[i] == window.min() and (window == src[i]).sum() == 1:
            out[i + rb] = src[i]
    return out


# ── Supertrend ───────────────────────────────────────────────────────────────

def supertrend(high, low, close, factor, atr_period):
    h, l, c = _arr(high), _arr(low), _arr(close)
    a = atr(h, l, c, atr_period)
    hl2 = (h + l) / 2.0
    upper = hl2 + float(factor) * a
    lower = hl2 - float(factor) * a
    n = len(c)
    st = np.full(n, np.nan)
    direction = np.full(n, 1.0)
    fu, fl = upper.copy(), lower.copy()
    for i in range(1, n):
        if np.isnan(a[i]):
            continue
        fl[i] = lower[i] if (lower[i] > fl[i - 1] or c[i - 1] < fl[i - 1]) else fl[i - 1]
        fu[i] = upper[i] if (upper[i] < fu[i - 1] or c[i - 1] > fu[i - 1]) else fu[i - 1]
        if np.isnan(st[i - 1]) or st[i - 1] == fu[i - 1]:
            direction[i] = -1 if c[i] > fu[i] else 1
        else:
            direction[i] = 1 if c[i] < fl[i] else -1
        st[i] = fl[i] if direction[i] == -1 else fu[i]
    return st, direction
