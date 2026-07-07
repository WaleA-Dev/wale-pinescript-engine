"""
Accuracy tests against TradingView's OFFICIAL Pine reference documentation.

Each reference implementation below is a literal bar-by-bar transcription of
the pseudocode published in the Pine Script v6 language reference (pine_ema,
pine_rma, pine_atr, pine_rsi, pine_stdev, ta.stoch formula, ta.crossover
definition), so any divergence in src/pine_ta.py is a genuine accuracy bug.

Broker tests transcribe the broker-emulator rules from the official
"Strategies" documentation: next-bar-open fills, gap fills at open, and the
intrabar path assumption (open closer to high => open->high->low->close).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src import pine_ta as ta
from src.pine_runtime import Broker, BrokerConfig

RNG = np.random.default_rng(123)


def _walk(n=500, with_nan_head=0):
    x = 100 + np.cumsum(RNG.normal(0, 1, n))
    x = np.maximum(x, 1.0)
    if with_nan_head:
        x[:with_nan_head] = np.nan
    return x


def _bars(n=500):
    close = _walk(n)
    open_ = np.concatenate(([close[0]], close[:-1]))
    spread = np.abs(RNG.normal(0, 0.5, n))
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    return open_, high, low, close


def assert_series_equal(actual, expected, atol=1e-9, label=""):
    a, e = np.asarray(actual, float), np.asarray(expected, float)
    assert a.shape == e.shape
    both_nan = np.isnan(a) & np.isnan(e)
    close = np.isclose(a, e, atol=atol, rtol=1e-9)
    ok = both_nan | close
    assert ok.all(), f"{label}: first mismatch at {int(np.argmax(~ok))}: {a[~ok][:3]} vs {e[~ok][:3]}"


# ── Official pseudocode reference implementations ────────────────────────────

def pine_ema_ref(src, length):
    """alpha = 2/(length+1); sum := na(sum[1]) ? src : alpha*src + (1-alpha)*nz(sum[1])"""
    alpha = 2 / (length + 1)
    out = np.full(len(src), np.nan)
    prev = np.nan
    for i, x in enumerate(src):
        if np.isnan(x):
            out[i] = prev  # na source ignored
            continue
        prev = x if np.isnan(prev) else alpha * x + (1 - alpha) * prev
        out[i] = prev
    return out


def pine_sma_ref(src, length):
    out = np.full(len(src), np.nan)
    for i in range(length - 1, len(src)):
        w = src[i - length + 1: i + 1]
        if not np.isnan(w).any():
            out[i] = w.mean()
    return out


def pine_rma_ref(src, length):
    """alpha = 1/length; sum := na(sum[1]) ? ta.sma(src, length) : alpha*src + (1-alpha)*nz(sum[1])"""
    alpha = 1 / length
    sma = pine_sma_ref(src, length)
    out = np.full(len(src), np.nan)
    prev = np.nan
    for i, x in enumerate(src):
        if np.isnan(prev):
            prev = sma[i]
        elif not np.isnan(x):
            prev = alpha * x + (1 - alpha) * prev
        out[i] = prev
    return out


def pine_atr_ref(high, low, close, length):
    """trueRange = na(high[1]) ? high-low : max(high-low, |high-close[1]|, |low-close[1]|); ta.rma(trueRange, length)"""
    n = len(high)
    tr = np.full(n, np.nan)
    for i in range(n):
        if i == 0 or np.isnan(high[i - 1]):
            tr[i] = high[i] - low[i]
        else:
            tr[i] = max(high[i] - low[i],
                        abs(high[i] - close[i - 1]),
                        abs(low[i] - close[i - 1]))
    return pine_rma_ref(tr, length)


def pine_rsi_ref(src, length):
    """u = max(change,0); d = max(-change,0); rs = rma(u)/rma(d); 100 - 100/(1+rs)"""
    n = len(src)
    u = np.full(n, np.nan)
    d = np.full(n, np.nan)
    for i in range(1, n):
        change = src[i] - src[i - 1]
        u[i] = max(change, 0.0)
        d[i] = max(-change, 0.0)
    ru, rd = pine_rma_ref(u, length), pine_rma_ref(d, length)
    out = np.full(n, np.nan)
    for i in range(n):
        if np.isnan(ru[i]) or np.isnan(rd[i]):
            continue
        if rd[i] == 0:
            out[i] = 100.0 if ru[i] > 0 else 50.0
        else:
            out[i] = 100 - 100 / (1 + ru[i] / rd[i])
    return out


def pine_stdev_ref(src, length):
    """population stdev over the window (divide by length)"""
    out = np.full(len(src), np.nan)
    for i in range(length - 1, len(src)):
        w = src[i - length + 1: i + 1]
        if not np.isnan(w).any():
            out[i] = math.sqrt(((w - w.mean()) ** 2).sum() / length)
    return out


# ── ta.* accuracy ─────────────────────────────────────────────────────────────

def test_ema_matches_official_pseudocode():
    src = _walk()
    for length in (3, 9, 21, 50):
        assert_series_equal(ta.ema(src, length), pine_ema_ref(src, length),
                            label=f"ema{length}")


def test_ema_seeds_with_first_source_value_not_sma():
    src = _walk(50)
    out = ta.ema(src, 21)
    assert out[0] == src[0]          # value exists from bar 0
    assert not np.isnan(out[:21]).any()


def test_ema_ignores_nan_head():
    src = _walk(300, with_nan_head=40)
    out = ta.ema(src, 10)
    assert np.isnan(out[:40]).all()
    assert out[40] == src[40]
    assert_series_equal(out, pine_ema_ref(src, 10), label="ema nan head")


def test_rma_matches_official_pseudocode():
    src = _walk()
    for length in (5, 14, 50):
        assert_series_equal(ta.rma(src, length), pine_rma_ref(src, length),
                            label=f"rma{length}")


def test_atr_matches_official_pseudocode():
    o, h, l, c = _bars()
    for length in (7, 14):
        assert_series_equal(ta.atr(h, l, c, length), pine_atr_ref(h, l, c, length),
                            label=f"atr{length}")


def test_rsi_matches_official_pseudocode():
    src = _walk()
    for length in (7, 14):
        assert_series_equal(ta.rsi(src, length), pine_rsi_ref(src, length),
                            atol=1e-8, label=f"rsi{length}")


def test_stdev_is_population_not_sample():
    src = _walk()
    assert_series_equal(ta.stdev(src, 20), pine_stdev_ref(src, 20),
                        atol=1e-8, label="stdev")


def test_macd_consistent_with_doc_ema():
    src = _walk()
    line, sig, hist = ta.macd(src, 12, 26, 9)
    ref_line = pine_ema_ref(src, 12) - pine_ema_ref(src, 26)
    ref_sig = pine_ema_ref(ref_line, 9)
    assert_series_equal(line, ref_line, atol=1e-8, label="macd line")
    assert_series_equal(sig, ref_sig, atol=1e-8, label="macd signal")
    assert_series_equal(hist, ref_line - ref_sig, atol=1e-8, label="macd hist")


def test_bb_returns_basis_upper_lower_order():
    src = _walk()
    basis, upper, lower = ta.bb(src, 20, 2.0)
    ref_basis = pine_sma_ref(src, 20)
    ref_dev = 2.0 * pine_stdev_ref(src, 20)
    assert_series_equal(basis, ref_basis, atol=1e-8, label="bb basis")
    assert_series_equal(upper, ref_basis + ref_dev, atol=1e-8, label="bb upper")
    assert_series_equal(lower, ref_basis - ref_dev, atol=1e-8, label="bb lower")


def test_stoch_formula():
    o, h, l, c = _bars()
    length = 14
    out = ta.stoch(c, h, l, length)
    for i in (50, 200, 400):
        hh = h[i - length + 1: i + 1].max()
        ll = l[i - length + 1: i + 1].min()
        expected = 100 * (c[i] - ll) / (hh - ll)
        assert abs(out[i] - expected) < 1e-9


def test_crossover_official_definition():
    """crossover: source1 > source2 on current bar AND source1 <= source2 on previous bar."""
    a = np.array([1.0, 2.0, 3.0, 2.0, 3.0, 3.0])
    b = np.array([2.5, 2.5, 2.5, 2.5, 2.5, 2.5])
    out = ta.crossover(a, b)
    assert list(out) == [False, False, True, False, True, False]
    out2 = ta.crossunder(a, b)
    assert list(out2) == [False, False, False, True, False, False]


def test_valuewhen_and_barssince():
    cond = np.array([False, True, False, False, True, False])
    src = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    vw = ta.valuewhen(cond, src, 0)
    assert np.isnan(vw[0])
    assert list(vw[1:]) == [20.0, 20.0, 20.0, 50.0, 50.0]
    vw1 = ta.valuewhen(cond, src, 1)
    assert np.isnan(vw1[:4]).all() and vw1[4] == 20.0 and vw1[5] == 20.0
    bs = ta.barssince(cond)
    assert np.isnan(bs[0])
    assert list(bs[1:]) == [0, 1, 2, 0, 1]


def test_highest_lowest_change():
    src = _walk()
    hi = ta.highest(src, 10)
    lo = ta.lowest(src, 10)
    for i in (30, 100):
        assert hi[i] == src[i - 9: i + 1].max()
        assert lo[i] == src[i - 9: i + 1].min()
    ch = ta.change(src, 3)
    assert abs(ch[10] - (src[10] - src[7])) < 1e-12


# ── Broker emulator vs official execution docs ───────────────────────────────

def _mk_broker(**kw):
    cfg = BrokerConfig(initial_capital=kw.pop("capital", 10000.0),
                       qty_type=kw.pop("qty_type", "fixed"),
                       qty_value=kw.pop("qty_value", 10.0),
                       commission_pct=kw.pop("commission", 0.0), **kw)
    return cfg


def test_entry_fills_at_next_bar_open():
    """Docs: 'the next available tick where the emulator fills a generated
    order is at the open of the following bar.'"""
    b = Broker(_mk_broker(), 4)
    bars = [(100, 101, 99, 100.5), (102, 103, 101, 102.5),
            (104, 105, 103, 104.5), (106, 107, 105, 106.5)]
    for i, (o, h, l, c) in enumerate(bars):
        b.begin_bar(i, o, h, l, c)
        if i == 0:
            b.entry("L", 1)          # placed during bar 0
        b.end_bar(i, c)
    assert len(b.trades) == 1
    assert b.trades[0].entry_bar == 1
    assert b.trades[0].entry_price == 102  # open of bar 1


def test_close_fills_at_next_bar_open():
    b = Broker(_mk_broker(), 4)
    bars = [(100, 101, 99, 100), (100, 101, 99, 100),
            (100, 101, 99, 100), (95, 96, 94, 95)]
    for i, (o, h, l, c) in enumerate(bars):
        b.begin_bar(i, o, h, l, c)
        if i == 0:
            b.entry("L", 1)
        if i == 2:
            b.close("L", comment="X")
        b.end_bar(i, c)
    t = b.trades[0]
    assert t.exit_bar == 3 and t.exit_price == 95  # open of bar 3


def test_stop_fills_at_stop_price_intrabar():
    b = Broker(_mk_broker(), 4)
    #                 o    h    l    c
    bars = [(100, 101, 99, 100), (100, 101, 99, 100),
            (100, 101, 95, 96), (95, 96, 94, 95)]
    for i, (o, h, l, c) in enumerate(bars):
        b.begin_bar(i, o, h, l, c)
        if i == 0:
            b.entry("L", 1)
        if b.position_size > 0:
            b.exit("SL", stop=97.0)
        b.end_bar(i, c)
    t = b.trades[0]
    assert t.exit_bar == 2 and t.exit_price == 97.0  # exact stop level


def test_gap_through_stop_fills_at_open():
    """Docs: 'if the market price crosses an order's price during the gap
    between two bars, the emulator fills the order at the current bar's open.'"""
    b = Broker(_mk_broker(), 3)
    bars = [(100, 101, 99, 100), (100, 101, 99, 100), (90, 92, 89, 91)]
    for i, (o, h, l, c) in enumerate(bars):
        b.begin_bar(i, o, h, l, c)
        if i == 0:
            b.entry("L", 1)
        if b.position_size > 0:
            b.exit("SL", stop=97.0)
        b.end_bar(i, c)
    t = b.trades[0]
    assert t.exit_bar == 2 and t.exit_price == 90  # gap: filled at open, not 97


def test_both_hit_open_closer_to_high_fills_limit_for_long():
    """Open closer to high => path open->high->low->close => limit (above) first."""
    b = Broker(_mk_broker(), 3)
    # bar 2: o=100 h=101 l=90 c=95 -> open is closer to high
    bars = [(100, 101, 99, 100), (100, 101, 99, 100), (100, 101, 90, 95)]
    for i, (o, h, l, c) in enumerate(bars):
        b.begin_bar(i, o, h, l, c)
        if i == 0:
            b.entry("L", 1)
        if b.position_size > 0:
            b.exit("X", stop=94.0, limit=100.5)
        b.end_bar(i, c)
    t = b.trades[0]
    assert t.exit_price == 100.5 and "limit" in t.exit_reason


def test_both_hit_open_closer_to_low_fills_stop_for_long():
    """Open closer to low => path open->low->high->close => stop (below) first."""
    b = Broker(_mk_broker(), 3)
    # bar 2: o=95 h=110 l=94 c=105 -> open closer to low
    bars = [(100, 101, 99, 100), (95, 101, 94, 100), (95, 110, 94, 105)]
    for i, (o, h, l, c) in enumerate(bars):
        b.begin_bar(i, o, h, l, c)
        if i == 0:
            b.entry("L", 1)
        if b.position_size > 0:
            b.exit("X", stop=94.5, limit=105.0)
        b.end_bar(i, c)
    t = b.trades[0]
    assert t.exit_price == 94.5 and "stop" in t.exit_reason


def test_percent_of_equity_sizing_and_commission():
    cfg = BrokerConfig(initial_capital=10000.0, qty_type="percent_of_equity",
                       qty_value=50.0, commission_pct=1.0)
    b = Broker(cfg, 3)
    bars = [(100, 101, 99, 100), (100, 101, 99, 100), (110, 111, 109, 110)]
    for i, (o, h, l, c) in enumerate(bars):
        b.begin_bar(i, o, h, l, c)
        if i == 0:
            b.entry("L", 1)
        if i == 1:
            b.close("L")
        b.end_bar(i, c)
    t = b.trades[0]
    assert t.qty == 50.0  # floor(10000*0.5 / 100)
    # pnl = 50*(110-100) - 1% of both fills = 500 - (50*100*.01 + 50*110*.01)
    assert abs(t.pnl - (500 - 50 - 55)) < 1e-9


def test_reversal_closes_then_flips():
    b = Broker(_mk_broker(), 4)
    bars = [(100, 101, 99, 100)] * 4
    for i, (o, h, l, c) in enumerate(bars):
        b.begin_bar(i, o, h, l, c)
        if i == 0:
            b.entry("L", 1)
        if i == 2:
            b.entry("S", -1)
        b.end_bar(i, c)
    assert len(b.trades) == 2
    assert b.trades[0].exit_reason == "Reverse"
    assert b.trades[1].direction == "SHORT"
    assert b.position_size < 0
