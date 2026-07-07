# Saty Phase Strategy — AAPL 1D Validation vs TradingView (2026-07-04)

End-to-end validation of the Pine→Python translator against a real TradingView
strategy tester run. Strategy: user's "Saty Phase Strategy" (oscillator entries,
overbought + bearish-divergence exits via `ta.pivothigh`). Data: AAPL 1D.

## Result

Every TradingView trade in the overlapping data window (Alpaca IEX history for
AAPL starts 2020-07-27) is reproduced by the engine, with **one** exit flipped
by feed differences — not by logic:

| TV # | TradingView                      | Engine (Alpaca IEX, splits+div adj)  |
|------|----------------------------------|--------------------------------------|
| 47   | 03/02/21 @124.96 → 07/19/21 @140.12 | 03/02/21 @124.90 → 07/19/21 @140.04 ✅ |
| 48   | 01/31/22 @166.35 → 02/15/22 @167.36 | 01/31/22 @166.34 → merged, see below ⚠️ |
| 49   | 03/16/22 @153.73 → 08/17/22 @169.61 | (merged) → 08/19/22 @169.98 ⚠️        |
| 50   | 09/08/22 @151.81 → 04/25/23 @162.68 | 09/08/22 @151.76 → 04/25/23 @162.63 ✅ |
| 51   | 08/23/23 @176.29 → 01/30/24 @188.80 | 08/23/23 @176.32 → 01/30/24 @188.73 ✅ |
| 52   | 03/11/24 @171.22 → 06/21/24 @208.58 | 03/11/24 @171.26 → 06/21/24 @208.58 ✅ |
| 53   | 01/16/25 @235.84 → 09/11/25 @226.23 | 01/16/25 @235.91 → 09/11/25 @226.13 ✅ |
| 54   | 01/14/26 @259.01 → open (marked @308.63) | 01/14/26 @258.94 → open ✅       |

Prices agree to within pennies (IEX open vs consolidated open).

## The one flipped exit, dissected

TV exited trade 48 on 2022-02-15 via the **bearish divergence** branch.
At the 2022-02-14 bar both feeds confirm the same two pivots (center 02-09,
`pivothigh(x, 3, 3)` confirms 3 bars later):

- price pivot 172.91 > previous stored pivot 172.25 → higher high ✅ (both feeds)
- oscillator pivot (ours): **32.17 vs previous 31.89** → higher → no divergence
- oscillator pivot (TV): fractionally **below** its previous → divergence → exit

A 0.28-point oscillator difference — caused by IEX bars vs TV's consolidated
tape flowing through `ta.atr(14)` — flips the comparison. Once TV re-entered
on 03/16 and we held, both trade lists realign from 09/08/22 onward.

## Lessons encoded in the engine

1. **Dividend adjustment matters.** The user's TV chart had the "adj" dividend
   toggle ON. With Alpaca `adjustment=split` (TV's *default*), entry dates were
   still 14/16 exact but prices ran ~2.7% high in 2021 decaying to ~0% in 2026 —
   the signature of dividend adjustment. The Fetch panel now has an Adjustment
   selector; match it to your TV chart.
2. **TV sizing defaults.** `strategy()` without `default_qty_type` means
   `strategy.fixed`, qty `1`, initial capital `1,000,000` (Pine v6 reference).
   The translator now emits exactly that.
3. **For exact 1:1 trade lists, use identical bars.** Export the TV chart data
   to CSV and load it via "Upload CSV" — threshold-crossing signals (oscillator
   crossing ±100, pivot comparisons) are legitimately sensitive to sub-percent
   OHLC differences between feeds. This is data sensitivity, not translation
   error: the same borderline bars flip in either direction depending on feed.

## Follow-up validations (same script, no code changes)

- **SPY 1D (2026-07-05)**: 7/8 overlapping trades date-exact on split-adjusted
  data; dividend-adjusted prices match TV to pennies. Two diffs dissected:
  oscillator grazed -62.72 vs the -61.8 trigger (Aug 2023) and 100.55 vs the
  100 threshold (Jul 2025) — sub-point feed noise.
- **GOOGL 1D (2026-07-06)**: 6/7 trades bar-identical on BOTH legs
  (12/12 legs, Oct 2021 → Apr 2026). The single diff is the earliest trade,
  adjacent to the IEX data-window start (TV history: 2004; IEX: ~2018) where
  warm-up plus the COVID-crash oscillator extremes amplify feed differences.
- In-app verification: Trades tab → "Compare to TV export" diffs any backtest
  against TradingView's own List-of-trades CSV.
