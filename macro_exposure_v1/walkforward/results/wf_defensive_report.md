# Macro-Exposure v1 — Balanced + Defensive Overlay Walk-Forward OOS Validation

**Same fixed Balanced chromosome** as `wf_report.md`. The ONLY change is a
defensive overlay applied to long allocations:

```
defensive = (SPY < 200-day MA) OR (VIX 20d percentile > 0.8)
when defensive AND target > 0:   target *= 0.5
shorts and flats untouched
```

**Column mapping** (no `spy_above_200dma` or `vix_pct` column exists verbatim
in the feature panel; mapped to canonical equivalents):

| Brief column        | Repo column                              | Notes |
|---------------------|------------------------------------------|-------|
| `spy_above_200dma`  | `spy_above_200dma_sign > 0` from `spy_dma200_dist` | 1 when SPY above 200-day SMA |
| `vix_pct`           | `vix_pct_20d` (recovered from `vix_pct_20d_centered`) | 20-day rolling VIX percentile (0..1). 20d is a short window, so the >0.8 trigger is hit often (~37% of days). |

**Overlay coverage**: defensive flag is true 37.4% of days; longs were scaled
on 494 of 2528 long-days across the OOT span (i.e. ~19.5% of long-days
de-risked).

---

## Pooled OOS metrics

| Metric            | Original Balanced | Defensive Overlay | Delta     |
|-------------------|-------------------|-------------------|-----------|
| Pooled CAGR       | **21.54%**        | **16.46%**        | -5.08 pp  |
| Pooled max DD     | 12.38%            | 12.33%            | -0.05 pp  |
| Pooled Sortino    | 1.55              | 1.43              | -0.12     |
| Pooled Sharpe     | 1.43              | 1.38              | -0.05     |
| Worst month       | -8.83%            | -6.94%            | +1.89 pp  |
| Hit rate (folds)  | 100.0%            | 90.9%             | -9.1 pp   |

CAGR gives up ~5pp in exchange for a meaningfully shallower worst-month tail
and bear-regime Sortino clearing the gate.

---

## Per-regime stratification

| Regime | n  | Orig CAGR | Def CAGR | Orig DD | Def DD | Orig Sortino | **Def Sortino** | Hit rate (def) |
|--------|----|-----------|----------|---------|--------|--------------|------------------|----------------|
| bull   | 15 | +28.63%   | +22.24%  | 8.07%   | 7.20%  | 2.16         | **2.01**         | 100.0%         |
| bear   | 3  | +4.90%    | +5.75%   | 9.43%   | 6.60%  | 0.39         | **0.50**         | 66.7%          |
| chop   | 4  | +20.04%   | +11.74%  | 6.55%   | 6.58%  | 1.33         | **1.01**         | 75.0%          |

Bear regime: CAGR slightly improved (4.9% -> 5.7%), DD cut from 9.4% to 6.6%,
Sortino moved from 0.39 to **0.504** — clears the 0.50 gate by 0.004.
Bull regime cost: ~6pp CAGR per year (still strong). Chop cost: ~8pp CAGR
and one previously +tiny fold turned slightly negative.

## Per-fold table (defensive)

| fold | OOT start  | OOT end    | regime | SPY CAGR | fold CAGR | fold DD | fold Sortino |
|---|---|---|---|---|---|---|---|
| 0  | 2015-01-01 | 2015-12-31 | chop  | +1.3%   | +2.51%  | 5.63%  | 0.30 |
| 1  | 2015-07-01 | 2016-06-30 | chop  | +3.1%   | -0.41%  | 8.08%  | -0.01|
| 2  | 2016-01-01 | 2016-12-31 | bull  | +13.5%  | +7.59%  | 8.09%  | 0.67 |
| 3  | 2016-07-01 | 2017-06-30 | bull  | +17.5%  | +19.07% | 5.43%  | 2.17 |
| 4  | 2017-01-01 | 2017-12-31 | bull  | +20.9%  | +22.54% | 3.74%  | 2.82 |
| 5  | 2017-07-01 | 2018-06-30 | bull  | +14.2%  | +30.06% | 9.28%  | 2.58 |
| 6  | 2018-01-01 | 2018-12-31 | bear  | -5.3%   | +13.26% | 9.28%  | 1.09 |
| 7  | 2018-07-01 | 2019-06-30 | chop  | +10.0%  | +17.62% | 2.90%  | 1.97 |
| 8  | 2019-01-01 | 2019-12-31 | bull  | +31.1%  | +23.17% | 9.70%  | 1.92 |
| 9  | 2019-07-01 | 2020-06-30 | chop  | +6.4%   | +27.24% | 9.70%  | 1.80 |
| 10 | 2020-01-01 | 2020-12-31 | bull  | +17.2%  | +46.89% | 6.61%  | 2.89 |
| 11 | 2020-07-01 | 2021-06-30 | bull  | +39.9%  | +24.32% | 4.41%  | 2.13 |
| 12 | 2021-01-01 | 2021-12-31 | bull  | +30.5%  | +8.31%  | 2.14%  | 1.76 |
| 13 | 2021-07-01 | 2022-06-30 | bear  | -11.1%  | +4.13%  | 4.90%  | 0.39 |
| 14 | 2022-01-01 | 2022-12-31 | bear  | -18.7%  | -0.16%  | 5.62%  | 0.03 |
| 15 | 2022-07-01 | 2023-06-30 | bull  | +18.3%  | +28.91% | 8.25%  | 2.91 |
| 16 | 2023-01-01 | 2023-12-31 | bull  | +26.8%  | +34.41% | 8.25%  | 3.11 |
| 17 | 2023-07-01 | 2024-06-30 | bull  | +24.5%  | +13.34% | 6.59%  | 1.41 |
| 18 | 2024-01-01 | 2024-12-31 | bull  | +25.6%  | +16.58% | 6.36%  | 1.12 |
| 19 | 2024-07-01 | 2025-06-30 | bull  | +14.7%  | +15.73% | 11.24% | 1.12 |
| 20 | 2025-01-01 | 2025-12-31 | bull  | +17.9%  | +13.81% | 9.17%  | 1.14 |
| 21 | 2025-07-01 | 2026-06-04 | bull  | +25.4%  | +28.84% | 8.79%  | 2.36 |

---

## Gate audit

| Gate                              | Threshold | Result | Pass? |
|-----------------------------------|-----------|--------|-------|
| Pooled OOS CAGR                   | >= 10%    | 16.46% | YES   |
| Pooled OOS max DD                 | <= 18%    | 12.33% | YES   |
| All-regime avg Sortino            | >= 0.50   | min 0.504 (bear) | YES (margin 0.004) |
| Fold hit rate (positive CAGR)     | >= 65%    | 90.9%  | YES   |

---

## Verdict

**PASS** — all four walk-forward gates satisfied.

**READY FOR PAPER-TRADE per HC #534 R1.**

### Caveats / honest assessment

1. **Bear-regime margin is razor-thin.** Sortino is 0.504 vs the 0.500 cutoff.
   A single bad bear fold added to the panel could flip this. Recommend
   tracking bear-regime Sortino monthly in paper-trade.
2. **CAGR cost is real (~5pp pooled, ~6pp in bull regime).** The overlay is
   defensive, not free.
3. **20-day VIX percentile is a short lookback.** A longer window (252d) might
   trigger less often and preserve more bull-regime upside while still helping
   in genuine stress. Worth exploring as overlay-v2 if paper-trade shows
   excessive defensive scaling in healthy bull markets.
4. **Hit rate dropped from 100% to 90.9%** — two folds turned slightly
   negative (one chop fold went from +0.something% to -0.41%, fold 14 bear
   went from a tiny positive to -0.16%). Both losses are <0.5pp so this is
   acceptable; the headline pass is intact.

### Suggested next steps

- Move to paper-trade with the defensive-overlay chromosome (Balanced + this overlay).
- In parallel, prototype overlay-v2 using **252-day** VIX percentile and the
  same 200-DMA test, to see if we can recover some of the bull-regime CAGR.
- Add a deeper "flat-flatten in severe stress" tier (e.g. when VIX >= 0.95
  AND SPY < 200-DMA, scale longs to 0) and re-check whether it pushes bear
  Sortino comfortably above 0.6 without further bull-regime CAGR loss.
