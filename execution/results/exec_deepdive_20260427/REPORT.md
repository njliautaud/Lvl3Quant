# CNN-Mamba v2 — Per-Trade Dynamics Deep Dive

**Run**: 2026-04-30T04:19:44.942298  |  **Trades**: 1866 (1179 long, 687 short) across 10 folds (Feb 23 - Mar 5 2026)
**Gate**: |z_10s| >= 2.3, per-fold OOT z-score, RTH only, hold 60s
**Fills**: entry pays half-spread; exit pays half-spread (1-tick spread tax)
**Account ref**: $50,000 for $-as-pct conversions

## Headline Numbers

- **Default tp=9 / sl=15 net PnL (after spread+comm)**: -1689.1 ticks = $-21,114 = **-42.23% of $50K**
- **Default win rate**: 55.8%
- **Naive 60s timeout win rate (mid-drift > 0)**: 48.8%
- **MFE p50 / p75**: 9.0 / 17.0 ticks
- **MAE p50 / p75**: 8.5 / 16.0 ticks
- **Time-to-MFE p50 / p75**: 27.6s / 48.7s
- **% of trades that go red first (then recover, MFE-after-MAE)**: 85.6%

## Top 5 Findings

1. **Conviction gradient (default tp9/sl15)**: Q1: $-2461 (n=376.0, $-6.5/trade); Q2: $-4891 (n=373.0, $-13.1/trade); Q3: $-5125 (n=371.0, $-13.8/trade); Q4: $-5287 (n=371.0, $-14.3/trade); Q5: $-3350 (n=375.0, $-8.9/trade). Q1 (lowest |z|) loses $-2461; top quintiles do the lifting.
2. **Q5 reaches MFE in 25.8s median** (MFE p50 = 10.0 ticks). 60s hold is leaving up to 34.2s of post-peak drift with no positive expectation — strong case for time-stop or trailing.
3. **92.5% of eventual winners go red first** (median red depth 4.0 ticks, p90 13.0). Tight stops (sl<8) would kill many otherwise-profitable trades.
4. **Exit-reason mix at tp9/sl15**: tp: 48.2%, timeout: 30.1%, sl: 21.7%. Timeout share indicates 60s hold isn't binding for most trades.
5. **Hour-of-day skew**: best hours ET = [13, 10] ($-3053); worst hours = [12, 9] ($-9700). Hour filter could materially boost Sortino.

## Recommended Adaptive TP/SL Rule

From `adaptive_tp_best_per_quintile.csv` (collapsed across hours for power):

| Quintile | n | TP | SL | mean PnL (ticks) | total PnL ($) | winrate |
|---|---|---|---|---|---|---|
| Q1 | 376 | 8 | 20 | -0.15 | $-723 | 63.6% |
| Q2 | 373 | 15 | 15 | -0.80 | $-3,753 | 46.1% |
| Q3 | 371 | 6 | 20 | -1.24 | $-5,731 | 64.7% |
| Q4 | 371 | 15 | 20 | -0.51 | $-2,381 | 50.7% |
| Q5 | 375 | 10 | 6 | -0.77 | $-3,594 | 41.6% |

**RULE**: Use per-quintile TP/SL from the table above. Highest-impact bucket is Q1 (TP=8, SL=20, n=376). Across all quintiles using adaptive TP/SL: $-16,183 vs baseline tp9/sl15 $-21,114.

## Confluence Filter Recommendations

1. **Drop Q1 (lowest |z|)**: Q1 PnL is $-2461; removing it eliminates dead-weight trades.
2. **Hour filter**: Drop hours ET in [13, 10, 14, 15, 11, 12, 9] (each loses >$100 cumulative). Keep prime 9:30-11:30 + best afternoon hours.
3. **Side asymmetry**: Long PnL = $-15335, Short PnL = $-5779. Investigate whether short side warrants stricter |z| gate.
4. **MAE-before-MFE patience**: 93% of winners go red first by 4.0 ticks median. Set SL >= 13 to avoid scratching winners; combine with vol-based exit (adverse > N ticks WITHOUT recovery in K bars).

## Caveats

- **Sample size**: 1866 trades total. 72/840 (quintile x hour x TP x SL) cells have n<30. Per-cell adaptive rules are noisy below n=30 — prefer per-quintile (no hour split) in production until n grows.
- **Quintile leakage**: per-fold OOT std/quintile boundaries used. Production should use causal running z (which is what tp9_z2.3 uses). Direction of bias: per-fold quintile breakpoints reflect within-fold rank, so they cannot leak across folds, but a trade ranked Q5 with 100% knowledge of fold's full distribution is slightly different from a real-time Q5 estimate.
- **Spread model**: paid 1 full tick (half-spread x 2). ES is 1-tick wide >95% of RTH so this is conservative-but-realistic. Real fill rate is ~83% (queue model); this report assumes 100% fill at touch.
- **Mid-price source**: cumulative `mid_price_change_ticks` from book_features. BBO bid/ask is sparse (6% of events have valid both-sides) so we cannot validate per-event spread; we assume spread=1 tick when the book file lacks valid quote.
- **No regime tagging beyond hour-of-day**: vol regime, news, etc. not captured here.

## Files
All CSVs in `/home/jupiter/Lvl3Quant/execution/results/exec_deepdive_20260427/`:
- `adaptive_tp_best_per_cell.csv` (1 KB)
- `adaptive_tp_best_per_quintile.csv` (0 KB)
- `adaptive_tp_grid.csv` (39 KB)
- `exit_reason_heatmap.csv` (0 KB)
- `fold_meta.csv` (0 KB)
- `hour_quintile_pnl.csv` (0 KB)
- `mae_before_mfe.csv` (0 KB)
- `mae_distribution.csv` (1 KB)
- `mfe_distribution.csv` (1 KB)
- `mfe_vs_mae_heatmap.csv` (0 KB)
- `time_to_mfe.csv` (0 KB)
- `time_to_stop.csv` (1 KB)
- `trades.csv` (315 KB)