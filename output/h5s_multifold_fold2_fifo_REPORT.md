# HC #495.1 FOLD-2 FIFO Grade — h5s low-LR multifold

**Date**: 2026-05-29
**Model**: CNN-Mamba v3 h5s low-LR multifold (fold 2)
**OOT Date**: 2026-04-14 (1 day)
**Prediction Count**: 21,016 windows
**Grader**: `scripts/grade_h5s_fold_N.py --fold 2` (apples-to-apples replica of `h5s_lowlr_fifo_sweep.py`)
**Status**: COMPLETE

---

## Headline Result (1s_top10%)

| Variant | Net Ticks | T/Trade | Trades | Sharpe | Verdict |
|---------|-----------|---------|--------|--------|---------|
| Ungated | +549.15 | **+0.2613** | 2,102 | 473.30 | MARGINAL_1DAY |
| Fill-prob gate q>=0.50 | +534.84 | **+0.2628** | 2,035 | 471.79 | MARGINAL_1DAY |

All 12 ungated cells and 11 of 12 gated cells are FIFO-net-positive. The gate is virtually a no-op on this day (1s gate passes 20342/21016, 5s_proxy3s gate passes 20352/21016).

---

## Full Sweep — Fold 2 Only

| Variant | Horizon | Top% | Net Ticks | T/Trade | Count | Sharpe |
|---------|---------|------|-----------|---------|-------|--------|
| ungated | 1s | 1% | 81.66 | +0.3870 | 211 | 872.53 |
| ungated | 1s | 2% | 150.70 | +0.3580 | 421 | 633.13 |
| ungated | 1s | 5% | 284.32 | +0.2705 | 1,051 | 528.56 |
| ungated | 1s | 10% | 549.15 | **+0.2613** | 2,102 | 473.30 |
| ungated | 1s | 20% | 997.80 | +0.2373 | 4,204 | 443.23 |
| ungated | 1s | 50% | 1384.99 | +0.1318 | 10,508 | 239.84 |
| ungated | 5s | 1% | 99.16 | +0.4700 | 211 | 600.40 |
| ungated | 5s | 2% | 226.20 | +0.5373 | 421 | 623.10 |
| ungated | 5s | 5% | 392.32 | +0.3733 | 1,051 | 371.94 |
| ungated | 5s | 10% | 687.15 | +0.3269 | 2,102 | 335.41 |
| ungated | 5s | 20% | 1267.80 | +0.3016 | 4,204 | 317.12 |
| ungated | 5s | 50% | 2120.99 | +0.2018 | 10,508 | 217.20 |
| gate>=0.5 | 1s | 1% | 78.80 | +0.3863 | 204 | 865.85 |
| gate>=0.5 | 1s | 2% | 147.97 | +0.3636 | 407 | 636.57 |
| gate>=0.5 | 1s | 5% | 279.23 | +0.2743 | 1,018 | 531.45 |
| gate>=0.5 | 1s | 10% | 534.84 | **+0.2628** | 2,035 | 471.79 |
| gate>=0.5 | 1s | 20% | 976.56 | +0.2400 | 4,069 | 444.34 |
| gate>=0.5 | 1s | 50% | 1387.20 | +0.1364 | 10,171 | 246.04 |
| gate>=0.5 | 5s | 1% | 101.80 | +0.4990 | 204 | 633.14 |
| gate>=0.5 | 5s | 2% | 217.09 | +0.5321 | 408 | 613.92 |
| gate>=0.5 | 5s | 5% | 383.23 | +0.3765 | 1,018 | 370.52 |
| gate>=0.5 | 5s | 10% | 672.46 | +0.3303 | 2,036 | 336.38 |
| gate>=0.5 | 5s | 20% | 1215.30 | +0.2985 | 4,071 | 310.71 |
| gate>=0.5 | 5s | 50% | 2119.32 | +0.2083 | 10,176 | 222.11 |

---

## Cumulative Multifold Tally (Folds 0 + 1 + 2)

### Ungated — folds positive of 3 per cell

All 12 ungated cells are 3/3 positive across folds 0, 1, 2.

| Cell | F0 | F1 | F2 | pos/3 | median t/trd |
|------|----|----|----|-------|---------------|
| 1s_top1% | +2.024 | +0.366 | +0.387 | 3/3 | +0.387 |
| 1s_top2% | +1.203 | +0.324 | +0.358 | 3/3 | +0.358 |
| 1s_top5% | +7.252 | +0.266 | +0.270 | 3/3 | +0.270 |
| **1s_top10%** | **+3.890** | **+0.225** | **+0.261** | **3/3** | **+0.261** |
| 1s_top20% | +1.994 | +0.201 | +0.237 | 3/3 | +0.237 |
| 1s_top50% | +0.957 | +0.098 | +0.132 | 3/3 | +0.132 |
| 5s_top1% | +16.124 | +0.904 | +0.470 | 3/3 | +0.904 |
| 5s_top2% | +8.229 | +0.549 | +0.537 | 3/3 | +0.549 |
| 5s_top5% | +3.709 | +0.445 | +0.373 | 3/3 | +0.445 |
| 5s_top10% | +3.720 | +0.382 | +0.327 | 3/3 | +0.382 |
| 5s_top20% | +1.887 | +0.260 | +0.302 | 3/3 | +0.302 |
| 5s_top50% | +0.905 | +0.142 | +0.202 | 3/3 | +0.202 |

### Fill-prob gate q>=0.50 — folds positive of 3 per cell

11 of 12 cells are 3/3 positive. Only `gate 5s_top2%` is 2/3 (fold 0 = −0.184).

| Cell | F0 | F1 | F2 | pos/3 | median t/trd |
|------|----|----|----|-------|---------------|
| 1s_top1% | +0.053 | +0.371 | +0.386 | 3/3 | +0.371 |
| 1s_top2% | +0.432 | +0.333 | +0.364 | 3/3 | +0.364 |
| 1s_top5% | +0.077 | +0.271 | +0.274 | 3/3 | +0.271 |
| **1s_top10%** | **+0.218** | **+0.228** | **+0.263** | **3/3** | **+0.228** |
| 1s_top20% | +0.124 | +0.204 | +0.240 | 3/3 | +0.204 |
| 1s_top50% | +0.091 | +0.101 | +0.136 | 3/3 | +0.101 |
| 5s_top1% | +0.053 | +0.844 | +0.499 | 3/3 | +0.499 |
| 5s_top2% | −0.184 | +0.547 | +0.532 | **2/3** | +0.532 |
| 5s_top5% | +0.351 | +0.453 | +0.377 | 3/3 | +0.377 |
| 5s_top10% | +0.447 | +0.377 | +0.330 | 3/3 | +0.377 |
| 5s_top20% | +0.147 | +0.269 | +0.299 | 3/3 | +0.269 |
| 5s_top50% | +0.053 | +0.142 | +0.208 | 3/3 | +0.142 |

---

## HC #495.1 Interim Read

**Bar (from HC #495.1)**: ≥30 of ~46 OOT days positive at 1s_top10%, regime skew ≤0.50, Sharpe ≥1.0 → revival confirmed.

**Where we stand at fold 2**:
- Days positive on 1s_top10% (ungated): **3 of 3 folds** (= 3 of 3 OOT days, since each fold = 1 OOT day in this WF setup).
- Days positive on 1s_top10% (gated): **3 of 3 folds**.
- Per-day intraday Sharpe ≥320 (extrapolated) across all 3 folds — Sharpe gate trivially satisfied.
- Regime skew not computable yet (need green vs red day classification, and at least 1 of each — current 3 OOT days appear all green).

**Trajectory**: On the linear extrapolation from 3/3 positive at f2, expected positive days at fold 9 ≈ 9-10 (assuming 1 OOT day per fold). Reported window says ~46 days total — that implies later folds will carry multi-day OOT, not 1-day. Fold 1 had 22,647 windows / fold 2 had 21,016 windows (vs fold 0's 939) suggesting the multifold WF expands OOT progressively. The 1-day-per-fold assumption is wrong starting at fold 1 — investigate `oot_files` count per fold once all complete.

**Interim verdict**: **Revival case STRENGTHENING.** 3/3 folds net-positive at the headline cell under both variants; gate is essentially a no-op on these days (>96% of windows pass q≥0.50, which itself is a data quality flag worth tracking).

---

## Caveats / Watch-Items

1. **Label-FIFO, not queue-FIFO**: This is `sign(pred)*label − 0.376`, NOT full FIFOReplayEngine. Apples-to-apples with the prior fold 0 report. Queue-aware grader (when built) is expected to soften results 30–50%.
2. **Gate behavior anomalous**: q≥0.50 passes 96–97% of windows in folds 1 and 2. Either (a) the XGBoost fill-prob head trained May 1 is now severely miscalibrated, (b) the day regime is unusually liquid, or (c) the 3s proxy for 5s is too permissive. The gate provides almost no selectivity here — a tighter q≥0.80 stress test is warranted.
3. **Regime skew unknown**: All 3 OOT days (Apr 12, 13, 14 2026) appear to be green/strong days for the signal. Need at least one red-day OOT before any regime-skew read is meaningful (per HC #428 R1).
4. **Fold 0 magnitudes outlier**: Fold 0 (939 windows, 94 trades at top10%) has 1–2 orders of magnitude larger t/trade than folds 1, 2 (~21k windows, ~2.1k trades). Suggests fold 0 OOT day was both shorter and stronger — its +3.89 is a small-sample artifact, not the representative signal. Fold 1/2 numbers (~+0.23 t/trade) are the credible base rate going forward.

---

## Files

- Predictions: `output/cnn_mamba_v3_h5s_lowlr_multifold/fold_02_oot_predictions.npz`
- Sweep dir: `output/h5s_multifold_fold2_fifo/` (summary.csv + result.json)
- Cumulative aggregator: `output/h5s_multifold_cumulative.json`
- Persistent grader: `scripts/grade_h5s_fold_N.py --fold N` (HC #491 R5 infra — usable for folds 3–9)

---

**Generated**: 2026-05-29 (auto)
**Authority**: HC #495.1 Phase 2 (cumulative multifold viability)
