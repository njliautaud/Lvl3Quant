# cross_asset_day_classifier_v1 — REPORT

**Generated:** 2026-05-22T13:38:03.787817
**Runtime:** 1.0s
**Sample:** 15 OOT days, 8 profitable, 7 unprofitable

## Verdict: ACCEPT-NEW-AXIS — cross-asset feature `VIX_change_5d` AUC=0.848 lifts above ES-only baseline 0.759

**Logistic combo:** Logistic combo AUC=0.750 ~tied with ES-only baseline 0.759; no improvement.

## Comparison vs ES-only baseline

- ES-only champion: `trend_ticks_open_to_945` (asc) AUC = **0.759**
- Best ES feature in this run: `trend_ticks_open_to_945` (asc) AUC = 0.759
- Best CROSS-ASSET feature: `VIX_change_5d` (desc) AUC = **0.848**
- Logistic regression combo (6 feats, L2 C=0.25, LOO): AUC = 0.750

## Cross-asset feature AUCs (ranked)

| feature | kind | direction | AUC |
|---|---|---|---:|
| `VIX_change_5d` | XA | desc | 0.848 |
| `SPX_5d_return` | XA | asc | 0.795 |
| `DXY_5d_return` | XA | desc | 0.723 |
| `NQ_vs_ES_5d_corr` | XA | desc | 0.688 |
| `VIX_zscore_20d` | XA | desc | 0.688 |
| `VIX_close_prev` | XA | desc | 0.670 |
| `week_of_month` | XA | desc | 0.634 |
| `NQ_overnight_return` | XA | desc | 0.634 |
| `YM_overnight_return` | XA | desc | 0.562 |
| `day_of_month` | XA | desc | 0.554 |

## All features (top 10)

| feature | kind | direction | AUC |
|---|---|---|---:|
| `VIX_change_5d` | XA | desc | 0.848 |
| `SPX_5d_return` | XA | asc | 0.795 |
| `trend_ticks_open_to_945` | ES | asc | 0.759 |
| `DXY_5d_return` | XA | desc | 0.723 |
| `dow_thu` | ES | desc | 0.714 |
| `spread_avg_early` | ES | desc | 0.714 |
| `NQ_vs_ES_5d_corr` | XA | desc | 0.688 |
| `VIX_zscore_20d` | XA | desc | 0.688 |
| `VIX_close_prev` | XA | desc | 0.670 |
| `abs_trend_to_10` | ES | desc | 0.670 |

## Method

- Same LOO framework as `day_classifier_v1`. Single-feature rank-AUC is equivalent
  to leave-one-out threshold rule under monotonic scoring.
- Cross-asset features pulled via yfinance:
  - yfinance ok = `True` (ok)
  - symbols: ['^VIX', 'NQ=F', 'YM=F', '^GSPC', 'DX-Y.NYB']
  - lookback range: 2026-01-15 .. 2026-05-15
- All cross-asset features are CAUSAL (use only data with bar timestamp
  STRICTLY BEFORE the trading date — i.e. T-1 close and earlier).
- Overnight return is implemented as prev-day full-return because Yahoo
  daily bars do not separate cleanly into pre-RTH vs RTH for futures;
  see code comment in `_prev_day_return`.
- NQ_vs_ES_5d_corr uses ^GSPC as the ES proxy (no daily ES future fetched).
- Logistic regression: standardised features fit on train fold only,
  L2 C=0.25 (heavy regularisation), LOO predicted prob, AUC computed pooled.

## Honest caveats

- **15-day sample** — same overfit risk class as day_classifier_v1. A single
  label flip can swing single-feature AUC by ~0.07.
- We tried 10 cross-asset features and 13 ES features; with 23 candidates the
  Bonferroni-adjusted noise-floor AUC is roughly 0.75+. Anything below ~0.78
  is plausibly chance.
- Yahoo end-of-day bars may have stale closes for some futures contracts;
  treat single-day spikes with caution.
- We did NOT run multi-feature LGBM (already shown to overfit at this sample).
- Validation on next 15+ OOT days is the only way to certify any feature.

## Files produced
- `combined_features.parquet`
- `feature_auc.csv`
- `loo_logistic_predictions.csv`
- `REPORT.md` (this file)