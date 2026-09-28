# day_classifier_v1 — REPORT

**Generated:** 2026-05-22T13:32:50.214597
**Runtime:** 24.8s

## Verdict: ACCEPT (single-feature) — `trend_ticks_open_to_945` AUC=0.759; LGBM overfit (AUC=0.179)

- **LGBM LOO AUC:** 0.179  (gate threshold for ACCEPT: > 0.65)
- **Best single-feature AUC:** 0.759  (feature: `trend_ticks_open_to_945`, direction: asc)
- **Baseline profitable-day ratio (no gating):** 8/15 = 53.33%

## Gated P&L Simulation — LGBM (top-K predicted-profitable days)

| K | days_sel | profit_days | profit_ratio | mean_t/trade (pooled) | day_sharpe_ann | pooled_sharpe_ann |
|---|----------|-------------|--------------|----------------------:|---------------:|------------------:|
| 6 | 6 | 1 | 16.67% | 0.913 | -10.56 | nan |
| 8 | 8 | 3 | 37.50% | 2.766 | 3.80 | nan |
| 10 | 10 | 4 | 40.00% | 2.364 | 3.28 | nan |
| 12 | 12 | 5 | 41.67% | 6.162 | 4.09 | nan |
| 15 | 15 | 8 | 53.33% | 5.030 | 4.54 | nan |

## Gated P&L Simulation — SINGLE FEATURE (`trend_ticks_open_to_945` asc)

| K | days_sel | profit_days | profit_ratio | mean_t/trade (pooled) | day_sharpe_ann |
|---|----------|-------------|--------------|----------------------:|---------------:|
| 6 | 6 | 5 | 83.33% | 2.998 | 9.36 |
| 8 | 8 | 6 | 75.00% | 2.723 | 7.49 |
| 10 | 10 | 6 | 60.00% | 2.585 | 5.12 |
| 12 | 12 | 8 | 66.67% | 5.266 | 7.42 |

## Top-3 LGBM feature importances
- `n_events_early`: 111.9
- `spread_avg_early`: 92.2
- `dow_fri`: 84.9

## Top-3 single-feature classifiers (LOO-equiv AUC)
- `trend_ticks_open_to_945` (asc): AUC=0.759
- `spread_avg_early` (desc): AUC=0.714
- `dow_thu` (desc): AUC=0.714

## Honest overfit-risk assessment

**Sample size: 16 days.** This is dangerously small for an ML classifier.
Mitigations applied:
- Strict leave-one-out CV (no held-out day ever touches its training fold)
- Tiny LGBM: num_leaves=8, max_depth=3, min_data_in_leaf=2, bagging+feature_frac
- Single-feature baseline reported alongside; if a simple rule matches LGBM AUC,
  prefer the rule (robust > fitted on 16 obs).

**Caveats remaining:**
- LOO with 16 samples still has high variance — single-day swing changes AUC by ~0.06.
- Feature engineering choices (cutoff 10:00 ET, window 9:30-10:00) were not
  themselves cross-validated — implicit researcher degrees of freedom.
- Three macro days (FOMC/NFP/CPI) in 16 may dominate; check is_fomc/is_nfp/is_cpi flags.
- Recommended: validate on the NEXT batch of 16+ OOT days before any live deployment.

## Files produced
- `day_features.parquet`
- `cv_results.csv`
- `feature_importance.csv`
- `single_feature_auc.csv`
- `simulated_gated_pnl.csv`
- `REPORT.md` (this file)