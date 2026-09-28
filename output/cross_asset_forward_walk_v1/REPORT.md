# cross_asset_forward_walk_v1 — REPORT

**Generated:** 2026-05-22T17:40:50.800167+00:00
**Runtime:** 2.7s
**Candidate:** `short_10s_thr55`
**Days available:** 15
**K grid:** [4, 6, 8, 10, 12]
**Directions:** ['asc', 'desc']
**Features searched per fold (augmented w/ cross-asset):** 31

## Verdict: REJECT

forward-walk profit_days/days_sel = 1/3 = 33.3% < 60%. Cross-asset feature edge does not survive strict train/test separation.

## Cross-asset headline-feature stability

| feature | direction | folds_selected | freq |
|---|---|---:|---:|
| `VIX_change_5d` | desc | 0/15 | 0.00% |
| `SPX_5d_return` | asc | 0/15 | 0.00% |

## Forward-walk LOO out-of-sample metrics

| metric | value |
|---|---|
| days_selected (held-out days included by train rule) | 3 |
| profit_days | 1 |
| profit_ratio | 33.33% |
| pooled t/trade | +13.117 |
| day-Sharpe (annualised) | +5.22 |

## Baseline (no gate, all 15 days)

| metric | value |
|---|---|
| profit_days / total | 8/15 |
| pooled t/trade | +5.030 |
| day-Sharpe (annualised) | +4.54 |

## Rule stability (selection frequency across LOO folds)

| feature | direction | folds_selected | freq |
|---|---|---:|---:|
| `VIX_change_5d` | asc | 9/15 | 60.00% |
| `NQ_vs_ES_5d_corr` | asc | 6/15 | 40.00% |

## Rule stability incl. K (top entries)

| feature | direction | K | folds_selected | freq |
|---|---|---:|---:|---:|
| `VIX_change_5d` | asc | 4 | 9/15 | 60.00% |
| `NQ_vs_ES_5d_corr` | asc | 4 | 4/15 | 26.67% |
| `NQ_vs_ES_5d_corr` | asc | 6 | 2/15 | 13.33% |

## Per-fold detail

| fold | held_out_date | sel_feature | dir | K | train_pr | train_pooled | included | day_pnl_tt |
|---:|---:|---|---|---:|---:|---:|---|---:|
| 0 | 20260316 | `VIX_change_5d` | asc | 4 | 1.0 | 5.8162 | no | -1.637 |
| 1 | 20260317 | `VIX_change_5d` | asc | 4 | 1.0 | 5.8162 | no | -2.326 |
| 2 | 20260318 | `VIX_change_5d` | asc | 4 | 1.0 | 5.8162 | no | +2.910 |
| 3 | 20260319 | `VIX_change_5d` | asc | 4 | 1.0 | 5.8162 | no | -5.501 |
| 4 | 20260401 | `VIX_change_5d` | asc | 4 | 1.0 | 5.8162 | no | -1.444 |
| 5 | 20260402 | `VIX_change_5d` | asc | 4 | 1.0 | 5.8162 | no | -1.104 |
| 6 | 20260403 | `VIX_change_5d` | asc | 4 | 1.0 | 5.8162 | no | +0.074 |
| 7 | 20260406 | `VIX_change_5d` | asc | 4 | 1.0 | 5.8162 | no | +19.196 |
| 8 | 20260407 | `NQ_vs_ES_5d_corr` | asc | 4 | 0.75 | 6.0259 | YES | +15.161 |
| 9 | 20260408 | `NQ_vs_ES_5d_corr` | asc | 4 | 0.75 | 13.8038 | no | +1.504 |
| 10 | 20260409 | `NQ_vs_ES_5d_corr` | asc | 4 | 1.0 | 13.0744 | YES | -1.947 |
| 11 | 20260410 | `NQ_vs_ES_5d_corr` | asc | 6 | 0.8333 | 8.3161 | no | +2.190 |
| 12 | 20260412 | `VIX_change_5d` | asc | 4 | 1.0 | 5.8162 | YES | -3.126 |
| 13 | 20260413 | `NQ_vs_ES_5d_corr` | asc | 6 | 0.8333 | 8.3161 | no | +2.506 |
| 14 | 20260414 | `NQ_vs_ES_5d_corr` | asc | 4 | 0.75 | 13.8038 | no | +2.250 |

## Honest small-sample discussion

**15 days is tiny.** Adding 7 cross-asset features to ~13 ES intrinsic features and 5 calendar features puts the candidate-rule space at ~31 features x 2 directions x 5 K-values = ~310 candidate rules per train fold of 14 days. Per-fold selection variance dominates.

Bonferroni-style noise floor for AUC on this sample size with ~46 hypotheses is roughly 0.78. The original headline AUC of 0.848 for VIX_change_5d desc was suggestive but not conclusive on the post-hoc selection.

**Required before any live deployment:** validate on the NEXT 16+ OOT days (target: 40+ days, all regimes per HC #428 R1).

## Files produced
- `REPORT.md` (this file)
- `loo_folds.csv` — one row per held-out day
- `rule_stability.csv` — (feature, direction) selection frequency
- `rule_stability_with_k.csv` — (feature, direction, K) selection frequency
- `.regen_complete.json` — per HC #485 R5