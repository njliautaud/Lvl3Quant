# day_classifier_forward_walk_v1 — REPORT

**Generated:** 2026-05-22T17:36:48.399653+00:00
**Runtime:** 1.7s
**Candidate:** `short_10s_thr55`
**Days available:** 15
**K grid:** [4, 6, 8, 10, 12]
**Directions:** ['asc', 'desc']
**Features searched per fold:** 21

## Verdict: REJECT

forward-walk profit_days/days_sel = 1/6 = 16.7% < 60%. K-selection drove the original +2.72 t/trade.

## Forward-walk LOO out-of-sample metrics

| metric | value |
|---|---|
| days_selected (held-out days included by train rule) | 6 |
| profit_days | 1 |
| profit_ratio | 16.67% |
| pooled t/trade | -0.223 |
| day-Sharpe (annualised) | -10.12 |

## Baseline (no gate, all 15 days)

| metric | value |
|---|---|
| profit_days / total | 8/15 |
| pooled t/trade | +5.030 |
| day-Sharpe (annualised) | +4.54 |

## Rule stability (selection frequency across LOO folds)

| feature | direction | folds_selected | freq |
|---|---|---:|---:|
| `abs_trend_to_10` | asc | 5/15 | 33.33% |
| `trend_ticks_open_to_945` | desc | 3/15 | 20.00% |
| `dow_thu` | asc | 2/15 | 13.33% |
| `n_events_early` | desc | 2/15 | 13.33% |
| `spread_avg_early` | asc | 2/15 | 13.33% |
| `n_trades_early` | desc | 1/15 | 6.67% |

## Rule stability incl. K (top entries)

| feature | direction | K | folds_selected | freq |
|---|---|---:|---:|---:|
| `abs_trend_to_10` | asc | 8 | 4/15 | 26.67% |
| `dow_thu` | asc | 10 | 2/15 | 13.33% |
| `n_events_early` | desc | 8 | 2/15 | 13.33% |
| `trend_ticks_open_to_945` | desc | 4 | 2/15 | 13.33% |
| `n_trades_early` | desc | 6 | 1/15 | 6.67% |
| `spread_avg_early` | asc | 8 | 1/15 | 6.67% |
| `trend_ticks_open_to_945` | desc | 8 | 1/15 | 6.67% |
| `spread_avg_early` | asc | 4 | 1/15 | 6.67% |
| `abs_trend_to_10` | asc | 6 | 1/15 | 6.67% |

## Per-fold detail

| fold | held_out_date | sel_feature | dir | K | train_pr | train_pooled | included | day_pnl_tt |
|---:|---:|---|---|---:|---:|---:|---|---:|
| 0 | 20260316 | `dow_thu` | asc | 10 | 0.8 | 5.7544 | YES | -1.637 |
| 1 | 20260317 | `dow_thu` | asc | 10 | 0.8 | 5.5977 | YES | -2.326 |
| 2 | 20260318 | `n_events_early` | desc | 8 | 0.75 | 5.4648 | no | +2.910 |
| 3 | 20260319 | `abs_trend_to_10` | asc | 8 | 0.75 | 9.9479 | no | -5.501 |
| 4 | 20260401 | `trend_ticks_open_to_945` | desc | 4 | 1.0 | 2.305 | YES | -1.444 |
| 5 | 20260402 | `n_trades_early` | desc | 6 | 0.8333 | 7.2389 | YES | -1.104 |
| 6 | 20260403 | `n_events_early` | desc | 8 | 0.75 | 5.4648 | no | +0.074 |
| 7 | 20260406 | `spread_avg_early` | asc | 8 | 0.75 | 4.9434 | no | +19.196 |
| 8 | 20260407 | `trend_ticks_open_to_945` | desc | 8 | 0.75 | 2.7227 | no | +15.161 |
| 9 | 20260408 | `spread_avg_early` | asc | 4 | 0.75 | 2.1795 | no | +1.504 |
| 10 | 20260409 | `abs_trend_to_10` | asc | 6 | 0.8333 | 10.9077 | YES | -1.947 |
| 11 | 20260410 | `trend_ticks_open_to_945` | desc | 4 | 0.75 | 2.0875 | YES | +2.190 |
| 12 | 20260412 | `abs_trend_to_10` | asc | 8 | 0.75 | 9.9479 | no | -3.126 |
| 13 | 20260413 | `abs_trend_to_10` | asc | 8 | 0.75 | 9.9479 | no | +2.506 |
| 14 | 20260414 | `abs_trend_to_10` | asc | 8 | 0.75 | 9.9479 | no | +2.250 |

## Honest small-sample discussion

**15 days is tiny.** Forward-walk LOO removes the K-selection-on-all-days bias of the original day_classifier_v1 result, but it does NOT eliminate small-sample risk. With 15 observations:

- One outlier day flipping inclusion can swing profit_ratio by ~7 percentage points.
- The training fold has only 14 days; the (feature, direction, K) search has ~21 features × 2 directions × 5 Ks = ~210 candidate rules per fold, vastly more candidates than training observations — so the per-fold rule choice itself carries selection variance.
- Stratified regime check (HC #428 R1) requires ≥40 OOT days. We have 15.

**Required before any live deployment:** validate on the NEXT 16+ OOT days (target: 40+ days, all regimes per HC #428 R1).

## Files produced
- `REPORT.md` (this file)
- `loo_folds.csv` — one row per held-out day
- `rule_stability.csv` — (feature, direction) selection frequency
- `rule_stability_with_k.csv` — (feature, direction, K) selection frequency
- `.regen_complete.json` — per HC #485 R5