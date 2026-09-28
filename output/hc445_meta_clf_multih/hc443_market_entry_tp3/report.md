# HC #445 Meta-Classifier v2 (multi-h) — hc443_market_entry_tp3

Fills: **4299** across **36** dates.

## OOS time-ordered split
- train: 3114 fills (25 dates)
- test: 1185 fills (11 dates)
- test baseline (no filter): mean_tk=-0.4950  PF=0.347  WR=12.49%
- AUC_oos = **0.6512**   (best_iter=6)

## Threshold sweep on OOS test

| thr | n | mean_tk | PF | WR | sharpe |
|---|---:|---:|---:|---:|---:|
| 0.30 | 0 | — | — | — | — |
| 0.35 | 0 | — | — | — | — |
| 0.40 | 0 | — | — | — | — |
| 0.45 | 0 | — | — | — | — |
| 0.50 | 0 | — | — | — | — |
| 0.55 | 0 | — | — | — | — |
| 0.60 | 0 | — | — | — | — |
| 0.65 | 0 | — | — | — | — |
| 0.70 | 0 | — | — | — | — |
| 0.75 | 0 | — | — | — | — |
| 0.80 | 0 | — | — | — | — |

## Verdict
❌ **FAIL** — no threshold yields n≥50, mean_tk>0, PF≥1.10 on time-ordered OOS test.

## Top-10 feature importance (gain)

- `minute_of_day`: 343.8
- `pred_strength`: 338.9
- `pred_5s_10s_diff`: 141.6
- `pred_abs_10s_z_in_day`: 127.3
- `pred_abs_1s`: 95.0
- `pred_1s_5s_diff`: 93.9
- `pred_decay_1_10`: 65.1
- `pred_10s`: 61.0
- `pred_abs_5s_z_in_day`: 57.6
- `pred_abs_5s`: 42.3
