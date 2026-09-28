# HC #445 Meta-Classifier v2 (multi-h) — hc442_primary_cancel10s_match

Fills: **1225** across **8** dates.

## OOS time-ordered split
- train: 1076 fills (5 dates)
- test: 149 fills (3 dates)
- test baseline (no filter): mean_tk=-0.2183  PF=0.678  WR=22.15%
- AUC_oos = **0.5008**   (best_iter=4)

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

- `minute_of_day`: 72.5
- `pred_1s_5s_diff`: 59.2
- `pred_abs_5s`: 34.3
- `pred_5s_10s_diff`: 20.4
- `signal_density`: 18.1
- `pred_abs_10s_z_in_day`: 15.8
- `pred_h_range`: 11.8
- `queue_ahead_log1p`: 11.3
- `pred_abs_10s`: 10.4
- `pred_10s`: 6.5
