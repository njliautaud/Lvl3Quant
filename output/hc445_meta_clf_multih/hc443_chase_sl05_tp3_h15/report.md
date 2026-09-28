# HC #445 Meta-Classifier v2 (multi-h) — hc443_chase_sl05_tp3_h15

Fills: **2431** across **32** dates.

## OOS time-ordered split
- train: 1901 fills (22 dates)
- test: 530 fills (10 dates)
- test baseline (no filter): mean_tk=-0.3807  PF=0.411  WR=24.91%
- AUC_oos = **0.5483**   (best_iter=28)

## Threshold sweep on OOS test

| thr | n | mean_tk | PF | WR | sharpe |
|---|---:|---:|---:|---:|---:|
| 0.30 | 80 | -0.4073 | 0.332 | 27.50% | -4.183 |
| 0.35 | 16 | -0.5635 | 0.099 | 25.00% | -4.544 |
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

- `minute_of_day`: 411.1
- `pred_abs_10s`: 362.9
- `pred_5s_10s_diff`: 359.2
- `pred_10s`: 237.0
- `queue_ahead_log1p`: 233.1
- `pred_1s_5s_diff`: 211.3
- `pred_abs_1s_z_in_day`: 202.2
- `pred_decay_1_10`: 189.2
- `signal_density`: 181.0
- `pred_abs_5s`: 176.2
