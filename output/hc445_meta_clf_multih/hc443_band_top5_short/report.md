# HC #445 Meta-Classifier v2 (multi-h) — hc443_band_top5_short

Fills: **19774** across **36** dates.

## OOS time-ordered split
- train: 14023 fills (25 dates)
- test: 5751 fills (11 dates)
- test baseline (no filter): mean_tk=-0.3476  PF=0.428  WR=28.85%
- AUC_oos = **0.5500**   (best_iter=51)

## Threshold sweep on OOS test

| thr | n | mean_tk | PF | WR | sharpe |
|---|---:|---:|---:|---:|---:|
| 0.30 | 1184 | -0.3536 | 0.368 | 33.36% | -14.728 |
| 0.35 | 170 | -0.3966 | 0.299 | 34.12% | -6.804 |
| 0.40 | 7 | -0.4474 | 0.285 | 28.57% | -1.625 |
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

- `minute_of_day`: 1561.2
- `signal_density`: 1048.8
- `pred_strength`: 901.9
- `pred_abs_1s_z_in_day`: 875.8
- `queue_ahead_log1p`: 863.4
- `pred_5s_10s_diff`: 738.3
- `pred_abs_1s`: 720.4
- `pred_decay_1_10`: 622.7
- `pred_5s`: 568.5
- `pred_abs_10s_z_in_day`: 559.7
