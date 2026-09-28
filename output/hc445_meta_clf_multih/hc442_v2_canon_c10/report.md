# HC #445 Meta-Classifier v2 (multi-h) — hc442_v2_canon_c10

Fills: **3455** across **32** dates.

## OOS time-ordered split
- train: 2513 fills (22 dates)
- test: 942 fills (10 dates)
- test baseline (no filter): mean_tk=-0.3309  PF=0.440  WR=30.57%
- AUC_oos = **0.5654**   (best_iter=18)

## Threshold sweep on OOS test

| thr | n | mean_tk | PF | WR | sharpe |
|---|---:|---:|---:|---:|---:|
| 0.30 | 88 | -0.3419 | 0.377 | 34.09% | -4.006 |
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

- `minute_of_day`: 329.2
- `queue_ahead_log1p`: 325.1
- `pred_h_range`: 277.9
- `pred_5s_10s_diff`: 270.0
- `pred_abs_1s_z_in_day`: 231.6
- `pred_strength`: 228.5
- `pred_abs_5s`: 194.8
- `pred_decay_1_10`: 189.1
- `pred_1s_5s_diff`: 180.1
- `pred_abs_10s_z_in_day`: 148.1
