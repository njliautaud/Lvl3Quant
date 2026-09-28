# HC #445 Meta-Classifier v2 (multi-h) — hc443_upside_sl3_tp8_h10

Fills: **2431** across **32** dates.

## OOS time-ordered split
- train: 1901 fills (22 dates)
- test: 530 fills (10 dates)
- test baseline (no filter): mean_tk=-0.7864  PF=0.550  WR=38.87%
- AUC_oos = **0.5353**   (best_iter=1)

## Threshold sweep on OOS test

| thr | n | mean_tk | PF | WR | sharpe |
|---|---:|---:|---:|---:|---:|
| 0.30 | 530 | -0.7864 | 0.550 | 38.87% | -5.858 |
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

- `queue_ahead_log1p`: 42.2
- `signal_density`: 26.2
- `pred_abs_5s_z_in_day`: 25.5
- `pred_decay_1_10`: 17.2
- `pred_abs_1s_z_in_day`: 12.1
- `pred_10s`: 12.1
- `pred_strength`: 10.9
- `pred_abs_10s`: 8.0
- `pred_1s_5s_diff`: 4.5
- `hour`: 4.5
