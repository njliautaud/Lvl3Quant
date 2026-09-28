# HC #445 Meta-Classifier v2 (multi-h) — v342_short_5s_top0.5_t1422_R2fix

Fills: **699** across **10** dates.

## OOS time-ordered split
- train: 646 fills (7 dates)
- test: 53 fills (3 dates)
- test baseline (no filter): mean_tk=-0.3949  PF=0.437  WR=49.06%
- AUC_oos = **0.5755**   (best_iter=114)

## Threshold sweep on OOS test

| thr | n | mean_tk | PF | WR | sharpe |
|---|---:|---:|---:|---:|---:|
| 0.30 | 52 | -0.3760 | 0.453 | 50.00% | -2.711 |
| 0.35 | 50 | -0.3760 | 0.453 | 50.00% | -2.659 |
| 0.40 | 43 | -0.3527 | 0.475 | 51.16% | -2.314 |
| 0.45 | 33 | -0.3457 | 0.482 | 51.52% | -1.987 |
| 0.50 | 26 | -0.2991 | 0.529 | 53.85% | -1.530 |
| 0.55 | 15 | -0.3093 | 0.518 | 53.33% | -1.201 |
| 0.60 | 7 | -0.2331 | 0.605 | 57.14% | -0.623 |
| 0.65 | 2 | — | — | — | — |
| 0.70 | 1 | — | — | — | — |
| 0.75 | 0 | — | — | — | — |
| 0.80 | 0 | — | — | — | — |

## Verdict
❌ **FAIL** — no threshold yields n≥50, mean_tk>0, PF≥1.10 on time-ordered OOS test.

## Top-10 feature importance (gain)

- `pred_h_range`: 293.1
- `pred_5s_10s_diff`: 288.0
- `pred_decay_1_10`: 285.1
- `pred_1s_5s_diff`: 279.0
- `pred_strength`: 207.8
- `pred_abs_1s_z_in_day`: 183.2
- `minute_of_day`: 157.4
- `pred_1s`: 146.5
- `pred_abs_1s`: 144.7
- `signal_density`: 142.4
