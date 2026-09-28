# PRELIMINARY — HC #432 Verdict — v342_short_10s_top0.5_t2831_R2fix

- Dates present: **15 / 47** (waiting on Neptune)
- Config: side=short, horizon=10s, conf_band=top0.5, TP=2.0, SL=1.5, hold=2.4s, cancel=10.0s, order=passive_at_touch

## Overall (FIFO realized)
| metric | value |
|---|---|
| n_fills | 1427 |
| sum_net_ticks | -322.05 |
| mean_net_tk/fill | -0.2257 |
| Sharpe (sqrt-N) | -5.612 |
| Sortino | -17.347 |
| PF | 0.725 |
| WR | 47.93% |
| day_concentration | 0.460 |

## Per-regime
| regime | n | sum_tk | mean_tk | Sharpe | PF | WR |
|---|---:|---:|---:|---:|---:|---:|
| green | 717 | -180.09 | -0.2512 | -4.720 | 0.675 | 48.12% |
| red | 710 | -141.96 | -0.1999 | -3.311 | 0.770 | 47.75% |
| flat | 0 | 0.00 | 0.0000 | 0.000 | 0.000 | 0.00% |

## HC #428 R1 — regime-agnostic OOT
- ratio |Sh_g − Sh_r| / max = **0.299** (threshold ≤ 0.50)
- day_concentration = **0.460** (cap ≤ 0.70)
- **R1 verdict: PASS**

## HC #428 R2 — MFE within horizon
| check | value | pass |
|---|---|---|
| TP ≤ p90_MFE@h | 2.0 vs 162400.000 | True |
| hold ≤ 1.5h | 2.4s vs 15.0s | True |
| cancel ≤ h | 10.0s vs 10.0s | True |
- **R2 verdict: PASS**

## Combined
- **FAIL** (R1=P, R2=P, Sharpe>0=F)

## Per-day
| date | regime | n | sum_tk | mean_tk | Sharpe | PF | WR |
|---|---|---:|---:|---:|---:|---:|---:|
| 20260223 | red | 88 | 13.41 | 0.1524 | 0.823 | 1.193 | 57.95% |
| 20260224 | green | 46 | 11.70 | 0.2544 | 0.999 | 1.347 | 60.87% |
| 20260225 | green | 4 | -0.50 | -0.1260 | -0.125 | 0.866 | 50.00% |
| 20260226 | red | 7 | -8.13 | -1.1617 | -2.762 | 0.121 | 14.29% |
| 20260227 | red | 13 | -10.39 | -0.7991 | -1.714 | 0.385 | 30.77% |
| 20260302 | green | 505 | -176.88 | -0.3503 | -6.104 | 0.533 | 45.94% |
| 20260303 | green | 21 | -6.90 | -0.3284 | -0.972 | 0.627 | 42.86% |
| 20260304 | green | 99 | -0.72 | -0.0073 | -0.044 | 0.991 | 53.54% |
| 20260305 | red | 24 | -27.52 | -1.1468 | -3.869 | 0.228 | 20.83% |
| 20260306 | red | 17 | -6.89 | -0.4054 | -1.000 | 0.592 | 47.06% |
| 20260309 | green | 42 | -6.79 | -0.1617 | -0.601 | 0.828 | 50.00% |
| 20260310 | red | 449 | -108.32 | -0.2413 | -3.311 | 0.715 | 46.55% |
| 20260311 | red | 1 | 1.62 | 1.6240 | 0.000 | inf | 100.00% |
| 20260312 | red | 5 | 4.62 | 0.9240 | 1.320 | 3.463 | 80.00% |
| 20260313 | red | 106 | -0.36 | -0.0034 | -0.020 | 0.996 | 52.83% |
