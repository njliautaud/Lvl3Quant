# PRELIMINARY — HC #432 Verdict — v342_short_5s_top0.5_t1422_R2fix

- Dates present: **15 / 47** (waiting on Neptune)
- Config: side=short, horizon=5s, conf_band=top0.5, TP=1.0, SL=1.0, hold=1.1s, cancel=5.0s, order=passive_at_touch

## Overall (FIFO realized)
| metric | value |
|---|---|
| n_fills | 1337 |
| sum_net_ticks | -496.71 |
| mean_net_tk/fill | -0.3715 |
| Sharpe (sqrt-N) | -13.659 |
| Sortino | -118.016 |
| PF | 0.454 |
| WR | 49.96% |
| day_concentration | 0.280 |

## Per-regime
| regime | n | sum_tk | mean_tk | Sharpe | PF | WR |
|---|---:|---:|---:|---:|---:|---:|
| green | 505 | -176.38 | -0.3493 | -7.857 | 0.477 | 51.29% |
| red | 832 | -320.33 | -0.3850 | -11.192 | 0.441 | 49.16% |
| flat | 0 | 0.00 | 0.0000 | 0.000 | 0.000 | 0.00% |

## HC #428 R1 — regime-agnostic OOT
- ratio |Sh_g − Sh_r| / max = **0.298** (threshold ≤ 0.50)
- day_concentration = **0.280** (cap ≤ 0.70)
- **R1 verdict: PASS**

## HC #428 R2 — MFE within horizon
| check | value | pass |
|---|---|---|
| TP ≤ p90_MFE@h | 1.0 vs 162400.000 | True |
| hold ≤ 1.5h | 1.1s vs 7.5s | True |
| cancel ≤ h | 5.0s vs 5.0s | True |
- **R2 verdict: PASS**

## Combined
- **FAIL** (R1=P, R2=P, Sharpe>0=F)

## Per-day
| date | regime | n | sum_tk | mean_tk | Sharpe | PF | WR |
|---|---|---:|---:|---:|---:|---:|---:|
| 20260223 | red | 62 | -23.31 | -0.3760 | -2.937 | 0.453 | 50.00% |
| 20260224 | green | 76 | -30.58 | -0.4023 | -3.485 | 0.430 | 48.68% |
| 20260225 | green | 11 | -7.14 | -0.6487 | -2.132 | 0.259 | 36.36% |
| 20260226 | red | 19 | -8.14 | -0.4286 | -1.821 | 0.408 | 47.37% |
| 20260227 | red | 294 | -97.04 | -0.3301 | -5.727 | 0.491 | 52.04% |
| 20260302 | green | 265 | -82.14 | -0.3100 | -5.064 | 0.516 | 53.21% |
| 20260303 | green | 98 | -34.85 | -0.3556 | -3.503 | 0.472 | 51.02% |
| 20260304 | green | 6 | -2.26 | -0.3760 | -0.841 | 0.453 | 50.00% |
| 20260305 | red | 88 | -29.59 | -0.3362 | -3.152 | 0.488 | 52.27% |
| 20260306 | red | 313 | -139.19 | -0.4447 | -7.947 | 0.391 | 45.69% |
| 20260309 | green | 49 | -19.42 | -0.3964 | -2.747 | 0.435 | 48.98% |
| 20260310 | red | 3 | -2.13 | -0.7093 | -1.064 | 0.227 | 33.33% |
| 20260311 | red | 4 | 0.50 | 0.1240 | 0.248 | 1.360 | 75.00% |
| 20260312 | red | 34 | -20.78 | -0.6113 | -3.613 | 0.281 | 38.24% |
| 20260313 | red | 15 | -0.64 | -0.0427 | -0.169 | 0.907 | 66.67% |
