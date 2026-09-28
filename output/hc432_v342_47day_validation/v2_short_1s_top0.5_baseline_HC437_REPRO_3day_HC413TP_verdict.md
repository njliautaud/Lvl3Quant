# HC #432 Verdict — v2_short_1s_top0.5_baseline_HC437_REPRO_3day_HC413TP

- Dates present: **3 / 3**
- Config: side=short, horizon=1s, conf_band=top0.5, TP=0.4782, SL=0.5686, hold=1.5s, cancel=1.0s, order=passive_at_touch

## Overall (FIFO realized)
| metric | value |
|---|---|
| n_fills | 160 |
| sum_net_ticks | -83.00 |
| mean_net_tk/fill | -0.5188 |
| Sharpe (sqrt-N) | -12.728 |
| Sortino | -932.341 |
| PF | 0.074 |
| WR | 40.62% |
| day_concentration | 0.569 |

## Per-regime
| regime | n | sum_tk | mean_tk | Sharpe | PF | WR |
|---|---:|---:|---:|---:|---:|---:|
| green | 68 | -33.85 | -0.4979 | -7.866 | 0.081 | 42.65% |
| red | 92 | -49.15 | -0.5342 | -9.986 | 0.070 | 39.13% |
| flat | 0 | 0.00 | 0.0000 | 0.000 | 0.000 | 0.00% |

## HC #428 R1 — regime-agnostic OOT
- ratio |Sh_g − Sh_r| / max = **0.212** (threshold ≤ 0.50)
- day_concentration = **0.569** (cap ≤ 0.70)
- **R1 verdict: PASS**

## HC #428 R2 — MFE within horizon
| check | value | pass |
|---|---|---|
| TP ≤ p90_MFE@h | 0.4782 vs 69600.000 | True |
| hold ≤ 1.5h | 1.5s vs 1.5s | True |
| cancel ≤ h | 1.0s vs 1.0s | True |
- **R2 verdict: PASS**

## Combined
- **FAIL** (R1=P, R2=P, Sharpe>0=F)

## Per-day
| date | regime | n | sum_tk | mean_tk | Sharpe | PF | WR |
|---|---|---:|---:|---:|---:|---:|---:|
| 20260224 | green | 68 | -33.85 | -0.4979 | -7.866 | 0.081 | 42.65% |
| 20260226 | red | 90 | -47.26 | -0.5251 | -9.671 | 0.072 | 40.00% |
| 20260301 | red | 2 | -1.89 | -0.9446 | 0.000 | 0.000 | 0.00% |
