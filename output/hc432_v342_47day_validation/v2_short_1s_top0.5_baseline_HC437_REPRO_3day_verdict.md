# HC #432 Verdict — v2_short_1s_top0.5_baseline_HC437_REPRO_3day

- Dates present: **3 / 3**
- Config: side=short, horizon=1s, conf_band=top0.5, TP=1.0, SL=0.5, hold=1.5s, cancel=1.0s, order=passive_at_touch

## Overall (FIFO realized)
| metric | value |
|---|---|
| n_fills | 160 |
| sum_net_ticks | -43.16 |
| mean_net_tk/fill | -0.2697 |
| Sharpe (sqrt-N) | -4.634 |
| Sortino | -7642815007195892.000 |
| PF | 0.481 |
| WR | 40.62% |
| day_concentration | 0.576 |

## Per-regime
| regime | n | sum_tk | mean_tk | Sharpe | PF | WR |
|---|---:|---:|---:|---:|---:|---:|
| green | 68 | -16.57 | -0.2436 | -2.711 | 0.515 | 42.65% |
| red | 92 | -26.59 | -0.2890 | -3.766 | 0.458 | 39.13% |
| flat | 0 | 0.00 | 0.0000 | 0.000 | 0.000 | 0.00% |

## HC #428 R1 — regime-agnostic OOT
- ratio |Sh_g − Sh_r| / max = **0.280** (threshold ≤ 0.50)
- day_concentration = **0.576** (cap ≤ 0.70)
- **R1 verdict: PASS**

## HC #428 R2 — MFE within horizon
| check | value | pass |
|---|---|---|
| TP ≤ p90_MFE@h | 1.0 vs 69600.000 | True |
| hold ≤ 1.5h | 1.5s vs 1.5s | True |
| cancel ≤ h | 1.0s vs 1.0s | True |
- **R2 verdict: PASS**

## Combined
- **FAIL** (R1=P, R2=P, Sharpe>0=F)

## Per-day
| date | regime | n | sum_tk | mean_tk | Sharpe | PF | WR |
|---|---|---:|---:|---:|---:|---:|---:|
| 20260224 | green | 68 | -16.57 | -0.2436 | -2.711 | 0.515 | 42.65% |
| 20260226 | red | 90 | -24.84 | -0.2760 | -3.543 | 0.475 | 40.00% |
| 20260301 | red | 2 | -1.75 | -0.8760 | 0.000 | 0.000 | 0.00% |
