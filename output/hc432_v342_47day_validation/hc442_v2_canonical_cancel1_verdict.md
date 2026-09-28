# PRELIMINARY — HC #432 Verdict — hc442_v2_canonical_cancel1

- Dates present: **1 / 48** (waiting on Neptune)
- Config: side=short, horizon=1s, conf_band=top0.5, TP=3.0, SL=0.5, hold=1.5s, cancel=1.0s, order=passive_at_touch

## Overall (FIFO realized)
| metric | value |
|---|---|
| n_fills | 1 |
| sum_net_ticks | -0.88 |
| mean_net_tk/fill | -0.8760 |
| Sharpe (sqrt-N) | 0.000 |
| Sortino | 0.000 |
| PF | 0.000 |
| WR | 0.00% |
| day_concentration | 1.000 |

## Per-regime
| regime | n | sum_tk | mean_tk | Sharpe | PF | WR |
|---|---:|---:|---:|---:|---:|---:|
| green | 0 | 0.00 | 0.0000 | 0.000 | 0.000 | 0.00% |
| red | 1 | -0.88 | -0.8760 | 0.000 | 0.000 | 0.00% |
| flat | 0 | 0.00 | 0.0000 | 0.000 | 0.000 | 0.00% |

## HC #428 R1 — regime-agnostic OOT
- ratio |Sh_g − Sh_r| / max = **0.000** (threshold ≤ 0.50)
- day_concentration = **1.000** (cap ≤ 0.70)
- **R1 verdict: FAIL**

## HC #428 R2 — MFE within horizon
| check | value | pass |
|---|---|---|
| TP ≤ p90_MFE@h | 3.0 vs 69600.000 | True |
| hold ≤ 1.5h | 1.5s vs 1.5s | True |
| cancel ≤ h | 1.0s vs 1.0s | True |
- **R2 verdict: PASS**

## Combined
- **FAIL** (R1=F, R2=P, Sharpe>0=F)

## Per-day
| date | regime | n | sum_tk | mean_tk | Sharpe | PF | WR |
|---|---|---:|---:|---:|---:|---:|---:|
| 20260301 | red | 1 | -0.88 | -0.8760 | 0.000 | 0.000 | 0.00% |
