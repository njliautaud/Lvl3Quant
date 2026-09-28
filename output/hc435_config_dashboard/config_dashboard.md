# HC #435 — Config Performance Dashboard

_Generated: 2026-05-19 17:05 ET_  
_Source: aggregated from HC #408/#413/#415/#417/#428/#429 artifacts_  
_Costs: passive_at_touch = 0.376 tk · market = 1.376 tk (commission $4.70 + spread)_

## Leaderboard

| Config | Family | Hzn | Side | Conf | TP | SL | hold(s) | cancel(s) | order | n | days | day_conc | net/fill | WR% | Sharpe√N | Sortino | PF | MFE/MAE@conf | HC408 | HC415 | R1 | R2 | OOT | Status |
|---|---|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|
| v2_1s_short_top05 | v2 | 1s | short | top 0.5% | 2.0 | 1.0 | 1.10 | 5.0 | passive_at_touch | 639 | 25 | 0.200 | 0.274 | 85.0 | 12.77 | 707 | 2.92 | 0.956/0.569 | PASS | PASS | PASS | PASS | 25/36d | VALIDATED |
| v342_1s_long_top05 | v3.4.2 | 1s | long | top 0.5% | 1.0 | 0.5 | 1.50 | 1.0 | passive_at_touch | 1206 | 17 | n/a | 1.005 | 68.2 | n/a | n/a | n/a | 1.381/0.291 | — | — | — | PASS | 17/17d | PROVISIONAL (17-day) |
| t1422_R2fix | v3.4.2 | 5s | short | conf_thr=0.083 | 2.0 | 1.5 | 1.10 | 5.0 | passive_at_touch_plus_2 | 1038 | 14 | 0.182 | 2.003 | n/a | 19.92 | n/a | n/a | 0.879/0.886 | — | — | PASS | PASS | 14/17d | PROVISIONAL (17-day) |
| t2831_R2fix | v3.4.2 | 10s | short | conf_thr~0.06 | 3.0 | 2.0 | 2.37 | 6.0 | passive_at_touch_plus_2 | 314 | 5 | 0.240 | 2.249 | n/a | 80.63 | n/a | n/a | 0.689/1.253 | — | — | PASS | PASS | 5/5d | PROVISIONAL (5-day) |
| ens_l1954_t1422 | v3.4.2 ensemble | 5s | long+short | mixed | 2.0 | 1.5 | 1.25 | 5.0 | passive_at_touch_plus_2 | 3365 | 17 | 0.129 | n/a | n/a | 31.94 | n/a | n/a | n/a | — | — | PASS | PASS | 17/17d | PROVISIONAL (17-day) |
| v33_30s_short_top05_REJECTED | v3.3 | 30s | short | top 0.5% | n/a | n/a | n/a | n/a | n/a | 1196 | 5 | n/a | -0.299 | 51.8 | n/a | n/a | n/a | 0.077/-5.580 | — | — | — | FAIL | 5/5d | REJECTED |
| t1554_v342_30s_short | v3.4.2 | 30s | short | conf_thr=0.0858 | 3.0 | 2.5 | 2.14 | 9.5 | passive_at_touch_plus_2 | 218 | 17 | 0.230 | 1.752 | n/a | 10.90 | n/a | n/a | 0.579/-3.763 | — | — | FAIL | PASS | 17/17d | REJECTED |

## R1 Stratified Sharpe (Green vs Red days)

| Config | Sharpe_green | Sharpe_red | |Δ|/max | R1 (≤0.50) |
|---|---:|---:|---:|---|
| t1422_R2fix | 10.24 | 10.68 | 0.041 | PASS |
| t2831_R2fix | 13.18 | 14.16 | 0.069 | PASS |
| ens_l1954_t1422 | 8.95 | 9.26 | 0.033 | PASS |
| t1554_v342_30s_short | 9.45 | 37.34 | 0.750 | FAIL |

## HC #428 R2 Compliance (TP ≤ p90 MFE · hold ≤ 1.5h · cancel ≤ h)

| Config | Horizon (s) | TP | 1.5*p90 proxy | TP gate | hold | 1.5h | hold gate | cancel | h | cancel gate |
|---|---:|---:|---:|---|---:|---:|---|---:|---:|---|
| v2_1s_short_top05 | 1.0 | 2.0 | 1.43 | PASS | 1.10 | 1.50 | PASS | 5.0 | 1.0 | FAIL |
| v342_1s_long_top05 | 1.0 | 1.0 | 2.07 | PASS | 1.50 | 1.50 | PASS | 1.0 | 1.0 | PASS |
| t1422_R2fix | 5.0 | 2.0 | 1.32 | PASS | 1.10 | 7.50 | PASS | 5.0 | 5.0 | PASS |
| t2831_R2fix | 10.0 | 3.0 | 1.03 | FAIL | 2.37 | 15.00 | PASS | 6.0 | 10.0 | PASS |
| ens_l1954_t1422 | 5.0 | 2.0 | n/a | PASS | 1.25 | 7.50 | PASS | 5.0 | 5.0 | PASS |
| v33_30s_short_top05_REJECTED | 30.0 | n/a | 0.12 | PASS | n/a | 45.00 | PASS | n/a | 30.0 | PASS |
| t1554_v342_30s_short | 30.0 | 3.0 | 0.87 | FAIL | 2.14 | 45.00 | PASS | 9.5 | 30.0 | PASS |

## Status Legend

- **VALIDATED** — passed full HC #408+#415+#428 R1+R2 on ≥30-day OOT
- **PROVISIONAL (N-day)** — passed available gates but on truncated OOT (data gap, inference not yet complete on full 47-day window)
- **REJECTED** — failed a binding HC gate

## Notes per config

- **v2_1s_short_top05** — 36-day OOT champion (HC #413)
- **v342_1s_long_top05** — Pre-FIFO conditional-MFE leader (HC #429). Needs FIFO+R1 audit.
- **t1422_R2fix** — R2-fix: cancel 80->20 evals, hold 1.10s. R1 PASS / R2 PASS.
- **t2831_R2fix** — R2-fix. n_fills trebled vs original — flag for review.
- **ens_l1954_t1422** — 50/50 ensemble. R1 ratio 0.033 — best regime balance. Day_conc 0.129.
- **v33_30s_short_top05_REJECTED** — Negative conditional MFE (0.077tk). Model failure mode (HC #429).
- **t1554_v342_30s_short** — R1 ratio 0.75 — regime-tailored to RED days. HC #428 R1 FAIL.