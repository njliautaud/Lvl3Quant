# HC #428 R2-FIX Re-Validation

_Generated: 2026-05-19_

## Summary

| Config | n_trades | Sharpe_ovr | Sharpe_green | Sharpe_red | Sharpe_flat | day_conc | R1 ratio | R1 | R2 | cancel(s) | hold(s) |
|--------|---------:|-----------:|-------------:|-----------:|------------:|---------:|---------:|:--:|:--:|----------:|--------:|
| t1422_R2fix | 1038 | 19.92 | 10.24 | 10.68 | 4.54 | 0.182 | 0.041 | PASS | PASS | 5.0 | 1.10 |
| t2831_R2fix | 314 | 80.63 | 13.18 | 14.16 | 0.00 | 0.240 | 0.069 | PASS | PASS | 6.0 | 2.37 |

## R2-Fix Parameter Changes

| Config | cancel_window evals (orig→fix) | cancel_sec (orig→fix) | hold_seconds (orig→fix) |
|--------|-------------------------------:|----------------------:|-------------------------:|
| t1422 | 80 → 20 | 20.0 → 5.0 | 1.10 → 1.10 |
| t2831 | 57 → 24 | 14.25 → 6.0 | 2.37 → 2.37 |

## t1422_R2fix

- cancel_window: 20 evals (5.0s)
- hold_seconds: 1.10
- n_trades: 1038, total_net_ticks: 2079.6
- Sharpe overall: 19.92
- Per regime: GREEN 10.24 | RED 10.68 | FLAT 4.54
- day_conc: 0.182
- R1 ratio: 0.041 → PASS
- R2: PASS

Per-day breakdown:

| date | regime | n | net_ticks | mean |
|------|:------:|--:|----------:|-----:|
| 20260223 | RED | 34 | 99.2 | 2.918 |
| 20260224 | GREEN | 88 | 174.7 | 1.985 |
| 20260225 | GREEN | 88 | 142.3 | 1.617 |
| 20260226 | RED | 68 | 161.8 | 2.380 |
| 20260227 | GREEN | 27 | 90.8 | 3.365 |
| 20260301 | nan | 17 | 53.6 | 3.153 |
| 20260302 | GREEN | 169 | 375.5 | 2.222 |
| 20260303 | GREEN | 18 | 22.2 | 1.235 |
| 20260304 | GREEN | 145 | 294.9 | 2.033 |
| 20260305 | FLAT | 31 | 71.3 | 2.301 |
| 20260306 | FLAT | 112 | 84.9 | 0.758 |
| 20260309 | GREEN | 196 | 379.3 | 1.935 |
| 20260310 | FLAT | 29 | 74.1 | 2.555 |
| 20260311 | FLAT | 16 | 55.0 | 3.437 |

## t2831_R2fix

- cancel_window: 24 evals (6.0s)
- hold_seconds: 2.37
- n_trades: 314, total_net_ticks: 706.3
- Sharpe overall: 80.63
- Per regime: GREEN 13.18 | RED 14.16 | FLAT 0.00
- day_conc: 0.240
- R1 ratio: 0.069 → PASS
- R2: PASS

Per-day breakdown:

| date | regime | n | net_ticks | mean |
|------|:------:|--:|----------:|-----:|
| 20260223 | RED | 38 | 97.7 | 2.571 |
| 20260224 | GREEN | 78 | 153.5 | 1.969 |
| 20260225 | GREEN | 102 | 169.6 | 1.663 |
| 20260226 | RED | 53 | 153.6 | 2.898 |
| 20260227 | GREEN | 43 | 131.8 | 3.066 |
