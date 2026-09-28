## SUMMARY

| Config | n_trades | Sharpe_ovr | Sharpe_green | Sharpe_red | day_conc | R1 | R2 |
|--------|---------:|----------:|-------------:|-----------:|--------:|:--:|:--:|
| v3.4.2 trial 1554 · 30s short | 218 | 10.90 | 9.45 | 37.34 | 0.23 | FAIL | PASS |
| v3.4.2 trial 1422 · 5s short | 387 | 12.52 | 12.01 | 16.86 | 0.21 | PASS | FAIL |
| v3.3 trial 2831 · 10s short | 89 | 20.24 | 23.30 | 15.21 | 0.41 | PASS | FAIL |

# HC #428 R1: Top-3 Stratified OOT Validation

_Generated: 2026-05-19 08:35 ET_

## Data Availability (CRITICAL)

Predictions NPZs available cover **17 OOT days** for v3.4.2 (5d + 12d extended)
and **5 OOT days** for v3.3. There are NO predictions for 20260316..20260429.
Re-validation is bounded by this data gap; running inference to fill the gap
requires GPU access (Razer/Neptune) which are reserved by current directives.

## Regime Classification of OOT Days

| Date | Regime | Pct Return |
|------|--------|-----------:|
| 20260223 | RED | -0.7781% |
| 20260224 | GREEN | 0.8105% |
| 20260225 | GREEN | 0.4184% |
| 20260226 | RED | -0.5674% |
| 20260227 | GREEN | 0.4741% |
| 20260301 | nan | nan% |
| 20260302 | GREEN | 1.1158% |
| 20260303 | GREEN | 0.7862% |
| 20260304 | GREEN | 0.5008% |
| 20260305 | FLAT | -0.106% |
| 20260306 | FLAT | -0.1444% |
| 20260308 | nan | nan% |
| 20260309 | GREEN | 1.7997% |
| 20260310 | FLAT | -0.103% |
| 20260311 | FLAT | -0.1877% |
| 20260312 | RED | -0.747% |
| 20260313 | RED | -1.0363% |
| 20260315 | nan | nan% |
| 20260316 | FLAT | 0.1038% |
| 20260317 | RED | -0.2357% |
| 20260318 | RED | -1.0265% |
| 20260319 | GREEN | 0.445% |
| 20260401 | GREEN | 0.2006% |
| 20260402 | GREEN | 1.4592% |
| 20260403 | nan | nan% |
| 20260405 | nan | nan% |
| 20260406 | GREEN | 0.4719% |
| 20260407 | GREEN | 0.3922% |
| 20260408 | FLAT | -0.0879% |
| 20260409 | GREEN | 0.7192% |
| 20260410 | RED | -0.2836% |
| 20260412 | nan | nan% |
| 20260413 | GREEN | 1.2656% |
| 20260414 | GREEN | 0.9837% |
| 20260415 | GREEN | 0.6558% |
| 20260416 | FLAT | 0.0707% |
| 20260417 | GREEN | 0.5652% |
| 20260419 | nan | nan% |
| 20260420 | FLAT | 0.0% |
| 20260421 | RED | -0.8656% |
| 20260422 | GREEN | 0.2797% |
| 20260423 | FLAT | -0.1468% |
| 20260424 | GREEN | 0.4468% |
| 20260426 | nan | nan% |
| 20260427 | GREEN | 0.2922% |
| 20260428 | FLAT | -0.007% |
| 20260429 | FLAT | -0.0175% |

## v3.4.2 trial 1554 · 30s short

**Config**: side=short, horizon=30s, order_type=passive_at_touch_plus_2, cancel=38 evals, hold=2.14s, conf_thr=0.08575

**Dates evaluated**: 17 days (20260223..20260315)

| Metric | Value |
|--------|------:|
| n_trades | 218 |
| total_net_ticks | 382.03 |
| sharpe_overall | 10.90 |
| sharpe_green | 9.45 |
| sharpe_red | 37.34 |
| sharpe_flat | 16.30 |
| day_conc | 0.23 |
| R1 ratio |ΔSharpe|/max | 0.75 |
| **R1 pass (≤0.50)** | **False** |

Per-regime breakdown:
```
       count      sum     mean       std
FLAT       9   14.616  1.62400  1.581139
GREEN    174  289.576  1.66423  2.796280
RED       35   77.840  2.22400  0.945578
```

## v3.4.2 trial 1422 · 5s short

**Config**: side=short, horizon=5s, order_type=passive_at_touch_plus_2, cancel=80 evals, hold=1.10s, conf_thr=0.08300

**Dates evaluated**: 17 days (20260223..20260315)

| Metric | Value |
|--------|------:|
| n_trades | 387 |
| total_net_ticks | 786.49 |
| sharpe_overall | 12.52 |
| sharpe_green | 12.01 |
| sharpe_red | 16.86 |
| sharpe_flat | 12.62 |
| day_conc | 0.21 |
| R1 ratio |ΔSharpe|/max | 0.29 |
| **R1 pass (≤0.50)** | **True** |

Per-regime breakdown:
```
       count      sum      mean       std
FLAT      22   45.728  2.078545  2.613650
GREEN    319  637.056  1.997041  2.638973
RED       46  103.704  2.254435  2.122459
```

## v3.3 trial 2831 · 10s short

**Config**: side=short, horizon=10s, order_type=passive_at_touch_plus_2, cancel=57 evals, hold=2.37s, conf_thr=0.06210

**Dates evaluated**: 5 days (20260223..20260227)

| Metric | Value |
|--------|------:|
| n_trades | 89 |
| total_net_ticks | 200.54 |
| sharpe_overall | 20.24 |
| sharpe_green | 23.30 |
| sharpe_red | 15.21 |
| sharpe_flat | n/a |
| day_conc | 0.41 |
| R1 ratio |ΔSharpe|/max | 0.35 |
| **R1 pass (≤0.50)** | **True** |

Per-regime breakdown:
```
       count      sum      mean       std
GREEN     63  146.312  2.322413  1.582515
RED       26   54.224  2.085538  2.176801
```
