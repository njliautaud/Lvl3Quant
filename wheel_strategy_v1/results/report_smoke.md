# Wheel Strategy v1 — Report (smoke)

Pareto front size: 13

| Tier | Yield % | MaxDD % | Worst Month % | Sortino | Trades | Assignments | Avg DTE | Avg Delta |
|------|---------|---------|---------------|---------|--------|-------------|---------|-----------|
| Conservative | 1.68 | 0.68 | -0.27 | 1.34 | 43 | 0 | 22.0 | 0.132 |
| Balanced | 7.12 | 1.81 | -0.39 | 2.49 | 90 | 0 | 22.0 | 0.385 |
| Aggressive | 7.23 | 2.19 | -0.18 | 1.95 | 63 | 0 | 29.0 | 0.385 |

## Conservative — config

```
put_delta_target = 0.13236742809799135
call_delta_target = 0.10942875570602029
dte_min = 16
dte_max = 29
profit_take_pct = 0.5042853455823514
roll_dte_trigger = 13
max_concurrent_names = 11
sector_cap_pct = 0.2525957307589074
vix_max_gate = 41.44428984900671
naaim_min_gate = -65.68027517625663
fund_score_floor = 7.697990982879299
```

### Conservative — 5 example trades

| Open | Close | Ticker | Kind | Strike | DTE | Delta | PnL |
|------|-------|--------|------|--------|-----|-------|-----|
| 2024-03-18 | 2024-03-21 | NVDA | CSP | 73.69 | 22 | 0.132 | $217.07 |
| 2024-08-20 | 2024-08-27 | NVDA | CSP | 103.52 | 22 | 0.132 | $123.87 |
| 2024-02-28 | 2024-03-01 | NVDA | CSP | 65.77 | 22 | 0.132 | $108.33 |
| 2024-08-09 | 2024-08-13 | NVDA | CSP | 85.76 | 22 | 0.132 | $104.82 |
| 2024-03-04 | 2024-03-06 | NVDA | CSP | 72.38 | 22 | 0.132 | $103.80 |

## Balanced — config

```
put_delta_target = 0.384665661176
call_delta_target = 0.3896896099223679
dte_min = 13
dte_max = 30
profit_take_pct = 0.34983689107917987
roll_dte_trigger = 9
max_concurrent_names = 29
sector_cap_pct = 0.3231090082225676
vix_max_gate = 39.182820833586305
naaim_min_gate = -4.847298294795436
fund_score_floor = 59.78999788110851
```

### Balanced — 5 example trades

| Open | Close | Ticker | Kind | Strike | DTE | Delta | PnL |
|------|-------|--------|------|--------|-----|-------|-----|
| 2024-08-14 | 2024-08-19 | NVDA | CSP | 113.65 | 22 | 0.385 | $446.76 |
| 2024-08-12 | 2024-08-14 | NVDA | CSP | 104.94 | 22 | 0.385 | $311.76 |
| 2024-07-05 | 2024-07-10 | AAPL | CSP | 220.27 | 22 | 0.385 | $309.20 |
| 2024-08-22 | 2024-08-27 | NVDA | CSP | 119.18 | 22 | 0.385 | $301.99 |
| 2024-03-19 | 2024-03-21 | NVDA | CSP | 86.33 | 22 | 0.385 | $286.02 |

## Aggressive — config

```
put_delta_target = 0.384665661176
call_delta_target = 0.17255568727013554
dte_min = 18
dte_max = 40
profit_take_pct = 0.34983689107917987
roll_dte_trigger = 9
max_concurrent_names = 29
sector_cap_pct = 0.3231090082225676
vix_max_gate = 27.077493680933905
naaim_min_gate = -4.847298294795436
fund_score_floor = 59.78999788110851
```

### Aggressive — 5 example trades

| Open | Close | Ticker | Kind | Strike | DTE | Delta | PnL |
|------|-------|--------|------|--------|-----|-------|-----|
| 2024-08-13 | 2024-08-16 | NVDA | CSP | 111.56 | 29 | 0.385 | $364.09 |
| 2024-08-08 | 2024-08-13 | NVDA | CSP | 100.92 | 29 | 0.385 | $361.74 |
| 2024-09-24 | 2024-10-03 | NVDA | CSP | 116.77 | 29 | 0.385 | $361.46 |
| 2024-08-16 | 2024-08-23 | NVDA | CSP | 119.73 | 29 | 0.385 | $328.18 |
| 2024-06-05 | 2024-06-13 | NVDA | CSP | 118.88 | 29 | 0.385 | $294.68 |