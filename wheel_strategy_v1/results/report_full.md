# Wheel Strategy v1 — Report (full)

Pareto front size: 25

| Tier | Yield % | MaxDD % | Worst Month % | Sortino | Trades | Assignments | Avg DTE | Avg Delta |
|------|---------|---------|---------------|---------|--------|-------------|---------|-----------|
| Conservative | 6.56 | 0.73 | -0.62 | 5.14 | 2837 | 0 | nan | nan |
| Balanced | 53.19 | 3.14 | -1.91 | 5.76 | 7442 | 0 | nan | nan |
| Aggressive | 112.69 | 10.57 | -4.96 | 2.36 | 935 | 0 | 26.0 | 0.369 |

## Conservative — config

```
put_delta_target = 0.16157017803409277
call_delta_target = 0.18277676592043096
dte_min = 8
dte_max = 21
profit_take_pct = 0.2770238136748633
roll_dte_trigger = 13
max_concurrent_names = 25
sector_cap_pct = 0.1703524412567311
vix_max_gate = 39.09474711851081
naaim_min_gate = -58.8140956480605
fund_score_floor = 82.24510909700663
```

### Conservative — 5 example trades

(no trades)

## Balanced — config

```
put_delta_target = 0.32601643447595074
call_delta_target = 0.3636402874257779
dte_min = 8
dte_max = 21
profit_take_pct = 0.2770238136748633
roll_dte_trigger = 13
max_concurrent_names = 25
sector_cap_pct = 0.38936200475952404
vix_max_gate = 43.41797879556319
naaim_min_gate = -58.8140956480605
fund_score_floor = 82.24510909700663
```

### Balanced — 5 example trades

(no trades)

## Aggressive — config

```
put_delta_target = 0.36902962738792755
call_delta_target = 0.17897420395540337
dte_min = 13
dte_max = 40
profit_take_pct = 0.41899077523491807
roll_dte_trigger = 5
max_concurrent_names = 28
sector_cap_pct = 0.3786029864812399
vix_max_gate = 48.387786338424306
naaim_min_gate = -84.26047323727377
fund_score_floor = 3.9922532563751223
```

### Aggressive — 5 example trades

| Open | Close | Ticker | Kind | Strike | DTE | Delta | PnL |
|------|-------|--------|------|--------|-----|-------|-----|
| 2026-02-24 | 2026-03-04 | PYPL | CSP | 44.77 | 26 | 0.369 | $107208.69 |
| 2026-02-25 | 2026-03-04 | COIN | CSP | 174.49 | 26 | 0.369 | $99247.32 |
| 2025-10-24 | 2025-11-03 | AMD | CSP | 240.46 | 26 | 0.369 | $89988.42 |
| 2025-11-13 | 2025-11-28 | MRNA | CSP | 23.86 | 26 | 0.369 | $80551.06 |
| 2025-09-03 | 2025-09-04 | SHOP | CSP | 134.06 | 26 | 0.369 | $80430.47 |