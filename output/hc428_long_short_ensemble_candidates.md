# HC #428 LONG-Side Counterparts + LONG+SHORT Ensemble Candidates

_Generated: 2026-05-19_

## Top LONG-side candidates (R2-clipped up front)

| trial | horizon | n_trades | Sharpe_ovr | Sharpe_g | Sharpe_r | day_conc | R1 ratio | R1 | R2 | original_optuna_sharpe |
|------:|--------:|---------:|-----------:|---------:|---------:|---------:|---------:|:--:|:--:|-----------------------:|
| 1954 | 5s | 2327 | 24.32 | 8.48 | 9.01 | 0.163 | 0.058 | PASS | PASS | 24.34 |
| 1123 | 5s | 1454 | 24.27 | 8.48 | 10.18 | 0.151 | 0.167 | PASS | PASS | 22.73 |
| 479 | 5s | 879 | 23.63 | 9.14 | 10.33 | 0.195 | 0.115 | PASS | PASS | 32.11 |
| 2105 | 5s | 1502 | 22.09 | 8.92 | 11.83 | 0.184 | 0.246 | PASS | PASS | 27.92 |
| 1155 | 5s | 2073 | 21.29 | 8.02 | 11.27 | 0.173 | 0.288 | PASS | PASS | 21.14 |
| 2153 | 5s | 1490 | 21.20 | 8.72 | 10.89 | 0.185 | 0.199 | PASS | PASS | 21.23 |
| 2515 | 5s | 1216 | 20.88 | 8.25 | 10.35 | 0.179 | 0.202 | PASS | PASS | 22.22 |
| 1102 | 5s | 1037 | 20.40 | 8.57 | 9.54 | 0.199 | 0.101 | PASS | PASS | 20.94 |
| 2065 | 5s | 981 | 20.29 | 9.71 | 10.89 | 0.207 | 0.109 | PASS | PASS | 21.58 |
| 2277 | 5s | 1142 | 19.03 | 9.79 | 11.69 | 0.174 | 0.163 | PASS | PASS | 21.36 |

## LONG+SHORT 50/50 Ensemble Candidates (long X + t1422_R2fix)

| long_trial | n_trades | Sharpe_ovr | Sharpe_g | Sharpe_r | day_conc | R1 ratio | R1 | R2 |
|-----------:|---------:|-----------:|---------:|---------:|---------:|---------:|:--:|:--:|
| 1954 | 3365 | 31.94 | 8.95 | 9.26 | 0.129 | 0.033 | PASS | PASS |
| 479 | 1917 | 31.80 | 9.61 | 10.44 | 0.121 | 0.079 | PASS | PASS |
| 1102 | 2075 | 31.12 | 9.25 | 9.82 | 0.128 | 0.058 | PASS | PASS |
| 2515 | 2254 | 30.95 | 9.06 | 10.43 | 0.123 | 0.131 | PASS | PASS |
| 1123 | 2492 | 30.73 | 9.13 | 10.29 | 0.116 | 0.113 | PASS | PASS |
| 2153 | 2528 | 30.46 | 9.16 | 10.84 | 0.133 | 0.155 | PASS | PASS |
| 2065 | 2019 | 29.18 | 9.78 | 10.85 | 0.134 | 0.098 | PASS | PASS |
| 1155 | 3111 | 28.85 | 8.70 | 11.18 | 0.134 | 0.222 | PASS | PASS |
| 2277 | 2180 | 28.65 | 9.85 | 11.47 | 0.123 | 0.141 | PASS | PASS |
| 2105 | 2540 | 28.49 | 9.27 | 11.60 | 0.134 | 0.201 | PASS | PASS |

## Configs (top 3 long)

### Trial 1954 (5s long)
```json
{
  "trial": 1954,
  "head_horizon": "5s",
  "side": "long",
  "conf_thr": 0.1766153777590755,
  "order_type": "passive_at_touch_plus_2",
  "cancel_window": 20,
  "hold_seconds": 1.253765956789527,
  "spread_ticks": 0.9187530711112109,
  "tod_start_hour": 12,
  "tod_end_hour": 14,
  "pred_strength_min": 0.9322761253015284,
  "sigma_halt_mult": 3.575961907605784,
  "commission_ticks": 0.4650432588394875,
  "use_fifo_confluence": true,
  "fifo_confluence_head": "pred_fifo_tp8sl5_net",
  "fifo_confluence_thr_ticks": -0.2865740543904416,
  "use_horizon_confluence": false,
  "confluence_horizon": "10s",
  "source": "v3.4.2"
}
```

### Trial 1123 (5s long)
```json
{
  "trial": 1123,
  "head_horizon": "5s",
  "side": "long",
  "conf_thr": 0.1130796774096535,
  "order_type": "passive_at_touch_plus_2",
  "cancel_window": 20,
  "hold_seconds": 1.9501178081064705,
  "spread_ticks": 0.9780510697524256,
  "tod_start_hour": 14,
  "tod_end_hour": 15,
  "pred_strength_min": 0.202725259034124,
  "sigma_halt_mult": 8.259615690657052,
  "commission_ticks": 0.3799498840823806,
  "use_fifo_confluence": true,
  "fifo_confluence_head": "pred_fifo_tp8sl5_net",
  "fifo_confluence_thr_ticks": -0.2542209660793549,
  "use_horizon_confluence": true,
  "confluence_horizon": "1s",
  "source": "v3.4.2"
}
```

### Trial 479 (5s long)
```json
{
  "trial": 479,
  "head_horizon": "5s",
  "side": "long",
  "conf_thr": 0.0675831971542619,
  "order_type": "passive_at_touch_plus_2",
  "cancel_window": 20,
  "hold_seconds": 1.258247572052733,
  "spread_ticks": 0.8300516907961991,
  "tod_start_hour": 14,
  "tod_end_hour": 15,
  "pred_strength_min": 0.4481147887872781,
  "sigma_halt_mult": 4.379729472109966,
  "commission_ticks": 0.4551312737085735,
  "use_fifo_confluence": true,
  "fifo_confluence_head": "pred_fifo_tp4sl3_net",
  "fifo_confluence_thr_ticks": 0.409951286886922,
  "use_horizon_confluence": false,
  "confluence_horizon": "10s",
  "source": "v3.4.2"
}
```
