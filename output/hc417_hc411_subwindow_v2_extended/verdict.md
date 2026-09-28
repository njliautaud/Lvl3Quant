# HC #417 — Extended HC #411 sub-window stability (N=8, N=10)

Cell tested: `v2_1s_short_top05`
NPZ: `hc417_v2_full_oot_wrapped_for_hc413.npz`
MFE config: V2-NATIVE (`hc417_v2_native_mfe_matrix.csv`)
OOT: 20260306 -> 20260429 (36 dates)
Cost: passive_at_touch = 0.376 tk; canonical FIFO replay (HC #74/#377)

**Power gate**: n_fills < 10 in a sub-window = STAT-UNDERPOWERED (not a falsification). Only sub-windows with n_fills >= 10 AND net < 0 count as a true regime flip.

## Summary table

| N | all_net_pos | true_flips (n>=10, net<0) | underpowered (n<10, net<=0) | ci_pos | FALSIFIED? |
|---:|:-:|---:|---:|---:|:-:|
| 8 | no | 0 | 0 | 6/8 | no |
| 10 | no | 0 | 0 | 9/10 | no |

## Sub-window detail

### N=8

| win | dates | n_fills | n_days | net/fill | CI95lo | day_conc | pdpr | wr% | label |
|---:|---|---:|---:|---:|---:|---:|---:|---:|:-:|
| 0 | 20260306..20260312 | 59 | 5 | 0.400 | 0.273 | 0.288 | 1.000 | 88.140 | pass |
| 1 | 20260313..20260318 | 154 | 4 | 0.274 | 0.187 | 0.435 | 1.000 | 84.420 | pass |
| 2 | 20260319..20260405 | 78 | 4 | 0.136 | -0.005 | 0.462 | 1.000 | 75.640 | pass |
| 3 | 20260406..20260410 | 113 | 5 | 0.253 | 0.148 | 0.319 | 1.000 | 83.190 | pass |
| 4 | 20260412..20260415 | 84 | 3 | 0.211 | 0.089 | 0.429 | 1.000 | 82.140 | pass |
| 5 | 20260416..20260420 | 109 | 3 | 0.367 | 0.285 | 0.440 | 1.000 | 91.740 | pass |
| 6 | 20260421..20260424 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | empty |
| 7 | 20260426..20260429 | 42 | 1 | 0.298 | 0.159 | 1.000 | 1.000 | 90.480 | pass |

### N=10

| win | dates | n_fills | n_days | net/fill | CI95lo | day_conc | pdpr | wr% | label |
|---:|---|---:|---:|---:|---:|---:|---:|---:|:-:|
| 0 | 20260306..20260311 | 45 | 4 | 0.411 | 0.269 | 0.378 | 1.000 | 88.890 | pass |
| 1 | 20260312..20260316 | 51 | 3 | 0.223 | 0.051 | 0.372 | 1.000 | 78.430 | pass |
| 2 | 20260317..20260401 | 170 | 4 | 0.238 | 0.152 | 0.394 | 1.000 | 82.350 | pass |
| 3 | 20260402..20260406 | 38 | 3 | 0.241 | 0.066 | 0.632 | 1.000 | 84.210 | pass |
| 4 | 20260407..20260410 | 100 | 4 | 0.254 | 0.143 | 0.360 | 1.000 | 83.000 | pass |
| 5 | 20260412..20260415 | 84 | 3 | 0.211 | 0.089 | 0.429 | 1.000 | 82.140 | pass |
| 6 | 20260416..20260419 | 76 | 2 | 0.372 | 0.275 | 0.632 | 1.000 | 92.110 | pass |
| 7 | 20260420..20260422 | 33 | 1 | 0.355 | 0.199 | 1.000 | 1.000 | 90.910 | pass |
| 8 | 20260423..20260426 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | empty |
| 9 | 20260427..20260429 | 42 | 1 | 0.298 | 0.159 | 1.000 | 1.000 | 90.480 | pass |

## Bottom line

- **Not falsified** at N in [8, 10]. No sub-window with n_fills>=10 had net<0.
- Some sub-windows may be underpowered (n_fills<10); see power_label column.