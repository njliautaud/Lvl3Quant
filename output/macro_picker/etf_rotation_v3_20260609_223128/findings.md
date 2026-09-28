# ETF rotation v3 — macro regime gate findings

Run dir: `/home/jupiter/Lvl3Quant/output/macro_picker/etf_rotation_v3_20260609_223128`

## v1 (overlay-only baseline) vs v3-best — side by side

| Metric | v1 (overlay) | v3 (best macro gate) |
|---|---|---|
| Sharpe | -0.082 | 1.354 |
| Sortino | -0.128 | 1.662 |
| Calmar | -0.116 | 1.672 |
| MaxDD | -18.01% | -8.11% |
| CAGR | -2.09% | 13.55% |
| PF | 0.986 | 1.393 |
| WR | 38.19% | 23.54% |
| day-conc | 0.014 | 0.027 |

## Regime split (HC #428 R1)

| | v1 (overlay) | v3 (best) |
|---|---|---|
| Sharpe_green | -2.499 | 0.054 |
| Sharpe_red   | 2.483 | 2.547 |
| Sharpe_flat  | 0.755 | 2.303 |
| skew ratio   | 1.994 | 0.979 |
| skew ≤ 0.50  | False | False |

## v3 gating accounting

- macro-flat days (sat in cash): 286
- traded days: 326

## Best thresholds

- `thr_vix_chg` = 7.0
- `thr_vix_term` = 1.05
- `thr_dxy_chg` = 1.5
- `thr_disp_pct` = 10.0

## Screening table (all configs)

| axis | val | Sharpe | Sortino | Calmar | MaxDD | Sharpe_g | Sharpe_r | skew | skew_ok |
|---|---|---|---|---|---|---|---|---|---|
| thr_vix_chg | 3.0 | 0.57 | 0.77 | 0.55 | -9.6% | -1.82 | 4.34 | 1.42 | False |
| thr_vix_chg | 5.0 | 0.56 | 0.76 | 0.54 | -10.3% | -1.78 | 3.92 | 1.46 | False |
| thr_vix_chg | 7.0 | 0.78 | 1.11 | 0.83 | -10.0% | -1.69 | 4.07 | 1.41 | False |
| thr_vix_term | 1.05 | 0.56 | 0.76 | 0.54 | -10.3% | -1.78 | 3.92 | 1.46 | False |
| thr_vix_term | 1.1 | 0.56 | 0.76 | 0.54 | -10.3% | -1.78 | 3.92 | 1.46 | False |
| thr_vix_term | 1.15 | 0.56 | 0.76 | 0.54 | -10.3% | -1.78 | 3.92 | 1.46 | False |
| thr_dxy_chg | 1.5 | 1.15 | 1.38 | 1.37 | -8.1% | -0.17 | 2.12 | 1.08 | False |
| thr_dxy_chg | 2.5 | 0.56 | 0.76 | 0.54 | -10.3% | -1.78 | 3.92 | 1.46 | False |
| thr_dxy_chg | 3.5 | 0.26 | 0.37 | 0.21 | -11.2% | -1.89 | 2.68 | 1.71 | False |
| thr_disp_pct | 5.0 | -0.31 | -0.41 | -0.24 | -17.0% | -2.97 | 2.63 | 1.88 | False |
| thr_disp_pct | 10.0 | 0.56 | 0.76 | 0.54 | -10.3% | -1.78 | 3.92 | 1.46 | False |
| thr_disp_pct | 15.0 | -0.01 | -0.02 | -0.05 | -14.1% | -1.73 | 2.68 | 1.65 | False |
| combined | None | 1.35 | 1.66 | 1.67 | -8.1% | 0.05 | 2.55 | 0.98 | False |
| baseline_v1_overlay | None | -0.08 | -0.13 | -0.12 | -18.0% | -2.50 | 2.48 | 1.99 | False |

## Verdict

**REJECT**

- Sharpe ≥ 1.5: False  (got 1.35)
- regime_skew ≤ 0.50: False  (got 0.979)
- MaxDD > -25%: True  (got -8.1%)
