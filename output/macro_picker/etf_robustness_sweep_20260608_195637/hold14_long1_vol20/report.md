# ETF rotation v1 — sector SPDR cross-sectional ridge

**Config**: hold=14d, long top-1, long-only, txn=5.0bps, vol_target=0.20, lev_clip=[0.25,2.00]

**HONEST DEPLOY GATE**: median per-fold Calmar **4.45** (need ≥ 1.0), worst-fold MaxDD **-10.3%** (need > -25%), folds passing Calmar≥1: **7/10** (need ≥ 50%).
**RESULT: PASS**

_(For reference — pooled-PnL stats, NOT the deploy gate)_: Sharpe 1.65 | pooled-Calmar 2.16 | CAGR 30.6% | pooled-MaxDD -14.2% | WR 48.4%
**Folds**: 10 | **Annualised turnover (legs/yr)**: 66.9

## Per-fold OOT

| OOT Window | Sharpe | Calmar | CAGR | MaxDD | WR | n_rebal | alpha |
|---|---|---|---|---|---|---|---|
| 2023-06-07 → 2023-12-07 | 2.10 | 5.29 | 29.8% | -5.6% | 47.6% | 7 | 0.1 |
| 2023-09-07 → 2024-03-07 | -1.47 | -2.45 | -21.3% | -8.7% | 46.0% | 6 | 0.1 |
| 2023-12-07 → 2024-06-07 | -0.08 | -0.39 | -3.5% | -8.9% | 39.3% | 8 | 0.1 |
| 2024-03-07 → 2024-09-07 | 0.13 | 0.07 | 0.6% | -8.9% | 45.3% | 8 | 0.1 |
| 2024-06-07 → 2024-12-07 | 2.61 | 7.90 | 53.7% | -6.8% | 51.8% | 9 | 0.1 |
| 2024-09-07 → 2025-03-07 | 3.16 | 16.86 | 60.0% | -3.6% | 52.0% | 7 | 0.1 |
| 2024-12-07 → 2025-06-07 | 4.16 | 23.63 | 77.2% | -3.3% | 56.6% | 5 | 0.1 |
| 2025-03-07 → 2025-09-07 | 1.35 | 3.62 | 22.1% | -6.1% | 52.8% | 5 | 0.1 |
| 2025-06-07 → 2025-12-07 | 1.80 | 3.49 | 34.3% | -9.8% | 50.5% | 9 | 0.1 |
| 2025-09-07 → 2026-03-07 | 3.05 | 7.21 | 74.3% | -10.3% | 44.8% | 9 | 0.1 |

## Feature importance (avg coef across folds)

| Feature | Avg coef |
|---|---|
| `rel_strength_spy` | -0.0163 |
| `ret_20d` | +0.0139 |
| `fp_beat_rate_4q_sec` | -0.0101 |
| `fp_eps_yoy_growth_sec` | -0.0085 |
| `fp_margin_trend_4q_sec` | +0.0072 |
| `fp_debt_to_equity_sec` | -0.0059 |
| `fp_fcf_yield_sec` | +0.0049 |
| `fp_current_ratio_sec` | -0.0048 |
| `fp_gross_margin_sec` | +0.0037 |
| `fp_net_margin_sec` | +0.0033 |
| `fp_rev_yoy_growth_sec` | -0.0031 |
| `ret_60d` | -0.0029 |
| `fp_ni_yoy_growth_sec` | +0.0029 |
| `momentum_cross_20_60` | +0.0026 |
| `fp_roe_sec` | +0.0022 |
