# ETF rotation v1 — sector SPDR cross-sectional ridge

**Config**: hold=14d, long top-2, long-only, txn=5.0bps, vol_target=0.20, lev_clip=[0.25,2.00]

**HONEST DEPLOY GATE**: median per-fold Calmar **5.68** (need ≥ 1.0), worst-fold MaxDD **-10.1%** (need > -25%), folds passing Calmar≥1: **9/10** (need ≥ 50%).
**RESULT: PASS**

_(For reference — pooled-PnL stats, NOT the deploy gate)_: Sharpe 1.90 | pooled-Calmar 2.73 | CAGR 34.5% | pooled-MaxDD -12.6% | WR 47.3%
**Folds**: 10 | **Annualised turnover (legs/yr)**: 133.8

## Per-fold OOT

| OOT Window | Sharpe | Calmar | CAGR | MaxDD | WR | n_rebal | alpha |
|---|---|---|---|---|---|---|---|
| 2023-06-07 → 2023-12-07 | 0.65 | 1.07 | 8.3% | -7.8% | 44.4% | 7 | 0.1 |
| 2023-09-07 → 2024-03-07 | 1.65 | 5.03 | 28.2% | -5.6% | 52.4% | 6 | 0.1 |
| 2023-12-07 → 2024-06-07 | 0.10 | -0.00 | -0.0% | -10.1% | 41.7% | 8 | 0.1 |
| 2024-03-07 → 2024-09-07 | 1.45 | 4.56 | 26.1% | -5.7% | 45.3% | 8 | 0.1 |
| 2024-06-07 → 2024-12-07 | 3.13 | 12.79 | 63.6% | -5.0% | 54.1% | 9 | 0.1 |
| 2024-09-07 → 2025-03-07 | 3.01 | 15.04 | 42.1% | -2.8% | 44.0% | 7 | 0.1 |
| 2024-12-07 → 2025-06-07 | 3.16 | 13.47 | 57.7% | -4.3% | 47.2% | 5 | 0.1 |
| 2025-03-07 → 2025-09-07 | 0.83 | 1.43 | 11.5% | -8.0% | 50.9% | 5 | 0.1 |
| 2025-06-07 → 2025-12-07 | 2.45 | 6.32 | 52.3% | -8.3% | 51.6% | 9 | 0.1 |
| 2025-09-07 → 2026-03-07 | 3.82 | 13.80 | 99.5% | -7.2% | 49.0% | 9 | 0.1 |

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
