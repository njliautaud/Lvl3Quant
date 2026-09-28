# ETF rotation v1 — sector SPDR cross-sectional ridge

**Config**: hold=14d, long top-3, long-only, txn=5.0bps, vol_target=0.20, lev_clip=[0.25,2.00]

**HONEST DEPLOY GATE**: median per-fold Calmar **6.74** (need ≥ 1.0), worst-fold MaxDD **-11.0%** (need > -25%), folds passing Calmar≥1: **9/10** (need ≥ 50%).
**RESULT: PASS**

_(For reference — pooled-PnL stats, NOT the deploy gate)_: Sharpe 1.94 | pooled-Calmar 3.22 | CAGR 35.2% | pooled-MaxDD -10.9% | WR 46.0%
**Folds**: 10 | **Annualised turnover (legs/yr)**: 200.7

## Per-fold OOT

| OOT Window | Sharpe | Calmar | CAGR | MaxDD | WR | n_rebal | alpha |
|---|---|---|---|---|---|---|---|
| 2023-06-07 → 2023-12-07 | 2.16 | 8.65 | 32.6% | -3.8% | 46.0% | 7 | 0.1 |
| 2023-09-07 → 2024-03-07 | 2.11 | 7.68 | 41.3% | -5.4% | 55.6% | 6 | 0.1 |
| 2023-12-07 → 2024-06-07 | 0.25 | 0.27 | 2.9% | -11.0% | 40.5% | 8 | 0.1 |
| 2024-03-07 → 2024-09-07 | 1.65 | 5.02 | 32.2% | -6.4% | 44.2% | 8 | 0.1 |
| 2024-06-07 → 2024-12-07 | 3.70 | 18.68 | 79.1% | -4.2% | 57.6% | 9 | 0.1 |
| 2024-09-07 → 2025-03-07 | 0.87 | 1.98 | 10.8% | -5.4% | 38.7% | 7 | 0.1 |
| 2024-12-07 → 2025-06-07 | 2.87 | 13.85 | 41.9% | -3.0% | 43.4% | 5 | 0.1 |
| 2025-03-07 → 2025-09-07 | 0.79 | 1.72 | 11.4% | -6.6% | 45.3% | 5 | 0.1 |
| 2025-06-07 → 2025-12-07 | 2.38 | 7.52 | 51.5% | -6.8% | 48.4% | 9 | 0.1 |
| 2025-09-07 → 2026-03-07 | 2.58 | 5.96 | 59.1% | -9.9% | 45.8% | 9 | 0.1 |

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
