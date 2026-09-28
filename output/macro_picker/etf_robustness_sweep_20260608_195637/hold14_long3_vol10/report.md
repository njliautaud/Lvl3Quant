# ETF rotation v1 — sector SPDR cross-sectional ridge

**Config**: hold=14d, long top-3, long-only, txn=5.0bps, vol_target=0.10, lev_clip=[0.25,2.00]

**HONEST DEPLOY GATE**: median per-fold Calmar **5.77** (need ≥ 1.0), worst-fold MaxDD **-6.3%** (need > -25%), folds passing Calmar≥1: **9/10** (need ≥ 50%).
**RESULT: PASS**

_(For reference — pooled-PnL stats, NOT the deploy gate)_: Sharpe 1.88 | pooled-Calmar 2.86 | CAGR 16.5% | pooled-MaxDD -5.8% | WR 46.0%
**Folds**: 10 | **Annualised turnover (legs/yr)**: 200.7

## Per-fold OOT

| OOT Window | Sharpe | Calmar | CAGR | MaxDD | WR | n_rebal | alpha |
|---|---|---|---|---|---|---|---|
| 2023-06-07 → 2023-12-07 | 2.11 | 7.76 | 15.1% | -2.0% | 46.0% | 7 | 0.1 |
| 2023-09-07 → 2024-03-07 | 2.11 | 7.11 | 19.3% | -2.7% | 55.6% | 6 | 0.1 |
| 2023-12-07 → 2024-06-07 | 0.19 | 0.24 | 1.4% | -5.9% | 40.5% | 8 | 0.1 |
| 2024-03-07 → 2024-09-07 | 1.65 | 4.82 | 15.5% | -3.2% | 44.2% | 8 | 0.1 |
| 2024-06-07 → 2024-12-07 | 3.70 | 16.12 | 34.3% | -2.1% | 57.6% | 9 | 0.1 |
| 2024-09-07 → 2025-03-07 | 0.99 | 2.40 | 6.8% | -2.9% | 38.7% | 7 | 0.1 |
| 2024-12-07 → 2025-06-07 | 2.63 | 9.68 | 18.7% | -1.9% | 43.4% | 5 | 0.1 |
| 2025-03-07 → 2025-09-07 | 0.79 | 1.75 | 5.9% | -3.4% | 45.3% | 5 | 0.1 |
| 2025-06-07 → 2025-12-07 | 2.35 | 6.71 | 24.3% | -3.6% | 48.4% | 9 | 0.1 |
| 2025-09-07 → 2026-03-07 | 2.16 | 3.71 | 23.3% | -6.3% | 45.8% | 9 | 0.1 |

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
