# ETF rotation v1 — sector SPDR cross-sectional ridge

**Config**: hold=14d, long top-2, long-only, txn=5.0bps, vol_target=0.10, lev_clip=[0.25,2.00]

**HONEST DEPLOY GATE**: median per-fold Calmar **5.25** (need ≥ 1.0), worst-fold MaxDD **-5.2%** (need > -25%), folds passing Calmar≥1: **9/10** (need ≥ 50%).
**RESULT: PASS**

_(For reference — pooled-PnL stats, NOT the deploy gate)_: Sharpe 1.89 | pooled-Calmar 2.55 | CAGR 16.3% | pooled-MaxDD -6.4% | WR 47.3%
**Folds**: 10 | **Annualised turnover (legs/yr)**: 133.8

## Per-fold OOT

| OOT Window | Sharpe | Calmar | CAGR | MaxDD | WR | n_rebal | alpha |
|---|---|---|---|---|---|---|---|
| 2023-06-07 → 2023-12-07 | 0.65 | 1.09 | 4.3% | -4.0% | 44.4% | 7 | 0.1 |
| 2023-09-07 → 2024-03-07 | 1.65 | 4.80 | 13.6% | -2.8% | 52.4% | 6 | 0.1 |
| 2023-12-07 → 2024-06-07 | 0.10 | 0.09 | 0.5% | -5.2% | 41.7% | 8 | 0.1 |
| 2024-03-07 → 2024-09-07 | 1.45 | 4.42 | 12.7% | -2.9% | 45.3% | 8 | 0.1 |
| 2024-06-07 → 2024-12-07 | 3.13 | 11.31 | 28.3% | -2.5% | 54.1% | 9 | 0.1 |
| 2024-09-07 → 2025-03-07 | 3.00 | 13.88 | 19.5% | -1.4% | 44.0% | 7 | 0.1 |
| 2024-12-07 → 2025-06-07 | 3.16 | 12.28 | 26.4% | -2.2% | 47.2% | 5 | 0.1 |
| 2025-03-07 → 2025-09-07 | 0.83 | 1.44 | 5.8% | -4.1% | 50.9% | 5 | 0.1 |
| 2025-06-07 → 2025-12-07 | 2.45 | 5.69 | 23.9% | -4.2% | 51.6% | 9 | 0.1 |
| 2025-09-07 → 2026-03-07 | 3.66 | 9.93 | 40.4% | -4.1% | 49.0% | 9 | 0.1 |

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
