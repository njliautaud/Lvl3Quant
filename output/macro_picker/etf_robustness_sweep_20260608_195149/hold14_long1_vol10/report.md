# ETF rotation v1 — sector SPDR cross-sectional ridge

**Config**: hold=14d, long top-1, long-only, txn=5.0bps, vol_target=0.10, lev_clip=[0.25,2.00]

**HONEST DEPLOY GATE**: median per-fold Calmar **4.25** (need ≥ 1.0), worst-fold MaxDD **-5.3%** (need > -25%), folds passing Calmar≥1: **7/10** (need ≥ 50%).
**RESULT: PASS**

_(For reference — pooled-PnL stats, NOT the deploy gate)_: Sharpe 1.65 | pooled-Calmar 2.03 | CAGR 14.7% | pooled-MaxDD -7.2% | WR 48.4%
**Folds**: 10 | **Annualised turnover (legs/yr)**: 66.9

## Per-fold OOT

| OOT Window | Sharpe | Calmar | CAGR | MaxDD | WR | n_rebal | alpha |
|---|---|---|---|---|---|---|---|
| 2023-06-07 → 2023-12-07 | 2.10 | 4.99 | 14.2% | -2.8% | 47.6% | 7 | 0.1 |
| 2023-09-07 → 2024-03-07 | -1.47 | -2.51 | -11.0% | -4.4% | 46.0% | 6 | 0.1 |
| 2023-12-07 → 2024-06-07 | -0.08 | -0.28 | -1.3% | -4.5% | 39.3% | 8 | 0.1 |
| 2024-03-07 → 2024-09-07 | 0.13 | 0.18 | 0.8% | -4.5% | 45.3% | 8 | 0.1 |
| 2024-06-07 → 2024-12-07 | 2.61 | 7.16 | 24.4% | -3.4% | 51.8% | 9 | 0.1 |
| 2024-09-07 → 2025-03-07 | 3.16 | 15.08 | 26.9% | -1.8% | 52.0% | 7 | 0.1 |
| 2024-12-07 → 2025-06-07 | 4.16 | 20.39 | 33.5% | -1.6% | 56.6% | 5 | 0.1 |
| 2025-03-07 → 2025-09-07 | 1.35 | 3.52 | 10.8% | -3.1% | 52.8% | 5 | 0.1 |
| 2025-06-07 → 2025-12-07 | 1.80 | 3.26 | 16.3% | -5.0% | 50.5% | 9 | 0.1 |
| 2025-09-07 → 2026-03-07 | 3.05 | 6.20 | 32.6% | -5.3% | 44.8% | 9 | 0.1 |

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
