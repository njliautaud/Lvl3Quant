# ETF rotation v1 — sector SPDR cross-sectional ridge

**Config**: hold=10d, long top-2, short bot-2, txn=5.0bps, vol_target=0.15, lev_clip=[0.25,2.00]

**HONEST DEPLOY GATE**: median per-fold Calmar **1.64** (need ≥ 1.0), worst-fold MaxDD **-16.1%** (need > -25%), folds passing Calmar≥1: **6/10** (need ≥ 50%).
**RESULT: PASS**

_(For reference — pooled-PnL stats, NOT the deploy gate)_: Sharpe 0.78 | pooled-Calmar 0.68 | CAGR 10.2% | pooled-MaxDD -15.0% | WR 41.1%
**Folds**: 10 | **Annualised turnover (legs/yr)**: 398.4

## Per-fold OOT

| OOT Window | Sharpe | Calmar | CAGR | MaxDD | WR | n_rebal | alpha |
|---|---|---|---|---|---|---|---|
| 2023-06-07 → 2023-12-07 | 0.99 | 2.73 | 12.3% | -4.5% | 42.1% | 8 | 0.1 |
| 2023-09-07 → 2024-03-07 | 1.03 | 2.14 | 11.0% | -5.1% | 41.1% | 8 | 0.1 |
| 2023-12-07 → 2024-06-07 | -3.19 | -2.38 | -38.2% | -16.1% | 35.8% | 11 | 0.1 |
| 2024-03-07 → 2024-09-07 | 0.14 | 0.17 | 1.0% | -5.9% | 36.0% | 12 | 0.1 |
| 2024-06-07 → 2024-12-07 | 2.28 | 8.37 | 41.5% | -5.0% | 43.3% | 12 | 0.1 |
| 2024-09-07 → 2025-03-07 | 0.77 | 1.37 | 10.7% | -7.9% | 42.4% | 10 | 0.1 |
| 2024-12-07 → 2025-06-07 | 5.65 | 46.46 | 66.9% | -1.4% | 55.0% | 6 | 0.1 |
| 2025-03-07 → 2025-09-07 | -0.66 | -1.01 | -9.5% | -9.4% | 44.1% | 8 | 0.1 |
| 2025-06-07 → 2025-12-07 | 1.20 | 1.90 | 17.0% | -8.9% | 43.8% | 13 | 0.1 |
| 2025-09-07 → 2026-03-07 | -0.44 | -0.82 | -8.0% | -9.8% | 38.9% | 11 | 0.1 |

## Feature importance (avg coef across folds)

| Feature | Avg coef |
|---|---|
| `rel_strength_spy` | -0.0095 |
| `fp_beat_rate_4q_sec` | -0.0077 |
| `ret_20d` | +0.0074 |
| `fp_eps_yoy_growth_sec` | -0.0059 |
| `fp_margin_trend_4q_sec` | +0.0050 |
| `fp_debt_to_equity_sec` | -0.0040 |
| `fp_fcf_yield_sec` | +0.0036 |
| `fp_current_ratio_sec` | -0.0030 |
| `fp_rev_yoy_growth_sec` | -0.0024 |
| `fp_roe_sec` | +0.0023 |
| `fp_gross_margin_sec` | +0.0020 |
| `ret_60d` | -0.0020 |
| `fp_ni_yoy_growth_sec` | +0.0018 |
| `momentum_cross_20_60` | +0.0017 |
| `rs_rank_among_sectors` | +0.0016 |
