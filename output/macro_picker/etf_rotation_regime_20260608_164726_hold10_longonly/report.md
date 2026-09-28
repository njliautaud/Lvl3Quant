# ETF rotation v1 — sector SPDR cross-sectional ridge

**Config**: hold=10d, long top-2, long-only, txn=5.0bps, vol_target=0.15, lev_clip=[0.25,2.00]

**HONEST DEPLOY GATE**: median per-fold Calmar **8.70** (need ≥ 1.0), worst-fold MaxDD **-9.4%** (need > -25%), folds passing Calmar≥1: **9/10** (need ≥ 50%).
**RESULT: PASS**

_(For reference — pooled-PnL stats, NOT the deploy gate)_: Sharpe 2.06 | pooled-Calmar 3.78 | CAGR 28.7% | pooled-MaxDD -7.6% | WR 47.9%
**Folds**: 10 | **Annualised turnover (legs/yr)**: 199.2

## Per-fold OOT

| OOT Window | Sharpe | Calmar | CAGR | MaxDD | WR | n_rebal | alpha |
|---|---|---|---|---|---|---|---|
| 2023-06-07 → 2023-12-07 | 3.89 | 23.49 | 59.0% | -2.5% | 49.1% | 8 | 0.1 |
| 2023-09-07 → 2024-03-07 | 4.63 | 21.30 | 61.8% | -2.9% | 60.7% | 8 | 0.1 |
| 2023-12-07 → 2024-06-07 | -1.09 | -1.64 | -15.4% | -9.4% | 45.7% | 11 | 0.1 |
| 2024-03-07 → 2024-09-07 | 0.56 | 1.51 | 6.6% | -4.4% | 40.4% | 12 | 0.1 |
| 2024-06-07 → 2024-12-07 | 3.16 | 17.46 | 52.2% | -3.0% | 51.5% | 12 | 0.1 |
| 2024-09-07 → 2025-03-07 | 3.85 | 20.28 | 52.3% | -2.6% | 49.4% | 10 | 0.1 |
| 2024-12-07 → 2025-06-07 | 2.51 | 9.73 | 29.1% | -3.0% | 37.5% | 6 | 0.1 |
| 2025-03-07 → 2025-09-07 | 0.59 | 2.16 | 5.7% | -2.6% | 45.8% | 8 | 0.1 |
| 2025-06-07 → 2025-12-07 | 0.85 | 1.61 | 9.9% | -6.1% | 43.8% | 13 | 0.1 |
| 2025-09-07 → 2026-03-07 | 2.77 | 7.68 | 47.6% | -6.2% | 51.1% | 11 | 0.1 |

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
