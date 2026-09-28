# ETF rotation v1 — sector SPDR cross-sectional ridge

**Config**: hold=10d, long top-2, long-only, txn=20.0bps, vol_target=0.15, lev_clip=[0.25,2.00]

**HONEST DEPLOY GATE**: median per-fold Calmar **2.34** (need ≥ 1.0), worst-fold MaxDD **-10.9%** (need > -25%), folds passing Calmar≥1: **6/10** (need ≥ 50%).
**RESULT: PASS**

_(For reference — pooled-PnL stats, NOT the deploy gate)_: Sharpe 1.55 | pooled-Calmar 2.87 | CAGR 24.8% | pooled-MaxDD -8.6% | WR 51.6%
**Folds**: 10 | **Annualised turnover (legs/yr)**: 205.1

## Per-fold OOT

| OOT Window | Sharpe | Calmar | CAGR | MaxDD | WR | n_rebal | alpha |
|---|---|---|---|---|---|---|---|
| 2023-06-07 → 2023-12-07 | 1.38 | 2.63 | 20.7% | -7.9% | 47.0% | 13 | 0.1 |
| 2023-09-07 → 2024-03-07 | 1.22 | 2.06 | 16.2% | -7.8% | 52.7% | 13 | 0.1 |
| 2023-12-07 → 2024-06-07 | -0.19 | -0.51 | -3.5% | -6.9% | 48.4% | 13 | 0.1 |
| 2024-03-07 → 2024-09-07 | -0.25 | -0.68 | -5.2% | -7.6% | 46.9% | 13 | 0.1 |
| 2024-06-07 → 2024-12-07 | 2.39 | 6.67 | 42.0% | -6.3% | 52.8% | 13 | 0.1 |
| 2024-09-07 → 2025-03-07 | 2.12 | 4.67 | 32.1% | -6.9% | 52.4% | 13 | 0.1 |
| 2024-12-07 → 2025-06-07 | 0.17 | 0.13 | 1.4% | -10.9% | 49.5% | 13 | 0.1 |
| 2025-03-07 → 2025-09-07 | 1.59 | 7.29 | 28.9% | -4.0% | 49.0% | 13 | 0.1 |
| 2025-06-07 → 2025-12-07 | 0.30 | 0.46 | 3.0% | -6.4% | 45.8% | 13 | 0.1 |
| 2025-09-07 → 2026-03-07 | 1.75 | 4.08 | 28.5% | -7.0% | 53.5% | 13 | 0.1 |

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
