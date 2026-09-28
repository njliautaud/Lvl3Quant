# ETF rotation v1 — sector SPDR cross-sectional ridge

**Config**: hold=10d, long top-2, long-only, txn=5bps, vol_target=0.15, lev_clip=[0.25,2.00]

**HONEST DEPLOY GATE**: median per-fold Calmar **3.60** (need ≥ 1.0), worst-fold MaxDD **-10.7%** (need > -25%), folds passing Calmar≥1: **7/10** (need ≥ 50%).
**RESULT: PASS**

_(For reference — pooled-PnL stats, NOT the deploy gate)_: Sharpe 1.65 | pooled-Calmar 2.98 | CAGR 26.7% | pooled-MaxDD -9.0% | WR 50.4%
**Folds**: 10 | **Annualised turnover (legs/yr)**: 205.1

## Per-fold OOT

| OOT Window | Sharpe | Calmar | CAGR | MaxDD | WR | n_rebal | alpha |
|---|---|---|---|---|---|---|---|
| 2023-06-07 → 2023-12-07 | 1.76 | 3.91 | 27.3% | -7.0% | 47.0% | 13 | 0.1 |
| 2023-09-07 → 2024-03-07 | 1.71 | 3.28 | 23.5% | -7.1% | 52.7% | 13 | 0.1 |
| 2023-12-07 → 2024-06-07 | 0.16 | 0.20 | 1.3% | -6.8% | 48.4% | 13 | 0.1 |
| 2024-03-07 → 2024-09-07 | -0.07 | -0.24 | -2.3% | -9.5% | 43.8% | 13 | 0.1 |
| 2024-06-07 → 2024-12-07 | 2.77 | 8.14 | 49.8% | -6.1% | 52.8% | 13 | 0.1 |
| 2024-09-07 → 2025-03-07 | 2.71 | 6.56 | 42.5% | -6.5% | 53.4% | 13 | 0.1 |
| 2024-12-07 → 2025-06-07 | 0.31 | 0.40 | 4.3% | -10.7% | 48.5% | 13 | 0.1 |
| 2025-03-07 → 2025-09-07 | 1.83 | 8.87 | 34.2% | -3.9% | 49.0% | 13 | 0.1 |
| 2025-06-07 → 2025-12-07 | 0.80 | 1.56 | 9.6% | -6.1% | 45.8% | 13 | 0.1 |
| 2025-09-07 → 2026-03-07 | 1.81 | 4.00 | 31.5% | -7.9% | 50.5% | 13 | 0.1 |

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
| `fp_current_ratio_sec` | -0.0029 |
| `fp_roe_sec` | +0.0024 |
| `fp_rev_yoy_growth_sec` | -0.0024 |
| `fp_gross_margin_sec` | +0.0021 |
| `ret_60d` | -0.0020 |
| `fp_ni_yoy_growth_sec` | +0.0017 |
| `momentum_cross_20_60` | +0.0017 |
| `rs_rank_among_sectors` | +0.0016 |
