# ETF rotation v1 — sector SPDR cross-sectional ridge

**Config**: hold=10d, long top-2, short bot-2, txn=5bps, vol_target=0.15, lev_clip=[0.25,2.00]

**HONEST DEPLOY GATE**: median per-fold Calmar **2.23** (need ≥ 1.0), worst-fold MaxDD **-14.7%** (need > -25%), folds passing Calmar≥1: **6/10** (need ≥ 50%).
**RESULT: PASS**

_(For reference — pooled-PnL stats, NOT the deploy gate)_: Sharpe 1.57 | pooled-Calmar 1.99 | CAGR 24.4% | pooled-MaxDD -12.3% | WR 47.1%
**Folds**: 10 | **Annualised turnover (legs/yr)**: 410.1

## Per-fold OOT

| OOT Window | Sharpe | Calmar | CAGR | MaxDD | WR | n_rebal | alpha |
|---|---|---|---|---|---|---|---|
| 2023-06-07 → 2023-12-07 | 2.10 | 5.38 | 31.1% | -5.8% | 47.0% | 13 | 0.1 |
| 2023-09-07 → 2024-03-07 | 1.95 | 5.05 | 30.2% | -6.0% | 46.2% | 13 | 0.1 |
| 2023-12-07 → 2024-06-07 | -2.42 | -2.09 | -30.8% | -14.7% | 35.8% | 13 | 0.1 |
| 2024-03-07 → 2024-09-07 | -0.02 | -0.15 | -0.9% | -5.9% | 39.6% | 13 | 0.1 |
| 2024-06-07 → 2024-12-07 | 1.99 | 6.81 | 35.1% | -5.2% | 45.3% | 13 | 0.1 |
| 2024-09-07 → 2025-03-07 | 1.50 | 3.01 | 23.6% | -7.9% | 46.6% | 13 | 0.1 |
| 2024-12-07 → 2025-06-07 | 1.25 | 2.18 | 18.2% | -8.4% | 52.5% | 13 | 0.1 |
| 2025-03-07 → 2025-09-07 | 0.33 | 0.37 | 3.5% | -9.4% | 44.9% | 13 | 0.1 |
| 2025-06-07 → 2025-12-07 | 1.38 | 2.28 | 20.4% | -8.9% | 46.9% | 13 | 0.1 |
| 2025-09-07 → 2026-03-07 | 0.29 | 0.36 | 3.5% | -9.8% | 43.6% | 13 | 0.1 |

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
