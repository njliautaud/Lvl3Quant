# ETF rotation v1 — sector SPDR cross-sectional ridge

**Config**: hold=10d, long top-2, short bot-2, txn=5bps, vol_target=0.15, lev_clip=[0.25,2.00]

**HONEST DEPLOY GATE**: median per-fold Calmar **0.99** (need ≥ 1.0), worst-fold MaxDD **-13.3%** (need > -25%), folds passing Calmar≥1: **4/10** (need ≥ 50%).
**RESULT: FAIL**

_(For reference — pooled-PnL stats, NOT the deploy gate)_: Sharpe 1.19 | pooled-Calmar 1.42 | CAGR 17.6% | pooled-MaxDD -12.4% | WR 46.0%
**Folds**: 10 | **Annualised turnover (legs/yr)**: 410.1

## Per-fold OOT

| OOT Window | Sharpe | Calmar | CAGR | MaxDD | WR | n_rebal | alpha |
|---|---|---|---|---|---|---|---|
| 2023-06-07 → 2023-12-07 | 2.07 | 4.91 | 30.0% | -6.1% | 47.0% | 13 | 0.1 |
| 2023-09-07 → 2024-03-07 | 1.09 | 2.65 | 15.9% | -6.0% | 42.9% | 13 | 0.1 |
| 2023-12-07 → 2024-06-07 | -2.24 | -2.08 | -27.8% | -13.3% | 36.8% | 13 | 0.1 |
| 2024-03-07 → 2024-09-07 | -0.21 | -0.42 | -2.9% | -6.9% | 41.7% | 13 | 0.1 |
| 2024-06-07 → 2024-12-07 | 0.50 | 0.73 | 7.6% | -10.5% | 45.3% | 13 | 0.1 |
| 2024-09-07 → 2025-03-07 | 1.66 | 3.63 | 25.9% | -7.2% | 48.5% | 13 | 0.1 |
| 2024-12-07 → 2025-06-07 | 1.17 | 2.57 | 18.2% | -7.1% | 49.5% | 13 | 0.1 |
| 2025-03-07 → 2025-09-07 | 0.59 | 0.83 | 7.8% | -9.4% | 44.9% | 13 | 0.1 |
| 2025-06-07 → 2025-12-07 | 0.68 | 1.00 | 9.1% | -9.1% | 44.8% | 13 | 0.1 |
| 2025-09-07 → 2026-03-07 | 0.71 | 0.98 | 10.8% | -11.1% | 45.5% | 13 | 0.1 |

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
