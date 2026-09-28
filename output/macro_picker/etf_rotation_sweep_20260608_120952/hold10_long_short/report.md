# ETF rotation v1 — sector SPDR cross-sectional ridge

**Config**: hold=10d, long top-2, short bot-2, txn=5bps, vol_target=0.15, lev_clip=[0.25,2.00]

**HONEST DEPLOY GATE**: median per-fold Calmar **-1.00** (need ≥ 1.0), worst-fold MaxDD **-100.0%** (need > -25%), folds passing Calmar≥1: **0/10** (need ≥ 50%).
**RESULT: FAIL**

_(For reference — pooled-PnL stats, NOT the deploy gate)_: Sharpe 0.90 | pooled-Calmar -1.00 | CAGR -100.0% | pooled-MaxDD -100.0% | WR 45.9%
**Folds**: 10 | **Annualised turnover (legs/yr)**: 410.1

## Per-fold OOT

| OOT Window | Sharpe | Calmar | CAGR | MaxDD | WR | n_rebal | alpha |
|---|---|---|---|---|---|---|---|
| 2023-06-07 → 2023-12-07 | 1.63 | -1.00 | -100.0% | -100.0% | 48.0% | 13 | 0.1 |
| 2023-09-07 → 2024-03-07 | 0.50 | -1.00 | -100.0% | -100.0% | 42.9% | 13 | 0.1 |
| 2023-12-07 → 2024-06-07 | -1.74 | -1.00 | -100.0% | -100.0% | 36.8% | 13 | 0.1 |
| 2024-03-07 → 2024-09-07 | -0.86 | -1.00 | -100.0% | -100.0% | 41.7% | 13 | 0.1 |
| 2024-06-07 → 2024-12-07 | -0.01 | -1.00 | -100.0% | -100.0% | 45.3% | 13 | 0.1 |
| 2024-09-07 → 2025-03-07 | 1.85 | -1.00 | -100.0% | -100.0% | 48.5% | 13 | 0.1 |
| 2024-12-07 → 2025-06-07 | 0.87 | -1.00 | -100.0% | -100.0% | 47.5% | 13 | 0.1 |
| 2025-03-07 → 2025-09-07 | 0.20 | -1.00 | -100.0% | -100.0% | 43.9% | 13 | 0.1 |
| 2025-06-07 → 2025-12-07 | 0.45 | -1.00 | -100.0% | -100.0% | 44.8% | 13 | 0.1 |
| 2025-09-07 → 2026-03-07 | 0.79 | -1.00 | -100.0% | -100.0% | 45.5% | 13 | 0.1 |

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
