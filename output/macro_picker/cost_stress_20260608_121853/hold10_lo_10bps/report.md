# ETF rotation v1 — sector SPDR cross-sectional ridge

**Config**: hold=10d, long top-2, long-only, txn=10.0bps, vol_target=0.15, lev_clip=[0.25,2.00]

**HONEST DEPLOY GATE**: median per-fold Calmar **3.17** (need ≥ 1.0), worst-fold MaxDD **-10.4%** (need > -25%), folds passing Calmar≥1: **7/10** (need ≥ 50%).
**RESULT: PASS**

_(For reference — pooled-PnL stats, NOT the deploy gate)_: Sharpe 1.78 | pooled-Calmar 3.47 | CAGR 29.2% | pooled-MaxDD -8.4% | WR 51.6%
**Folds**: 10 | **Annualised turnover (legs/yr)**: 205.1

## Per-fold OOT

| OOT Window | Sharpe | Calmar | CAGR | MaxDD | WR | n_rebal | alpha |
|---|---|---|---|---|---|---|---|
| 2023-06-07 → 2023-12-07 | 1.63 | 3.44 | 25.0% | -7.3% | 47.0% | 13 | 0.1 |
| 2023-09-07 → 2024-03-07 | 1.55 | 2.89 | 21.0% | -7.3% | 52.7% | 13 | 0.1 |
| 2023-12-07 → 2024-06-07 | 0.05 | -0.05 | -0.3% | -6.8% | 48.4% | 13 | 0.1 |
| 2024-03-07 → 2024-09-07 | 0.01 | -0.16 | -1.1% | -6.9% | 46.9% | 13 | 0.1 |
| 2024-06-07 → 2024-12-07 | 2.64 | 7.63 | 47.2% | -6.2% | 52.8% | 13 | 0.1 |
| 2024-09-07 → 2025-03-07 | 2.42 | 5.67 | 37.3% | -6.6% | 52.4% | 13 | 0.1 |
| 2024-12-07 → 2025-06-07 | 0.33 | 0.44 | 4.6% | -10.4% | 49.5% | 13 | 0.1 |
| 2025-03-07 → 2025-09-07 | 1.75 | 8.33 | 32.4% | -3.9% | 49.0% | 13 | 0.1 |
| 2025-06-07 → 2025-12-07 | 0.60 | 1.12 | 7.0% | -6.2% | 45.8% | 13 | 0.1 |
| 2025-09-07 → 2026-03-07 | 2.04 | 5.31 | 34.2% | -6.4% | 53.5% | 13 | 0.1 |

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
