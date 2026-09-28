# ETF rotation v1 — sector SPDR cross-sectional ridge

**Config**: hold=5d, long top-2, long-only, txn=5bps, vol_target=0.15, lev_clip=[0.25,2.00]

**HONEST DEPLOY GATE**: median per-fold Calmar **1.57** (need ≥ 1.0), worst-fold MaxDD **-12.9%** (need > -25%), folds passing Calmar≥1: **6/10** (need ≥ 50%).
**RESULT: PASS**

_(For reference — pooled-PnL stats, NOT the deploy gate)_: Sharpe 0.68 | pooled-Calmar 0.77 | CAGR 8.4% | pooled-MaxDD -10.9% | WR 40.7%
**Folds**: 10 | **Annualised turnover (legs/yr)**: 396.1

## Per-fold OOT

| OOT Window | Sharpe | Calmar | CAGR | MaxDD | WR | n_rebal | alpha |
|---|---|---|---|---|---|---|---|
| 2023-06-07 → 2023-12-07 | 1.18 | 1.59 | 16.2% | -10.2% | 43.1% | 26 | 0.1 |
| 2023-09-07 → 2024-03-07 | 0.72 | 0.94 | 8.5% | -9.0% | 41.8% | 25 | 0.1 |
| 2023-12-07 → 2024-06-07 | -0.34 | -0.91 | -4.4% | -4.9% | 41.2% | 25 | 0.1 |
| 2024-03-07 → 2024-09-07 | -0.49 | -1.00 | -7.0% | -7.0% | 34.0% | 26 | 0.1 |
| 2024-06-07 → 2024-12-07 | 2.98 | 8.69 | 40.7% | -4.7% | 45.5% | 26 | 0.1 |
| 2024-09-07 → 2025-03-07 | 1.03 | 1.71 | 14.0% | -8.2% | 45.9% | 25 | 0.1 |
| 2024-12-07 → 2025-06-07 | -0.40 | -0.71 | -9.2% | -12.9% | 39.4% | 25 | 0.1 |
| 2025-03-07 → 2025-09-07 | 1.69 | 8.25 | 27.3% | -3.3% | 38.8% | 26 | 0.1 |
| 2025-06-07 → 2025-12-07 | 0.72 | 1.54 | 9.0% | -5.8% | 41.0% | 26 | 0.1 |
| 2025-09-07 → 2026-03-07 | 1.37 | 1.80 | 19.7% | -11.0% | 47.7% | 25 | 0.1 |

## Feature importance (avg coef across folds)

| Feature | Avg coef |
|---|---|
| `rel_strength_spy` | -0.0047 |
| `fp_beat_rate_4q_sec` | -0.0038 |
| `ret_20d` | +0.0033 |
| `fp_eps_yoy_growth_sec` | -0.0027 |
| `fp_margin_trend_4q_sec` | +0.0023 |
| `fp_debt_to_equity_sec` | -0.0019 |
| `fp_fcf_yield_sec` | +0.0014 |
| `fp_current_ratio_sec` | -0.0013 |
| `fp_roe_sec` | +0.0012 |
| `fp_gross_margin_sec` | +0.0012 |
| `fp_rev_yoy_growth_sec` | -0.0010 |
| `fp_net_margin_sec` | +0.0009 |
| `rs_rank_among_sectors` | +0.0009 |
| `fp_fcf_yoy_growth_sec` | +0.0009 |
| `momentum_cross_20_60` | +0.0009 |
