# ETF rotation v1 — sector SPDR cross-sectional ridge

**Config**: hold=28d, long top-2, long-only, txn=5.0bps, vol_target=0.15, lev_clip=[0.25,2.00]

**HONEST DEPLOY GATE**: median per-fold Calmar **11.62** (need ≥ 1.0), worst-fold MaxDD **-6.2%** (need > -25%), folds passing Calmar≥1: **9/10** (need ≥ 50%).
**RESULT: PASS**

_(For reference — pooled-PnL stats, NOT the deploy gate)_: Sharpe 2.61 | pooled-Calmar 4.02 | CAGR 36.6% | pooled-MaxDD -9.1% | WR 48.7%
**Folds**: 10 | **Annualised turnover (legs/yr)**: 75.2

## Per-fold OOT

| OOT Window | Sharpe | Calmar | CAGR | MaxDD | WR | n_rebal | alpha |
|---|---|---|---|---|---|---|---|
| 2023-06-07 → 2023-12-07 | 3.21 | 13.26 | 38.4% | -2.9% | 49.3% | 4 | 0.1 |
| 2023-09-07 → 2024-03-07 | 2.87 | 11.83 | 42.8% | -3.6% | 54.7% | 3 | 0.1 |
| 2023-12-07 → 2024-06-07 | 1.30 | 4.15 | 19.1% | -4.6% | 46.3% | 5 | 0.1 |
| 2024-03-07 → 2024-09-07 | 2.80 | 12.57 | 46.7% | -3.7% | 52.6% | 4 | 0.1 |
| 2024-06-07 → 2024-12-07 | 3.66 | 16.47 | 50.0% | -3.0% | 54.2% | 5 | 0.1 |
| 2024-09-07 → 2025-03-07 | 0.49 | 1.37 | 4.8% | -3.5% | 38.5% | 3 | 0.1 |
| 2024-12-07 → 2025-06-07 | 5.90 | 33.25 | 89.7% | -2.7% | 54.0% | 3 | 0.1 |
| 2025-03-07 → 2025-09-07 | 1.01 | 3.46 | 12.3% | -3.6% | 50.9% | 3 | 0.1 |
| 2025-06-07 → 2025-12-07 | 0.37 | 0.68 | 4.2% | -6.2% | 47.4% | 5 | 0.1 |
| 2025-09-07 → 2026-03-07 | 3.53 | 11.41 | 61.7% | -5.4% | 50.5% | 5 | 0.1 |

## Feature importance (avg coef across folds)

| Feature | Avg coef |
|---|---|
| `fp_beat_rate_4q_sec` | -0.0183 |
| `fp_eps_yoy_growth_sec` | -0.0169 |
| `fp_net_margin_sec` | +0.0166 |
| `fp_margin_trend_4q_sec` | +0.0136 |
| `fp_ebitda_margin_sec` | -0.0123 |
| `fp_current_ratio_sec` | -0.0102 |
| `rel_strength_spy` | -0.0093 |
| `fp_gross_margin_sec` | +0.0082 |
| `fp_fcf_yield_sec` | +0.0082 |
| `fp_debt_to_equity_sec` | -0.0073 |
| `fp_ni_yoy_growth_sec` | +0.0067 |
| `fp_rev_yoy_growth_sec` | -0.0053 |
| `ret_60d` | -0.0048 |
| `momentum_cross_20_60` | +0.0044 |
| `rs_rank_among_sectors` | +0.0040 |
