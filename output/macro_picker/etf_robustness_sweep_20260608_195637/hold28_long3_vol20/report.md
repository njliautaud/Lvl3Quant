# ETF rotation v1 — sector SPDR cross-sectional ridge

**Config**: hold=28d, long top-3, long-only, txn=5.0bps, vol_target=0.20, lev_clip=[0.25,2.00]

**HONEST DEPLOY GATE**: median per-fold Calmar **12.01** (need ≥ 1.0), worst-fold MaxDD **-7.4%** (need > -25%), folds passing Calmar≥1: **8/10** (need ≥ 50%).
**RESULT: PASS**

_(For reference — pooled-PnL stats, NOT the deploy gate)_: Sharpe 2.79 | pooled-Calmar 7.63 | CAGR 54.7% | pooled-MaxDD -7.2% | WR 48.1%
**Folds**: 10 | **Annualised turnover (legs/yr)**: 112.8

## Per-fold OOT

| OOT Window | Sharpe | Calmar | CAGR | MaxDD | WR | n_rebal | alpha |
|---|---|---|---|---|---|---|---|
| 2023-06-07 → 2023-12-07 | 3.05 | 11.91 | 50.3% | -4.2% | 48.0% | 4 | 0.1 |
| 2023-09-07 → 2024-03-07 | 2.89 | 13.19 | 58.9% | -4.5% | 56.6% | 3 | 0.1 |
| 2023-12-07 → 2024-06-07 | 1.67 | 4.88 | 31.6% | -6.5% | 47.4% | 5 | 0.1 |
| 2024-03-07 → 2024-09-07 | 2.79 | 12.30 | 63.8% | -5.2% | 46.1% | 4 | 0.1 |
| 2024-06-07 → 2024-12-07 | 2.82 | 12.12 | 54.1% | -4.5% | 53.1% | 5 | 0.1 |
| 2024-09-07 → 2025-03-07 | -0.85 | -2.54 | -10.8% | -4.2% | 32.7% | 3 | 0.1 |
| 2024-12-07 → 2025-06-07 | 3.96 | 20.78 | 72.9% | -3.5% | 48.0% | 3 | 0.1 |
| 2025-03-07 → 2025-09-07 | 0.38 | 0.94 | 4.7% | -4.9% | 45.5% | 3 | 0.1 |
| 2025-06-07 → 2025-12-07 | 1.33 | 3.19 | 22.4% | -7.0% | 47.4% | 5 | 0.1 |
| 2025-09-07 → 2026-03-07 | 4.26 | 16.19 | 120.1% | -7.4% | 52.6% | 5 | 0.1 |

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
