# ETF rotation v1 — sector SPDR cross-sectional ridge

**Config**: hold=28d, long top-3, long-only, txn=5.0bps, vol_target=0.10, lev_clip=[0.25,2.00]

**HONEST DEPLOY GATE**: median per-fold Calmar **10.83** (need ≥ 1.0), worst-fold MaxDD **-4.3%** (need > -25%), folds passing Calmar≥1: **9/10** (need ≥ 50%).
**RESULT: PASS**

_(For reference — pooled-PnL stats, NOT the deploy gate)_: Sharpe 2.71 | pooled-Calmar 6.54 | CAGR 24.7% | pooled-MaxDD -3.8% | WR 48.1%
**Folds**: 10 | **Annualised turnover (legs/yr)**: 112.8

## Per-fold OOT

| OOT Window | Sharpe | Calmar | CAGR | MaxDD | WR | n_rebal | alpha |
|---|---|---|---|---|---|---|---|
| 2023-06-07 → 2023-12-07 | 3.05 | 10.76 | 22.9% | -2.1% | 48.0% | 4 | 0.1 |
| 2023-09-07 → 2024-03-07 | 2.89 | 11.82 | 26.5% | -2.2% | 56.6% | 3 | 0.1 |
| 2023-12-07 → 2024-06-07 | 1.56 | 4.07 | 14.5% | -3.6% | 47.4% | 5 | 0.1 |
| 2024-03-07 → 2024-09-07 | 2.79 | 10.90 | 28.5% | -2.6% | 46.1% | 4 | 0.1 |
| 2024-06-07 → 2024-12-07 | 2.82 | 10.94 | 24.5% | -2.2% | 53.1% | 5 | 0.1 |
| 2024-09-07 → 2025-03-07 | -0.78 | -2.20 | -5.3% | -2.4% | 32.7% | 3 | 0.1 |
| 2024-12-07 → 2025-06-07 | 3.77 | 16.07 | 32.3% | -2.0% | 48.0% | 3 | 0.1 |
| 2025-03-07 → 2025-09-07 | 0.38 | 1.04 | 2.6% | -2.5% | 45.5% | 3 | 0.1 |
| 2025-06-07 → 2025-12-07 | 1.06 | 2.29 | 9.9% | -4.3% | 47.4% | 5 | 0.1 |
| 2025-09-07 → 2026-03-07 | 4.27 | 13.17 | 49.8% | -3.8% | 52.6% | 5 | 0.1 |

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
