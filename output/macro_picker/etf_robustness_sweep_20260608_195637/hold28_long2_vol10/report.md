# ETF rotation v1 — sector SPDR cross-sectional ridge

**Config**: hold=28d, long top-2, long-only, txn=5.0bps, vol_target=0.10, lev_clip=[0.25,2.00]

**HONEST DEPLOY GATE**: median per-fold Calmar **10.88** (need ≥ 1.0), worst-fold MaxDD **-4.1%** (need > -25%), folds passing Calmar≥1: **9/10** (need ≥ 50%).
**RESULT: PASS**

_(For reference — pooled-PnL stats, NOT the deploy gate)_: Sharpe 2.61 | pooled-Calmar 3.81 | CAGR 23.3% | pooled-MaxDD -6.1% | WR 48.7%
**Folds**: 10 | **Annualised turnover (legs/yr)**: 75.2

## Per-fold OOT

| OOT Window | Sharpe | Calmar | CAGR | MaxDD | WR | n_rebal | alpha |
|---|---|---|---|---|---|---|---|
| 2023-06-07 → 2023-12-07 | 3.21 | 12.60 | 24.4% | -1.9% | 49.3% | 4 | 0.1 |
| 2023-09-07 → 2024-03-07 | 2.87 | 11.19 | 27.1% | -2.4% | 54.7% | 3 | 0.1 |
| 2023-12-07 → 2024-06-07 | 1.30 | 4.12 | 12.6% | -3.1% | 46.3% | 5 | 0.1 |
| 2024-03-07 → 2024-09-07 | 2.80 | 11.82 | 29.4% | -2.5% | 52.6% | 4 | 0.1 |
| 2024-06-07 → 2024-12-07 | 3.66 | 15.39 | 31.2% | -2.0% | 54.2% | 5 | 0.1 |
| 2024-09-07 → 2025-03-07 | 0.49 | 1.41 | 3.3% | -2.3% | 38.5% | 3 | 0.1 |
| 2024-12-07 → 2025-06-07 | 5.90 | 29.63 | 53.5% | -1.8% | 54.0% | 3 | 0.1 |
| 2025-03-07 → 2025-09-07 | 1.01 | 3.46 | 8.2% | -2.4% | 50.9% | 3 | 0.1 |
| 2025-06-07 → 2025-12-07 | 0.37 | 0.73 | 3.0% | -4.1% | 47.4% | 5 | 0.1 |
| 2025-09-07 → 2026-03-07 | 3.53 | 10.57 | 38.1% | -3.6% | 50.5% | 5 | 0.1 |

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
