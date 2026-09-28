# ETF rotation v1 — sector SPDR cross-sectional ridge

**Config**: hold=28d, long top-1, long-only, txn=5.0bps, vol_target=0.20, lev_clip=[0.25,2.00]

**HONEST DEPLOY GATE**: median per-fold Calmar **7.29** (need ≥ 1.0), worst-fold MaxDD **-12.1%** (need > -25%), folds passing Calmar≥1: **9/10** (need ≥ 50%).
**RESULT: PASS**

_(For reference — pooled-PnL stats, NOT the deploy gate)_: Sharpe 2.05 | pooled-Calmar 2.17 | CAGR 40.2% | pooled-MaxDD -18.5% | WR 48.3%
**Folds**: 10 | **Annualised turnover (legs/yr)**: 37.6

## Per-fold OOT

| OOT Window | Sharpe | Calmar | CAGR | MaxDD | WR | n_rebal | alpha |
|---|---|---|---|---|---|---|---|
| 2023-06-07 → 2023-12-07 | 3.06 | 8.66 | 49.0% | -5.7% | 50.7% | 4 | 0.1 |
| 2023-09-07 → 2024-03-07 | 0.59 | 1.75 | 8.8% | -5.0% | 50.9% | 3 | 0.1 |
| 2023-12-07 → 2024-06-07 | 0.73 | 1.79 | 12.6% | -7.0% | 43.2% | 5 | 0.1 |
| 2024-03-07 → 2024-09-07 | 2.00 | 5.93 | 45.4% | -7.7% | 52.6% | 4 | 0.1 |
| 2024-06-07 → 2024-12-07 | 2.90 | 9.59 | 54.8% | -5.7% | 51.0% | 5 | 0.1 |
| 2024-09-07 → 2025-03-07 | 3.68 | 26.79 | 86.7% | -3.2% | 51.9% | 3 | 0.1 |
| 2024-12-07 → 2025-06-07 | 5.74 | 46.68 | 130.1% | -2.8% | 62.0% | 3 | 0.1 |
| 2025-03-07 → 2025-09-07 | 1.17 | 4.33 | 20.4% | -4.7% | 50.9% | 3 | 0.1 |
| 2025-06-07 → 2025-12-07 | 0.19 | 0.15 | 1.8% | -12.1% | 46.3% | 5 | 0.1 |
| 2025-09-07 → 2026-03-07 | 3.53 | 13.51 | 99.4% | -7.4% | 52.6% | 5 | 0.1 |

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
