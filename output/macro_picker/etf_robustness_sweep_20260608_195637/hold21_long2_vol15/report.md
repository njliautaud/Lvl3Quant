# ETF rotation v1 — sector SPDR cross-sectional ridge

**Config**: hold=21d, long top-2, long-only, txn=5.0bps, vol_target=0.15, lev_clip=[0.25,2.00]

**HONEST DEPLOY GATE**: median per-fold Calmar **10.61** (need ≥ 1.0), worst-fold MaxDD **-6.4%** (need > -25%), folds passing Calmar≥1: **8/10** (need ≥ 50%).
**RESULT: PASS**

_(For reference — pooled-PnL stats, NOT the deploy gate)_: Sharpe 1.92 | pooled-Calmar 3.24 | CAGR 26.5% | pooled-MaxDD -8.2% | WR 46.5%
**Folds**: 10 | **Annualised turnover (legs/yr)**: 113.1

## Per-fold OOT

| OOT Window | Sharpe | Calmar | CAGR | MaxDD | WR | n_rebal | alpha |
|---|---|---|---|---|---|---|---|
| 2023-06-07 → 2023-12-07 | 3.13 | 18.51 | 44.9% | -2.4% | 46.0% | 5 | 0.1 |
| 2023-09-07 → 2024-03-07 | 3.50 | 16.02 | 57.4% | -3.6% | 56.7% | 4 | 0.1 |
| 2023-12-07 → 2024-06-07 | -0.29 | -0.77 | -4.4% | -5.7% | 42.9% | 6 | 0.1 |
| 2024-03-07 → 2024-09-07 | 1.96 | 4.44 | 24.0% | -5.4% | 43.6% | 5 | 0.1 |
| 2024-06-07 → 2024-12-07 | 3.55 | 14.14 | 52.3% | -3.7% | 53.1% | 5 | 0.1 |
| 2024-09-07 → 2025-03-07 | 2.38 | 9.22 | 29.6% | -3.2% | 50.0% | 4 | 0.1 |
| 2024-12-07 → 2025-06-07 | 0.06 | 0.04 | 0.1% | -3.3% | 37.8% | 3 | 0.1 |
| 2025-03-07 → 2025-09-07 | 2.77 | 13.15 | 31.0% | -2.4% | 53.2% | 3 | 0.1 |
| 2025-06-07 → 2025-12-07 | 1.66 | 5.67 | 25.0% | -4.4% | 48.9% | 6 | 0.1 |
| 2025-09-07 → 2026-03-07 | 3.59 | 12.00 | 77.1% | -6.4% | 50.5% | 6 | 0.1 |

## Feature importance (avg coef across folds)

| Feature | Avg coef |
|---|---|
| `fp_beat_rate_4q_sec` | -0.0142 |
| `fp_eps_yoy_growth_sec` | -0.0129 |
| `rel_strength_spy` | -0.0114 |
| `fp_margin_trend_4q_sec` | +0.0107 |
| `fp_net_margin_sec` | +0.0091 |
| `fp_current_ratio_sec` | -0.0073 |
| `fp_debt_to_equity_sec` | -0.0071 |
| `fp_fcf_yield_sec` | +0.0064 |
| `ret_20d` | +0.0063 |
| `fp_gross_margin_sec` | +0.0060 |
| `fp_ebitda_margin_sec` | -0.0056 |
| `fp_ni_yoy_growth_sec` | +0.0045 |
| `rs_rank_among_sectors` | +0.0040 |
| `momentum_cross_20_60` | +0.0039 |
| `fp_rev_yoy_growth_sec` | -0.0039 |
