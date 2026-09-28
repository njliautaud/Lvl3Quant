# ETF rotation v1 — sector SPDR cross-sectional ridge

**Config**: hold=21d, long top-2, long-only, txn=5.0bps, vol_target=0.10, lev_clip=[0.25,2.00]

**HONEST DEPLOY GATE**: median per-fold Calmar **9.86** (need ≥ 1.0), worst-fold MaxDD **-4.3%** (need > -25%), folds passing Calmar≥1: **8/10** (need ≥ 50%).
**RESULT: PASS**

_(For reference — pooled-PnL stats, NOT the deploy gate)_: Sharpe 1.92 | pooled-Calmar 3.13 | CAGR 17.1% | pooled-MaxDD -5.5% | WR 46.5%
**Folds**: 10 | **Annualised turnover (legs/yr)**: 113.1

## Per-fold OOT

| OOT Window | Sharpe | Calmar | CAGR | MaxDD | WR | n_rebal | alpha |
|---|---|---|---|---|---|---|---|
| 2023-06-07 → 2023-12-07 | 3.13 | 17.45 | 28.3% | -1.6% | 46.0% | 5 | 0.1 |
| 2023-09-07 → 2024-03-07 | 3.50 | 14.85 | 35.6% | -2.4% | 56.7% | 4 | 0.1 |
| 2023-12-07 → 2024-06-07 | -0.29 | -0.73 | -2.8% | -3.8% | 42.9% | 6 | 0.1 |
| 2024-03-07 → 2024-09-07 | 1.96 | 4.30 | 15.6% | -3.6% | 43.6% | 5 | 0.1 |
| 2024-06-07 → 2024-12-07 | 3.55 | 13.16 | 32.6% | -2.5% | 53.1% | 5 | 0.1 |
| 2024-09-07 → 2025-03-07 | 2.38 | 8.87 | 19.0% | -2.1% | 50.0% | 4 | 0.1 |
| 2024-12-07 → 2025-06-07 | 0.06 | 0.09 | 0.2% | -2.2% | 37.8% | 3 | 0.1 |
| 2025-03-07 → 2025-09-07 | 2.77 | 12.61 | 19.8% | -1.6% | 53.2% | 3 | 0.1 |
| 2025-06-07 → 2025-12-07 | 1.66 | 5.52 | 16.3% | -2.9% | 48.9% | 6 | 0.1 |
| 2025-09-07 → 2026-03-07 | 3.59 | 10.86 | 46.8% | -4.3% | 50.5% | 6 | 0.1 |

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
