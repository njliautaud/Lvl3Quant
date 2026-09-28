# ETF rotation v1 — sector SPDR cross-sectional ridge

**Config**: hold=21d, long top-2, long-only, txn=5.0bps, vol_target=0.15, lev_clip=[0.25,2.00]

**HONEST DEPLOY GATE**: median per-fold Calmar **5.23** (need ≥ 1.0), worst-fold MaxDD **-10.8%** (need > -25%), folds passing Calmar≥1: **7/10** (need ≥ 50%).
**RESULT: PASS**

_(For reference — pooled-PnL stats, NOT the deploy gate)_: Sharpe 1.07 | pooled-Calmar 1.25 | CAGR 15.0% | pooled-MaxDD -12.0% | WR 50.6%
**Folds**: 10 | **Annualised turnover (legs/yr)**: 113.1

## Per-fold OOT

| OOT Window | Sharpe | Calmar | CAGR | MaxDD | WR | n_rebal | alpha |
|---|---|---|---|---|---|---|---|
| 2023-06-07 → 2023-12-07 | 1.67 | 4.55 | 23.7% | -5.2% | 49.2% | 5 | 0.1 |
| 2023-09-07 → 2024-03-07 | 3.41 | 15.54 | 55.7% | -3.6% | 56.7% | 4 | 0.1 |
| 2023-12-07 → 2024-06-07 | -0.40 | -1.00 | -6.5% | -6.4% | 50.5% | 6 | 0.1 |
| 2024-03-07 → 2024-09-07 | 1.50 | 3.58 | 19.7% | -5.5% | 51.3% | 5 | 0.1 |
| 2024-06-07 → 2024-12-07 | 3.55 | 14.14 | 52.3% | -3.7% | 53.1% | 5 | 0.1 |
| 2024-09-07 → 2025-03-07 | 0.11 | 0.08 | 0.6% | -7.6% | 53.2% | 4 | 0.1 |
| 2024-12-07 → 2025-06-07 | -3.42 | -3.83 | -41.3% | -10.8% | 42.2% | 3 | 0.1 |
| 2025-03-07 → 2025-09-07 | 2.77 | 13.15 | 31.0% | -2.4% | 53.2% | 3 | 0.1 |
| 2025-06-07 → 2025-12-07 | 1.68 | 5.91 | 26.0% | -4.4% | 51.1% | 6 | 0.1 |
| 2025-09-07 → 2026-03-07 | 3.03 | 10.32 | 66.3% | -6.4% | 57.0% | 6 | 0.1 |

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
