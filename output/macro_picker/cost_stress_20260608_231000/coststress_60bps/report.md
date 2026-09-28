# ETF rotation v1 — sector SPDR cross-sectional ridge

**Config**: hold=21d, long top-2, long-only, txn=60.0bps, vol_target=0.15, lev_clip=[0.25,2.00]

**HONEST DEPLOY GATE**: median per-fold Calmar **5.69** (need ≥ 1.0), worst-fold MaxDD **-8.2%** (need > -25%), folds passing Calmar≥1: **8/10** (need ≥ 50%).
**RESULT: PASS**

_(For reference — pooled-PnL stats, NOT the deploy gate)_: Sharpe 0.81 | pooled-Calmar 0.93 | CAGR 10.2% | pooled-MaxDD -11.1% | WR 46.5%
**Folds**: 10 | **Annualised turnover (legs/yr)**: 113.1

## Per-fold OOT

| OOT Window | Sharpe | Calmar | CAGR | MaxDD | WR | n_rebal | alpha |
|---|---|---|---|---|---|---|---|
| 2023-06-07 → 2023-12-07 | 2.12 | 9.28 | 29.4% | -3.2% | 46.0% | 5 | 0.1 |
| 2023-09-07 → 2024-03-07 | 2.67 | 10.15 | 42.4% | -4.2% | 56.7% | 4 | 0.1 |
| 2023-12-07 → 2024-06-07 | -0.96 | -1.49 | -12.3% | -8.2% | 42.9% | 6 | 0.1 |
| 2024-03-07 → 2024-09-07 | 0.92 | 1.70 | 10.7% | -6.3% | 43.6% | 5 | 0.1 |
| 2024-06-07 → 2024-12-07 | 2.32 | 8.95 | 33.1% | -3.7% | 53.1% | 5 | 0.1 |
| 2024-09-07 → 2025-03-07 | 1.32 | 4.12 | 15.9% | -3.9% | 50.0% | 4 | 0.1 |
| 2024-12-07 → 2025-06-07 | -0.92 | -2.76 | -9.8% | -3.5% | 37.8% | 3 | 0.1 |
| 2025-03-07 → 2025-09-07 | 1.95 | 7.25 | 21.5% | -3.0% | 53.2% | 3 | 0.1 |
| 2025-06-07 → 2025-12-07 | 0.88 | 2.38 | 12.4% | -5.2% | 48.9% | 6 | 0.1 |
| 2025-09-07 → 2026-03-07 | 2.79 | 8.03 | 57.4% | -7.1% | 50.5% | 6 | 0.1 |

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
