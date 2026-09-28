# ETF rotation v1 — sector SPDR cross-sectional ridge

**Config**: hold=21d, long top-3, long-only, txn=5.0bps, vol_target=0.10, lev_clip=[0.25,2.00]

**HONEST DEPLOY GATE**: median per-fold Calmar **6.36** (need ≥ 1.0), worst-fold MaxDD **-3.8%** (need > -25%), folds passing Calmar≥1: **9/10** (need ≥ 50%).
**RESULT: PASS**

_(For reference — pooled-PnL stats, NOT the deploy gate)_: Sharpe 2.12 | pooled-Calmar 3.67 | CAGR 19.0% | pooled-MaxDD -5.2% | WR 46.1%
**Folds**: 10 | **Annualised turnover (legs/yr)**: 169.6

## Per-fold OOT

| OOT Window | Sharpe | Calmar | CAGR | MaxDD | WR | n_rebal | alpha |
|---|---|---|---|---|---|---|---|
| 2023-06-07 → 2023-12-07 | 3.67 | 18.39 | 30.6% | -1.7% | 49.2% | 5 | 0.1 |
| 2023-09-07 → 2024-03-07 | 3.66 | 15.08 | 36.2% | -2.4% | 58.3% | 4 | 0.1 |
| 2023-12-07 → 2024-06-07 | 0.80 | 1.92 | 6.6% | -3.5% | 44.0% | 6 | 0.1 |
| 2024-03-07 → 2024-09-07 | 2.11 | 5.57 | 18.1% | -3.2% | 41.0% | 5 | 0.1 |
| 2024-06-07 → 2024-12-07 | 2.57 | 7.16 | 23.6% | -3.3% | 56.2% | 5 | 0.1 |
| 2024-09-07 → 2025-03-07 | 1.68 | 5.19 | 13.5% | -2.6% | 41.9% | 4 | 0.1 |
| 2024-12-07 → 2025-06-07 | -0.49 | -1.74 | -3.5% | -2.0% | 31.1% | 3 | 0.1 |
| 2025-03-07 → 2025-09-07 | 2.93 | 10.61 | 20.5% | -1.9% | 51.1% | 3 | 0.1 |
| 2025-06-07 → 2025-12-07 | 1.72 | 5.42 | 17.3% | -3.2% | 46.8% | 6 | 0.1 |
| 2025-09-07 → 2026-03-07 | 3.79 | 12.41 | 47.0% | -3.8% | 51.6% | 6 | 0.1 |

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
