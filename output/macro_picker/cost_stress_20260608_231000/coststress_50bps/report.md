# ETF rotation v1 — sector SPDR cross-sectional ridge

**Config**: hold=21d, long top-2, long-only, txn=50.0bps, vol_target=0.15, lev_clip=[0.25,2.00]

**HONEST DEPLOY GATE**: median per-fold Calmar **6.66** (need ≥ 1.0), worst-fold MaxDD **-7.8%** (need > -25%), folds passing Calmar≥1: **8/10** (need ≥ 50%).
**RESULT: PASS**

_(For reference — pooled-PnL stats, NOT the deploy gate)_: Sharpe 1.01 | pooled-Calmar 1.24 | CAGR 13.0% | pooled-MaxDD -10.5% | WR 46.5%
**Folds**: 10 | **Annualised turnover (legs/yr)**: 113.1

## Per-fold OOT

| OOT Window | Sharpe | Calmar | CAGR | MaxDD | WR | n_rebal | alpha |
|---|---|---|---|---|---|---|---|
| 2023-06-07 → 2023-12-07 | 2.31 | 10.61 | 32.1% | -3.0% | 46.0% | 5 | 0.1 |
| 2023-09-07 → 2024-03-07 | 2.82 | 11.07 | 45.0% | -4.1% | 56.7% | 4 | 0.1 |
| 2023-12-07 → 2024-06-07 | -0.84 | -1.40 | -10.9% | -7.8% | 42.9% | 6 | 0.1 |
| 2024-03-07 → 2024-09-07 | 1.11 | 2.20 | 13.0% | -5.9% | 43.6% | 5 | 0.1 |
| 2024-06-07 → 2024-12-07 | 2.54 | 9.84 | 36.4% | -3.7% | 53.1% | 5 | 0.1 |
| 2024-09-07 → 2025-03-07 | 1.51 | 4.93 | 18.3% | -3.7% | 50.0% | 4 | 0.1 |
| 2024-12-07 → 2025-06-07 | -0.75 | -2.30 | -8.0% | -3.5% | 37.8% | 3 | 0.1 |
| 2025-03-07 → 2025-09-07 | 2.10 | 8.38 | 23.2% | -2.8% | 53.2% | 3 | 0.1 |
| 2025-06-07 → 2025-12-07 | 1.02 | 2.88 | 14.6% | -5.1% | 48.9% | 6 | 0.1 |
| 2025-09-07 → 2026-03-07 | 2.94 | 8.67 | 60.8% | -7.0% | 50.5% | 6 | 0.1 |

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
