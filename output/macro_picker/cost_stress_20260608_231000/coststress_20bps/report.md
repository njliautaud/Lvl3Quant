# ETF rotation v1 — sector SPDR cross-sectional ridge

**Config**: hold=21d, long top-2, long-only, txn=20.0bps, vol_target=0.15, lev_clip=[0.25,2.00]

**HONEST DEPLOY GATE**: median per-fold Calmar **9.35** (need ≥ 1.0), worst-fold MaxDD **-6.6%** (need > -25%), folds passing Calmar≥1: **8/10** (need ≥ 50%).
**RESULT: PASS**

_(For reference — pooled-PnL stats, NOT the deploy gate)_: Sharpe 1.62 | pooled-Calmar 2.44 | CAGR 21.8% | pooled-MaxDD -9.0% | WR 46.5%
**Folds**: 10 | **Annualised turnover (legs/yr)**: 113.1

## Per-fold OOT

| OOT Window | Sharpe | Calmar | CAGR | MaxDD | WR | n_rebal | alpha |
|---|---|---|---|---|---|---|---|
| 2023-06-07 → 2023-12-07 | 2.86 | 15.63 | 40.5% | -2.6% | 46.0% | 5 | 0.1 |
| 2023-09-07 → 2024-03-07 | 3.28 | 14.20 | 53.2% | -3.7% | 56.7% | 4 | 0.1 |
| 2023-12-07 → 2024-06-07 | -0.48 | -1.03 | -6.6% | -6.4% | 42.9% | 6 | 0.1 |
| 2024-03-07 → 2024-09-07 | 1.68 | 3.63 | 20.2% | -5.6% | 43.6% | 5 | 0.1 |
| 2024-06-07 → 2024-12-07 | 3.22 | 12.66 | 46.8% | -3.7% | 53.1% | 5 | 0.1 |
| 2024-09-07 → 2025-03-07 | 2.09 | 7.89 | 25.7% | -3.3% | 50.0% | 4 | 0.1 |
| 2024-12-07 → 2025-06-07 | -0.21 | -0.79 | -2.7% | -3.4% | 37.8% | 3 | 0.1 |
| 2025-03-07 → 2025-09-07 | 2.55 | 11.41 | 28.3% | -2.5% | 53.2% | 3 | 0.1 |
| 2025-06-07 → 2025-12-07 | 1.45 | 4.63 | 21.4% | -4.6% | 48.9% | 6 | 0.1 |
| 2025-09-07 → 2026-03-07 | 3.38 | 10.80 | 71.5% | -6.6% | 50.5% | 6 | 0.1 |

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
