# ETF rotation v1 — sector SPDR cross-sectional ridge

**Config**: hold=21d, long top-2, long-only, txn=75.0bps, vol_target=0.15, lev_clip=[0.25,2.00]

**HONEST DEPLOY GATE**: median per-fold Calmar **4.44** (need ≥ 1.0), worst-fold MaxDD **-8.9%** (need > -25%), folds passing Calmar≥1: **8/10** (need ≥ 50%).
**RESULT: PASS**

_(For reference — pooled-PnL stats, NOT the deploy gate)_: Sharpe 0.51 | pooled-Calmar 0.51 | CAGR 6.2% | pooled-MaxDD -12.0% | WR 46.5%
**Folds**: 10 | **Annualised turnover (legs/yr)**: 113.1

## Per-fold OOT

| OOT Window | Sharpe | Calmar | CAGR | MaxDD | WR | n_rebal | alpha |
|---|---|---|---|---|---|---|---|
| 2023-06-07 → 2023-12-07 | 1.85 | 7.53 | 25.5% | -3.4% | 46.0% | 5 | 0.1 |
| 2023-09-07 → 2024-03-07 | 2.43 | 8.88 | 38.5% | -4.3% | 56.7% | 4 | 0.1 |
| 2023-12-07 → 2024-06-07 | -1.14 | -1.61 | -14.3% | -8.9% | 42.9% | 6 | 0.1 |
| 2024-03-07 → 2024-09-07 | 0.65 | 1.07 | 7.3% | -6.8% | 43.6% | 5 | 0.1 |
| 2024-06-07 → 2024-12-07 | 1.99 | 7.64 | 28.2% | -3.7% | 53.1% | 5 | 0.1 |
| 2024-09-07 → 2025-03-07 | 1.04 | 3.04 | 12.4% | -4.1% | 50.0% | 4 | 0.1 |
| 2024-12-07 → 2025-06-07 | -1.16 | -3.41 | -12.3% | -3.6% | 37.8% | 3 | 0.1 |
| 2025-03-07 → 2025-09-07 | 1.73 | 5.83 | 19.0% | -3.3% | 53.2% | 3 | 0.1 |
| 2025-06-07 → 2025-12-07 | 0.67 | 1.65 | 9.1% | -5.5% | 48.9% | 6 | 0.1 |
| 2025-09-07 → 2026-03-07 | 2.57 | 7.14 | 52.4% | -7.3% | 50.5% | 6 | 0.1 |

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
