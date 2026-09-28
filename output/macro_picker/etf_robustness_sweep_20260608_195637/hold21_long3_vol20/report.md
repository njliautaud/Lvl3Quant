# ETF rotation v1 — sector SPDR cross-sectional ridge

**Config**: hold=21d, long top-3, long-only, txn=5.0bps, vol_target=0.20, lev_clip=[0.25,2.00]

**HONEST DEPLOY GATE**: median per-fold Calmar **7.03** (need ≥ 1.0), worst-fold MaxDD **-7.4%** (need > -25%), folds passing Calmar≥1: **9/10** (need ≥ 50%).
**RESULT: PASS**

_(For reference — pooled-PnL stats, NOT the deploy gate)_: Sharpe 2.10 | pooled-Calmar 4.00 | CAGR 39.3% | pooled-MaxDD -9.8% | WR 46.1%
**Folds**: 10 | **Annualised turnover (legs/yr)**: 169.6

## Per-fold OOT

| OOT Window | Sharpe | Calmar | CAGR | MaxDD | WR | n_rebal | alpha |
|---|---|---|---|---|---|---|---|
| 2023-06-07 → 2023-12-07 | 3.67 | 20.70 | 68.8% | -3.3% | 49.2% | 5 | 0.1 |
| 2023-09-07 → 2024-03-07 | 3.66 | 17.65 | 84.2% | -4.8% | 58.3% | 4 | 0.1 |
| 2023-12-07 → 2024-06-07 | 0.81 | 1.93 | 13.1% | -6.8% | 44.0% | 6 | 0.1 |
| 2024-03-07 → 2024-09-07 | 2.14 | 6.08 | 39.0% | -6.4% | 41.0% | 5 | 0.1 |
| 2024-06-07 → 2024-12-07 | 2.58 | 7.98 | 51.2% | -6.4% | 56.2% | 5 | 0.1 |
| 2024-09-07 → 2025-03-07 | 1.77 | 5.85 | 26.4% | -4.5% | 41.9% | 4 | 0.1 |
| 2024-12-07 → 2025-06-07 | -0.46 | -1.83 | -6.4% | -3.5% | 31.1% | 3 | 0.1 |
| 2025-03-07 → 2025-09-07 | 2.93 | 11.60 | 44.6% | -3.8% | 51.1% | 3 | 0.1 |
| 2025-06-07 → 2025-12-07 | 1.74 | 5.95 | 35.6% | -6.0% | 46.8% | 6 | 0.1 |
| 2025-09-07 → 2026-03-07 | 3.78 | 14.74 | 109.8% | -7.4% | 51.6% | 6 | 0.1 |

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
