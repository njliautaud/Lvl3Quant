# ETF rotation v1 — sector SPDR cross-sectional ridge

**Config**: hold=21d, long top-3, long-only, txn=5.0bps, vol_target=0.15, lev_clip=[0.25,2.00]

**HONEST DEPLOY GATE**: median per-fold Calmar **6.66** (need ≥ 1.0), worst-fold MaxDD **-5.7%** (need > -25%), folds passing Calmar≥1: **9/10** (need ≥ 50%).
**RESULT: PASS**

_(For reference — pooled-PnL stats, NOT the deploy gate)_: Sharpe 2.12 | pooled-Calmar 3.82 | CAGR 29.4% | pooled-MaxDD -7.7% | WR 46.1%
**Folds**: 10 | **Annualised turnover (legs/yr)**: 169.6

## Per-fold OOT

| OOT Window | Sharpe | Calmar | CAGR | MaxDD | WR | n_rebal | alpha |
|---|---|---|---|---|---|---|---|
| 2023-06-07 → 2023-12-07 | 3.67 | 19.63 | 49.0% | -2.5% | 49.2% | 5 | 0.1 |
| 2023-09-07 → 2024-03-07 | 3.66 | 16.30 | 58.5% | -3.6% | 58.3% | 4 | 0.1 |
| 2023-12-07 → 2024-06-07 | 0.80 | 1.90 | 9.8% | -5.2% | 44.0% | 6 | 0.1 |
| 2024-03-07 → 2024-09-07 | 2.11 | 5.78 | 28.0% | -4.8% | 41.0% | 5 | 0.1 |
| 2024-06-07 → 2024-12-07 | 2.57 | 7.53 | 37.0% | -4.9% | 56.2% | 5 | 0.1 |
| 2024-09-07 → 2025-03-07 | 1.68 | 5.32 | 20.7% | -3.9% | 41.9% | 4 | 0.1 |
| 2024-12-07 → 2025-06-07 | -0.49 | -1.78 | -5.3% | -3.0% | 31.1% | 3 | 0.1 |
| 2025-03-07 → 2025-09-07 | 2.93 | 11.09 | 32.0% | -2.9% | 51.1% | 3 | 0.1 |
| 2025-06-07 → 2025-12-07 | 1.72 | 5.58 | 26.6% | -4.8% | 46.8% | 6 | 0.1 |
| 2025-09-07 → 2026-03-07 | 3.79 | 13.71 | 77.5% | -5.7% | 51.6% | 6 | 0.1 |

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
