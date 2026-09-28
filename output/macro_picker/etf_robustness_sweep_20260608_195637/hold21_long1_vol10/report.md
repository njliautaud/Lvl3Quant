# ETF rotation v1 — sector SPDR cross-sectional ridge

**Config**: hold=21d, long top-1, long-only, txn=5.0bps, vol_target=0.10, lev_clip=[0.25,2.00]

**HONEST DEPLOY GATE**: median per-fold Calmar **5.75** (need ≥ 1.0), worst-fold MaxDD **-4.7%** (need > -25%), folds passing Calmar≥1: **10/10** (need ≥ 50%).
**RESULT: PASS**

_(For reference — pooled-PnL stats, NOT the deploy gate)_: Sharpe 1.78 | pooled-Calmar 2.88 | CAGR 14.9% | pooled-MaxDD -5.2% | WR 46.5%
**Folds**: 10 | **Annualised turnover (legs/yr)**: 56.5

## Per-fold OOT

| OOT Window | Sharpe | Calmar | CAGR | MaxDD | WR | n_rebal | alpha |
|---|---|---|---|---|---|---|---|
| 2023-06-07 → 2023-12-07 | 4.99 | 29.15 | 41.5% | -1.4% | 52.4% | 5 | 0.1 |
| 2023-09-07 → 2024-03-07 | 1.47 | 5.27 | 13.7% | -2.6% | 46.7% | 4 | 0.1 |
| 2023-12-07 → 2024-06-07 | 0.72 | 1.84 | 5.6% | -3.0% | 42.9% | 6 | 0.1 |
| 2024-03-07 → 2024-09-07 | 2.48 | 6.24 | 21.2% | -3.4% | 47.4% | 5 | 0.1 |
| 2024-06-07 → 2024-12-07 | 3.52 | 15.23 | 30.7% | -2.0% | 64.1% | 5 | 0.1 |
| 2024-09-07 → 2025-03-07 | 1.60 | 8.08 | 15.2% | -1.9% | 51.6% | 4 | 0.1 |
| 2024-12-07 → 2025-06-07 | 1.83 | 9.32 | 14.5% | -1.6% | 40.0% | 3 | 0.1 |
| 2025-03-07 → 2025-09-07 | 0.80 | 2.99 | 6.4% | -2.2% | 46.8% | 3 | 0.1 |
| 2025-06-07 → 2025-12-07 | 0.96 | 2.05 | 8.8% | -4.3% | 44.7% | 6 | 0.1 |
| 2025-09-07 → 2026-03-07 | 1.79 | 3.90 | 18.3% | -4.7% | 44.1% | 6 | 0.1 |

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
