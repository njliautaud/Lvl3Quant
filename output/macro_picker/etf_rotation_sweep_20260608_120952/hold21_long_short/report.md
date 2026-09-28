# ETF rotation v1 — sector SPDR cross-sectional ridge

**Config**: hold=21d, long top-2, short bot-2, txn=5bps, vol_target=0.15, lev_clip=[0.25,2.00]

**HONEST DEPLOY GATE**: median per-fold Calmar **-1.00** (need ≥ 1.0), worst-fold MaxDD **-100.0%** (need > -25%), folds passing Calmar≥1: **0/10** (need ≥ 50%).
**RESULT: FAIL**

_(For reference — pooled-PnL stats, NOT the deploy gate)_: Sharpe 0.09 | pooled-Calmar -1.00 | CAGR -100.0% | pooled-MaxDD -100.0% | WR 46.5%
**Folds**: 10 | **Annualised turnover (legs/yr)**: 233.5

## Per-fold OOT

| OOT Window | Sharpe | Calmar | CAGR | MaxDD | WR | n_rebal | alpha |
|---|---|---|---|---|---|---|---|
| 2023-06-07 → 2023-12-07 | -0.31 | -1.00 | -100.0% | -100.0% | 44.2% | 7 | 0.1 |
| 2023-09-07 → 2024-03-07 | 1.87 | -1.00 | -100.0% | -99.9% | 50.0% | 6 | 0.1 |
| 2023-12-07 → 2024-06-07 | -2.07 | -1.00 | -100.0% | -100.0% | 40.7% | 6 | 0.1 |
| 2024-03-07 → 2024-09-07 | 0.62 | -1.00 | -100.0% | -100.0% | 47.4% | 7 | 0.1 |
| 2024-06-07 → 2024-12-07 | 0.18 | -1.00 | -100.0% | -100.0% | 46.9% | 7 | 0.1 |
| 2024-09-07 → 2025-03-07 | 2.78 | -1.00 | -99.7% | -99.5% | 52.2% | 6 | 0.1 |
| 2024-12-07 → 2025-06-07 | 1.52 | -1.00 | -100.0% | -99.8% | 52.7% | 6 | 0.1 |
| 2025-03-07 → 2025-09-07 | 0.72 | -1.00 | -100.0% | -100.0% | 47.3% | 6 | 0.1 |
| 2025-06-07 → 2025-12-07 | -0.59 | -1.00 | -100.0% | -100.0% | 44.7% | 6 | 0.1 |
| 2025-09-07 → 2026-03-07 | 1.73 | -1.00 | -100.0% | -99.6% | 53.8% | 6 | 0.1 |

## Feature importance (avg coef across folds)

| Feature | Avg coef |
|---|---|
| `fp_beat_rate_4q_sec` | -0.0143 |
| `fp_eps_yoy_growth_sec` | -0.0129 |
| `rel_strength_spy` | -0.0113 |
| `fp_margin_trend_4q_sec` | +0.0107 |
| `fp_net_margin_sec` | +0.0092 |
| `fp_current_ratio_sec` | -0.0073 |
| `fp_debt_to_equity_sec` | -0.0071 |
| `fp_fcf_yield_sec` | +0.0065 |
| `ret_20d` | +0.0062 |
| `fp_gross_margin_sec` | +0.0058 |
| `fp_ebitda_margin_sec` | -0.0056 |
| `fp_ni_yoy_growth_sec` | +0.0045 |
| `rs_rank_among_sectors` | +0.0040 |
| `momentum_cross_20_60` | +0.0039 |
| `fp_rev_yoy_growth_sec` | -0.0038 |
