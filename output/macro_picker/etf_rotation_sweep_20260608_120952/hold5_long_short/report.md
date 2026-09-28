# ETF rotation v1 — sector SPDR cross-sectional ridge

**Config**: hold=5d, long top-2, short bot-2, txn=5bps, vol_target=0.15, lev_clip=[0.25,2.00]

**HONEST DEPLOY GATE**: median per-fold Calmar **-1.00** (need ≥ 1.0), worst-fold MaxDD **-100.0%** (need > -25%), folds passing Calmar≥1: **0/10** (need ≥ 50%).
**RESULT: FAIL**

_(For reference — pooled-PnL stats, NOT the deploy gate)_: Sharpe 0.52 | pooled-Calmar -1.00 | CAGR -100.0% | pooled-MaxDD -100.0% | WR 37.0%
**Folds**: 10 | **Annualised turnover (legs/yr)**: 792.1

## Per-fold OOT

| OOT Window | Sharpe | Calmar | CAGR | MaxDD | WR | n_rebal | alpha |
|---|---|---|---|---|---|---|---|
| 2023-06-07 → 2023-12-07 | 1.68 | -1.00 | -100.0% | -100.0% | 41.3% | 26 | 0.1 |
| 2023-09-07 → 2024-03-07 | -1.45 | -1.00 | -100.0% | -100.0% | 30.6% | 25 | 0.1 |
| 2023-12-07 → 2024-06-07 | -3.15 | -1.00 | -100.0% | -100.0% | 28.9% | 25 | 0.1 |
| 2024-03-07 → 2024-09-07 | -0.79 | -1.00 | -100.0% | -100.0% | 33.0% | 26 | 0.1 |
| 2024-06-07 → 2024-12-07 | 0.87 | -1.00 | -100.0% | -100.0% | 39.6% | 26 | 0.1 |
| 2024-09-07 → 2025-03-07 | 2.43 | -1.00 | -99.4% | -99.8% | 42.3% | 25 | 0.1 |
| 2024-12-07 → 2025-06-07 | 0.62 | -1.00 | -100.0% | -100.0% | 40.4% | 25 | 0.1 |
| 2025-03-07 → 2025-09-07 | -1.78 | -1.00 | -100.0% | -100.0% | 34.0% | 26 | 0.1 |
| 2025-06-07 → 2025-12-07 | 0.59 | -1.00 | -100.0% | -100.0% | 37.0% | 26 | 0.1 |
| 2025-09-07 → 2026-03-07 | 0.47 | -1.00 | -100.0% | -100.0% | 39.4% | 25 | 0.1 |

## Feature importance (avg coef across folds)

| Feature | Avg coef |
|---|---|
| `rel_strength_spy` | -0.0046 |
| `fp_beat_rate_4q_sec` | -0.0039 |
| `ret_20d` | +0.0032 |
| `fp_eps_yoy_growth_sec` | -0.0027 |
| `fp_margin_trend_4q_sec` | +0.0023 |
| `fp_debt_to_equity_sec` | -0.0019 |
| `fp_fcf_yield_sec` | +0.0015 |
| `fp_current_ratio_sec` | -0.0013 |
| `fp_roe_sec` | +0.0012 |
| `fp_gross_margin_sec` | +0.0011 |
| `fp_rev_yoy_growth_sec` | -0.0010 |
| `fp_net_margin_sec` | +0.0010 |
| `fp_fcf_yoy_growth_sec` | +0.0009 |
| `momentum_cross_20_60` | +0.0009 |
| `rs_rank_among_sectors` | +0.0009 |
