# ETF rotation v1 — sector SPDR cross-sectional ridge

**Config**: hold=5d, long top-2, short bot-2, txn=5bps, vol_target=0.15, lev_clip=[0.25,2.00]

**HONEST DEPLOY GATE**: median per-fold Calmar **0.67** (need ≥ 1.0), worst-fold MaxDD **-17.1%** (need > -25%), folds passing Calmar≥1: **4/10** (need ≥ 50%).
**RESULT: FAIL**

_(For reference — pooled-PnL stats, NOT the deploy gate)_: Sharpe 0.36 | pooled-Calmar 0.22 | CAGR 3.9% | pooled-MaxDD -17.6% | WR 37.1%
**Folds**: 10 | **Annualised turnover (legs/yr)**: 792.1

## Per-fold OOT

| OOT Window | Sharpe | Calmar | CAGR | MaxDD | WR | n_rebal | alpha |
|---|---|---|---|---|---|---|---|
| 2023-06-07 → 2023-12-07 | 2.25 | 4.58 | 31.2% | -6.8% | 40.4% | 26 | 0.1 |
| 2023-09-07 → 2024-03-07 | -0.71 | -1.11 | -9.3% | -8.5% | 30.6% | 25 | 0.1 |
| 2023-12-07 → 2024-06-07 | -3.37 | -2.25 | -38.4% | -17.1% | 28.9% | 25 | 0.1 |
| 2024-03-07 → 2024-09-07 | -1.57 | -1.87 | -13.9% | -7.4% | 32.0% | 26 | 0.1 |
| 2024-06-07 → 2024-12-07 | 1.39 | 3.18 | 18.5% | -5.8% | 39.6% | 26 | 0.1 |
| 2024-09-07 → 2025-03-07 | 1.27 | 3.04 | 19.0% | -6.2% | 42.3% | 25 | 0.1 |
| 2024-12-07 → 2025-06-07 | 1.02 | 2.33 | 15.1% | -6.5% | 42.3% | 25 | 0.1 |
| 2025-03-07 → 2025-09-07 | -1.49 | -1.28 | -15.6% | -12.2% | 35.0% | 26 | 0.1 |
| 2025-06-07 → 2025-12-07 | 0.48 | 0.85 | 5.8% | -6.8% | 38.0% | 26 | 0.1 |
| 2025-09-07 → 2026-03-07 | 0.37 | 0.48 | 4.3% | -9.0% | 39.4% | 25 | 0.1 |

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
