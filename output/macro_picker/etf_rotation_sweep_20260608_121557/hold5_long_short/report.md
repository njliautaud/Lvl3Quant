# ETF rotation v1 — sector SPDR cross-sectional ridge

**Config**: hold=5d, long top-2, short bot-2, txn=5bps, vol_target=0.15, lev_clip=[0.25,2.00]

**HONEST DEPLOY GATE**: median per-fold Calmar **0.44** (need ≥ 1.0), worst-fold MaxDD **-17.9%** (need > -25%), folds passing Calmar≥1: **3/10** (need ≥ 50%).
**RESULT: FAIL**

_(For reference — pooled-PnL stats, NOT the deploy gate)_: Sharpe -0.19 | pooled-Calmar -0.13 | CAGR -3.2% | pooled-MaxDD -25.2% | WR 35.9%
**Folds**: 10 | **Annualised turnover (legs/yr)**: 792.1

## Per-fold OOT

| OOT Window | Sharpe | Calmar | CAGR | MaxDD | WR | n_rebal | alpha |
|---|---|---|---|---|---|---|---|
| 2023-06-07 → 2023-12-07 | 2.25 | 4.58 | 31.2% | -6.8% | 40.4% | 26 | 0.1 |
| 2023-09-07 → 2024-03-07 | -1.12 | -1.31 | -13.2% | -10.0% | 29.6% | 25 | 0.1 |
| 2023-12-07 → 2024-06-07 | -3.67 | -2.24 | -40.1% | -17.9% | 26.8% | 25 | 0.1 |
| 2024-03-07 → 2024-09-07 | -4.34 | -2.21 | -36.1% | -16.4% | 29.0% | 26 | 0.1 |
| 2024-06-07 → 2024-12-07 | 0.98 | 2.16 | 12.6% | -5.8% | 39.6% | 26 | 0.1 |
| 2024-09-07 → 2025-03-07 | 0.67 | 1.06 | 9.4% | -8.8% | 42.3% | 25 | 0.1 |
| 2024-12-07 → 2025-06-07 | 0.27 | 0.58 | 2.9% | -4.9% | 40.4% | 25 | 0.1 |
| 2025-03-07 → 2025-09-07 | -1.59 | -1.42 | -16.3% | -11.5% | 34.0% | 26 | 0.1 |
| 2025-06-07 → 2025-12-07 | 0.26 | 0.40 | 2.7% | -6.8% | 38.0% | 26 | 0.1 |
| 2025-09-07 → 2026-03-07 | 0.37 | 0.48 | 4.3% | -9.0% | 39.4% | 25 | 0.1 |

## Feature importance (avg coef across folds)

| Feature | Avg coef |
|---|---|
| `rel_strength_spy` | -0.0047 |
| `fp_beat_rate_4q_sec` | -0.0038 |
| `ret_20d` | +0.0033 |
| `fp_eps_yoy_growth_sec` | -0.0027 |
| `fp_margin_trend_4q_sec` | +0.0023 |
| `fp_debt_to_equity_sec` | -0.0019 |
| `fp_fcf_yield_sec` | +0.0014 |
| `fp_current_ratio_sec` | -0.0013 |
| `fp_roe_sec` | +0.0012 |
| `fp_gross_margin_sec` | +0.0012 |
| `fp_rev_yoy_growth_sec` | -0.0010 |
| `fp_net_margin_sec` | +0.0009 |
| `rs_rank_among_sectors` | +0.0009 |
| `fp_fcf_yoy_growth_sec` | +0.0009 |
| `momentum_cross_20_60` | +0.0009 |
