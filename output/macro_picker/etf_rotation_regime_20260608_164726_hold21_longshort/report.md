# ETF rotation v1 — sector SPDR cross-sectional ridge

**Config**: hold=21d, long top-2, short bot-2, txn=5.0bps, vol_target=0.15, lev_clip=[0.25,2.00]

**HONEST DEPLOY GATE**: median per-fold Calmar **0.69** (need ≥ 1.0), worst-fold MaxDD **-12.2%** (need > -25%), folds passing Calmar≥1: **3/10** (need ≥ 50%).
**RESULT: FAIL**

_(For reference — pooled-PnL stats, NOT the deploy gate)_: Sharpe -0.08 | pooled-Calmar -0.12 | CAGR -2.1% | pooled-MaxDD -18.0% | WR 38.2%
**Folds**: 10 | **Annualised turnover (legs/yr)**: 226.1

## Per-fold OOT

| OOT Window | Sharpe | Calmar | CAGR | MaxDD | WR | n_rebal | alpha |
|---|---|---|---|---|---|---|---|
| 2023-06-07 → 2023-12-07 | -0.81 | -1.86 | -10.5% | -5.7% | 36.5% | 5 | 0.1 |
| 2023-09-07 → 2024-03-07 | 0.04 | -0.06 | -0.3% | -4.7% | 45.0% | 4 | 0.1 |
| 2023-12-07 → 2024-06-07 | -2.48 | -2.08 | -25.3% | -12.2% | 31.9% | 6 | 0.1 |
| 2024-03-07 → 2024-09-07 | 2.46 | 8.42 | 42.8% | -5.1% | 43.6% | 5 | 0.1 |
| 2024-06-07 → 2024-12-07 | 0.25 | 0.58 | 2.7% | -4.7% | 42.2% | 5 | 0.1 |
| 2024-09-07 → 2025-03-07 | 2.69 | 14.77 | 31.0% | -2.1% | 46.8% | 4 | 0.1 |
| 2024-12-07 → 2025-06-07 | 1.94 | 8.12 | 17.4% | -2.1% | 35.6% | 3 | 0.1 |
| 2025-03-07 → 2025-09-07 | 0.65 | 0.97 | 8.9% | -9.2% | 48.9% | 3 | 0.1 |
| 2025-06-07 → 2025-12-07 | -0.59 | -0.90 | -10.3% | -11.4% | 41.5% | 6 | 0.1 |
| 2025-09-07 → 2026-03-07 | 0.53 | 0.79 | 7.7% | -9.7% | 43.0% | 6 | 0.1 |

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
