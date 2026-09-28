# ETF rotation v1 — sector SPDR cross-sectional ridge

**Config**: hold=21d, long top-2, short bot-2, txn=5.0bps, vol_target=0.15, lev_clip=[0.25,2.00]

**HONEST DEPLOY GATE**: median per-fold Calmar **0.77** (need ≥ 1.0), worst-fold MaxDD **-12.3%** (need > -25%), folds passing Calmar≥1: **4/10** (need ≥ 50%).
**RESULT: FAIL**

_(For reference — pooled-PnL stats, NOT the deploy gate)_: Sharpe 0.30 | pooled-Calmar 0.24 | CAGR 3.5% | pooled-MaxDD -14.5% | WR 44.2%
**Folds**: 10 | **Annualised turnover (legs/yr)**: 226.1

## Per-fold OOT

| OOT Window | Sharpe | Calmar | CAGR | MaxDD | WR | n_rebal | alpha |
|---|---|---|---|---|---|---|---|
| 2023-06-07 → 2023-12-07 | -0.42 | -1.34 | -6.4% | -4.8% | 44.4% | 5 | 0.1 |
| 2023-09-07 → 2024-03-07 | 0.18 | 0.33 | 1.5% | -4.7% | 46.7% | 4 | 0.1 |
| 2023-12-07 → 2024-06-07 | -1.76 | -1.80 | -22.2% | -12.3% | 39.6% | 6 | 0.1 |
| 2024-03-07 → 2024-09-07 | 2.09 | 5.60 | 36.8% | -6.6% | 48.7% | 5 | 0.1 |
| 2024-06-07 → 2024-12-07 | 0.25 | 0.58 | 2.7% | -4.7% | 42.2% | 5 | 0.1 |
| 2024-09-07 → 2025-03-07 | 3.83 | 26.20 | 55.1% | -2.1% | 54.8% | 4 | 0.1 |
| 2024-12-07 → 2025-06-07 | 3.18 | 17.66 | 37.8% | -2.1% | 48.9% | 3 | 0.1 |
| 2025-03-07 → 2025-09-07 | 0.65 | 0.97 | 8.9% | -9.2% | 48.9% | 3 | 0.1 |
| 2025-06-07 → 2025-12-07 | -0.31 | -0.55 | -6.3% | -11.4% | 44.7% | 6 | 0.1 |
| 2025-09-07 → 2026-03-07 | 0.57 | 1.01 | 8.7% | -8.6% | 49.5% | 6 | 0.1 |

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
