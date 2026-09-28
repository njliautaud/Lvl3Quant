# ETF rotation v1 — sector SPDR cross-sectional ridge

**Config**: hold=21d, long top-2, long-only, txn=5bps
**Pooled OOT**: Sharpe **0.56** | Calmar **-0.98** | CAGR **-98.3%** | MaxDD -99.9% | WR 47.4%
**Deploy gate (Calmar >= 1.0)**: FAIL (-0.98)
**Folds**: 2 | **Annualised turnover (legs/yr)**: 84.9

## Per-fold OOT

| OOT Window | Sharpe | Calmar | CAGR | MaxDD | WR | n_rebal | alpha |
|---|---|---|---|---|---|---|---|
| 2024-06-07 → 2025-06-07 | 2.49 | 15.24 | 1287.3% | -84.5% | 51.4% | 12 | 0.1 |
| 2024-12-07 → 2025-12-07 | -0.41 | -1.00 | -100.0% | -99.9% | 45.1% | 12 | 0.1 |

## Feature importance (avg coef across folds)

| Feature | Avg coef |
|---|---|
| `fp_fcf_yield_sec` | +0.0139 |
| `fp_margin_trend_4q_sec` | +0.0138 |
| `fp_net_margin_sec` | +0.0135 |
| `fp_beat_rate_4q_sec` | -0.0115 |
| `fp_eps_yoy_growth_sec` | -0.0113 |
| `fp_current_ratio_sec` | -0.0106 |
| `fp_ebitda_margin_sec` | -0.0093 |
| `rel_strength_spy` | -0.0068 |
| `fp_rev_yoy_growth_sec` | -0.0056 |
| `fp_debt_to_equity_sec` | +0.0040 |
| `rs_rank_among_sectors` | +0.0037 |
| `fp_ni_yoy_growth_sec` | +0.0037 |
| `ret_60d` | -0.0037 |
| `momentum_cross_20_60` | +0.0037 |
| `ret_20d` | +0.0029 |
