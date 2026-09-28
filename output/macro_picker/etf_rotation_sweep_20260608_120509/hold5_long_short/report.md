# ETF rotation v1 — sector SPDR cross-sectional ridge

**Config**: hold=5d, long top-2, short bot-2, txn=5bps
**Pooled OOT**: Sharpe **1.78** | Calmar **1.20** | CAGR **117.3%** | MaxDD -98.1% | WR 41.4%
**Deploy gate (Calmar >= 1.0)**: PASS (1.20)
**Folds**: 2 | **Annualised turnover (legs/yr)**: 605.4

## Per-fold OOT

| OOT Window | Sharpe | Calmar | CAGR | MaxDD | WR | n_rebal | alpha |
|---|---|---|---|---|---|---|---|
| 2024-06-07 → 2025-06-07 | 1.64 | 0.52 | 50.9% | -98.0% | 39.3% | 50 | 0.1 |
| 2024-12-07 → 2025-12-07 | 1.52 | 0.01 | 1.1% | -96.1% | 41.8% | 50 | 0.1 |

## Feature importance (avg coef across folds)

| Feature | Avg coef |
|---|---|
| `rel_strength_spy` | -0.0065 |
| `ret_20d` | +0.0059 |
| `fp_beat_rate_4q_sec` | -0.0033 |
| `fp_net_margin_sec` | +0.0031 |
| `fp_margin_trend_4q_sec` | +0.0029 |
| `fp_fcf_yield_sec` | +0.0029 |
| `fp_current_ratio_sec` | -0.0021 |
| `fp_eps_yoy_growth_sec` | -0.0020 |
| `fp_ebitda_margin_sec` | -0.0020 |
| `fp_rev_yoy_growth_sec` | -0.0011 |
| `momentum_cross_20_60` | +0.0011 |
| `fp_debt_to_equity_sec` | +0.0010 |
| `fp_fcf_yoy_growth_sec` | +0.0009 |
| `fp_roe_sec` | +0.0007 |
| `ret_60d` | -0.0006 |
