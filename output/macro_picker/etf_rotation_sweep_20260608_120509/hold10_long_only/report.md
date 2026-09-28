# ETF rotation v1 — sector SPDR cross-sectional ridge

**Config**: hold=10d, long top-2, long-only, txn=5bps
**Pooled OOT**: Sharpe **-0.13** | Calmar **-1.00** | CAGR **-99.7%** | MaxDD -100.0% | WR 43.8%
**Deploy gate (Calmar >= 1.0)**: FAIL (-1.00)
**Folds**: 2 | **Annualised turnover (legs/yr)**: 155.6

## Per-fold OOT

| OOT Window | Sharpe | Calmar | CAGR | MaxDD | WR | n_rebal | alpha |
|---|---|---|---|---|---|---|---|
| 2024-06-07 → 2025-06-07 | 1.32 | -0.72 | -70.5% | -97.3% | 47.6% | 25 | 0.1 |
| 2024-12-07 → 2025-12-07 | -1.24 | -1.00 | -100.0% | -100.0% | 40.2% | 25 | 0.1 |

## Feature importance (avg coef across folds)

| Feature | Avg coef |
|---|---|
| `rel_strength_spy` | -0.0171 |
| `ret_20d` | +0.0163 |
| `fp_fcf_yield_sec` | +0.0065 |
| `fp_margin_trend_4q_sec` | +0.0062 |
| `fp_beat_rate_4q_sec` | -0.0062 |
| `fp_net_margin_sec` | +0.0049 |
| `fp_eps_yoy_growth_sec` | -0.0047 |
| `fp_current_ratio_sec` | -0.0043 |
| `fp_rev_yoy_growth_sec` | -0.0028 |
| `fp_ebitda_margin_sec` | -0.0026 |
| `ret_60d` | -0.0023 |
| `fp_debt_to_equity_sec` | +0.0022 |
| `momentum_cross_20_60` | +0.0015 |
| `fp_roe_sec` | +0.0015 |
| `fp_fcf_yoy_growth_sec` | +0.0015 |
