# ETF rotation v1 — sector SPDR cross-sectional ridge

**Config**: hold=21d, long top-2, long-only, txn=5bps, vol_target=0.15, lev_clip=[0.25,2.00]

**HONEST DEPLOY GATE**: median per-fold Calmar **5.72** (need ≥ 1.0), worst-fold MaxDD **-11.0%** (need > -25%), folds passing Calmar≥1: **8/10** (need ≥ 50%).
**RESULT: PASS**

_(For reference — pooled-PnL stats, NOT the deploy gate)_: Sharpe 1.63 | pooled-Calmar 2.45 | CAGR 25.6% | pooled-MaxDD -10.4% | WR 52.8%
**Folds**: 10 | **Annualised turnover (legs/yr)**: 116.7

## Per-fold OOT

| OOT Window | Sharpe | Calmar | CAGR | MaxDD | WR | n_rebal | alpha |
|---|---|---|---|---|---|---|---|
| 2023-06-07 → 2023-12-07 | -0.30 | -0.50 | -5.5% | -11.0% | 45.3% | 7 | 0.1 |
| 2023-09-07 → 2024-03-07 | 1.43 | 3.20 | 23.1% | -7.2% | 52.2% | 6 | 0.1 |
| 2023-12-07 → 2024-06-07 | -0.40 | -1.00 | -6.5% | -6.4% | 50.5% | 6 | 0.1 |
| 2024-03-07 → 2024-09-07 | 3.55 | 13.38 | 53.6% | -4.0% | 53.7% | 7 | 0.1 |
| 2024-06-07 → 2024-12-07 | 5.00 | 19.32 | 71.4% | -3.7% | 58.3% | 7 | 0.1 |
| 2024-09-07 → 2025-03-07 | 2.47 | 5.53 | 38.3% | -6.9% | 58.7% | 6 | 0.1 |
| 2024-12-07 → 2025-06-07 | 0.91 | 1.79 | 14.0% | -7.8% | 58.2% | 6 | 0.1 |
| 2025-03-07 → 2025-09-07 | 2.04 | 9.14 | 37.5% | -4.1% | 52.7% | 6 | 0.1 |
| 2025-06-07 → 2025-12-07 | 1.68 | 5.91 | 26.0% | -4.4% | 51.1% | 6 | 0.1 |
| 2025-09-07 → 2026-03-07 | 3.27 | 11.53 | 74.1% | -6.4% | 58.1% | 6 | 0.1 |

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
