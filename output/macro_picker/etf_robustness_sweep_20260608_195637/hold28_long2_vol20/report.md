# ETF rotation v1 — sector SPDR cross-sectional ridge

**Config**: hold=28d, long top-2, long-only, txn=5.0bps, vol_target=0.20, lev_clip=[0.25,2.00]

**HONEST DEPLOY GATE**: median per-fold Calmar **12.06** (need ≥ 1.0), worst-fold MaxDD **-8.2%** (need > -25%), folds passing Calmar≥1: **9/10** (need ≥ 50%).
**RESULT: PASS**

_(For reference — pooled-PnL stats, NOT the deploy gate)_: Sharpe 2.58 | pooled-Calmar 4.15 | CAGR 49.8% | pooled-MaxDD -12.0% | WR 48.7%
**Folds**: 10 | **Annualised turnover (legs/yr)**: 75.2

## Per-fold OOT

| OOT Window | Sharpe | Calmar | CAGR | MaxDD | WR | n_rebal | alpha |
|---|---|---|---|---|---|---|---|
| 2023-06-07 → 2023-12-07 | 3.21 | 13.95 | 53.9% | -3.9% | 49.3% | 4 | 0.1 |
| 2023-09-07 → 2024-03-07 | 2.87 | 12.52 | 60.3% | -4.8% | 54.7% | 3 | 0.1 |
| 2023-12-07 → 2024-06-07 | 1.30 | 4.17 | 25.7% | -6.2% | 46.3% | 5 | 0.1 |
| 2024-03-07 → 2024-09-07 | 2.80 | 13.37 | 65.9% | -4.9% | 52.6% | 4 | 0.1 |
| 2024-06-07 → 2024-12-07 | 3.59 | 17.02 | 68.7% | -4.0% | 54.2% | 5 | 0.1 |
| 2024-09-07 → 2025-03-07 | 0.49 | 1.33 | 6.2% | -4.7% | 38.5% | 3 | 0.1 |
| 2024-12-07 → 2025-06-07 | 5.95 | 37.91 | 128.9% | -3.4% | 54.0% | 3 | 0.1 |
| 2025-03-07 → 2025-09-07 | 1.01 | 3.45 | 16.4% | -4.7% | 50.9% | 3 | 0.1 |
| 2025-06-07 → 2025-12-07 | 0.38 | 0.65 | 5.3% | -8.2% | 47.4% | 5 | 0.1 |
| 2025-09-07 → 2026-03-07 | 3.43 | 11.60 | 83.4% | -7.2% | 50.5% | 5 | 0.1 |

## Feature importance (avg coef across folds)

| Feature | Avg coef |
|---|---|
| `fp_beat_rate_4q_sec` | -0.0183 |
| `fp_eps_yoy_growth_sec` | -0.0169 |
| `fp_net_margin_sec` | +0.0166 |
| `fp_margin_trend_4q_sec` | +0.0136 |
| `fp_ebitda_margin_sec` | -0.0123 |
| `fp_current_ratio_sec` | -0.0102 |
| `rel_strength_spy` | -0.0093 |
| `fp_gross_margin_sec` | +0.0082 |
| `fp_fcf_yield_sec` | +0.0082 |
| `fp_debt_to_equity_sec` | -0.0073 |
| `fp_ni_yoy_growth_sec` | +0.0067 |
| `fp_rev_yoy_growth_sec` | -0.0053 |
| `ret_60d` | -0.0048 |
| `momentum_cross_20_60` | +0.0044 |
| `rs_rank_among_sectors` | +0.0040 |
