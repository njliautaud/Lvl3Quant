# Sector picker v6 — Sparse-K (K=5, hold=10d)

**Filter**: univariate Spearman IC, top-5 per train fold (no OOT leakage)
**Fit**: ridge on selected K features
**Pool**: 76 candidate features (76-feature master_panel_v2 shelf)
**Walk-forward**: 36mo train / 12mo OOT, 6mo step

## Pooled-OOT

| Metric | Strategy | SPY 1x | SPY 1.5x | SPY 2x |
|---|---:|---:|---:|---:|
| Sharpe | 0.53 | 0.69 | 0.62 | 0.58 |
| Sortino | 0.79 | 0.83 | 0.75 | 0.71 |
| CAGR | 2.7% | 12.2% | 14.8% | 16.3% |
| MaxDD | -13.1% | -34.1% | -47.8% | -59.4% |
| Calmar | 0.21 | 0.36 | 0.31 | 0.27 |

**Deployable sectors (Calmar >= 1.0)**: 0 / 11

## Per-sector pooled-OOT

| Sector | Sharpe | CAGR | MaxDD | Calmar | PF | WR |
|---|---:|---:|---:|---:|---:|---:|
| Communication Services | 0.84 | 10.6% | -20.3% | 0.52 | 1.17 | 48.0% |
| Energy | 0.70 | 9.0% | -17.6% | 0.51 | 1.14 | 46.5% |
| Utilities | 0.63 | 5.4% | -14.9% | 0.36 | 1.13 | 46.4% |
| Real Estate | 0.43 | 3.4% | -14.0% | 0.24 | 1.09 | 41.9% |
| Technology | 0.35 | 4.7% | -38.7% | 0.12 | 1.07 | 45.0% |
| Industrials | 0.10 | 0.5% | -39.4% | 0.01 | 1.02 | 44.4% |
| Healthcare | 0.02 | -0.6% | -34.6% | -0.02 | 1.00 | 46.0% |
| Consumer Defensive | -0.00 | -0.4% | -21.8% | -0.02 | 1.00 | 44.8% |
| Financial Services | -0.11 | -2.3% | -36.2% | -0.06 | 0.98 | 44.7% |
| Consumer Cyclical | -0.14 | -4.6% | -60.1% | -0.08 | 0.97 | 44.1% |
| Basic Materials | -0.19 | -2.7% | -34.5% | -0.08 | 0.97 | 43.9% |

## Top-20 most-frequently-picked features (across folds x sectors)

| Rank | Feature | Pick count |
|---:|---|---:|
| 1 | `flag_accounting_change` | 64 |
| 2 | `flag_restatement` | 41 |
| 3 | `rv_cc_252d` | 40 |
| 4 | `fp_margin_trend_4q` | 34 |
| 5 | `rv_pk_20d` | 34 |
| 6 | `fp_revenue_ttm` | 33 |
| 7 | `flag_going_concern` | 32 |
| 8 | `rv_yz_60d` | 31 |
| 9 | `rv_cc_60d` | 28 |
| 10 | `fp_net_margin` | 26 |
| 11 | `fp_ebitda_ttm` | 26 |
| 12 | `fp_current_ratio` | 24 |
| 13 | `fp_rev_yoy_growth` | 23 |
| 14 | `fp_ebitda_margin` | 23 |
| 15 | `fp_debt_to_equity` | 23 |
| 16 | `fp_market_cap_pit` | 22 |
| 17 | `fp_ni_yoy_growth` | 22 |
| 18 | `fp_fcf_yield` | 21 |
| 19 | `fp_fcf_ttm` | 21 |
| 20 | `fp_fcf_yoy_growth` | 21 |
