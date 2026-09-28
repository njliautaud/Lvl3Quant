# Sector picker v6 — Sparse-K (K=8, hold=10d)

**Filter**: univariate Spearman IC, top-8 per train fold (no OOT leakage)
**Fit**: ridge on selected K features
**Pool**: 76 candidate features (76-feature master_panel_v2 shelf)
**Walk-forward**: 36mo train / 12mo OOT, 6mo step

## Pooled-OOT

| Metric | Strategy | SPY 1x | SPY 1.5x | SPY 2x |
|---|---:|---:|---:|---:|
| Sharpe | 0.43 | 0.69 | 0.62 | 0.58 |
| Sortino | 0.64 | 0.83 | 0.75 | 0.71 |
| CAGR | 2.2% | 12.2% | 14.8% | 16.3% |
| MaxDD | -13.8% | -34.1% | -47.8% | -59.4% |
| Calmar | 0.16 | 0.36 | 0.31 | 0.27 |

**Deployable sectors (Calmar >= 1.0)**: 0 / 11

## Per-sector pooled-OOT

| Sector | Sharpe | CAGR | MaxDD | Calmar | PF | WR |
|---|---:|---:|---:|---:|---:|---:|
| Communication Services | 0.90 | 11.0% | -20.9% | 0.52 | 1.18 | 46.1% |
| Energy | 0.75 | 9.8% | -23.2% | 0.42 | 1.15 | 46.9% |
| Utilities | 0.62 | 5.4% | -14.0% | 0.38 | 1.13 | 46.8% |
| Real Estate | 0.42 | 3.4% | -16.9% | 0.20 | 1.08 | 45.9% |
| Industrials | 0.17 | 1.4% | -25.5% | 0.05 | 1.03 | 45.3% |
| Technology | 0.14 | 0.9% | -38.3% | 0.02 | 1.03 | 44.2% |
| Healthcare | 0.11 | 0.6% | -38.4% | 0.02 | 1.02 | 46.1% |
| Consumer Defensive | -0.05 | -1.0% | -20.4% | -0.05 | 0.99 | 44.0% |
| Basic Materials | -0.16 | -2.2% | -28.9% | -0.08 | 0.97 | 43.7% |
| Consumer Cyclical | -0.20 | -5.5% | -62.4% | -0.09 | 0.96 | 44.3% |
| Financial Services | -0.42 | -6.6% | -53.5% | -0.12 | 0.92 | 44.9% |

## Top-20 most-frequently-picked features (across folds x sectors)

| Rank | Feature | Pick count |
|---:|---|---:|
| 1 | `flag_accounting_change` | 80 |
| 2 | `rv_cc_252d` | 59 |
| 3 | `fp_market_cap_pit` | 54 |
| 4 | `flag_restatement` | 50 |
| 5 | `rv_cc_60d` | 47 |
| 6 | `rv_pk_20d` | 47 |
| 7 | `fp_revenue_ttm` | 47 |
| 8 | `rv_yz_60d` | 46 |
| 9 | `fp_ni_ttm` | 44 |
| 10 | `fp_margin_trend_4q` | 44 |
| 11 | `flag_going_concern` | 43 |
| 12 | `fp_ebitda_ttm` | 42 |
| 13 | `fp_ebitda_margin` | 42 |
| 14 | `rv_cc_20d` | 41 |
| 15 | `fp_debt_to_equity` | 40 |
| 16 | `fp_current_ratio` | 39 |
| 17 | `fp_rev_yoy_growth` | 38 |
| 18 | `fp_net_margin` | 38 |
| 19 | `fp_fcf_yield` | 36 |
| 20 | `rv_yz_20d` | 36 |
