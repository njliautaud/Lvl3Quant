# Sector picker v6 — Sparse-K (K=5, hold=10d)

**Filter**: univariate Spearman IC, top-5 per train fold (no OOT leakage)
**Fit**: ridge on selected K features
**Pool**: 89 candidate features (76-feature master_panel_v2 shelf)
**Walk-forward**: 36mo train / 12mo OOT, 6mo step

## Pooled-OOT

| Metric | Strategy | SPY 1x | SPY 1.5x | SPY 2x |
|---|---:|---:|---:|---:|
| Sharpe | 0.30 | 0.69 | 0.62 | 0.58 |
| Sortino | 0.46 | 0.83 | 0.75 | 0.71 |
| CAGR | 1.5% | 12.2% | 14.8% | 16.3% |
| MaxDD | -13.9% | -34.1% | -47.8% | -59.4% |
| Calmar | 0.11 | 0.36 | 0.31 | 0.27 |

**Deployable sectors (Calmar >= 1.0)**: 0 / 11

## Per-sector pooled-OOT

| Sector | Sharpe | CAGR | MaxDD | Calmar | PF | WR |
|---|---:|---:|---:|---:|---:|---:|
| Real Estate | 0.79 | 6.8% | -11.7% | 0.58 | 1.17 | 42.1% |
| Communication Services | 0.61 | 7.6% | -27.3% | 0.28 | 1.12 | 47.8% |
| Technology | 0.38 | 5.4% | -38.2% | 0.14 | 1.07 | 45.3% |
| Energy | 0.31 | 3.3% | -34.1% | 0.10 | 1.06 | 45.9% |
| Utilities | 0.18 | 1.2% | -23.1% | 0.05 | 1.04 | 45.8% |
| Financial Services | 0.03 | -0.3% | -26.7% | -0.01 | 1.01 | 44.9% |
| Consumer Defensive | 0.01 | -0.4% | -22.4% | -0.02 | 1.00 | 45.1% |
| Healthcare | -0.14 | -2.9% | -41.9% | -0.07 | 0.97 | 45.6% |
| Consumer Cyclical | -0.19 | -4.9% | -62.0% | -0.08 | 0.97 | 43.9% |
| Industrials | -0.21 | -3.6% | -43.4% | -0.08 | 0.96 | 44.1% |
| Basic Materials | -0.22 | -3.1% | -36.2% | -0.09 | 0.96 | 43.1% |

## Top-20 most-frequently-picked features (across folds x sectors)

| Rank | Feature | Pick count |
|---:|---|---:|
| 1 | `flag_accounting_change` | 63 |
| 2 | `rh_sentiment_volatility` | 38 |
| 3 | `flag_restatement` | 38 |
| 4 | `rv_cc_252d` | 37 |
| 5 | `rv_pk_20d` | 34 |
| 6 | `fp_margin_trend_4q` | 33 |
| 7 | `rh_mention_count` | 30 |
| 8 | `flag_going_concern` | 30 |
| 9 | `fp_revenue_ttm` | 28 |
| 10 | `rv_yz_60d` | 26 |
| 11 | `fp_net_margin` | 23 |
| 12 | `fp_ebitda_ttm` | 23 |
| 13 | `rv_cc_60d` | 22 |
| 14 | `fp_current_ratio` | 21 |
| 15 | `fp_debt_to_equity` | 21 |
| 16 | `fp_fcf_yield` | 18 |
| 17 | `fp_market_cap_pit` | 18 |
| 18 | `fp_fcf_ttm` | 18 |
| 19 | `fp_fcf_yoy_growth` | 18 |
| 20 | `fp_rev_yoy_growth` | 18 |
