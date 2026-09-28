# Sector picker v6 — Sparse-K (K=8, hold=10d)

**Filter**: univariate Spearman IC, top-8 per train fold (no OOT leakage)
**Fit**: ridge on selected K features
**Pool**: 89 candidate features (76-feature master_panel_v2 shelf)
**Walk-forward**: 36mo train / 12mo OOT, 6mo step

## Pooled-OOT

| Metric | Strategy | SPY 1x | SPY 1.5x | SPY 2x |
|---|---:|---:|---:|---:|
| Sharpe | 0.27 | 0.69 | 0.62 | 0.58 |
| Sortino | 0.40 | 0.83 | 0.75 | 0.71 |
| CAGR | 1.3% | 12.2% | 14.8% | 16.3% |
| MaxDD | -13.8% | -34.1% | -47.8% | -59.4% |
| Calmar | 0.09 | 0.36 | 0.31 | 0.27 |

**Deployable sectors (Calmar >= 1.0)**: 0 / 11

## Per-sector pooled-OOT

| Sector | Sharpe | CAGR | MaxDD | Calmar | PF | WR |
|---|---:|---:|---:|---:|---:|---:|
| Energy | 0.71 | 9.0% | -24.1% | 0.37 | 1.14 | 47.4% |
| Utilities | 0.68 | 5.7% | -14.0% | 0.40 | 1.14 | 46.6% |
| Communication Services | 0.61 | 7.3% | -32.6% | 0.22 | 1.12 | 45.8% |
| Real Estate | 0.56 | 4.7% | -11.9% | 0.40 | 1.11 | 46.6% |
| Technology | 0.07 | -0.3% | -39.0% | -0.01 | 1.01 | 43.1% |
| Industrials | 0.07 | -0.0% | -26.4% | -0.00 | 1.01 | 46.8% |
| Financial Services | -0.00 | -0.8% | -31.2% | -0.03 | 1.00 | 45.5% |
| Consumer Defensive | -0.18 | -2.2% | -24.3% | -0.09 | 0.97 | 43.0% |
| Consumer Cyclical | -0.19 | -5.0% | -60.5% | -0.08 | 0.97 | 44.5% |
| Basic Materials | -0.35 | -4.4% | -44.9% | -0.10 | 0.94 | 43.9% |
| Healthcare | -0.38 | -6.0% | -45.9% | -0.13 | 0.93 | 45.5% |

## Top-20 most-frequently-picked features (across folds x sectors)

| Rank | Feature | Pick count |
|---:|---|---:|
| 1 | `flag_accounting_change` | 73 |
| 2 | `rv_cc_252d` | 56 |
| 3 | `rh_sentiment_volatility` | 50 |
| 4 | `fp_market_cap_pit` | 49 |
| 5 | `flag_restatement` | 47 |
| 6 | `rv_cc_60d` | 45 |
| 7 | `flag_going_concern` | 42 |
| 8 | `rv_yz_60d` | 40 |
| 9 | `rv_pk_20d` | 40 |
| 10 | `fp_revenue_ttm` | 40 |
| 11 | `fp_margin_trend_4q` | 39 |
| 12 | `rh_mention_count` | 37 |
| 13 | `fp_ebitda_ttm` | 37 |
| 14 | `rv_cc_20d` | 36 |
| 15 | `fp_rev_yoy_growth` | 36 |
| 16 | `fp_current_ratio` | 35 |
| 17 | `fp_net_margin` | 34 |
| 18 | `fp_ni_ttm` | 34 |
| 19 | `fp_fcf_ttm` | 33 |
| 20 | `fp_debt_to_equity` | 33 |
