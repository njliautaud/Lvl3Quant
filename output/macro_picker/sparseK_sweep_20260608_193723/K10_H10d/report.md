# Sector picker v6 — Sparse-K (K=10, hold=10d)

**Filter**: univariate Spearman IC, top-10 per train fold (no OOT leakage)
**Fit**: ridge on selected K features
**Pool**: 89 candidate features (76-feature master_panel_v2 shelf)
**Walk-forward**: 36mo train / 12mo OOT, 6mo step

## Pooled-OOT

| Metric | Strategy | SPY 1x | SPY 1.5x | SPY 2x |
|---|---:|---:|---:|---:|
| Sharpe | 0.34 | 0.69 | 0.62 | 0.58 |
| Sortino | 0.53 | 0.83 | 0.75 | 0.71 |
| CAGR | 1.7% | 12.2% | 14.8% | 16.3% |
| MaxDD | -13.0% | -34.1% | -47.8% | -59.4% |
| Calmar | 0.13 | 0.36 | 0.31 | 0.27 |

**Deployable sectors (Calmar >= 1.0)**: 0 / 11

## Per-sector pooled-OOT

| Sector | Sharpe | CAGR | MaxDD | Calmar | PF | WR |
|---|---:|---:|---:|---:|---:|---:|
| Communication Services | 0.64 | 7.5% | -31.7% | 0.24 | 1.12 | 46.3% |
| Real Estate | 0.57 | 4.8% | -12.5% | 0.38 | 1.11 | 46.1% |
| Utilities | 0.53 | 4.6% | -13.7% | 0.34 | 1.11 | 46.0% |
| Energy | 0.49 | 5.6% | -20.0% | 0.28 | 1.10 | 46.8% |
| Financial Services | 0.06 | -0.1% | -32.9% | -0.00 | 1.01 | 46.0% |
| Industrials | 0.06 | -0.1% | -27.5% | -0.00 | 1.01 | 45.9% |
| Technology | -0.03 | -2.1% | -36.1% | -0.06 | 0.99 | 44.6% |
| Healthcare | -0.09 | -2.5% | -37.3% | -0.07 | 0.98 | 45.8% |
| Consumer Cyclical | -0.10 | -3.4% | -53.5% | -0.06 | 0.98 | 45.9% |
| Basic Materials | -0.11 | -1.8% | -34.6% | -0.05 | 0.98 | 44.7% |
| Consumer Defensive | -0.12 | -1.7% | -24.6% | -0.07 | 0.98 | 43.7% |

## Top-20 most-frequently-picked features (across folds x sectors)

| Rank | Feature | Pick count |
|---:|---|---:|
| 1 | `flag_accounting_change` | 77 |
| 2 | `rv_cc_252d` | 67 |
| 3 | `fp_market_cap_pit` | 61 |
| 4 | `rh_sentiment_volatility` | 58 |
| 5 | `flag_restatement` | 55 |
| 6 | `rv_cc_60d` | 54 |
| 7 | `rv_yz_60d` | 51 |
| 8 | `fp_margin_trend_4q` | 51 |
| 9 | `fp_ni_ttm` | 51 |
| 10 | `fp_debt_to_equity` | 50 |
| 11 | `fp_revenue_ttm` | 49 |
| 12 | `rv_pk_20d` | 48 |
| 13 | `rh_mention_count` | 45 |
| 14 | `fp_ebitda_ttm` | 45 |
| 15 | `flag_going_concern` | 44 |
| 16 | `rv_cc_20d` | 43 |
| 17 | `fp_current_ratio` | 42 |
| 18 | `fp_fcf_ttm` | 42 |
| 19 | `fp_fcf_yoy_growth` | 42 |
| 20 | `rv_yz_20d` | 42 |
