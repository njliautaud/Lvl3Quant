# Sector picker v6 — Sparse-K (K=15, hold=10d)

**Filter**: univariate Spearman IC, top-15 per train fold (no OOT leakage)
**Fit**: ridge on selected K features
**Pool**: 89 candidate features (76-feature master_panel_v2 shelf)
**Walk-forward**: 36mo train / 12mo OOT, 6mo step

## Pooled-OOT

| Metric | Strategy | SPY 1x | SPY 1.5x | SPY 2x |
|---|---:|---:|---:|---:|
| Sharpe | 0.19 | 0.69 | 0.62 | 0.58 |
| Sortino | 0.28 | 0.83 | 0.75 | 0.71 |
| CAGR | 0.9% | 12.2% | 14.8% | 16.3% |
| MaxDD | -14.9% | -34.1% | -47.8% | -59.4% |
| Calmar | 0.06 | 0.36 | 0.31 | 0.27 |

**Deployable sectors (Calmar >= 1.0)**: 0 / 11

## Per-sector pooled-OOT

| Sector | Sharpe | CAGR | MaxDD | Calmar | PF | WR |
|---|---:|---:|---:|---:|---:|---:|
| Real Estate | 0.66 | 5.6% | -13.3% | 0.42 | 1.13 | 47.2% |
| Utilities | 0.51 | 4.2% | -14.5% | 0.29 | 1.11 | 46.7% |
| Energy | 0.35 | 3.8% | -20.6% | 0.18 | 1.07 | 45.4% |
| Technology | 0.26 | 3.2% | -40.6% | 0.08 | 1.05 | 44.5% |
| Communication Services | 0.17 | 1.3% | -42.0% | 0.03 | 1.03 | 45.5% |
| Consumer Defensive | 0.12 | 0.7% | -20.9% | 0.03 | 1.02 | 46.1% |
| Industrials | 0.04 | -0.2% | -27.8% | -0.01 | 1.01 | 46.8% |
| Financial Services | -0.12 | -2.5% | -40.1% | -0.06 | 0.98 | 44.8% |
| Basic Materials | -0.19 | -2.6% | -29.1% | -0.09 | 0.97 | 44.5% |
| Consumer Cyclical | -0.26 | -6.4% | -68.4% | -0.09 | 0.95 | 44.0% |
| Healthcare | -0.29 | -5.1% | -48.1% | -0.11 | 0.95 | 45.4% |

## Top-20 most-frequently-picked features (across folds x sectors)

| Rank | Feature | Pick count |
|---:|---|---:|
| 1 | `flag_accounting_change` | 95 |
| 2 | `rv_cc_252d` | 92 |
| 3 | `rv_cc_60d` | 77 |
| 4 | `rv_yz_20d` | 76 |
| 5 | `rv_pk_20d` | 73 |
| 6 | `fp_market_cap_pit` | 71 |
| 7 | `fp_ni_ttm` | 71 |
| 8 | `rv_yz_60d` | 70 |
| 9 | `rv_cc_20d` | 70 |
| 10 | `fp_eps_ttm` | 70 |
| 11 | `flag_restatement` | 70 |
| 12 | `fp_margin_trend_4q` | 69 |
| 13 | `fp_debt_to_equity` | 68 |
| 14 | `fp_revenue_ttm` | 68 |
| 15 | `fp_rev_yoy_growth` | 67 |
| 16 | `rh_sentiment_volatility` | 67 |
| 17 | `fp_fcf_yield` | 64 |
| 18 | `fp_fcf_yoy_growth` | 64 |
| 19 | `fp_ebitda_ttm` | 64 |
| 20 | `fp_current_ratio` | 63 |
