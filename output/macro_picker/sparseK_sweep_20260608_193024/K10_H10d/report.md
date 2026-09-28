# Sector picker v6 — Sparse-K (K=10, hold=10d)

**Filter**: univariate Spearman IC, top-10 per train fold (no OOT leakage)
**Fit**: ridge on selected K features
**Pool**: 76 candidate features (76-feature master_panel_v2 shelf)
**Walk-forward**: 36mo train / 12mo OOT, 6mo step

## Pooled-OOT

| Metric | Strategy | SPY 1x | SPY 1.5x | SPY 2x |
|---|---:|---:|---:|---:|
| Sharpe | 0.36 | 0.69 | 0.62 | 0.58 |
| Sortino | 0.54 | 0.83 | 0.75 | 0.71 |
| CAGR | 1.9% | 12.2% | 14.8% | 16.3% |
| MaxDD | -13.1% | -34.1% | -47.8% | -59.4% |
| Calmar | 0.14 | 0.36 | 0.31 | 0.27 |

**Deployable sectors (Calmar >= 1.0)**: 0 / 11

## Per-sector pooled-OOT

| Sector | Sharpe | CAGR | MaxDD | Calmar | PF | WR |
|---|---:|---:|---:|---:|---:|---:|
| Communication Services | 0.83 | 9.7% | -20.0% | 0.48 | 1.16 | 46.8% |
| Energy | 0.80 | 10.1% | -13.8% | 0.74 | 1.16 | 48.0% |
| Utilities | 0.49 | 4.1% | -20.3% | 0.20 | 1.10 | 46.4% |
| Industrials | 0.27 | 2.7% | -18.7% | 0.15 | 1.05 | 46.6% |
| Real Estate | 0.27 | 2.1% | -21.9% | 0.09 | 1.05 | 44.9% |
| Technology | 0.06 | -0.6% | -44.4% | -0.01 | 1.01 | 44.7% |
| Basic Materials | -0.07 | -1.4% | -32.3% | -0.04 | 0.99 | 45.2% |
| Healthcare | -0.10 | -2.5% | -44.0% | -0.06 | 0.98 | 44.6% |
| Financial Services | -0.14 | -3.0% | -46.8% | -0.06 | 0.97 | 45.4% |
| Consumer Defensive | -0.14 | -2.0% | -24.5% | -0.08 | 0.97 | 44.7% |
| Consumer Cyclical | -0.21 | -6.0% | -62.0% | -0.10 | 0.96 | 44.6% |

## Top-20 most-frequently-picked features (across folds x sectors)

| Rank | Feature | Pick count |
|---:|---|---:|
| 1 | `flag_accounting_change` | 86 |
| 2 | `rv_cc_252d` | 74 |
| 3 | `fp_market_cap_pit` | 63 |
| 4 | `fp_margin_trend_4q` | 61 |
| 5 | `fp_ni_ttm` | 59 |
| 6 | `flag_restatement` | 59 |
| 7 | `rv_cc_60d` | 58 |
| 8 | `fp_debt_to_equity` | 55 |
| 9 | `rv_yz_60d` | 55 |
| 10 | `rv_pk_20d` | 55 |
| 11 | `fp_revenue_ttm` | 54 |
| 12 | `rv_yz_20d` | 52 |
| 13 | `rv_cc_20d` | 51 |
| 14 | `fp_rev_yoy_growth` | 50 |
| 15 | `fp_ebitda_ttm` | 50 |
| 16 | `fp_fcf_ttm` | 47 |
| 17 | `fp_eps_ttm` | 47 |
| 18 | `fp_net_margin` | 47 |
| 19 | `flag_going_concern` | 47 |
| 20 | `fp_current_ratio` | 46 |
