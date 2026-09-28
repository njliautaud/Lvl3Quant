# Sector picker v6 — Sparse-K (K=15, hold=10d)

**Filter**: univariate Spearman IC, top-15 per train fold (no OOT leakage)
**Fit**: ridge on selected K features
**Pool**: 76 candidate features (76-feature master_panel_v2 shelf)
**Walk-forward**: 36mo train / 12mo OOT, 6mo step

## Pooled-OOT

| Metric | Strategy | SPY 1x | SPY 1.5x | SPY 2x |
|---|---:|---:|---:|---:|
| Sharpe | 0.35 | 0.69 | 0.62 | 0.58 |
| Sortino | 0.52 | 0.83 | 0.75 | 0.71 |
| CAGR | 1.8% | 12.2% | 14.8% | 16.3% |
| MaxDD | -13.0% | -34.1% | -47.8% | -59.4% |
| Calmar | 0.14 | 0.36 | 0.31 | 0.27 |

**Deployable sectors (Calmar >= 1.0)**: 0 / 11

## Per-sector pooled-OOT

| Sector | Sharpe | CAGR | MaxDD | Calmar | PF | WR |
|---|---:|---:|---:|---:|---:|---:|
| Energy | 0.92 | 11.7% | -20.6% | 0.57 | 1.19 | 46.3% |
| Utilities | 0.49 | 4.0% | -18.6% | 0.21 | 1.10 | 45.7% |
| Real Estate | 0.39 | 3.1% | -18.8% | 0.16 | 1.07 | 45.3% |
| Basic Materials | 0.30 | 2.7% | -23.0% | 0.12 | 1.05 | 45.6% |
| Industrials | 0.21 | 1.8% | -28.3% | 0.06 | 1.04 | 46.2% |
| Communication Services | 0.15 | 1.1% | -38.4% | 0.03 | 1.03 | 45.4% |
| Technology | 0.11 | 0.4% | -44.6% | 0.01 | 1.02 | 44.4% |
| Financial Services | -0.01 | -1.0% | -40.7% | -0.02 | 1.00 | 45.1% |
| Consumer Cyclical | -0.01 | -1.9% | -73.0% | -0.03 | 1.00 | 45.9% |
| Consumer Defensive | -0.29 | -3.4% | -29.2% | -0.12 | 0.95 | 43.6% |
| Healthcare | -0.35 | -5.8% | -53.5% | -0.11 | 0.94 | 44.3% |

## Top-20 most-frequently-picked features (across folds x sectors)

| Rank | Feature | Pick count |
|---:|---|---:|
| 1 | `flag_accounting_change` | 101 |
| 2 | `rv_cc_252d` | 99 |
| 3 | `rv_pk_20d` | 88 |
| 4 | `rv_cc_60d` | 86 |
| 5 | `fp_eps_ttm` | 84 |
| 6 | `rv_yz_20d` | 82 |
| 7 | `rv_yz_60d` | 81 |
| 8 | `rv_cc_20d` | 80 |
| 9 | `fp_rev_yoy_growth` | 79 |
| 10 | `fp_market_cap_pit` | 77 |
| 11 | `fp_ni_ttm` | 76 |
| 12 | `fp_margin_trend_4q` | 75 |
| 13 | `flag_restatement` | 75 |
| 14 | `fp_revenue_ttm` | 74 |
| 15 | `fp_debt_to_equity` | 73 |
| 16 | `fp_ebitda_ttm` | 73 |
| 17 | `fp_fcf_yoy_growth` | 72 |
| 18 | `fp_ebitda_margin` | 71 |
| 19 | `fp_fcf_ttm` | 70 |
| 20 | `fp_net_margin` | 70 |
