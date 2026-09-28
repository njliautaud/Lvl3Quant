# Sector picker v7 — LightGBM (HC #565 R2 / 2026-06-08 verdict)

Same panel as v5/v6 (master_panel_v2.parquet). Feature pool: 89 candidates.
Hold horizon: 10d. Walk-forward: 36mo train / 12mo OOT / 6mo step.

## Pooled equal-weight book
- Sharpe: -0.04
- Calmar: -0.01
- CAGR:   -0.2%
- MaxDD:  0.0%
- Deployable sectors (Calmar ≥ 1): **0 / 11**

## Per-sector pooled-OOT

| Sector | Sharpe | Calmar | CAGR | MaxDD | PF | WR |
|---|---|---|---|---|---|---|
| Basic Materials | -0.38 | -0.12 | -4.6% | 0.0% | 0.00 | 0.0% |
| Communication Services | 0.94 | 0.77 | 11.5% | 0.0% | 0.00 | 0.0% |
| Consumer Cyclical | -0.47 | -0.15 | -8.7% | 0.0% | 0.00 | 0.0% |
| Consumer Defensive | 0.07 | 0.01 | 0.2% | 0.0% | 0.00 | 0.0% |
| Energy | 0.59 | 0.37 | 7.3% | 0.0% | 0.00 | 0.0% |
| Financial Services | -0.59 | -0.14 | -8.6% | 0.0% | 0.00 | 0.0% |
| Healthcare | -0.07 | -0.05 | -2.0% | 0.0% | 0.00 | 0.0% |
| Industrials | -0.50 | -0.15 | -6.1% | 0.0% | 0.00 | 0.0% |
| Real Estate | 0.06 | 0.01 | 0.2% | 0.0% | 0.00 | 0.0% |
| Technology | 0.24 | 0.06 | 2.6% | 0.0% | 0.00 | 0.0% |
| Utilities | 0.03 | -0.01 | -0.1% | 0.0% | 0.00 | 0.0% |

## Top-15 features by average LGBM gain

| Feature | Avg gain |
|---|---|
| `rh_sentiment_volatility` | 3.1 |
| `flag_accounting_change` | 2.6 |
| `rh_mention_count` | 2.4 |
| `rv_cc_252d` | 1.7 |
| `flag_restatement` | 1.6 |
| `fp_ni_yoy_growth` | 1.5 |
| `fp_fcf_yoy_growth` | 1.4 |
| `rv_yz_60d` | 1.4 |
| `flag_going_concern` | 1.1 |
| `fp_debt_to_equity` | 1.1 |
| `fp_rev_yoy_growth` | 1.1 |
| `rh_upvote_sum` | 1.1 |
| `fp_margin_trend_4q` | 1.1 |
| `rf_delta_z` | 1.1 |
| `rv_cc_60d` | 1.1 |