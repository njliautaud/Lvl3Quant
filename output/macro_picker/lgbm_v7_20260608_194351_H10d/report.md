# Sector picker v7 — LightGBM (HC #565 R2 / 2026-06-08 verdict)

Same panel as v5/v6 (master_panel_v2.parquet). Feature pool: 89 candidates.
Hold horizon: 10d. Walk-forward: 36mo train / 12mo OOT / 6mo step.

## Pooled equal-weight book
- Sharpe: -0.16
- Calmar: -0.04
- CAGR:   -0.8%
- MaxDD:  0.0%
- Deployable sectors (Calmar ≥ 1): **0 / 11**

## Per-sector pooled-OOT

| Sector | Sharpe | Calmar | CAGR | MaxDD | PF | WR |
|---|---|---|---|---|---|---|
| Basic Materials | -0.18 | -0.11 | -2.6% | 0.0% | 0.00 | 0.0% |
| Communication Services | 0.45 | 0.22 | 4.9% | 0.0% | 0.00 | 0.0% |
| Consumer Cyclical | -0.29 | -0.10 | -6.4% | 0.0% | 0.00 | 0.0% |
| Consumer Defensive | -0.35 | -0.13 | -3.8% | 0.0% | 0.00 | 0.0% |
| Energy | 0.39 | 0.20 | 4.9% | 0.0% | 0.00 | 0.0% |
| Financial Services | -0.61 | -0.14 | -8.4% | 0.0% | 0.00 | 0.0% |
| Healthcare | 0.07 | -0.00 | -0.0% | 0.0% | 0.00 | 0.0% |
| Industrials | -0.04 | -0.05 | -1.1% | 0.0% | 0.00 | 0.0% |
| Real Estate | -0.15 | -0.06 | -1.5% | 0.0% | 0.00 | 0.0% |
| Technology | -0.07 | -0.07 | -2.5% | 0.0% | 0.00 | 0.0% |
| Utilities | 0.15 | 0.04 | 0.9% | 0.0% | 0.00 | 0.0% |

## Top-15 features by average LGBM gain

| Feature | Avg gain |
|---|---|
| `rh_sentiment_volatility` | 3.8 |
| `flag_accounting_change` | 2.9 |
| `rh_mention_count` | 2.2 |
| `flag_restatement` | 1.9 |
| `rv_cc_252d` | 1.8 |
| `fp_beat_rate_4q` | 1.6 |
| `fp_ni_yoy_growth` | 1.6 |
| `rv_yz_60d` | 1.5 |
| `flag_going_concern` | 1.4 |
| `fp_fcf_yoy_growth` | 1.4 |
| `rv_cc_60d` | 1.3 |
| `rf_delta_z` | 1.2 |
| `fp_margin_trend_4q` | 1.2 |
| `fp_eps_yoy_growth` | 1.2 |
| `rh_upvote_sum` | 1.1 |