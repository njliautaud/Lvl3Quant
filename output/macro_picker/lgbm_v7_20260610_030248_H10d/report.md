# Sector picker v7 — LightGBM (HC #565 R2 / 2026-06-08 verdict)

Same panel as v5/v6 (master_panel_v2.parquet). Feature pool: 89 candidates.
Hold horizon: 10d. Walk-forward: 36mo train / 12mo OOT / 6mo step.

## Pooled equal-weight book
- Sharpe: 0.18
- Calmar: 0.07
- CAGR:   0.7%
- MaxDD:  -9.8%
- Deployable sectors (Calmar ≥ 1): **0 / 12**

## Per-sector pooled-OOT

| Sector | Sharpe | Calmar | CAGR | MaxDD | PF | WR |
|---|---|---|---|---|---|---|
| Basic Materials | -0.40 | -0.13 | -4.7% | -37.0% | 0.93 | 43.8% |
| Communication Services | 0.55 | 0.44 | 7.0% | -15.9% | 1.11 | 46.8% |
| Consumer Cyclical | -0.44 | -0.14 | -9.4% | -67.7% | 0.92 | 42.5% |
| Consumer Defensive | -0.18 | -0.06 | -2.0% | -32.2% | 0.97 | 44.5% |
| Energy | -0.28 | -0.11 | -4.4% | -40.9% | 0.95 | 42.8% |
| Financial Services | -0.27 | -0.09 | -4.2% | -48.1% | 0.95 | 44.7% |
| Healthcare | 0.18 | 0.05 | 1.5% | -28.7% | 1.03 | 45.8% |
| Industrials | -0.13 | -0.07 | -4.3% | -64.3% | 0.97 | 42.3% |
| Information Technology | 0.51 | 0.37 | 12.8% | -34.7% | 1.11 | 45.0% |
| Real Estate | 0.10 | 0.02 | 0.4% | -25.1% | 1.02 | 46.3% |
| Technology | 0.57 | 0.27 | 8.4% | -31.7% | 1.12 | 44.5% |
| Utilities | 0.22 | 0.07 | 1.8% | -27.4% | 1.05 | 43.7% |

## Top-15 features by average LGBM gain

| Feature | Avg gain |
|---|---|
| `ret` | 36.0 |
| `log_ret` | 31.0 |
| `rh_sentiment_volatility` | 8.0 |
| `rh_sentiment_polarity` | 7.2 |
| `rh_mention_count` | 7.0 |
| `rh_upvote_sum` | 6.1 |
| `rh_comment_count` | 5.7 |
| `flag_accounting_change` | 1.3 |
| `rv_cc_252d` | 1.2 |
| `fp_ni_yoy_growth` | 0.8 |
| `rv_pk_20d` | 0.8 |
| `rv_yz_60d` | 0.7 |
| `fp_margin_trend_4q` | 0.7 |
| `flag_restatement` | 0.7 |
| `fp_fcf_yoy_growth` | 0.7 |