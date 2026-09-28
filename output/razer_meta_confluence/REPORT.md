# Meta-Confluence Stream-Continuation Report

Compliance: HC #466 + #467 + #428 R1 + #344 + #468 + #469 R2 (5-OOT smoke until 40-day rerun).

## Setup
- Source: razer_meta_confluence_train.py — 32 v4 heads + 88 pair-confluence indicators (binary + signed for top-50 robust pairs).
- Models: ['xgb', 'mlp'].  Walk-forward: 15-day-train / 1-day-test sliding, 17 OOT folds.
- Eval rows total: 60.  Rows passing regime + day_conc≤0.70: 8.

## Top 10 by Sharpe-per-trade (regime-pass, day_conc≤0.70)

| model | conf cut | exit_M | floor | n_trades | mean hold (s) | WR | mean net ticks | Sharpe | regime_skew | day_conc |
|-------|----------|--------|-------|----------|---------------|-----|-----------------|--------|-------------|----------|
| xgb | top 20.0% | 3 | 0.50 | 42914 | 27.3 | 39.6% | -0.203 | -0.520 | 0.35 | 0.14 |
| xgb | top 10.0% | 3 | 0.50 | 29372 | 24.2 | 42.8% | -0.214 | -0.558 | 0.37 | 0.14 |
| xgb | top 20.0% | 2 | 0.25 | 60199 | 15.6 | 47.0% | -0.230 | -0.627 | 0.48 | 0.17 |
| xgb | top 20.0% | 2 | 0.50 | 61843 | 14.6 | 47.3% | -0.230 | -0.637 | 0.44 | 0.18 |
| mlp | top 10.0% | 3 | 0.25 | 27559 | 27.5 | 39.4% | -0.272 | -0.698 | 0.07 | 0.20 |
| mlp | top 20.0% | 2 | 0.25 | 57669 | 15.6 | 46.7% | -0.259 | -0.705 | 0.38 | 0.21 |
| mlp | top 20.0% | 2 | 0.50 | 58900 | 14.9 | 47.0% | -0.261 | -0.721 | 0.39 | 0.21 |
| xgb | top 20.0% | 5 | 0.25 | 29268 | 48.9 | 15.8% | -0.240 | -0.846 | 0.45 | 0.12 |
