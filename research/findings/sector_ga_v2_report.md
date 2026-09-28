# Sector GA Sweep — v2 (HC #559 + HC #561 R4)

Generated: 2026-06-07 03:51:41

Universe: **70 names** in `wheel_strategy_v1/data/cache/universe.parquet (70-name baseline) + sp500_wikipedia (503 names — list-only, not yet priced)`.
Costs: 5 bps round-trip per turnover (ADV >= $20M), 30 bps otherwise.
Rebalance: weekly, long-top-decile / short-bottom-decile.
Regime EXCLUDED from per-name score (HC #561 R2).

## Verdict Table (HC #561 R4 acceptance gates)

| Sector | n | folds | verdict | WF med Sharpe | WF med Calmar | WF med CAGR | WF med MaxDD |
|---|---|---|---|---|---|---|---|
| Technology | 17 | 15 | FAILS CALMAR FLOOR | -1.359 | -0.782 | -0.183 | -0.242 |
| Financial Services | 15 | 15 | FAILS CALMAR FLOOR | -1.009 | -0.692 | -0.127 | -0.163 |
| Consumer Cyclical | 9 | 0 | FAILS CALMAR FLOOR | - | - | - | - |
| Consumer Defensive | 7 | - | SKIPPED (n_tickers<8) | - | - | - | - |
| Communication Services | 7 | - | SKIPPED (n_tickers<8) | - | - | - | - |
| Healthcare | 6 | - | SKIPPED (n_tickers<8) | - | - | - | - |
| Industrials | 5 | - | SKIPPED (n_tickers<8) | - | - | - | - |

## Per-sector details

### Technology — verdict: FAILS CALMAR FLOOR

- n_tickers: 17 | features considered: 64 | n_folds: 15
- Top weights:
  - `theme_industrials_r60`: +0.941
  - `theme_healthcare_inflow_z20`: +0.750
- WF summary (median / p25 / p75):
  - sharpe: med=-1.359  p25=-1.932  p75=-0.864
  - sortino: med=-2.226  p25=-3.111  p75=-1.200
  - cagr: med=-0.183  p25=-0.253  p75=-0.148
  - max_dd: med=-0.242  p25=-0.276  p75=-0.195
  - calmar: med=-0.782  p25=-0.930  p75=-0.625
  - pf: med=0.798  p25=0.720  p75=0.855
  - wr: med=0.425  p25=0.408  p75=0.444

### Financial Services — verdict: FAILS CALMAR FLOOR

- n_tickers: 15 | features considered: 64 | n_folds: 15
- Top weights:
  - `theme_industrials_r60`: +0.941
  - `theme_healthcare_inflow_z20`: +0.750
- WF summary (median / p25 / p75):
  - sharpe: med=-1.009  p25=-1.691  p75=-0.265
  - sortino: med=-1.544  p25=-2.343  p75=-0.343
  - cagr: med=-0.127  p25=-0.162  p75=-0.048
  - max_dd: med=-0.163  p25=-0.229  p75=-0.134
  - calmar: med=-0.692  p25=-0.854  p75=-0.297
  - pf: med=0.848  p25=0.736  p75=0.950
  - wr: med=0.458  p25=0.452  p75=0.488

### Consumer Cyclical — verdict: FAILS CALMAR FLOOR

- n_tickers: 9 | features considered: 64 | n_folds: 0
- Top weights:
  - `theme_industrials_r60`: +0.941
  - `theme_healthcare_inflow_z20`: +0.750
- WF summary (median / p25 / p75):
  - sharpe: med=nan  p25=nan  p75=nan
  - sortino: med=nan  p25=nan  p75=nan
  - cagr: med=nan  p25=nan  p75=nan
  - max_dd: med=nan  p25=nan  p75=nan
  - calmar: med=nan  p25=nan  p75=nan
  - pf: med=nan  p25=nan  p75=nan
  - wr: med=nan  p25=nan  p75=nan
