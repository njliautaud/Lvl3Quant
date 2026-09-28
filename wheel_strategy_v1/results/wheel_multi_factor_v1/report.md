# Wheel Multi-Factor Universe — HC #558 R2

Window: 2020-01-01 → 2025-12-31, $100k. Real IV + skew + slippage + regime overlay.
Universe: composite-ranked top-25 from 70-name pool.

Composite weights:
  25% fund_score | 20% sector flow z | 15% sector RS acceleration |
  15% name 60d momentum | 10% vol stability | 15% macro health

## Selected universe
AAPL, ABBV, ADBE, AMD, BLK, CRM, GE, GOOGL, JNJ, JPM, LLY, MA, META, MSFT, NFLX, NVDA, PANW, SCHW, SHOP, SLB, SMCI, T, TSLA, V, VZ

## Headline metrics

| Metric | Multi-Factor Balanced | v7 broad Balanced (HC #557 baseline) |
|---|---|---|
| Realized CAGR | 8.2% | 10.6% |
| Realized Sharpe | 0.92 | 1.02 |
| Realized Sortino | 0.39 | 0.41 |
| Realized MaxDD | -8.9% | -11.9% |
| WR | 89.4% | n/a |
| Assignment rate | 5.5% | n/a |
| n_trades | 557 | n/a |

vs SPY 1.5x margin (Sharpe 0.70, MaxDD -47%): multi-factor wheel wins if Realized Sharpe > 0.70.
