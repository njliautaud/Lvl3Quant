# Quality-Growth Screen — HC #557 R2 Test

**Verdict: WHEEL LOSES — JUST USE MARGIN SPY (reasons: Sharpe 0.42 <= margin-SPY 1.5x 0.70; income negative in some bucket: {'green': 0.0, 'red': 0.0, 'flat': 0.0})**

Window: 2020-01-01 to 2025-12-31. Real IV + skew + slippage + regime overlay (v7 canonical). $100k.

## Universe
11 quality-growth names: ABBV, ADBE, CRM, GOOGL, JNJ, META, MSFT, NVDA, PLTR, SCHW, V

Filter: fund_score >= 60, ebitda_margin > 10%, debt_to_equity < 1.0, net_margin > 0, market_cap > $50B.

## Headline

| Metric | Balanced (broad) | Balanced_QG (quality-growth) | SPY 1.5x margin |
|---|---|---|---|
| CAGR | 10.5% | 8.8% | 18.5% |
| Sharpe | 1.02 | 0.42 | 0.70 |
| Sortino | 0.41 | 0.43 | 0.87 |
| MaxDD | -11.9% | -40.6% | -47.3% |

## Monthly Realized Income by Regime
green: $0 | red: $0 | flat: $0

## Stratified Sharpe by Regime
green: nan | red: nan | flat: nan
