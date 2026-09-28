# Macro Picker v2 — Thematic Rotation + Acceleration

**Verdict: PICKER v2 LOSES — reasons: CAGR 13.6% <= margin-SPY 1.5x 18.5%; monthly return negative in some regime: {'green': '4.83%', 'red': '-3.93%', 'flat': '0.53%'}; regime sharpe asymmetric worst/best=0.08 < 0.50**

Window: 2020-01-01 → 2025-12-31, $100k starting. Monthly rebalance. Long-only.
Sectors selected by acceleration of 60d relative-strength vs SPY (forward-looking, catches rotation earlier).
Standalone thematic slots: SMH, SOXX, IGV, XBI, ARKK (top 2 by same acceleration signal).
Macro gate: 100% cash only if VIX > 40 OR (risk_off AND VIX > 30). Reduced 50% in risk_off otherwise.

Stand-downs: 6 / 71 rebalances.

## Headline

| Strategy | CAGR | Sharpe | Sortino | MaxDD |
|---|---|---|---|---|
| Picker v2 | 13.6% | 0.89 | 1.12 | -23.5% |
| SPY 1.0× | 14.8% | 0.77 | 0.95 | -33.7% |
| SPY 1.5× margin | 18.5% | 0.70 | 0.87 | -47.4% |
| SPY 2.0× margin | 21.0% | 0.67 | 0.83 | -58.9% |

## Regime Diagnostics (HC #557 R2)

Mean monthly return:
- green: 4.83% | red: -3.93% | flat: 0.53%

Stratified Sharpe:
- green: 15.38 | red: -14.33 | flat: 1.21
