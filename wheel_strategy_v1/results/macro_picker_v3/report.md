# Macro Picker v3 — Hedged with TLT + GLD

**Verdict: PICKER v3 LOSES — reasons: CAGR 12.3% <= 1.5x SPY 18.5%; red-month -2.55% still negative; regime sharpe asymmetric 0.10 < 0.50**

Window: 2020-01-01 → 2025-12-31, $100k starting. Monthly rebalance.
Normal regime: same equity-rotation logic as v2.
Risk-off (VIX > 30 OR regime_overlay.risk_off): 50% TLT + 25% GLD + 25% cash.
Hard gate (VIX > 40): 60% TLT + 40% GLD.

Regime state distribution: normal 38, risk-off 32, hard 1 of 71 rebalances.

## Headline

| Strategy | CAGR | Sharpe | Sortino | MaxDD |
|---|---|---|---|---|
| Picker v3 | 12.3% | 0.88 | 1.13 | -18.7% |
| SPY 1.0× | 14.8% | 0.77 | 0.95 | -33.7% |
| SPY 1.5× margin | 18.5% | 0.70 | 0.87 | -47.4% |

## Regime Diagnostics (HC #557 R2)

Mean monthly return:
- green: 3.35% | red: -2.55% | flat: 0.44%

Stratified Sharpe:
- green: 10.04 | red: -8.99 | flat: 1.02
