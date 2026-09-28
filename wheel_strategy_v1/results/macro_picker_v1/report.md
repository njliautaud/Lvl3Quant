# Macro Picker v1 — Sector Rotation

**Verdict: PICKER LOSES — reasons: CAGR 13.2% <= margin-SPY 1.5x 18.5%; monthly return negative in some regime: {'green': 0.029567046241660042, 'red': -0.02336756755867556, 'flat': 0.006863857981920467}; regime sharpe asymmetric worst/best=0.16 < 0.50**

Window: 2020-01-01 → 2025-12-31, $100k starting. Monthly rebalance. Long-only.
Top-3 sectors by 60d relative-strength + flow z-score; 5 highest-momentum names per sector.
Macro gate: stand down if regime_overlay.risk_off OR VIX > 35.

## Headline

| Strategy | CAGR | Sharpe | Sortino | MaxDD |
|---|---|---|---|---|
| Picker v1 | 13.2% | 0.91 | 0.85 | -17.9% |
| SPY 1.0× | 14.8% | 0.77 | 0.95 | -33.7% |
| SPY 1.5× margin | 18.5% | 0.70 | 0.87 | -47.4% |
| SPY 2.0× margin | 21.0% | 0.67 | 0.83 | -58.9% |

## Regime Diagnostics (HC #557 R2)

Mean monthly return by regime:
- green months: 2.96%
- red months:   -2.34%
- flat months:  0.69%

Stratified Sharpe by regime:
- green: 8.96
- red:   -7.62
- flat:  1.48

## Notes
- Costs: 1.0 bps one-way per trade (slippage + commission).
- No shorts in v1. No borrow cost.
- Top SaaS→AI/Semis rotation 2023-2024 should show up in holdings_log (look for XLK + SMH appearing).
