# Validation pack — hold21_longonly + regime overlay

Book days: 419 | rebals: 47 | years: 1.66

## Regime split (SPY > 60d-MA = bull)
- Bull: n=377, Sharpe=2.03, WR=51.7%
- Bear: n=42, Sharpe=0.00, WR=0.0%
- |ΔSharpe|/max = 1.00 (note: cash-gated strategy → bear≈0 by construction)

## Bootstrap (B=5000) of per-fold Calmar median
- Folds: 10, observed median Calmar: 10.61
- 90% CI: [4.44, 14.14]
- P(median Calmar ≥ 1.0) = 99.3%
- P(median Calmar ≥ 2.0) = 99.2%

## Cost stress (linear txn-cost drag)
- Base (5bps): Sharpe 1.92 on 419 days
- @5bps → Sharpe 1.92
- @10bps → Sharpe 1.81
- @20bps → Sharpe 1.59
