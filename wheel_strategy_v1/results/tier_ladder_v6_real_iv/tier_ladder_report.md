# Wheel Strategy Tier Ladder — Comparative Report
Pricing: **modeled_bs_calibrated** (MODELED unless real chain data wired in)
Window: **2020-01-01 -> 2025-12-31**  Capital per tier: **$100,000**

| Tier | Target % | Universe | CAGR % | Sharpe | Sortino | Max DD % | PF | WR % | Trades | Assign % |
|------|---------:|---------:|-------:|-------:|--------:|---------:|---:|-----:|-------:|---------:|
| Tier1_Conservative | 10 | 14 | 2.89 | 0.51 | 0.43 | -11.55 | 1.79 | 88.3 | 472 | 0.0 |
| Tier2_Balanced | 15 | 48 | 5.57 | 0.51 | 0.41 | -23.61 | 1.57 | 87.7 | 576 | 0.0 |
| Tier3_Income | 22 | 28 | 8.05 | 0.68 | 0.60 | -22.14 | 1.59 | 84.3 | 626 | 0.0 |
| Tier4_Aggressive | 32 | 16 | 15.06 | 1.12 | 1.16 | -18.24 | 1.90 | 82.2 | 572 | 0.0 |
| Tier5_Turbo | 45 | 13 | 12.13 | 1.13 | 1.15 | -16.03 | 1.61 | 74.3 | 689 | 0.0 |

## Tier Descriptions
- **Tier1_Conservative** (target 10%): Blue chips, 17Δ puts, 30-45 DTE, IV-rank > 30. Capital-preservation first.  Default capital share: 35%
- **Tier2_Balanced** (target 15%): Broad large-cap, 22Δ puts, 30-45 DTE, IV-rank > 25. Balanced yield/risk.  Default capital share: 30%
- **Tier3_Income** (target 22%): Higher-IV half of universe, 27Δ puts, 21-35 DTE, IV-rank > 40. Income focus.  Default capital share: 20%
- **Tier4_Aggressive** (target 32%): High-β + high-IV, 34Δ puts, 14-28 DTE, IV-rank > 50. Aggressive premium.  Default capital share: 10%
- **Tier5_Turbo** (target 45%): Weeklies on liquid high-IV, 40Δ puts, 7-14 DTE, IV-rank > 60. Turbo, regime-gated.  Default capital share: 5%

## Recommended capital allocation (default ladder split)
| Tier | Default Share |
|---|---:|
| Tier1_Conservative | 35% |
| Tier2_Balanced | 30% |
| Tier3_Income | 20% |
| Tier4_Aggressive | 10% |
| Tier5_Turbo | 5% |
