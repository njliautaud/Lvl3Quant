# Wheel Strategy Tier Ladder — Comparative Report
Pricing: **modeled_bs_calibrated** (MODELED unless real chain data wired in)
Window: **2020-01-01 -> 2025-12-31**  Capital per tier: **$100,000**

| Tier | Target % | Universe | CAGR % | Sharpe | Sortino | Max DD % | PF | WR % | Trades | Assign % |
|------|---------:|---------:|-------:|-------:|--------:|---------:|---:|-----:|-------:|---------:|
| Tier1_Conservative | 10 | 14 | 4.02 | 0.80 | 0.62 | -11.30 | 2.43 | 91.3 | 459 | 0.0 |
| Tier2_Balanced | 15 | 48 | 15.40 | 1.75 | 1.76 | -15.52 | 4.66 | 91.3 | 622 | 0.0 |
| Tier3_Income | 22 | 28 | 15.48 | 1.28 | 1.21 | -19.99 | 2.33 | 86.6 | 672 | 0.0 |
| Tier4_Aggressive | 32 | 16 | 18.83 | 1.35 | 1.47 | -18.19 | 2.31 | 83.3 | 551 | 0.0 |
| Tier5_Turbo | 45 | 13 | 19.30 | 1.59 | 1.67 | -11.12 | 2.20 | 77.2 | 675 | 0.0 |

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
