# Wheel Strategy Tier Ladder — Comparative Report
Pricing: **modeled_bs_calibrated** (MODELED unless real chain data wired in)
Window: **2020-01-01 -> 2025-12-31**  Capital per tier: **$100,000**

| Tier | Target % | Universe | CAGR % | Sharpe | Sortino | Max DD % | PF | WR % | Trades | Assign % |
|------|---------:|---------:|-------:|-------:|--------:|---------:|---:|-----:|-------:|---------:|
| Tier1_Conservative_FW | 10 | 14 | 9.35 | 0.45 | 0.42 | -45.32 | 3.00 | 94.2 | 466 | 2.4 |
| Tier2_Balanced_FW | 15 | 48 | 13.82 | 0.63 | 0.50 | -43.24 | 2.46 | 92.7 | 505 | 2.5 |
| Tier3_Income_FW | 22 | 28 | 12.38 | 0.52 | 0.47 | -50.16 | 2.10 | 89.7 | 658 | 4.0 |
| Tier4_Aggressive_FW | 32 | 16 | 16.42 | 0.57 | 0.49 | -53.82 | 1.65 | 84.8 | 605 | 8.4 |
| Tier5_Turbo_FW | 45 | 13 | 27.67 | 0.67 | 0.69 | -57.13 | 1.45 | 81.9 | 852 | 13.4 |

## Tier Descriptions
- **Tier1_Conservative_FW** (target 10%): Blue chips, 17Δ puts, 30-45 DTE, IV-rank > 30. Capital-preservation first. [FULL WHEEL: profit-take 0.65, roll DTE 1, allows assignment]  Default capital share: 35%
- **Tier2_Balanced_FW** (target 15%): Broad large-cap, 22Δ puts, 30-45 DTE, IV-rank > 25. Balanced yield/risk. [FULL WHEEL: profit-take 0.65, roll DTE 1, allows assignment]  Default capital share: 30%
- **Tier3_Income_FW** (target 22%): Higher-IV half of universe, 27Δ puts, 21-35 DTE, IV-rank > 40. Income focus. [FULL WHEEL: profit-take 0.65, roll DTE 1, allows assignment]  Default capital share: 20%
- **Tier4_Aggressive_FW** (target 32%): High-β + high-IV, 34Δ puts, 14-28 DTE, IV-rank > 50. Aggressive premium. [FULL WHEEL: profit-take 0.65, roll DTE 1, allows assignment]  Default capital share: 10%
- **Tier5_Turbo_FW** (target 45%): Weeklies on liquid high-IV, 40Δ puts, 7-14 DTE, IV-rank > 60. Turbo, regime-gated. [FULL WHEEL: profit-take 0.65, roll DTE 1, allows assignment]  Default capital share: 5%

## Recommended capital allocation (default ladder split)
| Tier | Default Share |
|---|---:|
| Tier1_Conservative_FW | 35% |
| Tier2_Balanced_FW | 30% |
| Tier3_Income_FW | 20% |
| Tier4_Aggressive_FW | 10% |
| Tier5_Turbo_FW | 5% |
