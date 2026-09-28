# Wheel Strategy Tier Ladder — Comparative Report
Pricing: **modeled_bs_calibrated** (MODELED unless real chain data wired in)
Window: **2020-01-01 -> 2025-12-31**  Capital per tier: **$100,000**

| Tier | Target % | Universe | CAGR % | Sharpe | Sortino | Max DD % | PF | WR % | Trades | Assign % |
|------|---------:|---------:|-------:|-------:|--------:|---------:|---:|-----:|-------:|---------:|
| Tier1_Conservative_FW | 10 | 14 | 9.46 | 0.44 | 0.40 | -62.79 | 3.91 | 94.2 | 588 | 2.8 |
| Tier2_Balanced_FW | 15 | 48 | 17.91 | 0.62 | 0.54 | -56.13 | 3.32 | 92.8 | 869 | 3.8 |
| Tier3_Income_FW | 22 | 28 | 21.18 | 0.75 | 0.67 | -40.49 | 2.76 | 88.0 | 906 | 4.2 |
| Tier4_Aggressive_FW | 32 | 16 | 25.00 | 0.66 | 0.72 | -50.12 | 1.79 | 85.4 | 850 | 11.2 |
| Tier5_Turbo_FW | 45 | 13 | 41.28 | 0.83 | 0.92 | -70.00 | 1.56 | 82.4 | 1168 | 11.0 |

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
