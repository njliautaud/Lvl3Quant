# Wheel Strategy Tier Ladder — Comparative Report
Pricing: **modeled_bs_calibrated** (MODELED unless real chain data wired in)
Window: **2018-01-01 -> 2026-06-06**  Capital per tier: **$100,000**

| Tier | Target % | Universe | CAGR % | Sharpe | Sortino | Max DD % | PF | WR % | Trades | Assign % |
|------|---------:|---------:|-------:|-------:|--------:|---------:|---:|-----:|-------:|---------:|
| Tier1_Conservative_FW | 10 | 14 | 5.64 | 0.34 | 0.31 | -82.36 | 2.68 | 93.2 | 745 | 4.1 |
| Tier2_Balanced_FW | 15 | 48 | 9.73 | 0.44 | 0.40 | -69.45 | 2.27 | 91.9 | 1010 | 2.8 |
| Tier3_Income_FW | 22 | 28 | 13.85 | 0.52 | 0.49 | -58.97 | 1.81 | 87.6 | 986 | 5.1 |
| Tier4_Aggressive_FW | 32 | 16 | 12.22 | 0.49 | 0.43 | -53.94 | 1.63 | 84.5 | 717 | 8.5 |
| Tier5_Turbo_FW | 45 | 13 | 28.78 | 0.72 | 0.69 | -52.87 | 1.32 | 81.6 | 1074 | 12.8 |

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
