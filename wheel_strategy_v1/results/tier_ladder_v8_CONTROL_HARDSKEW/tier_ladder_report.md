# Wheel Strategy Tier Ladder — Comparative Report
Pricing: **real_blend_vendor_iv_modeled_fallback** (MODELED unless real chain data wired in)
Window: **2020-01-01 -> 2025-12-31**  Capital per tier: **$100,000**

## MTM (mark-to-market) metrics

| Tier | Target % | Universe | CAGR % | Sharpe | Sortino | Max DD % | PF | WR % | Trades | Assign % |
|------|---------:|---------:|-------:|-------:|--------:|---------:|---:|-----:|-------:|---------:|
| Tier1_Conservative_FW | 10 | 14 | 4.86 | 0.47 | 0.46 | -18.82 | 1.80 | 92.9 | 462 | 6.0 |
| Tier2_Balanced_FW | 15 | 48 | 7.46 | 0.56 | 0.54 | -28.10 | 1.79 | 90.6 | 626 | 4.8 |
| Tier3_Income_FW | 22 | 28 | 2.48 | 0.24 | 0.22 | -30.71 | 1.36 | 85.5 | 511 | 6.1 |
| Tier4_Aggressive_FW | 32 | 16 | 3.83 | 0.30 | 0.28 | -32.71 | 1.10 | 83.4 | 505 | 9.2 |
| Tier5_Turbo_FW | 45 | 13 | 2.67 | 0.28 | 0.27 | -22.90 | 1.18 | 77.6 | 688 | 10.8 |

## REALIZED-CASH metrics (account cash, no MTM noise)

| Tier | Realized CAGR % | Realized Sharpe | Realized Sortino | Realized Max DD % | Final Equity $ |
|------|----------------:|----------------:|-----------------:|------------------:|---------------:|
| Tier1_Conservative_FW | 4.43 | 0.96 | 0.26 | -7.58 | $129,565 |
| Tier2_Balanced_FW | 8.81 | 1.10 | 0.36 | -12.09 | $165,696 |
| Tier3_Income_FW | 4.62 | 0.56 | 0.15 | -22.48 | $131,014 |
| Tier4_Aggressive_FW | 3.87 | 0.27 | 0.16 | -57.71 | $125,493 |
| Tier5_Turbo_FW | 5.16 | 0.51 | 0.26 | -17.47 | $135,141 |

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
