# Wheel Strategy Tier Ladder — Comparative Report
Pricing: **modeled_bs_calibrated** (MODELED unless real chain data wired in)
Window: **2020-01-01 -> 2025-12-31**  Capital per tier: **$100,000**

## MTM (mark-to-market) metrics

| Tier | Target % | Universe | CAGR % | Sharpe | Sortino | Max DD % | PF | WR % | Trades | Assign % |
|------|---------:|---------:|-------:|-------:|--------:|---------:|---:|-----:|-------:|---------:|
| Tier1_Conservative_FW | 10 | 14 | 8.05 | 0.99 | 0.99 | -12.19 | 3.11 | 94.0 | 645 | 2.3 |
| Tier2_Balanced_FW | 15 | 48 | 20.05 | 1.57 | 1.48 | -24.83 | 3.45 | 93.8 | 854 | 1.9 |
| Tier3_Income_FW | 22 | 28 | 21.90 | 1.76 | 1.79 | -22.30 | 2.96 | 89.8 | 888 | 2.9 |
| Tier4_Aggressive_FW | 32 | 16 | 9.90 | 0.59 | 0.62 | -47.39 | 1.32 | 84.7 | 757 | 10.7 |
| Tier5_Turbo_FW | 45 | 13 | 11.47 | 0.76 | 0.82 | -32.14 | 1.37 | 81.8 | 1118 | 12.9 |

## REALIZED-CASH metrics (account cash, no MTM noise)

| Tier | Realized CAGR % | Realized Sharpe | Realized Sortino | Realized Max DD % | Final Equity $ |
|------|----------------:|----------------:|-----------------:|------------------:|---------------:|
| Tier1_Conservative_FW | 8.54 | 2.18 | 0.70 | -7.86 | $163,265 |
| Tier2_Balanced_FW | 20.71 | 2.70 | 0.76 | -16.80 | $308,225 |
| Tier3_Income_FW | 24.02 | 2.19 | 0.90 | -15.19 | $362,350 |
| Tier4_Aggressive_FW | 12.21 | 0.39 | 0.30 | -63.43 | $199,151 |
| Tier5_Turbo_FW | 13.88 | 1.04 | 0.57 | -28.43 | $217,555 |

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
