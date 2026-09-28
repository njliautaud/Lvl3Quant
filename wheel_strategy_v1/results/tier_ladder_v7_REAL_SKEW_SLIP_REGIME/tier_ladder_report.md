# Wheel Strategy Tier Ladder — Comparative Report
Pricing: **real_blend_vendor_iv_modeled_fallback** (MODELED unless real chain data wired in)
Window: **2020-01-01 -> 2025-12-31**  Capital per tier: **$100,000**

## MTM (mark-to-market) metrics

| Tier | Target % | Universe | CAGR % | Sharpe | Sortino | Max DD % | PF | WR % | Trades | Assign % |
|------|---------:|---------:|-------:|-------:|--------:|---------:|---:|-----:|-------:|---------:|
| Tier1_Conservative_FW | 10 | 14 | -4.73 | 0.19 | 0.15 | -74.22 | 1.75 | 93.0 | 388 | 6.4 |
| Tier2_Balanced_FW | 15 | 48 | 8.75 | 0.47 | 0.41 | -82.21 | 2.08 | 91.1 | 570 | 4.6 |
| Tier3_Income_FW | 22 | 28 | 0.47 | 0.28 | 0.26 | -79.23 | 1.06 | 84.2 | 533 | 5.2 |
| Tier4_Aggressive_FW | 32 | 16 | 18.21 | 0.58 | 0.54 | -61.04 | 1.17 | 80.4 | 565 | 10.2 |
| Tier5_Turbo_FW | 45 | 13 | 13.00 | 0.51 | 0.45 | -56.63 | 1.22 | 78.7 | 728 | 11.7 |

## REALIZED-CASH metrics (account cash, no MTM noise)

| Tier | Realized CAGR % | Realized Sharpe | Realized Sortino | Realized Max DD % | Final Equity $ |
|------|----------------:|----------------:|-----------------:|------------------:|---------------:|
| Tier1_Conservative_FW | 3.34 | 0.85 | 0.18 | -5.72 | $121,723 |
| Tier2_Balanced_FW | 10.57 | 1.02 | 0.41 | -11.92 | $182,396 |
| Tier3_Income_FW | 1.01 | 0.14 | 0.05 | -45.24 | $106,221 |
| Tier4_Aggressive_FW | 7.56 | 0.36 | 0.19 | -65.73 | $154,627 |
| Tier5_Turbo_FW | 7.02 | 0.54 | 0.27 | -30.15 | $150,036 |

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
