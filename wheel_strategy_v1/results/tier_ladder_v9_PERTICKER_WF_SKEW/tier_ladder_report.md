# Wheel Strategy Tier Ladder — Comparative Report
Pricing: **real_blend_vendor_iv_modeled_fallback** (MODELED unless real chain data wired in)
Window: **2020-01-01 -> 2025-12-31**  Capital per tier: **$100,000**

## MTM (mark-to-market) metrics

| Tier | Target % | Universe | CAGR % | Sharpe | Sortino | Max DD % | PF | WR % | Trades | Assign % |
|------|---------:|---------:|-------:|-------:|--------:|---------:|---:|-----:|-------:|---------:|
| Tier1_Conservative_FW | 10 | 14 | 3.54 | 0.39 | 0.36 | -24.65 | 1.93 | 94.2 | 412 | 3.0 |
| Tier2_Balanced_FW | 15 | 48 | 11.46 | 0.82 | 0.77 | -26.80 | 2.45 | 91.2 | 602 | 5.3 |
| Tier3_Income_FW | 22 | 28 | 12.97 | 0.87 | 0.90 | -21.49 | 1.74 | 88.6 | 612 | 4.4 |
| Tier4_Aggressive_FW | 32 | 16 | 8.78 | 0.66 | 0.66 | -23.17 | 1.36 | 82.6 | 511 | 10.9 |
| Tier5_Turbo_FW | 45 | 13 | 5.71 | 0.52 | 0.51 | -20.86 | 1.32 | 79.6 | 737 | 12.5 |

## REALIZED-CASH metrics (account cash, no MTM noise)

| Tier | Realized CAGR % | Realized Sharpe | Realized Sortino | Realized Max DD % | Final Equity $ |
|------|----------------:|----------------:|-----------------:|------------------:|---------------:|
| Tier1_Conservative_FW | 3.62 | 0.87 | 0.20 | -15.30 | $123,699 |
| Tier2_Balanced_FW | 12.43 | 1.37 | 0.41 | -11.73 | $201,525 |
| Tier3_Income_FW | 14.21 | 0.77 | 0.32 | -35.11 | $221,322 |
| Tier4_Aggressive_FW | 8.59 | 0.65 | 0.27 | -30.12 | $163,727 |
| Tier5_Turbo_FW | 8.49 | 0.85 | 0.43 | -19.40 | $162,834 |

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
