# Wheel Strategy Tier Ladder — Comparative Report
Pricing: **real_blend_vendor_iv_modeled_fallback** (MODELED unless real chain data wired in)
Window: **2020-01-01 -> 2025-12-31**  Capital per tier: **$100,000**

## MTM (mark-to-market) metrics

| Tier | Target % | Universe | CAGR % | Sharpe | Sortino | Max DD % | PF | WR % | Trades | Assign % |
|------|---------:|---------:|-------:|-------:|--------:|---------:|---:|-----:|-------:|---------:|
| Tier1_Conservative_FW | 10 | 14 | 5.30 | 0.57 | 0.57 | -16.62 | 2.23 | 93.4 | 473 | 6.3 |
| Tier2_Balanced_FW | 15 | 48 | 13.01 | 0.93 | 0.84 | -26.81 | 2.73 | 91.0 | 558 | 4.9 |
| Tier3_Income_FW | 22 | 28 | 7.68 | 0.57 | 0.54 | -31.52 | 1.66 | 87.2 | 579 | 5.2 |
| Tier4_Aggressive_FW | 32 | 16 | 10.70 | 0.80 | 0.82 | -25.07 | 1.46 | 82.4 | 495 | 9.2 |
| Tier5_Turbo_FW | 45 | 13 | 7.81 | 0.65 | 0.67 | -20.07 | 1.36 | 77.6 | 751 | 13.2 |

## REALIZED-CASH metrics (account cash, no MTM noise)

| Tier | Realized CAGR % | Realized Sharpe | Realized Sortino | Realized Max DD % | Final Equity $ |
|------|----------------:|----------------:|-----------------:|------------------:|---------------:|
| Tier1_Conservative_FW | 4.80 | 1.10 | 0.25 | -9.12 | $132,378 |
| Tier2_Balanced_FW | 13.29 | 1.67 | 0.62 | -10.95 | $210,943 |
| Tier3_Income_FW | 9.24 | 0.95 | 0.28 | -16.67 | $169,667 |
| Tier4_Aggressive_FW | 11.23 | 0.82 | 0.38 | -17.61 | $189,010 |
| Tier5_Turbo_FW | 10.51 | 0.78 | 0.49 | -20.78 | $181,828 |

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
