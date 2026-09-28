# Wheel Strategy Tier Ladder — Comparative Report
Pricing: **real_blend_vendor_iv_modeled_fallback** (MODELED unless real chain data wired in)
Window: **2020-01-01 -> 2025-12-31**  Capital per tier: **$100,000**

## MTM (mark-to-market) metrics

| Tier | Target % | Universe | CAGR % | Sharpe | Sortino | Max DD % | PF | WR % | Trades | Assign % |
|------|---------:|---------:|-------:|-------:|--------:|---------:|---:|-----:|-------:|---------:|
| Tier1_Conservative_FW | 10 | 14 | 3.42 | 0.36 | 0.34 | -24.65 | 1.65 | 93.6 | 392 | 3.7 |
| Tier2_Balanced_FW | 15 | 48 | 12.98 | 0.90 | 0.84 | -26.80 | 2.47 | 90.3 | 629 | 5.7 |
| Tier3_Income_FW | 22 | 28 | 12.76 | 0.86 | 0.84 | -24.94 | 1.88 | 87.6 | 573 | 5.1 |
| Tier4_Aggressive_FW | 32 | 16 | 9.08 | 0.70 | 0.69 | -22.95 | 1.39 | 81.7 | 496 | 10.8 |
| Tier5_Turbo_FW | 45 | 13 | 4.81 | 0.44 | 0.42 | -23.27 | 1.29 | 79.1 | 723 | 15.2 |

## REALIZED-CASH metrics (account cash, no MTM noise)

| Tier | Realized CAGR % | Realized Sharpe | Realized Sortino | Realized Max DD % | Final Equity $ |
|------|----------------:|----------------:|-----------------:|------------------:|---------------:|
| Tier1_Conservative_FW | 2.67 | 0.73 | 0.15 | -15.30 | $117,088 |
| Tier2_Balanced_FW | 13.83 | 1.49 | 0.63 | -8.35 | $216,983 |
| Tier3_Income_FW | 14.39 | 0.77 | 0.31 | -37.00 | $223,450 |
| Tier4_Aggressive_FW | 8.81 | 0.67 | 0.29 | -30.12 | $165,660 |
| Tier5_Turbo_FW | 7.68 | 0.80 | 0.41 | -19.19 | $155,699 |

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
