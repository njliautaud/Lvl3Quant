# Strategy Leaderboard (as of 2026-07-18)

## Top Strategies by Sharpe Ratio

| # | Strategy | Sharpe | Sortino | CAGR | MaxDD | Calmar | Perm | SubP | R1 | Gates |
|---|----------|--------|---------|------|-------|--------|------|------|----|-------|
| 1 | VIX Ultra-Aggressive (12/16/22) | 3.91 | - | 38% | -5.7% | - | ✅ | ✅ | ❌ | 2/3 |
| 2 | VIX + Tail Hedge | 3.42 | 5.84 | 34.5% | -10.7% | 3.21 | ✅ | ✅ | ❌ | 2/3 |
| 3 | Anti-Fragile Portfolio | 3.31 | - | 35% | -7.0% | - | ✅ | ✅ | ❌ | 2/3 |
| 4 | VIX + Trend Confirm | 3.32 | 5.62 | 33.3% | -6.4% | 5.21 | ✅ | ✅ | ❌ | 2/3 |
| 5 | VIX Leverage Base | 3.22 | 5.46 | 33.2% | -7.0% | 4.73 | ✅ | ✅ | ❌ | 2/3 |
| 6 | Dynamic VIX/Trend | 3.06 | 4.93 | 22.3% | -4.6% | 4.82 | TBD | - | - | TBD |
| 7 | ML Trend Following v2 | **2.90** | 4.15 | 19.8% | -5.3% | 3.76 | ✅ | ✅ | **✅** | **3/4** |
| 8 | VIX-Scaled Factor | 2.59 | - | 37% | -8.9% | - | ✅ | ✅ | ❌ | 2/3 |
| 9 | Dual Momentum (GEM) | 1.66 | - | 24.5% | -12.5% | - | ✅ | ✅ | ❌ | 2/3 |
| 10 | VRP Harvesting | 1.42 | 1.56 | 11% | -9.5% | 1.16 | ✅ | ✅ | ❌ | 2/3 |
| - | SPY B&H | 0.87 | 1.07 | 14.8% | -33.7% | 0.44 | - | - | - | - |

## Key Findings

1. **VIX leverage timing is the strongest single signal** — Sharpe 3.0-3.9 across variants
2. **ALL VIX strategies fail R1** — they're inherently bull-biased (leverage when VIX is low = leverage in bull markets)
3. **ML Trend Following is the ONLY R1 PASS** with strong Sharpe (2.90) — works equally in bull AND bear markets
4. **Best risk-adjusted**: VIX + Trend Confirm (Calmar 5.21) — requires both low VIX AND bullish trend
5. **Blending doesn't help**: Adding trend following to VIX leverage dilutes returns without improving risk-adjusted metrics
6. **Monthly rebalance kills VIX strategy** (Sharpe drops from 3.2 to 0.75) — daily VIX checking essential
7. **Calendar anomalies add nothing** (+0.046 Sharpe when VIX signal present)

## Current Allocation (VIX = 18.8, SPY = $743.29)

For $100K:
- **VIX Base**: $80K SPY + $20K SHY
- **VIX + Trend Confirm**: $80K SPY + $20K SHY (SPY above 200 SMA ✅)
- **VIX + Tail Hedge**: $70K SPY + $15K TLT + $15K SHY

Rebalance when VIX crosses: 15, 20, 25, or 30
