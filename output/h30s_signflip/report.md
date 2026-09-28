# HC #494 R3 — H30s SIGN-FLIPPED FIFO Sweep Report

**Date**: 2026-05-28T19:54:33.491655

**OOT Range**: 20260224 → 20260427 (46 days)

## Results by Cell

| Horizon | Top % | Net Ticks | Count | Days | Pos Days | Trades/Day | Sharpe | Sharpe_Green | Sharpe_Red | Regime Skew | Verdict |
|---------|-------|-----------|-------|------|----------|------------|--------|--------------|------------|-------------|----------|
| 10s | 10% | -30844.14 | 99031 | 46 | 6 | 2152.8 | -127.15 | -157.81 | nan | nan | **FAIL** |
| 10s | 20% | -64283.27 | 196163 | 46 | 3 | 4264.4 | -132.59 | -169.31 | nan | nan | **FAIL** |
| 10s | 5% | -16756.60 | 49979 | 46 | 4 | 1086.5 | -137.16 | -176.16 | nan | nan | **FAIL** |
| 1s | 10% | -35178.39 | 98646 | 46 | 2 | 2144.5 | -406.98 | -442.75 | nan | nan | **FAIL** |
| 1s | 20% | -70591.58 | 195584 | 46 | 1 | 4251.8 | -409.94 | -441.57 | nan | nan | **FAIL** |
| 1s | 5% | -17884.96 | 49839 | 46 | 3 | 1083.5 | -417.05 | -444.84 | nan | nan | **FAIL** |
| 30s | 10% | -30792.01 | 99653 | 46 | 11 | 2166.4 | -78.68 | -135.20 | nan | nan | **FAIL** |
| 30s | 20% | -64406.46 | 197613 | 46 | 10 | 4295.9 | -80.71 | -134.07 | nan | nan | **FAIL** |
| 30s | 5% | -17914.96 | 50597 | 46 | 9 | 1099.9 | -90.44 | -158.74 | nan | nan | **FAIL** |
| 5s | 10% | -30125.07 | 98955 | 46 | 5 | 2151.2 | -163.82 | -196.28 | nan | nan | **FAIL** |
| 5s | 20% | -66391.09 | 196104 | 46 | 4 | 4263.1 | -182.98 | -229.41 | nan | nan | **FAIL** |
| 5s | 5% | -16222.73 | 49857 | 46 | 6 | 1083.8 | -177.16 | -190.90 | nan | nan | **FAIL** |

## Viability Bar (HC #494 R1)

A cell PASSES if ALL conditions are met:
- Net ticks > 0.0
- Regime skew ≤ 0.5
- Positive days ≥ 30
- Trades/day ≥ 5

MARGINAL = net_ticks > 0 but fails one or more conditions.
FAIL = net_ticks ≤ 0 or regime_skew too high.
