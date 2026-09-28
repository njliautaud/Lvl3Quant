# HC #494 R3 — H30s FIFO Sweep Report

**Date**: 2026-05-28T19:41:52.445011

**OOT Range**: 20260224 → 20260427 (46 days)

## Results by Cell

| Horizon | Top % | Net Ticks | Count | Days | Pos Days | Trades/Day | Sharpe | Sharpe_Green | Sharpe_Red | Regime Skew | Verdict |
|---------|-------|-----------|-------|------|----------|------------|--------|--------------|------------|-------------|----------|
| 10s | 10% | -34717.98 | 94746 | 46 | 4 | 2059.7 | -152.16 | -170.31 | nan | nan | **FAIL** |
| 10s | 20% | -69477.10 | 189991 | 46 | 3 | 4130.2 | -150.46 | -175.16 | nan | nan | **FAIL** |
| 10s | 5% | -15934.36 | 47239 | 46 | 4 | 1026.9 | -139.76 | -186.02 | nan | nan | **FAIL** |
| 1s | 10% | -34679.61 | 95737 | 46 | 2 | 2081.2 | -413.66 | -448.91 | nan | nan | **FAIL** |
| 1s | 20% | -69429.95 | 191089 | 46 | 1 | 4154.1 | -414.09 | -459.95 | nan | nan | **FAIL** |
| 1s | 5% | -16492.52 | 48153 | 46 | 4 | 1046.8 | -387.39 | -417.43 | nan | nan | **FAIL** |
| 30s | 10% | -41789.03 | 94173 | 46 | 11 | 2047.2 | -110.03 | -131.37 | nan | nan | **FAIL** |
| 30s | 20% | -80522.62 | 189769 | 46 | 9 | 4125.4 | -105.90 | -134.49 | nan | nan | **FAIL** |
| 30s | 5% | -21369.68 | 47192 | 46 | 12 | 1025.9 | -108.39 | -130.42 | nan | nan | **FAIL** |
| 5s | 10% | -33276.10 | 95233 | 46 | 5 | 2070.3 | -192.44 | -220.47 | nan | nan | **FAIL** |
| 5s | 20% | -68704.05 | 190443 | 46 | 4 | 4140.1 | -199.39 | -238.44 | nan | nan | **FAIL** |
| 5s | 5% | -14711.48 | 47734 | 46 | 4 | 1037.7 | -166.01 | -197.91 | nan | nan | **FAIL** |

## Viability Bar (HC #494 R1)

A cell PASSES if ALL conditions are met:
- Net ticks > 0.0
- Regime skew ≤ 0.5
- Positive days ≥ 30
- Trades/day ≥ 5

MARGINAL = net_ticks > 0 but fails one or more conditions.
FAIL = net_ticks ≤ 0 or regime_skew too high.
