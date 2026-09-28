# Quantile DLinear — Closest-to-Profit Analysis v1

## Context
- Data source: hc489_dlinear_quantile_asym_long_v1
- Variant: asymmetric_long (P10/P50/P90 quantile preds optimized for LONG side)
- Dates: 2026-04-27 (green, N=2.37M), 2026-04-28 (red, N=2.67M)
- Total events: 5,036,935
- Cost model: passive limit fill, 0.376 ticks round-trip

## Deploy Gates (HC #428 R1 + R2)
1. net_ticks_per_event > +0.10
2. win_rate >= 0.52
3. Profitable on BOTH days (2/2, not 1/2)
4. |Sharpe_green - Sharpe_red| / max <= 0.50

## Results Summary
- Total cells analyzed: 36
- Cells passing ALL deploy gates: **5**

## FINDING: 5 CELLS PASS ALL GATES

### Winning Cells
Best cell (net_ticks):
  horizon=1s side=long bucket=top_1pct
  net_ticks=+0.2595
  wr=0.5723
  sharpe_green=0.000, sharpe_red=0.000
  regime_imbalance=0.0000

horizon side    bucket      n  net_ticks_per_event  win_rate  profitable_days  regime_imbalance
     1s long  top_1pct  35183             0.259548  0.572322                2               0.0
     1s long  top_2pct  70366             0.211173  0.571967                2               0.0
     1s long  top_5pct 175915             0.160407  0.569161                2               0.0
     1s long top_10pct 351830             0.126776  0.564275                2               0.0
     5s long  top_5pct 204790             0.121776  0.548401                2               0.0

## Regime Asymmetry Check
- Green day (4/27) cells: 36
- Red day (4/28) cells: 36
- Sign flips (profit↔loss) green-to-red: 9

## Data Caveats
- Data type: ASYMMETRIC-LONG variant (P50 predictions optimized for long alpha)
- Missing fold_00 (no 4/29 data)
- MFE/MAE are PROXY (realized signed moves) for 1s/5s/10s horizons
  (true MFE/MAE not available in quantile predictions)
- Confidence ordering: P50 (median quantile prediction)
