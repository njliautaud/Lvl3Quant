# Boosting (b) verdict — meta-LGBM trade-success gate on v3.4.2-basis ensemble

HC #427 R5 boosting technique #2. Inputs: 11 LOO-robust configs from boosting (a). Per-day LOO meta-training (hold 1 day out, train on 4, gate that day).

## Summary
- Configs tested: 11
- Configs where meta-gate found a robust threshold that BEATS baseline worst-day Sharpe: **0**
- Thresholds swept: [0.25, 0.3, 0.35, 0.4, 0.5, 0.6]

## Boosted configs (worth promoting)

_No config saw worst-day Sharpe improvement under any threshold._

**Interpretation**: The 32-head feature vector at signal time does not consistently predict per-fill profitability beyond what the existing conf_thr+horizon_confluence+fifo_confluence already captures. Boosting (b) does not advance HC #427 R5 with this design; try alternative meta architectures (regime conditioning, weighted ensemble).