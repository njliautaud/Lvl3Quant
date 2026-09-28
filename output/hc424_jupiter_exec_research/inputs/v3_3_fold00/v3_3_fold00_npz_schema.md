# v3.3 fold_00 NPZ Schema (HC #424 §3 Fallback Path)

## Source
- **Origin**: `/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_predictions.npz` (already on Jupiter — NO SCP needed)
- **Copied to**: `/home/jupiter/Lvl3Quant/output/hc424_jupiter_exec_research/inputs/v3_3_fold00/fold_00_predictions.npz`
- **SHA256 (predictions)**: `b455f30373555ae71d991a8962d9f307dadba4c60ecd22c23eb723d9e38e06b0`
- **SHA256 (feature_stats)**: `4c0f1d88b95b028c7d9169fb0c27eb1b2b93f7429df8cb4e87f610fe6fe25e15`
- **fold_idx**: 0
- **n_samples**: 241,351
- **oot_dates**: ['20260223','20260224','20260225','20260226','20260227']

## Reported IC (Jupiter-side, scipy.stats.spearmanr, mask-applied)
- IC_1s  = 0.2625
- IC_5s  = 0.1303
- IC_10s = 0.0865

**vs CLAUDE.md champion claim** (IC_1s≈0.222, IC_5s≈0.141, IC_10s≈0.106):
- IC_1s BETTER than reported (+0.04)
- IC_5s slightly under (-0.011)
- IC_10s under (-0.020)
- Verdict still valid — proceeding with LGBM execution gate analysis.

## Schema (per-key)
Identical to v3.4.2 ep3 NPZ — same 30 prediction heads + targets + masks.
Heads: log_ret {1s,5s,10s,30s,60s,5min}, p_up {5s,10s,30s,60s},
quantiles {10s,30s,60s}x{q10,q50,q90}, mfe/mae {30s,60s}, time_to_mfe,
p_reversal {15s,30s,60s}, realized_vol_30s, fifo_tp4sl3 {net, hit_tp},
fifo_tp8sl5 {net, hit_tp}.

## Differences vs v3.4.2 ep3
- Same feature schema → identical 30-head LGBM possible (no adapter needed).
- Different training method (uncertainty-weighted MTL).
- IC trade-off: v3.3 has slightly higher 5s/10s than v3.4.2 ep3 (per HC #422
  mixed-IC finding); v3.4.2 ep3 has higher 1s.
