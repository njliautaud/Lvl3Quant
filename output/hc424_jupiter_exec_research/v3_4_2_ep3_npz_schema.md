# v3.4.2 ep3 OOT NPZ — Schema Inventory (HC #424 §3 Step 1)

**Source**: `nick@neptune:/home/nick/Lvl3Quant/output/cnn_mamba_v3_4_2_fixedmtl/fold_00_ep3_oot.npz`
**Local copy**: `/home/jupiter/Lvl3Quant/output/hc424_jupiter_exec_research/inputs/v3_4_2_ep3/fold_00_ep3_oot.npz`
**SHA256**: `bf513ec9eef5ffad47e841d30893222934f3537396fc75c59062e351e43b9838`
**Size**: 13,568,019 bytes (12.94 MB)
**N samples**: 241,351
**Fold**: 0
**OOT dates**: 20260223–20260227 (5 trading days, inferred from FIFO label alignment per HC #417 wrap)
**FIFO labels total**: 241,692 → diff 341 absorbed by proportional last-day trim

## Reported metrics (in NPZ scalars)
| Horizon | IC |
|---|---|
| 1s  | **0.2744** |
| 5s  | 0.1277 |
| 10s | 0.0897 |
| 30s | 0.0559 |
| MFE_30s corr | 0.2923 |
| MAE_30s corr | 0.2759 |

Confirms HC #422: IC_1s win (0.274 vs v3.3 0.222 baseline), IC_5s/10s mixed lower but multi-head signal usable.

## Heads available (multi-output schema; HC #423 §4 calls for ALL HEADS as features)

All arrays are `(241351,)` float32.

### Predicted heads (used as features for LGBM gate)
- `pred_log_ret_1s` (mean=0.0092, std=0.398)
- `pred_log_ret_5s`
- `pred_log_ret_10s`
- `pred_log_ret_30s`
- `pred_log_ret_60s`
- `pred_log_ret_5min`
- `pred_p_up_5s`, `pred_p_up_10s`, `pred_p_up_30s`, `pred_p_up_60s`
- `pred_log_ret_10s_q10/q50/q90`
- `pred_log_ret_30s_q10/q50/q90`
- `pred_log_ret_60s_q10/q50/q90`
- `pred_pred_mfe_30s_ticks` (mean=4.78, std=0.24)
- `pred_pred_mae_30s_ticks` (mean=-4.59, std=0.18)
- `pred_pred_mfe_60s_ticks`
- `pred_pred_mae_60s_ticks`
- `pred_pred_time_to_mfe_secs`
- `pred_p_reversal_15s`, `pred_p_reversal_30s`, `pred_p_reversal_60s`
- `pred_pred_realized_vol_30s_ticks` (mean=6.93, std=0.19)  ← VOL HEAD
- `pred_fifo_tp4sl3_net`, `pred_fifo_tp8sl5_net`  ← model's own FIFO-net predictions
- `pred_fifo_tp4sl3_hit_tp`, `pred_fifo_tp8sl5_hit_tp`

**Total predictor features**: 30 (full multi-head vector — HC #422 R8 satisfied)

### Targets present (for sanity / for LGBM target if desired)
- `target_log_ret_{1s,5s,10s,30s,60s,5min}` + p_up/quantile variants
- `target_pred_mfe_30s_ticks`, `target_pred_mae_30s_ticks`, `target_pred_mfe_60s_ticks`, `target_pred_mae_60s_ticks`, `target_pred_time_to_mfe_secs`
- `target_p_reversal_*`, `target_pred_realized_vol_30s_ticks`
- `target_fifo_tp4sl3_net`, `target_fifo_tp8sl5_net`, `target_fifo_tp4sl3_hit_tp`, `target_fifo_tp8sl5_hit_tp`

### Masks (validity per head)
- `mask_*` companion for every head

### Missing from NPZ (need external alignment)
- No `n_samples`, no `oot_dates`, no `ts_ns`, no `day_index` → recover via per-day FIFO label counts (HC #417 wrap method).
- True per-row timestamps come from FIFO labels' `ts_ns` once aligned by date.

## Decision: Sufficient for LGBM execution gate

YES — predictions cover full multi-head feature space and the per-row FIFO targets
(`target_fifo_tp4sl3_net`, etc.) are present so we can train an LGBM gate directly
on this NPZ. For canonical replay verdict (HC #397B) we align to per-day FIFO labels
to recover `ts_ns` and `day_idx` → day_conc (HC #344) gate.

## Feature stats (for live encoder consistency)
- `fold_00_feature_stats.npz`: `mean_t1` (39,), `std_t1` (39,) — book-feature
  normalizer stats. Not needed for the LGBM gate (gate operates on POST-encoder
  predictions, not raw book features).
