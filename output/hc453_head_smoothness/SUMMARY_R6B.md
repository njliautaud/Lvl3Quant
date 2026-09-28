# HC #453 R6b — Head JOINT (smoothness AND informativeness AND sign-balance)

## Why this exists

R6a ranked by lag-1 autocorr only. Top 7 heads passed the 0.30 floor — but
spot-checking showed pred_fifo_tp4sl3_net is 100% negative on every OOT date,
with p10-p90 spread of 0.06-0.51 in a full range of 0.94-2.03. The 'smoothness'
was a DEAD-SIGNAL artifact: model collapsed heads to near-constants because
FixedWeightMultiHeadLoss assigned them default weight 0.1. High autocorr
without dynamic range != stream-coherent signal.

Per HC #453 R4-amended (principle-first, number-second), the right gate is
JOINT: smooth AND informative AND sign-balanced. Score = ac_lag1 × spread_ratio × balanced_frac.

## Re-ranked top 10 by joint actionability score

| head | days | mean_ac1 | med_spread_ratio | balanced% | pos% | ACTIONABLE |
|---|---|---|---|---|---|---|
| pred_log_ret_30s_q50 | 32 | 0.086 | 0.354 | 1.00 | 0.526 | **0.0306** |
| pred_p_up_30s | 32 | 0.081 | 0.349 | 1.00 | 0.680 | **0.0282** |
| pred_pred_mfe_60s_ticks | 32 | 0.061 | 0.470 | 0.84 | 0.272 | **0.0242** |
| pred_pred_mae_60s_ticks | 32 | 0.054 | 0.436 | 1.00 | 0.609 | **0.0235** |
| pred_log_ret_10s_q50 | 32 | 0.119 | 0.220 | 0.75 | 0.742 | **0.0196** |
| pred_log_ret_10s | 32 | 0.034 | 0.578 | 1.00 | 0.601 | **0.0194** |
| pred_log_ret_5s | 32 | 0.029 | 0.640 | 1.00 | 0.609 | **0.0186** |
| pred_log_ret_30s | 32 | 0.121 | 0.154 | 0.94 | 0.311 | **0.0175** |
| pred_log_ret_60s_q50 | 32 | 0.024 | 0.719 | 1.00 | 0.608 | **0.0170** |
| pred_p_up_10s | 32 | 0.046 | 0.454 | 0.78 | 0.274 | **0.0163** |

## R6a 'smooth' heads now reassessed

| head | mean_ac1 | spread_ratio | balanced% | verdict |
|---|---|---|---|---|
| pred_fifo_tp4sl3_net | 0.615 | 0.168 | 0.00 | DEAD (constant) |
| pred_fifo_tp8sl5_net | 0.492 | 0.119 | 0.00 | DEAD (constant) |
| pred_p_up_60s | 0.415 | 0.100 | 0.00 | DEAD (constant) |
| pred_p_reversal_15s | 0.386 | 0.144 | 0.00 | DEAD (constant) |
| pred_p_reversal_60s | 0.385 | 0.118 | 0.31 | narrow |
| pred_p_reversal_30s | 0.356 | 0.200 | 0.00 | DEAD (constant) |
| pred_pred_realized_vol_30s_ticks | 0.325 | 0.162 | 0.00 | DEAD (constant) |

## Conclusion

If the top actionable_score across all 23 heads is still < ~0.05, then NO existing
v3.4.2 head is both stream-coherent AND informative. The HC #454 Phase 2 trainer
patch (new pressure/persistence/MFE-MAE heads with weight=1.0 and a smoothness
regularizer) is then the ONLY path forward — confirms user's reframing instinct.
