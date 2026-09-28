# v3.3 CONFLUENCE MATRIX — LONG Top 1.0% per head

Source: `fold_00_predictions.npz` (5 OOT days, 241,351 events, 32 heads)
Min n_both: 20.  Commission ticks: 0.376.  Eligible heads: 32

## Top 30 cross-head pairs by joint passive Sharpe

| Pair | n_both | Joint Sharpe | Joint net t | Solo i | Solo j | Lift vs best solo |
|---|---|---|---|---|---|---|
| log_ret_60s_q10 ∧ pred_realized_vol_30s_ticks | 25 | 0.128 | 0.448 | -0.244 | -0.011 | +0.139 |
| fifo_tp4sl3_hit_tp ∧ pred_mfe_30s_ticks | 428 | 0.098 | 0.337 | 0.020 | -0.000 | +0.078 |
| fifo_tp4sl3_hit_tp ∧ pred_realized_vol_30s_ticks | 403 | 0.075 | 0.260 | 0.020 | -0.011 | +0.055 |
| log_ret_60s_q10 ∧ pred_mfe_30s_ticks | 23 | 0.058 | 0.205 | -0.244 | -0.000 | +0.058 |
| log_ret_10s_q90 ∧ p_up_30s | 28 | 0.033 | 0.105 | -0.183 | -0.285 | +0.216 |
| log_ret_30s_q90 ∧ p_up_30s | 28 | 0.033 | 0.105 | -0.122 | -0.285 | +0.154 |
| fifo_tp4sl3_hit_tp ∧ log_ret_10s | 20 | 0.027 | 0.098 | 0.020 | -0.275 | +0.008 |
| p_reversal_60s ∧ pred_realized_vol_30s_ticks | 24 | 0.011 | 0.040 | -0.294 | -0.011 | +0.022 |
| fifo_tp8sl5_net ∧ log_ret_10s_q10 | 49 | 0.010 | 0.034 | -0.310 | -0.304 | +0.314 |
| fifo_tp4sl3_hit_tp ∧ fifo_tp8sl5_hit_tp | 369 | 0.008 | 0.027 | 0.020 | -0.140 | -0.012 |
| fifo_tp4sl3_hit_tp ∧ log_ret_30s_q90 | 382 | 0.001 | 0.005 | 0.020 | -0.122 | -0.018 |
| fifo_tp8sl5_net ∧ log_ret_30s_q10 | 50 | -0.013 | -0.042 | -0.310 | -0.343 | +0.297 |
| pred_mfe_30s_ticks ∧ pred_realized_vol_30s_ticks | 624 | -0.018 | -0.061 | -0.000 | -0.011 | -0.017 |
| log_ret_60s_q50 ∧ pred_mae_60s_ticks | 25 | -0.031 | -0.112 | -0.210 | -0.132 | +0.101 |
| log_ret_30s_q90 ∧ pred_mfe_30s_ticks | 406 | -0.042 | -0.149 | -0.122 | -0.000 | -0.042 |
| log_ret_30s_q90 ∧ pred_realized_vol_30s_ticks | 385 | -0.054 | -0.188 | -0.122 | -0.011 | -0.043 |
| p_reversal_60s ∧ pred_mfe_30s_ticks | 22 | -0.070 | -0.252 | -0.294 | -0.000 | -0.070 |
| fifo_tp4sl3_hit_tp ∧ log_ret_10s_q90 | 276 | -0.079 | -0.277 | 0.020 | -0.183 | -0.099 |
| log_ret_10s_q90 ∧ pred_mfe_30s_ticks | 256 | -0.087 | -0.307 | -0.183 | -0.000 | -0.087 |
| fifo_tp8sl5_hit_tp ∧ pred_realized_vol_30s_ticks | 559 | -0.092 | -0.321 | -0.140 | -0.011 | -0.081 |
| fifo_tp8sl5_net ∧ pred_time_to_mfe_secs | 62 | -0.092 | -0.300 | -0.310 | -0.308 | +0.216 |
| fifo_tp8sl5_hit_tp ∧ log_ret_30s_q90 | 360 | -0.094 | -0.330 | -0.140 | -0.122 | +0.027 |
| log_ret_10s_q90 ∧ pred_realized_vol_30s_ticks | 241 | -0.101 | -0.354 | -0.183 | -0.011 | -0.090 |
| fifo_tp8sl5_hit_tp ∧ pred_mfe_30s_ticks | 552 | -0.101 | -0.353 | -0.140 | -0.000 | -0.101 |
| p_reversal_30s ∧ pred_time_to_mfe_secs | 24 | -0.117 | -0.398 | -0.317 | -0.308 | +0.190 |
| fifo_tp8sl5_net ∧ p_up_60s | 37 | -0.134 | -0.441 | -0.310 | -0.315 | +0.176 |
| log_ret_10s_q10 ∧ p_up_60s | 47 | -0.137 | -0.454 | -0.304 | -0.315 | +0.167 |
| log_ret_30s_q10 ∧ p_up_60s | 47 | -0.137 | -0.454 | -0.343 | -0.315 | +0.178 |
| fifo_tp8sl5_hit_tp ∧ log_ret_10s_q90 | 225 | -0.148 | -0.516 | -0.140 | -0.183 | -0.008 |
| p_up_60s ∧ pred_time_to_mfe_secs | 118 | -0.171 | -0.519 | -0.315 | -0.308 | +0.137 |

## Solo LONG Sharpe (Top 1.0% reference)

| Head | n | Solo Sharpe | Solo net t |
|---|---|---|---|
| fifo_tp4sl3_hit_tp | 594 | 0.020 | 0.069 |
| pred_mfe_30s_ticks | 668 | -0.000 | -0.001 |
| pred_realized_vol_30s_ticks | 635 | -0.011 | -0.037 |
| log_ret_30s_q90 | 675 | -0.122 | -0.410 |
| pred_mae_60s_ticks | 598 | -0.132 | -0.463 |
| fifo_tp8sl5_hit_tp | 652 | -0.140 | -0.487 |
| log_ret_10s_q90 | 627 | -0.183 | -0.593 |
| log_ret_60s_q50 | 604 | -0.210 | -0.730 |
| log_ret_60s_q10 | 594 | -0.244 | -0.772 |
| log_ret_60s_q90 | 645 | -0.256 | -0.816 |
| p_up_10s | 607 | -0.266 | -0.889 |
| log_ret_30s_q50 | 598 | -0.273 | -0.926 |
| log_ret_10s | 598 | -0.275 | -0.897 |
| p_reversal_15s | 909 | -0.280 | -0.875 |
| p_up_30s | 599 | -0.285 | -0.974 |
| log_ret_1s | 597 | -0.285 | -0.889 |
| fifo_tp4sl3_net | 825 | -0.286 | -0.890 |
| p_reversal_60s | 610 | -0.294 | -0.931 |
| log_ret_5s | 595 | -0.296 | -0.941 |
| log_ret_10s_q50 | 592 | -0.299 | -0.963 |
