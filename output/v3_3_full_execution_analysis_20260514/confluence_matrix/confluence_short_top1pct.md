# v3.3 CONFLUENCE MATRIX — SHORT Top 1.0% per head

Source: `fold_00_predictions.npz` (5 OOT days, 241,351 events, 32 heads)
Min n_both: 20.  Commission ticks: 0.376.  Eligible heads: 32

## Top 30 cross-head pairs by joint passive Sharpe

| Pair | n_both | Joint Sharpe | Joint net t | Solo i | Solo j | Lift vs best solo |
|---|---|---|---|---|---|---|
| log_ret_5min ∧ pred_mfe_60s_ticks | 22 | 1.797 | 2.682 | 0.099 | 0.019 | +1.698 |
| log_ret_5min ∧ log_ret_60s_q90 | 40 | 0.564 | 1.600 | 0.099 | -0.113 | +0.465 |
| log_ret_10s ∧ p_reversal_30s | 22 | 0.414 | 1.227 | 0.023 | -0.235 | +0.392 |
| log_ret_5min ∧ pred_time_to_mfe_secs | 136 | 0.370 | 1.147 | 0.099 | 0.124 | +0.246 |
| log_ret_5min ∧ pred_mae_30s_ticks | 41 | 0.357 | 1.122 | 0.099 | -0.208 | +0.258 |
| log_ret_30s ∧ p_reversal_30s | 46 | 0.352 | 1.087 | 0.063 | -0.235 | +0.290 |
| fifo_tp8sl5_net ∧ log_ret_5s | 55 | 0.331 | 0.945 | 0.073 | 0.021 | +0.258 |
| fifo_tp8sl5_hit_tp ∧ log_ret_30s_q90 | 20 | 0.314 | 0.750 | -0.043 | 0.029 | +0.285 |
| fifo_tp8sl5_net ∧ log_ret_1s | 73 | 0.305 | 0.890 | 0.073 | 0.015 | +0.232 |
| fifo_tp4sl3_net ∧ pred_time_to_mfe_secs | 23 | 0.279 | 0.804 | 0.109 | 0.124 | +0.155 |
| fifo_tp4sl3_net ∧ log_ret_5s | 58 | 0.276 | 0.802 | 0.109 | 0.021 | +0.167 |
| fifo_tp4sl3_net ∧ log_ret_30s | 37 | 0.276 | 0.797 | 0.109 | 0.063 | +0.167 |
| log_ret_10s_q10 ∧ log_ret_30s | 102 | 0.262 | 0.814 | -0.019 | 0.063 | +0.199 |
| fifo_tp8sl5_net ∧ log_ret_10s | 41 | 0.258 | 0.756 | 0.073 | 0.023 | +0.185 |
| fifo_tp8sl5_net ∧ log_ret_30s | 41 | 0.258 | 0.756 | 0.073 | 0.063 | +0.185 |
| log_ret_30s ∧ log_ret_30s_q10 | 98 | 0.256 | 0.796 | 0.063 | -0.004 | +0.194 |
| fifo_tp8sl5_net ∧ p_reversal_15s | 21 | 0.255 | 0.810 | 0.073 | -0.244 | +0.182 |
| log_ret_30s_q50 ∧ p_reversal_30s | 75 | 0.248 | 0.800 | 0.047 | -0.235 | +0.200 |
| log_ret_5min ∧ p_reversal_30s | 35 | 0.243 | 0.800 | 0.099 | -0.235 | +0.144 |
| log_ret_10s ∧ log_ret_30s_q10 | 58 | 0.239 | 0.724 | 0.023 | -0.004 | +0.216 |
| log_ret_5s ∧ p_reversal_30s | 30 | 0.238 | 0.767 | 0.021 | -0.235 | +0.217 |
| log_ret_10s ∧ log_ret_10s_q10 | 60 | 0.223 | 0.683 | 0.023 | -0.019 | +0.201 |
| fifo_tp4sl3_net ∧ log_ret_1s | 89 | 0.223 | 0.663 | 0.109 | 0.015 | +0.114 |
| log_ret_10s_q10 ∧ log_ret_5s | 94 | 0.214 | 0.660 | -0.019 | 0.021 | +0.193 |
| fifo_tp4sl3_net ∧ log_ret_10s | 37 | 0.205 | 0.608 | 0.109 | 0.023 | +0.096 |
| log_ret_30s_q50 ∧ pred_time_to_mfe_secs | 142 | 0.200 | 0.641 | 0.047 | 0.124 | +0.075 |
| log_ret_30s ∧ pred_mae_30s_ticks | 33 | 0.199 | 0.667 | 0.063 | -0.208 | +0.136 |
| log_ret_30s ∧ pred_time_to_mfe_secs | 81 | 0.183 | 0.586 | 0.063 | 0.124 | +0.058 |
| fifo_tp4sl3_net ∧ fifo_tp8sl5_net | 652 | 0.179 | 0.564 | 0.109 | 0.073 | +0.070 |
| log_ret_30s_q10 ∧ log_ret_5s | 87 | 0.178 | 0.552 | -0.004 | 0.021 | +0.157 |

## Solo SHORT Sharpe (Top 1.0% reference)

| Head | n | Solo Sharpe | Solo net t |
|---|---|---|---|
| pred_time_to_mfe_secs | 624 | 0.124 | 0.417 |
| fifo_tp4sl3_net | 901 | 0.109 | 0.352 |
| log_ret_5min | 595 | 0.099 | 0.339 |
| log_ret_60s_q10 | 768 | 0.091 | 0.273 |
| fifo_tp8sl5_net | 1015 | 0.073 | 0.235 |
| log_ret_30s | 613 | 0.063 | 0.189 |
| p_up_5s | 594 | 0.058 | 0.172 |
| p_up_10s | 625 | 0.053 | 0.154 |
| p_reversal_60s | 596 | 0.052 | 0.179 |
| log_ret_30s_q50 | 608 | 0.047 | 0.147 |
| p_up_30s | 611 | 0.044 | 0.128 |
| pred_realized_vol_30s_ticks | 931 | 0.042 | 0.133 |
| log_ret_10s_q90 | 608 | 0.038 | 0.111 |
| log_ret_30s_q90 | 675 | 0.029 | 0.086 |
| pred_mfe_30s_ticks | 1233 | 0.025 | 0.079 |
| log_ret_10s | 602 | 0.023 | 0.070 |
| log_ret_5s | 596 | 0.021 | 0.066 |
| pred_mae_60s_ticks | 608 | 0.020 | 0.060 |
| pred_mfe_60s_ticks | 591 | 0.019 | 0.058 |
| log_ret_1s | 611 | 0.015 | 0.050 |
