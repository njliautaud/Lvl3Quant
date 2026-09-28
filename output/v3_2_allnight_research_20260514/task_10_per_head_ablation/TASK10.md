# Task 10 — Per-Head Ablation (drop-one, smaller MLP, 3-fold CV)

Baseline AUC (all heads): 0.5112

## Top 15 most-important heads (largest AUC drop when removed)
| Head | AUC w/o | Δ |
|---|---:|---:|
| pred_pred_realized_vol_30s_ticks | 0.4944 | -0.0168 |
| pred_log_ret_1s | 0.4967 | -0.0145 |
| pred_pred_mfe_60s_ticks | 0.4973 | -0.0140 |
| pred_log_ret_30s | 0.4980 | -0.0133 |
| pred_pred_time_to_mfe_secs | 0.4991 | -0.0121 |
| pred_pred_mae_60s_ticks | 0.4995 | -0.0117 |
| pred_log_ret_30s_q50 | 0.5003 | -0.0109 |
| pred_log_ret_30s_q90 | 0.5004 | -0.0108 |
| pred_log_ret_5min | 0.5008 | -0.0104 |
| pred_log_ret_10s_q90 | 0.5017 | -0.0095 |
| pred_log_ret_5s | 0.5022 | -0.0091 |
| pred_log_ret_30s_q10 | 0.5025 | -0.0087 |
| pred_p_reversal_60s | 0.5027 | -0.0086 |
| pred_fifo_tp8sl5_hit_tp | 0.5041 | -0.0072 |
| pred_fifo_tp4sl3_hit_tp | 0.5041 | -0.0072 |

## 10 LEAST-important heads (positive Δ = removable noise)
| Head | AUC w/o | Δ |
|---|---:|---:|
| pred_p_up_5s | 0.5102 | -0.0010 |
| pred_log_ret_60s_q50 | 0.5098 | -0.0015 |
| pred_log_ret_60s_q90 | 0.5095 | -0.0018 |
| pred_log_ret_10s_q50 | 0.5088 | -0.0024 |
| pred_fifo_tp4sl3_net | 0.5084 | -0.0028 |
| pred_p_up_60s | 0.5083 | -0.0029 |
| pred_p_up_30s | 0.5082 | -0.0031 |
| pred_p_reversal_15s | 0.5072 | -0.0041 |
| pred_log_ret_60s_q10 | 0.5068 | -0.0044 |
| pred_log_ret_60s | 0.5066 | -0.0046 |
