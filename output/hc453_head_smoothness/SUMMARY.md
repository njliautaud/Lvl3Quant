# HC #453 R6a — Head Smoothness Diagnostic

Source: /home/jupiter/Lvl3Quant/output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate (34 OOT NPZs)

## Top 5 smoothest heads by mean lag-1 autocorr

| head | n_days | mean ac_lag1 | flip/min | top10pct persist 10s |
|---|---|---|---|---|
| pred_fifo_tp4sl3_net | 32 | 0.6148 | 85.55 | 0.453 |
| pred_fifo_tp8sl5_net | 32 | 0.4918 | 84.30 | 0.424 |
| pred_p_up_60s | 32 | 0.4150 | 91.36 | 0.422 |
| pred_p_reversal_15s | 32 | 0.3860 | 97.40 | 0.269 |
| pred_p_reversal_60s | 32 | 0.3853 | 84.78 | 0.264 |

## Bottom 5 (most flickery)

| head | n_days | mean ac_lag1 | flip/min | top10pct persist 10s |
|---|---|---|---|---|
| pred_log_ret_5min | 32 | 0.0271 | 111.48 | 0.231 |
| pred_log_ret_60s_q50 | 32 | 0.0237 | 116.09 | 0.371 |
| pred_pred_mfe_30s_ticks | 32 | 0.0221 | 113.78 | 0.235 |
| pred_pred_mae_30s_ticks | 32 | 0.0220 | 120.51 | 0.300 |
| pred_log_ret_1s | 32 | 0.0194 | 116.62 | 0.183 |

## HC #453 R4 gate: lag-1 autocorr >= 0.30 required

**Heads passing R4 gate (candidates for new trading signal)**:
- pred_fifo_tp4sl3_net (ac_lag1=0.6148)
- pred_fifo_tp8sl5_net (ac_lag1=0.4918)
- pred_p_up_60s (ac_lag1=0.4150)
- pred_p_reversal_15s (ac_lag1=0.3860)
- pred_p_reversal_60s (ac_lag1=0.3853)
- pred_p_reversal_30s (ac_lag1=0.3556)
- pred_pred_realized_vol_30s_ticks (ac_lag1=0.3254)
