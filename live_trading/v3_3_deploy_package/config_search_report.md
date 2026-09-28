# v3.3 Config Search Report — HC #357/#363/#368

- Output dir: `/home/jupiter/Lvl3Quant/live_trading/v3_3_deploy_package/configs/candidate_configs`
- Gate: side=SHORT, hc357_sharpe ≥ 0.5, hc357_net > 0, n_fills ≥ 30, day_conc ≤ 0.95, ci_low_95 > -0.5

## Deploy-eligible candidates (0)

| Rank | Head | Band | n_fills | hc357_sharpe | hc357_net | day_conc | ci_low_95 |
|---|---|---|---|---|---|---|---|

## Rejected (213)

| Head | Side | Band | Reasons |
|---|---|---|---|
| fifo_tp8sl5_net | SHORT | Top0.1% | hc357_sharpe=0.054 < 0.5, day_conc=0.954 > 0.95 |
| p_reversal_60s | SHORT | Top0.1% | hc357_sharpe=0.045 < 0.5 |
| log_ret_60s_q50 | SHORT | Top1% | hc357_sharpe=0.003 < 0.5 |
| log_ret_60s_q90 | SHORT | Top5% | hc357_sharpe=-0.002 < 0.5, hc357_net=-0.590 <= 0, ci_low_95=-0.502 <= -0.5 |
| log_ret_10s_q50 | LONG | Top0.1% | side=LONG (need SHORT), hc357_sharpe=-0.006 < 0.5, hc357_net=-0.595 <= 0, ci_low_95=-0.775 <= -0.5 |
| log_ret_10s_q50 | SHORT | Top0.1% | hc357_sharpe=-0.009 < 0.5, hc357_net=-0.017 <= 0 |
| p_up_60s | SHORT | Top0.5% | hc357_sharpe=-0.010 < 0.5, hc357_net=-0.577 <= 0 |
| log_ret_60s | SHORT | Top10% | hc357_sharpe=-0.010 < 0.5, hc357_net=-0.577 <= 0, ci_low_95=-0.999 <= -0.5 |
| p_up_5s | LONG | Top20% | side=LONG (need SHORT), hc357_sharpe=-0.011 < 0.5, hc357_net=-0.577 <= 0, ci_low_95=-0.916 <= -0.5 |
| log_ret_60s_q90 | SHORT | Top10% | hc357_sharpe=-0.012 < 0.5, hc357_net=-0.602 <= 0 |
| p_reversal_30s | LONG | Top0.1% | side=LONG (need SHORT), hc357_sharpe=-0.013 < 0.5, hc357_net=-0.574 <= 0, ci_low_95=-0.679 <= -0.5 |
| log_ret_30s_q10 | SHORT | Top0.1% | hc357_sharpe=-0.013 < 0.5, hc357_net=-0.574 <= 0, ci_low_95=-0.579 <= -0.5 |
| log_ret_30s_q90 | LONG | Top1% | side=LONG (need SHORT), hc357_sharpe=-0.014 < 0.5, hc357_net=-0.604 <= 0 |
| log_ret_60s_q50 | SHORT | Top20% | hc357_sharpe=-0.020 < 0.5, hc357_net=-0.569 <= 0 |
| log_ret_60s_q90 | SHORT | Top20% | hc357_sharpe=-0.024 < 0.5, hc357_net=-0.565 <= 0 |
| p_reversal_15s | SHORT | Top1% | hc357_sharpe=-0.025 < 0.5, hc357_net=-0.563 <= 0 |
| log_ret_30s_q50 | SHORT | Top0.5% | hc357_sharpe=-0.027 < 0.5, hc357_net=-0.053 <= 0 |
| p_up_60s | SHORT | Top1% | hc357_sharpe=-0.030 < 0.5, hc357_net=-0.558 <= 0 |
| p_reversal_15s | LONG | Top0.1% | side=LONG (need SHORT), hc357_sharpe=-0.040 < 0.5, hc357_net=-0.640 <= 0, ci_low_95=-0.831 <= -0.5 |
| p_reversal_30s | SHORT | Top1% | hc357_sharpe=-0.042 < 0.5, hc357_net=-0.642 <= 0 |
| log_ret_30s_q90 | LONG | Top0.5% | side=LONG (need SHORT), hc357_sharpe=-0.056 < 0.5, hc357_net=-0.539 <= 0 |
| log_ret_10s_q90 | LONG | Top1% | side=LONG (need SHORT), hc357_sharpe=-0.064 < 0.5, hc357_net=-0.671 <= 0 |
| log_ret_60s_q50 | SHORT | Top10% | hc357_sharpe=-0.074 < 0.5, hc357_net=-0.698 <= 0, ci_low_95=-0.602 <= -0.5 |
| fifo_tp4sl3_net | SHORT | Top0.1% | hc357_sharpe=-0.081 < 0.5, hc357_net=-0.154 <= 0, day_conc=0.977 > 0.95 |
| log_ret_60s | SHORT | Top20% | hc357_sharpe=-0.082 < 0.5, hc357_net=-0.713 <= 0, ci_low_95=-0.970 <= -0.5 |
| log_ret_30s_q50 | SHORT | Top1% | hc357_sharpe=-0.083 < 0.5, hc357_net=-0.148 <= 0 |
| p_reversal_15s | SHORT | Top0.5% | hc357_sharpe=-0.084 < 0.5, hc357_net=-0.718 <= 0, ci_low_95=-0.739 <= -0.5 |
| p_up_5s | SHORT | Top0.1% | hc357_sharpe=-0.085 < 0.5, hc357_net=-0.707 <= 0, ci_low_95=-1.334 <= -0.5 |
| p_up_10s | SHORT | Top0.1% | hc357_sharpe=-0.085 < 0.5, hc357_net=-0.707 <= 0, ci_low_95=-1.407 <= -0.5 |
| log_ret_60s_q90 | LONG | Top1% | side=LONG (need SHORT), hc357_sharpe=-0.087 < 0.5, hc357_net=-0.707 <= 0, ci_low_95=-0.536 <= -0.5 |
| log_ret_10s_q90 | LONG | Top0.5% | side=LONG (need SHORT), hc357_sharpe=-0.088 < 0.5, hc357_net=-0.715 <= 0, ci_low_95=-0.543 <= -0.5 |
| log_ret_60s_q90 | SHORT | Top0.5% | hc357_sharpe=-0.093 < 0.5, hc357_net=-0.515 <= 0, day_conc=0.970 > 0.95, ci_low_95=-0.932 <= -0.5 |
| fifo_tp8sl5_net | SHORT | Top0.5% | hc357_sharpe=-0.097 < 0.5, hc357_net=-0.173 <= 0 |
| log_ret_30s_q90 | LONG | Top0.1% | side=LONG (need SHORT), hc357_sharpe=-0.099 < 0.5, hc357_net=-0.748 <= 0, ci_low_95=-0.876 <= -0.5 |
| log_ret_60s_q90 | LONG | Top0.5% | side=LONG (need SHORT), hc357_sharpe=-0.107 < 0.5, hc357_net=-0.744 <= 0, ci_low_95=-0.762 <= -0.5 |
| log_ret_10s_q90 | LONG | Top10% | side=LONG (need SHORT), hc357_sharpe=-0.119 < 0.5, hc357_net=-0.784 <= 0 |
| p_reversal_15s | LONG | Top5% | side=LONG (need SHORT), hc357_sharpe=-0.119 < 0.5, hc357_net=-0.770 <= 0 |
| log_ret_60s_q50 | LONG | Top1% | side=LONG (need SHORT), hc357_sharpe=-0.121 < 0.5, hc357_net=-0.795 <= 0, ci_low_95=-0.680 <= -0.5 |
| p_reversal_15s | LONG | Top10% | side=LONG (need SHORT), hc357_sharpe=-0.123 < 0.5, hc357_net=-0.788 <= 0 |
| p_reversal_30s | LONG | Top5% | side=LONG (need SHORT), hc357_sharpe=-0.123 < 0.5, hc357_net=-0.778 <= 0, ci_low_95=-0.506 <= -0.5 |

_(173 more rejections truncated)_