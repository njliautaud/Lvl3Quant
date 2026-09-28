# HC #475 R1 — Raw Signal Asymmetry Diagnostic

Generated: 2026-05-21T17:19:34
Source NPZ: `/home/jupiter/Lvl3Quant/output/hc432_v342_47day_validation/fold_00_ep1_oot_inference_47day_hc432.npz`

## Per-head sign distribution + magnitude symmetry

| Head | n | pos_share | neg_share | pos_p90 | neg_p90 | mag_ratio | count_ratio |
|------|---|-----------|-----------|---------|---------|-----------|-------------|
| `pred_log_ret_1s` | 900454 | 0.687 | 0.313 | 0.2061 | 0.2754 | 0.748 | 2.195 |
| `pred_log_ret_5s` | 900454 | 0.687 | 0.313 | 0.1982 | 0.3301 | 0.601 | 2.196 |
| `pred_log_ret_10s` | 900454 | 0.685 | 0.315 | 0.25 | 0.3008 | 0.831 | 2.171 |
| `pred_log_ret_30s` | 900454 | 0.274 | 0.726 | 4.219 | 1.828 | 2.308 | 0.378 |
| `pred_log_ret_60s` | 900454 | 0.004 | 0.996 | 0.2656 | 0.8438 | 0.315 | 0.004 |
| `pred_log_ret_5min` | 900454 | 0.701 | 0.299 | 1.547 | 0.5312 | 2.912 | 2.344 |
| `pred_p_up_5s` | 900454 | 0.024 | 0.976 | 0.006409 | 0.2832 | 0.023 | 0.025 |
| `pred_p_up_10s` | 900454 | 0.474 | 0.526 | 0.03711 | 0.1797 | 0.207 | 0.901 |
| `pred_p_up_30s` | 900454 | 0.760 | 0.240 | 0.09961 | 0.07227 | 1.378 | 3.161 |
| `pred_p_up_60s` | 900454 | 0.991 | 0.009 | 0.4043 | 0.5195 | 0.778 | 105.424 |
| `pred_log_ret_10s_q10` | 900454 | 0.000 | 1.000 | nan | 3.109 | 0.000 | 0.000 |
| `pred_log_ret_10s_q50` | 900454 | 0.755 | 0.245 | 0.04175 | 0.07568 | 0.552 | 3.089 |
| `pred_log_ret_10s_q90` | 900454 | 1.000 | 0.000 | 3.359 | nan | inf | inf |
| `pred_log_ret_30s_q10` | 900454 | 0.000 | 1.000 | nan | 4.844 | 0.000 | 0.000 |
| `pred_log_ret_30s_q50` | 900454 | 0.637 | 0.363 | 0.1396 | 0.3223 | 0.433 | 1.754 |
| `pred_log_ret_30s_q90` | 900454 | 1.000 | 0.000 | 5.25 | nan | inf | inf |
| `pred_log_ret_60s_q10` | 900454 | 0.000 | 1.000 | 1.027 | 0.8945 | 1.148 | 0.000 |
| `pred_log_ret_60s_q50` | 900454 | 0.687 | 0.313 | 1.477 | 2.297 | 0.643 | 2.190 |
| `pred_log_ret_60s_q90` | 900454 | 0.687 | 0.313 | 1.617 | 1.719 | 0.941 | 2.199 |
| `pred_pred_mfe_30s_ticks` | 900454 | 0.686 | 0.314 | 2.984 | 11.5 | 0.260 | 2.185 |
| `pred_pred_mfe_60s_ticks` | 900454 | 0.144 | 0.856 | 0.3926 | 1.242 | 0.316 | 0.169 |
| `pred_pred_time_to_mfe_secs` | 900454 | 1.000 | 0.000 | 12.81 | nan | inf | inf |
| `pred_pred_realized_vol_30s_ticks` | 900454 | 1.000 | 0.000 | 4.094 | nan | inf | inf |
| `pred_fifo_tp4sl3_net` | 900454 | 0.000 | 1.000 | nan | 1.531 | 0.000 | 0.000 |
| `pred_fifo_tp8sl5_net` | 900454 | 0.000 | 1.000 | 0.1822 | 1.977 | 0.092 | 0.000 |
| `pred_fifo_tp4sl3_hit_tp` | 900454 | 0.164 | 0.836 | 0.1079 | 0.1934 | 0.558 | 0.196 |
| `pred_fifo_tp8sl5_hit_tp` | 900454 | 0.000 | 1.000 | 0.1116 | 1.531 | 0.073 | 0.000 |

## Interpretation per HC #475 R1

Reject the model on the both-sides competency bar if **any** trading-horizon head shows:
- `mag_ratio_pos_over_neg_p90` outside [0.5, 2.0], AND
- `min(|IC_long|, |IC_short|) / max(|IC_long|, |IC_short|) < 0.5`

If both fail: model is short-tail-only at the representation level — alpha redev required.
If only count_ratio fails: bias is dataset/label-distribution, not representation.
If only mag_ratio fails but IC ratio is OK: bias is calibration, fixable post-hoc.