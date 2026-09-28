# v3.2 Fold 0 OOT Deep Analysis — Morning Briefing

**Generated**: 2026-05-13T01:56:09.927526
**Source**: `/home/jupiter/Lvl3Quant/output/v3_2_deep_sim_20260512/fold_00_oot_predictions.npz`
**Samples analyzed**: 241351
**OOT dates**: ['20260223', '20260224', '20260225', '20260226', '20260227']

## TL;DR — bottom-line findings per HC #307 deliverable

### 1. Aggregate IC by horizon (legacy compat)

| Head | n | IC | MagCorr | DA |
|---|---|---|---|---|
| log_ret_1s | 241351 | 0.2387 | 0.0374 | 0.6413 |
| log_ret_5s | 241351 | nan | nan | 0.5487 |
| log_ret_10s | 241351 | nan | nan | 0.5218 |
| log_ret_30s | 241351 | nan | nan | 0.5130 |
| pred_mfe_30s_ticks | 81154 | 0.2799 | 0.2802 | 0.9996 |
| pred_mae_30s_ticks | 81154 | 0.3623 | 0.3623 | 0.9995 |
| log_ret_10s_q10 | 241351 | nan | nan | 0.5019 |
| log_ret_10s_q50 | 241351 | nan | nan | 0.5250 |
| log_ret_10s_q90 | 241351 | nan | nan | 0.4934 |
| log_ret_30s_q10 | 241351 | nan | nan | 0.5067 |
| log_ret_30s_q50 | 241351 | nan | nan | 0.5102 |
| log_ret_30s_q90 | 241351 | nan | nan | 0.4843 |

### 2. CONFIDENCE BANDS — DA @ Top 1% (per HC #306 — the key cells)

| Head | n | DA_all | DA_long | DA_short | IC | MagCorr | Sharpe_toy |
|---|---|---|---|---|---|---|---|
| log_ret_1s | 2373 | 0.8478 | 0.6675 | 0.5909 | 0.4531 | 0.2349 | 26.064 |
| log_ret_5s | 2504 | 0.6746 | 0.6914 | 0.5485 | nan | nan | nan |
| log_ret_10s | 2493 | 0.6209 | 0.5789 | 0.5374 | nan | nan | nan |
| log_ret_30s | 2331 | 0.5442 | 0.6061 | 0.5061 | nan | nan | nan |
| pred_mfe_30s_ticks | 832 | 0.9988 | 0.9880 | nan | -0.2761 | -0.2761 | 47.324 |
| pred_mae_30s_ticks | 827 | 1.0000 | nan | 0.9613 | 0.2381 | 0.2381 | 44.123 |

### 3. ROLLING-AVG 1s EXIT CONFLUENCE (USER IDEA — HC #307)

Hypothesis: smoothing 1s preds via rolling mean cleans tick noise → sharper entry+exit.

| K_smooth | smooth_ms | tgt_horizon | n | DA_all | DA_long | DA_short | IC | sign_flip_rate |
|---|---|---|---|---|---|---|---|---|
| 1 | 250 | 1s | 241349 | 0.6413 | 0.4259 | 0.4103 | 0.2387 | 0.4694 |
| 1 | 250 | 5s | 241349 | 0.5597 | 0.4692 | 0.4633 | nan | 0.4694 |
| 1 | 250 | 10s | 241349 | 0.5377 | 0.4733 | 0.4722 | nan | 0.4694 |
| 1 | 250 | 30s | 241349 | 0.5150 | 0.4690 | 0.4851 | nan | 0.4694 |
| 3 | 750 | 1s | 241349 | 0.5895 | 0.4052 | 0.3686 | 0.1679 | 0.2316 |
| 3 | 750 | 5s | 241349 | 0.5423 | 0.4633 | 0.4431 | nan | 0.2316 |
| 3 | 750 | 10s | 241349 | 0.5254 | 0.4677 | 0.4576 | nan | 0.2316 |
| 3 | 750 | 30s | 241349 | 0.5058 | 0.4618 | 0.4739 | nan | 0.2316 |
| 5 | 1250 | 1s | 241347 | 0.5682 | 0.3952 | 0.3537 | 0.1389 | 0.1649 |
| 5 | 1250 | 5s | 241347 | 0.5375 | 0.4629 | 0.4376 | nan | 0.1649 |
| 5 | 1250 | 10s | 241347 | 0.5223 | 0.4679 | 0.4534 | nan | 0.1649 |
| 5 | 1250 | 30s | 241347 | 0.5023 | 0.4589 | 0.4699 | nan | 0.1649 |
| 10 | 2500 | 1s | 241341 | 0.5434 | 0.3833 | 0.3374 | 0.0967 | 0.0988 |
| 10 | 2500 | 5s | 241341 | 0.5284 | 0.4595 | 0.4289 | nan | 0.0988 |
| 10 | 2500 | 10s | 241341 | 0.5157 | 0.4642 | 0.4471 | nan | 0.0988 |
| 10 | 2500 | 30s | 241341 | 0.4963 | 0.4517 | 0.4647 | nan | 0.0988 |
| 20 | 5000 | 1s | 241331 | 0.5149 | 0.3630 | 0.3218 | 0.0485 | 0.0568 |
| 20 | 5000 | 5s | 241331 | 0.5162 | 0.4498 | 0.4199 | nan | 0.0568 |
| 20 | 5000 | 10s | 241331 | 0.5097 | 0.4592 | 0.4426 | nan | 0.0568 |
| 20 | 5000 | 30s | 241331 | 0.4934 | 0.4478 | 0.4620 | nan | 0.0568 |
| 50 | 12500 | 1s | 241301 | 0.4974 | 0.3493 | 0.3129 | 0.0000 | 0.0254 |
| 50 | 12500 | 5s | 241301 | 0.5002 | 0.4318 | 0.4098 | nan | 0.0254 |
| 50 | 12500 | 10s | 241301 | 0.4955 | 0.4419 | 0.4328 | nan | 0.0254 |
| 50 | 12500 | 30s | 241301 | 0.4860 | 0.4364 | 0.4566 | nan | 0.0254 |

### 4. PRICE-PATH FROM PREDICTIONS — avg signed forward return per band

Per top-X% confidence signal of each entry head, what's the realized signed return at each forward horizon?
(Signed = direction-correct trade pnl in log-ret units)

| entry_head | band | n | avg_signed_ret_1s | avg_signed_ret_5s | avg_signed_ret_10s | avg_signed_ret_30s | avg_signed_ret_60s |
|---|---|---|---|---|---|---|---|
| log_ret_1s | Top 0.1% | 244 | 1.602459 | nan | nan | nan | 0.000000 |
| log_ret_1s | Top 1% | 2373 | 0.946060 | nan | nan | nan | 0.000000 |
| log_ret_1s | Top 5% | 12083 | 0.818257 | nan | nan | nan | 0.000000 |
| log_ret_1s | All | 241351 | 0.314465 | nan | nan | nan | 0.000000 |
| log_ret_5s | Top 0.1% | 245 | 1.114286 | nan | nan | nan | 0.000000 |
| log_ret_5s | Top 1% | 2504 | 0.700879 | nan | nan | nan | 0.000000 |
| log_ret_5s | Top 5% | 12104 | 0.573199 | nan | nan | nan | 0.000000 |
| log_ret_5s | All | 241351 | 0.253732 | nan | nan | nan | 0.000000 |
| log_ret_10s | Top 0.1% | 239 | 1.092050 | 1.234310 | 0.920502 | 1.075314 | 0.000000 |
| log_ret_10s | Top 1% | 2493 | 0.664661 | nan | nan | nan | 0.000000 |
| log_ret_10s | Top 5% | 12008 | 0.503165 | nan | nan | nan | 0.000000 |
| log_ret_10s | All | 241351 | 0.186034 | nan | nan | nan | 0.000000 |
| log_ret_30s | Top 0.1% | 237 | 0.843882 | 0.759494 | 0.118143 | -0.257384 | 0.000000 |
| log_ret_30s | Top 1% | 2331 | 0.602317 | nan | nan | nan | 0.000000 |
| log_ret_30s | Top 5% | 12103 | 0.431629 | nan | nan | nan | 0.000000 |
| log_ret_30s | All | 241351 | 0.129407 | nan | nan | nan | 0.000000 |

### 5. MULTI-HEAD AGREEMENT / CONFLUENCE

When N of {1s,5s,10s,30s} heads agree on sign, what's DA on each target?

| target_horizon | agreement_count | n | DA | IC | avg_signed_ret |
|---|---|---|---|---|---|
| log_ret_1s | 3 | 66530 | 0.3215 | 0.0216 | -0.001841 |
| log_ret_1s | 4 | 152320 | 0.4265 | 0.2149 | 0.350433 |
| log_ret_5s | 3 | 66530 | 0.4161 | nan | nan |
| log_ret_5s | 4 | 152320 | 0.4728 | nan | nan |
| log_ret_10s | 3 | 66530 | 0.4365 | nan | nan |
| log_ret_10s | 4 | 152320 | 0.4766 | nan | nan |
| log_ret_30s | 3 | 66530 | 0.4567 | nan | nan |
| log_ret_30s | 4 | 152320 | 0.4854 | nan | nan |

### 6. QUANTILE CALIBRATION (is q90 actually the 90th percentile?)

| horizon | n | p_real_below_q10 (want ~0.10) | p_real_below_q50 (want ~0.50) | p_real_below_q90 (want ~0.90) | avg(q90-q10) |
|---|---|---|---|---|---|
| 10s | 241351 | 0.2816 | 0.4701 | 0.6888 | 3.840059 |
| 30s | 241351 | 0.3018 | 0.4688 | 0.6599 | 5.521449 |

### 7. MFE/MAE BAND CALIBRATION (do path heads predict path correctly?)

(See `mfe_mae_bands.csv` for full per-band breakdown — summary below)

| entry_head | path_horizon | n | avg_pred_mfe | avg_real_mfe | corr_mfe | mfe_cal_ratio |
|---|---|---|---|---|---|---|
| log_ret_30s | 30s | 2331 | 1.9257 | 2.0826 | -0.1494 | 0.9247 |
| log_ret_30s | 60s | 2331 | -0.2190 | 0.0000 | nan | -218990.5312 |

### 8. REVERSAL HEAD VALUE (when p_reversal > thr, does it reverse?)

| head | threshold | n | hit_rate | base_rate | lift |
|---|---|---|---|---|---|
| p_reversal_15s | 0.30 | 81158 | 0.8000 | 0.8000 | +0.0000 |
| p_reversal_15s | 0.40 | 81133 | 0.8000 | 0.8000 | +0.0000 |
| p_reversal_15s | 0.50 | 32290 | 0.8661 | 0.8000 | +0.0660 |
| p_reversal_15s | 0.60 | 1404 | 0.9330 | 0.8000 | +0.1330 |
| p_reversal_15s | 0.70 | 65 | 0.9385 | 0.8000 | +0.1384 |
| p_reversal_30s | 0.30 | 81158 | 0.8607 | 0.8607 | +0.0000 |
| p_reversal_30s | 0.40 | 81157 | 0.8608 | 0.8607 | +0.0000 |
| p_reversal_30s | 0.50 | 81140 | 0.8607 | 0.8607 | +0.0000 |
| p_reversal_30s | 0.60 | 26752 | 0.9201 | 0.8607 | +0.0594 |
| p_reversal_30s | 0.70 | 915 | 0.9399 | 0.8607 | +0.0791 |

### 9. TIMING ALIGNMENT (does pred_time_to_mfe sharpen MFE accuracy?)

| time_bucket | n | avg_pred_time | avg_pred_mfe | avg_real_mfe | corr_mfe | calibration |
|---|---|---|---|---|---|---|
| 15-20s | 26239 | 18.64 | 1.7773 | 6.9664 | 0.2445 | 0.2551 |
| 20-25s | 54884 | 23.05 | 1.5208 | 4.1706 | 0.1544 | 0.3647 |
| 25-30s | 30 | 25.66 | 0.8515 | 4.6000 | -0.4261 | 0.1851 |

### 10. PREDICTION DISTRIBUTIONS (we need this data, can't lose it)

(Full per-head stats in `pred_distributions.json` — top heads summarized below)

| head | n | mean | std | p05 | p95 | frac_pos | target_mean | target_std |
|---|---|---|---|---|---|---|---|---|
| pred_log_ret_1s | 241351 | -0.039515 | 0.285462 | -0.558594 | 0.390625 | 0.489 | 0.015171 | 1.634116 |
| pred_log_ret_5s | 241351 | -0.142029 | 0.407645 | -0.867188 | 0.498047 | 0.392 | nan | nan |
| pred_log_ret_10s | 241351 | -0.206044 | 0.491325 | -1.046875 | 0.550781 | 0.371 | nan | nan |
| pred_log_ret_30s | 241351 | -0.471259 | 0.799052 | -1.812500 | 0.660156 | 0.349 | nan | nan |
| pred_pred_mfe_30s_ticks | 241351 | 1.650981 | 0.211882 | 1.312500 | 1.937500 | 1.000 | nan | nan |
| pred_pred_mae_30s_ticks | 241351 | -1.315414 | 0.214277 | -1.617188 | -0.960938 | 0.000 | nan | nan |

## Files written

- `aggregate_ic.json` (1.4 KB)
- `confidence_bands.csv` (10.9 KB)
- `confidence_bands.json` (21.3 KB)
- `fold_00_oot_predictions.metrics.json` (0.8 KB)
- `fold_00_oot_predictions.npz` (12943.9 KB)
- `mfe_mae_bands.csv` (3.8 KB)
- `multi_head_agreement.csv` (0.5 KB)
- `pred_distributions.json` (13.5 KB)
- `price_path_from_preds.csv` (5.0 KB)
- `quantile_calibration.csv` (0.3 KB)
- `reversal_head_value.csv` (0.9 KB)
- `rolling_avg_exit_confluence.csv` (2.5 KB)
- `timing_alignment.csv` (0.4 KB)
