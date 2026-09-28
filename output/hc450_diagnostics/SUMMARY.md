# HC #450 R3+R4 Diagnostics Summary
Generated: 2026-05-20 07:49:59.400455  
Commission used: 0.376 ticks (= $4.70 / $12.50)  
Stride: 250 ms  

## Models processed
- **cnn_mamba_v3_4_2**: 32 OOT days, 1,579,225 pooled predictions @ 1s horizon
- **cnn_mamba_v3_3**: 32 OOT days, 1,579,225 pooled predictions @ 1s horizon
- **cnn_mamba_v2**: 41 OOT days, 2,298,912 pooled predictions @ 1s horizon
- **patchtst**: 32 OOT days, 1,929,756 pooled predictions @ 1s horizon

## 1. Are we trading pressure or noise?
Criterion: autocorr at lag-4 (=1s after prediction). >0.6 = pressure-like (signal persists, model is predicting a slow-moving order-book imbalance). <0.3 = noisy (signal flips faster than we can act on it).

| model | autocorr lag-1 (250ms) | autocorr lag-4 (1s) | autocorr lag-40 (10s) | sign-flips/sec | top-10% mean run (s) | verdict |
|---|---|---|---|---|---|---|
| cnn_mamba_v3_4_2 | 0.019 | 0.011 | 0.003 | 1.804 | 0.37 | NOISY |
| cnn_mamba_v3_3 | 0.076 | 0.069 | 0.029 | 1.816 | 0.30 | NOISY |
| cnn_mamba_v2 | 0.034 | 0.010 | 0.004 | 1.972 | 0.28 | NOISY |
| patchtst | 0.063 | 0.012 | 0.007 | 1.875 | 0.29 | NOISY |

## 2. Is PatchTST smoother than CNN-Mamba v3.4.2?
- autocorr lag-4 (1s): PatchTST = 0.012 vs v3.4.2 = 0.011
- sign-flips/sec: PatchTST = 1.875 vs v3.4.2 = 1.804
- top-10% mean run (s): PatchTST = 0.29 vs v3.4.2 = 0.37
- **verdict: YES, PatchTST is smoother**

## 3. Should we smooth our signal before trading?
- **cnn_mamba_v3_4_2**: lag-1 autocorr=0.019, lag-4=0.011 (decay 0.009). → YES — heavy smoothing (EMA ~1-2s) before any trade decision.
- **cnn_mamba_v3_3**: lag-1 autocorr=0.076, lag-4=0.069 (decay 0.007). → YES — heavy smoothing (EMA ~1-2s) before any trade decision.
- **cnn_mamba_v2**: lag-1 autocorr=0.034, lag-4=0.010 (decay 0.025). → YES — heavy smoothing (EMA ~1-2s) before any trade decision.
- **patchtst**: lag-1 autocorr=0.063, lag-4=0.012 (decay 0.051). → YES — heavy smoothing (EMA ~1-2s) before any trade decision.

## 4. Top-10% signal persistence (how many seconds does it stay 'extreme'?)
| model | mean run (ticks) | mean run (sec) | given top-10% at t, frac of next 10s also top-10% |
|---|---|---|---|
| cnn_mamba_v3_4_2 | 1.5 | 0.37 | 0.298 |
| cnn_mamba_v3_3 | 1.2 | 0.30 | 0.143 |
| cnn_mamba_v2 | 1.1 | 0.28 | 0.108 |
| patchtst | 1.2 | 0.29 | 0.108 |

## 5. Top-10% short signal MFE/MAE headline (per HC #450 R3)
Direction-adjusted mean realized move (ticks) for the top-10% short signals (pred<0, |pred| in top decile of |pred|):

| model | horizon | n | mean realized (tk) | p90 realized | win rate | net after 0.376tk commission |
|---|---|---|---|---|---|---|
| cnn_mamba_v3_4_2 | 1s | 128101 | 0.177 | 2.500 | 0.453 | -0.199 |
| cnn_mamba_v3_4_2 | 5s | 154836 | 0.155 | 5.000 | 0.482 | -0.221 |
| cnn_mamba_v3_4_2 | 10s | 145375 | 0.130 | 6.500 | 0.488 | -0.246 |
| cnn_mamba_v3_4_2 | 30s | 37551 | -0.252 | 9.000 | 0.458 | -0.628 |
| cnn_mamba_v3_3 | 1s | 78500 | 0.401 | 2.500 | 0.523 | 0.025 |
| cnn_mamba_v3_3 | 5s | 61018 | 0.396 | 4.500 | 0.522 | 0.020 |
| cnn_mamba_v3_3 | 10s | 45037 | 0.364 | 6.000 | 0.515 | -0.012 |
| cnn_mamba_v3_3 | 30s | 23675 | 0.389 | 9.000 | 0.506 | 0.013 |
| cnn_mamba_v2 | 1s | 28966 | 0.741 | 3.000 | 0.606 | 0.365 |
| cnn_mamba_v2 | 5s | 36547 | 0.786 | 5.000 | 0.581 | 0.410 |
| cnn_mamba_v2 | 10s | 56521 | 0.711 | 6.000 | 0.557 | 0.335 |
| patchtst | 1s | 75057 | -0.060 | 2.000 | 0.502 | -0.436 |

## 6. Top-1% MFE/MAE headline
| model | side | horizon | n | mean realized (tk) | net after commission |
|---|---|---|---|---|---|
| cnn_mamba_v3_4_2 | long | 1s | 8312 | -0.008 | -0.384 |
| cnn_mamba_v3_4_2 | long | 10s | 373 | 0.487 | 0.111 |
| cnn_mamba_v3_4_2 | long | 30s | 14674 | 0.188 | -0.188 |
| cnn_mamba_v3_4_2 | short | 1s | 7481 | 0.176 | -0.200 |
| cnn_mamba_v3_4_2 | short | 5s | 15776 | 0.290 | -0.086 |
| cnn_mamba_v3_4_2 | short | 10s | 15399 | 0.046 | -0.330 |
| cnn_mamba_v3_4_2 | short | 30s | 1077 | 0.858 | 0.481 |
| cnn_mamba_v3_3 | long | 1s | 1655 | 0.756 | 0.380 |
| cnn_mamba_v3_3 | long | 5s | 2491 | 0.960 | 0.584 |
| cnn_mamba_v3_3 | long | 10s | 4596 | 1.026 | 0.650 |
| cnn_mamba_v3_3 | long | 30s | 9437 | 0.954 | 0.578 |
| cnn_mamba_v3_3 | short | 1s | 14138 | 0.290 | -0.086 |
| cnn_mamba_v3_3 | short | 5s | 13290 | 0.304 | -0.072 |
| cnn_mamba_v3_3 | short | 10s | 11176 | 0.362 | -0.014 |
| cnn_mamba_v3_3 | short | 30s | 6314 | 0.564 | 0.188 |
| cnn_mamba_v2 | long | 1s | 21698 | 0.554 | 0.178 |
| cnn_mamba_v2 | long | 5s | 21399 | 0.645 | 0.269 |
| cnn_mamba_v2 | long | 10s | 21911 | 0.659 | 0.283 |
| cnn_mamba_v2 | short | 1s | 1292 | 1.076 | 0.700 |
| cnn_mamba_v2 | short | 5s | 1585 | 0.995 | 0.619 |
| cnn_mamba_v2 | short | 10s | 1068 | 0.893 | 0.517 |
| patchtst | long | 1s | 5362 | 0.551 | 0.175 |
| patchtst | long | 5s | 19296 | 0.448 | 0.072 |
| patchtst | long | 10s | 19295 | 0.390 | 0.014 |
| patchtst | short | 1s | 13936 | -1.379 | -1.755 |

## Caveat on MFE/MAE proxy
The NPZs do not contain per-sample MFE/MAE arrays for the OOT set (v3.4.2 has the fields but they are mask=0). We use the realized close-of-horizon move (`labels_h`, in ticks) as the proxy. True MFE within the horizon is **larger** than what we report; true MAE (downside excursion) is **worse** than what we report. To get true MFE/MAE, regenerate the OOT NPZs from MBO replay with the bookkeeping enabled.
