# PASS 7 — Robustness Validation of Pass 6 Pocket

Pocket = S_Top0.5% × agree_15 × no_reversal × golden-ToD × vol_mid (n=19, mean +1.29t)

## R1. Leave-One-Day-Out CV

| Holdout day | n_fill | mean_t | WR% | Sharpe | Total_t |
|---|---:|---:|---:|---:|---:|
| 0 | 0 | nan | nan | nan | nan |
| 1 | 0 | nan | nan | nan | nan |
| 2 | 19 | 1.289 | 68.4 | 2.70 | 24.50 |
| 3 | 0 | nan | nan | nan | nan |
| 4 | 0 | nan | nan | nan | nan |

## R2. Permutation Test (1000 random sign-shuffles)

- Observed pocket Sharpe: **2.697**
- Null mean Sharpe: 1.312
- Null 95th percentile: 2.643
- Null 99th percentile: 3.249
- **p-value (one-tail) = 0.0460**

## R3. ToD Bucket Sensitivity

| Bucket set | n_fill | mean_t | WR% | Sharpe |
|---|---:|---:|---:|---:|
| original_4_6_8 | 19 | 1.289 | 68.4 | 2.70 |
| shift_back_3_5_7 | 26 | -0.769 | 38.5 | -1.71 |
| shift_fwd_5_7_9 | 26 | -0.769 | 38.5 | -1.71 |
| only_4_6 | 19 | 1.289 | 68.4 | 2.70 |
| only_6_8 | 10 | 1.850 | 80.0 | 3.87 |
| only_4_8 | 9 | 0.667 | 55.6 | 0.79 |
| expanded_3_4_6_8 | 19 | 1.289 | 68.4 | 2.70 |
| expanded_4_6_8_9 | 19 | 1.289 | 68.4 | 2.70 |

## R4. Drop-One-Filter Ablation

| Variant | n_fill | mean_t | WR% | Sharpe |
|---|---:|---:|---:|---:|
| all_filters | 19 | 1.289 | 68.4 | 2.70 |
| drop_agree_15 | 19 | 1.289 | 68.4 | 2.70 |
| drop_no_rev | 19 | 1.289 | 68.4 | 2.70 |
| drop_vol_mid | 33 | 0.667 | 63.6 | 1.37 |
| drop_golden_tod | 54 | 0.241 | 51.9 | 0.72 |
| loose_band_Top1 | 48 | 0.688 | 62.5 | 1.92 |
| loose_band_Top5 | 262 | 0.639 | 61.8 | 3.85 |
| tight_band_Top0p1 | 2 | 3.000 | 100.0 | 0.00 |

## R5. tp4sl3 vs tp8sl5 Cross-Check

| FIFO | n_fill | mean_t | WR% | Sharpe |
|---|---:|---:|---:|---:|
| tp4sl3 | 19 | 1.289 | 68.4 | 2.70 |
| tp8sl5 | 19 | 1.263 | 68.4 | 2.45 |

## R6. Newey-West Effective Sample Size

- n_obs = 19
- lag-1 autocorr = 0.114
- n_eff = 15.5
- naive Sharpe = 2.697
- AR(1)-corrected Sharpe = 2.434

