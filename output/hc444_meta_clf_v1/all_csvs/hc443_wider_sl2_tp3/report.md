# HC #444 Meta-Classifier v1 Report

Fills source: `hc443_wider_sl2_tp3_fifo_fills.csv`  (2431 fills × 32 dates)

## Time-ordered 70/30 (OOS holdout)
- n_train = 1901, n_test = 530
- base positive-class rate (test) = 0.406
- AUC = 0.5159
- best_iter = 6

Threshold sweep (subset of test fills above prob threshold):

| thr | n | mean_tk | PF | WR | sharpe |
|---|---:|---:|---:|---:|---:|
| 0.30 | 530 | -0.601 | 0.40 | 40.6% | -8.70 |
| 0.35 | 496 | -0.604 | 0.40 | 40.3% | -8.47 |
| 0.40 | 138 | -0.601 | 0.34 | 42.0% | -4.98 |

Top-5 feature importance (gain):
- `pred_strength`: 149.3
- `queue_ahead_log1p`: 113.7
- `minute_of_day`: 108.7
- `signal_density`: 48.9
- `pred_pct_in_day`: 48.1

## Random 80/20 (learnability)
- n_train = 1949, n_test = 482
- base positive-class rate (test) = 0.402
- AUC = 0.5438
- best_iter = 30

Threshold sweep (subset of test fills above prob threshold):

| thr | n | mean_tk | PF | WR | sharpe |
|---|---:|---:|---:|---:|---:|
| 0.30 | 455 | -0.498 | 0.57 | 41.1% | -5.44 |
| 0.35 | 370 | -0.423 | 0.61 | 42.2% | -4.17 |
| 0.40 | 207 | -0.400 | 0.62 | 44.9% | -3.02 |
| 0.45 | 58 | -0.686 | 0.40 | 39.7% | -3.01 |

Top-5 feature importance (gain):
- `pred_strength`: 382.2
- `signal_density`: 343.5
- `minute_of_day`: 313.6
- `pred_strength_z_within_day`: 306.0
- `queue_ahead_log1p`: 292.3

## HC #444 R3 verdict
❌ NO threshold yields positive mean_tk + PF ≥ 1.10 + n ≥ 50 on the time-ordered OOS test set.

Interpretation: pre-trade-known features (pred_strength, TOD, DOW, queue, signal density, side) do not separate profitable from unprofitable fills well enough to rescue the canonical loss. Stronger features required (book imbalance at signal, recent realized vol, multi-model agreement).