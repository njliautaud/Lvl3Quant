# HC #444 Meta-Classifier v1 Report

Fills source: `hc443_band_top5_short_fifo_fills.csv`  (19774 fills × 36 dates)

## Time-ordered 70/30 (OOS holdout)
- n_train = 14023, n_test = 5751
- base positive-class rate (test) = 0.288
- AUC = 0.5337
- best_iter = 21

Threshold sweep (subset of test fills above prob threshold):

| thr | n | mean_tk | PF | WR | sharpe |
|---|---:|---:|---:|---:|---:|
| 0.30 | 274 | -0.398 | 0.31 | 32.5% | -8.45 |

Top-5 feature importance (gain):
- `pred_strength`: 624.6
- `minute_of_day`: 599.6
- `queue_ahead_log1p`: 311.9
- `pred_strength_z_within_day`: 304.0
- `pred_pct_in_day`: 288.7

## Random 80/20 (learnability)
- n_train = 14796, n_test = 4978
- base positive-class rate (test) = 0.264
- AUC = 0.5307
- best_iter = 26

Threshold sweep (subset of test fills above prob threshold):

| thr | n | mean_tk | PF | WR | sharpe |
|---|---:|---:|---:|---:|---:|
| 0.30 | 417 | -0.321 | 0.49 | 26.9% | -6.34 |

Top-5 feature importance (gain):
- `pred_strength`: 515.6
- `minute_of_day`: 500.6
- `queue_ahead_log1p`: 407.1
- `signal_density`: 297.6
- `pred_strength_z_within_day`: 282.6

## HC #444 R3 verdict
❌ NO threshold yields positive mean_tk + PF ≥ 1.10 + n ≥ 50 on the time-ordered OOS test set.

Interpretation: pre-trade-known features (pred_strength, TOD, DOW, queue, signal density, side) do not separate profitable from unprofitable fills well enough to rescue the canonical loss. Stronger features required (book imbalance at signal, recent realized vol, multi-model agreement).