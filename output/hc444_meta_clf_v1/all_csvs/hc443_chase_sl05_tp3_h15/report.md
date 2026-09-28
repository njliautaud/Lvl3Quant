# HC #444 Meta-Classifier v1 Report

Fills source: `hc443_chase_sl05_tp3_h15_fifo_fills.csv`  (2431 fills × 32 dates)

## Time-ordered 70/30 (OOS holdout)
- n_train = 1901, n_test = 530
- base positive-class rate (test) = 0.249
- AUC = 0.5267
- best_iter = 4

Threshold sweep (subset of test fills above prob threshold):

| thr | n | mean_tk | PF | WR | sharpe |
|---|---:|---:|---:|---:|---:|

Top-5 feature importance (gain):
- `minute_of_day`: 138.1
- `pred_strength`: 120.5
- `signal_density`: 68.6
- `pred_pct_in_day`: 21.7
- `queue_ahead_log1p`: 20.1

## Random 80/20 (learnability)
- n_train = 1949, n_test = 482
- base positive-class rate (test) = 0.234
- AUC = 0.5915
- best_iter = 30

Threshold sweep (subset of test fills above prob threshold):

| thr | n | mean_tk | PF | WR | sharpe |
|---|---:|---:|---:|---:|---:|
| 0.30 | 64 | -0.149 | 0.74 | 34.4% | -1.01 |

Top-5 feature importance (gain):
- `pred_strength`: 495.5
- `minute_of_day`: 360.4
- `queue_ahead_log1p`: 320.2
- `pred_strength_z_within_day`: 309.1
- `signal_density`: 270.8

## HC #444 R3 verdict
❌ NO threshold yields positive mean_tk + PF ≥ 1.10 + n ≥ 50 on the time-ordered OOS test set.

Interpretation: pre-trade-known features (pred_strength, TOD, DOW, queue, signal density, side) do not separate profitable from unprofitable fills well enough to rescue the canonical loss. Stronger features required (book imbalance at signal, recent realized vol, multi-model agreement).