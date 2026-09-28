# HC #444 Meta-Classifier v1 Report

Fills source: `hc443_tight_sl1_tp2_h05_fifo_fills.csv`  (2076 fills × 31 dates)

## Time-ordered 70/30 (OOS holdout)
- n_train = 1661, n_test = 415
- base positive-class rate (test) = 0.313
- AUC = 0.5197
- best_iter = 1

Threshold sweep (subset of test fills above prob threshold):

| thr | n | mean_tk | PF | WR | sharpe |
|---|---:|---:|---:|---:|---:|

Top-5 feature importance (gain):
- `minute_of_day`: 52.6
- `signal_density`: 19.6
- `pred_pct_in_day`: 16.2
- `dow`: 6.2
- `pred_strength_z_within_day`: 3.9

## Random 80/20 (learnability)
- n_train = 1522, n_test = 554
- base positive-class rate (test) = 0.285
- AUC = 0.5638
- best_iter = 8

Threshold sweep (subset of test fills above prob threshold):

| thr | n | mean_tk | PF | WR | sharpe |
|---|---:|---:|---:|---:|---:|
| 0.30 | 239 | -0.539 | 0.38 | 33.5% | -7.12 |

Top-5 feature importance (gain):
- `pred_strength`: 168.3
- `minute_of_day`: 155.4
- `pred_strength_z_within_day`: 61.3
- `queue_ahead_log1p`: 59.0
- `signal_density`: 52.4

## HC #444 R3 verdict
❌ NO threshold yields positive mean_tk + PF ≥ 1.10 + n ≥ 50 on the time-ordered OOS test set.

Interpretation: pre-trade-known features (pred_strength, TOD, DOW, queue, signal density, side) do not separate profitable from unprofitable fills well enough to rescue the canonical loss. Stronger features required (book imbalance at signal, recent realized vol, multi-model agreement).