# HC #444 Meta-Classifier v1 Report

Fills source: `v342_long_1s_top0.5_tp1.0_sl0.5_h1.5_c1.0_passive_at_touch_fifo_fills.csv`  (1728 fills × 16 dates)

## Time-ordered 70/30 (OOS holdout)
- n_train = 1067, n_test = 661
- base positive-class rate (test) = 0.496
- AUC = 0.5572
- best_iter = 25

Threshold sweep (subset of test fills above prob threshold):

| thr | n | mean_tk | PF | WR | sharpe |
|---|---:|---:|---:|---:|---:|
| 0.30 | 661 | -0.132 | 0.70 | 49.6% | -4.52 |
| 0.35 | 661 | -0.132 | 0.70 | 49.6% | -4.52 |
| 0.40 | 594 | -0.118 | 0.73 | 50.5% | -3.86 |
| 0.45 | 412 | -0.074 | 0.82 | 53.4% | -2.01 |
| 0.50 | 199 | -0.077 | 0.81 | 53.3% | -1.45 |
| 0.55 | 59 | -0.062 | 0.84 | 54.2% | -0.64 |

Top-5 feature importance (gain):
- `minute_of_day`: 263.2
- `queue_ahead_log1p`: 191.4
- `pred_strength_z_within_day`: 157.3
- `signal_density`: 145.1
- `pred_pct_in_day`: 144.4

## Random 80/20 (learnability)
- n_train = 1411, n_test = 317
- base positive-class rate (test) = 0.502
- AUC = 0.5420
- best_iter = 13

Threshold sweep (subset of test fills above prob threshold):

| thr | n | mean_tk | PF | WR | sharpe |
|---|---:|---:|---:|---:|---:|
| 0.30 | 317 | -0.122 | 0.72 | 50.2% | -2.91 |
| 0.35 | 317 | -0.122 | 0.72 | 50.2% | -2.91 |
| 0.40 | 312 | -0.115 | 0.73 | 50.6% | -2.71 |
| 0.45 | 272 | -0.108 | 0.75 | 51.1% | -2.37 |
| 0.50 | 53 | -0.055 | 0.86 | 54.7% | -0.53 |

Top-5 feature importance (gain):
- `minute_of_day`: 179.7
- `queue_ahead_log1p`: 168.0
- `pred_strength_z_within_day`: 102.4
- `pred_pct_in_day`: 92.5
- `signal_density`: 47.9

## HC #444 R3 verdict
❌ NO threshold yields positive mean_tk + PF ≥ 1.10 + n ≥ 50 on the time-ordered OOS test set.

Interpretation: pre-trade-known features (pred_strength, TOD, DOW, queue, signal density, side) do not separate profitable from unprofitable fills well enough to rescue the canonical loss. Stronger features required (book imbalance at signal, recent realized vol, multi-model agreement).