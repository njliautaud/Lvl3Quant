# HC #444 Meta-Classifier v1 Report

Fills source: `v342_long_5s_top0.5_for_ensemble_fifo_fills.csv`  (2645 fills × 16 dates)

## Time-ordered 70/30 (OOS holdout)
- n_train = 1728, n_test = 917
- base positive-class rate (test) = 0.486
- AUC = 0.5202
- best_iter = 1

Threshold sweep (subset of test fills above prob threshold):

| thr | n | mean_tk | PF | WR | sharpe |
|---|---:|---:|---:|---:|---:|
| 0.30 | 917 | -0.147 | 0.67 | 48.6% | -5.95 |
| 0.35 | 917 | -0.147 | 0.67 | 48.6% | -5.95 |
| 0.40 | 917 | -0.147 | 0.67 | 48.6% | -5.95 |
| 0.45 | 917 | -0.147 | 0.67 | 48.6% | -5.95 |

Top-5 feature importance (gain):
- `minute_of_day`: 32.8
- `pred_pct_in_day`: 13.9
- `signal_density`: 8.4
- `dow`: 6.0
- `pred_strength_z_within_day`: 3.4

## Random 80/20 (learnability)
- n_train = 2325, n_test = 320
- base positive-class rate (test) = 0.478
- AUC = 0.5713
- best_iter = 8

Threshold sweep (subset of test fills above prob threshold):

| thr | n | mean_tk | PF | WR | sharpe |
|---|---:|---:|---:|---:|---:|
| 0.30 | 320 | -0.159 | 0.65 | 47.8% | -3.79 |
| 0.35 | 320 | -0.159 | 0.65 | 47.8% | -3.79 |
| 0.40 | 320 | -0.159 | 0.65 | 47.8% | -3.79 |
| 0.45 | 304 | -0.156 | 0.66 | 48.0% | -3.61 |
| 0.50 | 41 | -0.071 | 0.82 | 53.7% | -0.60 |

Top-5 feature importance (gain):
- `queue_ahead_log1p`: 96.7
- `signal_density`: 90.4
- `pred_strength_z_within_day`: 89.8
- `minute_of_day`: 88.0
- `pred_pct_in_day`: 50.8

## HC #444 R3 verdict
❌ NO threshold yields positive mean_tk + PF ≥ 1.10 + n ≥ 50 on the time-ordered OOS test set.

Interpretation: pre-trade-known features (pred_strength, TOD, DOW, queue, signal density, side) do not separate profitable from unprofitable fills well enough to rescue the canonical loss. Stronger features required (book imbalance at signal, recent realized vol, multi-model agreement).