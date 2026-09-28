# HC #444 Meta-Classifier v1 Report

Fills source: `v342_lshort_5s_ensemble_50_50_fifo_fills.csv`  (3982 fills × 16 dates)

## Time-ordered 70/30 (OOS holdout)
- n_train = 2960, n_test = 1022
- base positive-class rate (test) = 0.486
- AUC = 0.5325
- best_iter = 45

Threshold sweep (subset of test fills above prob threshold):

| thr | n | mean_tk | PF | WR | sharpe |
|---|---:|---:|---:|---:|---:|
| 0.30 | 1022 | -0.087 | 0.64 | 48.6% | -7.09 |
| 0.35 | 1022 | -0.087 | 0.64 | 48.6% | -7.09 |
| 0.40 | 999 | -0.084 | 0.64 | 48.9% | -6.82 |
| 0.45 | 807 | -0.077 | 0.67 | 50.3% | -5.52 |
| 0.50 | 390 | -0.061 | 0.73 | 52.3% | -3.05 |
| 0.55 | 89 | -0.034 | 0.84 | 55.1% | -0.82 |

Top-5 feature importance (gain):
- `pred_strength_z_within_day`: 693.7
- `queue_ahead_log1p`: 504.4
- `minute_of_day`: 436.1
- `signal_density`: 393.6
- `pred_strength`: 312.9

## Random 80/20 (learnability)
- n_train = 3621, n_test = 361
- base positive-class rate (test) = 0.471
- AUC = 0.5175
- best_iter = 1

Threshold sweep (subset of test fills above prob threshold):

| thr | n | mean_tk | PF | WR | sharpe |
|---|---:|---:|---:|---:|---:|
| 0.30 | 361 | -0.101 | 0.59 | 47.1% | -4.88 |
| 0.35 | 361 | -0.101 | 0.59 | 47.1% | -4.88 |
| 0.40 | 361 | -0.101 | 0.59 | 47.1% | -4.88 |
| 0.45 | 361 | -0.101 | 0.59 | 47.1% | -4.88 |

Top-5 feature importance (gain):
- `pred_strength`: 28.5
- `queue_ahead_log1p`: 22.7
- `minute_of_day`: 19.3
- `signal_density`: 19.2
- `pred_strength_z_within_day`: 0.0

## HC #444 R3 verdict
❌ NO threshold yields positive mean_tk + PF ≥ 1.10 + n ≥ 50 on the time-ordered OOS test set.

Interpretation: pre-trade-known features (pred_strength, TOD, DOW, queue, signal density, side) do not separate profitable from unprofitable fills well enough to rescue the canonical loss. Stronger features required (book imbalance at signal, recent realized vol, multi-model agreement).