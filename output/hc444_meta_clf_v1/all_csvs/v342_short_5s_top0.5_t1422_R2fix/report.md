# HC #444 Meta-Classifier v1 Report

Fills source: `v342_short_5s_top0.5_t1422_R2fix_fifo_fills.csv`  (1337 fills × 15 dates)

## Time-ordered 70/30 (OOS holdout)
- n_train = 1232, n_test = 105
- base positive-class rate (test) = 0.486
- AUC = 0.4261
- best_iter = 1

Threshold sweep (subset of test fills above prob threshold):

| thr | n | mean_tk | PF | WR | sharpe |
|---|---:|---:|---:|---:|---:|
| 0.30 | 105 | -0.405 | 0.43 | 48.6% | -4.13 |
| 0.35 | 105 | -0.405 | 0.43 | 48.6% | -4.13 |
| 0.40 | 105 | -0.405 | 0.43 | 48.6% | -4.13 |
| 0.45 | 105 | -0.405 | 0.43 | 48.6% | -4.13 |
| 0.50 | 52 | -0.530 | 0.33 | 42.3% | -3.83 |

Top-5 feature importance (gain):
- `minute_of_day`: 29.0
- `pred_pct_in_day`: 21.5
- `signal_density`: 10.1
- `pred_strength`: 0.0
- `pred_strength_z_within_day`: 0.0

## Random 80/20 (learnability)
- n_train = 1036, n_test = 301
- base positive-class rate (test) = 0.522
- AUC = 0.5068
- best_iter = 1

Threshold sweep (subset of test fills above prob threshold):

| thr | n | mean_tk | PF | WR | sharpe |
|---|---:|---:|---:|---:|---:|
| 0.30 | 301 | -0.328 | 0.49 | 52.2% | -5.75 |
| 0.35 | 301 | -0.328 | 0.49 | 52.2% | -5.75 |
| 0.40 | 301 | -0.328 | 0.49 | 52.2% | -5.75 |
| 0.45 | 301 | -0.328 | 0.49 | 52.2% | -5.75 |

Top-5 feature importance (gain):
- `signal_density`: 33.7
- `pred_pct_in_day`: 14.9
- `queue_ahead_log1p`: 11.7
- `minute_of_day`: 2.6
- `pred_strength`: 0.0

## HC #444 R3 verdict
❌ NO threshold yields positive mean_tk + PF ≥ 1.10 + n ≥ 50 on the time-ordered OOS test set.

Interpretation: pre-trade-known features (pred_strength, TOD, DOW, queue, signal density, side) do not separate profitable from unprofitable fills well enough to rescue the canonical loss. Stronger features required (book imbalance at signal, recent realized vol, multi-model agreement).