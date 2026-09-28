# HC #444 Meta-Classifier v1 Report

Fills source: `v342_short_10s_top0.5_t2831_R2fix_fifo_fills.csv`  (1427 fills × 15 dates)

## Time-ordered 70/30 (OOS holdout)
- n_train = 824, n_test = 603
- base positive-class rate (test) = 0.483
- AUC = 0.5063
- best_iter = 1

Threshold sweep (subset of test fills above prob threshold):

| thr | n | mean_tk | PF | WR | sharpe |
|---|---:|---:|---:|---:|---:|
| 0.30 | 603 | -0.181 | 0.79 | 48.3% | -2.80 |
| 0.35 | 603 | -0.181 | 0.79 | 48.3% | -2.80 |
| 0.40 | 603 | -0.181 | 0.79 | 48.3% | -2.80 |
| 0.45 | 603 | -0.181 | 0.79 | 48.3% | -2.80 |

Top-5 feature importance (gain):
- `signal_density`: 21.8
- `queue_ahead_log1p`: 11.5
- `hour`: 10.2
- `dow`: 7.3
- `pred_strength`: 4.0

## Random 80/20 (learnability)
- n_train = 964, n_test = 463
- base positive-class rate (test) = 0.462
- AUC = 0.5376
- best_iter = 2

Threshold sweep (subset of test fills above prob threshold):

| thr | n | mean_tk | PF | WR | sharpe |
|---|---:|---:|---:|---:|---:|
| 0.30 | 463 | -0.253 | 0.71 | 46.2% | -3.51 |
| 0.35 | 463 | -0.253 | 0.71 | 46.2% | -3.51 |
| 0.40 | 463 | -0.253 | 0.71 | 46.2% | -3.51 |
| 0.45 | 463 | -0.253 | 0.71 | 46.2% | -3.51 |
| 0.50 | 75 | -0.029 | 0.96 | 52.0% | -0.16 |

Top-5 feature importance (gain):
- `signal_density`: 43.5
- `minute_of_day`: 19.9
- `queue_ahead_log1p`: 14.4
- `dow`: 13.3
- `hour`: 10.8

## HC #444 R3 verdict
❌ NO threshold yields positive mean_tk + PF ≥ 1.10 + n ≥ 50 on the time-ordered OOS test set.

Interpretation: pre-trade-known features (pred_strength, TOD, DOW, queue, signal density, side) do not separate profitable from unprofitable fills well enough to rescue the canonical loss. Stronger features required (book imbalance at signal, recent realized vol, multi-model agreement).