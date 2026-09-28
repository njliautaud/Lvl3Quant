# HC #444 Meta-Classifier v1 Report

Fills source: `hc443_upside_sl3_tp8_h10_fifo_fills.csv`  (2431 fills × 32 dates)

## Time-ordered 70/30 (OOS holdout)
- n_train = 1901, n_test = 530
- base positive-class rate (test) = 0.389
- AUC = 0.5174
- best_iter = 3

Threshold sweep (subset of test fills above prob threshold):

| thr | n | mean_tk | PF | WR | sharpe |
|---|---:|---:|---:|---:|---:|
| 0.30 | 530 | -0.786 | 0.55 | 38.9% | -5.85 |

Top-5 feature importance (gain):
- `minute_of_day`: 78.4
- `pred_strength`: 38.2
- `queue_ahead_log1p`: 33.2
- `signal_density`: 27.6
- `pred_pct_in_day`: 26.8

## Random 80/20 (learnability)
- n_train = 1949, n_test = 482
- base positive-class rate (test) = 0.342
- AUC = 0.5511
- best_iter = 22

Threshold sweep (subset of test fills above prob threshold):

| thr | n | mean_tk | PF | WR | sharpe |
|---|---:|---:|---:|---:|---:|
| 0.30 | 384 | -0.498 | 0.76 | 35.2% | -2.43 |
| 0.35 | 210 | -0.357 | 0.82 | 35.7% | -1.27 |
| 0.40 | 50 | +0.364 | 1.24 | 50.0% | +0.66 |

Top-5 feature importance (gain):
- `minute_of_day`: 326.9
- `pred_strength`: 313.2
- `queue_ahead_log1p`: 245.6
- `signal_density`: 226.6
- `pred_pct_in_day`: 211.4

## HC #444 R3 verdict
❌ NO threshold yields positive mean_tk + PF ≥ 1.10 + n ≥ 50 on the time-ordered OOS test set.

Interpretation: pre-trade-known features (pred_strength, TOD, DOW, queue, signal density, side) do not separate profitable from unprofitable fills well enough to rescue the canonical loss. Stronger features required (book imbalance at signal, recent realized vol, multi-model agreement).