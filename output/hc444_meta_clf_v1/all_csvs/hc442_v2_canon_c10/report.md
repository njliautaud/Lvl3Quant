# HC #444 Meta-Classifier v1 Report

Fills source: `hc442_v2_canon_c10_fifo_fills.csv`  (3455 fills × 32 dates)

## Time-ordered 70/30 (OOS holdout)
- n_train = 2513, n_test = 942
- base positive-class rate (test) = 0.306
- AUC = 0.5508
- best_iter = 24

Threshold sweep (subset of test fills above prob threshold):

| thr | n | mean_tk | PF | WR | sharpe |
|---|---:|---:|---:|---:|---:|
| 0.30 | 195 | -0.261 | 0.50 | 38.5% | -3.98 |
| 0.35 | 44 | -0.331 | 0.32 | 43.2% | -3.22 |

Top-5 feature importance (gain):
- `pred_strength`: 425.8
- `minute_of_day`: 387.4
- `queue_ahead_log1p`: 341.7
- `signal_density`: 340.6
- `pred_pct_in_day`: 185.1

## Random 80/20 (learnability)
- n_train = 2772, n_test = 683
- base positive-class rate (test) = 0.253
- AUC = 0.5982
- best_iter = 75

Threshold sweep (subset of test fills above prob threshold):

| thr | n | mean_tk | PF | WR | sharpe |
|---|---:|---:|---:|---:|---:|
| 0.30 | 136 | -0.122 | 0.79 | 33.1% | -1.17 |
| 0.35 | 52 | +0.037 | 1.08 | 48.1% | +0.22 |

Top-5 feature importance (gain):
- `pred_strength`: 1054.7
- `minute_of_day`: 847.1
- `queue_ahead_log1p`: 842.8
- `signal_density`: 789.1
- `pred_strength_z_within_day`: 634.3

## HC #444 R3 verdict
❌ NO threshold yields positive mean_tk + PF ≥ 1.10 + n ≥ 50 on the time-ordered OOS test set.

Interpretation: pre-trade-known features (pred_strength, TOD, DOW, queue, signal density, side) do not separate profitable from unprofitable fills well enough to rescue the canonical loss. Stronger features required (book imbalance at signal, recent realized vol, multi-model agreement).