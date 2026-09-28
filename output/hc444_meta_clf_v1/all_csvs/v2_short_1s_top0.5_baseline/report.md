# HC #444 Meta-Classifier v1 Report

Fills source: `v2_short_1s_top0.5_baseline_fifo_fills.csv`  (2676 fills × 34 dates)

## Time-ordered 70/30 (OOS holdout)
- n_train = 1518, n_test = 1158
- base positive-class rate (test) = 0.454
- AUC = 0.4847
- best_iter = 1

Threshold sweep (subset of test fills above prob threshold):

| thr | n | mean_tk | PF | WR | sharpe |
|---|---:|---:|---:|---:|---:|
| 0.30 | 1158 | -0.198 | 0.58 | 45.4% | -9.16 |
| 0.35 | 1158 | -0.198 | 0.58 | 45.4% | -9.16 |
| 0.40 | 1158 | -0.198 | 0.58 | 45.4% | -9.16 |
| 0.45 | 1158 | -0.198 | 0.58 | 45.4% | -9.16 |

Top-5 feature importance (gain):
- `signal_density`: 62.7
- `minute_of_day`: 8.6
- `dow`: 5.8
- `pred_pct_in_day`: 5.5
- `pred_strength_z_within_day`: 3.2

## Random 80/20 (learnability)
- n_train = 2070, n_test = 606
- base positive-class rate (test) = 0.469
- AUC = 0.5153
- best_iter = 20

Threshold sweep (subset of test fills above prob threshold):

| thr | n | mean_tk | PF | WR | sharpe |
|---|---:|---:|---:|---:|---:|
| 0.30 | 606 | -0.180 | 0.61 | 46.9% | -5.99 |
| 0.35 | 606 | -0.180 | 0.61 | 46.9% | -5.99 |
| 0.40 | 579 | -0.164 | 0.64 | 48.0% | -5.33 |
| 0.45 | 409 | -0.155 | 0.66 | 48.7% | -4.21 |
| 0.50 | 143 | -0.208 | 0.57 | 44.8% | -3.34 |

Top-5 feature importance (gain):
- `minute_of_day`: 315.8
- `pred_strength_z_within_day`: 242.1
- `pred_strength`: 228.0
- `queue_ahead_log1p`: 163.3
- `pred_pct_in_day`: 162.4

## HC #444 R3 verdict
❌ NO threshold yields positive mean_tk + PF ≥ 1.10 + n ≥ 50 on the time-ordered OOS test set.

Interpretation: pre-trade-known features (pred_strength, TOD, DOW, queue, signal density, side) do not separate profitable from unprofitable fills well enough to rescue the canonical loss. Stronger features required (book imbalance at signal, recent realized vol, multi-model agreement).