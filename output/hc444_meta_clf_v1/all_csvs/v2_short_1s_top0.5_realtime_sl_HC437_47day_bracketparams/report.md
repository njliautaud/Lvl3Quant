# HC #444 Meta-Classifier v1 Report

Fills source: `v2_short_1s_top0.5_realtime_sl_HC437_47day_bracketparams_fifo_fills.csv`  (3455 fills × 32 dates)

## Time-ordered 70/30 (OOS holdout)
- n_train = 2513, n_test = 942
- base positive-class rate (test) = 0.461
- AUC = 0.5370
- best_iter = 10

Threshold sweep (subset of test fills above prob threshold):

| thr | n | mean_tk | PF | WR | sharpe |
|---|---:|---:|---:|---:|---:|
| 0.30 | 942 | -0.243 | 0.52 | 46.1% | -9.81 |
| 0.35 | 942 | -0.243 | 0.52 | 46.1% | -9.81 |
| 0.40 | 846 | -0.242 | 0.52 | 46.1% | -9.26 |
| 0.45 | 366 | -0.215 | 0.56 | 47.8% | -5.40 |

Top-5 feature importance (gain):
- `pred_strength`: 277.2
- `signal_density`: 148.0
- `minute_of_day`: 142.2
- `queue_ahead_log1p`: 136.2
- `pred_pct_in_day`: 129.4

## Random 80/20 (learnability)
- n_train = 2772, n_test = 683
- base positive-class rate (test) = 0.458
- AUC = 0.5643
- best_iter = 57

Threshold sweep (subset of test fills above prob threshold):

| thr | n | mean_tk | PF | WR | sharpe |
|---|---:|---:|---:|---:|---:|
| 0.30 | 666 | -0.245 | 0.52 | 45.9% | -8.31 |
| 0.35 | 606 | -0.221 | 0.55 | 47.5% | -7.13 |
| 0.40 | 496 | -0.189 | 0.60 | 49.6% | -5.52 |
| 0.45 | 322 | -0.174 | 0.63 | 50.6% | -4.10 |
| 0.50 | 138 | -0.226 | 0.55 | 47.1% | -3.48 |
| 0.55 | 42 | -0.146 | 0.68 | 52.4% | -1.23 |

Top-5 feature importance (gain):
- `pred_strength`: 790.5
- `queue_ahead_log1p`: 701.6
- `signal_density`: 592.8
- `pred_strength_z_within_day`: 569.4
- `minute_of_day`: 488.3

## HC #444 R3 verdict
❌ NO threshold yields positive mean_tk + PF ≥ 1.10 + n ≥ 50 on the time-ordered OOS test set.

Interpretation: pre-trade-known features (pred_strength, TOD, DOW, queue, signal density, side) do not separate profitable from unprofitable fills well enough to rescue the canonical loss. Stronger features required (book imbalance at signal, recent realized vol, multi-model agreement).