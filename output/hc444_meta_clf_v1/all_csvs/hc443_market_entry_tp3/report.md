# HC #444 Meta-Classifier v1 Report

Fills source: `hc443_market_entry_tp3_fifo_fills.csv`  (4299 fills × 36 dates)

## Time-ordered 70/30 (OOS holdout)
- n_train = 3114, n_test = 1185
- base positive-class rate (test) = 0.125
- AUC = 0.6487
- best_iter = 52

Threshold sweep (subset of test fills above prob threshold):

| thr | n | mean_tk | PF | WR | sharpe |
|---|---:|---:|---:|---:|---:|

Top-5 feature importance (gain):
- `pred_strength`: 1577.7
- `signal_density`: 697.7
- `pred_strength_z_within_day`: 609.5
- `minute_of_day`: 598.4
- `pred_pct_in_day`: 432.6

## Random 80/20 (learnability)
- n_train = 3133, n_test = 1166
- base positive-class rate (test) = 0.046
- AUC = 0.7814
- best_iter = 74

Threshold sweep (subset of test fills above prob threshold):

| thr | n | mean_tk | PF | WR | sharpe |
|---|---:|---:|---:|---:|---:|

Top-5 feature importance (gain):
- `pred_strength`: 1631.8
- `minute_of_day`: 1441.3
- `signal_density`: 1011.8
- `pred_strength_z_within_day`: 687.2
- `pred_pct_in_day`: 565.6

## HC #444 R3 verdict
❌ NO threshold yields positive mean_tk + PF ≥ 1.10 + n ≥ 50 on the time-ordered OOS test set.

Interpretation: pre-trade-known features (pred_strength, TOD, DOW, queue, signal density, side) do not separate profitable from unprofitable fills well enough to rescue the canonical loss. Stronger features required (book imbalance at signal, recent realized vol, multi-model agreement).