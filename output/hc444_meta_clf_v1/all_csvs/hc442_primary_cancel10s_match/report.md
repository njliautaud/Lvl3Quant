# HC #444 Meta-Classifier v1 Report

Fills source: `hc442_primary_cancel10s_match_fifo_fills.csv`  (2102 fills × 10 dates)

## Time-ordered 70/30 (OOS holdout)
- n_train = 1953, n_test = 149
- base positive-class rate (test) = 0.221
- AUC = 0.5935
- best_iter = 4

Threshold sweep (subset of test fills above prob threshold):

| thr | n | mean_tk | PF | WR | sharpe |
|---|---:|---:|---:|---:|---:|

Top-5 feature importance (gain):
- `signal_density`: 184.6
- `minute_of_day`: 93.9
- `queue_ahead_log1p`: 32.9
- `pred_strength_z_within_day`: 19.8
- `pred_pct_in_day`: 13.5

## Random 80/20 (learnability)
- SKIPPED (n_train=2042, n_test=60)

## HC #444 R3 verdict
❌ NO threshold yields positive mean_tk + PF ≥ 1.10 + n ≥ 50 on the time-ordered OOS test set.

Interpretation: pre-trade-known features (pred_strength, TOD, DOW, queue, signal density, side) do not separate profitable from unprofitable fills well enough to rescue the canonical loss. Stronger features required (book imbalance at signal, recent realized vol, multi-model agreement).