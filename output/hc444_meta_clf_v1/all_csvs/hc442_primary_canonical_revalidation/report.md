# HC #444 Meta-Classifier v1 Report

Fills source: `hc442_primary_canonical_revalidation_fifo_fills.csv`  (1613 fills × 10 dates)

## Time-ordered 70/30 (OOS holdout)
- n_train = 1490, n_test = 123
- base positive-class rate (test) = 0.171
- AUC = 0.5394
- best_iter = 12

Threshold sweep (subset of test fills above prob threshold):

| thr | n | mean_tk | PF | WR | sharpe |
|---|---:|---:|---:|---:|---:|

Top-5 feature importance (gain):
- `signal_density`: 195.6
- `minute_of_day`: 191.6
- `queue_ahead_log1p`: 158.5
- `hour`: 27.7
- `pred_strength_z_within_day`: 26.2

## Random 80/20 (learnability)
- SKIPPED (n_train=1568, n_test=45)

## HC #444 R3 verdict
❌ NO threshold yields positive mean_tk + PF ≥ 1.10 + n ≥ 50 on the time-ordered OOS test set.

Interpretation: pre-trade-known features (pred_strength, TOD, DOW, queue, signal density, side) do not separate profitable from unprofitable fills well enough to rescue the canonical loss. Stronger features required (book imbalance at signal, recent realized vol, multi-model agreement).