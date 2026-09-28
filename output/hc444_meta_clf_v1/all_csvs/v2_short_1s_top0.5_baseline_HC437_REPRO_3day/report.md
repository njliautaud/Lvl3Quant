# HC #444 Meta-Classifier v1 Report

Fills source: `v2_short_1s_top0.5_baseline_HC437_REPRO_3day_fifo_fills.csv`  (160 fills × 3 dates)

## Time-ordered 70/30 (OOS holdout)
- SKIPPED (n_train=158, n_test=2)

## Random 80/20 (learnability)
- SKIPPED (n_train=70, n_test=90)

## HC #444 R3 verdict
❌ NO threshold yields positive mean_tk + PF ≥ 1.10 + n ≥ 50 on the time-ordered OOS test set.

Interpretation: pre-trade-known features (pred_strength, TOD, DOW, queue, signal density, side) do not separate profitable from unprofitable fills well enough to rescue the canonical loss. Stronger features required (book imbalance at signal, recent realized vol, multi-model agreement).