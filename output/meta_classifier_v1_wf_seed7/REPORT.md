# Meta-Classifier v1 — Multi-Fold Walk-Forward Report

**Verdict: CONDITIONAL_ACCEPT**

Protocol: 4 folds, sliding window (HC #0). IS=16 days, OOT=4 days/fold, 16 OOT days total.
Total models trained: 32 (4 folds x 8 models).

## Best cell (pooled across all 4 folds)
- **long @ 1s, threshold 0.50**
- Pooled trades over 16 OOT days: 104934
- Net ticks/trade (pooled): 0.113  (gate > 0.1)
- Annualized Sharpe (pooled day-means): 13.26  (gate > 0.3)
- Win rate: 0.552
- Profitable days: 13/16  (gate >= 11)
- Regime imbalance (|Sg-Sr|/max): nan  (gate <= 0.5)
- Day concentration: 0.211  (gate <= 0.7)
- Per-fold consistency: 4/4  (gate >= 3)
- Gates passed: 5/6

## Did short_10s @ thr=0.50 (single-fold leader) survive?
- Pooled trades: 110053
- Net ticks/trade: 0.103
- Sharpe: 3.44
- Profitable days: 10/16
- Per-fold consistency: 2/4
- Gates passed: 3/6
- **SURVIVED MULTI-FOLD: NO**

## Top-3 features (avg gain across all folds and models)
1. pred_log_ret_1s
2. mean_queue_imb_W20
3. signed_trade_flow_W20

## Notes
- Labels: passive-limit net ticks (commission 0.376 baked in).
- Per-fold consistency gate (NEW): >= 3/4 folds with positive net AND >= 10 trades.
- Pdays gate raised to >= 11/16 (~69%) per multi-fold spec.
