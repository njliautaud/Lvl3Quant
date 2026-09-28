# Meta-Classifier v1 Report

**Verdict: REJECT**

- IS days: 20 (20260223 → 20260319)
- OOT days: 12 (20260401 → 20260414)
- Feature set: 11 prediction-stream + 6 raw-market = 20 features (vs morning's snapshot-only)

## Best cell across all (side × horizon × threshold)
- **short @ 10s, threshold 0.50**
- Trades over OOT: 101
- Net ticks/trade: 4.010  (gate > 0.1)
- Annualized Sharpe: 5.35  (gate > 0.3)
- Win rate: 0.792
- Profitable days: 6/12  (gate ≥ 8)
- Regime imbalance: nan  (gate ≤ 0.5)
- Day concentration: 0.591  (gate ≤ 0.7)
- Gates passed: 3/5

## Top-3 features by average gain across the 8 models
1. pred_log_ret_1s
2. mean_queue_imb_W20
3. signed_trade_flow_W20

## Notes
- Cost model: passive-limit (0.376 ticks commission baked into labels).
- Walk-forward: 1 contiguous IS block, 1 contiguous OOT block, no shuffling across days.
- Regime proxy: sign of day's y_long_30s_net mean (green/red/flat, threshold ±0.1 t).
- Gates: HC #428 (net > 0.1, Sharpe > 0.3, pdays ≥ 65%, regime_imb ≤ 0.5, dayconc ≤ 0.7).
- Inner-val: last 20% of IS days for early-stopping.
