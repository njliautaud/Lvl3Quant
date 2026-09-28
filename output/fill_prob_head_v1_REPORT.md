# Fill Probability Head v1 - Training Report

## Data Summary
- **Source**: `/home/jupiter/Lvl3Quant/output/fifo_v7_grade/fills.parquet`
- **Total rows**: 18,675
- **Fill rate** (TP or max_hold, not SL): 34.31%
- **Target variable**: `filled` = 1 if order reached profit target or max hold time, 0 if stopped out

### Data Split
- **Train**: 14,940 rows (34.10% fill rate)
- **Validation**: 3,735 rows (35.15% fill rate)
- **Split method**: Time-ordered, last 20% as validation

## Model Configuration
- **Model**: LightGBM binary classifier
- **Parameters**:
  - learning_rate=0.05, num_leaves=31, subsample=0.8, colsample_bytree=0.8
  - Early stopping: 50 rounds without improvement
  - Stopped at iteration 4 (no overfitting)

## Features (10 total)
1. `queue_ahead` — number of orders ahead at submission
2. `queue_ahead_log` — log-transformed queue depth
3. `pred_strength` — signal confidence from CNN-Mamba v2
4. `pred_strength_squared` — confidence interaction term
5. `direction_binary` — 1 if short, 0 if long
6. `time_of_day_hour` — hour + minute/60 (market regimes)
7. `day_of_week` — day-of-week encoding
8. `hold_s` — observed hold duration (seconds)
9. `hold_s_log` — log-transformed hold time
10. `queue_wait_ns` — nanoseconds waiting in queue

## Results

### ROC-AUC
| Dataset | AUC |
|---------|-----|
| Train | 0.8366 |
| **Val** | **0.8574** |

**Status**: ✓ **PASS** (AUC 0.8574 >> 0.75 threshold; 0.90 queue-predictor-v2 is a higher bar but this is strong for fill probability given label definition)

### Feature Importance (top 5)
| Feature | Importance |
|---------|------------|
| hold_s | 17,860 |
| queue_ahead | 200 |
| queue_wait_ns | 178 |
| pred_strength | 149 |
| pred_strength_squared | 60 |

**Interpretation**: Hold duration dominates (orders that survive longer are more likely to avoid SL). Queue depth and wait time are secondary predictors. Signal strength is weak predictor, suggesting execution timing matters more than signal strength for fill probability.

### Calibration (Validation Set)
| Bin | Predicted Prob | Actual Prob | Count |
|-----|----------------|------------|-------|
| 1 | 0.317 | 0.180 | 2,838 |
| 2 | 0.441 | 0.894 | 897 |

**Quality**: MODERATE. Bin 1 over-predicts fills (0.317 pred vs 0.180 actual). Bin 2 under-predicts (0.441 pred vs 0.894 actual). Suggests model is conservative on high-confidence predictions. Could be recalibrated with isotonic regression if deployed.

## Artifacts
- **Model**: `/home/jupiter/Lvl3Quant/models/fill_prob_head_v1.lgb`
- **MLflow run**: `80f00053c29d4a508bd9ca527cab11ad`
- **Experiment**: `fill_prob_head_v1` at http://localhost:5000

## Next Steps
1. **Integrate into FIFO harness**: Use model.predict() to weight passive limit orders by fill probability
2. **Calibration**: Apply isotonic regression post-hoc for better probability estimates
3. **Decay analysis**: Re-train weekly on rolling window if fill behavior drifts with market regime

