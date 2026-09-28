# LGBM Execution Filter — Deployment Specification
# Created: 2026-05-08 00:50 ET
# Status: Research complete, deployment design phase

## Summary

LightGBM execution filter that gates signal-model entries. Trained on CNN-Mamba v2
predictions + embeddings, validated on 1.18M OOT samples across 47 walk-forward folds.

## Best Configuration

| Setting | Value | Rationale |
|---------|-------|-----------|
| Profit threshold | 1.0 ticks | Best top1% WR (57.4%) — cleaner boundary |
| Training folds | 5 | 10 folds help top5% but not top1%; 5 folds = more OOT validation |
| Embeddings | YES | +2-3% WR contribution, mandatory |
| Feature count | 117 | 21 signal + 96 embedding features |

## Performance (Walk-Forward OOT)

| Filter Tier | N Trades | WR | Edge/Trade | PF | Sortino |
|-------------|----------|-----|-----------|-----|---------|
| Top 1% | 11,834 | 57.4% | +0.381t | 1.31 | 0.085 |
| Top 5% | 59,606 | 53.9% | +0.195t | 1.10 | 0.029 |
| Top 10% | 118,349 | 53.4% | +0.152t | 1.08 | 0.022 |

## Integration Architecture

### Current Flow (paper_trading_mamba_v2_patched.py):
```
MBO Events → CNN-Mamba v2 → prediction (3 horizons) → tier check (Top5%) → ORDER
```

### Proposed Flow (with LGBM gate):
```
MBO Events → CNN-Mamba v2 → prediction (3 horizons) + embeddings (96-dim)
          → tier check (Top5%)
          → LGBM filter (signal features + embeddings → probability)
          → probability > threshold → ORDER
```

### Implementation Steps

1. **Save LGBM model on Razer**: Copy best fold model (`.txt` format) to Razer
2. **Add LGBM inference to paper trader**:
   - `import lightgbm as lgb`
   - `model = lgb.Booster(model_file="path/to/lgbm_model.txt")`
   - After CNN-Mamba produces prediction + embeddings:
     - Build 117-dim feature vector (same as training)
     - `prob = model.predict(features.reshape(1, -1))[0]`
     - Only trade if `prob > lgbm_threshold`
3. **Threshold selection**: 
   - Conservative: Top 1% (~prob > 0.6-0.7) — fewer trades, higher quality
   - Moderate: Top 5% (~prob > 0.5-0.55) — more trades, still positive edge
4. **Walk-forward model updates**: Retrain LGBM weekly with latest data

### Key Considerations

- LGBM inference is ~0.1ms per prediction (negligible latency)
- Model file is ~100KB (tiny)
- Requires `lightgbm` pip package on Razer (need to install)
- Embeddings must be extracted from CNN-Mamba intermediate layer (already exposed)
- Feature engineering must exactly match training (use shared function)

### Risk: Overfitting

The LGBM filter was trained on the SAME CNN-Mamba predictions used in the paper trader.
This means the filter and the signal model share information. The walk-forward validation
mitigates this, but we should monitor:
- Does LGBM filter improve paper trading WR over raw tier filtering?
- Does the edge persist on new dates not in the training set?
- Are we just concentrating trades in favorable time periods (time-of-day bias)?

### Monitoring

After deployment, track these metrics daily:
- Trades filtered out by LGBM (should be 80-95% of tier-passing signals)
- WR of LGBM-approved trades vs LGBM-rejected trades
- Edge decay over time (retrain trigger: if WR drops below 53%)
