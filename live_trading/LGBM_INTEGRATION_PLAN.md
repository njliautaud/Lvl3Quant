# LGBM Fold 35 Integration Plan
## Fast-Track Deployment for Monday April 21, 2026

**Model**: LGBM Fold 35 (trained April 16, 2026)  
**Location**: `/home/jupiter/Lvl3Quant/models/lgbm_60_5_fold35/`  
**Target**: Live trading Monday 9:30 AM ET

---

## Model Performance Summary

### Base Performance (All Predictions)
- **IC (1s)**: 0.122 (12.2%)
- **IC (5s)**: 0.055 (5.5%)
- **IC (10s)**: 0.041 (4.1%)
- **IC (30s)**: 0.018 (1.8%)

### Gated Performance (Top Conviction)
| Tier | 1s IC | Dir Acc | Population |
|------|-------|---------|------------|
| Top 10% | **21.4%** | 62.6% | 10% of predictions |
| Top 25% | **19.1%** | 60.3% | 25% of predictions |
| Top 50% | **16.3%** | 58.1% | 50% of predictions |

**Deployment Strategy**: Trade only **Top 10-25%** strongest predictions

---

## Technical Integration Tasks

### Task 1: Model Loading & Inference (2 hours)

**Files Required**:
```
/home/jupiter/Lvl3Quant/models/lgbm_60_5_fold35/
├── labels_1s_lgbm.pkl       (762 KB)
├── labels_5s_lgbm.pkl       (823 KB)
├── labels_10s_lgbm.pkl      (1.1 MB)
├── labels_30s_lgbm.pkl      (931 KB)
├── fold_meta.json           (5.6 KB)
└── labels_1s_calibration.json (773 B)
```

**Integration Code** (create `/home/jupiter/Lvl3Quant/live_trading/models/lgbm_inference.py`):

```python
"""
LGBM Inference Wrapper for Live Trading
"""
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

class LGBMInferenceEngine:
    """Loads LGBM fold 35 and provides real-time predictions"""
    
    def __init__(self, model_dir: str = "/home/jupiter/Lvl3Quant/models/lgbm_60_5_fold35"):
        self.model_dir = Path(model_dir)
        self.models = {}
        self.calibration = {}
        
        # Load all horizon models
        for horizon in ['1s', '5s', '10s', '30s']:
            model_path = self.model_dir / f"labels_{horizon}_lgbm.pkl"
            with open(model_path, 'rb') as f:
                self.models[horizon] = pickle.load(f)
        
        # Load calibration thresholds
        calib_path = self.model_dir / "labels_1s_calibration.json"
        with open(calib_path) as f:
            import json
            self.calibration = json.load(f)
        
        print(f"✓ Loaded LGBM models for horizons: {list(self.models.keys())}")
        print(f"✓ Calibration thresholds: top10={self.calibration['tier_thresholds']['top10']:.3f}")
    
    def predict(self, features: np.ndarray, horizon: str = '1s') -> Tuple[float, float]:
        """
        Generate prediction for given features
        
        Args:
            features: Feature vector (must match training features)
            horizon: Prediction horizon ('1s', '5s', '10s', '30s')
        
        Returns:
            (prediction_value, abs_prediction) tuple
        """
        model = self.models[horizon]
        pred = model.predict(features.reshape(1, -1))[0]
        abs_pred = abs(pred)
        return pred, abs_pred
    
    def get_conviction_tier(self, abs_prediction: float) -> str:
        """
        Map absolute prediction value to conviction tier
        
        Returns: 'top1', 'top5', 'top10', 'top25', 'top50', or 'below50'
        """
        thresholds = self.calibration['tier_thresholds']
        
        if abs_prediction >= thresholds['top1']:
            return 'top1'
        elif abs_prediction >= thresholds['top5']:
            return 'top5'
        elif abs_prediction >= thresholds['top10']:
            return 'top10'
        elif abs_prediction >= thresholds['top25']:
            return 'top25'
        elif abs_prediction >= thresholds['top50']:
            return 'top50'
        else:
            return 'below50'
    
    def predict_with_tier(self, features: np.ndarray, horizon: str = '1s') -> Dict:
        """
        Generate prediction with conviction tier
        
        Returns:
            {
                'prediction': float,
                'abs_prediction': float,
                'tier': str,
                'expected_ic': float,  # From calibration
                'horizon': str
            }
        """
        pred, abs_pred = self.predict(features, horizon)
        tier = self.get_conviction_tier(abs_pred)
        
        # Look up expected IC for this tier (from fold_meta.json)
        expected_ic_map = {
            'top1': 0.214,   # From fold 35 results
            'top5': 0.214,
            'top10': 0.214,
            'top25': 0.191,
            'top50': 0.163,
            'below50': 0.122
        }
        
        return {
            'prediction': pred,
            'abs_prediction': abs_pred,
            'tier': tier,
            'expected_ic': expected_ic_map.get(tier, 0.0),
            'horizon': horizon,
            'timestamp': time.time()
        }
```

**Testing**:
```python
# Test script: test_lgbm_inference.py
engine = LGBMInferenceEngine()

# Generate mock features (replace with real feature extraction)
mock_features = np.random.randn(100)  # Adjust dimension to match training

result = engine.predict_with_tier(mock_features, horizon='1s')
print(f"Prediction: {result['prediction']:.4f}")
print(f"Tier: {result['tier']}")
print(f"Expected IC: {result['expected_ic']:.3f}")
```

---

### Task 2: Feature Extraction Integration (3 hours)

**Challenge**: LGBM was trained on specific features. Must match exactly.

**Action Required**: 
1. Identify feature list from LGBM training
2. Verify feature extraction pipeline produces same features
3. Test feature alignment with mock data

**Top Features for LGBM (from fold_meta.json)**:
1. `price` (IC = 0.056)
2. `price_sign_mom_200` (IC = 0.056)
3. `add_side_asym_100` (IC = 0.042)
4. `qty_price_mom_20` (IC = 0.041)
5. `qty_add_imbalance_100` (IC = 0.039)

**Feature Extraction Code** (integrate with existing pipeline):
```python
def extract_lgbm_features(mbo_buffer: List[MBOEvent]) -> np.ndarray:
    """
    Extract features that match LGBM training
    
    Must produce same feature vector as used in training
    """
    # TODO: Implement feature extraction
    # This needs to match whatever was used in LGBM training
    pass
```

**BLOCKER**: Need to locate LGBM training script to get exact feature list.

**Search locations**:
- `/home/jupiter/Lvl3Quant/alpha_discovery/experiments/`
- `/home/jupiter/Lvl3Quant/models/lgbm_60_5_fold35/` (check for feature metadata)

---

### Task 3: Inference Engine Integration (2 hours)

**Modify**: `/home/jupiter/Lvl3Quant/live_trading/inference_engine.py`

**Add LGBM to model registry**:
```python
from live_trading.models.lgbm_inference import LGBMInferenceEngine

# In InferenceEngine.__init__():
self.lgbm_engine = LGBMInferenceEngine()

# In InferenceEngine.process_event():
async def process_event(self, event: MBOEvent) -> List[Prediction]:
    # Extract features from event buffer
    features = self.extract_features()
    
    # Generate LGBM prediction
    lgbm_result = self.lgbm_engine.predict_with_tier(features, horizon='10s')
    
    # Create Prediction object
    pred = Prediction(
        model_name='lgbm_fold35',
        value=lgbm_result['prediction'],
        tier=lgbm_result['tier'],
        expected_ic=lgbm_result['expected_ic'],
        latency_ms=0.0,  # LGBM is very fast
        timestamp=event.timestamp
    )
    
    # Emit to callbacks
    await self._emit_prediction(pred)
    
    return [pred]
```

---

### Task 4: Trading Card Configuration (1 hour)

**Create**: Conservative LGBM trading card

**Card Config** (`configs/paper_config.json`):
```json
{
  "cards": [
    {
      "name": "lgbm_conservative",
      "model_name": "lgbm_fold35",
      "threshold": 0.0,
      "min_tier": "top10",
      "max_position_size": 1,
      "take_profit_ticks": 2.0,
      "stop_loss_ticks": 1.5,
      "max_hold_seconds": 300,
      "conviction_decay": false,
      "enabled": true,
      "description": "Ultra-conservative LGBM: top 10% only, 1 contract, tight stops"
    }
  ]
}
```

**Key Settings**:
- `min_tier: "top10"` → Only trade top 10% strongest predictions (IC = 21%)
- `max_position_size: 1` → 1 contract maximum
- `take_profit_ticks: 2.0` → Exit at +2 ticks ($25 profit)
- `stop_loss_ticks: 1.5` → Exit at -1.5 ticks ($18.75 loss)
- `max_hold_seconds: 300` → Max 5 minute hold time

**Phase 2 (Tuesday if Monday successful)**:
```json
{
  "name": "lgbm_moderate",
  "min_tier": "top25",
  "max_position_size": 2,
  "take_profit_ticks": 3.0,
  "stop_loss_ticks": 2.0
}
```

---

### Task 5: Leakage Audit (1 hour)

**Verification Checklist**:
- [ ] Training dates: Jan 5 - Mar 6, 2026 (before test)
- [ ] Test dates: Mar 8 - Mar 12, 2026 (after training)
- [ ] No overlap between train/test
- [ ] Feature extraction uses only past data
- [ ] No forward-looking labels
- [ ] Walk-forward training confirmed

**Audit Script** (create `audit_lgbm_leakage.py`):
```python
import json
from pathlib import Path

def audit_lgbm_fold35():
    """Verify no leakage in LGBM fold 35"""
    
    meta_path = Path("/home/jupiter/Lvl3Quant/models/lgbm_60_5_fold35/fold_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    
    print("=== LEAKAGE AUDIT: LGBM Fold 35 ===")
    print(f"Train period: {meta['train']}")
    print(f"Test period: {meta['test']}")
    
    # Parse dates
    train_end = "20260306"
    test_start = "20260308"
    
    if train_end < test_start:
        print("✓ PASS: No temporal leakage (train ends before test starts)")
    else:
        print("✗ FAIL: Temporal leakage detected")
    
    # Check for future features
    print("\n=== Top Features ===")
    for horizon, features in meta['raw_ic_top'].items():
        print(f"\n{horizon}:")
        for feat_name, ic in features:
            # Check if feature name suggests future data
            if any(word in feat_name.lower() for word in ['future', 'forward', 'next']):
                print(f"  ⚠️  {feat_name}: {ic:.4f} (potential future leakage)")
            else:
                print(f"  ✓ {feat_name}: {ic:.4f}")
    
    print("\n=== AUDIT COMPLETE ===")

if __name__ == "__main__":
    audit_lgbm_fold35()
```

**Expected Output**: ✓ PASS on all checks

---

### Task 6: Paper Trading Validation (24 hours)

**Sunday Schedule**:
- **08:00**: Start paper trading with LGBM
- **10:00**: First checkpoint (2 hours, ~10-20 predictions)
- **14:00**: Mid-day checkpoint (6 hours)
- **20:00**: Evening checkpoint (12 hours)
- **Monday 08:00**: Final validation (24 hours complete)

**Monitoring Dashboard**:
```python
# Real-time metrics to track
metrics = {
    'predictions_generated': 0,
    'predictions_top10': 0,
    'orders_submitted': 0,
    'fills_received': 0,
    'positions_opened': 0,
    'positions_closed': 0,
    'pnl_ticks': 0.0,
    'pnl_dollars': 0.0,
    'win_rate': 0.0,
    'avg_hold_time_seconds': 0.0,
    'prediction_latency_ms': [],
}
```

**GO/NO-GO Criteria** (evaluate at 20:00 Sunday):
- ✅ GO if:
  - Win rate > 50% (or insufficient samples)
  - No safety violations (no rejected orders, no circuit breakers)
  - Prediction latency < 50ms
  - PnL not significantly negative (< -$200)
  - System stable (no crashes, no errors)
  
- ❌ NO-GO if:
  - Win rate < 40% (with >20 trades)
  - Multiple safety violations
  - Prediction latency > 100ms
  - PnL < -$500
  - System instability

---

### Task 7: Monday Deployment (Go-Live)

**Pre-Market Checklist** (Monday 9:00-9:25 AM):
- [ ] Paper trading validation passed
- [ ] All safety mechanisms tested
- [ ] Kill switch verified working
- [ ] Rithmic connection live
- [ ] Position limits configured (1 contract max)
- [ ] Daily loss limit set ($500)
- [ ] Monitoring dashboard ready
- [ ] User available for first hour of trading

**9:30 AM Launch**:
1. Switch from paper trading to live mode
2. Enable LGBM conservative card
3. Monitor first prediction closely
4. Watch for any unexpected behavior

**First Hour** (9:30-10:30 AM):
- Close monitoring required
- User should be available
- Ready to activate kill switch if needed

**First Day** (Monday PM):
- Limit to 10 trades maximum
- Conservative thresholds
- Close all positions by 3:30 PM (no overnight)

---

## Risk Mitigation

### Fallback Plan
If paper trading fails Sunday evening:
1. Activate kill switch immediately
2. Assess failure mode
3. **Delay to April 28** (Option C)
4. User notified with detailed analysis

### Backup Model
Continue CNN-Jamba validation in parallel:
- Sunday: Run CNN-Jamba test on Neptune (if GPU freed)
- Next week: Full CNN-Jamba training
- Future: Upgrade from LGBM to CNN-Jamba when validated

### Safety Nets
- Kill switch always available
- Circuit breakers active (max 5 consecutive losses)
- Position limits enforced (1 contract max)
- Daily loss limit ($500)
- Real-time monitoring with Discord alerts

---

## Timeline Summary

**Saturday PM (12:00-18:00)**: 6 hours
- [ ] 12:00-14:00: Implement LGBM inference wrapper
- [ ] 14:00-16:00: Integrate with inference engine
- [ ] 16:00-17:00: Configure trading card
- [ ] 17:00-18:00: Run leakage audit

**Sunday (08:00-20:00)**: 12 hours
- [ ] 08:00: Start 24-hour paper trading
- [ ] Throughout day: Monitor performance
- [ ] 20:00: GO/NO-GO decision

**Monday 9:30 AM**: Live trading launch (if GO)

---

## Success Criteria

**Minimum for Deployment**:
- Leakage audit: PASS ✅
- Paper trading: Stable 24 hours, no crashes ✅
- Prediction latency: < 50ms ✅
- Safety mechanisms: All functional ✅
- Win rate: > 50% OR insufficient data ✅

**Ideal for Deployment**:
- Paper trading: Profitable or breakeven
- Multiple trades executed successfully
- No safety violations
- User confidence: HIGH

---

## Blockers & Dependencies

**CRITICAL BLOCKER**: Feature extraction
- Must match LGBM training exactly
- Need to locate training script or feature metadata
- Without this, cannot generate predictions

**Dependencies**:
1. User approval of Option D
2. Feature extraction pipeline identified
3. Rithmic connection confirmed live
4. Paper trading infrastructure tested

---

## Next Steps

**Awaiting User Decision**: Option D approval

**If Approved**: Begin Task 1 (LGBM inference wrapper) immediately

**If Not Approved**: Clarify concerns and adjust plan

---

*Plan created: April 19, 2026*  
*Target deployment: Monday April 21, 2026, 9:30 AM ET*
