# IC Boosting Scripts - Meta Model Stacker & Execution Optimizer

Two research scripts for boosting alpha signal IC from ~0.04 (weak LGBM baseline) via ensemble stacking and execution timing optimization.

## Scripts

### 1. `meta_model_stacker.py`
**Purpose**: Stack multiple base model predictions + meta-features to boost IC

**Mechanism**:
- Takes OOT predictions from base models (LGBM, CNN, Mamba, etc.)
- Extracts meta-features from book data: spread, depth imbalance, volatility, momentum
- Trains lightweight MLP or LGBM meta-learner on expanded test window
- **Prevents leakage**: meta-model only sees fold-specific OOT base predictions

**Meta-features extracted**:
- Spread (ask_1 - bid_1)
- Bid-ask size imbalance (log scale)
- Rolling volatility (10-bar window)
- Mid-price change magnitude
- Cumulative delta (order flow)
- Depth imbalance

**Walk-forward approach**:
```
Fold 0: Train on [day1..dayN], test on [dayN+1..dayN+5]
Fold 1: Train on [day1..dayN+5], test on [dayN+6..dayN+10]
Fold 2: Train on [day1..dayN+10], test on [dayN+11..dayN+15]
...
```

**Usage**:
```bash
# LGBM meta-learner
python3 meta_model_stacker.py --meta-learner lgbm --base-models lgbm

# MLP meta-learner (supports GPU)
python3 meta_model_stacker.py --meta-learner mlp --base-models lgbm

# Multi-base ensemble (future extension)
# python3 meta_model_stacker.py --meta-learner mlp --base-models lgbm cnn mamba
```

**Outputs**:
- `/alpha_discovery/results/meta_stacker/fold{N:02d}_stacked_preds.npz`
  - `preds`: stacked predictions (OOT)
  - `labels`: realized returns
  - `base_preds`: base model predictions
  - `meta_features`: extracted meta-features
- MLflow experiment: `Meta_Stacker`
- Metrics: concat IC by confidence bucket (all, top50, top25, top10), directional accuracy

**Expected improvement**: 5-15% IC lift if base models are uncorrelated and meta-features are informative

---

### 2. `execution_optimizer.py`
**Purpose**: Learn optimal execution timing to improve realized PnL

**Problem**: Signal quality ≠ trade quality
- Strong prediction but wide spread → expensive execution
- Weak prediction but tight spread → profitable trade
- Need to learn WHEN to execute

**Mechanism**:
- Takes base alpha predictions (LGBM or stacked)
- Extracts execution state: signal magnitude/sign, spread, queue depth, vol, time
- Trains MLP to predict realized PnL given (signal, market_state)
- Execution scores indicate trade profitability

**Execution features**:
- |prediction| (signal magnitude)
- sign(prediction)
- Spread
- Queue depth (bid/ask size)
- Book imbalance (log ratio)
- Volatility (5-bar, 20-bar)
- Mid-price momentum
- Net order flow (cumulative delta)

**Approaches**:
1. **Supervised MLP** (`--approach supervised`):
   - Predict realized PnL from (signal, market_state)
   - High predicted PnL → execute now
   - Low predicted PnL → wait/skip
   - Metric: correlation between exec scores and realized PnL

2. **Policy Gradient RL** (planned):
   - Learn {BUY, SELL, WAIT} policy
   - Optimize for Sortino ratio
   - Current: baseline random strategy

**Usage**:
```bash
# Supervised MLP (current)
python3 execution_optimizer.py --approach supervised

# Policy gradient RL (to be implemented)
# python3 execution_optimizer.py --approach policy_gradient
```

**Outputs**:
- `/alpha_discovery/results/execution_optimizer/fold{N:02d}_exec_scores.npz`
  - `exec_scores`: predicted PnL per trade
  - `base_preds`: base model predictions
  - `labels`: realized returns
  - `features`: execution features
- MLflow experiment: `Execution_Optimizer`
- Metrics: overall IC, directional accuracy, Sortino ratio, execution correlation, PnL spread (high-exec vs low-exec)

**Expected improvement**: 20-50% Sortino improvement if execution state is predictive

---

## Data Flow

```
Book features (30-dim)
├─ LGBM train (expanding window)
├─ Extract meta-features (spread, imbalance, vol)
├─ Train meta-stacker MLP
└─ Output: stacked OOT predictions (improved IC)

Book features + base predictions
├─ Extract execution features
├─ Train execution MLP
└─ Output: execution scores (when to trade)

Combined:
1. Run meta_model_stacker.py → improved predictions
2. Run execution_optimizer.py on stacked predictions → execution timing
3. Result: higher IC + better execution = improved Sortino
```

---

## Implementation Details

### Walk-forward guarantees (ABSOLUTE):
- ✅ Expanding window (no sliding, no lookahead)
- ✅ OOT predictions (fold N trained on data before fold N)
- ✅ Meta-learner trained ONLY on fold N OOT base predictions
- ✅ No label leakage into features (labels used only for training supervision)

### Key design decisions:

**Meta Stacker**:
- LGBM base model (same as book features script)
- 30 book features + 6 meta-features = 36-dim input to meta-learner
- MLP: 36 → 64 → 1 (simple to prevent overfitting on small OOT sets)
- LGBM: 100 trees, depth=5 (simpler than base model)
- Why lightweight? OOT folds are often <100k samples, risk of overfitting

**Execution Optimizer**:
- Base model: LGBM (same config as meta_stacker)
- 11-12 execution features (signal + market state)
- MLP: {feat_dim} → 128 → 64 → 1 (predict PnL)
- Train on full fold (not OOT) because execution is forward-looking
- Metrics: correlation between exec_scores and realized PnL

### MLflow logging:
- Per-fold: IC by confidence, directional accuracy, n_train, n_test
- Summary: average IC/DA/Sortino across all folds
- Artifacts: fold prediction .npz files, results JSON

---

## Running Locally

```bash
cd /home/jupiter/Lvl3Quant

# Start MLflow (if not running)
# mlflow server --host 0.0.0.0 --port 5000 &

# Run meta stacker
python3 alpha_discovery/meta_model_stacker.py --meta-learner lgbm

# Run execution optimizer
python3 alpha_discovery/execution_optimizer.py --approach supervised

# View results
# MLflow UI: http://localhost:5000
# Files: ls alpha_discovery/results/meta_stacker/
#        ls alpha_discovery/results/execution_optimizer/
```

---

## Future Extensions

1. **Multi-base ensemble stacking**:
   - Add CNN, Mamba, Hawkes predictions to base_models
   - Meta-learner: 1 LGBM + 1 CNN + 1 Mamba + 6 meta = 9-dim input

2. **Policy gradient RL for execution**:
   - Replace MLP with A2C or PPO
   - Action space: {BUY_NOW, WAIT_100ms, WAIT_500ms, SKIP}
   - Reward: realized_pnl - execution_cost

3. **Online calibration**:
   - Update meta/execution weights as new folds complete
   - Drift detection if IC degrades

4. **Fill simulator integration**:
   - Predicted exec score → simulated fill price
   - Measure actual PnL improvement vs backtest

---

## Debugging

**Script hangs loading data**:
```bash
# Check book features exist
ls -lh /home/jupiter/Lvl3Quant/data/processed/mbo_book_features/ | head

# Check one file
python3 -c "import numpy as np; d = np.load('/path/to/file.npz'); print(list(d.keys()))"
```

**MLflow not logging**:
```bash
# Check MLflow server
curl http://localhost:5000/health

# Check tracking URI in script (should be http://localhost:5000)
```

**Out of memory**:
- Reduce TEST_DAYS (currently 5)
- Reduce fold window for testing
- Use LGBM meta-learner instead of MLP

---

## References

- Base model: `train_lgbm_book_features.py` (IC ~0.04, 30 book features)
- Book feature schema: `data/processed/mbo_book_features/*.npz`
- Event data: `data/processed/mbo_events_feat/*.npz` (15-dim events)
- Prior work: EventCNN1D (IC 0.132), Mamba SSM (training)
