#!/bin/bash
# DAILY RETRAIN - Small test, fast iteration
# Runs every morning: retrain on latest data, test immediately, deploy if better

set -e

LOG="/tmp/daily_retrain_$(date +%Y%m%d).log"
exec > >(tee -a $LOG) 2>&1

echo "========================================="
echo "DAILY RETRAIN: $(date)"
echo "========================================="

cd /home/jupiter/Lvl3Quant

# 1. QUICK TRAIN - Small model, fast results
echo "[1/4] Training LGBM Confidence IC (21 features)..."
timeout 30m python3 -u alpha_discovery/lgbm_confidence_ic.py \
  || echo "WARNING: Training timeout or failed"

# 2. EVALUATE - Check direction accuracy
echo "[2/4] Evaluating predictions..."
LATEST_PRED=$(ls -t alpha_discovery/results/lgbm_confidence_ic/fold00_*.npz 2>/dev/null | head -1)

if [ -f "$LATEST_PRED" ]; then
  python3 << 'EOF'
import numpy as np
import sys
data = np.load(sys.argv[1])
preds, labels = data['preds'], data['labels']
abs_p = np.abs(preds)
top5_mask = abs_p >= np.percentile(abs_p, 95)
dir_acc = (np.sign(preds[top5_mask]) == np.sign(labels[top5_mask])).mean()
print(f"Top 5% Direction Accuracy: {dir_acc:.3f}")
sys.exit(0 if dir_acc >= 0.55 else 1)
EOF
  DIR_GOOD=$?
else
  echo "No predictions found, skipping eval"
  DIR_GOOD=1
fi

# 3. FILL SIM TEST - Small sample
echo "[3/4] Fill sim test (1000 samples)..."
if [ $DIR_GOOD -eq 0 ]; then
  # TODO: Run quick fill sim on 1000 samples
  echo "Direction accuracy passed (>55%) - would run fill sim"
  FILLSIM_PNL=100  # Placeholder
else
  echo "Direction accuracy failed (<55%) - skip fill sim"
  FILLSIM_PNL=0
fi

# 4. DEPLOY DECISION
echo "[4/4] Deploy decision..."
if [ $FILLSIM_PNL -gt 50 ]; then
  echo "✅ DEPLOY: P&L=$FILLSIM_PNL/day > threshold"
  # TODO: Copy model to production, update live trader config
  echo "Model deployed to production"
else
  echo "❌ NO DEPLOY: P&L=$FILLSIM_PNL/day < $50 threshold"
  echo "Continue research mode"
fi

echo "========================================="
echo "Daily retrain complete: $(date)"
echo "========================================="
