#!/bin/bash
# Deploy v4 multi-head training to Neptune
# Run this when: (1) pressure targets are built, (2) Neptune is released from gaming
set -e

NEPTUNE="nick@neptune"
REMOTE_ROOT="/home/nick/Lvl3Quant"
LOCAL_ROOT="/home/jupiter/Lvl3Quant"

echo "=== V4 MULTI-HEAD DEPLOYMENT TO NEPTUNE ==="

# 1. Sync smooth pressure targets to Neptune
echo "[1/4] Syncing smooth pressure targets..."
rsync -avz --progress \
  "$LOCAL_ROOT/data/processed/smooth_pressure_targets/" \
  "$NEPTUNE:$REMOTE_ROOT/data/processed/smooth_pressure_targets/"

# 2. Sync training script
echo "[2/4] Syncing v4 training script..."
scp "$LOCAL_ROOT/scripts/train_v4_multihead.py" \
    "$NEPTUNE:$REMOTE_ROOT/scripts/train_v4_multihead.py"

# 3. Create output directory
echo "[3/4] Creating output directory..."
ssh "$NEPTUNE" "mkdir -p $REMOTE_ROOT/output/v4_multihead_pressure"

# 4. Launch training
echo "[4/4] Launching v4 multi-head training..."
ssh "$NEPTUNE" "cd $REMOTE_ROOT && \
  source /home/nick/miniconda3/etc/profile.d/conda.sh && \
  conda activate py311-train && \
  nohup python scripts/train_v4_multihead.py \
    --data-dir data/processed/mbo_events_smart_v3 \
    --pressure-dir data/processed/smooth_pressure_targets \
    --output-dir output/v4_multihead_pressure \
    --device cuda \
    --epochs 15 \
    --batch-size 256 \
    --seq-len 100 \
    --lr 1e-4 \
    --wf-window 60 \
    --mlflow-uri http://localhost:5000 \
    > logs/v4_multihead_training.log 2>&1 &
  echo 'PID='\$!"

echo "=== DEPLOYMENT COMPLETE ==="
echo "Monitor: ssh $NEPTUNE 'tail -f $REMOTE_ROOT/logs/v4_multihead_training.log'"
echo "GPU check: ssh $NEPTUNE 'nvidia-smi'"
