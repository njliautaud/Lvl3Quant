#!/bin/bash
# Wait for current Mamba training to finish, then launch CUDA Mamba
set -e

CURRENT_PID=1473040
SCRIPT="/home/nick/Lvl3Quant/alpha_discovery/deep_models/train_event_mamba_cuda.py"
OUTPUT_DIR="/home/nick/Lvl3Quant/alpha_discovery/deep_models/results/event_mamba_cuda"
DATA_DIR="/tmp/mbo_clean_neptune"
PYTHON="/home/nick/miniconda3/envs/py311-train/bin/python"

echo "[$(date)] Waiting for PID $CURRENT_PID to finish..."
while kill -0 $CURRENT_PID 2>/dev/null; do
    sleep 30
done
echo "[$(date)] PID $CURRENT_PID finished. Launching CUDA Mamba in 10s..."
sleep 10

mkdir -p "$OUTPUT_DIR"

echo "[$(date)] Launching CUDA Mamba training..."
cd /home/nick/Lvl3Quant

PYTHONUNBUFFERED=1 \
MLFLOW_TRACKING_URI=http://jupiter:5000 \
MAMBA_D_MODEL=128 \
MAMBA_D_STATE=64 \
MAMBA_N_LAYERS=4 \
MAMBA_D_CONV=4 \
EVENT_WINDOW_SIZE=500 \
EVENT_BATCH_SIZE=64 \
EVENT_STRIDE=500 \
EVENT_EPOCHS=3 \
EVENT_N_FOLDS=5 \
EVENT_LR=3e-4 \
nohup $PYTHON -u "$SCRIPT" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --n-folds 5 \
    > /tmp/mamba_cuda_run.log 2>&1 &

NEW_PID=$!
echo "[$(date)] CUDA Mamba launched! PID=$NEW_PID"
echo "[$(date)] Log: /tmp/mamba_cuda_run.log"
echo "[$(date)] Output: $OUTPUT_DIR"
echo ""
echo "Config: d_model=128, d_state=64, n_layers=4, d_conv=4"
echo "        window=500, batch=64, stride=500, epochs=3, folds=5"
