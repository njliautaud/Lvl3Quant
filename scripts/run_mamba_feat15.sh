#!/bin/bash
# Mamba v5 feat15 + normalization experiment
# Same architecture as v4 (d=192, 6 layers, w=1000), but with 15 engineered features
# Using PID-file locking to prevent double-launch

set -e
cd /home/nick/Lvl3Quant

source scripts/training_lock.sh
acquire_lock "mamba_training" || exit 1

export MAMBA_FEATURE_SET=feat15
export DISABLE_MLFLOW=1
export WF_WINDOW_DAYS=60
export EVENT_BATCH_SIZE=64
export EVENT_WINDOW_SIZE=1000
export EVENT_STRIDE=500
export MAMBA_D_MODEL=192
export MAMBA_D_STATE=96
export MAMBA_N_LAYERS=6
export MAMBA_EPOCHS=5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== MAMBA v5 feat15 LAUNCH ==="
echo "Feature set: feat15 (15 features)"
echo "Architecture: d=192, state=96, layers=6"
echo "WF: sliding 60d, 1d OOT, 10 folds"
echo "Batch: 64, Window: 1000, Stride: 500"
echo "Time: $(date)"

/home/nick/miniconda3/envs/py311-train/bin/python -u \
    alpha_discovery/deep_models/train_event_mamba_cuda.py \
    --output-dir /home/nick/Lvl3Quant/output/mamba_v5_sliding60d_feat15 \
    --n-folds 10

echo "=== MAMBA v5 feat15 COMPLETE ==="
echo "Time: $(date)"
