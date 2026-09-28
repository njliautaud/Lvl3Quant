#!/bin/bash
# CNN-Mamba v2 FULL RETRAIN through April 2026
# Launch on Neptune AFTER Optuna exec MLP completes
# Architecture: d_model=128, d_state=64, dt_rank=16, n_layers=4 (script defaults + CUDA Mamba)
# Data: 248 days (Jul 2025 - Apr 2026), sliding 60-day window
# Expected runtime: ~24-36 hours for full WF on 3090

set -e

export EVENT_WINDOW_SIZE=3000
export EVENT_STRIDE=500
export EVENT_BATCH_SIZE=256
export EVENT_LR=3e-4
export EVENT_N_FOLDS=180
export WF_WINDOW_DAYS=60
export FEATURE_SET=smart_v3
export SKIP_NORMALIZE=1

# Fresh output directory — DO NOT share with old runs
export OUTPUT_DIR="/home/nick/Lvl3Quant/output/cnn_mamba_v2_retrain_apr2026"
mkdir -p "$OUTPUT_DIR"

cd /home/nick/Lvl3Quant

echo "=========================================="
echo "CNN-Mamba v2 RETRAIN — April 2026 Data"
echo "=========================================="
echo "Start time: $(date)"
echo "Output: $OUTPUT_DIR"
echo "Folds: $EVENT_N_FOLDS (sliding 60-day window)"
echo "Window: $EVENT_WINDOW_SIZE events, stride $EVENT_STRIDE"
echo "Batch: $EVENT_BATCH_SIZE"
echo ""

# Activate conda env
source /home/nick/miniconda3/etc/profile.d/conda.sh
conda activate py311-train

# Set MLflow tracking
export MLFLOW_TRACKING_URI="http://localhost:5000"
export MLFLOW_EXPERIMENT_NAME="cnn_mamba_v2_retrain_apr2026"

# Launch training
python -u alpha_discovery/deep_models/train_cnn_mamba_v2.py \
    --data-dir /home/nick/Lvl3Quant/data/processed/mbo_events_smart_v3 \
    --output-dir "$OUTPUT_DIR" \
    --n-folds $EVENT_N_FOLDS \
    --skip-transfer \
    --device cuda \
    2>&1 | tee "$OUTPUT_DIR/training.log"

echo ""
echo "Training complete: $(date)"
