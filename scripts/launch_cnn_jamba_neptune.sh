#!/bin/bash
# Launch CNN-Jamba training on Neptune (Ubuntu + RTX 3090)
# Auto-generated launch script

set -e

echo "=== CNN-Jamba Training Launch ==="
echo "Node: Neptune (neptune)"
echo "GPU: RTX 3090 24GB"
echo "Start time: $(date)"
echo ""

# Set environment variables for optimal performance
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export MLFLOW_TRACKING_URI=http://jupiter:5000

# CNN-Jamba hyperparameters
export CNN_CHANNELS=64
export CNN_LAYERS=3
export CNN_KERNEL=5
export JAMBA_D_MODEL=128
export JAMBA_D_STATE=64
export JAMBA_N_BLOCKS=4
export JAMBA_N_HEADS=4
export JAMBA_DROPOUT=0.1
export EVENT_WINDOW_SIZE=500
export EVENT_STRIDE=250
export EVENT_BATCH_SIZE=32
export EVENT_LR=0.0003
export EVENT_EPOCHS=5
export EVENT_N_FOLDS=5

# Paths
DATA_DIR="/home/nick/Lvl3Quant/data/processed/mbo_events"
OUTPUT_DIR="/home/nick/Lvl3Quant/alpha_discovery/deep_models/results/cnn_jamba_neptune_$(date +%Y%m%d)"
SCRIPT_PATH="/home/nick/Lvl3Quant/alpha_discovery/deep_models/train_cnn_jamba.py"

echo "Config:"
echo "  Window size: $EVENT_WINDOW_SIZE events"
echo "  Batch size: $EVENT_BATCH_SIZE"
echo "  Folds: $EVENT_N_FOLDS"
echo "  CNN layers: $CNN_LAYERS x $CNN_CHANNELS channels"
echo "  Jamba blocks: $JAMBA_N_BLOCKS (d_model=$JAMBA_D_MODEL)"
echo "  Output: $OUTPUT_DIR"
echo ""

# Activate virtual environment
source /home/nick/Lvl3Quant/venv_training/bin/activate

# Check PyTorch installation
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')" || {
    echo "ERROR: PyTorch not installed or CUDA not available"
    exit 1
}

# Check GPU
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || {
    echo "ERROR: nvidia-smi failed"
    exit 1
}

# Dependencies already in venv

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Launch training
echo ""
echo "=== Starting CNN-Jamba Training ==="
cd /home/nick/Lvl3Quant

python -u "$SCRIPT_PATH" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --n-folds "$EVENT_N_FOLDS" \
    --device cuda \
    --skip-transfer 2>&1 | tee "$OUTPUT_DIR/training.log"

echo ""
echo "=== Training Complete ==="
echo "End time: $(date)"
