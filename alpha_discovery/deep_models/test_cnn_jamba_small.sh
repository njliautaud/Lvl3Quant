#!/bin/bash
# Quick CNN-Jamba test with minimal dataset
# Tests if architecture works without waiting hours

cd ~/Lvl3Quant/alpha_discovery/deep_models || exit 1

# Activate environment
source ~/ray-env/bin/activate

echo "========================================"
echo "CNN-Jamba Quick Test"
echo "========================================"
echo "Config: 1 fold, reduced dataset"
echo "Purpose: Validate architecture works"
echo "Expected time: 15-30 minutes"
echo ""
echo "⚠️  IMPORTANT: Close Overwatch and other GPU apps first!"
echo ""

# Run with minimal config (using existing flags only):
# - 1 fold only
# - Uses default config (window=1000, but will work with less data)
python3 train_cnn_jamba.py \
  --data-dir ~/Lvl3Quant/data/processed/mbo_events \
  --output-dir results/cnn_jamba_test \
  --n-folds 1

echo ""
echo "Test complete. Check results/cnn_jamba_test/ for output"
echo "If successful, run full training with --n-folds 5"
