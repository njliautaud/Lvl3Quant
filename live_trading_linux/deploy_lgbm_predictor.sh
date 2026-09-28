#!/bin/bash
# Deploy LGBM Book Predictor for Live Trading
# Runs on Saturn/Jupiter CPU nodes

set -e

ROOT="/home/jupiter/Lvl3Quant"
MODEL_PATH="$ROOT/alpha_discovery/results/lgbm_book_features/lgbm_latest.pkl"
PREDICTOR="$ROOT/live_trading_linux/lgbm_book_inference.py"
DATA_PATH="/tmp/current_book_features.npz"  # Live book state
OUTPUT_PATH="/tmp/lgbm_predictions.npz"

echo "=== LGBM Live Predictor Deployment ==="
echo "Model: $MODEL_PATH"
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_PATH"
echo

# Check model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    echo "Train model first: python alpha_discovery/train_lgbm_book_features.py"
    exit 1
fi

# Check data exists
if [ ! -f "$DATA_PATH" ]; then
    echo "WARNING: No live data at $DATA_PATH"
    echo "Using sample data for testing..."
    # Use latest book feature file as sample
    SAMPLE=$(ls -t $ROOT/data/processed/mbo_book_features/2026*.npz | head -1)
    DATA_PATH="$SAMPLE"
fi

# Run prediction
echo "Running prediction..."
python3 "$PREDICTOR" \
    --model-path "$MODEL_PATH" \
    --data-path "$DATA_PATH" \
    --top-n 20 \
    --min-confidence 0.3 \
    --output "$OUTPUT_PATH"

echo
echo "✅ Predictions complete!"
echo "Output: $OUTPUT_PATH"
echo
echo "Top signals saved. Ready for trading engine integration."
