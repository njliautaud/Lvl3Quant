#!/bin/bash
# Extended LGBM walk-forward training including fresh April 2026 data
# Run this AFTER syncing fresh data with sync_fresh_data.sh

set -e

echo "=== Extended LGBM Walk-Forward Training ==="
echo "Start time: $(date)"
echo ""

# Count total available files
DATA_DIR="/home/jupiter/Lvl3Quant/data/processed/mbo_events"
TOTAL_FILES=$(ls "$DATA_DIR"/*.npz 2>/dev/null | wc -l)

echo "Total MBO files available: $TOTAL_FILES"
echo "Expected: ~110+ files (Jan 2025 - April 2026)"
echo ""

if [ "$TOTAL_FILES" -lt 100 ]; then
    echo "⚠️  WARNING: Only $TOTAL_FILES files found. Did fresh data sync complete?"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Training config
export LGBM_N_FOLDS=10  # More folds with more data
export LGBM_LOOKBACK_DAYS=60
export LGBM_EXPANDING_WINDOW=1
OUTPUT_DIR="/home/jupiter/Lvl3Quant/models/lgbm_extended_$(date +%Y%m%d)"

echo "Config:"
echo "  Folds: $LGBM_N_FOLDS (expanding window)"
echo "  Lookback: $LGBM_LOOKBACK_DAYS days"
echo "  Output: $OUTPUT_DIR"
echo ""

mkdir -p "$OUTPUT_DIR"

# Launch training
cd /home/jupiter/Lvl3Quant

echo "Launching LGBM training..."
python3 alpha_discovery/train_lgbm_sliding_60d.py \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --n-folds "$LGBM_N_FOLDS" \
    --lookback-days "$LGBM_LOOKBACK_DAYS" \
    --expanding-window 2>&1 | tee "$OUTPUT_DIR/training.log"

echo ""
echo "=== Training Complete ==="
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Next: Evaluate IC on most recent folds to assess current performance"
