#!/bin/bash
# v4_fold141_auto_analyze.sh — Sync fold 141 from Neptune and run full validation pipeline
# Run this when fold 141 training completes
set -e

echo "=== Checking fold 141 on Neptune ==="
PRED_FILE="fold_141_oot_predictions.npz"
REMOTE="nick@neptune:/home/nick/Lvl3Quant/output/v4_multihead_pressure_v1"
LOCAL="/home/jupiter/Lvl3Quant/output/v4_multihead_pressure_v1"

# Check if predictions exist on Neptune
if ssh nick@neptune "test -f /home/nick/Lvl3Quant/output/v4_multihead_pressure_v1/${PRED_FILE}"; then
    echo "Fold 141 predictions found! Syncing..."
    scp "${REMOTE}/${PRED_FILE}" "${LOCAL}/"
    scp "${REMOTE}/fold_141_best.pt" "${LOCAL}/" 2>/dev/null || true
    echo "Sync complete."
else
    echo "ERROR: Fold 141 predictions not yet available"
    exit 1
fi

echo ""
echo "=== Running label-based analysis (all folds 126-141) ==="
cd /home/jupiter/Lvl3Quant
python3 scripts/v4_deep_validation.py 2>&1 | tee /home/jupiter/Lvl3Quant/logs/v4_fold141_validation.log

echo ""
echo "=== Running FIFO tick-level simulation (fold 141 = Mar 12) ==="
python3 scripts/v4_fifo_tick_sim_multi.py --fold 141 2>&1 | tee /home/jupiter/Lvl3Quant/logs/v4_fold141_fifo.log

echo ""
echo "=== DONE ==="
echo "Check logs in /home/jupiter/Lvl3Quant/logs/"
