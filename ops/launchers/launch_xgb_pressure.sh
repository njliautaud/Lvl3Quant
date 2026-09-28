#!/bin/bash
cd /home/nick/Lvl3Quant
source /home/nick/miniconda3/etc/profile.d/conda.sh
conda activate py311-train
export MLFLOW_TRACKING_URI=http://localhost:5000

LOG=logs/xgb_pressure_combined.log
rm -f "$LOG"

for TARGET in eofi pdi; do
  for HORIZON in 10s 30s; do
    echo "=== Starting ${TARGET} ${HORIZON} ===" >> "$LOG"
    python scripts/train_pressure_xgb_v2.py \
      --data-dir data/processed/mbo_events_smart_v3 \
      --pressure-dir data/processed/smooth_pressure_targets \
      --output-dir output/pressure_xgb_${TARGET}_${HORIZON} \
      --target "$TARGET" --horizon "$HORIZON" \
      --subsample-ratio 0.05 \
      --rolling-windows "" \
      --mlflow-uri http://localhost:5000 \
      >> "$LOG" 2>&1
    echo "=== Done ${TARGET} ${HORIZON} ===" >> "$LOG"
  done
done

echo "=== ALL DONE ===" >> "$LOG"
