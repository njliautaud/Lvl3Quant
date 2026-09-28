#!/bin/bash
# CNN Mamba v2 execution research sweep on Jupiter — HC #30
# Sweeps signal_threshold × horizon × hold_ms across all available v2 folds.
# Output: /home/jupiter/Lvl3Quant/output/cnn_mamba_v2_fillsim_sweep_<TS>/

set -u
TS=$(date +%Y%m%d_%H%M)
ROOT=/home/jupiter/Lvl3Quant
OUT=$ROOT/output/cnn_mamba_v2_fillsim_sweep_$TS
mkdir -p "$OUT"
LOG=$OUT/sweep.log

echo "[$(date)] CNN Mamba v2 fillsim sweep starting" | tee -a "$LOG"
echo "Output: $OUT" | tee -a "$LOG"

# Discover available folds
FOLD_COUNT=$(ls $ROOT/output/cnn_mamba_v2_smart_v3_mar/fold_*_oot_predictions.npz 2>/dev/null | wc -l)
LAST_FOLD=$((FOLD_COUNT - 1))
echo "Found $FOLD_COUNT v2 folds (0-$LAST_FOLD)" | tee -a "$LOG"

cd $ROOT

# Sweep grid: 3 horizons × 5 thresholds × 4 hold times = 60 configs
for HORIZON in 1s 5s 10s; do
  for THRESH in 0.0 0.5 1.0 1.5 2.0; do
    for HOLD in 1000 5000 10000 30000; do
      TAG="h${HORIZON}_t${THRESH}_hold${HOLD}"
      echo "[$(date)] Running config: $TAG" | tee -a "$LOG"
      python3 scripts/deep_pred_to_fillsim.py \
        --model-dir output/cnn_mamba_v2_smart_v3_mar \
        --model-name cnn_mamba_v2_$TAG \
        --horizon $HORIZON \
        --window 500 --stride 250 \
        --folds 0-$LAST_FOLD \
        --signal-threshold $THRESH \
        --hold-ms $HOLD \
        > $OUT/run_$TAG.log 2>&1 || echo "  FAILED: $TAG"
    done
  done
done

echo "[$(date)] Sweep complete" | tee -a "$LOG"

# Aggregate results
python3 <<'PYEOF' >> "$LOG" 2>&1
import json, glob, os
from pathlib import Path
out = os.environ.get('OUT', '$OUT')
print("Aggregating results from", out)
# Look for fillsim summary jsons
for j in sorted(glob.glob(f"{out}/run_*.log")):
    name = Path(j).stem.replace("run_", "")
    print(name, "→", "see log")
PYEOF

echo "[$(date)] DONE" | tee -a "$LOG"
