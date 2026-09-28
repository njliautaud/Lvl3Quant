#!/bin/bash
# Waits for precompute to finish, then launches Mamba v5 feat15
# Runs as a background process so it survives SSH disconnects

PRECOMPUTE_PID=148788
LOG="/home/nick/Lvl3Quant/logs/mamba_v5_feat15.log"

echo "$(date): Waiting for precompute PID $PRECOMPUTE_PID to finish..." >> "$LOG"

# Wait for precompute to finish
while kill -0 $PRECOMPUTE_PID 2>/dev/null; do
    sleep 30
done

echo "$(date): Precompute done. Checking file count..." >> "$LOG"

# Verify we have enough files
NFILES=$(ls /home/nick/Lvl3Quant/data/processed/mbo_events_feat15/*.npz 2>/dev/null | wc -l)
echo "$(date): Found $NFILES feat15 NPZ files" >> "$LOG"

if [ "$NFILES" -lt 200 ]; then
    echo "$(date): ERROR: Only $NFILES files, expected ~209. Aborting." >> "$LOG"
    exit 1
fi

# Launch Mamba v5 feat15
echo "$(date): Launching Mamba v5 feat15..." >> "$LOG"
cd /home/nick/Lvl3Quant

# PID locking
LOCKDIR="/home/nick/Lvl3Quant/locks"
mkdir -p "$LOCKDIR"
LOCKFILE="$LOCKDIR/mamba_training.lock"
if [ -f "$LOCKFILE" ] && kill -0 $(cat "$LOCKFILE") 2>/dev/null; then
    echo "$(date): ERROR: Mamba already running (PID $(cat $LOCKFILE))" >> "$LOG"
    exit 1
fi

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

/home/nick/miniconda3/envs/py311-train/bin/python -u \
    alpha_discovery/deep_models/train_event_mamba_cuda.py \
    --output-dir /home/nick/Lvl3Quant/output/mamba_v5_sliding60d_feat15 \
    --n-folds 10 \
    >> "$LOG" 2>&1 &

MAMBA_PID=$!
echo "$MAMBA_PID" > "$LOCKFILE"
echo "$(date): Mamba v5 feat15 launched (PID $MAMBA_PID)" >> "$LOG"
echo "$(date): Output: /home/nick/Lvl3Quant/output/mamba_v5_sliding60d_feat15" >> "$LOG"
echo "$(date): Lock: $LOCKFILE" >> "$LOG"

# Wait and clean lock on exit
wait $MAMBA_PID
rm -f "$LOCKFILE"
echo "$(date): Mamba v5 feat15 exited (code $?)" >> "$LOG"
