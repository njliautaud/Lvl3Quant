#!/bin/bash
# Wait for current fusion v3 fold 0 (PID $V3_PID) to exit, then verify smart_v4 backfill done,
# then launch fusion v1 fold 0 with smart_v4. Logs to /home/nick/Lvl3Quant/logs/queue_smart_v4.log
set -u
V3_PID="${V3_PID:-7938}"
BACKFILL_PID="${BACKFILL_PID:-11829}"
SMART_V4_DIR=/home/nick/Lvl3Quant/data/processed/mbo_events_smart_v4
LOG=/home/nick/Lvl3Quant/logs/queue_smart_v4.log
ts() { date "+%Y-%m-%d %H:%M:%S"; }

echo "[$(ts)] queue_smart_v4 START. Watching V3_PID=$V3_PID, BACKFILL=$BACKFILL_PID" >> "$LOG"

# 1) Wait for current GPU job to exit
while kill -0 "$V3_PID" 2>/dev/null; do
    sleep 60
done
echo "[$(ts)] V3 fusion (PID $V3_PID) exited" >> "$LOG"

# 2) Wait up to 20 min for backfill to finish
WAIT=0
while kill -0 "$BACKFILL_PID" 2>/dev/null; do
    sleep 30
    WAIT=$((WAIT+30))
    if [ $WAIT -ge 1200 ]; then
        echo "[$(ts)] BACKFILL still running after 20 min — proceeding anyway with what is already in v4 dir" >> "$LOG"
        break
    fi
done
echo "[$(ts)] backfill (PID $BACKFILL_PID) exited or timed out (waited ${WAIT}s)" >> "$LOG"

# 3) Sanity check: smart_v4 dir has files
N_V4=$(ls "$SMART_V4_DIR" 2>/dev/null | wc -l)
echo "[$(ts)] smart_v4 has $N_V4 files" >> "$LOG"
if [ "$N_V4" -lt 150 ]; then
    echo "[$(ts)] FATAL: smart_v4 has too few files ($N_V4 < 150). Aborting." >> "$LOG"
    exit 1
fi

# 4) Launch fusion v1 fold 0 with smart_v4
export LVL3=/home/nick/Lvl3Quant
export PYTHON_BIN=/home/nick/miniconda3/envs/py311-train/bin/python3
export SMART_DATA_DIR="$SMART_V4_DIR"
export OUTPUT_DIR=/home/nick/Lvl3Quant/output/fusion_bakeoff_v1_smart_v4
export MLFLOW_TRACKING_URI=http://jupiter:5000
echo "[$(ts)] LAUNCHING fusion v1 fold 0 with SMART_V4..." >> "$LOG"
nohup bash /home/nick/Lvl3Quant/scripts/launch_fusion_v1_fold0.sh >> "$LOG" 2>&1 &
LAUNCH_PID=$!
echo "[$(ts)] launched PID $LAUNCH_PID" >> "$LOG"
