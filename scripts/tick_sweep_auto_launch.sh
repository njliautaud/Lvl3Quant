#!/bin/bash
# Auto-launch tick-level sweep when preprocessing completes
# Monitors preprocessed_mbo/ for all 34 files, then launches sweep

PREPROC_DIR="/home/jupiter/Lvl3Quant/data/preprocessed_mbo"
LOG="/home/jupiter/Lvl3Quant/logs/tick_sweep_auto.log"
EXPECTED=34

echo "$(date) — Waiting for $EXPECTED preprocessed files in $PREPROC_DIR" | tee -a "$LOG"

while true; do
    COUNT=$(ls "$PREPROC_DIR"/mbo_*.npz 2>/dev/null | wc -l)
    # Check if preprocess PID is still alive
    PREPROC_PID=$(pgrep -f "tick_replay_fast.py.*preprocess" 2>/dev/null)

    if [ "$COUNT" -ge "$EXPECTED" ]; then
        echo "$(date) — All $COUNT/$EXPECTED files ready. Launching sweep." | tee -a "$LOG"
        break
    elif [ -z "$PREPROC_PID" ] && [ "$COUNT" -lt "$EXPECTED" ]; then
        if [ "$COUNT" -ge 20 ]; then
            echo "$(date) — Preprocess finished with $COUNT files (some dates were weekends/empty). Launching sweep." | tee -a "$LOG"
            break
        else
            echo "$(date) — Preprocess died with only $COUNT/$EXPECTED files. Too few. Aborting." | tee -a "$LOG"
            exit 1
        fi
    fi

    echo "$(date) — $COUNT/$EXPECTED files ready, PID=$PREPROC_PID still running..." >> "$LOG"
    sleep 60
done

# Launch comprehensive sweep with multiple threshold levels
# Signal characteristics: 1s horizon, mean≈0.06, std≈0.19
# Test thresholds that capture different confidence tiers

cd /home/jupiter/Lvl3Quant

echo "$(date) — Launching sweep: TP 2-16, SL 2-10, threshold 0.20, 100 perms" | tee -a "$LOG"

nohup python3 -u engines/tick_replay_fast.py \
    --mode sweep \
    --tp-min 2 --tp-max 16 --tp-step 2 \
    --sl-min 2 --sl-max 10 --sl-step 2 \
    --threshold 0.20 \
    --hold 30 --cancel 15 \
    --perms 100 \
    --output engines/tick_sweep_t020_results.json \
    >> logs/tick_sweep_t020.log 2>&1 &
SWEEP_PID=$!

echo "$(date) — Sweep PID=$SWEEP_PID launched (threshold=0.20)" | tee -a "$LOG"

# Wait for first sweep to finish before launching second (CPU constrained)
wait $SWEEP_PID

# Second sweep with threshold=0.25 (7.8% of predictions, good L/S balance)
echo "$(date) — First sweep done. Launching threshold=0.25 sweep." | tee -a "$LOG"
nohup python3 -u engines/tick_replay_fast.py \
    --mode sweep \
    --tp-min 2 --tp-max 16 --tp-step 2 \
    --sl-min 2 --sl-max 10 --sl-step 2 \
    --threshold 0.25 \
    --hold 30 --cancel 15 \
    --perms 100 \
    --output engines/tick_sweep_t025_results.json \
    >> logs/tick_sweep_t025.log 2>&1 &
SWEEP_PID2=$!

echo "$(date) — Sweep PID=$SWEEP_PID2 launched (threshold=0.25)" | tee -a "$LOG"

wait $SWEEP_PID2

# Third sweep: high threshold (selective, 0.45% of predictions)
echo "$(date) — Second sweep done. Launching threshold=0.30 sweep." | tee -a "$LOG"
nohup python3 -u engines/tick_replay_fast.py \
    --mode sweep \
    --tp-min 2 --tp-max 16 --tp-step 2 \
    --sl-min 2 --sl-max 10 --sl-step 2 \
    --threshold 0.30 \
    --hold 30 --cancel 15 \
    --perms 100 \
    --output engines/tick_sweep_t030_results.json \
    >> logs/tick_sweep_t030.log 2>&1 &
SWEEP_PID3=$!

echo "$(date) — Sweep PID=$SWEEP_PID3 launched (threshold=0.30)" | tee -a "$LOG"

wait $SWEEP_PID3

# Auto-analyze all results
echo "$(date) — All sweeps complete. Running analysis..." | tee -a "$LOG"
python3 /home/jupiter/Lvl3Quant/scripts/analyze_tick_sweep.py >> "$LOG" 2>&1
echo "$(date) — Analysis complete. Results in output/tick_sweep_analysis.md" | tee -a "$LOG"
