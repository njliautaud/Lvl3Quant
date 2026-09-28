#!/bin/bash
# Chain remaining sweeps after sweep 1 (threshold=0.20) completes
# Launched manually after auto-launcher bug

cd /home/jupiter/Lvl3Quant
LOG="logs/tick_sweep_auto.log"

# Wait for sweep 1 to finish
echo "$(date) — Waiting for sweep 1 (PID $1) to complete..." | tee -a "$LOG"
wait $1 2>/dev/null || while kill -0 $1 2>/dev/null; do sleep 60; done

echo "$(date) — Sweep 1 done. Launching threshold=0.25 sweep." | tee -a "$LOG"
python3 -u engines/tick_replay_fast.py \
    --mode sweep \
    --tp-min 2 --tp-max 16 --tp-step 2 \
    --sl-min 2 --sl-max 10 --sl-step 2 \
    --threshold 0.25 \
    --hold 30 --cancel 15 \
    --perms 100 \
    --output engines/tick_sweep_t025_results.json \
    >> logs/tick_sweep_t025.log 2>&1

echo "$(date) — Sweep 2 done. Launching threshold=0.30 sweep." | tee -a "$LOG"
python3 -u engines/tick_replay_fast.py \
    --mode sweep \
    --tp-min 2 --tp-max 16 --tp-step 2 \
    --sl-min 2 --sl-max 10 --sl-step 2 \
    --threshold 0.30 \
    --hold 30 --cancel 15 \
    --perms 100 \
    --output engines/tick_sweep_t030_results.json \
    >> logs/tick_sweep_t030.log 2>&1

echo "$(date) — All 3 sweeps complete. Running analysis..." | tee -a "$LOG"
python3 scripts/analyze_tick_sweep.py >> "$LOG" 2>&1
echo "$(date) — Analysis complete." | tee -a "$LOG"
