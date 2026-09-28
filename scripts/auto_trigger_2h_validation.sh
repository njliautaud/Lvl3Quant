#!/bin/bash
# Auto-trigger 2h ES clean walk-forward validation when minute bar rebuild completes
# Checks every 60 seconds for rebuild completion, then launches validation

BARS_DIR="/home/jupiter/Lvl3Quant/data/processed/mbo_minute_bars_v1"
LOG="/home/jupiter/Lvl3Quant/logs/auto_trigger_2h.log"
VALIDATION_SCRIPT="/home/jupiter/Lvl3Quant/scripts/lh_2h_clean_walkforward.py"

echo "[$(date)] Auto-trigger watcher started" >> "$LOG"

while true; do
    BAR_COUNT=$(ls "$BARS_DIR"/*.parquet 2>/dev/null | wc -l)
    WORKERS=$(ps aux | grep build_minute_bars | grep python | grep -v grep | wc -l)

    echo "[$(date)] Bars: $BAR_COUNT, Workers: $WORKERS" >> "$LOG"

    # If workers are done and we have a meaningful number of bars
    if [ "$WORKERS" -eq 0 ] && [ "$BAR_COUNT" -gt 100 ]; then
        echo "[$(date)] Rebuild COMPLETE ($BAR_COUNT bars). Launching 2h clean walk-forward..." >> "$LOG"

        cd /home/jupiter/Lvl3Quant
        nohup python3 "$VALIDATION_SCRIPT" > logs/lh_2h_clean_walkforward_v2.log 2>&1 &
        VALPID=$!
        echo "[$(date)] Walk-forward launched as PID $VALPID" >> "$LOG"

        exit 0
    fi

    sleep 60
done
