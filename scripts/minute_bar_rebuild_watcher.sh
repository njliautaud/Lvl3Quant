#!/bin/bash
# Watch for minute bar rebuild completion, then trigger clean walk-forward validation
BARS_DIR="/home/jupiter/Lvl3Quant/data/processed/mbo_minute_bars_v1"
TARGET=238
LOG="/home/jupiter/Lvl3Quant/logs/rebuild_watcher.log"

while true; do
    COUNT=$(ls "$BARS_DIR"/*.parquet 2>/dev/null | wc -l)
    WORKERS=$(ps aux | grep build_minute_bars | grep -v grep | wc -l)
    echo "[$(date)] Bars: $COUNT/$TARGET, Workers: $WORKERS" >> "$LOG"
    
    if [ "$WORKERS" -eq 0 ] && [ "$COUNT" -gt 50 ]; then
        echo "[$(date)] Rebuild COMPLETE ($COUNT bars). Workers finished." >> "$LOG"
        # Signal completion
        touch /home/jupiter/Lvl3Quant/data/processed/mbo_minute_bars_v1/.rebuild_complete
        echo "[$(date)] Marked .rebuild_complete. Exiting watcher." >> "$LOG"
        exit 0
    fi
    sleep 60
done
