#!/bin/bash
# Watch minute bar rebuild and auto-trigger clean walkforward when done
# Checks every 60s if rebuild process is finished and all 238 files exist

BAR_DIR="/home/jupiter/Lvl3Quant/data/processed/mbo_minute_bars_v1"
EXPECTED=238
LOG="/home/jupiter/Lvl3Quant/logs/minbar_rebuild_watcher.log"
SCRIPT="/home/jupiter/Lvl3Quant/scripts/lh_2h_clean_walkforward.py"
OUT_LOG="/home/jupiter/Lvl3Quant/logs/lh_2h_clean_walkforward_$(date +%Y%m%d_%H%M%S).log"

echo "[$(date)] Watcher started. Waiting for $EXPECTED minute bar files..." >> "$LOG"

while true; do
    COUNT=$(ls "$BAR_DIR"/*.parquet 2>/dev/null | wc -l)
    REBUILD_RUNNING=$(pgrep -f "rebuild_minute_bars" | wc -l)

    echo "[$(date)] Files: $COUNT/$EXPECTED, Rebuild procs: $REBUILD_RUNNING" >> "$LOG"

    if [ "$COUNT" -ge "$EXPECTED" ] && [ "$REBUILD_RUNNING" -eq 0 ]; then
        echo "[$(date)] REBUILD COMPLETE! $COUNT files ready. Launching clean walkforward..." >> "$LOG"

        cd /home/jupiter/Lvl3Quant
        nohup python3 "$SCRIPT" > "$OUT_LOG" 2>&1 &
        WF_PID=$!
        echo "[$(date)] Clean walkforward launched, PID=$WF_PID, log=$OUT_LOG" >> "$LOG"

        exit 0
    fi

    if [ "$REBUILD_RUNNING" -eq 0 ] && [ "$COUNT" -lt "$EXPECTED" ]; then
        echo "[$(date)] WARNING: Rebuild not running but only $COUNT/$EXPECTED files. Rebuild may have failed." >> "$LOG"
        # Don't exit — maybe rebuild is between batches
    fi

    sleep 60
done
