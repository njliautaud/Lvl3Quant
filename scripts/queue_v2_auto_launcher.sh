#!/bin/bash
# Auto-launcher: checks if enough queue features exist, then runs v2 entry selector
# Run from cron every 30 minutes

QUEUE_DIR="/home/jupiter/Lvl3Quant/output/queue_features_universal"
FIFO_DIR="/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3_fifo_labels"
LOCKFILE="/tmp/queue_v2_running.lock"
MIN_OVERLAP=35  # Need at least 35 dates (25 train + 5 OOT + buffer)

# Don't run if already running (check lockfile AND process list)
if [ -f "$LOCKFILE" ]; then
    pid=$(cat "$LOCKFILE")
    if kill -0 "$pid" 2>/dev/null; then
        echo "$(date): v2 already running (PID $pid)"
        exit 0
    fi
    rm -f "$LOCKFILE"
fi
# Belt-and-suspenders: check by process name too
if pgrep -f 'queue_entry_selector_v2' >/dev/null 2>&1; then
    echo "$(date): v2 process already running (found by pgrep). Skipping."
    exit 0
fi

# Count overlapping dates
OVERLAP=$(python3 -c "
from pathlib import Path
import re
q = {f.stem.replace('features_','') for f in Path('$QUEUE_DIR').glob('features_*.parquet')}
f = {re.search(r'(\d{8})', f.stem).group(1) for f in Path('$FIFO_DIR').glob('*.npz') if re.search(r'(\d{8})', f.stem)}
print(len(q & f))
" 2>/dev/null)

echo "$(date): Queue-FIFO overlap: $OVERLAP dates (need $MIN_OVERLAP)"

if [ "$OVERLAP" -ge "$MIN_OVERLAP" ]; then
    # Don't re-launch if results already exist with same or more overlap dates
    RESULTS_FILE="/home/jupiter/Lvl3Quant/output/queue_entry_selector_v2/results.json"
    if [ -f "$RESULTS_FILE" ]; then
        RESULT_AGE=$(( $(date +%s) - $(stat -c %Y "$RESULTS_FILE") ))
        if [ "$RESULT_AGE" -lt 14400 ]; then  # Less than 4 hours old
            echo "$(date): Results already exist (${RESULT_AGE}s old). Skipping."
            exit 0
        fi
    fi
    echo "$(date): Enough data! Launching v2 entry selector..."
    cd /home/jupiter/Lvl3Quant
    nohup python3 -u alpha_discovery/queue_entry_selector_v2.py > logs/queue_entry_selector_v2.log 2>&1 &
    echo $! > "$LOCKFILE"
    echo "$(date): Launched PID $!"
else
    echo "$(date): Not enough data yet. Waiting..."
fi
