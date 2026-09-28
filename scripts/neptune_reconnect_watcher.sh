#!/bin/bash
# Watch for Neptune connectivity restoration, then rsync book data + upgrade to smart_v4_book
LOG="/home/jupiter/Lvl3Quant/logs/neptune_reconnect.log"
MAX_ATTEMPTS=120  # 120 * 30s = 1 hour max

echo "$(date) — Starting Neptune reconnect watcher" >> "$LOG"

for i in $(seq 1 $MAX_ATTEMPTS); do
    if ssh -o ConnectTimeout=5 -o BatchMode=yes nick@neptune "echo ok" > /dev/null 2>&1; then
        echo "$(date) — Neptune CONNECTED! Starting book data rsync..." >> "$LOG"
        
        # Check if smart_v4 preprocessing is done on Neptune
        V4_COUNT=$(ssh nick@neptune "ls /home/nick/Lvl3Quant/data/processed/mbo_events_smart_v4/ 2>/dev/null | wc -l")
        echo "$(date) — Neptune smart_v4 files: $V4_COUNT" >> "$LOG"
        
        # Check what's running
        RUNNING=$(ssh nick@neptune "ps aux | grep -E 'train_triple|train_event' | grep -v grep | head -3")
        echo "$(date) — Running processes: $RUNNING" >> "$LOG"
        
        # Rsync book normalized data
        BOOK_COUNT=$(ls /home/jupiter/Lvl3Quant/data/processed/mbo_book_normalized/*.npz 2>/dev/null | wc -l)
        echo "$(date) — Rsyncing $BOOK_COUNT book normalized files to Neptune..." >> "$LOG"
        ssh nick@neptune "mkdir -p /home/nick/Lvl3Quant/data/processed/mbo_book_normalized"
        rsync -az /home/jupiter/Lvl3Quant/data/processed/mbo_book_normalized/ nick@neptune:/home/nick/Lvl3Quant/data/processed/mbo_book_normalized/ >> "$LOG" 2>&1
        echo "$(date) — Book data rsync COMPLETE" >> "$LOG"
        
        exit 0
    fi
    sleep 30
done

echo "$(date) — Gave up after $MAX_ATTEMPTS attempts" >> "$LOG"
