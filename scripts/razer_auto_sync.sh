#!/bin/bash
# Auto-detect Razer coming online and trigger bulk sync
# Runs via crontab every 30 min. Only acts when Razer is reachable AND we have a data gap.
# After successful sync, triggers minute bar conversion and notifies via Discord webhook.

RAZER_HOST="claude@razer"
LOCK_FILE="/tmp/razer_auto_sync.lock"
LAST_SYNC_FILE="/home/jupiter/Lvl3Quant/logs/razer_last_sync.txt"
LOG="/home/jupiter/Lvl3Quant/logs/razer_auto_sync.log"
SYNC_SCRIPT="/home/jupiter/Lvl3Quant/scripts/razer_bulk_sync.sh"

# Prevent concurrent runs
if [ -f "$LOCK_FILE" ]; then
    LOCK_AGE=$(($(date +%s) - $(stat -c%Y "$LOCK_FILE" 2>/dev/null || echo 0)))
    if [ "$LOCK_AGE" -lt 1800 ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') Sync already running (lock age ${LOCK_AGE}s)" >> "$LOG"
        exit 0
    fi
    # Stale lock, remove it
    rm -f "$LOCK_FILE"
fi

# Only run during market hours or shortly after (6 AM - 8 PM ET)
HOUR_ET=$(TZ=America/New_York date +%H)
if [ "$HOUR_ET" -lt 6 ] || [ "$HOUR_ET" -gt 22 ]; then
    exit 0  # Silent exit outside useful hours
fi

# Quick reachability check (2s timeout)
if ! ssh -o ConnectTimeout=2 -o BatchMode=yes "${RAZER_HOST}" "echo ok" >/dev/null 2>&1; then
    # Try with password
    if ! sshpass -p "${CLUSTER_SSH_PASSWORD}" ssh -o ConnectTimeout=3 -o StrictHostKeyChecking=no \
        "${RAZER_HOST}" "echo ok" >/dev/null 2>&1; then
        exit 0  # Razer offline, silent exit
    fi
fi

# Razer is online! Check if we already synced today
TODAY=$(date +%Y-%m-%d)
LAST_SYNC=$(cat "$LAST_SYNC_FILE" 2>/dev/null || echo "never")
if [ "$LAST_SYNC" = "$TODAY" ]; then
    exit 0  # Already synced today
fi

# Create lock
touch "$LOCK_FILE"
echo "$(date '+%Y-%m-%d %H:%M:%S') Razer detected online! Starting auto-sync..." >> "$LOG"

# Run the bulk sync
RESULT=$("$SYNC_SCRIPT" 2>&1)
SYNC_EXIT=$?
echo "$(date '+%Y-%m-%d %H:%M:%S') Sync result: ${RESULT}" >> "$LOG"

if [ $SYNC_EXIT -eq 0 ]; then
    # Extract sync count from result
    SYNCED_COUNT=$(echo "$RESULT" | grep -oP 'Synced: \K[0-9]+' || echo "0")

    if [ "$SYNCED_COUNT" -gt 0 ]; then
        echo "$TODAY" > "$LAST_SYNC_FILE"
        echo "$(date '+%Y-%m-%d %H:%M:%S') SUCCESS: ${SYNCED_COUNT} new files synced" >> "$LOG"

        # Trigger minute bar conversion if script exists
        if [ -f "/home/jupiter/Lvl3Quant/scripts/run_minute_bars_bulk.sh" ]; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') Running minute bar conversion..." >> "$LOG"
            bash /home/jupiter/Lvl3Quant/scripts/run_minute_bars_bulk.sh >> "$LOG" 2>&1
        fi

        # Alert via autonomy inject (will be picked up by next Claude session)
        echo "RAZER_AUTO_SYNC_COMPLETE: ${SYNCED_COUNT} new MBO files synced from Razer on ${TODAY}. Minute bars converted. Paper engine will auto-detect on next 30-min reload." > /tmp/razer_sync_alert.txt
    else
        echo "$(date '+%Y-%m-%d %H:%M:%S') No new files to sync (already up to date)" >> "$LOG"
        echo "$TODAY" > "$LAST_SYNC_FILE"
    fi
else
    echo "$(date '+%Y-%m-%d %H:%M:%S') SYNC FAILED: exit ${SYNC_EXIT}" >> "$LOG"
fi

# Remove lock
rm -f "$LOCK_FILE"
