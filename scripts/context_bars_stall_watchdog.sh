#!/bin/bash
# HC #451 context-bars stall watchdog — added 2026-05-21 00:15 ET after 3rd multiprocessing deadlock.
# Pattern: Pool workers stay alive (CPU%) but stop producing dayfiles. Driver imap frozen.
# Detection: if newest .parquet in output dir is > STALL_MIN min old AND driver process alive → restart.
#
# Cron: */5 * * * *
# Safe to run when no context_bars driver is running (silent no-op).
#
# Author: HC #451 + HC #456 R3 pattern.

set -euo pipefail

OUT_DIR="/home/jupiter/Lvl3Quant/output/hc451_context_bars/per_day"
DRIVER_SCRIPT="/home/jupiter/Lvl3Quant/scripts/hc451_research/run_context_bars_all_days.py"
STATE_FILE="/home/jupiter/Lvl3Quant/logs/context_bars_watchdog_state.txt"
LOG_DIR="/home/jupiter/Lvl3Quant/logs"
STALL_MIN=45            # Newest parquet older than this AND driver alive → stall.
                        # Tuned for: big MBO .dbn.zst files take 5-15min/date × 12 workers,
                        # but some October 2025 dates with 600MB+ inputs take 30min+.
GRACE_AFTER_RESTART=1800 # 30-min cooldown after restart (workers need ramp-up on large files).
PROCESS_AGE_FLOOR_MIN=30 # Don't even consider stall until driver has been alive ≥ this long.

NOW=$(date +%s)

# Check driver alive
DRIVER_PIDS=$(pgrep -f "run_context_bars_all_days.py" || true)
if [ -z "$DRIVER_PIDS" ]; then
    # No driver running — count completed parquets. If 238 = done, silent. Else warn (someone should relaunch).
    PARQUET_COUNT=$(ls "$OUT_DIR"/*.parquet 2>/dev/null | wc -l)
    if [ "$PARQUET_COUNT" -lt 238 ]; then
        # Driver gone but build incomplete. Check if state file shows we just restarted (in grace).
        if [ -f "$STATE_FILE" ]; then
            LAST_RESTART=$(cat "$STATE_FILE" 2>/dev/null | head -1)
            if [ -n "$LAST_RESTART" ] && [ $((NOW - LAST_RESTART)) -lt "$GRACE_AFTER_RESTART" ]; then
                exit 0
            fi
        fi
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] WATCHDOG: driver gone, $PARQUET_COUNT/238 parquets — relaunching"
        cd /home/jupiter/Lvl3Quant
        nohup python3 "$DRIVER_SCRIPT" > "$LOG_DIR/hc451_context_bars_watchdog_relaunch_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
        echo "$NOW" > "$STATE_FILE"
    fi
    exit 0
fi

# Driver alive — check process age first (don't kill young drivers)
DRIVER_PID=$(echo "$DRIVER_PIDS" | head -1)
DRIVER_AGE_S=$(($(date +%s) - $(stat -c %Y /proc/$DRIVER_PID 2>/dev/null || echo "$NOW")))
DRIVER_AGE_MIN=$((DRIVER_AGE_S / 60))
if [ "$DRIVER_AGE_MIN" -lt "$PROCESS_AGE_FLOOR_MIN" ]; then
    exit 0
fi

# Check stall via run_log.jsonl growth (more reliable than parquet mtime — driver writes one line per done date)
LOG_FILE="$OUT_DIR/../run_log.jsonl"
if [ ! -f "$LOG_FILE" ]; then
    exit 0
fi
LOG_AGE_S=$((NOW - $(stat -c %Y "$LOG_FILE")))
LOG_AGE_MIN=$((LOG_AGE_S / 60))

if [ "$LOG_AGE_MIN" -lt "$STALL_MIN" ]; then
    # Healthy — run_log got an append within last STALL_MIN min (= a date completed)
    exit 0
fi

NEWEST_PARQUET=$(ls -t "$OUT_DIR"/*.parquet 2>/dev/null | head -1)
if [ -z "$NEWEST_PARQUET" ]; then
    exit 0
fi
NEWEST_AGE_S=$((NOW - $(stat -c %Y "$NEWEST_PARQUET")))
NEWEST_AGE_MIN=$((NEWEST_AGE_S / 60))

# Stall: driver alive but parquet output frozen
# Check grace period
if [ -f "$STATE_FILE" ]; then
    LAST_RESTART=$(cat "$STATE_FILE" 2>/dev/null | head -1)
    if [ -n "$LAST_RESTART" ] && [ $((NOW - LAST_RESTART)) -lt "$GRACE_AFTER_RESTART" ]; then
        # Recently restarted — let it settle
        exit 0
    fi
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] WATCHDOG: stall detected (newest parquet ${NEWEST_AGE_MIN}min old, driver alive) — killing + relaunching"
pkill -f "run_context_bars_all_days.py" || true
sleep 3
pkill -9 -f "run_context_bars_all_days.py" || true
sleep 2

cd /home/jupiter/Lvl3Quant
nohup python3 "$DRIVER_SCRIPT" > "$LOG_DIR/hc451_context_bars_watchdog_relaunch_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
echo "$NOW" > "$STATE_FILE"

# Single Discord inject on first stall — let user know watchdog acted
INJECT_SCRIPT="/home/jupiter/Lvl3Quant/scripts/autonomy_inject.sh"
if [ -x "$INJECT_SCRIPT" ]; then
    "$INJECT_SCRIPT" "WATCHDOG_FIRED: Context-bars build stalled (worker deadlock pattern, 4th occurrence). Watchdog killed + relaunched with skip-existing. Verify next parquet write within 5 min and root-cause the deadlock if time permits."
fi

exit 0
