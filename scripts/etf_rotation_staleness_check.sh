#!/bin/bash
# HC #590 P1-1: ETF rotation staleness watchdog.
# Reads mtime of etf_rotation_paper_state.json; if older than 36h on a weekday
# (Mon=1..Fri=5), fires a QCC alert AND a Discord webhook notification.
# Skips on weekends.
set -u

STATE_FILE="/home/jupiter/Lvl3Quant/live_trading_linux/data/etf_rotation_paper_state.json"
LOG="/home/jupiter/Lvl3Quant/logs/etf_rotation_staleness.log"
WEBHOOK="/home/jupiter/teleclaude-main/utils/webhook_notifier.js"
INJECT_SH="/home/jupiter/Lvl3Quant/scripts/autonomy_inject.sh"
THRESHOLD_SEC=$((36 * 3600))

mkdir -p "$(dirname "$LOG")"
TS=$(date '+%Y-%m-%d %H:%M:%S %Z')
DOW=$(date +%u)   # 1=Mon .. 7=Sun

# Skip Saturday/Sunday
if [ "$DOW" -ge 6 ]; then
    echo "[$TS] skip weekend (dow=$DOW)" >> "$LOG"
    exit 0
fi

if [ ! -f "$STATE_FILE" ]; then
    MSG="ETF rotation state file MISSING ($STATE_FILE) — engine may have never run today."
    echo "[$TS] ALERT $MSG" >> "$LOG"
    [ -f "$WEBHOOK" ] && node "$WEBHOOK" "ETF rotation: $MSG" 2>/dev/null
    [ -f "$INJECT_SH" ] && "$INJECT_SH" "ETF_ROTATION_STALE: $MSG"
    exit 0
fi

NOW=$(date +%s)
MT=$(stat -c %Y "$STATE_FILE" 2>/dev/null || echo "$NOW")
AGE=$((NOW - MT))
AGE_HOURS=$((AGE / 3600))

if [ "$AGE" -gt "$THRESHOLD_SEC" ]; then
    MSG="ETF rotation state stale ${AGE_HOURS}h (>36h). Daily 9:42 cron may have silently failed."
    echo "[$TS] ALERT $MSG" >> "$LOG"
    [ -f "$WEBHOOK" ] && node "$WEBHOOK" "ETF rotation: $MSG" 2>/dev/null
    [ -f "$INJECT_SH" ] && "$INJECT_SH" "ETF_ROTATION_STALE: $MSG Check logs/etf_rotation_paper.log."
else
    echo "[$TS] OK age=${AGE_HOURS}h" >> "$LOG"
fi
