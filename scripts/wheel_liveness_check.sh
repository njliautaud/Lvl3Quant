#!/bin/bash
# HC #590 P1-3: Wheel paper-engine liveness watchdog.
# The wheel daemon writes wheel_paper_state/state.json on each decision loop.
# If the file is older than 6h on a weekday, the daemon is likely hung even if
# PM2 reports "online". Fires QCC alert + Discord webhook + autonomy inject.
set -u

STATE_FILE="/home/jupiter/Lvl3Quant/live_trading_linux/wheel_paper_state/state.json"
LOG="/home/jupiter/Lvl3Quant/logs/wheel_liveness.log"
WEBHOOK="/home/jupiter/teleclaude-main/utils/webhook_notifier.js"
INJECT_SH="/home/jupiter/Lvl3Quant/scripts/autonomy_inject.sh"
THRESHOLD_SEC=$((6 * 3600))

mkdir -p "$(dirname "$LOG")"
TS=$(date '+%Y-%m-%d %H:%M:%S %Z')
DOW=$(date +%u)   # 1=Mon .. 7=Sun

if [ "$DOW" -ge 6 ]; then
    echo "[$TS] skip weekend (dow=$DOW)" >> "$LOG"
    exit 0
fi

# If wheel-paper-engine is intentionally stopped in PM2, don't alert.
PM2_BIN="${PM2_BIN:-$(command -v pm2 2>/dev/null || echo /home/jupiter/.nvm/versions/node/v22.22.2/bin/pm2)}"
PM2_STATUS=$("$PM2_BIN" jlist 2>/dev/null | python3 -c "
import json,sys
for p in json.load(sys.stdin):
    if p.get('name')=='wheel-paper-engine':
        print(p.get('pm2_env',{}).get('status','unknown'))
        break
" 2>/dev/null || echo "unknown")
if [ "$PM2_STATUS" = "stopped" ]; then
    echo "[$TS] skip — wheel-paper-engine intentionally stopped in PM2" >> "$LOG"
    exit 0
fi

if [ ! -f "$STATE_FILE" ]; then
    MSG="Wheel paper state file MISSING ($STATE_FILE) — engine may have never run."
    echo "[$TS] ALERT $MSG" >> "$LOG"
    [ -f "$WEBHOOK" ] && node "$WEBHOOK" "Wheel: $MSG" 2>/dev/null
    [ -f "$INJECT_SH" ] && "$INJECT_SH" "WHEEL_STALE: $MSG"
    exit 0
fi

NOW=$(date +%s)
MT=$(stat -c %Y "$STATE_FILE" 2>/dev/null || echo "$NOW")
AGE=$((NOW - MT))
AGE_HOURS=$((AGE / 3600))

if [ "$AGE" -gt "$THRESHOLD_SEC" ]; then
    MSG="Wheel paper state stale ${AGE_HOURS}h (>6h). Daemon may be hung — restart wheel-paper-engine via PM2."
    echo "[$TS] ALERT $MSG" >> "$LOG"
    [ -f "$WEBHOOK" ] && node "$WEBHOOK" "Wheel: $MSG" 2>/dev/null
    [ -f "$INJECT_SH" ] && "$INJECT_SH" "WHEEL_STALE: $MSG Check pm2 logs wheel-paper-engine."
else
    echo "[$TS] OK age=${AGE_HOURS}h" >> "$LOG"
fi
