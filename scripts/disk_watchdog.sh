#!/bin/bash
# Disk Watchdog - prevents runaway disk/log usage on ALL nodes
# Runs via cron every 4 hours on Jupiter
# Covers: Jupiter (local), Neptune (SSH), Razer (SSH)

LOG_MAX_MB=10
THRESHOLD_WARN=70
THRESHOLD_CRIT=85
WEBHOOK="/home/jupiter/teleclaude-main/utils/webhook_notifier.js"

truncate_logs() {
    local dir="$1"
    for f in "$dir"/*.log "$dir"/**/*.log; do
        [ -f "$f" ] || continue
        SIZE_MB=$(du -m "$f" 2>/dev/null | cut -f1)
        if [ "${SIZE_MB:-0}" -gt "$LOG_MAX_MB" ]; then
            echo "  Truncating $f (${SIZE_MB}MB)"
            tail -500 "$f" > "${f}.tmp" && mv "${f}.tmp" "$f"
        fi
    done
}

alert() {
    echo "[$(date)] $1"
    [ -f "$WEBHOOK" ] && node "$WEBHOOK" "⚠️ Disk: $1" 2>/dev/null
}

# ── JUPITER (local) ──
echo "[$(date)] === Jupiter ==="
DISK_PCT=$(df / --output=pcent | tail -1 | tr -d ' %')
echo "  Disk: ${DISK_PCT}%"
truncate_logs "/home/jupiter/.pm2/logs"
truncate_logs "/home/jupiter/teleclaude-main/logs"
truncate_logs "/home/jupiter/teleclaude/logs"
truncate_logs "/home/jupiter/Lvl3Quant/logs"
find /tmp -maxdepth 2 -type f -size +100M -delete 2>/dev/null
[ "$DISK_PCT" -ge "$THRESHOLD_CRIT" ] && alert "Jupiter CRITICAL ${DISK_PCT}%"

# ── NEPTUNE (SSH) ──
echo "[$(date)] === Neptune ==="
ssh -o ConnectTimeout=8 -o StrictHostKeyChecking=no nick@neptune bash -s << 'REMOTE'
DISK_PCT=$(df / --output=pcent | tail -1 | tr -d ' %')
echo "  Disk: ${DISK_PCT}%"
for dir in /home/nick/Lvl3Quant/logs /home/nick/Lvl3Quant/output/*/logs /tmp; do
    for f in "$dir"/*.log "$dir"/**/*.log; do
        [ -f "$f" ] || continue
        SIZE_MB=$(du -m "$f" 2>/dev/null | cut -f1)
        if [ "${SIZE_MB:-0}" -gt 10 ]; then
            echo "  Truncating $f (${SIZE_MB}MB)"
            tail -500 "$f" > "${f}.tmp" && mv "${f}.tmp" "$f"
        fi
    done
done
find /tmp -maxdepth 2 -type f -size +100M -delete 2>/dev/null
# Report back disk pct for alerting
echo "DISK_PCT=${DISK_PCT}"
REMOTE
NEPTUNE_PCT=$(ssh -o ConnectTimeout=5 nick@neptune "df / --output=pcent | tail -1 | tr -d ' %'" 2>/dev/null)
[ "${NEPTUNE_PCT:-0}" -ge "$THRESHOLD_CRIT" ] && alert "Neptune CRITICAL ${NEPTUNE_PCT}%"

# ── RAZER (SSH) ──
echo "[$(date)] === Razer ==="
ssh -o ConnectTimeout=8 -o StrictHostKeyChecking=no claude@razer bash -s << 'REMOTE'
# Windows/Cygwin/Git Bash - check common log locations
for dir in /c/Users/claude/Lvl3Quant/logs /c/Users/claude/Lvl3Quant/output/*/logs /c/Users/claude/AppData/Local/Temp; do
    [ -d "$dir" ] || continue
    for f in "$dir"/*.log; do
        [ -f "$f" ] || continue
        SIZE_MB=$(du -m "$f" 2>/dev/null | cut -f1)
        if [ "${SIZE_MB:-0}" -gt 10 ]; then
            echo "  Truncating $f (${SIZE_MB}MB)"
            tail -500 "$f" > "${f}.tmp" && mv "${f}.tmp" "$f"
        fi
    done
done
REMOTE
echo "[$(date)] === Done ==="
