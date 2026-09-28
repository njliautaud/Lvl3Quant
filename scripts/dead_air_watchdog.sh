#!/bin/bash
# HC #461 R5 — Dead-air detection.
# If no substantive Discord message in the last 90 min AND user has been idle,
# log a HIGH-PRIORITY entry to accountability.log. This is a tripwire so the user
# can immediately see if the bot died vs just had nothing to say.
set -u

LOG=/home/jupiter/Lvl3Quant/logs/accountability.log
WINDOW_MIN=90

mkdir -p "$(dirname "$LOG")"

TS=$(date '+%Y-%m-%d %H:%M:%S %Z')
NOW_EPOCH=$(date +%s)
CUTOFF=$((NOW_EPOCH - WINDOW_MIN * 60))

BRIDGE_LOG="/home/jupiter/teleclaude-main/logs/bridge-$(date '+%Y-%m-%d').log"
if [ ! -f "$BRIDGE_LOG" ]; then
    echo "[$TS] DEAD_AIR_CHECK no_bridge_log_yet" >> "$LOG"
    exit 0
fi

# Last "Sending Discord message" — but ignore boilerplate (Still working, Session recovered, Morning Briefing)
LAST_SUBSTANTIVE_EPOCH=0
while IFS= read -r ISO_TS; do
    [ -z "$ISO_TS" ] && continue
    EPOCH=$(date -d "$ISO_TS" +%s 2>/dev/null || echo 0)
    [ "$EPOCH" -gt "$LAST_SUBSTANTIVE_EPOCH" ] && LAST_SUBSTANTIVE_EPOCH=$EPOCH
done < <(grep -B1 "Sending Discord message" "$BRIDGE_LOG" 2>/dev/null \
    | grep -vE "Still working|Session recovered|Morning Briefing|EOD|Proactive context reset|stream hung" \
    | grep -oE "2026-[0-9-]+T[0-9:.]+Z" \
    | tail -50)

# Last user-message receipt
LAST_USER_EPOCH=0
LAST_USER_ISO=$(grep -oE "2026-[0-9-]+T[0-9:.]+Z.*Discord message received" "$BRIDGE_LOG" 2>/dev/null \
    | tail -1 | grep -oE "2026-[0-9-]+T[0-9:.]+Z" | head -1)
[ -n "$LAST_USER_ISO" ] && LAST_USER_EPOCH=$(date -d "$LAST_USER_ISO" +%s 2>/dev/null || echo 0)

# Dead-air condition: last substantive msg older than cutoff AND user not currently active
USER_RECENT_THRESHOLD=$((NOW_EPOCH - 600))  # User active in last 10 min — don't tripwire
if [ "$LAST_SUBSTANTIVE_EPOCH" -lt "$CUTOFF" ] && [ "$LAST_USER_EPOCH" -lt "$USER_RECENT_THRESHOLD" ]; then
    LAST_REAL_HUMAN=$(date -d "@$LAST_SUBSTANTIVE_EPOCH" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo NEVER)
    MIN_SINCE=$(( (NOW_EPOCH - LAST_SUBSTANTIVE_EPOCH) / 60 ))
    echo "[$TS] !!! DEAD_AIR window=${WINDOW_MIN}min last_substantive_msg=\"$LAST_REAL_HUMAN\" minutes_since=$MIN_SINCE — bot may be silent without reason" >> "$LOG"

    # HC #710 — Auto-recovery: if dead air > 120 min, inject a recovery prompt
    RECOVERY_THRESHOLD=120
    RECOVERY_LOCKFILE="/tmp/dead_air_recovery.lock"
    if [ "$MIN_SINCE" -ge "$RECOVERY_THRESHOLD" ]; then
        # Only trigger once per dead-air episode (lockfile expires after 60 min)
        if [ ! -f "$RECOVERY_LOCKFILE" ] || [ $(( NOW_EPOCH - $(stat -c %Y "$RECOVERY_LOCKFILE" 2>/dev/null || echo 0) )) -gt 3600 ]; then
            touch "$RECOVERY_LOCKFILE"
            echo "[$TS] >>> DEAD_AIR_RECOVERY triggered (${MIN_SINCE}min silent) — injecting recovery prompt" >> "$LOG"
            curl -s -m 5 -X POST http://127.0.0.1:7731/inject \
                -H 'Content-Type: application/json' \
                -d '{"message":"DEAD_AIR_RECOVERY: Bot has been silent for '"$MIN_SINCE"' minutes. Session likely hit context limits. Run /recovery skill immediately, then send a brief status update to Discord."}' \
                >> /dev/null 2>&1
        fi
    fi
else
    MIN_SINCE=$(( (NOW_EPOCH - LAST_SUBSTANTIVE_EPOCH) / 60 ))
    echo "[$TS] dead_air_check OK min_since_last_real_msg=$MIN_SINCE" >> "$LOG"
fi
