#!/bin/bash
# Polls Razer's front_month_guard.ps1 every cron tick. If MISMATCH found,
# fires a Discord alert via the inject endpoint.
set -uo pipefail

LOG=/home/jupiter/Lvl3Quant/logs/front_month_guard_check.log
STATE=/tmp/front_month_guard_last_alert
SSH_OPTS="-o ConnectTimeout=8 -o StrictHostKeyChecking=no -o BatchMode=no"

OUT=$(ssh $SSH_OPTS claude@razer "powershell -ExecutionPolicy Bypass -File C:\\Users\\claude\\Lvl3Quant\\scripts\\front_month_guard.ps1" 2>&1)
RC=$?

echo "[$(date)] rc=$RC" >> "$LOG"
echo "$OUT" >> "$LOG"

if [[ $RC -ne 0 ]]; then
    # SSH/script failure — log only, do not spam
    exit 0
fi

if echo "$OUT" | grep -q "MISMATCH"; then
    # debounce: only alert once per hour for the same mismatch
    HASH=$(echo "$OUT" | grep MISMATCH | sha1sum | cut -d' ' -f1)
    LAST=$(cat "$STATE" 2>/dev/null || echo "")
    NOW=$(date +%s)
    LAST_TS=$(stat -c %Y "$STATE" 2>/dev/null || echo 0)
    AGE=$((NOW - LAST_TS))
    if [[ "$HASH" != "$LAST" || $AGE -gt 3600 ]]; then
        MSG=$(echo "$OUT" | grep -E "expected|MISMATCH" | head -5)
        BODY="**⚠ Front-month guard tripped**\n\n\`\`\`\n$MSG\n\`\`\`"
        curl -s -X POST http://127.0.0.1:7731/inject \
            -H "Content-Type: application/json" \
            -d "{\"message\": $(echo "$BODY" | jq -Rs .)}" >/dev/null 2>&1
        echo "$HASH" > "$STATE"
    fi
fi
