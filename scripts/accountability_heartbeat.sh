#!/bin/bash
# HC #461 R2 — Hourly accountability heartbeat.
# Writes one compact line to /home/jupiter/Lvl3Quant/logs/accountability.log proving work was done.
# Survives Claude crashes — runs from OS cron independent of teleclaude session state.
set -u

LOG=/home/jupiter/Lvl3Quant/logs/accountability.log
mkdir -p "$(dirname "$LOG")"

TS=$(date '+%Y-%m-%d %H:%M:%S %Z')

# Count cron-injected queries in the last hour (logged by lib/discord.js patch)
CRON_CYCLES_LAST_HR=$(grep -c "CRON_QUERY_START" "$LOG" 2>/dev/null | head -1 || echo 0)
# Last substantive bot message timestamp from bridge log (anything that's NOT a Still-working / Session-recovered boilerplate)
BRIDGE_LOG="/home/jupiter/teleclaude-main/logs/bridge-$(date '+%Y-%m-%d').log"
LAST_REAL_MSG="never"
if [ -f "$BRIDGE_LOG" ]; then
    LAST_REAL_MSG=$(grep -E "Sending Discord message" "$BRIDGE_LOG" 2>/dev/null | tail -1 | awk -F'[][]' '{print $2}' || echo "never")
fi

# Quick node status snapshot (no SSH — uses qcc heartbeat file if present, else "unknown")
NODE_STATUS="unknown"
QCC_DB=/home/jupiter/.qcc/state.json
if [ -f "$QCC_DB" ]; then
    NODE_STATUS=$(python3 -c "
import json, sys
try:
    d = json.load(open('$QCC_DB'))
    n = d.get('nodes', {})
    parts = []
    for name in ['neptune', 'jupiter', 'razer', 'saturn']:
        node = n.get(name, {})
        st = node.get('status', '?')
        gpu = node.get('gpu_util', '?')
        parts.append(f'{name}:{st}/{gpu}')
    print(' '.join(parts))
except Exception as e:
    print('parse_err:' + str(e)[:40])
" 2>/dev/null || echo "qcc_read_err")
fi

# Active training processes on Neptune (best-effort via SSH, short timeout)
NEP_PROC="?"
NEP_PROC=$(timeout 5 ssh -o ConnectTimeout=3 -o StrictHostKeyChecking=no nick@neptune \
    "ps -eo pid,etime,rss,cmd --sort=-rss 2>/dev/null | grep -E 'python|cnn_mamba' | grep -v grep | head -1 | awk '{print \$1\":\"\$2}'" 2>/dev/null || echo "ssh_fail")
[ -z "$NEP_PROC" ] && NEP_PROC="none"

echo "[$TS] HEARTBEAT cron_cycles_last_hr=$CRON_CYCLES_LAST_HR | last_real_msg=$LAST_REAL_MSG | nodes=$NODE_STATUS | nep_top_proc=$NEP_PROC" >> "$LOG"

# Token usage (best-effort — only if claude_usage MCP is reachable via CLI helper, otherwise skip)
TOKENS_TODAY="unknown"
if command -v claude >/dev/null 2>&1; then
    # Optional: extend later via API call. Placeholder for now.
    :
fi
echo "[$TS] TOKENS today=$TOKENS_TODAY" >> "$LOG"

# Self-trim accountability log if >5MB
SIZE=$(stat -c%s "$LOG" 2>/dev/null || echo 0)
if [ "$SIZE" -gt 5242880 ]; then
    tail -c 2097152 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"
fi
