#!/bin/bash
# Fast recovery snapshot — outputs ONLY what the next session needs
# Called by autonomy_inject on recovery. Replaces reading 44K-token state files.
# Built 2026-09-26 to fix the "2hr context reset burns tokens re-reading" problem.

set -e
echo "=== FAST RECOVERY $(date '+%Y-%m-%d %H:%M:%S %Z') ==="

# 1. Current state (first 15 lines of SESSION_STATE only)
echo ""
echo "--- STATE (latest entry) ---"
head -15 /home/jupiter/Lvl3Quant/SESSION_STATE.md

# 2. Account (cached — updated by position check cron)
echo ""
echo "--- ACCOUNT ---"
if [ -f /home/jupiter/Lvl3Quant/state/account_snapshot.json ]; then
    python3 -c "import json; d=json.load(open('/home/jupiter/Lvl3Quant/state/account_snapshot.json')); print(f'Balance: \${d.get(\"total\",\"?\")}, Positions: {d.get(\"positions\",0)}')" 2>/dev/null || echo "Flat, ~\$391"
else
    echo "Flat, ~\$391 (no snapshot file)"
fi

# 3. Cluster health (fast — just ping)
echo ""
echo "--- CLUSTER ---"
for node in "jupiter:localhost" "neptune:neptune"; do
    name=${node%%:*}
    ip=${node##*:}
    if ping -c1 -W2 "$ip" >/dev/null 2>&1; then
        echo "  $name: UP"
    else
        echo "  $name: DOWN"
    fi
done

# 4. Neptune GPU (1 line)
echo ""
echo "--- NEPTUNE GPU ---"
ssh -o ConnectTimeout=3 -o StrictHostKeyChecking=no nick@neptune \
    "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader" 2>/dev/null \
    || echo "SSH failed"

# 5. Crontab health (count only)
echo ""
echo "--- CRONS ---"
count=$(crontab -l 2>/dev/null | grep -c 'autonomy_inject\|watchdog')
echo "  Autonomy+watchdog entries: $count"

# 6. Market status
echo ""
echo "--- MARKET ---"
day=$(date +%u)  # 1=Mon, 7=Sun
hour=$(date +%H)
if [ "$day" -ge 6 ]; then
    echo "  Weekend — closed"
elif [ "$hour" -lt 9 ] || [ "$hour" -ge 16 ]; then
    echo "  After hours"
else
    echo "  MARKET OPEN"
fi

# 7. Last user message (from state file)
echo ""
echo "--- LAST USER ---"
grep "LAST USER" /home/jupiter/Lvl3Quant/SESSION_STATE.md | head -1

echo ""
echo "=== END FAST RECOVERY ==="
