#!/bin/bash
# Mamba training monitor - runs every 35min from crontab
# Checks Neptune GPU and alerts if idle (training died)

LOG="/home/jupiter/Lvl3Quant/logs/mamba_monitor.log"
WEBHOOK_SCRIPT="/home/jupiter/teleclaude-main/utils/webhook_notifier.js"

echo "[$(date)] Mamba monitor check" >> "$LOG"

# Check Neptune GPU via SSH
GPU_UTIL=$(ssh -o ConnectTimeout=8 -o StrictHostKeyChecking=no nick@neptune \
  "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits" 2>/dev/null | tr -d ' ')

if [ -z "$GPU_UTIL" ]; then
    echo "  Neptune SSH failed" >> "$LOG"
    exit 0
fi

echo "  Neptune GPU: ${GPU_UTIL}%" >> "$LOG"

if [ "$GPU_UTIL" -lt 10 ] 2>/dev/null; then
    PROCS=$(ssh -o ConnectTimeout=8 nick@neptune "ps aux | grep -E 'train.*python|python.*train' | grep -v grep | wc -l" 2>/dev/null)
    echo "  Training processes: $PROCS" >> "$LOG"
    if [ "${PROCS:-0}" -eq 0 ]; then
        echo "  ALERT: Neptune idle, no training!" >> "$LOG"
        # Send webhook alert if notifier exists
        if [ -f "$WEBHOOK_SCRIPT" ]; then
            node "$WEBHOOK_SCRIPT" "⚠️ Neptune GPU IDLE — no training process found. Mamba may have crashed." 2>/dev/null
        fi
    fi
fi

# Trim log
if [ -f "$LOG" ] && [ "$(stat -c%s "$LOG" 2>/dev/null || echo 0)" -gt 2097152 ]; then
    tail -200 "$LOG" > "${LOG}.tmp" && mv "${LOG}.tmp" "$LOG"
fi
