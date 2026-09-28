#!/bin/bash
# Autonomous training monitor - runs from system crontab every 15 min
# Self-sufficient: checks GPU, alerts on idle, auto-dispatches next experiment
# Does NOT depend on Claude session being alive

LOG="/home/jupiter/Lvl3Quant/logs/autonomous_monitor.log"
QUEUE="/home/jupiter/Lvl3Quant/data/research_queue_persistent.json"
WEBHOOK="/home/jupiter/teleclaude-main/utils/webhook_notifier.js"
EVAL_SCRIPT="/home/jupiter/Lvl3Quant/scripts/auto_evaluate.py"
DISPATCH_SCRIPT="/home/jupiter/Lvl3Quant/scripts/auto_dispatch.py"
CRASH_RECOVERY="/home/jupiter/Lvl3Quant/scripts/crash_recovery.sh"

TS="[$(date '+%Y-%m-%d %H:%M:%S')]"
echo "$TS === Autonomous Monitor ===" >> "$LOG"

alert() {
    echo "$TS ALERT: $1" >> "$LOG"
    [ -f "$WEBHOOK" ] && node "$WEBHOOK" "$1" 2>/dev/null
}

# --- Neptune GPU check ---
NEPTUNE_GPU=$(ssh -o ConnectTimeout=8 -o StrictHostKeyChecking=no nick@neptune \
    "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits" 2>/dev/null | tr -d ' \r')
NEPTUNE_PROCS=$(ssh -o ConnectTimeout=5 nick@neptune \
    "ps aux | grep -E 'python.*train' | grep -v grep | wc -l" 2>/dev/null | tr -d ' \r')
echo "$TS Neptune: GPU=${NEPTUNE_GPU:-?}% procs=${NEPTUNE_PROCS:-?}" >> "$LOG"

if [ "${NEPTUNE_GPU:-100}" -lt 10 ] && [ "${NEPTUNE_PROCS:-1}" -eq 0 ]; then
    alert "🔴 Neptune IDLE — no training process! GPU at ${NEPTUNE_GPU}%"
    # Try crash recovery first (restarts same experiment with same config)
    [ -f "$CRASH_RECOVERY" ] && bash "$CRASH_RECOVERY" >> "$LOG" 2>&1
fi

# --- Razer GPU check ---
RAZER_GPU=$(ssh -o ConnectTimeout=8 -o StrictHostKeyChecking=no claude@razer \
    "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits" 2>/dev/null | tr -d ' \r')
RAZER_PROCS=$(ssh -o ConnectTimeout=5 claude@razer \
    "tasklist /FI \"IMAGENAME eq python.exe\" /NH 2>nul | find /c \"python\"" 2>/dev/null | tr -d ' \r')
echo "$TS Razer: GPU=${RAZER_GPU:-?}% procs=${RAZER_PROCS:-?}" >> "$LOG"

if [ "${RAZER_GPU:-100}" -lt 10 ] && [ "${RAZER_PROCS:-1}" -eq 0 ]; then
    alert "🔴 Razer IDLE — no training process! GPU at ${RAZER_GPU}%"
    # Try crash recovery (restarts same experiment with same config)
    [ -f "$CRASH_RECOVERY" ] && bash "$CRASH_RECOVERY" >> "$LOG" 2>&1
fi

# --- Jupiter LGBM check ---
JUPITER_PROCS=$(ps aux | grep "python.*train" | grep -v grep | wc -l)
echo "$TS Jupiter: training_procs=$JUPITER_PROCS" >> "$LOG"

# --- Check for new completed results ---
[ -f "$EVAL_SCRIPT" ] && python3 "$EVAL_SCRIPT" --local-only >> "$LOG" 2>&1

# --- Update QCC DB with latest GPU stats ---
python3 -c "
import sqlite3
conn = sqlite3.connect('/home/jupiter/teleclaude-main/data/qcc.db')
if '${NEPTUNE_GPU}' and '${NEPTUNE_GPU}' != '?':
    conn.execute('UPDATE compute_nodes SET last_gpu_util=? WHERE name=\"neptune\"', (${NEPTUNE_GPU:-0},))
if '${RAZER_GPU}' and '${RAZER_GPU}' != '?':
    conn.execute('UPDATE compute_nodes SET last_gpu_util=? WHERE name=\"razer\"', (${RAZER_GPU:-0},))
conn.commit(); conn.close()
" 2>/dev/null

# Trim log
[ -f "$LOG" ] && SIZE=$(stat -c%s "$LOG" 2>/dev/null) && [ "${SIZE:-0}" -gt 5242880 ] && tail -500 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"

echo "$TS Done" >> "$LOG"
