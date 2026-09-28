#!/bin/bash
# Crash Recovery System — auto-restarts training that died on GPU nodes
# Designed to be called from autonomous_monitor.sh or crontab
# Reads training_state.json for current experiment configs
# Uses PID locking to prevent duplicate launches

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STATE_FILE="$SCRIPT_DIR/training_state.json"
LOG="/home/jupiter/Lvl3Quant/logs/crash_recovery.log"
WEBHOOK="node /home/jupiter/teleclaude-main/utils/webhook_notifier.js"

TS="[$(date '+%Y-%m-%d %H:%M:%S')]"

log() { echo "$TS $1" >> "$LOG"; }
alert() {
    log "ALERT: $1"
    $WEBHOOK "$1" 2>/dev/null || true
}

# Check if state file exists
if [ ! -f "$STATE_FILE" ]; then
    log "ERROR: $STATE_FILE not found"
    exit 1
fi

###############################################################################
# NEPTUNE CHECK
###############################################################################
recover_neptune() {
    local SSH="nick@neptune"
    local status
    status=$(python3 -c "import json; d=json.load(open('$STATE_FILE')); print(d['nodes']['neptune']['status'])" 2>/dev/null)

    if [ "$status" != "running" ]; then
        log "Neptune: status=$status, skipping"
        return
    fi

    # Check if training process is alive
    local procs
    procs=$(ssh -o ConnectTimeout=8 -o BatchMode=yes "$SSH" \
        "ps aux | grep -E 'python.*train_event_mamba' | grep -v grep | wc -l" 2>/dev/null | tr -d ' \r\n')

    if [ "${procs:-0}" -gt 0 ]; then
        log "Neptune: OK (${procs} mamba processes running)"
        return
    fi

    # Process is dead — check GPU to confirm
    local gpu
    gpu=$(ssh -o ConnectTimeout=8 -o BatchMode=yes "$SSH" \
        "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits" 2>/dev/null | tr -d ' \r\n')

    if [ "${gpu:-100}" -gt 50 ]; then
        log "Neptune: No mamba process but GPU at ${gpu}% — something else running?"
        return
    fi

    alert "🔴 Neptune: Mamba training CRASHED! GPU=${gpu:-?}%, 0 procs. Attempting restart..."

    # Read config from state file
    local config
    config=$(python3 -c "
import json
d = json.load(open('$STATE_FILE'))
n = d['nodes']['neptune']
env_str = ' '.join(f'{k}={v}' for k, v in n['env'].items())
print(f\"{env_str}|{n['python']}|{n['script']}|{n['args']}|{n['cwd']}|{n['log_file']}|{n['lock_file']}\")
" 2>/dev/null)

    if [ -z "$config" ]; then
        alert "🔴 Neptune: Failed to read config from state file!"
        return
    fi

    IFS='|' read -r ENV_STR PYTHON SCRIPT ARGS CWD LOGF LOCKF <<< "$config"

    # Check lock file for stale PID
    ssh -o ConnectTimeout=8 -o BatchMode=yes "$SSH" "
        if [ -f '$LOCKF' ]; then
            PID=\$(cat '$LOCKF')
            if kill -0 \$PID 2>/dev/null; then
                echo 'LOCKED'
                exit 0
            fi
            rm -f '$LOCKF'
        fi
        echo 'UNLOCKED'
    " 2>/dev/null | grep -q "LOCKED" && {
        log "Neptune: Lock file active, process alive despite grep miss. Skipping."
        return
    }

    # Find the latest fold to resume from
    local last_fold
    last_fold=$(ssh -o ConnectTimeout=8 -o BatchMode=yes "$SSH" \
        "ls -d $(python3 -c "import json; print(json.load(open('$STATE_FILE'))['nodes']['neptune']['output_dir'])")/fold_* 2>/dev/null | wc -l" 2>/dev/null | tr -d ' \r\n')

    log "Neptune: Relaunching from fold ${last_fold:-0}..."

    # Relaunch with nohup
    ssh -o ConnectTimeout=15 -o BatchMode=yes "$SSH" "
        cd $CWD && \
        rm -f $LOCKF && \
        $ENV_STR nohup $PYTHON -u $SCRIPT $ARGS > $LOGF 2>&1 & \
        MPID=\$!; echo \"\$MPID\" > $LOCKF; echo \"Launched PID: \$MPID\"
    " 2>/dev/null

    local rc=$?
    if [ $rc -eq 0 ]; then
        alert "🟢 Neptune: Mamba training RESTARTED successfully (was at fold ${last_fold:-?})"
    else
        alert "🔴 Neptune: Restart FAILED (exit code $rc)"
    fi
}

###############################################################################
# RAZER CHECK
###############################################################################
recover_razer() {
    local SSH="claude@razer"
    local status
    status=$(python3 -c "import json; d=json.load(open('$STATE_FILE')); print(d['nodes']['razer']['status'])" 2>/dev/null)

    if [ "$status" != "running" ]; then
        log "Razer: status=$status, skipping"
        return
    fi

    # Check if python is running on Razer
    local procs
    procs=$(ssh -o ConnectTimeout=8 -o BatchMode=yes "$SSH" \
        'tasklist /FI "IMAGENAME eq python.exe" /NH 2>nul | find /c "python"' 2>/dev/null | tr -d ' \r\n')

    if [ "${procs:-0}" -gt 0 ]; then
        log "Razer: OK (${procs} python processes running)"
        return
    fi

    # Check GPU
    local gpu
    gpu=$(ssh -o ConnectTimeout=8 -o BatchMode=yes "$SSH" \
        "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits" 2>/dev/null | tr -d ' \r\n')

    if [ "${gpu:-100}" -gt 50 ]; then
        log "Razer: No python but GPU at ${gpu}% — GPU busy without python?"
        return
    fi

    alert "🔴 Razer: Training CRASHED! GPU=${gpu:-?}%, 0 procs. Attempting restart..."

    # Read bat file path from state
    local bat_file
    bat_file=$(python3 -c "import json; print(json.load(open('$STATE_FILE'))['nodes']['razer'].get('bat_file',''))" 2>/dev/null)

    if [ -z "$bat_file" ]; then
        alert "🔴 Razer: No bat_file in state config! Cannot restart."
        return
    fi

    # Use wmic to launch detached process (proven method for Windows SSH)
    ssh -o ConnectTimeout=15 -o BatchMode=yes "$SSH" \
        "wmic process call create \"cmd /c $bat_file\"" 2>/dev/null

    local rc=$?
    if [ $rc -eq 0 ]; then
        alert "🟢 Razer: Training RESTARTED via wmic ($bat_file)"
    else
        alert "🔴 Razer: Restart FAILED (exit code $rc)"
    fi
}

###############################################################################
# MAIN
###############################################################################
log "=== Crash Recovery Check ==="
recover_neptune
recover_razer
log "=== Done ==="

# Trim log
[ -f "$LOG" ] && SIZE=$(stat -c%s "$LOG" 2>/dev/null || echo 0) && [ "${SIZE:-0}" -gt 5242880 ] && tail -500 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"
