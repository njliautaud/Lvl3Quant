#!/bin/bash
# Neptune Auto-Dispatcher — launches training when GPU has free VRAM
# Built 2026-09-26 after user called out 3 months of idle GPU
# Cron: */30 * * * * /home/jupiter/Lvl3Quant/scripts/neptune_auto_dispatcher.sh
#
# Logic:
#   1. SSH to Neptune, check VRAM usage
#   2. If <4GB used AND no training running → launch passive exec optimizer MLP
#   3. If training already running → do nothing
#   4. Log everything

set -e

LOGFILE="/home/jupiter/Lvl3Quant/logs/neptune_auto_dispatcher.log"
LOCKFILE="/tmp/neptune_training.lock"
SSH_TARGET="nick@neptune"
SSH_OPTS="-o ConnectTimeout=5 -o StrictHostKeyChecking=no -o BatchMode=yes"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [AUTO-DISPATCH] $1" >> "$LOGFILE"
}

# Check if we already know training is running
if [ -f "$LOCKFILE" ]; then
    # Verify it's still actually running
    pid=$(cat "$LOCKFILE" 2>/dev/null)
    if ssh $SSH_OPTS "$SSH_TARGET" "ps -p $pid > /dev/null 2>&1" 2>/dev/null; then
        log "Training PID $pid still running. Skipping."
        exit 0
    else
        log "Stale lock (PID $pid dead). Removing lock."
        rm -f "$LOCKFILE"
    fi
fi

# Check Neptune GPU status
GPU_INFO=$(ssh $SSH_OPTS "$SSH_TARGET" \
    "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits" 2>/dev/null)

if [ -z "$GPU_INFO" ]; then
    log "SSH to Neptune failed. Skipping."
    exit 1
fi

MEM_USED=$(echo "$GPU_INFO" | cut -d',' -f1 | tr -d ' ')
MEM_TOTAL=$(echo "$GPU_INFO" | cut -d',' -f2 | tr -d ' ')
GPU_UTIL=$(echo "$GPU_INFO" | cut -d',' -f3 | tr -d ' ')

log "Neptune GPU: ${MEM_USED}/${MEM_TOTAL} MiB, ${GPU_UTIL}% util"

# Check if there's enough VRAM (need at least 8GB free for training)
FREE_MEM=$((MEM_TOTAL - MEM_USED))
if [ "$FREE_MEM" -lt 8000 ]; then
    log "Not enough free VRAM (${FREE_MEM} MiB free, need 8000+). Desktop/gaming active. Skipping."
    exit 0
fi

# Check if any python training is already running
TRAINING_CHECK=$(ssh $SSH_OPTS "$SSH_TARGET" \
    "pgrep -f 'train_passive_exec|train_split_dqn|train_fifo_rl|train_ppo|train_sac'" 2>/dev/null || true)

if [ -n "$TRAINING_CHECK" ]; then
    log "Training process already running (PIDs: $TRAINING_CHECK). Skipping."
    exit 0
fi

# GPU is free! Launch the passive exec optimizer MLP (best result: Spearman 0.78)
log "GPU FREE (${FREE_MEM} MiB available). Launching passive exec optimizer MLP..."

# Launch training on Neptune via SSH
LAUNCH_CMD="cd /home/nick/Lvl3Quant && \
    PYTHONPATH=/home/nick/Lvl3Quant:\$PYTHONPATH \
    MLFLOW_TRACKING_URI=http://jupiter:5000 \
    nohup /home/nick/miniconda3/envs/py311-train/bin/python \
    scripts/train_passive_exec_optimizer_mlp.py \
    > /home/nick/Lvl3Quant/logs/passive_exec_mlp_auto.log 2>&1 & echo \$!"

REMOTE_PID=$(ssh $SSH_OPTS "$SSH_TARGET" "$LAUNCH_CMD" 2>/dev/null)

if [ -n "$REMOTE_PID" ] && [ "$REMOTE_PID" -gt 0 ] 2>/dev/null; then
    echo "$REMOTE_PID" > "$LOCKFILE"
    log "SUCCESS: Launched passive exec optimizer MLP on Neptune (PID: $REMOTE_PID)"

    # Notify via Discord webhook if available
    if [ -f /home/jupiter/Lvl3Quant/scripts/discord_notify.sh ]; then
        /home/jupiter/Lvl3Quant/scripts/discord_notify.sh \
            "Auto-dispatcher launched passive exec optimizer MLP on Neptune (GPU had ${FREE_MEM}MB free)" 2>/dev/null || true
    fi
else
    log "FAILED: Could not launch training on Neptune. Output: $REMOTE_PID"
fi
