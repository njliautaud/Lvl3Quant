#!/bin/bash
# v4_completion_watcher.sh — polls Neptune for V4 multihead training completion
# When all 176 folds have predictions, auto-triggers v4_completion_eval.py
# Run: nohup bash scripts/v4_completion_watcher.sh &
set -u

NEPTUNE="nick@neptune"
REMOTE_DIR="/home/nick/Lvl3Quant/output/v4_multihead_pressure_v1"
EVAL_SCRIPT="/home/jupiter/Lvl3Quant/scripts/v4_completion_eval.py"
INJECT="http://127.0.0.1:7731/inject"
LOG="/home/jupiter/Lvl3Quant/logs/v4_completion_watcher.log"
TARGET_FOLDS=50  # folds 126-175 = 50 total predictions
CHECK_INTERVAL=300  # 5 minutes

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }

log "V4 completion watcher started. Target: $TARGET_FOLDS folds."

while true; do
    # Count prediction files on Neptune
    count=$(ssh -o ConnectTimeout=10 "$NEPTUNE" \
        "ls ${REMOTE_DIR}/fold_*_oot_predictions.npz 2>/dev/null | wc -l" 2>/dev/null)

    if [ -z "$count" ]; then
        log "SSH to Neptune failed, retrying in ${CHECK_INTERVAL}s"
        sleep "$CHECK_INTERVAL"
        continue
    fi

    log "Folds completed: ${count}/${TARGET_FOLDS}"

    # Check if training process is still running
    pid_alive=$(ssh -o ConnectTimeout=10 "$NEPTUNE" \
        "ps aux | grep train_v4_multihead | grep -v grep | wc -l" 2>/dev/null)

    if [ "$count" -ge "$TARGET_FOLDS" ]; then
        log "ALL $TARGET_FOLDS FOLDS COMPLETE! Launching evaluation..."

        # Run the completion eval
        cd /home/jupiter/Lvl3Quant
        python3 "$EVAL_SCRIPT" --auto-launch 2>&1 | tee -a "$LOG"

        # Inject result to Claude session
        msg="V4 multihead training COMPLETE ($TARGET_FOLDS folds). Evaluation results saved. Check output/v4_completion_eval/ for IC analysis."
        curl -s -m 5 -X POST "$INJECT" -H 'Content-Type: application/json' \
            -d "{\"message\":\"$msg\"}" >> "$LOG" 2>&1

        log "Evaluation complete. Watcher exiting."
        exit 0
    fi

    # If training process died but folds < target, alert
    if [ "$pid_alive" = "0" ] && [ "$count" -lt "$TARGET_FOLDS" ]; then
        log "WARNING: Training process dead but only $count/$TARGET_FOLDS folds. Alerting..."
        msg="V4 multihead training DIED at fold $count/$TARGET_FOLDS. Process not running on Neptune. Investigate and relaunch."
        curl -s -m 5 -X POST "$INJECT" -H 'Content-Type: application/json' \
            -d "{\"message\":\"$msg\"}" >> "$LOG" 2>&1
        log "Alert sent. Watcher exiting."
        exit 1
    fi

    sleep "$CHECK_INTERVAL"
done
