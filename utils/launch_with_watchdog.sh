#!/bin/bash
# Launch training with mandatory logging watchdog
# Usage: ./launch_with_watchdog.sh <script> <log_file> <watchdog_minutes>
#
# Example:
#   ./launch_with_watchdog.sh "python train.py" /tmp/train.log 10

set -e

SCRIPT=$1
LOG_FILE=$2
MAX_SILENT=${3:-10}  # Default 10 minutes

if [ -z "$SCRIPT" ] || [ -z "$LOG_FILE" ]; then
    echo "Usage: $0 <script> <log_file> [max_silent_minutes]"
    exit 1
fi

# Ensure log directory exists
mkdir -p "$(dirname "$LOG_FILE")"

# Launch training in background, capturing PID
echo "[LAUNCH] Starting: $SCRIPT"
echo "[LAUNCH] Log: $LOG_FILE"
$SCRIPT > "$LOG_FILE" 2>&1 &
TRAIN_PID=$!

echo "[LAUNCH] Training PID: $TRAIN_PID"

# Launch watchdog
echo "[LAUNCH] Starting watchdog (max silent: ${MAX_SILENT} min)"
python utils/training_watchdog.py \
    --log-file "$LOG_FILE" \
    --pid $TRAIN_PID \
    --max-silent-minutes $MAX_SILENT \
    --check-interval 30 &
WATCHDOG_PID=$!

echo "[LAUNCH] Watchdog PID: $WATCHDOG_PID"
echo "[LAUNCH] Both processes running. Watchdog will kill training if log stays empty."
echo ""
echo "Monitor with:"
echo "  tail -f $LOG_FILE"
echo "  ps aux | grep $TRAIN_PID"
echo ""
echo "Kill both:"
echo "  kill $TRAIN_PID $WATCHDOG_PID"
