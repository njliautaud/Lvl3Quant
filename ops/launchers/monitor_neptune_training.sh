#!/bin/bash
# Continuous monitoring of Neptune Split DQN training (logs-only)
# HC #194: Monitor by logs, not GPU%

LOG_FILE="/home/nick/Lvl3Quant/split_dqn_v1_r15_proper_wf.log"
LAST_LINE_COUNT=0
STALL_MINUTES=5
CHECK_INTERVAL=60

echo "=== Neptune Split DQN Monitoring (Logs-Only) ==="
echo "Monitoring: $LOG_FILE"
echo "Check interval: ${CHECK_INTERVAL}s"
echo "Stall detection: >${STALL_MINUTES} min no progress"
echo ""

while true; do
  TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")

  # Check if process is running
  PROC_COUNT=$(ssh -o ConnectTimeout=5 nick@neptune "ps aux | grep -i split_dqn | grep -v grep | wc -l" 2>/dev/null || echo "0")

  if [ "$PROC_COUNT" -eq "0" ]; then
    echo "[$TIMESTAMP] 🔴 TRAINING STOPPED (process not found)"
    break
  fi

  # Check log file growth
  CURRENT_LINE_COUNT=$(ssh -o ConnectTimeout=5 nick@neptune "wc -l < $LOG_FILE" 2>/dev/null || echo "0")

  if [ "$CURRENT_LINE_COUNT" -eq "0" ]; then
    echo "[$TIMESTAMP] ⚠️  Cannot read log"
    sleep $CHECK_INTERVAL
    continue
  fi

  # Check for progress
  LATEST_LOG=$(ssh -o ConnectTimeout=5 nick@neptune "tail -20 $LOG_FILE" 2>/dev/null)

  # Look for training indicators
  if echo "$LATEST_LOG" | grep -q "Epoch"; then
    EPOCH=$(echo "$LATEST_LOG" | grep Epoch | tail -1 | grep -oP 'Epoch \K[0-9]+' || echo "?")
    FOLD=$(echo "$LATEST_LOG" | grep -oP 'FOLD \K[0-9]+' || echo "?")
    echo "[$TIMESTAMP] ✅ Training: Fold $FOLD Epoch $EPOCH | Log lines: $CURRENT_LINE_COUNT"
    LAST_LINE_COUNT=$CURRENT_LINE_COUNT
  elif echo "$LATEST_LOG" | grep -q "loss\|update"; then
    echo "[$TIMESTAMP] ✅ Training active | Log lines: $CURRENT_LINE_COUNT"
    LAST_LINE_COUNT=$CURRENT_LINE_COUNT
  else
    # Check if log is growing
    if [ "$CURRENT_LINE_COUNT" -gt "$LAST_LINE_COUNT" ]; then
      echo "[$TIMESTAMP] ✅ Log growing | Lines: $LAST_LINE_COUNT → $CURRENT_LINE_COUNT"
      LAST_LINE_COUNT=$CURRENT_LINE_COUNT
    else
      echo "[$TIMESTAMP] ⚠️  Log not growing (potential stall)"
    fi
  fi

  # Check for errors
  if echo "$LATEST_LOG" | grep -qi "error\|exception\|traceback"; then
    echo "[$TIMESTAMP] 🔴 ERROR DETECTED in log:"
    echo "$LATEST_LOG" | grep -i "error\|exception" | tail -3
  fi

  sleep $CHECK_INTERVAL
done
