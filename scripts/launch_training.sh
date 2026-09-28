#!/bin/bash
# Safe training launcher with PID-file locking
# Prevents double-launch from SSH timeouts
# Usage: ./launch_training.sh <lock_name> <command...>
# Example: ./launch_training.sh mamba_training python -u train.py --args

set -e

LOCK_DIR="/home/nick/Lvl3Quant/locks"
mkdir -p "$LOCK_DIR"

LOCK_NAME="$1"
shift

if [ -z "$LOCK_NAME" ]; then
    echo "Usage: $0 <lock_name> <command...>"
    exit 1
fi

LOCKFILE="$LOCK_DIR/${LOCK_NAME}.lock"

# Check for existing lock
if [ -f "$LOCKFILE" ]; then
    OLD_PID=$(cat "$LOCKFILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "ABORT: $LOCK_NAME already running (PID $OLD_PID)"
        echo "  To force: rm $LOCKFILE && $0 $LOCK_NAME $@"
        exit 1
    else
        echo "Cleaning stale lock (PID $OLD_PID dead)"
        rm -f "$LOCKFILE"
    fi
fi

# Launch the command
echo "Launching: $@"
"$@" &
CHILD_PID=$!
echo "$CHILD_PID" > "$LOCKFILE"
echo "Started $LOCK_NAME (PID $CHILD_PID), lock: $LOCKFILE"

# Wait for child and clean up
wait $CHILD_PID
EXIT_CODE=$?
rm -f "$LOCKFILE"
echo "$LOCK_NAME finished (exit code $EXIT_CODE), lock released"
exit $EXIT_CODE
