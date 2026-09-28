#!/bin/bash
# Training PID-file locking mechanism
# Prevents double-launches from SSH timeouts
# Usage: source training_lock.sh; acquire_lock "mamba_training"

LOCK_DIR="/home/nick/Lvl3Quant/locks"
mkdir -p "$LOCK_DIR"

acquire_lock() {
    local name="$1"
    local lockfile="$LOCK_DIR/${name}.lock"
    
    if [ -f "$lockfile" ]; then
        local old_pid=$(cat "$lockfile")
        if kill -0 "$old_pid" 2>/dev/null; then
            echo "ERROR: $name already running (PID $old_pid). Aborting."
            echo "If stale, run: rm $lockfile"
            return 1
        else
            echo "WARNING: Stale lock found (PID $old_pid dead). Cleaning up."
            rm -f "$lockfile"
        fi
    fi
    
    echo $$ > "$lockfile"
    echo "Lock acquired: $name (PID $$)"
    
    # Set trap to clean up lock on exit
    trap "rm -f '$lockfile'; echo 'Lock released: $name'" EXIT INT TERM
    return 0
}

release_lock() {
    local name="$1"
    local lockfile="$LOCK_DIR/${name}.lock"
    rm -f "$lockfile"
    echo "Lock released: $name"
}

check_lock() {
    local name="$1"
    local lockfile="$LOCK_DIR/${name}.lock"
    
    if [ -f "$lockfile" ]; then
        local pid=$(cat "$lockfile")
        if kill -0 "$pid" 2>/dev/null; then
            echo "LOCKED: $name running (PID $pid)"
            return 0
        else
            echo "STALE: $name lock exists but PID $pid is dead"
            return 2
        fi
    else
        echo "FREE: $name is not locked"
        return 1
    fi
}
