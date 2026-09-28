#!/bin/bash
# Safe CPU wrapper for Neptune — protects Mamba GPU training
# Usage: ./safe_cpu_wrapper.sh <command> [args...]
#
# Enforces:
#   - nice 19 (lowest CPU priority — Mamba always gets CPU first)
#   - max 4 cores via taskset (cores 16-19, away from Mamba's cores)
#   - memory limit via ulimit (max 6GB virtual memory)
#   - IO scheduling: idle class (Mamba gets disk priority)

set -e

if [ $# -eq 0 ]; then
    echo "Usage: $0 <command> [args...]"
    echo "Runs command with low priority to protect Mamba training"
    exit 1
fi

# Check Mamba is still running
MAMBA_PID=$(pgrep -f "train_event_mamba" || true)
if [ -n "$MAMBA_PID" ]; then
    echo "[safe_cpu] Mamba detected (PID $MAMBA_PID) — applying resource limits"
else
    echo "[safe_cpu] No Mamba detected — running with light limits only"
fi

# Check available memory before starting
AVAIL_MB=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
echo "[safe_cpu] Available memory: ${AVAIL_MB}MB"

if [ "$AVAIL_MB" -lt 4000 ]; then
    echo "[safe_cpu] ERROR: Less than 4GB available — refusing to start CPU work"
    echo "[safe_cpu] Mamba needs memory headroom. Aborting."
    exit 1
fi

if [ "$AVAIL_MB" -lt 8000 ]; then
    echo "[safe_cpu] WARNING: Less than 8GB available — using conservative 3GB limit"
    MEM_LIMIT=$((3 * 1024 * 1024))  # 3GB in KB for ulimit
else
    MEM_LIMIT=$((6 * 1024 * 1024))  # 6GB in KB for ulimit
fi

# Set memory limit (virtual memory, in KB)
ulimit -v $MEM_LIMIT 2>/dev/null || echo "[safe_cpu] Could not set memory limit"

# Run with:
#   nice 19 = lowest CPU priority
#   taskset 0xF0000 = cores 16-19 only (4 cores, away from core 0-15)
#   ionice -c3 = idle IO class
echo "[safe_cpu] Launching: $@"
echo "[safe_cpu] nice=19, cores=16-19, ionice=idle, mem_limit=${MEM_LIMIT}KB"

exec nice -n 19 ionice -c3 taskset -c 16-19 "$@"
