#!/bin/bash
# Start Ray head node - for PM2 management
export PATH="/home/jupiter/.local/bin:$PATH"

# Check if Ray is already running
if ray status 2>/dev/null | grep -q "ALIVE"; then
    echo "Ray already running"
    exit 0
fi

ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265 2>&1
echo "Ray head started at $(date)"

# Keep process alive for PM2
while true; do
    if ! ray status 2>/dev/null | grep -q "ALIVE"; then
        echo "Ray died, restarting at $(date)"
        ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265 2>&1
    fi
    sleep 300
done
