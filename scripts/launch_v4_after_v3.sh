#!/bin/bash
# Wait for v3 to finish, then launch v4 comprehensive decay analysis
set -e

V3_PID=522282
V4_SCRIPT="/home/jupiter/Lvl3Quant/alpha_discovery/deep_models/test_model_decay_v4_comprehensive.py"
V4_LOG="/home/jupiter/Lvl3Quant/output/decay_v4_log.txt"

echo "$(date): Waiting for v3 (PID $V3_PID) to finish..."

# Wait for the v3 parent process
while kill -0 $V3_PID 2>/dev/null; do
    sleep 30
done

echo "$(date): v3 finished. Launching v4 comprehensive decay analysis..."
cd /home/jupiter/Lvl3Quant
python3 "$V4_SCRIPT" --workers 6 2>&1 | tee "$V4_LOG"
echo "$(date): v4 complete."
