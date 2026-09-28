#!/bin/bash
# Wait for supervised_exec_v2 (PID 2837913) to finish, then launch fill timing model
set -e

V2_PID=2837913
LOG="/home/nick/Lvl3Quant/output/fill_timing_v1.log"

echo "[$(date)] Waiting for supervised_exec_v2 (PID $V2_PID) to finish..."

# Wait for v2 to complete
while kill -0 $V2_PID 2>/dev/null; do
    sleep 30
done

echo "[$(date)] supervised_exec_v2 finished. Launching fill timing model in 10s..."
sleep 10

# Activate conda and launch
source /home/nick/miniconda3/etc/profile.d/conda.sh
conda activate py311-train
cd /home/nick/Lvl3Quant

echo "[$(date)] Launching fill_timing_model.py..."
exec python experiments/fill_timing_model.py \
    --n-train-days 60 \
    --n-eval-days 5 \
    --epochs 30 \
    --batch-size 4096 \
    --lr 1e-3 \
    --hidden-dims 256 128 64 32 \
    --dropout 0.3
