#!/bin/bash
# Split DQN v5 Launch Script — waits for precompute then trains
set -e

cd /home/nick/Lvl3Quant
source /home/nick/miniconda3/etc/profile.d/conda.sh
conda activate py311-train
export PYTHONPATH=/home/nick/Lvl3Quant

# Wait for precompute to finish
echo "[Tue Jun 16 12:17:29 AM EDT 2026] Waiting for precompute to finish..."
while pgrep -f precompute_observations > /dev/null 2>&1; do
    sleep 30
    echo "[Tue Jun 16 12:17:29 AM EDT 2026] Precompute still running... (0 files done)"
done
echo "[Tue Jun 16 12:17:29 AM EDT 2026] Precompute complete. 0 observation files ready."

# Verify we have enough data
OBS_COUNT=0
if [ "" -lt 30 ]; then
    echo "ERROR: Only  obs files. Need at least 30 for meaningful training."
    exit 1
fi

echo "[Tue Jun 16 12:17:29 AM EDT 2026] Launching Split DQN v5 training on GPU..."

# Phase 3: Train
MLFLOW_TRACKING_URI=http://jupiter:5000 python alpha_discovery/execution/train_split_dqn_v5.py     --data-dir /home/nick/Lvl3Quant/data/processed/mbo_events_smart_v3     --pred-dir /home/nick/Lvl3Quant/output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_env_format/     --patchtst-dir /home/nick/Lvl3Quant/output/patchtst_smart_v3_mar/     --precomputed-dir /home/nick/Lvl3Quant/data/precomputed_obs_v342/     --output-dir /home/nick/Lvl3Quant/output/split_dqn_v5_v342/     --experiment-name split_dqn_v5_v342     --wall-cap 600     --epochs 12     --train-days 25     --eval-days 3     --hidden-dim 128     --lr 1e-4     --batch-size 4096     --n-step 20     --gpu-updates-per-sec 200     2>&1 | tee logs/split_dqn_v5_v342.log

echo "[Tue Jun 16 12:17:29 AM EDT 2026] Split DQN v5 training complete."
