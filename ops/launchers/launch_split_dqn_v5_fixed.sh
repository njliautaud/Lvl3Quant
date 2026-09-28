#!/bin/bash
set -e
cd /home/nick/Lvl3Quant
source /home/nick/miniconda3/etc/profile.d/conda.sh
conda activate py311-train
export PYTHONPATH=/home/nick/Lvl3Quant

echo "[$(date)] Waiting for precompute to finish..."
while pgrep -f precompute_observations > /dev/null 2>&1; do
    OBS_NOW=$(ls data/precomputed_obs_v342/*.npy 2>/dev/null | wc -l)
    echo "[$(date)] Precompute still running... ($OBS_NOW .npy files done)"
    sleep 60
done

OBS_COUNT=$(ls data/precomputed_obs_v342/*.npy 2>/dev/null | wc -l)
echo "[$(date)] Precompute complete. $OBS_COUNT observation files ready."

if [ "$OBS_COUNT" -lt 30 ]; then
    echo "ERROR: Only $OBS_COUNT obs files. Need at least 30."
    exit 1
fi

echo "[$(date)] Launching Split DQN v5 training on GPU..."
MLFLOW_TRACKING_URI=http://jupiter:5000 python alpha_discovery/execution/train_split_dqn_v5.py \
    --data-dir /home/nick/Lvl3Quant/data/processed/mbo_events_smart_v3 \
    --pred-dir /home/nick/Lvl3Quant/output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_env_format/ \
    --patchtst-dir /home/nick/Lvl3Quant/output/patchtst_smart_v3_mar/ \
    --precomputed-dir /home/nick/Lvl3Quant/data/precomputed_obs_v342/ \
    --output-dir /home/nick/Lvl3Quant/output/split_dqn_v5_v342/ \
    --experiment-name split_dqn_v5_v342 \
    --wall-cap 600 \
    --epochs 12 \
    --train-days 25 \
    --eval-days 3 \
    --hidden-dim 128 \
    --lr 1e-4 \
    --batch-size 4096 \
    --n-step 20 \
    --gpu-updates-per-sec 200 \
    2>&1 | tee logs/split_dqn_v5_v342.log

echo "[$(date)] Split DQN v5 training complete."
