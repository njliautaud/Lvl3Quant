#!/bin/bash
# Launch SAC RL on Neptune GPU after CNN-Mamba WF completes
# HC #117: SAC alongside PPO, HC #111: PatchTST confluence, HC #116: best RL for Monday

source /home/nick/miniconda3/bin/activate py311-train
export MLFLOW_TRACKING_URI="http://jupiter:5000"
export CUDA_VISIBLE_DEVICES=0

cd /home/nick/Lvl3Quant/alpha_discovery/execution

mkdir -p /home/nick/Lvl3Quant/output/fifo_sac_rl_neptune_v1

# Different config from Razer: larger buffer, more warmup, lower lr, PatchTST confluence
nohup python train_fifo_rl_sac.py \
    --epochs 80 \
    --train-days 40 \
    --eval-days 5 \
    --lr 1e-4 \
    --hidden-dim 256 \
    --buffer-size 1000000 \
    --batch-size 512 \
    --warmup-steps 2000 \
    --updates-per-step 2 \
    --data-dir /home/nick/Lvl3Quant/data/processed/mbo_events_smart_v3 \
    --pred-dir /home/nick/Lvl3Quant/output/cnn_mamba_v2_smart_v3_mar \
    --patchtst-dir /home/nick/Lvl3Quant/output/patchtst_smart_v3_mar \
    --output-dir /home/nick/Lvl3Quant/output/fifo_sac_rl_neptune_v1 \
    > /home/nick/Lvl3Quant/output/fifo_sac_rl_neptune_v1/sac_launch.log 2>&1 &

echo "SAC PID: $!"
echo "Log: /home/nick/Lvl3Quant/output/fifo_sac_rl_neptune_v1/sac_launch.log"
