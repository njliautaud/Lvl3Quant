#!/bin/bash
export MLFLOW_TRACKING_URI=http://jupiter:5000
export MAMBA_D_MODEL=192
export MAMBA_D_STATE=96
export MAMBA_N_LAYERS=6
export EVENT_STRIDE=2500
export EVENT_BATCH_SIZE=64

cd /home/nick/Lvl3Quant/alpha_discovery/deep_models

nohup python train_event_mamba_cuda.py \
  --data-dir /home/nick/Lvl3Quant/data/processed/mbo_events \
  --output-dir results/event_mamba_v3_bs64 \
  --n-folds 5 \
  --device cuda \
  > results/mamba_v3_bs64.log 2>&1 &

echo "PID=$!"
