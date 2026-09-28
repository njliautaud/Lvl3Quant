#!/bin/bash
cd /home/nick/Lvl3Quant
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export MLFLOW_TRACKING_URI=""
export PYTHONFAULTHANDLER=1
mkdir -p output/fifo_ppo_v5_multimodel
/home/nick/miniconda3/envs/py311-train/bin/python -u   alpha_discovery/execution/train_fifo_rl.py   --data-dir /home/nick/Lvl3Quant/data/processed/mbo_events_smart_v3   --pred-dir /home/nick/Lvl3Quant/output/cnn_mamba_v2_bulk_oot   --patchtst-dir /home/nick/Lvl3Quant/output/patchtst_bulk_oot   --output-dir /home/nick/Lvl3Quant/output/fifo_ppo_v5_multimodel   --hidden-dim 128   --epochs 10   --train-days 35   --eval-days 5   --lr 3e-4   --rollout-len 8192   --mini-batch 1024   --entropy-coef 0.02
