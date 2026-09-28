#!/bin/bash
cd /home/nick/Lvl3Quant
# Kill old training (exclude this script)
ps aux | grep train_split_dqn.py | grep -v grep | awk '{print $2}' | xargs kill 2>/dev/null
sleep 2
exec /home/nick/miniconda3/envs/py311-train/bin/python alpha_discovery/execution/train_split_dqn.py   --data-dir /home/nick/Lvl3Quant/data/processed/mbo_events_smart_v3   --pred-dir /home/nick/Lvl3Quant/output/cnn_mamba_v2_smart_v3_mar   --patchtst-dir /home/nick/Lvl3Quant/output/patchtst_cache   --output-dir /home/nick/Lvl3Quant/output/split_dqn_v2_perhead_r1   --n-step 20   --epochs 50
