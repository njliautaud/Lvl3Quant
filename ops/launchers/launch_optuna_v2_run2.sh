#!/bin/bash
cd /home/nick/Lvl3Quant
source /home/nick/miniconda3/bin/activate py311-train
nohup python -u alpha_discovery/deep_models/optuna_exec_mlp_v2.py --n-trials 100 > /home/nick/Lvl3Quant/output/optuna_exec_mlp_v2_run2.log 2>&1 &
echo "PID: "
