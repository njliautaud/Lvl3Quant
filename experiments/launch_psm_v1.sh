#!/bin/bash
source /home/nick/miniconda3/etc/profile.d/conda.sh
conda activate py311-train
cd /home/nick/Lvl3Quant
nohup python3 experiments/pred_stream_momentum_v1.py > output/pred_stream_momentum_v1/train.log 2>&1 &
echo "PID=$!"
