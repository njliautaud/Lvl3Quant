#!/bin/bash
source /home/nick/miniconda3/bin/activate py311-train
cd /home/nick/Lvl3Quant
export PYTHONUNBUFFERED=1
python alpha_discovery/deep_models/train_event_transformer_fast.py --train-window-days 30 2>&1 | tee output/event_transformer_w1000_sliding.log
echo "EXIT CODE: $?" >> output/event_transformer_w1000_sliding.log
