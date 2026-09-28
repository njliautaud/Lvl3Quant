#!/bin/bash
cd /home/jupiter/Lvl3Quant/alpha_discovery/deep_models
python3 precompute_tensors.py --window-size 1000 --stride 500 --output-dir /home/jupiter/Lvl3Quant/data/processed/mbo_tensors_mamba > /home/jupiter/tensors_mamba.log 2>&1
echo "Mamba tensors DONE exit=$?" >> /home/jupiter/tensors_mamba.log
