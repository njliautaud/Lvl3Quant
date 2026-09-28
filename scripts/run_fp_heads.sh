#!/bin/bash
cd /home/nick/Lvl3Quant
/home/nick/miniconda3/envs/py311-train/bin/python3 -u scripts/train_firstpassage_heads.py > output/direct_firstpassage_heads_v1/train.log 2>&1
echo "EXIT_CODE=$?" >> output/direct_firstpassage_heads_v1/train.log
