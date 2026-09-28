#!/bin/bash
cd /home/nick/Lvl3Quant/wheel_strategy_v1
mkdir -p output/regime_predictor
/home/nick/ray-env/bin/python3 regime_predictor_train.py >> output/regime_predictor_train.log 2>&1
