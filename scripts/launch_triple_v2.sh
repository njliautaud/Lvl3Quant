#!/bin/bash
cd /home/nick/Lvl3Quant/alpha_discovery/deep_models
export MAMBA_FEATURE_SET=smart_v4_book
export EVENT_WINDOW_SIZE=1000
export EVENT_STRIDE=250
export EVENT_NUM_WORKERS=4
export ENABLE_MFE_MAE=1
export DECAY_HALFLIFE_DAYS=15
export DISABLE_MLFLOW=1
export WF_WINDOW_DAYS=60

nohup /home/nick/miniconda3/envs/py311-train/bin/python -u \
    train_triple_fusion.py \
    --output-dir /home/nick/Lvl3Quant/output/triple_fusion_v1_smart_v4_book_mar \
    --window-mode sliding \
    --train-days 60 \
    --oot-days 1 \
    --oot-start-date 2026-03-01 \
    --n-folds 13 \
    > /home/nick/Lvl3Quant/logs/triple_fusion_v1_smart_v4_book_mar.log 2>&1 &

echo "PID=$!"
