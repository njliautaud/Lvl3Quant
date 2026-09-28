#!/bin/bash
cd /home/nick/Lvl3Quant
export MAMBA_D_MODEL=96
export MAMBA_D_STATE=32
export MAMBA_N_LAYERS=3
export MAMBA_EPOCHS=5
export EVENT_BATCH_SIZE=128
export EVENT_WINDOW_SIZE=1000
export EVENT_STRIDE=250
export EVENT_NUM_WORKERS=0
export DISABLE_MLFLOW=1
export WF_WINDOW_DAYS=60

OUTDIR=output/fusion_v3_warmfix_20260429
mkdir -p 

exec /home/nick/miniconda3/envs/py311-train/bin/python3   alpha_discovery/deep_models/train_cnn_patchtst_mamba.py   --n-folds 9 --train-days 60 --oot-days 1   --smart-data-dir data/processed/mbo_events_smart_v4   --raw-data-dir data/processed/mbo_events   --vol-pred-dir output/vol_lgbm_v3   --output-dir    --warm-start-cnn-mamba output/cnn_mamba_v2_smart_v3_mar/fold_10_best.pt
