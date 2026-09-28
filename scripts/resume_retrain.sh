#!/bin/bash
source /home/nick/miniconda3/etc/profile.d/conda.sh
conda activate py311-train

cd /home/nick/Lvl3Quant/alpha_discovery/deep_models
export START_FOLD=10
export MAMBA_FEATURE_SET=smart_v3
export WF_WINDOW_DAYS=60
export EVENT_BATCH_SIZE=128
export EVENT_WINDOW_SIZE=1000
export EVENT_STRIDE=500
export MAMBA_D_MODEL=96
export MAMBA_D_STATE=32
export MAMBA_N_LAYERS=3
export MAMBA_EPOCHS=5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MLFLOW_TRACKING_URI=file:///home/nick/Lvl3Quant/mlflow_local
python3 -u train_cnn_mamba.py   --data-dir /home/nick/Lvl3Quant/data/processed/mbo_events_smart_v3   --output-dir /home/nick/Lvl3Quant/output/cnn_mamba_v2_smart_v3_mar   --n-folds 25   --device cuda   2>&1 | tee /home/nick/Lvl3Quant/output/cnn_mamba_v2_retrain_fold10plus.log
