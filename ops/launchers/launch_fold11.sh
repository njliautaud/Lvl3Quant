#!/bin/bash
source /home/nick/miniconda3/etc/profile.d/conda.sh
conda activate py311-train
export START_FOLD=11
export MAMBA_FEATURE_SET=smart_v3
export MLFLOW_TRACKING_URI=file:///home/nick/Lvl3Quant/mlruns
cd /home/nick/Lvl3Quant
exec python3 -u alpha_discovery/deep_models/train_cnn_mamba.py --data-dir /home/nick/Lvl3Quant/data/processed/mbo_events_smart_v3 --output-dir /home/nick/Lvl3Quant/output/cnn_mamba_v2_smart_v3_mar --n-folds 25 --device cuda
