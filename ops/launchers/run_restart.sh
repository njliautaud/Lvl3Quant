#!/bin/bash
LOCK=/tmp/cnn_mamba_training.lock
if [ -f "" ]; then echo "Already running"; exit 0; fi
touch ""
trap "rm -f " EXIT
source /home/nick/miniconda3/bin/activate py311-train
cd /home/nick/Lvl3Quant
export MAMBA_FEATURE_SET=smart_v3
export START_FOLD=11
python alpha_discovery/deep_models/train_cnn_mamba.py   --data-dir data/processed/mbo_events_smart_v3   --output-dir output/cnn_mamba_v2_smart_v3_mar   --n-folds 25 --skip-transfer   >> output/cnn_mamba_v2_retrain_fold11_v3.log 2>&1
