#!/bin/bash
cd /home/nick/Lvl3Quant
source /home/nick/miniconda3/etc/profile.d/conda.sh
conda activate py311-train
export MLFLOW_TRACKING_URI=http://localhost:5000
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
rm -f logs/v4_multihead_run1.log
rm -rf outputs/v4_multihead_run1
python scripts/train_v4_multihead.py \
    --data-dir data/processed/mbo_events_smart_v3 \
    --pressure-dir data/processed/smooth_pressure_targets \
    --output-dir outputs/v4_multihead_run1 \
    --batch-size 256 \
    --stride 1000 \
    --epochs 3 \
    --lr 0.0003 \
    --mlflow-uri http://localhost:5000 \
    > logs/v4_multihead_run1.log 2>&1
