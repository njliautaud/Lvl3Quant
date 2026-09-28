#!/bin/bash
set -e
exec 200>/tmp/v4_train.flock
flock -n 200 || { echo "Another instance running"; exit 1; }
source /home/nick/training-env/bin/activate
export DISABLE_MLFLOW=1 PYTHONUNBUFFERED=1
cd /home/nick/Lvl3Quant
python3 -u scripts/train_v4_multihead.py \
  --data-dir /home/nick/Lvl3Quant/data/processed/mbo_events_smart_v3 \
  --pressure-dir /home/nick/Lvl3Quant/data/processed/smooth_pressure_targets \
  --output-dir /home/nick/Lvl3Quant/output/v4_multihead_pressure_v1 \
  --start-fold 137 --batch-size 256 --stride 2000 --epochs 2 --device cuda
