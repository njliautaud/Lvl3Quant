#!/bin/bash
# EventCNN1D raw 6-channel on smart_v3_mar fold layout (cnn_mamba_v2 baseline match)
# Resolves Mamba-vs-CNN backbone question per DIRECTIVES 22:05 ET 2026-04-27
set -e
cd /home/nick/Lvl3Quant/alpha_discovery/deep_models

OUT=/home/nick/Lvl3Quant/output/event_cnn_1d_smart_v3_mar
mkdir -p "$OUT"

export MLFLOW_TRACKING_URI=http://jupiter:5000
# Match cnn_mamba_v2_smart_v3_mar window/stride
export EVENT_WINDOW_SIZE=1000
export EVENT_STRIDE=500
export EVENT_BATCH_SIZE=128
export EVENT_EPOCHS=5
export EVENT_LR=3e-4
export EVENT_N_FOLDS=11
# Raw 6-channel mode (no CNN_FEATURE_SET) — per directive
export CNN_CHANNELS=128
export CNN_KERNEL=5
export CNN_LAYERS=6
export CNN_DROPOUT=0.1
export PYTHONUNBUFFERED=1

PY=/home/nick/miniconda3/envs/py311-train/bin/python

nohup $PY -u train_event_cnn_1d.py \
  --data-dir /home/nick/Lvl3Quant/data/processed/mbo_events_smart_v3_mar_layout \
  --output-dir "$OUT" \
  --n-folds 11 \
  --train-days 60 \
  --oot-days 1 \
  --device cuda \
  > "$OUT/training.log" 2>&1 &

PID=$!
echo "$PID" > "$OUT/run.pid"
echo "[$(date)] EventCNN1D smart_v3_mar layout launched. PID=$PID"
echo "Log: $OUT/training.log"
