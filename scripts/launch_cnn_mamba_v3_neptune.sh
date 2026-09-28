#!/bin/bash
# Launch CNN-Mamba v3 (FIFO-aware multi-head) on Neptune RTX 3090
# Per CNN_MAMBA_V3_SPEC.md, HC #281 (code mods authorized), HC #282 (weekly retrain),
# HC #283 (user-authorized trading research code).
#
# Usage:
#   ./launch_cnn_mamba_v3_neptune.sh                  # full 10-fold run
#   ./launch_cnn_mamba_v3_neptune.sh --n-folds 1      # smoke test, fold 0 only
#
# This script is meant to be SSH-invoked from Jupiter:
#   ssh nick@neptune "bash -lc 'cd /home/nick/Lvl3Quant && ./scripts/launch_cnn_mamba_v3_neptune.sh'"

set -euo pipefail

# Resolve repo root (works whether invoked from / or from anywhere)
LVL3_ROOT="${LVL3_ROOT:-/home/nick/Lvl3Quant}"
cd "$LVL3_ROOT"

# Run-time identity
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$LVL3_ROOT/logs"
PID_DIR="$LVL3_ROOT/logs/pids"
OUT_DIR="$LVL3_ROOT/output/cnn_mamba_v3_smart_v3_fifo"
mkdir -p "$LOG_DIR" "$PID_DIR" "$OUT_DIR"

LOG="$LOG_DIR/cnn_mamba_v3_${RUN_TAG}.log"
PID_FILE="$PID_DIR/cnn_mamba_v3.pid"

# Env (per HC #281(H) — Tailscale IP for cross-node MLflow)
export LVL3_ROOT
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://jupiter:5000}"
export MAMBA_FEATURE_SET="smart_v3"
# Allow caller override via env vars (so we can tune for GPU memory)
export EVENT_WINDOW_SIZE="${EVENT_WINDOW_SIZE:-3000}"
export EVENT_STRIDE="${EVENT_STRIDE:-250}"
export EVENT_BATCH_SIZE="${EVENT_BATCH_SIZE:-128}"
export PYTHONPATH="$LVL3_ROOT:${PYTHONPATH:-}"
# CUDA / dataloader
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
# Force unbuffered Python so log lines flush immediately
export PYTHONUNBUFFERED=1

# venv (Neptune training env) — HC #285: use /home/nick/training-env which has
# mamba_ssm 2.3.1 + causal_conv1d 1.6.1 prebuilt (vs venv_training which lacks them
# and forced the pure-PyTorch S6 fallback that caused smoke-5 to OOM-slow at 17s/batch).
VENV_PY="${VENV_PY:-/home/nick/training-env/bin/python}"
if [ ! -x "$VENV_PY" ]; then
    echo "ERROR: training-env python not found at $VENV_PY" >&2
    exit 1
fi

# Sanity: required input dirs exist
DATA_DIR="$LVL3_ROOT/data/processed/mbo_events_smart_v3"
LABEL_DIR="$LVL3_ROOT/data/processed/mbo_events_smart_v3_fifo_labels"
WARMSTART="$LVL3_ROOT/output/cnn_mamba_v2_smart_v3_mar/fold_10_best.pt"

for p in "$DATA_DIR" "$LABEL_DIR" "$WARMSTART"; do
    if [ ! -e "$p" ]; then
        echo "ERROR: missing required path: $p" >&2
        exit 2
    fi
done

N_DATA=$(ls "$DATA_DIR"/*_mbo_events.npz 2>/dev/null | wc -l)
N_LABEL=$(ls "$LABEL_DIR"/*_fifo_labels.npz 2>/dev/null | wc -l)
echo "[launcher] data files=$N_DATA  label files=$N_LABEL  warmstart=$(ls -la "$WARMSTART" | awk '{print $5}') bytes"

# Refuse to start if a prior run is still alive
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE" 2>/dev/null || echo "")
    if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
        echo "ERROR: prior CNN-Mamba v3 still running (PID=$OLD_PID). Refusing to start." >&2
        exit 3
    fi
fi

# Kick off
echo "[launcher] Launching CNN-Mamba v3 trainer"
echo "[launcher] Log:    $LOG"
echo "[launcher] PID:    $PID_FILE"
echo "[launcher] Output: $OUT_DIR"
echo "[launcher] MLflow: $MLFLOW_TRACKING_URI  (experiment=CNNMamba_v3_FIFO)"

nohup "$VENV_PY" -u "$LVL3_ROOT/alpha_discovery/deep_models/train_cnn_mamba_v3.py" \
    --data-dir "$DATA_DIR" \
    --label-dir "$LABEL_DIR" \
    --output-dir "$OUT_DIR" \
    --warmstart-ckpt "$WARMSTART" \
    "$@" \
    > "$LOG" 2>&1 &

CHILD_PID=$!
echo "$CHILD_PID" > "$PID_FILE"
echo "[launcher] Started PID=$CHILD_PID"

# Brief liveness check (5s) — confirm process didn't immediately die
sleep 5
if ! kill -0 "$CHILD_PID" 2>/dev/null; then
    echo "[launcher] FATAL: process $CHILD_PID died within 5s. Last log:" >&2
    tail -50 "$LOG" >&2
    exit 4
fi
echo "[launcher] Process alive after 5s. Tail of log so far:"
tail -20 "$LOG"
