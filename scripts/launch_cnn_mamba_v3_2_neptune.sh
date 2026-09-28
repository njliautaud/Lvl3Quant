#!/bin/bash
# Launch CNN-Mamba v3.2 (long-context multi-head) on Neptune RTX 3090
# Per docs/cnn_mamba_v3.2_spec.md, HC #294.
#
# Usage:
#   ./launch_cnn_mamba_v3_2_neptune.sh                # full 10-fold run
#   ./launch_cnn_mamba_v3_2_neptune.sh --n-folds 1    # smoke test, fold 0 only

set -euo pipefail

LVL3_ROOT="${LVL3_ROOT:-/home/nick/Lvl3Quant}"
cd "$LVL3_ROOT"

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$LVL3_ROOT/logs"
PID_DIR="$LVL3_ROOT/logs/pids"
OUT_DIR="$LVL3_ROOT/output/cnn_mamba_v3_2_long_context"
mkdir -p "$LOG_DIR" "$PID_DIR" "$OUT_DIR"

LOG="$LOG_DIR/cnn_mamba_v3_2_${RUN_TAG}.log"
PID_FILE="$PID_DIR/cnn_mamba_v3_2.pid"

# Env (Tailscale IP for MLflow per HC #281(H))
export LVL3_ROOT
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://jupiter:5000}"
export MAMBA_FEATURE_SET="smart_v3"
export EVENT_WINDOW_SIZE="${EVENT_WINDOW_SIZE:-1500}"
export EVENT_STRIDE="${EVENT_STRIDE:-250}"
export V32_BATCH_SIZE="${V32_BATCH_SIZE:-96}"
export V32_EPOCHS="${V32_EPOCHS:-5}"
export PYTHONPATH="$LVL3_ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1

# venv (per HC #285)
VENV_PY="${VENV_PY:-/home/nick/training-env/bin/python}"
if [ ! -x "$VENV_PY" ]; then
    echo "ERROR: training-env python not found at $VENV_PY" >&2
    exit 1
fi

# Sanity: required input dirs exist
DATA_DIR="$LVL3_ROOT/data/processed/mbo_events_smart_v3"
FIFO_LABEL_DIR="$LVL3_ROOT/data/processed/mbo_events_smart_v3_fifo_labels"
ALPHA_LABEL_DIR="$LVL3_ROOT/data/processed/mbo_events_smart_v3_alpha_labels"
PT_PRED_DIR="$LVL3_ROOT/data/processed/mbo_events_smart_v3_pt_pred"
WARMSTART="$LVL3_ROOT/output/cnn_mamba_v3_smart_v3_fifo/fold_00_best.pt"

for p in "$DATA_DIR" "$FIFO_LABEL_DIR" "$WARMSTART"; do
    if [ ! -e "$p" ]; then
        echo "ERROR: missing required path: $p" >&2
        exit 2
    fi
done

# Alpha labels + pt_pred are optional (heads mask out if missing)
[ -d "$ALPHA_LABEL_DIR" ] || echo "[launcher] WARN: alpha labels missing — alpha heads will be masked"
[ -d "$PT_PRED_DIR" ] || echo "[launcher] WARN: pt_pred missing — PatchTST features will be zeros"

N_DATA=$(ls "$DATA_DIR"/*_mbo_events.npz 2>/dev/null | wc -l)
N_FIFO=$(ls "$FIFO_LABEL_DIR"/*_fifo_labels.npz 2>/dev/null | wc -l)
N_ALPHA=$(ls "$ALPHA_LABEL_DIR"/*_alpha_labels.npz 2>/dev/null | wc -l || echo 0)
N_PT=$(ls "$PT_PRED_DIR"/*_pt_pred_event_aligned.npz 2>/dev/null | wc -l || echo 0)
echo "[launcher] data=$N_DATA  fifo=$N_FIFO  alpha=$N_ALPHA  pt=$N_PT"

# Refuse to start if a prior run is still alive
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE" 2>/dev/null || echo "")
    if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
        echo "ERROR: prior v3.2 still running (PID=$OLD_PID)" >&2
        exit 3
    fi
fi

echo "[launcher] Launching CNN-Mamba v3.2 trainer"
echo "[launcher] Log:    $LOG"
echo "[launcher] PID:    $PID_FILE"
echo "[launcher] Output: $OUT_DIR"
echo "[launcher] MLflow: $MLFLOW_TRACKING_URI  (experiment=CNNMamba_v3_2_long_context)"

nohup "$VENV_PY" -u "$LVL3_ROOT/alpha_discovery/deep_models/train_cnn_mamba_v3_2.py" \
    --data-dir "$DATA_DIR" \
    --fifo-label-dir "$FIFO_LABEL_DIR" \
    --alpha-label-dir "$ALPHA_LABEL_DIR" \
    --pt-pred-dir "$PT_PRED_DIR" \
    --output-dir "$OUT_DIR" \
    --warmstart-ckpt "$WARMSTART" \
    "$@" \
    > "$LOG" 2>&1 &

CHILD_PID=$!
echo "$CHILD_PID" > "$PID_FILE"
echo "[launcher] Started PID=$CHILD_PID"

sleep 8
if ! kill -0 "$CHILD_PID" 2>/dev/null; then
    echo "[launcher] FATAL: process $CHILD_PID died within 8s. Tail of log:" >&2
    tail -80 "$LOG" >&2
    exit 4
fi
echo "[launcher] Process alive after 8s. Tail of log so far:"
tail -30 "$LOG"
