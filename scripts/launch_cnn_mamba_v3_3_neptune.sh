#!/bin/bash
# Launch CNN-Mamba v3.3 (uncertainty-weighted multi-task) on Neptune RTX 3090
#
# Same model architecture and data pipeline as v3.2; the only difference is
# the loss class — per-head learnable log_sigma (Kendall et al. 2018) instead
# of static lambdas. Drop-in replacement for launch_cnn_mamba_v3_2_neptune.sh.
#
# Usage:
#   ./launch_cnn_mamba_v3_3_neptune.sh                # full 10-fold run
#   ./launch_cnn_mamba_v3_3_neptune.sh --n-folds 1    # smoke test, fold 0 only

set -euo pipefail

LVL3_ROOT="${LVL3_ROOT:-/home/nick/Lvl3Quant}"
cd "$LVL3_ROOT"

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$LVL3_ROOT/logs"
PID_DIR="$LVL3_ROOT/logs/pids"
OUT_DIR="$LVL3_ROOT/output/cnn_mamba_v3_3_uncertainty_weighted"
mkdir -p "$LOG_DIR" "$PID_DIR" "$OUT_DIR"

LOG="$LOG_DIR/cnn_mamba_v3_3_${RUN_TAG}.log"
PID_FILE="$PID_DIR/cnn_mamba_v3_3.pid"

# Env (Tailscale IP for MLflow per HC #281(H))
export LVL3_ROOT
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://jupiter:5000}"
export MAMBA_FEATURE_SET="smart_v3"
export EVENT_WINDOW_SIZE="${EVENT_WINDOW_SIZE:-1500}"
export EVENT_STRIDE="${EVENT_STRIDE:-250}"
# Reuse the v3.2 env names so the same wrapper tooling/cron lines apply.
export V32_BATCH_SIZE="${V32_BATCH_SIZE:-96}"
export V32_EPOCHS="${V32_EPOCHS:-5}"
# v3.3-specific: initial log_sigma per head (0.0 => sigma=1.0)
export V33_LOG_SIGMA_INIT="${V33_LOG_SIGMA_INIT:-0.0}"
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
TIER2_PARQUET="$LVL3_ROOT/data/derived/tier2_orderflow_features_v1.parquet"
TIER3_PARQUET="$LVL3_ROOT/data/derived/tier3_session_features_v1.parquet"
WARMSTART="$LVL3_ROOT/output/cnn_mamba_v3_smart_v3_fifo/fold_00_best.pt"

for p in "$DATA_DIR" "$FIFO_LABEL_DIR" "$WARMSTART"; do
    if [ ! -e "$p" ]; then
        echo "ERROR: missing required path: $p" >&2
        exit 2
    fi
done

# Alpha labels + pt_pred + tier2/3 parquet are optional (heads/tiers mask out)
[ -d "$ALPHA_LABEL_DIR" ] || echo "[launcher] WARN: alpha labels missing — alpha heads will be masked"
[ -d "$PT_PRED_DIR" ] || echo "[launcher] WARN: pt_pred missing — PatchTST features will be zeros"
[ -e "$TIER2_PARQUET" ] || echo "[launcher] WARN: tier2 parquet missing — Tier 2 inputs will be zero"
[ -e "$TIER3_PARQUET" ] || echo "[launcher] WARN: tier3 parquet missing — Tier 3 inputs will be zero"

N_DATA=$(ls "$DATA_DIR"/*_mbo_events.npz 2>/dev/null | wc -l)
N_FIFO=$(ls "$FIFO_LABEL_DIR"/*_fifo_labels.npz 2>/dev/null | wc -l)
N_ALPHA=$(ls "$ALPHA_LABEL_DIR"/*_alpha_labels.npz 2>/dev/null | wc -l || echo 0)
N_PT=$(ls "$PT_PRED_DIR"/*_pt_pred_event_aligned.npz 2>/dev/null | wc -l || echo 0)
echo "[launcher] data=$N_DATA  fifo=$N_FIFO  alpha=$N_ALPHA  pt=$N_PT"

# Refuse to start if a prior run is still alive
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE" 2>/dev/null || echo "")
    if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
        echo "ERROR: prior v3.3 still running (PID=$OLD_PID)" >&2
        exit 3
    fi
fi

echo "[launcher] Launching CNN-Mamba v3.3 (uncertainty-weighted) trainer"
echo "[launcher] Log:    $LOG"
echo "[launcher] PID:    $PID_FILE"
echo "[launcher] Output: $OUT_DIR"
echo "[launcher] MLflow: $MLFLOW_TRACKING_URI  (experiment=CNNMamba_v3_3_uncertainty_weighted)"
echo "[launcher] log_sigma_init=$V33_LOG_SIGMA_INIT"

nohup "$VENV_PY" -u "$LVL3_ROOT/alpha_discovery/deep_models/train_cnn_mamba_v3_3.py" \
    --data-dir "$DATA_DIR" \
    --fifo-label-dir "$FIFO_LABEL_DIR" \
    --alpha-label-dir "$ALPHA_LABEL_DIR" \
    --pt-pred-dir "$PT_PRED_DIR" \
    --tier2-parquet-root "$TIER2_PARQUET" \
    --tier3-parquet-root "$TIER3_PARQUET" \
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
