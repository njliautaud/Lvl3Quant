#!/bin/bash
# Launch Fusion v1 (CNN + PatchTST + smart + vol → Mamba) — fold 0 only.
#
# This is the GATING run for the user's morning deliverable (Apr 28 ~7-9 AM ET).
# It runs ONLY the first OOT fold (~20260223) so we get a result before the user wakes up.
# A full 11-fold sweep is launched separately via the same script with --max-folds removed.
#
# Layout matches the cnn_mamba_v2_smart_v3_mar bake-off:
#   sliding 60d train + 1d OOT, n_folds=11, fold-0 OOT date ≈ 20260223
#
# Usage:
#   bash /home/jupiter/Lvl3Quant/scripts/launch_fusion_v1_fold0.sh
#
# Env overrides:
#   DEVICE=cuda|cpu                    (default cuda; auto-falls-back if no GPU)
#   MAX_SILENT_MIN=N                   (watchdog inactivity threshold; default 15)
#   MLFLOW_TRACKING_URI=http://...     (default: http://jupiter:5000)
#   SMART_DATA_DIR=...                 (default: data/processed/mbo_events_smart_v3)
#   RAW_DATA_DIR=...                   (default: data/processed/mbo_events)
#   VOL_PRED_DIR=...                   (default: output/vol_lgbm_v3)
#   OUTPUT_DIR=...                     (default: output/fusion_bakeoff_v1)
#   WARM_CNN_MAMBA=ckpt.pt             (optional warm-start path)
#   WARM_PATCHTST=ckpt.pt              (optional warm-start path)

set -e

LVL3="${LVL3:-/home/jupiter/Lvl3Quant}"
cd "$LVL3"

# ---- Config (env-overridable) ----
DEVICE="${DEVICE:-cuda}"
MAX_SILENT_MIN="${MAX_SILENT_MIN:-15}"
MLFLOW_URI="${MLFLOW_TRACKING_URI:-http://jupiter:5000}"
SMART_DATA_DIR="${SMART_DATA_DIR:-$LVL3/data/processed/mbo_events_smart_v3}"
RAW_DATA_DIR="${RAW_DATA_DIR:-$LVL3/data/processed/mbo_events}"
VOL_PRED_DIR="${VOL_PRED_DIR:-$LVL3/output/vol_lgbm_v3}"
OUTPUT_DIR="${OUTPUT_DIR:-$LVL3/output/fusion_bakeoff_v1}"
LOG_DIR="$LVL3/logs"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/fusion_v1_fold0_${STAMP}.log"

echo "[LAUNCH] Fusion v1 — fold 0 only"
echo "[LAUNCH] STAMP        = $STAMP"
echo "[LAUNCH] LOG_FILE     = $LOG_FILE"
echo "[LAUNCH] OUTPUT_DIR   = $OUTPUT_DIR"
echo "[LAUNCH] SMART_DATA   = $SMART_DATA_DIR"
echo "[LAUNCH] RAW_DATA     = $RAW_DATA_DIR"
echo "[LAUNCH] VOL_PRED_DIR = $VOL_PRED_DIR"
echo "[LAUNCH] DEVICE       = $DEVICE"
echo "[LAUNCH] MLFLOW       = $MLFLOW_URI"

WARM_ARGS=""
if [ -n "$WARM_CNN_MAMBA" ]; then
    WARM_ARGS="$WARM_ARGS --warm-start-cnn-mamba $WARM_CNN_MAMBA"
    echo "[LAUNCH] WARM_CNN_MAMBA = $WARM_CNN_MAMBA"
fi
if [ -n "$WARM_PATCHTST" ]; then
    WARM_ARGS="$WARM_ARGS --warm-start-patchtst $WARM_PATCHTST"
    echo "[LAUNCH] WARM_PATCHTST  = $WARM_PATCHTST"
fi

# Export env (used by both training script + helper imports)
export MLFLOW_TRACKING_URI="$MLFLOW_URI"
export MLFLOW_EXPERIMENT="FusionBakeoff_v1_mamba"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Build the training command (matches train_event_cnn_1d_v3_mar layout: 60d/1d sliding)
PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAIN_CMD="$PYTHON_BIN -u alpha_discovery/deep_models/train_cnn_patchtst_mamba.py \
    --smart-data-dir $SMART_DATA_DIR \
    --raw-data-dir   $RAW_DATA_DIR \
    --vol-pred-dir   $VOL_PRED_DIR \
    --output-dir     $OUTPUT_DIR \
    --n-folds 11 \
    --train-days 60 \
    --oot-days 1 \
    --max-folds 1 \
    --device $DEVICE \
    $WARM_ARGS"

echo "[LAUNCH] CMD          = $TRAIN_CMD"
echo ""

# Use the project's standard watchdog wrapper (kills training if log goes silent > MAX_SILENT_MIN)
exec bash "$LVL3/utils/launch_with_watchdog.sh" "$TRAIN_CMD" "$LOG_FILE" "$MAX_SILENT_MIN"
