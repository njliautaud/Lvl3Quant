#!/usr/bin/env bash
# HC #410 P1 — Permanent launcher for v3.4.2 RESUME from book_gate_fix checkpoint.
#
# Reuses MLflow run e5f0f79b313d4ac4aa461df8b7af2385 (HC #407 rule 4).
# Loads /home/nick/Lvl3Quant/output/cnn_mamba_v3_4_2_fixedmtl/fold_00_intra_ckpt.book_gate_fix.pt
# Uses memsafe dataset class (SmartV34MemmapDualTrunkDataset).
#
# Inner launcher /tmp/v342_resume_launcher.py monkey-patches dispatch_v34_2_fixedmtl
# to: (a) use memmap dataset, (b) reuse MLflow run, (c) load full v3.4.2 state.
#
# USAGE:
#   bash /home/nick/Lvl3Quant/scripts/v3_4_research/launch_v342_resume_book_gate_fix.sh
#
# Writes PID to /tmp/v342_resume.pid, log to /home/nick/Lvl3Quant/logs/v3_4_2/v342_resume_<ts>.log
set -euo pipefail

CKPT="/home/nick/Lvl3Quant/output/cnn_mamba_v3_4_2_fixedmtl/fold_00_intra_ckpt.book_gate_fix.pt"
MLFLOW_RUN_ID="${V32_MLFLOW_RUN_ID:-e5f0f79b313d4ac4aa461df8b7af2385}"
LAUNCHER="/tmp/v342_resume_launcher.py"
PY="/home/nick/miniconda3/envs/py311-train/bin/python"
LOG_DIR="/home/nick/Lvl3Quant/logs/v3_4_2"
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/v342_resume_book_gate_fix_${TS}.log"

[ -f "$CKPT" ]     || { echo "FATAL: ckpt missing: $CKPT" >&2; exit 1; }
[ -f "$LAUNCHER" ] || { echo "FATAL: launcher missing: $LAUNCHER (recreate from scripts/v3_4_research/launch_v342_resume_book_gate_fix.sh adjacent file)" >&2; exit 1; }
mkdir -p "$LOG_DIR"

cd /home/nick/Lvl3Quant

V32_RESUME_CKPT="$CKPT" \
V32_MLFLOW_RUN_ID="$MLFLOW_RUN_ID" \
V32_WF_TRAIN_DAYS="${V32_WF_TRAIN_DAYS:-60}" \
V32_BATCH_SIZE="${V32_BATCH_SIZE:-8}" \
V32_NUM_WORKERS="${V32_NUM_WORKERS:-1}" \
V32_EPOCHS="${V32_EPOCHS:-5}" \
V32_N_FOLDS="${V32_N_FOLDS:-1}" \
V32_CKPT_EVERY_N_BATCHES="${V32_CKPT_EVERY_N_BATCHES:-500}" \
MLFLOW_TRACKING_URI="http://jupiter:5000" \
PYTHONPATH="/home/nick/Lvl3Quant" \
nohup "$PY" -u -X faulthandler "$LAUNCHER" --device cuda --n-folds 1 \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "$PID" > /tmp/v342_resume.pid
echo "Launched v3.4.2 resume: PID=$PID  LOG=$LOG_FILE"
echo "Reusing MLflow run: $MLFLOW_RUN_ID"
echo "Ckpt: $CKPT"
