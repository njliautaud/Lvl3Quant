#!/bin/bash
# v3.4.3 DUAL-TRUNK REPAIR v2 LAUNCH WRAPPER
# HC #424 — post-OOM-crash relaunch with v2 patches
# - Kills desktop apps (Firefox/Discord/Steam) freeing ~3 GB RSS for training.
# - Launches the v2 launcher under nohup with stdout/stderr -> timestamped log.
# - Writes PID to logs/pids/v343_repair_v2.pid.

set -u

PROJECT_ROOT="/home/nick/Lvl3Quant"
PY="/home/nick/miniconda3/envs/py311-train/bin/python"
LAUNCHER="/tmp/v343_repair_launcher_v2.py"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${PROJECT_ROOT}/logs"
PID_DIR="${LOG_DIR}/pids"
LOG_PATH="${LOG_DIR}/v343_repair_v2_${TS}.log"
PID_PATH="${PID_DIR}/v343_repair_v2.pid"

mkdir -p "${PID_DIR}"

echo "[$(date -Iseconds)] === v3.4.3 REPAIR v2 LAUNCH ==="
echo "[$(date -Iseconds)] Killing desktop apps (firefox/discord/steam) to free RAM..."
pkill -f firefox 2>/dev/null || true
pkill -f discord 2>/dev/null || true
pkill -f steam 2>/dev/null || true
sleep 1
echo "[$(date -Iseconds)] free -h after desktop kill:"
free -h

cd "${PROJECT_ROOT}"

# Verify launcher exists
if [ ! -f "${LAUNCHER}" ]; then
  echo "[FATAL] launcher not found at ${LAUNCHER}"
  exit 2
fi

# Verify previous PID is dead before launching new one
if [ -f "${PID_PATH}" ]; then
  OLD_PID="$(cat "${PID_PATH}")"
  if [ -n "${OLD_PID}" ] && kill -0 "${OLD_PID}" 2>/dev/null; then
    echo "[FATAL] previous v2 PID ${OLD_PID} still alive — refusing to launch a duplicate"
    exit 3
  fi
fi

echo "[$(date -Iseconds)] log -> ${LOG_PATH}"
echo "[$(date -Iseconds)] pid file -> ${PID_PATH}"

V32_BATCH_SIZE=16 \
V32_NUM_WORKERS=2 \
V32_WF_TRAIN_DAYS=60 \
V32_N_FOLDS=1 \
V32_EPOCHS=1 \
MLFLOW_TRACKING_URI=http://jupiter:5000 \
nohup "${PY}" -u "${LAUNCHER}" --device cuda --n-folds 1 \
  > "${LOG_PATH}" 2>&1 &

NEW_PID=$!
echo "${NEW_PID}" > "${PID_PATH}"
echo "[$(date -Iseconds)] launched PID ${NEW_PID}"
echo "${LOG_PATH}"  # last line = log path for caller
