#!/bin/bash
# launch_v343_repair.sh — HC #423 §2 dual-trunk repair launcher (v3.4.3)
#
# Ships /tmp/v343_repair_launcher.py to Neptune and starts it inside the
# training-env venv. Mirrors the conventions used by
# scripts/launch_cnn_mamba_v3_3_neptune.sh (PID file, log file, alive-check),
# but the actual training process runs on Neptune over SSH.
#
# Does NOT auto-launch from Jupiter outside RTH+after-close hours unless the
# operator explicitly passes --force. Default behaviour is: SCP + start, then
# detach. Background watchdog handling is delegated to the persistent
# monitor (heartbeat + GPU-idle alerts).
#
# Usage (operator, post-16:00 ET only per HC #423 §2 sequencing):
#   ./scripts/launch_v343_repair.sh
#       — SCP + launch on Neptune, 60d / 1-fold / 1-epoch default run.
#   ./scripts/launch_v343_repair.sh --smoke-test
#       — SCP + run a CPU smoke test (constructs model, prints param count).
#   ./scripts/launch_v343_repair.sh --dry-build
#       — SCP + V342_DRY_RUN=1 (build datasets, print RSS, exit).

set -euo pipefail

# ----------------------------------------------------------------
# Config
# ----------------------------------------------------------------
NEPTUNE_HOST="${NEPTUNE_HOST:-neptune}"   # uses ~/.ssh/config Host entry
NEPTUNE_USER="${NEPTUNE_USER:-nick}"
NEPTUNE_ROOT="${NEPTUNE_ROOT:-/home/nick/Lvl3Quant}"
NEPTUNE_PY="${NEPTUNE_PY:-/home/nick/miniconda3/envs/py311-train/bin/python}"

LOCAL_LAUNCHER="/tmp/v343_repair_launcher.py"
REMOTE_LAUNCHER="/tmp/v343_repair_launcher.py"

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
REMOTE_LOG_DIR="$NEPTUNE_ROOT/logs"
REMOTE_OUTPUT_DIR="$NEPTUNE_ROOT/output/cnn_mamba_v3_4_3_repair"
REMOTE_LOG="$REMOTE_LOG_DIR/v343_repair_${RUN_TAG}.log"
REMOTE_PID_FILE="$REMOTE_LOG_DIR/pids/v343_repair.pid"

LOCAL_LOG_DIR="/home/jupiter/Lvl3Quant/logs"
LOCAL_LOG="$LOCAL_LOG_DIR/v343_repair_launch_${RUN_TAG}.log"
mkdir -p "$LOCAL_LOG_DIR"

# Training env (passed to the remote python process)
export V32_BATCH_SIZE="${V32_BATCH_SIZE:-16}"
export V32_WF_TRAIN_DAYS="${V32_WF_TRAIN_DAYS:-60}"
export V32_N_FOLDS="${V32_N_FOLDS:-1}"
export V32_EPOCHS="${V32_EPOCHS:-1}"
export V32_NUM_WORKERS="${V32_NUM_WORKERS:-2}"
export V343_AUX_LAMBDA="${V343_AUX_LAMBDA:-0.075}"
export V343_PHASE1_STEPS="${V343_PHASE1_STEPS:-10000}"
export V343_PHASE2_BOOK_GATE_RAW="${V343_PHASE2_BOOK_GATE_RAW:-0.5}"
export V343_TELEMETRY_EVERY="${V343_TELEMETRY_EVERY:-200}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://jupiter:5000}"

# ----------------------------------------------------------------
# Arg parsing
# ----------------------------------------------------------------
SMOKE_TEST=0
DRY_BUILD=0
FORCE=0
for arg in "$@"; do
    case "$arg" in
        --smoke-test) SMOKE_TEST=1 ;;
        --dry-build)  DRY_BUILD=1 ;;
        --force)      FORCE=1 ;;
        *) echo "[launch_v343_repair] unknown arg: $arg" >&2; exit 64 ;;
    esac
done

# ----------------------------------------------------------------
# Sanity: launcher file exists locally
# ----------------------------------------------------------------
if [ ! -f "$LOCAL_LAUNCHER" ]; then
    echo "[launch_v343_repair] FATAL: local launcher not found: $LOCAL_LAUNCHER" >&2
    exit 2
fi
LOCAL_LC=$(wc -l < "$LOCAL_LAUNCHER")
echo "[launch_v343_repair] local launcher: $LOCAL_LAUNCHER ($LOCAL_LC lines)"

# ----------------------------------------------------------------
# Sanity: SSH to Neptune reachable
# ----------------------------------------------------------------
echo "[launch_v343_repair] probing Neptune SSH ($NEPTUNE_HOST)..."
if ! ssh -o ConnectTimeout=10 -o BatchMode=yes "$NEPTUNE_HOST" "true"; then
    echo "[launch_v343_repair] FATAL: cannot SSH to $NEPTUNE_HOST" >&2
    exit 3
fi
echo "[launch_v343_repair] Neptune reachable."

# ----------------------------------------------------------------
# Refuse to start if a v3.4.3 run is already alive on Neptune
# ----------------------------------------------------------------
EXISTING=$(ssh "$NEPTUNE_HOST" "if [ -f $REMOTE_PID_FILE ]; then \
    pid=\$(cat $REMOTE_PID_FILE 2>/dev/null || echo ''); \
    if [ -n \"\$pid\" ] && kill -0 \$pid 2>/dev/null; then echo \"\$pid\"; fi; \
fi")
if [ -n "$EXISTING" ] && [ "$FORCE" = "0" ]; then
    echo "[launch_v343_repair] FATAL: prior v3.4.3 run alive on Neptune (PID=$EXISTING)" >&2
    echo "[launch_v343_repair] use --force to override (will NOT kill the old one)" >&2
    exit 4
fi

# ----------------------------------------------------------------
# Ensure remote dirs exist + SCP launcher
# ----------------------------------------------------------------
ssh "$NEPTUNE_HOST" "mkdir -p $REMOTE_LOG_DIR $REMOTE_LOG_DIR/pids $REMOTE_OUTPUT_DIR"
echo "[launch_v343_repair] SCP $LOCAL_LAUNCHER -> $NEPTUNE_HOST:$REMOTE_LAUNCHER"
scp -q "$LOCAL_LAUNCHER" "$NEPTUNE_HOST:$REMOTE_LAUNCHER"

# Verify Neptune sees the file with right line count
REMOTE_LC=$(ssh "$NEPTUNE_HOST" "wc -l < $REMOTE_LAUNCHER 2>/dev/null || echo 0")
if [ "$REMOTE_LC" -lt 100 ]; then
    echo "[launch_v343_repair] FATAL: remote launcher line count too low ($REMOTE_LC)" >&2
    exit 5
fi
echo "[launch_v343_repair] remote launcher OK ($REMOTE_LC lines)."

# ----------------------------------------------------------------
# Build env-var string for the remote process
# ----------------------------------------------------------------
REMOTE_ENV="\
PYTHONUNBUFFERED=1 \
PYTHONPATH=$NEPTUNE_ROOT \
CUDA_VISIBLE_DEVICES=0 \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI \
MAMBA_FEATURE_SET=smart_v3 \
EVENT_WINDOW_SIZE=1500 EVENT_STRIDE=250 \
V32_BATCH_SIZE=$V32_BATCH_SIZE V32_WF_TRAIN_DAYS=$V32_WF_TRAIN_DAYS \
V32_N_FOLDS=$V32_N_FOLDS V32_EPOCHS=$V32_EPOCHS V32_NUM_WORKERS=$V32_NUM_WORKERS \
V343_AUX_LAMBDA=$V343_AUX_LAMBDA V343_PHASE1_STEPS=$V343_PHASE1_STEPS \
V343_PHASE2_BOOK_GATE_RAW=$V343_PHASE2_BOOK_GATE_RAW \
V343_TELEMETRY_EVERY=$V343_TELEMETRY_EVERY"

# ----------------------------------------------------------------
# Smoke test path: import-only sanity check, exits in seconds.
# ----------------------------------------------------------------
if [ "$SMOKE_TEST" = "1" ]; then
    echo "[launch_v343_repair] SMOKE TEST — importing launcher under V343_SMOKE=1 on Neptune"
    ssh "$NEPTUNE_HOST" "$REMOTE_ENV V343_SMOKE=1 $NEPTUNE_PY -c '\
import importlib.util, sys; \
spec = importlib.util.spec_from_file_location(\"v343\", \"$REMOTE_LAUNCHER\"); \
print(\"smoke-import-ok\", spec is not None)'"
    exit 0
fi

# ----------------------------------------------------------------
# Dry-build path: V342_DRY_RUN=1 (builds dataset, prints RSS, exits)
# ----------------------------------------------------------------
DRY_FLAG=""
if [ "$DRY_BUILD" = "1" ]; then
    DRY_FLAG="V342_DRY_RUN=1"
    echo "[launch_v343_repair] DRY-BUILD mode — datasets only, no training."
fi

# ----------------------------------------------------------------
# Launch
# ----------------------------------------------------------------
echo "[launch_v343_repair] starting Neptune training process..."
echo "[launch_v343_repair] remote log: $NEPTUNE_HOST:$REMOTE_LOG"
echo "[launch_v343_repair] remote pid: $NEPTUNE_HOST:$REMOTE_PID_FILE"
echo "[launch_v343_repair] MLflow experiment: CNNMamba_v3_4_3_dualtrunk_repair"

REMOTE_CMD="\
cd $NEPTUNE_ROOT && \
nohup env $REMOTE_ENV $DRY_FLAG $NEPTUNE_PY -u $REMOTE_LAUNCHER \
    --device cuda --n-folds $V32_N_FOLDS \
    > $REMOTE_LOG 2>&1 < /dev/null & \
echo \$! > $REMOTE_PID_FILE; \
cat $REMOTE_PID_FILE"

REMOTE_PID=$(ssh "$NEPTUNE_HOST" "$REMOTE_CMD")
echo "[launch_v343_repair] started Neptune PID=$REMOTE_PID"

# ----------------------------------------------------------------
# Alive check after 15s
# ----------------------------------------------------------------
sleep 15
STILL=$(ssh "$NEPTUNE_HOST" "kill -0 $REMOTE_PID 2>/dev/null && echo alive || echo dead")
if [ "$STILL" != "alive" ]; then
    echo "[launch_v343_repair] FATAL: Neptune process $REMOTE_PID died within 15s" >&2
    echo "[launch_v343_repair] tail of remote log:" >&2
    ssh "$NEPTUNE_HOST" "tail -120 $REMOTE_LOG" >&2 || true
    exit 6
fi

echo "[launch_v343_repair] OK — Neptune PID=$REMOTE_PID still alive after 15s."
echo "[launch_v343_repair] tail of remote log so far:"
ssh "$NEPTUNE_HOST" "tail -40 $REMOTE_LOG" || true
echo ""
echo "[launch_v343_repair] follow live:  ssh $NEPTUNE_HOST tail -f $REMOTE_LOG"
echo "[launch_v343_repair] telemetry:    ssh $NEPTUNE_HOST tail -f $REMOTE_OUTPUT_DIR/telemetry_*.jsonl"
echo "[launch_v343_repair] kill if bad:  ssh $NEPTUNE_HOST 'kill \$(cat $REMOTE_PID_FILE)'"
