#!/bin/bash
# launch_headc_K1_when_free.sh — relaunch head-C K=1 sensitivity variant (HC #599 R3)
# Guarded per HC #599 R2: refuses to launch while user is gaming.
# Usage: bash /home/nick/Lvl3Quant/scripts/launch_headc_K1_when_free.sh
set -u
PY=/home/nick/miniconda3/envs/py311-train/bin/python
MLFLOW_BIN=/home/nick/miniconda3/envs/py311-train/bin/mlflow
ROOT=/home/nick/Lvl3Quant
SCRIPT=$ROOT/scripts/p_alpha_headc_firstpassage_v1_K1.py
TS=$(date +%Y%m%d_%H%M%S)
LOG=$ROOT/logs/p_alpha_headc_v1_K1_relaunch_${TS}.log

# 0. ALREADY-RUNNING GUARD
if pgrep -f "p_alpha_headc_firstpassage_v1_K1.py" >/dev/null; then
    echo "[SKIP] K=1 training already running."
    exit 0
fi

# 1. GAMING CHECK (HC #599 R2) — game executables only (Steam client idling is OK)
if pgrep -af 'deadlock\.exe|wineserver|gamescope|\.exe -steam' | grep -vi 'launch_headc' >/dev/null; then
    echo "[ABORT] Gaming processes detected on Neptune — HC #599 R2. Not launching."
    pgrep -af 'deadlock\.exe|wineserver' | head -3
    exit 1
fi
# GPU-free check: gate on COMPUTE apps, not raw util (desktop/Steam UI renders 30-40% on an attached GPU).
# Abort only if a compute app is using >500 MiB VRAM (i.e. real training/inference/game compute).
BIGCOMPUTE=$(nvidia-smi --query-compute-apps=name,used_memory --format=csv,noheader,nounits | awk -F", " '$2 > 500 {print}')
if [ -n "$BIGCOMPUTE" ]; then
    echo "[ABORT] Active compute app(s) on GPU — node not free. Not launching."
    echo "$BIGCOMPUTE"
    exit 1
fi

# 2. Ensure MLflow server (sqlite backend, same store as completed K=2 run)
if ! curl -s --max-time 3 http://localhost:5000/api/2.0/mlflow/experiments/search -d '{"max_results":1}' -H 'Content-Type: application/json' >/dev/null 2>&1; then
    echo "[INFO] MLflow server down — starting (sqlite:///mlflow.db, artifacts mlruns/)"
    cd $ROOT
    nohup $MLFLOW_BIN server --backend-store-uri sqlite:///$ROOT/mlflow.db \
        --default-artifact-root $ROOT/mlruns --host 0.0.0.0 --port 5000 \
        > $ROOT/logs/mlflow_server_${TS}.log 2>&1 &
    sleep 10
fi

# 3. Launch K=1 sensitivity (10h wall cap, MLflow mandatory, fresh output dir)
cd $ROOT
export MLFLOW_TRACKING_URI=http://localhost:5000
nohup $PY $SCRIPT --wall-cap-min 600 > $LOG 2>&1 &
PID=$!
echo "[LAUNCHED] PID=$PID LOG=$LOG"
sleep 90
if ! kill -0 $PID 2>/dev/null; then echo "[FAIL] process died — check $LOG"; tail -20 $LOG; exit 1; fi
if grep -q 'MLflow run started' $LOG; then grep 'MLflow run started' $LOG; else echo "[WARN] no MLflow run yet after 90s — verify within 5 min or kill (HC: MLflow mandatory)"; fi
tail -5 $LOG
