#!/bin/bash
# Fusion v2 auto-launch watcher (HC #8 / HC #15 / HC #18 / HC #26).
# Polls Jupiter MLflow for `df1c5ff4` (EventCNN1D WF-FIXED) status=FINISHED.
# When df1c5ff4 finishes (or Neptune GPU goes idle for >5min), SSH-launches
# fusion v2 on Neptune with warm-started CNN+Mamba and PatchTST branches.
# Idempotent via marker file. Push Discord alert before+after launch (HC #27).
#
# Cron: */10 * * * * /home/jupiter/Lvl3Quant/scripts/fusion_v2_auto_launch_watcher.sh >> /home/jupiter/Lvl3Quant/logs/fusion_v2_watcher.log 2>&1

set -e

LVL3=/home/jupiter/Lvl3Quant
MARKER=/tmp/fusion_v2_launched.marker
LOG=$LVL3/logs/fusion_v2_watcher.log
NEPTUNE=nick@neptune
MLFLOW_URI=http://jupiter:5000
DF1C5FF4=df1c5ff4e92a4526a1c8ff1ac8762321
INJECT="$LVL3/scripts/autonomy_inject.sh"

mkdir -p "$LVL3/logs"
ts() { date '+%Y-%m-%d %H:%M:%S'; }

# Idempotent: stop if already launched
if [ -f "$MARKER" ]; then
    echo "[$(ts)] Marker present — fusion v2 already launched. Exiting."
    exit 0
fi

# Check df1c5ff4 status via MLflow REST
STATUS=$(curl -s -m 15 "$MLFLOW_URI/api/2.0/mlflow/runs/get?run_id=$DF1C5FF4" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('run',{}).get('info',{}).get('status','UNKNOWN'))" 2>/dev/null || echo "UNKNOWN")

echo "[$(ts)] df1c5ff4 status = $STATUS"

# Only proceed when df1c5ff4 is FINISHED, FAILED, or KILLED (i.e. Neptune is free)
if [ "$STATUS" != "FINISHED" ] && [ "$STATUS" != "FAILED" ] && [ "$STATUS" != "KILLED" ]; then
    echo "[$(ts)] df1c5ff4 still RUNNING — skipping."
    exit 0
fi

# Confirm Neptune GPU actually idle (defense-in-depth)
GPU_UTIL=$(ssh -o ConnectTimeout=10 -o BatchMode=yes "$NEPTUNE" \
    "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits" 2>/dev/null | tr -d ' ' || echo "999")
echo "[$(ts)] Neptune GPU util = ${GPU_UTIL}%"
if [ "$GPU_UTIL" -gt 30 ] 2>/dev/null; then
    echo "[$(ts)] Neptune GPU still busy (${GPU_UTIL}%) — skipping."
    exit 0
fi

# Mark launched BEFORE invocation (prevents double-launch races)
touch "$MARKER"
STAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR=/home/nick/Lvl3Quant/output/fusion_v2_warmstart_$STAMP
LOGREMOTE=/home/nick/Lvl3Quant/logs/fusion_v2_warmstart_$STAMP.log

# Pre-launch Discord push (HC #27)
[ -x "$INJECT" ] && "$INJECT" "FUSION v2 AUTO-LAUNCH FIRING NOW: df1c5ff4 status=$STATUS, Neptune GPU=${GPU_UTIL}%. Launching warm-started fusion (CNN+Mamba ckpt fold_10 + PatchTST ckpt fold_10) on Neptune. Output: $OUTDIR. Push fold-0 IC table within ~50 min." || true

ssh -o ConnectTimeout=15 "$NEPTUNE" bash <<EOF >> "$LOG" 2>&1
set -e
cd /home/nick/Lvl3Quant
export MLFLOW_TRACKING_URI=$MLFLOW_URI
export MLFLOW_EXPERIMENT=FusionBakeoff_v2_warmstart
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export EVENT_NUM_WORKERS=2
export EVENT_BATCH_SIZE=64
mkdir -p logs $OUTDIR
nohup /home/nick/miniconda3/envs/py311-train/bin/python -u \
    alpha_discovery/deep_models/train_cnn_patchtst_mamba.py \
    --smart-data-dir /home/nick/Lvl3Quant/data/processed/mbo_events_smart_v4 \
    --raw-data-dir   /home/nick/Lvl3Quant/data/processed/mbo_events \
    --vol-pred-dir   /home/nick/Lvl3Quant/output/vol_lgbm_v3 \
    --output-dir     $OUTDIR \
    --n-folds 11 \
    --train-days 60 \
    --oot-days 1 \
    --max-folds 1 \
    --device cuda \
    --warm-start-cnn-mamba /home/nick/Lvl3Quant/output/cnn_mamba_v2_smart_v3_mar/fold_10_best.pt \
    --warm-start-patchtst  /home/nick/Lvl3Quant/output/patchtst_razer_weights/fold_10_best.pt \
    > $LOGREMOTE 2>&1 &
echo "FUSION_V2_PID=\$!"
EOF

echo "[$(ts)] Fusion v2 launched on Neptune. Output dir: $OUTDIR"

# Post-launch Discord push (HC #27)
[ -x "$INJECT" ] && "$INJECT" "FUSION v2 LAUNCHED. Output dir: $OUTDIR. Log: $LOGREMOTE. Watch for first MLflow run in 'FusionBakeoff_v2_warmstart' experiment within 5 min — kill+restart if absent (HC #25/#27 mandate)." || true

exit 0
