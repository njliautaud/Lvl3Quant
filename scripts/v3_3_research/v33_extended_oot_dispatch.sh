#!/bin/bash
# v33_extended_oot_dispatch.sh — Queue extended OOT inference on Neptune POST-v3.4.2.
#
# DOES NOT modify any trainer code. Reuses the existing Neptune-side script:
#   /home/nick/Lvl3Quant/scripts/v3_3_research/v32_run_oot_inference.py
# which already supports a --dates list and uses --max-vram-frac to share GPU.
#
# Strategy: wait for Neptune GPU to drop below 30% (v3.4.2 finished), then
# SSH-launch this with a 30+ day extended OOT date list. Output predictions.npz
# enables the entire HC #374 execution science framework on 17+ firing days
# rather than the current 3-4 day data desert.
#
# Internal Lvl3Quant research dispatcher.
set -euo pipefail

NEPTUNE_HOST="nick@neptune"
SSH_KEY="${HOME}/.ssh/id_ed25519"
CKPT="/home/nick/Lvl3Quant/output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_intra_ckpt.pt"
FEAT_STATS="/home/nick/Lvl3Quant/output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_feature_stats.npz"
FOLD_SCHED="/home/nick/Lvl3Quant/output/cnn_mamba_v3_3_uncertainty_weighted/fold_schedule.json"
OUT_NPZ="/home/nick/Lvl3Quant/output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_extended_oot_predictions.npz"
SCRIPT="/home/nick/Lvl3Quant/scripts/v3_3_research/v32_run_oot_inference.py"

# 38-day extended OOT (20260301 → 20260423) excluding weekends
# Pre-OOT (20260223-0227) is the existing 5-day window; we extend post-OOT.
DATES=(
  20260301 20260302 20260303 20260304 20260305 20260306
  20260308 20260309 20260310 20260311 20260312 20260313
  20260315 20260316 20260317 20260318 20260319 20260320
  20260322 20260323 20260324 20260325 20260326 20260327
  20260329 20260330 20260331 20260401 20260402 20260403
  20260405 20260406 20260407 20260408 20260409 20260410
  20260412 20260413
)

# Safety: refuse to run if Neptune GPU >30% (v3.4.2 still using it)
GPU_UTIL=$(ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "$NEPTUNE_HOST" \
  "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1")
echo "Neptune GPU util: ${GPU_UTIL}%"
if [ "$GPU_UTIL" -gt 30 ]; then
  echo "REFUSE: Neptune GPU >30% — v3.4.2 still training. Re-run after Ep 5 verdict."
  exit 1
fi

# Verify ckpt + feat_stats exist on Neptune
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "$NEPTUNE_HOST" \
  "ls -la $CKPT $FEAT_STATS" || { echo "Neptune files missing"; exit 1; }

LOG="/home/nick/Lvl3Quant/logs/v3_3_extended_oot_$(date +%Y%m%d_%H%M%S).log"

# Launch under setsid+nohup so it survives our SSH disconnect
CMD="cd /home/nick/Lvl3Quant && setsid nohup env PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 \
PYTHONPATH=/home/nick/Lvl3Quant /home/nick/miniconda3/envs/py311-train/bin/python -u \
$SCRIPT \
  --ckpt $CKPT \
  --fold-schedule $FOLD_SCHED \
  --output $OUT_NPZ \
  --feature-stats $FEAT_STATS \
  --max-vram-frac 0.50 \
  --batch-size 4 \
  --dates ${DATES[*]} \
  > $LOG 2>&1 < /dev/null & echo \$!"

echo "Launching v3.3 extended OOT inference on Neptune..."
echo "Dates: ${#DATES[@]} = ${DATES[0]}..${DATES[-1]}"
echo "Log:  $LOG"
echo "Out:  $OUT_NPZ"

PID=$(ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "$NEPTUNE_HOST" "$CMD")
echo "Launched PID=$PID on Neptune"
echo "Tail with: ssh nick@neptune 'tail -f $LOG'"
