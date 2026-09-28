#!/bin/bash
# Resume 47-day per-date inference after Neptune reboot 2026-05-19 18:04 UTC
# 29/47 done, resuming dates 30/47 onward
set -uo pipefail
cd /home/nick/Lvl3Quant

LOGFILE=logs/v342_oot_47day_perdate_resume_$(date +%Y%m%d_%H%M%S).log
OUTDIR=output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate
mkdir -p "$OUTDIR"

REMAINING=(20260409 20260410 20260412 20260413 20260414 20260415 20260416 20260417 20260419 20260420 20260421 20260422 20260423 20260424 20260426 20260427 20260428 20260429)

i=29
total=47
for d in "${REMAINING[@]}"; do
    i=$((i+1))
    out="$OUTDIR/oot_$d.npz"
    if [ -f "$out" ] && [ $(stat -c %s "$out") -gt 50000 ]; then
        echo "[$i/$total] $d already exists, skip" | tee -a "$LOGFILE"
        continue
    fi
    echo "[$i/$total] $d - starting at $(date +%H:%M:%S)" | tee -a "$LOGFILE"
    V32_BATCH_SIZE=16 V32_WF_TRAIN_DAYS=10 \
      PYTHONPATH=/home/nick/Lvl3Quant \
      /home/nick/miniconda3/envs/py311-train/bin/python -u -X faulthandler \
      scripts/v3_4_research/v342_run_oot_inference.py \
      --device cuda \
      --ckpt output/cnn_mamba_v3_4_2_fixedmtl/fold_00_intra_ckpt.pt \
      --dates "$d" \
      --output "$out" >> "$LOGFILE" 2>&1
    rc=$?
    rss=$(awk '/MemAvailable/ {printf "%dMB", $2/1024}' /proc/meminfo)
    echo "[$i/$total] $d done rc=$rc at $(date +%H:%M:%S) - RSS-free: $rss" | tee -a "$LOGFILE"
done
echo "ALL DONE at $(date +%H:%M:%S)" | tee -a "$LOGFILE"
