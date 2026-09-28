#!/bin/bash
# v4_fold_sync.sh — Sync v4 multihead fold predictions from Neptune to Jupiter
# Run via cron every 30 minutes during training
# Cron: */30 * * * * /home/jupiter/Lvl3Quant/scripts/v4_fold_sync.sh >> /home/jupiter/Lvl3Quant/logs/v4_fold_sync.log 2>&1

set -u
LOG_PFX="[$(date '+%Y-%m-%d %H:%M:%S')]"
LOCAL_DIR=/home/jupiter/Lvl3Quant/output/v4_multihead_pressure_v1
REMOTE_DIR=nick@neptune:/home/nick/Lvl3Quant/output/v4_multihead_pressure_v1

# Count existing local folds
local_before=$(ls ${LOCAL_DIR}/fold_*_oot_predictions.npz 2>/dev/null | wc -l)

# Sync
rsync -az --timeout=30 ${REMOTE_DIR}/fold_*_oot_predictions.npz ${LOCAL_DIR}/ 2>/dev/null

local_after=$(ls ${LOCAL_DIR}/fold_*_oot_predictions.npz 2>/dev/null | wc -l)
new_folds=$((local_after - local_before))

echo "${LOG_PFX} Synced: ${local_before} → ${local_after} folds (${new_folds} new)"

# If we got new folds and have 20+, run the analysis
if [ ${new_folds} -gt 0 ] && [ ${local_after} -ge 20 ]; then
    echo "${LOG_PFX} Running v4 deep validation with ${local_after} folds..."
    cd /home/jupiter/Lvl3Quant
    python3 scripts/v4_deep_validation.py > /home/jupiter/Lvl3Quant/logs/v4_deep_validation_latest.log 2>&1
    echo "${LOG_PFX} Validation complete. Check logs/v4_deep_validation_latest.log"
fi
