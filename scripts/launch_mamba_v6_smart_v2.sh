#!/bin/bash
# Launch Mamba v6 with smart_v2 (22 features) on Neptune
# Prerequisites: smart_v2 data must be rsynced to Neptune first
#
# This is the NEXT Mamba experiment after v5 feat15 completes or yields enough baseline folds.
# Changes from v5: 22 smart-normalized features (was 15 feat15 with per-fold z-score)
# Same architecture: d_model=192, d_state=96, n_layers=6, epochs=5

set -e

NEPTUNE="nick@neptune"
NEPTUNE_ROOT="/home/nick/Lvl3Quant"

# Verify data exists
echo "Verifying smart_v2 data on Neptune..."
COUNT=$(ssh -o ConnectTimeout=10 $NEPTUNE "ls -1 $NEPTUNE_ROOT/data/processed/mbo_events_smart_v2/*.npz 2>/dev/null | wc -l")
echo "  Found $COUNT files"
if [ "$COUNT" -lt 200 ]; then
    echo "ERROR: Only $COUNT files — run rsync_smart_v2_to_neptune.sh first!"
    exit 1
fi

# Kill any existing Mamba training (only if user confirms)
EXISTING=$(ssh -o ConnectTimeout=10 $NEPTUNE "pgrep -f train_event_mamba || echo none")
if [ "$EXISTING" != "none" ]; then
    echo "WARNING: Existing Mamba process(es) found: $EXISTING"
    echo "Kill them first with: ssh $NEPTUNE 'kill $EXISTING'"
    echo "Or wait for current run to finish."
    exit 1
fi

echo "Launching Mamba v6 smart_v2 on Neptune..."
ssh -o ConnectTimeout=10 $NEPTUNE "cd $NEPTUNE_ROOT && \
    rm -f locks/mamba_training.lock && \
    START_FOLD=0 \
    MAMBA_FEATURE_SET=smart_v2 \
    DISABLE_MLFLOW=1 \
    WF_WINDOW_DAYS=60 \
    EVENT_BATCH_SIZE=64 \
    EVENT_WINDOW_SIZE=1000 \
    EVENT_STRIDE=500 \
    MAMBA_D_MODEL=192 \
    MAMBA_D_STATE=96 \
    MAMBA_N_LAYERS=6 \
    MAMBA_EPOCHS=5 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    nohup /home/nick/miniconda3/envs/py311-train/bin/python -u \
        alpha_discovery/deep_models/train_event_mamba_cuda.py \
        --output-dir $NEPTUNE_ROOT/output/mamba_v6_sliding60d_smart_v2 \
        --n-folds 10 \
    > $NEPTUNE_ROOT/logs/mamba_v6_smart_v2.log 2>&1 & \
    MPID=\$!; echo \"\$MPID\" > locks/mamba_training.lock; echo \"Launched PID: \$MPID\""

echo ""
echo "Monitor with: ssh $NEPTUNE 'tail -f $NEPTUNE_ROOT/logs/mamba_v6_smart_v2.log'"
