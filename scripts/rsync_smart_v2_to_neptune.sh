#!/bin/bash
# Rsync smart_v2 preprocessed features to Neptune
# Run this when precompute_features_smart_v2.py completes on Jupiter

set -e

SRC="/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v2/"
DST="nick@neptune:/home/nick/Lvl3Quant/data/processed/mbo_events_smart_v2/"

echo "$(date) — Starting rsync of smart_v2 features to Neptune..."
echo "  Source: $SRC"
echo "  Dest:   $DST"

# Count source files
SRC_COUNT=$(ls -1 "$SRC"*.npz 2>/dev/null | wc -l)
echo "  Source files: $SRC_COUNT"

if [ "$SRC_COUNT" -lt 200 ]; then
    echo "ERROR: Only $SRC_COUNT files found — preprocessing may not be complete!"
    exit 1
fi

# Rsync with compression
rsync -avz --progress "$SRC" "$DST"

# Verify on Neptune
echo ""
echo "$(date) — Verifying on Neptune..."
NEPTUNE_COUNT=$(ssh -o ConnectTimeout=10 nick@neptune "ls -1 /home/nick/Lvl3Quant/data/processed/mbo_events_smart_v2/*.npz 2>/dev/null | wc -l")
echo "  Neptune files: $NEPTUNE_COUNT"

if [ "$NEPTUNE_COUNT" -eq "$SRC_COUNT" ]; then
    echo "✓ Rsync complete and verified! $NEPTUNE_COUNT files on Neptune."
    echo ""
    echo "Ready to launch Mamba v6 smart_v2 on Neptune with:"
    echo "  MAMBA_FEATURE_SET=smart_v2 DISABLE_MLFLOW=1 WF_WINDOW_DAYS=60 ..."
else
    echo "WARNING: Count mismatch! Jupiter=$SRC_COUNT, Neptune=$NEPTUNE_COUNT"
fi
