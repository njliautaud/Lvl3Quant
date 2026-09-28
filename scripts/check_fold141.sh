#!/bin/bash
# Check if fold 141 predictions have landed on Neptune
PRED=$(ssh nick@neptune "ls -la /home/nick/Lvl3Quant/output/v4_multihead_pressure_v1/fold_141_oot_predictions.npz 2>/dev/null" 2>/dev/null)
if [ -n "$PRED" ]; then
    echo "FOLD 141 COMPLETE: $PRED"
    # Sync to Jupiter
    scp nick@neptune:/home/nick/Lvl3Quant/output/v4_multihead_pressure_v1/fold_141_oot_predictions.npz /home/jupiter/Lvl3Quant/output/v4_multihead_pressure_v1/ 2>/dev/null
    scp nick@neptune:/home/nick/Lvl3Quant/output/v4_multihead_pressure_v1/fold_141_best.pt /home/jupiter/Lvl3Quant/output/v4_multihead_pressure_v1/ 2>/dev/null
    echo "SYNCED to Jupiter"
else
    echo "FOLD 141 still training"
fi
