#!/bin/bash
# Rerun any missing dates after bulk_oot_inference.py completes
# Run this if any dates were skipped or had bad files deleted
cd /home/nick/Lvl3Quant
source /home/nick/miniconda3/bin/activate py311-train

echo "Checking for missing CNN-Mamba dates..."
CNN_OUT="output/cnn_mamba_v2_bulk_oot"
TST_OUT="output/patchtst_bulk_oot"
DATA="data/processed/mbo_events_smart_v3"

missing=0
for f in ; do
    if [ ! -f "/_predictions.npz" ]; then
        echo "Missing CNN: "
        missing=1
    fi
    if [ ! -f "/_predictions.npz" ]; then
        echo "Missing TST: "
    fi
done
echo "Total missing CNN dates: "
