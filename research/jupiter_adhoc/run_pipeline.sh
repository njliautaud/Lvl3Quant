#!/bin/bash
# Wait for event processing to finish
echo "[pipeline] Waiting for event processing to complete..."
while pgrep -f process_missing_mbo > /dev/null 2>&1; do
    COUNT=$(ls /home/jupiter/Lvl3Quant/data/processed/mbo_events/*.npz 2>/dev/null | wc -l)
    echo "[pipeline] Events: $COUNT/209 — still processing..."
    sleep 60
done

COUNT=$(ls /home/jupiter/Lvl3Quant/data/processed/mbo_events/*.npz 2>/dev/null | wc -l)
echo "[pipeline] Events DONE: $COUNT files"

# Run microbatch conversion
echo "[pipeline] Starting microbatch conversion..."
cd /home/jupiter/Lvl3Quant/alpha_discovery/deep_models
python3 preprocess_microbatch.py     --data-dir /home/jupiter/Lvl3Quant/data/processed/mbo_events     --output-dir /home/jupiter/Lvl3Quant/data/processed/mbo_microbatch

MICRO=$(ls /home/jupiter/Lvl3Quant/data/processed/mbo_microbatch/*.npz 2>/dev/null | wc -l)
echo "[pipeline] Microbatch DONE: $MICRO files"

echo "[pipeline] COMPLETE. Ready for sync to GPU nodes."
