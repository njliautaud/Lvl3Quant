#!/bin/bash
# process_new_mbo_data.sh — Automated pipeline for new MBO data
# Runs on Neptune after GLBX zip download completes
set -euo pipefail

DOWNLOAD_DIR="/home/nick/Downloads"
LVL3="/home/nick/Lvl3Quant"
RAW_DIR="$LVL3/data/raw/mbo"
BASE_DIR="$LVL3/data/processed/mbo_events"
SMART_DIR="$LVL3/data/processed/mbo_events_smart_v3"
LOG="/home/nick/Lvl3Quant/output/mbo_pipeline_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$LOG") 2>&1
echo "=== MBO Data Pipeline Started: $(date) ==="

# Step 0: Find the completed zip
ZIP=$(ls -t "$DOWNLOAD_DIR"/GLBX-*.zip 2>/dev/null | head -1)
if [ -z "$ZIP" ] || [ ! -s "$ZIP" ]; then
    echo "ERROR: No completed GLBX zip found in $DOWNLOAD_DIR"
    exit 1
fi
echo "Found zip: $ZIP ($(du -h "$ZIP" | cut -f1))"

# Step 1: Extract
EXTRACT_DIR="/tmp/glbx_extract_$$"
mkdir -p "$EXTRACT_DIR"
echo "Extracting to $EXTRACT_DIR..."
unzip -o "$ZIP" -d "$EXTRACT_DIR"
echo "Extraction complete"

# Step 2: Find and move .dbn.zst files
mkdir -p "$RAW_DIR"
DBN_COUNT=0
for f in $(find "$EXTRACT_DIR" -name "*.dbn.zst" -type f); do
    fname=$(basename "$f")
    if [ ! -f "$RAW_DIR/$fname" ]; then
        cp "$f" "$RAW_DIR/$fname"
        echo "  Copied: $fname"
        DBN_COUNT=$((DBN_COUNT + 1))
    else
        echo "  Skipped (exists): $fname"
    fi
done
echo "Copied $DBN_COUNT new .dbn.zst files to $RAW_DIR"

# Step 3: Process raw → base NPZ
echo ""
echo "=== Step 3: Converting .dbn.zst → base NPZ ==="
source /home/nick/miniconda3/etc/profile.d/conda.sh
conda activate py311-train
cd "$LVL3"
python3 process_missing_mbo.py --workers 4 --out-dir "$BASE_DIR"
echo "Base NPZ conversion complete"

# Step 4: Generate smart_v3 features
echo ""
echo "=== Step 4: Generating smart_v3 features ==="
cd "$LVL3/alpha_discovery/deep_models"
SMART_INPUT_DIR="$BASE_DIR" SMART_OUTPUT_DIR="$SMART_DIR" python3 precompute_features_smart_v3.py
echo "Smart_v3 feature generation complete"

# Step 5: Count new dates
NEW_BASE=$(ls "$BASE_DIR"/*.npz 2>/dev/null | wc -l)
NEW_SMART=$(ls "$SMART_DIR"/*.npz 2>/dev/null | wc -l)
echo ""
echo "=== Pipeline Complete ==="
echo "Base NPZ total: $NEW_BASE files"
echo "Smart_v3 total: $NEW_SMART files"
echo "Finished at: $(date)"

# Cleanup extract dir
rm -rf "$EXTRACT_DIR"
echo "Cleaned up temp extract dir"
