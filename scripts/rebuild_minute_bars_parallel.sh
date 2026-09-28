#!/bin/bash
# Parallel rebuild of minute bars with fixed microprice
# Uses GNU parallel or xargs to process multiple days concurrently
# Each day is independent — no cross-day state needed

SCRIPT="/home/jupiter/Lvl3Quant/scripts/build_minute_bars_v1.py"
RAW_DIR="/home/jupiter/Lvl3Quant/data/raw/mbo"
OUT_DIR="/home/jupiter/Lvl3Quant/data/processed/mbo_minute_bars_v1"
LOG_DIR="/home/jupiter/Lvl3Quant/logs/minute_bars_rebuild"
JOBS=8  # 8 parallel jobs (each uses ~500MB RAM, so 8 × 500MB = 4GB)

mkdir -p "$OUT_DIR" "$LOG_DIR"

# Get list of dates from raw files
dates=()
for f in "$RAW_DIR"/glbx-mdp3-*.mbo.dbn.zst; do
    date=$(basename "$f" | sed 's/glbx-mdp3-\([0-9]*\).*/\1/')
    # Skip if output already exists
    if [ ! -f "$OUT_DIR/${date}.parquet" ]; then
        dates+=("$date")
    fi
done

echo "Total dates to process: ${#dates[@]}"
echo "Parallel jobs: $JOBS"
echo "Starting at $(date)"

# Process in parallel using xargs
printf '%s\n' "${dates[@]}" | xargs -P $JOBS -I {} bash -c "
    python3 $SCRIPT --date {} > $LOG_DIR/{}.log 2>&1
    if [ \$? -eq 0 ]; then
        echo '[OK] {}'
    else
        echo '[FAIL] {}'
    fi
"

echo "Completed at $(date)"
echo "Files built: $(ls $OUT_DIR/*.parquet 2>/dev/null | wc -l)"
