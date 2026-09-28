#!/bin/bash
# Bulk minute-bar builder launcher
# Processes all 238 MBO files with progress tracking and resumability

SCRIPT_DIR="/home/jupiter/Lvl3Quant/scripts"
LOG_DIR="/home/jupiter/Lvl3Quant/logs"
OUTPUT_DIR="/home/jupiter/Lvl3Quant/data/processed/mbo_minute_bars_v1"

# Create log directory
mkdir -p "$LOG_DIR"

echo "Starting minute-bars bulk build at $(date)"
echo "Log: $LOG_DIR/minute_bars_build.log"
echo ""

# Run the builder (idempotent: skips existing parquets automatically)
python3 "$SCRIPT_DIR/build_minute_bars_v1.py" \
  --input-dir /home/jupiter/Lvl3Quant/data/raw/mbo \
  --output-dir "$OUTPUT_DIR" \
  --log-dir "$LOG_DIR"

EXIT_CODE=$?

echo ""
echo "Bulk build completed at $(date)"
echo "Output directory: $OUTPUT_DIR"
echo "File count: $(ls -1 $OUTPUT_DIR/*.parquet 2>/dev/null | wc -l)"
echo "Exit code: $EXIT_CODE"
