#!/bin/bash
# Wrapper: runs a paper engine then fires the signal callback (HC #791)
# Usage: run_engine_with_callback.sh <engine_script> [engine_name]
# Example: run_engine_with_callback.sh paper_engines/sector_combined_v93_paper.py sector_combined_v93

ENGINE_SCRIPT="$1"
ENGINE_NAME="${2:-$(basename "$1" .py)}"
BASE="/home/jupiter/Lvl3Quant"

cd "$BASE"

# Run the engine
/usr/bin/python3 "$ENGINE_SCRIPT" 2>&1

# Fire the callback chain (aggregator → pre-validator → inject if threshold met)
"$BASE/scripts/signal_callback.sh" "$ENGINE_NAME" 2>&1
