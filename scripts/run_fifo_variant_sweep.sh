#!/bin/bash
# HC #265 FIFO replay variant sweep — runs after default (mfe/all/limit) finishes.
# Each variant ~8-12 min. Total ~30-40 min.
set -u
LVL3=/home/jupiter/Lvl3Quant
LOGDIR=$LVL3/logs
mkdir -p "$LOGDIR"

# Wait for the default run (PID 2959280) to finish if still running
while pgrep -f "validate_via_fifo_replay" > /dev/null; do
    echo "[$(date +%H:%M:%S)] waiting for prior FIFO run..."
    sleep 60
done

cd "$LVL3"

run_variant() {
    local name=$1; shift
    local logfile="$LOGDIR/fifo_replay_v3_${name}.log"
    echo "[$(date +%H:%M:%S)] === ${name} ===" | tee -a "$logfile"
    python3 -u scripts/validate_via_fifo_replay.py "$@" >> "$logfile" 2>&1
    echo "[$(date +%H:%M:%S)] ${name} DONE — tail:" | tee -a "$logfile"
    tail -25 "$logfile"
}

# Variant A — pred_pnl ranker (most likely to help: ranks by predicted realized PnL)
run_variant "pnl_all_limit" --ranker pnl --side all --order-type limit

# Variant B — chase order type (move limit toward market every 1s)
run_variant "mfe_all_chase" --ranker mfe --side all --order-type chase

# Variant C — short-side only (decay analysis: short has stronger edge)
run_variant "mfe_short_limit" --ranker mfe --side short --order-type limit

# Variant D — pred_pnl + chase combined
run_variant "pnl_all_chase" --ranker pnl --side all --order-type chase

echo "[$(date +%H:%M:%S)] FIFO variant sweep complete."
