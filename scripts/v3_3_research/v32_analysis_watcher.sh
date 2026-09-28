#!/usr/bin/env bash
# v3.2 OOT Deep-Sim Watcher — polls Neptune for predictions.npz, SCPs it
# to Jupiter when ready, runs the deep analysis, and writes the morning
# briefing markdown.
#
# Usage: nohup ./v32_analysis_watcher.sh > /home/jupiter/Lvl3Quant/logs/v32_watcher.log 2>&1 &

set -uo pipefail

NEPTUNE_HOST="nick@neptune"
NEPTUNE_NPZ="/home/nick/Lvl3Quant/output/v3_2_deep_sim_20260512/fold_00_oot_predictions.npz"
NEPTUNE_META="/home/nick/Lvl3Quant/output/v3_2_deep_sim_20260512/fold_00_oot_predictions.metrics.json"
JUPITER_OUT="/home/jupiter/Lvl3Quant/output/v3_2_deep_sim_20260512"
LOCAL_NPZ="$JUPITER_OUT/fold_00_oot_predictions.npz"
LOCAL_META="$JUPITER_OUT/fold_00_oot_predictions.metrics.json"
PYBIN="/home/jupiter/miniconda3/envs/ray311/bin/python"
SCRIPT="/home/jupiter/Lvl3Quant/scripts/v3_3_research/v32_deep_analysis.py"
MAX_WAIT_HOURS=14
POLL_SECS=180
START_TS=$(date +%s)

mkdir -p "$JUPITER_OUT"
echo "[$(date -Is)] watcher started — polling every ${POLL_SECS}s for ${MAX_WAIT_HOURS}h"

while true; do
    NOW=$(date +%s)
    ELAPSED_H=$(( (NOW - START_TS) / 3600 ))
    if [[ $ELAPSED_H -ge $MAX_WAIT_HOURS ]]; then
        echo "[$(date -Is)] TIMEOUT after ${MAX_WAIT_HOURS}h — exiting without running analysis"
        exit 2
    fi

    # Check if NPZ exists on Neptune (must be > 1MB and stable for 2 consecutive polls)
    SIZE_NOW=$(ssh -o ConnectTimeout=10 "$NEPTUNE_HOST" "stat -c%s '$NEPTUNE_NPZ' 2>/dev/null || echo 0")
    if [[ "$SIZE_NOW" == "0" ]] || [[ "$SIZE_NOW" -lt 1000000 ]]; then
        echo "[$(date -Is)] npz not ready (size=$SIZE_NOW) — elapsed ${ELAPSED_H}h, sleeping ${POLL_SECS}s"
        sleep "$POLL_SECS"
        continue
    fi

    echo "[$(date -Is)] npz detected size=$SIZE_NOW bytes — confirming stability"
    sleep 30
    SIZE_VERIFY=$(ssh -o ConnectTimeout=10 "$NEPTUNE_HOST" "stat -c%s '$NEPTUNE_NPZ' 2>/dev/null || echo 0")
    if [[ "$SIZE_NOW" != "$SIZE_VERIFY" ]]; then
        echo "[$(date -Is)] npz still being written ($SIZE_NOW -> $SIZE_VERIFY) — waiting"
        sleep "$POLL_SECS"
        continue
    fi

    echo "[$(date -Is)] npz stable. SCPing to Jupiter..."
    scp "$NEPTUNE_HOST:$NEPTUNE_NPZ" "$LOCAL_NPZ"
    scp "$NEPTUNE_HOST:$NEPTUNE_META" "$LOCAL_META" 2>/dev/null || echo "(no meta.json on Neptune yet, continuing)"
    echo "[$(date -Is)] SCP done. Local size: $(stat -c%s "$LOCAL_NPZ") bytes"

    echo "[$(date -Is)] launching deep analysis..."
    META_ARG=""
    [[ -f "$LOCAL_META" ]] && META_ARG="--meta-json $LOCAL_META"
    "$PYBIN" "$SCRIPT" \
        --predictions-npz "$LOCAL_NPZ" \
        --output-dir "$JUPITER_OUT" \
        $META_ARG
    ANALYSIS_RC=$?
    echo "[$(date -Is)] analysis exited rc=$ANALYSIS_RC"
    echo "[$(date -Is)] morning_briefing.md preview:"
    head -60 "$JUPITER_OUT/morning_briefing.md" 2>/dev/null || true
    exit $ANALYSIS_RC
done
