#!/bin/bash
# signal_callback.sh — TRUE EVENT-DRIVEN EXECUTION (HC #791)
#
# Called by paper engines AFTER they update their state files.
# Chain: engine updates → this script → aggregator → pre-validator → autonomy inject
#
# Usage: signal_callback.sh [engine_name]
# Example: signal_callback.sh "sector_combined_v93"
#
# This replaces 15-min polling as the PRIMARY trigger.
# Cron polling remains as a safety net only.

set -euo pipefail

ENGINE_NAME="${1:-unknown}"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
BASE="/home/jupiter/Lvl3Quant"
LOG="$BASE/logs/signal_callback.log"
LOCKFILE="/tmp/signal_callback.lock"

# Prevent concurrent runs (aggregator is not re-entrant)
if [ -f "$LOCKFILE" ]; then
    LOCK_AGE=$(( $(date +%s) - $(stat -c %Y "$LOCKFILE" 2>/dev/null || echo 0) ))
    if [ "$LOCK_AGE" -lt 60 ]; then
        echo "[$TIMESTAMP] SKIP: callback already running (age ${LOCK_AGE}s, triggered by $ENGINE_NAME)" >> "$LOG"
        exit 0
    fi
    # Stale lock (>60s), remove it
    rm -f "$LOCKFILE"
fi

echo $$ > "$LOCKFILE"
trap 'rm -f "$LOCKFILE"' EXIT

echo "[$TIMESTAMP] CALLBACK triggered by: $ENGINE_NAME" >> "$LOG"

# Step 1: Run the aggregator
echo "[$TIMESTAMP] Running aggregator..." >> "$LOG"
cd "$BASE"
AGG_OUTPUT=$(python3 paper_engines/agentic_signal_aggregator.py 2>&1) || true
echo "[$TIMESTAMP] Aggregator done" >> "$LOG"

# Step 2: Check if any signal crossed threshold
SIGNALS_FILE="$BASE/state/agentic_signals.json"
if [ ! -f "$SIGNALS_FILE" ]; then
    echo "[$TIMESTAMP] No signals file found, exiting" >> "$LOG"
    exit 0
fi

# Check for any ticker with confidence >= 0.78 using python
HAS_SIGNAL=$(python3 -c "
import json
try:
    with open('$SIGNALS_FILE') as f:
        data = json.load(f)
    signals = data.get('signals', [])
    # Handle both list format (agentic_signals) and dict format
    if isinstance(signals, list):
        for sig in signals:
            ticker = sig.get('ticker', '?')
            conf = sig.get('confidence_score', sig.get('confidence', 0))
            if isinstance(conf, (int, float)) and conf >= 0.78:
                print(f'YES:{ticker}:{conf:.2f}')
                break
        else:
            print('NO')
    elif isinstance(signals, dict):
        for ticker, info in signals.items():
            conf = info.get('confidence', info.get('score', 0))
            if isinstance(conf, (int, float)) and conf >= 0.78:
                print(f'YES:{ticker}:{conf:.2f}')
                break
        else:
            print('NO')
    else:
        print('NO')
except Exception as e:
    print(f'ERROR:{e}')
" 2>&1)

echo "[$TIMESTAMP] Signal check: $HAS_SIGNAL" >> "$LOG"

if [[ "$HAS_SIGNAL" == NO* ]] || [[ "$HAS_SIGNAL" == ERROR* ]]; then
    echo "[$TIMESTAMP] No signals above threshold. Done." >> "$LOG"
    exit 0
fi

# Step 3: Signal found! Run pre-validator
echo "[$TIMESTAMP] HIGH SIGNAL DETECTED ($HAS_SIGNAL) — running pre-validator..." >> "$LOG"
PRE_OUTPUT=$(python3 scripts/execution_pre_validator.py 2>&1) || true
echo "[$TIMESTAMP] Pre-validator done" >> "$LOG"

# Step 4: Check if execution_ready.json has actionable trades
READY_FILE="$BASE/state/execution_ready.json"
if [ ! -f "$READY_FILE" ]; then
    echo "[$TIMESTAMP] No execution_ready.json, exiting" >> "$LOG"
    exit 0
fi

HAS_TRADES=$(python3 -c "
import json
try:
    with open('$READY_FILE') as f:
        data = json.load(f)
    trades = data.get('trades', data.get('ready', []))
    if trades and len(trades) > 0:
        print(f'YES:{len(trades)} trades ready')
    else:
        print('NO')
except Exception as e:
    print(f'ERROR:{e}')
" 2>&1)

echo "[$TIMESTAMP] Trade check: $HAS_TRADES" >> "$LOG"

if [[ "$HAS_TRADES" == NO* ]] || [[ "$HAS_TRADES" == ERROR* ]]; then
    echo "[$TIMESTAMP] Pre-validator rejected all signals. Done." >> "$LOG"
    exit 0
fi

# Step 5: FIRE! Inject execution prompt to wake Claude
echo "[$TIMESTAMP] FIRING EXECUTION INJECT ($HAS_TRADES) — triggered by $ENGINE_NAME" >> "$LOG"
"$BASE/scripts/autonomy_inject.sh" "SIGNAL_CALLBACK_EXECUTION: Paper engine '$ENGINE_NAME' just fired a signal above threshold. Pre-validator has approved trades. Check state/execution_ready.json and EXECUTE IMMEDIATELY. This is event-driven per HC #791 — do not defer."

echo "[$TIMESTAMP] Inject sent. Full chain complete: engine → aggregator → pre-validator → inject" >> "$LOG"
