#!/bin/bash
# pre_action_check.sh — HC #469 R6(d) — pre-send / pre-dispatch checklist.
#
# Source this from autonomy_inject.sh and / or run before every Discord send.
# Writes warnings to accountability.log. Non-blocking by default (return 0)
# unless STRICT=1 is set — then returns nonzero on any failure.

set -uo pipefail

LVL3=/home/jupiter/Lvl3Quant
ACC_LOG=$LVL3/logs/accountability.log
TS=$(date '+%Y-%m-%d %H:%M:%S')
STRICT=${STRICT:-0}
warnings=0

flag() {
    echo "[$TS] PRE_ACTION_WARNING: $*" >> "$ACC_LOG"
    warnings=$((warnings + 1))
}

# Check 1: any node idle > 10 min?
if [[ -f $LVL3/logs/idle_watchdog/cron.log ]]; then
    recent=$(tail -20 $LVL3/logs/idle_watchdog/cron.log 2>/dev/null || echo "")
    if echo "$recent" | grep -qE "idle for [0-9]{2,}\s*min|IDLE_ALERT"; then
        flag "node(s) idle — check $LVL3/logs/idle_watchdog/cron.log"
    fi
fi

# Check 2: is the most recent stream_backtest_v2 report using canonical FIFO?
latest_report=$(ls -t $LVL3/output/stream_backtest_v2/*REPORT*.md 2>/dev/null | head -1)
if [[ -n "$latest_report" && ! "$latest_report" =~ canonical_fifo ]]; then
    flag "most recent report '$latest_report' is NOT canonical FIFO — only smoke-check valid"
fi

# Check 3: most recent report uses 40+ days?
if [[ -n "$latest_report" ]]; then
    days_count=$(grep -oP '\b[0-9]{1,3}\s*(?:days|OOT|day)\b' "$latest_report" 2>/dev/null | head -3 | grep -oP '[0-9]+' | sort -rn | head -1)
    if [[ -n "$days_count" && "$days_count" -lt 40 ]]; then
        flag "most recent report claims $days_count days — below the 40-day threshold (HC #469 R2)"
    fi
fi

# Check 4: time-based exit baseline used as production claim?
if [[ -n "$latest_report" ]] && grep -qiE "production exit|tradable exit|exit policy" "$latest_report" 2>/dev/null; then
    if ! grep -qiE "adaptive|stream.continuation|learned" "$latest_report" 2>/dev/null; then
        flag "report claims a production exit but doesn't reference adaptive/stream/learned (HC #469 R4)"
    fi
fi

if [[ $warnings -gt 0 ]]; then
    echo "[$TS] PRE_ACTION_CHECK: $warnings warning(s) logged to $ACC_LOG" >&2
    if [[ "$STRICT" -eq 1 ]]; then
        exit 2
    fi
fi

exit 0
