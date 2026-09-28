#!/bin/bash
# HC #590 P1-2: K=6 Megacap weekly-rebal miss watchdog.
# Runs Tuesday 10am: reads megacap_paper_state.json last_rebal_date, asserts
# it equals the most recent Monday. If not, fires QCC + Discord alert.
set -u

STATE_FILE="/home/jupiter/Lvl3Quant/live_trading_linux/data/megacap_paper_state.json"
LOG="/home/jupiter/Lvl3Quant/logs/megacap_weekly_check.log"
WEBHOOK="/home/jupiter/teleclaude-main/utils/webhook_notifier.js"
INJECT_SH="/home/jupiter/Lvl3Quant/scripts/autonomy_inject.sh"

mkdir -p "$(dirname "$LOG")"
TS=$(date '+%Y-%m-%d %H:%M:%S %Z')

if [ ! -f "$STATE_FILE" ]; then
    MSG="K=6 megacap state file MISSING — weekly rebal likely never ran."
    echo "[$TS] ALERT $MSG" >> "$LOG"
    [ -f "$WEBHOOK" ] && node "$WEBHOOK" "Megacap: $MSG" 2>/dev/null
    [ -f "$INJECT_SH" ] && "$INJECT_SH" "MEGACAP_MISSED: $MSG"
    exit 0
fi

# Expected last rebal date = most recent Monday (yesterday if today is Tue)
EXPECTED=$(date -d "last Monday" +%Y-%m-%d)
# If today IS Monday, "last Monday" is today itself
TODAY_DOW=$(date +%u)
if [ "$TODAY_DOW" = "1" ]; then
    EXPECTED=$(date +%Y-%m-%d)
fi

LAST=$(python3 -c "import json,sys
try:
    d=json.load(open('$STATE_FILE'))
    print(d.get('last_rebal_date','MISSING'))
except Exception as e:
    print('PARSE_ERROR:'+str(e))
" 2>/dev/null)

if [ "$LAST" = "$EXPECTED" ]; then
    echo "[$TS] OK last_rebal=$LAST expected=$EXPECTED" >> "$LOG"
else
    MSG="K=6 megacap last_rebal_date=$LAST but expected $EXPECTED. Monday 9:37 cron likely missed."
    echo "[$TS] ALERT $MSG" >> "$LOG"
    [ -f "$WEBHOOK" ] && node "$WEBHOOK" "Megacap: $MSG" 2>/dev/null
    [ -f "$INJECT_SH" ] && "$INJECT_SH" "MEGACAP_MISSED: $MSG Check logs/megacap_paper_rebal.log."
fi
