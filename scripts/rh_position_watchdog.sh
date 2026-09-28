#!/bin/bash
# RH Position Watchdog — verifies hourly position check cron exists
# Runs as a bash watchdog (survives session resets, no Python deps)
# If the RH position check cron line is missing, restores it immediately

CRON_PATTERN="rh_position_check"
INJECT_SCRIPT="/home/jupiter/Lvl3Quant/scripts/autonomy_inject.sh"
PROMPT_FILE="/home/jupiter/Lvl3Quant/scripts/prompts/rh_position_check.txt"
LOG="/home/jupiter/Lvl3Quant/logs/rh_watchdog.log"

mkdir -p "$(dirname "$LOG")"

if ! crontab -l 2>/dev/null | grep -q "$CRON_PATTERN"; then
    echo "[$(date)] ALERT: RH position check cron MISSING — restoring" >> "$LOG"
    (crontab -l 2>/dev/null; echo "# RH position monitor: hourly during market hours (9:30-16:00 ET, weekdays)"; echo "30 9,10,11,12,13,14,15 * * 1-5 $INJECT_SCRIPT \"\$(cat $PROMPT_FILE)\"") | crontab -
    echo "[$(date)] RESTORED RH position check cron" >> "$LOG"
else
    echo "[$(date)] OK: RH position check cron present" >> "$LOG"
fi
