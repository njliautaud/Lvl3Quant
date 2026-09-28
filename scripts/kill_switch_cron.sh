#!/bin/bash
# kill_switch_cron.sh — Runs kill switch monitor every 15 min during market hours
# Cron: */15 9-16 * * 1-5  (covers 9:30-16:00 ET via the 9:30 offset check below)
#
# Install: crontab -e → add:
#   */15 9-16 * * 1-5 /home/jupiter/Lvl3Quant/scripts/kill_switch_cron.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MONITOR="/home/jupiter/Lvl3Quant/live_trading_linux/kill_switch_monitor.py"
LOG_DIR="/home/jupiter/Lvl3Quant/live_trading_linux/kill_switch_state"

mkdir -p "$LOG_DIR"

# Check if market is open (ET timezone)
HOUR_ET=$(TZ='America/New_York' date +%H)
MIN_ET=$(TZ='America/New_York' date +%M)
DOW=$(TZ='America/New_York' date +%u)  # 1=Mon, 7=Sun

# Skip weekends
if [ "$DOW" -gt 5 ]; then
    exit 0
fi

# Skip before 9:30 ET and after 16:00 ET
if [ "$HOUR_ET" -lt 9 ] || ([ "$HOUR_ET" -eq 9 ] && [ "$MIN_ET" -lt 30 ]); then
    exit 0
fi
if [ "$HOUR_ET" -ge 16 ]; then
    exit 0
fi

# Run the kill switch check with --alert flag (sends to inject endpoint if active)
cd /home/jupiter/Lvl3Quant
python3 "$MONITOR" --alert >> "$LOG_DIR/cron_output.log" 2>&1 || true
