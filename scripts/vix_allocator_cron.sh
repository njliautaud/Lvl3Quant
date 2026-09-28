#!/bin/bash
# VIX Daily Allocator — runs at 3:30 PM ET, injects result into Claude for execution
# If action != HOLD, Claude should execute via Robinhood MCP

set -u
LOG="/home/jupiter/Lvl3Quant/logs/vix_allocator.log"
cd /home/jupiter/Lvl3Quant

echo "=== VIX Allocator Run $(date) ===" >> "$LOG"
OUTPUT=$(python3 -u scripts/growth_research/vix_daily_allocator.py 2>&1)
echo "$OUTPUT" >> "$LOG"

# Extract summary line
SUMMARY=$(echo "$OUTPUT" | grep "^--- SUMMARY:" | head -1)
ACTION=$(echo "$OUTPUT" | grep "^Action:" | head -1)

if [ -z "$SUMMARY" ]; then
    SUMMARY="VIX allocator failed — check logs"
fi

# Inject into Claude session for action
/home/jupiter/Lvl3Quant/scripts/autonomy_inject.sh "VIX_ALLOCATOR_SIGNAL: $SUMMARY. $ACTION. If action is BUY or SELL, execute on Robinhood account ${ROBINHOOD_ACCOUNT_NUMBER:-} using place_equity_order MCP tool. If HOLD, no action needed. Check actual UPRO positions before trading (use get_equity_positions)."
