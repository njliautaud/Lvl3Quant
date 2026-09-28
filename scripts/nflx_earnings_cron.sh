#!/bin/bash
# NFLX Earnings Gap Check — One-shot cron for Jul 17 9:35 AM ET
# Runs the gap checker, captures output, injects into Claude session for action
set -u

LOG="/home/jupiter/Lvl3Quant/logs/nflx_earnings_gap.log"
SCRIPT="/home/jupiter/Lvl3Quant/live_trading_linux/nflx_earnings_gap_check.py"

echo "=== NFLX Earnings Gap Check $(date) ===" >> "$LOG"

# Run the gap checker and capture output
OUTPUT=$(python3 "$SCRIPT" 2>&1)
echo "$OUTPUT" >> "$LOG"

# Inject into Claude session with the full result
MSG="NFLX_EARNINGS_GAP_CHECK triggered at 9:35 AM. Here is the gap check output:

${OUTPUT}

INSTRUCTIONS: Send the gap result to the user on Discord (plain English summary). If gap >= 10%, look up NFLX option chains via Robinhood MCP (get_option_chains, get_option_instruments) for Jul 24 expiry, find ATM \$5-wide debit spread pricing, and report the trade setup. NOTE: Robinhood MCP only supports single-leg orders — remind user to place the spread via Robinhood app, or place each leg separately if user approves. Max budget \$110. If gap < 10%, just tell the user no trade today."

/home/jupiter/Lvl3Quant/scripts/autonomy_inject.sh "$MSG"

echo "Inject sent at $(date)" >> "$LOG"
