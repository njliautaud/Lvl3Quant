#!/bin/bash
# Durable autonomy: POST a prompt to teleclaude inject endpoint.
# Fires regardless of Claude session state. Survives reboots.
set -u
# --- HC #817 USAGE GATE (2026-09-28) ---
# Usage: autonomy_inject.sh [--priority critical|normal|low] "<message>"
#   (or env INJECT_PRIORITY=...). critical = order/exit/trade-safety -> ALWAYS sent.
#   Unlabeled callers are auto-classified by keywords below (trade words -> critical).
#   INJECT_DRY_RUN=1 -> print/log what WOULD be sent, send nothing.
PRIORITY="${INJECT_PRIORITY:-}"
if [ "${1:-}" = "--priority" ]; then PRIORITY="${2:-}"; shift 2; fi
RAW_MSG="${1:-AUTO_CHECK: routine 35-min check. Run /deep-check skill.}"
if [ -z "$PRIORITY" ]; then
  if printf '%s' "$RAW_MSG" | head -c 600 | grep -qiE 'exit|trigger|stop|kill|execut|order|position|flip|reversal|capture|spread|trade|buy|sell|bracket|expir|fill|guard'; then
    PRIORITY=critical
  elif printf '%s' "$RAW_MSG" | head -c 200 | grep -qiE 'pulse|briefing|summary|heartbeat|productivity|status'; then
    PRIORITY=low
  else
    PRIORITY=normal
  fi
fi
if ! /home/jupiter/agent/bin/usage_gate.sh "$PRIORITY" "autonomy_inject: ${RAW_MSG:0:60}"; then
  echo "[$(date '+%F %T')] SKIPPED (usage_gate prio=$PRIORITY): ${RAW_MSG:0:60}" \
    >> /home/jupiter/Lvl3Quant/logs/autonomy_inject.log
  exit 0
fi
# --- CONTINUITY GATE: pause self-prompts while on the free fallback provider ---
# When Claude subscription usage is exhausted we run a LEAN fallback (tools+memory,
# no sub-agents, no self-prompts). Automatic pulses must NOT fire in that window.
source "$HOME/agent/fallback/preflight.sh" 2>/dev/null || true
if [ "${CONTINUITY_PROCEED:-1}" = "0" ]; then
  echo "[$(date '+%F %T')] SKIPPED (continuity lean/fallback): ${RAW_MSG:0:60}" \
    >> /home/jupiter/Lvl3Quant/logs/autonomy_inject.log
  exit 0
fi
# Prepend market status so Claude always knows the day/time/market state
# OPTIMIZATION: Skip verbose market prepend if TOKEN_SAVER brake is active
if [ -f "$HOME/agent/state/TOKEN_SAVER" ]; then
  MSG="${RAW_MSG}"  # Brake on: skip context overhead
else
  MARKET_LINE=$(python3 /home/jupiter/Lvl3Quant/scripts/market_status.py --oneliner 2>/dev/null || echo "[market_status unavailable]")
  MSG="${MARKET_LINE} | ${RAW_MSG}"
fi
# JSON-escape the message (replace \, ", newlines)
ESCAPED=$(printf '%s' "$MSG" | python3 -c 'import sys, json; print(json.dumps(sys.stdin.read()))')
if [ "${INJECT_DRY_RUN:-0}" = "1" ]; then
  echo "[DRY-RUN] would inject prio=$PRIORITY len=${#MSG}: ${MSG:0:160}"
  echo "[$(date '+%F %T')] DRYRUN prio=$PRIORITY msg='${MSG:0:80}'" >> /home/jupiter/Lvl3Quant/logs/autonomy_inject.log
  exit 0
fi
# stamp the pulse so the continuity guard can detect a non-responding (exhausted) session
date +%s > "$HOME/agent/state/last_pulse" 2>/dev/null || true
RESP=$(curl -s -m 5 -X POST http://127.0.0.1:7731/inject \
  -H 'Content-Type: application/json' \
  -d "{\"message\":${ESCAPED}}" 2>&1)
echo "[$(date '+%Y-%m-%d %H:%M:%S')] prio=${PRIORITY} resp='${RESP}' msg='${MSG:0:80}'" >> /home/jupiter/Lvl3Quant/logs/autonomy_inject.log
