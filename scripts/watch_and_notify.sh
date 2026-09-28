#!/bin/bash
# watch_and_notify.sh - poll a PID, send Discord when it dies.
# Created per HC #476 R4 (mandatory completion/crash notifications).
#
# Usage: watch_and_notify.sh <PID> <JOB_NAME> <LOG_PATH> [OUTPUT_FILE_TO_CHECK]
# Example: watch_and_notify.sh 2051429 "HC475_AB" /tmp/hc475_ab.log /home/jupiter/Lvl3Quant/output/hc475_ab/done.flag

set -u
PID="${1:?need pid}"
JOB_NAME="${2:?need job name}"
LOG_PATH="${3:?need log path}"
OUTPUT_FILE="${4:-}"

NOTIFY_JS="/home/jupiter/teleclaude-main/utils/discord_notify.js"
POLL_S=60

echo "[watch_and_notify] watching pid=$PID job=$JOB_NAME log=$LOG_PATH"

while kill -0 "$PID" 2>/dev/null; do
  sleep "$POLL_S"
done

# Process gone. Determine success vs crash.
STATUS="CRASH"
DETAIL=""
if [[ -n "$OUTPUT_FILE" ]] && [[ -e "$OUTPUT_FILE" ]]; then
  STATUS="DONE"
fi

# If no explicit output file, infer from log tail: presence of "Traceback" or "Error" => CRASH; "done" or "saved" => DONE.
if [[ -z "$OUTPUT_FILE" ]] && [[ -e "$LOG_PATH" ]]; then
  if tail -50 "$LOG_PATH" | grep -qiE "traceback|error:|killed|oom"; then
    STATUS="CRASH"
  elif tail -50 "$LOG_PATH" | grep -qiE "saved|complete|done|finished"; then
    STATUS="DONE"
  fi
fi

# Take last meaningful line of log for context.
if [[ -e "$LOG_PATH" ]]; then
  DETAIL=$(tail -3 "$LOG_PATH" | tr '\n' ' ' | cut -c1-200)
fi

MSG="**${JOB_NAME}**: ${STATUS}. Last log: ${DETAIL}"
node "$NOTIFY_JS" "$MSG" systemStatus 2>/dev/null || \
  node -e "const n=require('$NOTIFY_JS'); n.send(process.argv[1],'systemStatus').then(()=>process.exit(0)).catch(e=>{console.error(e);process.exit(1);});" "$MSG"

echo "[watch_and_notify] notified: $STATUS"
