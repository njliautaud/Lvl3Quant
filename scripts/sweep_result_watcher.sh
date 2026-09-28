#!/bin/bash
# sweep_result_watcher.sh — HC #269 implementation
# Tails topx_sweep_master.log; on each new "DONE" line, extracts the matching
# log file's raw FIFO result block (per HC #268 — no implied $/day) and POSTs
# to Discord #general via the bot token (mirrors persistent_monitor.js mechanism).
# Posting to #general means the teleclaude bridge wakes Claude as if user pinged.
set -u
LOGDIR=/home/jupiter/Lvl3Quant/logs
MASTER=$LOGDIR/topx_sweep_master.log
GENERAL_CHANNEL_ID="${DISCORD_GENERAL_CHANNEL_ID:-}"
TOKEN=$(python3 -c "import json; print(json.load(open('/home/jupiter/teleclaude-main/config.json'))['discordToken'])" 2>/dev/null)
STATE=/home/jupiter/Lvl3Quant/logs/.sweep_watcher_seen

if [ -z "$TOKEN" ]; then
  echo "[$(date +%H:%M:%S)] FATAL: no discord token" >&2
  exit 2
fi
touch "$STATE"

post_discord() {
  local msg="$1"
  # Discord limits a single message to 2000 chars. Truncate, escape JSON.
  local truncated
  truncated=$(printf "%s" "$msg" | head -c 1900)
  local payload
  payload=$(python3 -c "import json,sys; print(json.dumps({'content': sys.stdin.read()}))" <<< "$truncated")
  curl -sS -X POST \
    -H "Authorization: Bot $TOKEN" \
    -H "Content-Type: application/json" \
    -d "$payload" \
    "https://discord.com/api/v10/channels/${GENERAL_CHANNEL_ID}/messages" >/dev/null 2>&1
}

# HC #274(B): also POST to localhost inject endpoint so the live Claude session
# wakes immediately. Discord-bridge auto-relay was unreliable for bot messages.
post_inject() {
  local msg="$1"
  local truncated
  truncated=$(printf "%s" "$msg" | head -c 1900)
  local payload
  payload=$(python3 -c "import json,sys; print(json.dumps({'message': sys.stdin.read()}))" <<< "$truncated")
  curl -sS -m 5 -X POST \
    -H "Content-Type: application/json" \
    -d "$payload" \
    "http://127.0.0.1:7731/inject" >/dev/null 2>&1
}

extract_result_block() {
  # Find the most-recent log file whose name contains the tag and emit the summary block.
  local tag="$1"
  local f
  f=$(ls -1t "$LOGDIR"/fifo_replay_v3_*"${tag}"*.log 2>/dev/null | head -n 1)
  if [ -z "$f" ]; then
    echo "(no log file found for tag=$tag)"
    return
  fi
  echo "log: $(basename "$f")"
  grep -E "Folds with realized|Total realized|Sum gross|Sum NET|Avg ticks/trade gross|Avg ticks/trade NET|Median fold|% folds with positive|VERDICT|FAIL|PASS" "$f" 2>/dev/null | head -15
}

# Re-emit only NEW DONE lines past the seen-marker.
echo "[$(date +%H:%M:%S)] sweep_result_watcher started; tailing $MASTER" >> $LOGDIR/sweep_watcher.log
tail -n 0 -F "$MASTER" 2>/dev/null | while read -r line; do
  case "$line" in
    *" DONE")
      tag=$(echo "$line" | sed -E 's/^.*\] (.*) DONE$/\1/')
      block=$(extract_result_block "$tag")
      msg="**[FIFO sweep] ${tag} DONE** (HC #268 raw output)"$'\n''```'$'\n'"$block"$'\n''```'
      post_discord "$msg"
      post_inject "EVENT_TRIGGER [SWEEP_DONE]: $tag finished. Read latest result block, evaluate vs HC #254/#272 (NET ticks, %folds positive, fill%), update SESSION_STATE, post headline if interesting."
      echo "[$(date +%H:%M:%S)] posted+injected: $tag" >> $LOGDIR/sweep_watcher.log
      ;;
    *"sweep complete"*|*"ALL DONE"*)
      msg="**[FIFO sweep] ${line}** — chain finished, autonomous queue is empty. Next decision needed."
      post_discord "$msg"
      post_inject "EVENT_TRIGGER [SWEEP_CHAIN_DONE]: autonomous queue empty. Decide next experiment per DIRECTIVES (Phase 3 dispatch, new sweep dimension, or hold)."
      echo "[$(date +%H:%M:%S)] posted+injected COMPLETE: $line" >> $LOGDIR/sweep_watcher.log
      ;;
  esac
done
