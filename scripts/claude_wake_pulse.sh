#!/bin/bash
# claude_wake_pulse.sh — HC #273 cross-session wake-up
# Spawns a fresh `claude -p` session every 35 min that:
#  - reads SESSION_STATE.md + DIRECTIVES.md
#  - checks cluster state
#  - acts on idle GPUs / crashed runs / completed sweeps
#  - posts to Discord ONLY if action taken or anomaly found
#
# This is the durable fix for "Discord posts don't wake Claude."
# Even if no live Claude session exists, this cron spins one up.
#
# Created 2026-05-10 02:04 ET in response to user "fix the reason" (HC #273).
#
set -u
LOG=/home/jupiter/Lvl3Quant/logs/claude_wake_pulse.log
LOCK=/tmp/claude_wake_pulse.lock
CLAUDE_BIN=/home/jupiter/.local/bin/claude

# Single-flight: skip if previous pulse still running
if [ -f "$LOCK" ]; then
  PID=$(cat "$LOCK" 2>/dev/null)
  if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
    echo "[$(date)] previous pulse PID=$PID still running — skipping" >> "$LOG"
    exit 0
  fi
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

PROMPT='AUTONOMOUS WAKE-PULSE (cross-session, system cron). HC #273 mandatory channel.
1) Read /home/jupiter/Lvl3Quant/SESSION_STATE.md (last_updated + current state)
2) Read /home/jupiter/Lvl3Quant/DIRECTIVES.md top 3 HCs
3) Run /cluster-status (or check Jupiter procs + Neptune nvidia-smi via SSH)
4) If Neptune GPU idle AND book_features rebuild done AND enriched parquets ready: dispatch Phase 3 LGBM training to Neptune (rsync + SSH-launch).
5) If Phase 3 just completed: rsync results back to Jupiter, run Phase 4 validator (HC #271 dominance + regime).
6) If sweep-result-watcher posted DONE since last pulse: evaluate vs HC #254/#272, post headline.
7) ONLY post to Discord if (a) action taken, (b) anomaly found, or (c) result worth reporting. Silence is fine.
8) Update SESSION_STATE.md timestamp.
Be terse. No "I will now check..." narration. Just act and report.'

echo "[$(date)] wake-pulse firing" >> "$LOG"
timeout 600 "$CLAUDE_BIN" -p "$PROMPT" >> "$LOG" 2>&1
RC=$?
echo "[$(date)] wake-pulse done rc=$RC" >> "$LOG"
