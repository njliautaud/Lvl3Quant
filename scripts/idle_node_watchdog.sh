#!/bin/bash
# HC #431 R1 + R2: OS-level idle node watchdog.
# Cron-fires every 10 min, checks GPU/CPU idle, injects autonomy prompt
# to teleclaude inject endpoint when nodes are idle > 15 min.
#
# Durability: cron survives reboots/session restarts. Inject endpoint is
# the only path that fires prompts to Claude regardless of session state.
set -u

STATE_DIR="/home/jupiter/Lvl3Quant/logs/idle_watchdog"
mkdir -p "$STATE_DIR"
LOG="$STATE_DIR/watchdog.log"
NOW=$(date +%s)
TS=$(date '+%Y-%m-%d %H:%M:%S')

# Idle threshold (seconds)
IDLE_THRESH_SEC=900   # 15 min
NEPTUNE_SSH="nick@neptune"
INJECT_URL="http://127.0.0.1:7731/inject"

log() { echo "[$TS] $*" >> "$LOG"; }

# Suppression check: if $STATE_DIR/{node}_idle_acceptable.flag exists and the
# first line is "valid_until=<unix_ts>" with ts > NOW, suppress the alert.
# This lets the operator declare "idle is the correct state right now per HC #X"
# and stop the 30-min cron from repeatedly waking Claude on a node that has no
# valid dispatchable work. Remove or expire the flag to re-enable alerts.
idle_alert_suppressed() {
  local node="$1"
  local flag="$STATE_DIR/${node}_idle_acceptable.flag"
  [ -f "$flag" ] || return 1
  local valid_until
  valid_until=$(grep -m1 '^valid_until=' "$flag" 2>/dev/null | cut -d= -f2 | tr -d ' ')
  [ -z "$valid_until" ] && return 1
  if [ "$valid_until" -gt "$NOW" ] 2>/dev/null; then
    local reason
    reason=$(grep -m1 '^reason=' "$flag" 2>/dev/null | cut -d= -f2-)
    log "${node} idle alert SUPPRESSED by flag (valid_until=${valid_until}, reason='${reason:-unspecified}')"
    return 0
  fi
  # Expired flag — clean up so the alert resumes
  log "${node} idle suppression flag EXPIRED at ${valid_until}, removing"
  rm -f "$flag"
  return 1
}

# -----------------------------------------------------------------------------
# Check Neptune GPU
# HC #491 R5 fix 2026-05-22 23:30 ET: ATOMIC single-SSH probe. Two separate
# SSH calls were racing — when the second timed out, train_procs=0 silently
# even with v342_run_oot_inference.py alive. Result: 5 false IDLE alerts in
# one session. Single-trip query + GPU-mem-guard + fail-safe (any partial
# failure → treat as BUSY, NEVER fire idle from incomplete data).
# -----------------------------------------------------------------------------
NEPTUNE_PROBE=$(ssh -o ConnectTimeout=8 -o BatchMode=yes "$NEPTUNE_SSH" \
  "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | head -1; echo '---'; ps -eo cmd= 2>/dev/null | grep -iE 'train_cnn|train_v2|train_event|run_oot|v342_run|book_cnn|cnn_mamba|patchtst|inference' | grep -v grep | wc -l" 2>/dev/null)
NEPTUNE_GPU=$(echo "$NEPTUNE_PROBE" | awk -F',' '/^[0-9]/ {gsub(/ /,"",$1); print $1; exit}')
NEPTUNE_GPU_MEM=$(echo "$NEPTUNE_PROBE" | awk -F',' '/^[0-9]/ {gsub(/ /,"",$2); print $2; exit}')
NEPTUNE_TRAIN_PROCS=$(echo "$NEPTUNE_PROBE" | awk '/^---/{f=1; next} f{print; exit}' | tr -d ' ')
NEPTUNE_TRAIN_PROCS=${NEPTUNE_TRAIN_PROCS:-0}
NEPTUNE_GPU_MEM=${NEPTUNE_GPU_MEM:-0}

if [ -z "$NEPTUNE_GPU" ] || ! echo "$NEPTUNE_PROBE" | grep -q '^---'; then
  log "Neptune SSH/GPU probe INCOMPLETE — treating as BUSY (fail-safe), skipping idle eval"
  # Refresh state file so a future flaky run does not accumulate phantom idle time
  echo "$NOW" > "$STATE_DIR/neptune_last_busy.ts"
else
  STATE_FILE="$STATE_DIR/neptune_last_busy.ts"
  # HC #491 R5: BUSY if ANY of: gpu_util≥5, train proc alive, OR GPU memory ≥200 MiB.
  # GPU-mem-held is the PRIMARY busy signal on Neptune because hidepid=2 (or similar)
  # makes ps-via-SSH-as-nick miss processes owned by other UIDs. Empty/idle GPU
  # holds <50 MiB. Any active model loaded in eval mode holds 200 MiB+.
  if [ "$NEPTUNE_GPU" -ge 5 ] || [ "$NEPTUNE_TRAIN_PROCS" -ge 1 ] || [ "$NEPTUNE_GPU_MEM" -ge 200 ]; then
    echo "$NOW" > "$STATE_FILE"
    log "Neptune GPU=${NEPTUNE_GPU}% mem=${NEPTUNE_GPU_MEM}MiB procs=${NEPTUNE_TRAIN_PROCS} (BUSY)"
  else
    LAST_BUSY=$(cat "$STATE_FILE" 2>/dev/null || echo "$NOW")
    IDLE_FOR=$(( NOW - LAST_BUSY ))
    log "Neptune GPU=${NEPTUNE_GPU}% (idle ${IDLE_FOR}s)"
    if [ "$IDLE_FOR" -ge "$IDLE_THRESH_SEC" ] && ! idle_alert_suppressed neptune; then
      # Fire autonomy prompt — Claude must launch work
      MSG="IDLE_ALERT: Neptune GPU idle ${IDLE_FOR}s, GPU=${NEPTUNE_GPU}%. READ /home/jupiter/Lvl3Quant/DIRECTIVES.md (top 30 lines) and /home/jupiter/Lvl3Quant/SESSION_STATE.md (top 30 lines) FIRST. Dispatch only from the CURRENT highest-numbered HC's valid lanes. If no GPU work exists in current lanes, set suppression flag at /home/jupiter/Lvl3Quant/logs/idle_watchdog/neptune_idle_acceptable.flag and STOP."
      ESCAPED=$(printf '%s' "$MSG" | python3 -c 'import sys, json; print(json.dumps(sys.stdin.read()))')
      RESP=$(curl -s -m 5 -X POST "$INJECT_URL" -H 'Content-Type: application/json' -d "{\"message\":${ESCAPED}}" 2>&1)
      log "Neptune IDLE injected: resp='${RESP}'"
      # Reset to avoid spam (re-warn after another full threshold)
      echo "$NOW" > "$STATE_FILE"
    fi
  fi
fi

# -----------------------------------------------------------------------------
# Check Jupiter local CPU (training/sweep processes)
# -----------------------------------------------------------------------------
# Look for any optuna/training/sweep python proc owned by jupiter
JUP_BUSY=$(ps -u jupiter -o cmd= 2>/dev/null | \
  grep -iE "optuna|train_cnn|train_event|patchtst|sweep|fifo_validate|inference|launcher" | \
  grep -v grep | wc -l)

STATE_FILE="$STATE_DIR/jupiter_last_busy.ts"
if [ "$JUP_BUSY" -ge 1 ]; then
  echo "$NOW" > "$STATE_FILE"
  log "Jupiter CPU procs=${JUP_BUSY} (BUSY)"
else
  LAST_BUSY=$(cat "$STATE_FILE" 2>/dev/null || echo "$NOW")
  IDLE_FOR=$(( NOW - LAST_BUSY ))
  log "Jupiter procs=0 (idle ${IDLE_FOR}s)"
  if [ "$IDLE_FOR" -ge "$IDLE_THRESH_SEC" ] && ! idle_alert_suppressed jupiter; then
    MSG="IDLE_ALERT: Jupiter CPU idle ${IDLE_FOR}s. READ /home/jupiter/Lvl3Quant/DIRECTIVES.md (top 200 lines) FIRST and dispatch only from the CURRENT highest-numbered HC's valid lanes. Per HC #518 R1+R6 (newest as of 2026-06-03 15:55 ET), ES taker-execution experiments are DEAD; the prior e1-e6 menu naming canonical-FIFO sweeps, adaptive-exit IL, ES-targeting microstructure features, latency-aware ES FIFO replay is SUPERSEDED if it's aimed at ES taker. Valid current lanes: HC #518 R5 (SPY-shares port scoping — data-feed/broker research, infra mapping, timeline — fits Jupiter CPU well, no spend without sign-off), HC #518 R6 (raw-MBO migration to Razer — file orchestration on Jupiter). Also valid: any Meridian work per HC #516-522 (the user's active project). If no dispatchable lane, report 'Jupiter idle is correct state per HC #518' once and STOP re-firing. If a newer HC supersedes #518, follow that instead."
    ESCAPED=$(printf '%s' "$MSG" | python3 -c 'import sys, json; print(json.dumps(sys.stdin.read()))')
    RESP=$(curl -s -m 5 -X POST "$INJECT_URL" -H 'Content-Type: application/json' -d "{\"message\":${ESCAPED}}" 2>&1)
    log "Jupiter IDLE injected: resp='${RESP}'"
    echo "$NOW" > "$STATE_FILE"
  fi
fi

# -----------------------------------------------------------------------------
# Check Razer GPU (HC #468 R5 — every alpha sweep gets a parallel Razer dispatch)
# HC #469 R6(a) — auto-dispatch when Razer GPU idle ≥10min
# -----------------------------------------------------------------------------
RAZER_SSH="claude@razer"
# HC #491 R5 fix 2026-05-22: ATOMIC single-SSH probe (same race fix as Neptune).
RAZER_PROBE=$(sshpass -p "${CLUSTER_SSH_PASSWORD:-}" ssh -o ConnectTimeout=8 -o StrictHostKeyChecking=no "$RAZER_SSH" \
  "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits & powershell -Command \"(Get-Process python -ErrorAction SilentlyContinue | Measure-Object).Count\"" 2>/dev/null | tr -d '\r')
RAZER_GPU=$(echo "$RAZER_PROBE" | awk -F',' '/^[0-9]/ {gsub(/ /,"",$1); print $1; exit}')
RAZER_GPU_MEM=$(echo "$RAZER_PROBE" | awk -F',' '/^[0-9]/ {gsub(/ /,"",$2); print $2; exit}')
RAZER_PROCS=$(echo "$RAZER_PROBE" | grep -E '^[0-9]+$' | tail -1 | tr -d ' ')
RAZER_GPU=${RAZER_GPU:-}
RAZER_GPU_MEM=${RAZER_GPU_MEM:-0}
RAZER_PROCS=${RAZER_PROCS:-0}
if [ -z "$RAZER_GPU" ] || [ -z "$RAZER_PROCS" ]; then
  log "Razer SSH/probe INCOMPLETE — treating as BUSY (fail-safe)"
  echo "$NOW" > "$STATE_DIR/razer_last_busy.ts"
elif [ -n "${RAZER_GPU:-}" ]; then
  STATE_FILE="$STATE_DIR/razer_last_busy.ts"
  # HC #491 R5: BUSY if gpu_util≥5, python proc alive, OR VRAM ≥200 MiB held.
  if [ "$RAZER_GPU" -ge 5 ] || [ "$RAZER_PROCS" -ge 1 ] || [ "$RAZER_GPU_MEM" -ge 200 ]; then
    echo "$NOW" > "$STATE_FILE"
    log "Razer GPU=${RAZER_GPU}% mem=${RAZER_GPU_MEM}MiB procs=${RAZER_PROCS} (BUSY)"
  else
    LAST_BUSY=$(cat "$STATE_FILE" 2>/dev/null || echo "$NOW")
    IDLE_FOR=$(( NOW - LAST_BUSY ))
    log "Razer GPU=${RAZER_GPU}% (idle ${IDLE_FOR}s)"
    if [ "$IDLE_FOR" -ge 600 ] && ! idle_alert_suppressed razer; then  # 10 min — stricter than Neptune/Jupiter per HC #468 R5
      MSG="IDLE_ALERT: Razer GPU idle ${IDLE_FOR}s. READ /home/jupiter/Lvl3Quant/DIRECTIVES.md (top 200 lines) FIRST and dispatch only from the CURRENT highest-numbered HC's valid lanes. Per HC #518 R1+R6 (newest as of 2026-06-03 15:55 ET), ES taker-execution experiments are DEAD and the e1-e5 menu (confluence meta, queue predictor, MFE relabel, PatchTST extension, head ablation) is EXPLICITLY SUPERSEDED — do not dispatch those. Valid current lanes: HC #518 R5 (SPY-shares port scoping, requires user sign-off for spend), HC #518 R6 (raw-MBO migration to Razer — standing background work). If neither lane has dispatchable GPU work tonight, report 'Razer idle is correct state per HC #518' once and STOP re-firing. If a newer HC supersedes #518, follow that instead."
      ESCAPED=$(printf '%s' "$MSG" | python3 -c 'import sys, json; print(json.dumps(sys.stdin.read()))')
      RESP=$(curl -s -m 5 -X POST "$INJECT_URL" -H 'Content-Type: application/json' -d "{\"message\":${ESCAPED}}" 2>&1)
      log "Razer IDLE injected: resp='${RESP}'"
      echo "$NOW" > "$STATE_FILE"
    fi
  fi
fi

# -----------------------------------------------------------------------------
# Check Razer live stack heartbeat (paper trader)
# -----------------------------------------------------------------------------
# Razer heartbeat file is sync'd via razer_npz_sync.sh; check freshness if present
RAZER_HB="/home/jupiter/Lvl3Quant/live_trading/status/razer_heartbeat.json"
if [ -f "$RAZER_HB" ]; then
  HB_AGE=$(( NOW - $(stat -c %Y "$RAZER_HB" 2>/dev/null || echo "$NOW") ))
  log "Razer heartbeat age=${HB_AGE}s"
  if [ "$HB_AGE" -gt 300 ]; then
    MSG="HEARTBEAT_STALE (HC #431): Razer heartbeat is ${HB_AGE}s old (>300s). Paper trader may be down. SSH Razer NOW and verify spec trader PID alive."
    ESCAPED=$(printf '%s' "$MSG" | python3 -c 'import sys, json; print(json.dumps(sys.stdin.read()))')
    curl -s -m 5 -X POST "$INJECT_URL" -H 'Content-Type: application/json' -d "{\"message\":${ESCAPED}}" >/dev/null 2>&1
    log "Razer heartbeat-stale injected"
  fi
fi

# Rotate log if >5MB
LOG_SIZE=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
if [ "$LOG_SIZE" -gt 5242880 ]; then
  tail -1000 "$LOG" > "${LOG}.tmp" && mv "${LOG}.tmp" "$LOG"
fi
