#!/bin/bash
# external_trigger_daemon.sh — HC #274 event-driven external trigger
#
# Watches multiple event sources and POSTs to teleclaude inject endpoint
# (http://127.0.0.1:7731/inject) on each interesting event so the live
# Claude session wakes IMMEDIATELY (target latency <30 sec) instead of
# waiting for the next 35-min cron.
#
# Sources watched:
#   1. New enriched parquet drops in output/meta_lgbm_features/
#   2. New book_features NPZ drops in data/processed/mbo_book_features/
#   3. Neptune GPU idle↔busy transitions (polled every 60s)
#   4. Razer GPU idle↔busy transitions (polled every 60s)
#   5. Training process deaths (PIDs from /tmp/active_training_pids)
#
# Dedupe via state file. Single-flight via PM2.
#
# Created 2026-05-10 02:11 ET in response to user "BUILD an external trigger" (HC #274).
#
set -u

LOGDIR=/home/jupiter/Lvl3Quant/logs
LOG=$LOGDIR/external_trigger_daemon.log
STATE=$LOGDIR/.external_trigger_state
ROOT=/home/jupiter/Lvl3Quant
INJECT_URL=http://127.0.0.1:7731/inject

mkdir -p "$LOGDIR"
touch "$STATE"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }

post_inject() {
  local label="$1"
  local detail="$2"
  local msg="EVENT_TRIGGER [$label]: $detail"
  local payload
  payload=$(python3 -c "import json,sys; print(json.dumps({'message': sys.stdin.read()}))" <<< "$msg")
  local rc
  rc=$(curl -sS -m 5 -o /dev/null -w '%{http_code}' -X POST \
        -H "Content-Type: application/json" \
        -d "$payload" \
        "$INJECT_URL" 2>&1)
  log "inject label=$label http=$rc msg='${msg:0:120}'"
}

# Dedupe: returns 0 if event is NEW (caller should fire), 1 if seen.
seen_check() {
  local key="$1"
  if grep -qxF "$key" "$STATE" 2>/dev/null; then
    return 1
  fi
  echo "$key" >> "$STATE"
  return 0
}

# --- Source 1+2: Filesystem watchers (inotify) ---
watch_filesystems() {
  if ! command -v inotifywait >/dev/null 2>&1; then
    log "inotifywait not installed — falling back to polling"
    poll_filesystems &
    return
  fi
  inotifywait -m -e close_write,moved_to \
    "$ROOT/output/meta_lgbm_features/" \
    "$ROOT/data/processed/mbo_book_features/" \
    --format '%w|%f' 2>/dev/null | while IFS='|' read -r dir file; do
      [ -z "$file" ] && continue
      key="fs:${dir}${file}"
      seen_check "$key" || continue
      case "$file" in
        *_signals_enriched.parquet)
          post_inject "ENRICHED_PARQUET" "New file: $file. Phase 2 enricher produced output. Check if 46/46 dates done → fire Phase 3 LGBM gate on Neptune."
          ;;
        *_book_features.npz)
          post_inject "BOOK_FEATURES" "New file: $file. Book features rebuild progressing. Tally remaining count; when 0 → trigger Phase 2 enricher for these dates."
          ;;
        *.parquet|*.npz)
          # other parquet/npz drops — generic
          log "fs event ignored (not actionable): $file"
          seen_check "discard:$key" || true
          ;;
      esac
    done &
  log "inotify watcher started PID=$!"
}

poll_filesystems() {
  log "polling fallback active (inotify unavailable)"
  while true; do
    for d in "$ROOT/output/meta_lgbm_features" "$ROOT/data/processed/mbo_book_features"; do
      [ -d "$d" ] || continue
      find "$d" -type f -newer "$STATE" \( -name '*.parquet' -o -name '*.npz' \) 2>/dev/null | while read -r f; do
        bn=$(basename "$f")
        key="fs:$f"
        seen_check "$key" || continue
        case "$bn" in
          *_signals_enriched.parquet)
            post_inject "ENRICHED_PARQUET" "New: $bn"
            ;;
          *_book_features.npz)
            post_inject "BOOK_FEATURES" "New: $bn"
            ;;
        esac
      done
    done
    touch "$STATE"
    sleep 60
  done &
}

# --- Source 3+4: GPU idle/busy transitions ---
# HYSTERESIS FIX 2026-05-16 18:34 ET: require HYSTERESIS_COUNT consecutive reads
# of the new state before firing a transition event. Prevents false-positive
# pairs from single-sample nvidia-smi reads that land between training batches.
# At HYSTERESIS_COUNT=3 and 60s poll, real transitions confirmed within ~3 min.
#
# STEAM FALSE-POSITIVE FIX 2026-05-23 00:35 ET (HC #491 R5 — real fix that
# survives context resets): replace util-threshold (≥5%) with python-compute-app
# presence check. Neptune's steamwebhelper GUI rendering was crossing the 5%
# util threshold during overnight idle hours, firing 9+ false BUSY/IDLE
# transitions in 3.5h, each spawning a new Claude session and burning ~625K
# cache tokens. Training = python process on GPU. No python compute app = idle,
# regardless of GUI util. Reduces false events to ~0 while still catching real
# training starts/deaths within ~3 min via the existing hysteresis.
GPU_STATE_NEPTUNE=""
GPU_STATE_RAZER=""
NEPTUNE_CANDIDATE_STATE=""
NEPTUNE_CANDIDATE_COUNT=0
RAZER_CANDIDATE_STATE=""
RAZER_CANDIDATE_COUNT=0
HYSTERESIS_COUNT=3

# HC #499 R3 — nodes reserved for user (gaming). Skip ALL GPU-transition events
# (BUSY and IDLE) on these nodes; user game activity is not a research signal.
# Set via env: GAMING_NODES="neptune,foo". Default: neptune (per user 2026-05-30).
# HC #600 (2026-06-10, token-conscious): razer added. Razer is the LIVE host — its GPU
# idle/busy is driven by Windows desktop UI flapping around the 5% threshold and fired
# repeated useless session-waking injections. Live-stack health is monitored via QCC /
# paper-engine checks, not GPU util. Remove razer from this list when ES live inference
# resumes (Rithmic creds fixed) if GPU-transition events become meaningful again.
GAMING_NODES="${GAMING_NODES:-neptune,razer}"
gaming_skip() {
  # returns 0 if $1 (lowercase node name) is in $GAMING_NODES
  local n="$1"
  case ",${GAMING_NODES,,}," in
    *",$n,"*) return 0 ;;
    *) return 1 ;;
  esac
}

poll_gpus() {
  while true; do
    # HC #499 R3: skip Neptune polling entirely if reserved for user (gaming).
    if gaming_skip neptune; then
      :  # no-op; user controls neptune
    else
    # Neptune via SSH — single call returns "util|python_compute_app_count"
    nep_data=$(ssh -o ConnectTimeout=8 -o BatchMode=yes nick@neptune \
      "U=\$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '); P=\$(nvidia-smi --query-compute-apps=process_name --format=csv,noheader 2>/dev/null | grep -ic python); echo \"\${U:-0}|\${P:-0}\"" 2>/dev/null)
    nep_util=${nep_data%|*}
    nep_pyapp=${nep_data#*|}
    if [ -n "$nep_util" ] && [ -n "$nep_pyapp" ]; then
      # FIXED (HC #491 R5, 2026-05-23): busy iff EITHER a python compute app
      # is consuming GPU OR util >= 50%. The SSH piped double nvidia-smi query
      # intermittently fails the compute-apps part within the 3s ConnectTimeout,
      # returning pyapp=0 while training runs at 100% util. Contradictory
      # readings (high util + 0 pyapps) = SSH query bug, NOT idle.
      if [ "$nep_pyapp" -ge 1 ] 2>/dev/null; then
        current="busy"
      elif [ "$nep_util" -ge 50 ] 2>/dev/null; then
        current="busy"  # util >= 50% without pyapp = SSH query partial failure
      else
        current="idle"
      fi
      if [ -z "$GPU_STATE_NEPTUNE" ]; then
        GPU_STATE_NEPTUNE="$current"
      elif [ "$current" = "$GPU_STATE_NEPTUNE" ]; then
        NEPTUNE_CANDIDATE_STATE=""
        NEPTUNE_CANDIDATE_COUNT=0
      else
        if [ "$current" = "$NEPTUNE_CANDIDATE_STATE" ]; then
          NEPTUNE_CANDIDATE_COUNT=$((NEPTUNE_CANDIDATE_COUNT + 1))
        else
          NEPTUNE_CANDIDATE_STATE="$current"
          NEPTUNE_CANDIDATE_COUNT=1
        fi
        if [ "$NEPTUNE_CANDIDATE_COUNT" -ge "$HYSTERESIS_COUNT" ]; then
          post_inject "NEPTUNE_GPU_${current^^}" "Neptune RTX 3090 transitioned $GPU_STATE_NEPTUNE → $current (util=$nep_util%, confirmed ${HYSTERESIS_COUNT} consecutive reads). $([ "$current" = "idle" ] && echo "Dispatch next experiment per DIRECTIVES." || echo "Training started.")"
          GPU_STATE_NEPTUNE="$current"
          NEPTUNE_CANDIDATE_STATE=""
          NEPTUNE_CANDIDATE_COUNT=0
        fi
      fi
    fi
    fi  # end gaming_skip neptune guard

    # Razer (status from QCC daemon — already polled there, just read)
    # HC #600: honor the suppression list for razer too (was neptune-only).
    if gaming_skip razer; then
      :  # no-op; razer GPU transitions suppressed (live host, desktop-UI noise)
    else
    raz_util=$(curl -sS -m 3 http://localhost:3456/api/nodes 2>/dev/null | python3 -c "
import json, sys
try:
    nodes = json.load(sys.stdin)
    for n in (nodes.get('nodes') if isinstance(nodes, dict) else nodes) or []:
        if n.get('name') == 'razer':
            print(int(n.get('last_gpu_util') or 0))
            break
except Exception:
    pass
" 2>/dev/null)
    if [ -n "$raz_util" ]; then
      if [ "$raz_util" -ge 5 ] 2>/dev/null; then current="busy"; else current="idle"; fi
      if [ -z "$GPU_STATE_RAZER" ]; then
        GPU_STATE_RAZER="$current"
      elif [ "$current" = "$GPU_STATE_RAZER" ]; then
        RAZER_CANDIDATE_STATE=""
        RAZER_CANDIDATE_COUNT=0
      else
        if [ "$current" = "$RAZER_CANDIDATE_STATE" ]; then
          RAZER_CANDIDATE_COUNT=$((RAZER_CANDIDATE_COUNT + 1))
        else
          RAZER_CANDIDATE_STATE="$current"
          RAZER_CANDIDATE_COUNT=1
        fi
        if [ "$RAZER_CANDIDATE_COUNT" -ge "$HYSTERESIS_COUNT" ]; then
          post_inject "RAZER_GPU_${current^^}" "Razer RTX 3070 transitioned $GPU_STATE_RAZER → $current (util=$raz_util%, confirmed ${HYSTERESIS_COUNT} consecutive reads)."
          GPU_STATE_RAZER="$current"
          RAZER_CANDIDATE_STATE=""
          RAZER_CANDIDATE_COUNT=0
        fi
      fi
    fi
    fi  # end gaming_skip razer guard
    sleep 60
  done &
  log "GPU poller started PID=$! (HYSTERESIS_COUNT=$HYSTERESIS_COUNT)"
}

# --- Source 5: Training process deaths ---
poll_pids() {
  PIDFILE=/tmp/active_training_pids
  [ -f "$PIDFILE" ] || return
  while true; do
    if [ -f "$PIDFILE" ]; then
      while read -r pid label; do
        [ -z "$pid" ] && continue
        if ! kill -0 "$pid" 2>/dev/null; then
          key="dead:$pid:$label"
          seen_check "$key" || continue
          post_inject "TRAINING_DEAD" "PID $pid ($label) is gone. Diagnose: check logs, decide relaunch or pivot."
        fi
      done < "$PIDFILE"
    fi
    sleep 30
  done &
}

# --- Boot ---
log "=== external_trigger_daemon starting (HC #274) ==="
# HC #600 (2026-06-10): DAEMON_START is log-only — boot injections woke a Claude
# session on every daemon/pm2 restart for zero actionable content (token burn).
log "DAEMON_START: online. Watching: enriched parquets, book_features NPZ, GPU transitions (suppressed: $GAMING_NODES), training PIDs."

watch_filesystems
poll_gpus
poll_pids

# Keep parent alive
wait
