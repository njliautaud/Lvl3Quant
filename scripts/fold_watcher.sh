#!/bin/bash
# fold_watcher.sh — polls MLflow for new completed folds in active EventCNN1D / fusion runs
# and triggers Claude (via /inject) to push the IC table to Discord.
# Bypasses the broken cron-%-truncation path. Designed to be cron-safe (no % in this script).
#
# Cron: */5 * * * * /home/jupiter/Lvl3Quant/scripts/fold_watcher.sh >> /home/jupiter/Lvl3Quant/logs/fold_watcher.log 2>&1
set -u
STATE_DIR=/home/jupiter/Lvl3Quant/state
mkdir -p "$STATE_DIR"
MLFLOW=http://localhost:5000
INJECT=http://127.0.0.1:7731/inject
LOG_PFX="[$(date '+%Y-%m-%d %H:%M:%S')]"

# Experiments to watch (id : name)
EXPERIMENTS=(
  "935985340201974967:EventDriven_CNN1D"
  "664968863836573362:FusionBakeoff_v1_mamba"
)

post_inject () {
  local msg="$1"
  local esc
  esc=$(printf '%s' "$msg" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))')
  curl -s -m 5 -X POST "$INJECT" -H 'Content-Type: application/json' \
    -d "{\"message\":${esc}}" 2>&1 | head -c 200
}

for entry in "${EXPERIMENTS[@]}"; do
  exp_id="${entry%%:*}"
  exp_name="${entry##*:}"

  # Find the latest RUNNING run in this experiment
  resp=$(curl -s -m 8 "$MLFLOW/api/2.0/mlflow/runs/search" \
    -H 'Content-Type: application/json' \
    -d "{\"experiment_ids\":[\"$exp_id\"],\"max_results\":3,\"order_by\":[\"start_time DESC\"]}")
  run_id=$(echo "$resp" | python3 -c 'import json,sys
try:
  d=json.load(sys.stdin); runs=d.get("runs",[])
  for r in runs:
    if r["info"]["status"]=="RUNNING":
      print(r["info"]["run_id"]); break
except: pass' 2>/dev/null)

  [ -z "$run_id" ] && { echo "$LOG_PFX $exp_name: no RUNNING run"; continue; }

  # Pull metrics, extract max fold index that has oot_ic_1s logged
  metrics=$(curl -s -m 5 "$MLFLOW/api/2.0/mlflow/runs/get?run_id=$run_id")
  max_fold=$(echo "$metrics" | python3 -c 'import json,sys,re
try:
  d=json.load(sys.stdin); ms=d.get("run",{}).get("data",{}).get("metrics",[])
  folds=set()
  for m in ms:
    k=m["key"]
    mt=re.match(r"oot_ic_1s_fold(\d+)", k)
    if mt: folds.add(int(mt.group(1)))
  print(max(folds) if folds else -1)
except: print(-1)' 2>/dev/null)

  [ "$max_fold" = "-1" ] && { echo "$LOG_PFX $exp_name run=$run_id no oot folds yet"; continue; }

  state_file="$STATE_DIR/fold_watcher_${exp_id}_${run_id}.state"
  last_seen=$(cat "$state_file" 2>/dev/null || echo -1)

  if [ "$max_fold" -gt "$last_seen" ]; then
    echo "$LOG_PFX $exp_name run=$run_id NEW_FOLD seen=$max_fold last=$last_seen"
    msg="FOLD_LANDED: $exp_name run=$run_id reached fold $max_fold (was $last_seen). ACTION: pull oot_ic_1s/5s/10s for ALL folds 0..$max_fold from MLflow run $run_id, build the per-fold IC table plus running concat IC, compare to CNN Mamba v2 baseline (IC_1s=0.255 IC_5s=0.125 IC_10s=0.090 fold-0), post to Discord general channel. Single message, table format. No more silent gaps per HC #27."
    post_inject "$msg"
    echo "$max_fold" > "$state_file"
  else
    echo "$LOG_PFX $exp_name run=$run_id no_new_fold (max=$max_fold)"
  fi
done
