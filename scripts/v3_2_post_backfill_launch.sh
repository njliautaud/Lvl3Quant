#!/bin/bash
# v3.2 post-backfill autonomous pipeline (HC #295 compliant)
# Polls until Tier 2/3 backfill completes, then rsyncs parquets + patched
# trainer to Neptune and launches fold 0 via the v3.2 launcher.
# All status flushed to logs/v3_2_post_backfill.log.

set -uo pipefail

LOG=/home/jupiter/Lvl3Quant/logs/v3_2_post_backfill.log
T2_DIR=/home/jupiter/Lvl3Quant/data/derived/tier2_orderflow_features_v1.parquet
T3_DIR=/home/jupiter/Lvl3Quant/data/derived/tier3_session_features_v1.parquet
NEPTUNE=nick@neptune
NEPTUNE_LVL3=/home/nick/Lvl3Quant
END_DATE=2026-04-29
BACKFILL_PROC=build_v3_2_tier_features

mkdir -p /home/jupiter/Lvl3Quant/logs

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

log "==== v3.2 post-backfill pipeline START ===="

# 1. Poll until backfill python is gone AND target end date written
while true; do
    alive=$(pgrep -f "$BACKFILL_PROC" | head -1)
    last=$(ls "$T2_DIR" 2>/dev/null | sort | tail -1 | sed 's/date=//')
    log "poll: alive_pid=$alive last_day=$last"
    if [ -z "$alive" ]; then
        if [ "$last" = "$END_DATE" ]; then
            log "backfill complete — last day=$last"
            break
        else
            log "WARN: backfill process gone but last day=$last (expected $END_DATE)"
            log "WARN: proceeding anyway with whatever days exist"
            break
        fi
    fi
    sleep 300
done

# 2. Day count sanity
n_t2=$(ls "$T2_DIR" 2>/dev/null | wc -l)
n_t3=$(ls "$T3_DIR" 2>/dev/null | wc -l)
t2_size=$(du -sh "$T2_DIR" 2>/dev/null | cut -f1)
t3_size=$(du -sh "$T3_DIR" 2>/dev/null | cut -f1)
log "T2: $n_t2 days ($t2_size)   T3: $n_t3 days ($t3_size)"

if [ "$n_t2" -lt 80 ] || [ "$n_t3" -lt 80 ]; then
    log "ERROR: too few days to launch fold 0 (need ≥80, got T2=$n_t2 T3=$n_t3). ABORT."
    exit 2
fi

# 3. Rsync parquets to Neptune
log "rsync T2 to Neptune..."
rsync -avz --delete "$T2_DIR/" "$NEPTUNE:$NEPTUNE_LVL3/data/derived/tier2_orderflow_features_v1.parquet/" >> "$LOG" 2>&1
log "rsync T2 done (rc=$?)"

log "rsync T3 to Neptune..."
rsync -avz --delete "$T3_DIR/" "$NEPTUNE:$NEPTUNE_LVL3/data/derived/tier3_session_features_v1.parquet/" >> "$LOG" 2>&1
log "rsync T3 done (rc=$?)"

# 4. Rsync patched trainer
log "rsync trainer..."
rsync -avz /home/jupiter/Lvl3Quant/alpha_discovery/deep_models/train_cnn_mamba_v3_2.py \
    "$NEPTUNE:$NEPTUNE_LVL3/alpha_discovery/deep_models/train_cnn_mamba_v3_2.py" >> "$LOG" 2>&1
log "rsync trainer done (rc=$?)"

# 5. Kill any stale launcher PID file
ssh "$NEPTUNE" "rm -f $NEPTUNE_LVL3/logs/pids/cnn_mamba_v3_2.pid" >> "$LOG" 2>&1

# 6. Launch fold 0 on Neptune (smoke gate per HC #295H — fold 0 only first)
log "launching v3.2 fold 0 on Neptune (smoke gate)..."
ssh "$NEPTUNE" "cd $NEPTUNE_LVL3 && bash scripts/launch_cnn_mamba_v3_2_neptune.sh --n-folds 1" >> "$LOG" 2>&1
launch_rc=$?
log "launch rc=$launch_rc"

# 7. Confirm process alive and report PID + MLflow run
sleep 15
NEPTUNE_INFO=$(ssh "$NEPTUNE" "ps aux | grep train_cnn_mamba_v3_2 | grep -v grep | head -1; echo '---'; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader")
log "Neptune post-launch info:"
echo "$NEPTUNE_INFO" | tee -a "$LOG"

log "==== v3.2 post-backfill pipeline DONE ===="
