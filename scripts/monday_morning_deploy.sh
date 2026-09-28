#!/bin/bash
# Monday Morning Deployment Orchestrator
# ========================================
# Chains: Razer health check → MBO bulk sync → minute bar build → data validation
# Run at ~8:00 ET Monday when Razer comes online
#
# Usage: ./scripts/monday_morning_deploy.sh [--dry-run]
# Exit codes: 0 = success, 1 = Razer unreachable, 2 = sync failed, 3 = bar build failed

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$ROOT_DIR/logs"
LOG="$LOG_DIR/monday_deploy_$(date +%Y%m%d).log"
MBO_DIR="$ROOT_DIR/data/processed/mbo_events"
BAR_DIR="$ROOT_DIR/data/processed/mbo_minute_bars_v1"
RAZER_HOST="claude@razer"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

mkdir -p "$LOG_DIR"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $1" | tee -a "$LOG"
}

# ─── Phase 0: Pre-flight ───
log "=== MONDAY MORNING DEPLOY START ==="
PRE_MBO_COUNT=$(ls -1 "$MBO_DIR"/*.npz 2>/dev/null | wc -l)
PRE_BAR_COUNT=$(ls -1 "$BAR_DIR"/*.parquet 2>/dev/null | wc -l)
LAST_MBO=$(ls -1 "$MBO_DIR"/*.npz 2>/dev/null | sort | tail -1 | xargs basename 2>/dev/null || echo "none")
log "Pre-sync: ${PRE_MBO_COUNT} MBO files, ${PRE_BAR_COUNT} minute bar files"
log "Latest MBO: ${LAST_MBO}"

# ─── Phase 1: Check Razer is online ───
log "Phase 1: Checking Razer connectivity..."
RAZER_ONLINE=false
for attempt in 1 2 3; do
    if ping -c 1 -W 5 razer &>/dev/null; then
        RAZER_ONLINE=true
        log "Razer reachable (attempt $attempt)"
        break
    fi
    log "Razer ping attempt $attempt failed, waiting 10s..."
    sleep 10
done

if [ "$RAZER_ONLINE" = false ]; then
    log "ERROR: Razer unreachable after 3 attempts. Aborting."
    echo "RESULT: Razer offline — cannot sync. Retry later."
    exit 1
fi

# Verify SSH works
if ! sshpass -p "${CLUSTER_SSH_PASSWORD:-}" ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no \
    "${RAZER_HOST}" "echo ok" 2>/dev/null | grep -q ok; then
    log "ERROR: Razer ping OK but SSH failed. Check credentials."
    echo "RESULT: Razer reachable but SSH auth failed."
    exit 1
fi
log "Razer SSH verified OK"

if [ "$DRY_RUN" = true ]; then
    log "DRY RUN — would proceed with sync. Exiting."
    exit 0
fi

# ─── Phase 2: Bulk sync MBO data ───
log "Phase 2: Running MBO bulk sync from Razer..."
SYNC_START=$(date +%s)

bash "$SCRIPT_DIR/razer_bulk_sync.sh" 2>&1 | tee -a "$LOG"
SYNC_STATUS=${PIPESTATUS[0]}

SYNC_END=$(date +%s)
SYNC_DURATION=$(( SYNC_END - SYNC_START ))

POST_MBO_COUNT=$(ls -1 "$MBO_DIR"/*.npz 2>/dev/null | wc -l)
NEW_MBO=$(( POST_MBO_COUNT - PRE_MBO_COUNT ))
LATEST_MBO=$(ls -1 "$MBO_DIR"/*.npz 2>/dev/null | sort | tail -1 | xargs basename 2>/dev/null || echo "none")

log "Sync complete in ${SYNC_DURATION}s: ${NEW_MBO} new files (${PRE_MBO_COUNT} → ${POST_MBO_COUNT})"
log "Latest MBO now: ${LATEST_MBO}"

if [ "$NEW_MBO" -eq 0 ]; then
    log "WARNING: No new MBO files synced. Razer may not have recorded during May-June."
    # Don't exit — still proceed with minute bar build in case some were partially done
fi

# ─── Phase 3: Build minute bars from new data ───
log "Phase 3: Building minute bars from MBO data..."
BAR_START=$(date +%s)

# The builder is idempotent — skips existing parquets
python3 "$SCRIPT_DIR/build_minute_bars_v1.py" \
    --input-dir "$ROOT_DIR/data/raw/mbo" \
    --output-dir "$BAR_DIR" \
    --log-dir "$LOG_DIR" 2>&1 | tee -a "$LOG"
BAR_STATUS=${PIPESTATUS[0]}

# Also try building from processed NPZ (alternate path if raw doesn't exist)
if [ -f "$SCRIPT_DIR/npz_to_minute_bars.py" ]; then
    log "Running NPZ→minute-bar conversion for any new files..."
    python3 "$SCRIPT_DIR/npz_to_minute_bars.py" 2>&1 | tee -a "$LOG" || true
fi

BAR_END=$(date +%s)
BAR_DURATION=$(( BAR_END - BAR_START ))

POST_BAR_COUNT=$(ls -1 "$BAR_DIR"/*.parquet 2>/dev/null | wc -l)
NEW_BARS=$(( POST_BAR_COUNT - PRE_BAR_COUNT ))
LATEST_BAR=$(ls -1 "$BAR_DIR"/*.parquet 2>/dev/null | sort | tail -1 | xargs basename 2>/dev/null || echo "none")

log "Bar build complete in ${BAR_DURATION}s: ${NEW_BARS} new bar files (${PRE_BAR_COUNT} → ${POST_BAR_COUNT})"
log "Latest minute bar: ${LATEST_BAR}"

# ─── Phase 4: Validate data integrity ───
log "Phase 4: Data validation..."
VALIDATION_OK=true

# Check no zero-size files
ZERO_FILES=$(find "$MBO_DIR" -name "*.npz" -empty 2>/dev/null | wc -l)
if [ "$ZERO_FILES" -gt 0 ]; then
    log "WARNING: ${ZERO_FILES} empty NPZ files found!"
    VALIDATION_OK=false
fi

ZERO_BARS=$(find "$BAR_DIR" -name "*.parquet" -empty 2>/dev/null | wc -l)
if [ "$ZERO_BARS" -gt 0 ]; then
    log "WARNING: ${ZERO_BARS} empty parquet files found!"
    VALIDATION_OK=false
fi

# Check data gap — list missing dates
log "Checking for data gaps in the last 60 trading days..."
python3 -c "
import os
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

bar_dir = Path('$BAR_DIR')
files = sorted(bar_dir.glob('*.parquet'))
dates = [f.stem for f in files if f.stem.isdigit()]

if len(dates) > 1:
    last = dates[-1]
    first_recent = dates[-min(60, len(dates))]
    print(f'Bar coverage: {first_recent} to {last} ({len(dates)} total days)')

    # Check for gaps > 3 consecutive missing weekdays
    from datetime import datetime as dt
    date_set = set(dates)
    start = dt.strptime(first_recent, '%Y%m%d')
    end = dt.strptime(last, '%Y%m%d')
    cur = start
    gap_start = None
    gaps = []
    while cur <= end:
        if cur.weekday() < 5:  # weekday
            ds = cur.strftime('%Y%m%d')
            if ds not in date_set:
                if gap_start is None:
                    gap_start = ds
            else:
                if gap_start is not None:
                    gaps.append((gap_start, (cur - timedelta(days=1)).strftime('%Y%m%d')))
                    gap_start = None
        cur += timedelta(days=1)
    if gap_start:
        gaps.append((gap_start, end.strftime('%Y%m%d')))

    if gaps:
        print(f'Data gaps found ({len(gaps)}):')
        for g in gaps:
            print(f'  {g[0]} - {g[1]}')
    else:
        print('No data gaps detected.')
else:
    print(f'Only {len(dates)} bar files — cannot check continuity')
" 2>&1 | tee -a "$LOG"

# ─── Summary ───
log "=== MONDAY DEPLOY SUMMARY ==="
log "MBO files: ${PRE_MBO_COUNT} → ${POST_MBO_COUNT} (+${NEW_MBO} new)"
log "Minute bars: ${PRE_BAR_COUNT} → ${POST_BAR_COUNT} (+${NEW_BARS} new)"
log "Latest MBO: ${LATEST_MBO}"
log "Latest bar: ${LATEST_BAR}"
log "Sync time: ${SYNC_DURATION}s, Build time: ${BAR_DURATION}s"
log "Validation: $([ "$VALIDATION_OK" = true ] && echo 'PASS' || echo 'WARNINGS — check log')"
log "=== DEPLOY COMPLETE ==="

# Output machine-readable summary for calling scripts
cat <<EOF

MONDAY_DEPLOY_RESULT:
  status: $([ "$VALIDATION_OK" = true ] && echo 'OK' || echo 'WARNINGS')
  new_mbo_files: ${NEW_MBO}
  new_bar_files: ${NEW_BARS}
  total_mbo: ${POST_MBO_COUNT}
  total_bars: ${POST_BAR_COUNT}
  latest_date: ${LATEST_BAR%.parquet}
  sync_seconds: ${SYNC_DURATION}
  build_seconds: ${BAR_DURATION}
  next_step: retrain_walkforward_with_new_data
EOF
