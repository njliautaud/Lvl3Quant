#!/bin/bash
# sync_and_run_fillsim.sh -- Sync MFE/MAE predictions from Uranus and run fill sim.
# Runs continuously, checking for new preds every 60s and processing new ones immediately.

set -e

URANUS_IP="winnode"
URANUS_USER="nick"
URANUS_PREDS_DIR="C:/Users/nick/Lvl3Quant/alpha_discovery/deep_models/results/mfe_mae_wf/fold_preds"
LOCAL_PREDS_DIR="/home/jupiter/Lvl3Quant/data/processed/mfe_mae_fold_preds"
FILL_SIM_CLI="/home/jupiter/Lvl3Quant/rust_cache_builder/target/release/fill_sim_cli"
MBO_DIR="/home/jupiter/Lvl3Quant/data/raw/mbo"
RESULTS_DIR="/home/jupiter/Lvl3Quant/results/mfe_mae_fill_sim"
FILL_SIM_SCRIPT="/home/jupiter/Lvl3Quant/scripts/run_mfe_mae_fill_sim.py"
LOG="/home/jupiter/Lvl3Quant/results/mfe_mae_fill_sim/sync_run.log"

mkdir -p "$LOCAL_PREDS_DIR" "$RESULTS_DIR"

log() {
    echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG"
}

log "=== MFE/MAE Sync + Fill Sim ==="
log "Uranus: $URANUS_USER@$URANUS_IP:$URANUS_PREDS_DIR"
log "Local preds: $LOCAL_PREDS_DIR"
log "Results: $RESULTS_DIR"

PROCESSED_FILES=""

while true; do
    # Sync from Uranus via rsync over SSH
    log "Syncing from Uranus..."

    # rsync from Windows path (need to convert to unix-style for OpenSSH on Windows)
    URANUS_PATH_UNIX=$(echo "$URANUS_PREDS_DIR" | sed 's|C:|/c|; s|\\|/|g')

    rsync -avz --no-perms --no-times \
        --include="*_mfe_mae_preds.npz" \
        --exclude="*" \
        "${URANUS_USER}@${URANUS_IP}:${URANUS_PATH_UNIX}/" \
        "$LOCAL_PREDS_DIR/" \
        -e "ssh -o StrictHostKeyChecking=no -o BatchMode=yes" \
        2>&1 | tail -5 | while read line; do log "  rsync: $line"; done

    # Find new pred files that haven't been processed yet
    for pred_file in "$LOCAL_PREDS_DIR"/*_mfe_mae_preds.npz; do
        [ -f "$pred_file" ] || continue
        basename=$(basename "$pred_file")
        date_str="${basename:0:10}"

        # Check if already processed (summary file exists)
        summary_check="$RESULTS_DIR/${date_str}_abs_2.0.json"
        if [ -f "$summary_check" ]; then
            continue  # Already processed
        fi

        # Check if MBO file exists for this date
        compact=$(echo "$date_str" | tr -d '-')
        mbo_file=$(ls "$MBO_DIR"/*${compact}*.dbn.zst 2>/dev/null | head -1)
        if [ -z "$mbo_file" ]; then
            log "  No MBO for $date_str, skipping"
            continue
        fi

        log "  Processing $date_str (MBO: $(basename $mbo_file))..."

        # Run fill sim at each threshold
        for thresh in "1.5" "2.0" "2.5" "3.0"; do
            out_file="$RESULTS_DIR/${date_str}_abs_${thresh}.json"
            [ -f "$out_file" ] && continue

            "$FILL_SIM_CLI" \
                --mbo-file "$mbo_file" \
                --predictions "$pred_file" \
                --output "$out_file" \
                --signal-threshold "$thresh" \
                --hold-ms 10000 \
                --stop-loss-ticks 8 \
                --take-profit-ticks 16 \
                --conviction-exit-bars 50 \
                --chase-entry \
                --chase-max-ticks 2 \
                --chase-max-reprices 5 \
                --prime-hours \
                --latency-ms 2 \
                2>&1 | tail -3 | while read line; do log "    [$date_str thresh=$thresh] $line"; done

            if [ -f "$out_file" ]; then
                n_trades=$(python3 -c "import json; d=json.load(open('$out_file')); s=d.get('summary', d); print(s.get('n_trades',0))" 2>/dev/null || echo "?")
                pnl=$(python3 -c "import json; d=json.load(open('$out_file')); s=d.get('summary', d); print(f'{s.get(\"total_pnl\",0):+.1f}')" 2>/dev/null || echo "?")
                log "    DONE: thresh=$thresh trades=$n_trades PnL=$pnl"
            fi
        done

        # Percentile thresholds
        python3 -c "
import numpy as np, json, subprocess, sys
data = np.load('$pred_file')
sig = np.abs(data['predictions'])
nonzero = sig[sig > 0]
if len(nonzero) == 0:
    print('No signals')
    sys.exit(0)
for pct, label in [(90, 'top10pct'), (95, 'top5pct'), (99, 'top1pct')]:
    threshold = float(np.percentile(nonzero, pct))
    out_file = '$RESULTS_DIR/${date_str}_' + label + '.json'
    if open(out_file).read() if __import__('os').path.exists(out_file) else None:
        continue
    cmd = ['$FILL_SIM_CLI',
        '--mbo-file', '$mbo_file',
        '--predictions', '$pred_file',
        '--output', out_file,
        '--signal-threshold', str(threshold),
        '--hold-ms', '10000',
        '--stop-loss-ticks', '8',
        '--take-profit-ticks', '16',
        '--conviction-exit-bars', '50',
        '--chase-entry', '--chase-max-ticks', '2',
        '--chase-max-reprices', '5',
        '--prime-hours', '--latency-ms', '2']
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode == 0:
        print(f'  {label}: threshold={threshold:.4f}')
    else:
        print(f'  {label} FAILED: {result.stderr[:100]}')
" 2>&1 | while read line; do log "    $line"; done

    done

    # Aggregate results and log summary
    python3 "$FILL_SIM_SCRIPT" --preds-dir "$LOCAL_PREDS_DIR" --mbo-dir "$MBO_DIR" 2>&1 | tail -20 | while read line; do log "SUMMARY: $line"; done

    # Check if inference is complete (no new files expected)
    n_preds=$(ls "$LOCAL_PREDS_DIR"/*.npz 2>/dev/null | wc -l)
    log "Total preds processed: $n_preds"

    # Wait before next sync cycle
    log "Waiting 60s for next sync..."
    sleep 60
done
