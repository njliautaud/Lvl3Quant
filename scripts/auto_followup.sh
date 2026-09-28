#!/bin/bash
# auto_followup.sh — HC #469 R6(a) + R6(b) — sweep-completion auto-dispatcher.
#
# Scans output/ for newly-created .DONE marker files. For each, launches the
# next item in a hard-coded chain queue. Idempotent — uses .CHAINED markers
# to avoid double-dispatching the same followup.
#
# Cron: */5 * * * *
#
# Chain (each .DONE triggers the NEXT entry):
#  1) full_pair_triplet_sweep.DONE → top10_stability_stratified.py
#  2) top10_stability_report.DONE → surviving_confluence_canonical_fifo.py
#  3) surviving_canonical_fifo.DONE → adaptive_exit_v0_train.py
#  4) adaptive_exit_v0.DONE → (queue for user review; no further auto)

set -uo pipefail

LVL3=/home/jupiter/Lvl3Quant
LOG=$LVL3/logs/auto_followup.log
OUT=$LVL3/output/stream_backtest_v2
TS=$(date '+%Y-%m-%d %H:%M:%S')

log() { echo "[$TS] $*" >> "$LOG"; }

# Helper: if DONE marker exists AND no CHAINED marker yet, fire callback and mark CHAINED.
maybe_fire() {
    local done_marker="$1"
    local chained_marker="$2"
    local cmd="$3"

    if [[ -f "$done_marker" && ! -f "$chained_marker" ]]; then
        log "FIRE: $done_marker → $cmd"
        eval "$cmd" >> "$LOG" 2>&1
        local rc=$?
        if [[ $rc -eq 0 ]]; then
            touch "$chained_marker"
            log "CHAINED: $chained_marker written"
        else
            log "ERROR: command exited $rc — leaving $chained_marker absent so we retry next cycle"
        fi
    fi
}

# Chain step 2: stability done → canonical FIFO
maybe_fire \
    "$OUT/stability_stratified.DONE" \
    "$OUT/stability_stratified.CHAINED" \
    "cd $LVL3 && nohup python3 -u scripts/surviving_confluence_canonical_fifo.py > logs/surviving_canonical_fifo_\$(date +%Y%m%d_%H%M%S).log 2>&1 & disown"

# Chain step 3: canonical FIFO done → adaptive exit v0
maybe_fire \
    "$OUT/surviving_canonical_fifo.DONE" \
    "$OUT/surviving_canonical_fifo.CHAINED" \
    "cd $LVL3 && nohup python3 -u scripts/adaptive_exit_v0_train.py > logs/adaptive_exit_v0_\$(date +%Y%m%d_%H%M%S).log 2>&1 & disown"

# Chain step 4: adaptive exit done → (terminal — log only, no further auto-dispatch)
if [[ -f "$LVL3/output/adaptive_exit_v0/adaptive_exit_v0.DONE" && ! -f "$LVL3/output/adaptive_exit_v0/adaptive_exit_v0.CHAINED" ]]; then
    log "TERMINAL: adaptive_exit_v0 complete — leaving for user review"
    touch "$LVL3/output/adaptive_exit_v0/adaptive_exit_v0.CHAINED"
fi

exit 0
