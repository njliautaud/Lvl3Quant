#!/bin/bash
# Full-run driver: ingest -> GA -> report.
# Logs to logs/full_run.log
set -u
cd "$(dirname "$0")"
mkdir -p logs results
LOG=logs/full_run.log
echo "=== full run start $(date -Iseconds) ===" | tee -a "$LOG"

step() { echo "--- $1 $(date -Iseconds) ---" | tee -a "$LOG"; }

step "ingest_universe"
python3 data/ingest_universe.py --no-enrich >> "$LOG" 2>&1
step "ingest_prices"
python3 data/ingest_prices.py --start 2015-01-01 >> "$LOG" 2>&1
step "ingest_fundamentals"
python3 data/ingest_fundamentals.py >> "$LOG" 2>&1
step "ingest_macro"
python3 data/ingest_macro.py --start 2015-01-01 >> "$LOG" 2>&1
step "ingest_options"
python3 data/ingest_options.py >> "$LOG" 2>&1
step "run_ga"
python3 ga/run_ga.py --pop 100 --gen 50 >> "$LOG" 2>&1
step "build_report"
python3 report/build_report.py >> "$LOG" 2>&1
echo "=== full run done $(date -Iseconds) ===" | tee -a "$LOG"
