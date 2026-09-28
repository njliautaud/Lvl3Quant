#!/bin/bash
# Full-run driver for macro_exposure_v1.
# Logs to logs/full_run.log
# DO NOT LAUNCH until wheel_strategy_v1 ingest_fundamentals / ingest_options is done,
# or yfinance will rate-limit both pipelines.
set -u
cd "$(dirname "$0")"
mkdir -p logs results
LOG=logs/full_run.log
echo "=== full run start $(date -Iseconds) ===" | tee -a "$LOG"

step() { echo "--- $1 $(date -Iseconds) ---" | tee -a "$LOG"; }

step "ingest_market"
python3 data/ingest_market.py --start 2010-01-01 >> "$LOG" 2>&1
step "ingest_macro_features"
python3 data/ingest_macro_features.py >> "$LOG" 2>&1
step "ingest_sentiment"
python3 data/ingest_sentiment.py --start 2010-01-01 >> "$LOG" 2>&1
step "ingest_breadth"
python3 data/ingest_breadth.py >> "$LOG" 2>&1
step "run_ga"
python3 ga/run_ga.py --pop 80 --gen 40 >> "$LOG" 2>&1
step "build_report"
python3 report/build_report.py >> "$LOG" 2>&1
echo "=== full run done $(date -Iseconds) ===" | tee -a "$LOG"
