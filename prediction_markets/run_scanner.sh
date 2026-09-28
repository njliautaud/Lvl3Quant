#!/bin/bash
export GROQ_API_KEY="${GROQ_API_KEY}"
cd /home/jupiter/prediction_markets
while true; do
  echo "[$(date)] Starting auto_scanner..."
  python3 auto_scanner.py --verbose 2>&1
  echo "[$(date)] Sleeping 30 minutes..."
  sleep 1800
done
