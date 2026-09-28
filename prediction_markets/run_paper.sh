#!/bin/bash
export GROQ_API_KEY="${GROQ_API_KEY}"
cd /home/jupiter/prediction_markets
echo "[$(date)] Starting paper trader..."
python3 paper_trader.py --loop --interval 1800 --min-score 70 2>&1
