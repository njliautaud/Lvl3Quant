#!/bin/bash
# Launch paper trading engine with latest LGBM model
# Markets: Sun 6PM ET → Fri 5PM ET (ES futures)

set -e
cd /home/jupiter/Lvl3Quant/live_trading_linux

# Load Rithmic credentials
source .env
export RITHMIC_USER RITHMIC_PASSWORD RITHMIC_URI
export RITHMIC_PB_DIR="/home/jupiter/teleclaude-main/downloads/rithmic_api/0.89.0.0/samples/samples.py"

MODEL="/home/jupiter/Lvl3Quant/live_trading_linux/models/labels_10s_lgbm.pkl"
CALIB="/home/jupiter/Lvl3Quant/live_trading_linux/models/labels_10s_calibration.json"
LOG_DIR="/home/jupiter/Lvl3Quant/live_trading_linux/logs"

echo "[$(date)] Starting paper engine..."
echo "  Model: $MODEL"
echo "  Symbol: ESM6 @ CME"
echo "  Tier: top10, Timeout: 30s, Slippage: 1 tick"

python3 -u -m live_trading_linux.paper_engine \
    --model "$MODEL" \
    --calibration "$CALIB" \
    --symbol ESM6 \
    --exchange CME \
    --tier top10 \
    --timeout 30 \
    --slippage 1 \
    --commission 0.50 \
    --max-spread 2.0 \
    --stats-interval 300 \
    2>&1 | tee "$LOG_DIR/paper_engine_$(date +%Y%m%d).log"
