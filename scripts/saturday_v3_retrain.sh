#!/bin/bash
# Saturday weekly CNN-Mamba v3 retrain (HC #282(A))
# Triggered by cron 0 6 * * 6 (Sat 6 AM ET)
# SSH-invokes the existing v3 launcher on Neptune. No new code; orchestration only.
set -euo pipefail
LOG=/home/jupiter/Lvl3Quant/logs/saturday_v3_retrain.log
echo "[$(date -Iseconds)] Saturday v3 retrain triggered" >> "$LOG"
ssh -o ConnectTimeout=15 nick@neptune \
  "bash -lc 'cd /home/nick/Lvl3Quant && ./scripts/launch_cnn_mamba_v3_neptune.sh'" \
  >> "$LOG" 2>&1
echo "[$(date -Iseconds)] Saturday v3 retrain dispatch returned $?" >> "$LOG"
