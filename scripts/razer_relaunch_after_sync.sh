#!/bin/bash
# Auto-launches Wider EventCNN1D on Razer once data sync completes
# Idempotent: only fires once, drops marker after success
set -e
MARKER=/tmp/razer_wider_cnn_launched.marker
SYNC_LOG=/tmp/razer_data_sync.log

if [ -f "$MARKER" ]; then
  echo "[$(date)] Already launched (marker exists). Exiting."
  exit 0
fi

# Sync still running?
if pgrep -f "tar -cf - .*mbo_events.npz" >/dev/null; then
  echo "[$(date)] Sync still running. Waiting next cycle."
  exit 0
fi

# Sync completed?
if ! grep -q "^\[end " "$SYNC_LOG" 2>/dev/null; then
  echo "[$(date)] Sync log has no [end] line yet. Waiting."
  exit 0
fi

# Verify Razer file count
RAZER_COUNT=$(sshpass -p "${CLUSTER_SSH_PASSWORD}" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 claude@razer 'dir C:\Users\claude\Lvl3Quant\data\processed\mbo_events\*.npz /b 2^>nul' 2>/dev/null | grep -c '\.npz')
echo "[$(date)] Razer mbo_events file count: $RAZER_COUNT"

if [ -z "$RAZER_COUNT" ] || [ "$RAZER_COUNT" -lt 200 ]; then
  echo "[$(date)] File count too low ($RAZER_COUNT). Sync incomplete. Waiting."
  exit 0
fi

# Update launch.bat to point at new data path
TS=$(date +%Y%m%d_%H%M)
LAUNCH_BAT=/tmp/razer_setup/launch_wider_cnn.bat
cat > "$LAUNCH_BAT" << 'BATEOF'
@echo off
set LOGFILE=C:\Users\claude\Lvl3Quant\logs\wider_cnn_TIMESTAMP.log
set MLFLOW_TRACKING_URI=http://jupiter:5000
set CNN_CHANNELS=256
set CNN_LAYERS=8
set CNN_KERNEL=5
set CNN_DROPOUT=0.1
set EVENT_WINDOW_SIZE=1000
set EVENT_BATCH_SIZE=32
set EVENT_EPOCHS=3
set EVENT_NUM_WORKERS=0
set EVENT_LR=3e-4
set EVENT_GRAD_CLIP=1.0
set EVENT_WARMUP=300
set DISABLE_MLFLOW=0
set PYTHONUNBUFFERED=1
cd /d C:\Users\claude\Lvl3Quant
echo === LAUNCH %DATE% %TIME% === > "%LOGFILE%"
echo CNN_CHANNELS=%CNN_CHANNELS% CNN_LAYERS=%CNN_LAYERS% BATCH=%EVENT_BATCH_SIZE% >> "%LOGFILE%"
python -u scripts\train_event_cnn_1d.py --data-dir C:\Users\claude\Lvl3Quant\data\processed\mbo_events --output-dir C:\Users\claude\Lvl3Quant\output\wider_event_cnn_1d_256x8_TIMESTAMP --n-folds 11 --train-days 60 --oot-days 1 --window-mode sliding --device cuda >> "%LOGFILE%" 2>&1
echo === EXIT %DATE% %TIME% (rc=%ERRORLEVEL%) === >> "%LOGFILE%"
BATEOF
sed -i "s/TIMESTAMP/$TS/g" "$LAUNCH_BAT"
sshpass -p "${CLUSTER_SSH_PASSWORD}" scp -o StrictHostKeyChecking=no "$LAUNCH_BAT" 'claude@razer:Lvl3Quant/scripts/launch_wider_cnn.bat'
echo "[$(date)] Pushed launch.bat with timestamp $TS"

# Launch detached on Razer
sshpass -p "${CLUSTER_SSH_PASSWORD}" ssh -o StrictHostKeyChecking=no claude@razer \
  "powershell -Command \"Start-Process -FilePath 'C:\\Users\\claude\\Lvl3Quant\\scripts\\launch_wider_cnn.bat' -WindowStyle Hidden\""
echo "[$(date)] Launched Wider EventCNN1D 256x8 on Razer."

touch "$MARKER"
echo "[$(date)] Marker set. Won't relaunch."
