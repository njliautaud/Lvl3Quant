#!/bin/bash
# launch_hc451_salience_neptune.sh — stage and launch HC #451 salience-augmented CNN-Mamba v3.4.2 retrain on Neptune
#
# Pre-conditions:
#   1. HC #450 book CNN training on Neptune (current PID 419431) has finished (no train_v2_branch_book_cnn process)
#   2. Salience parquets present on Jupiter at /home/jupiter/Lvl3Quant/output/hc451_salience_tags/per_day/ (238 days)
#   3. Dispatch script present on Jupiter at /home/jupiter/Lvl3Quant/scripts/v3_4_research/dispatch_v34_2_salience.py
#   4. Neptune has canonical baseline ckpt at /home/nick/Lvl3Quant/output/cnn_mamba_v3_4_2_fixedmtl/fold_00_intra_ckpt.pt
#
# This script:
#   a. Rsyncs salience parquets + dispatch script from Jupiter to Neptune
#   b. Launches the salience-augmented retrain on Neptune via setsid+nohup (the pattern that survives SSH drops)
#   c. Tails 30 seconds of log and confirms training started

set -euo pipefail

NEPTUNE_HOST="nick@neptune"
JUPITER_REPO="/home/jupiter/Lvl3Quant"
NEPTUNE_REPO="/home/nick/Lvl3Quant"

echo "=== HC #451 salience-augmented v3.4.2 retrain launch ==="
echo

# 1. Verify book CNN is not still running on Neptune
echo "[1/4] Verifying Neptune is free..."
BOOK_CNN_RUNNING=$(ssh -o ConnectTimeout=10 "$NEPTUNE_HOST" "ps aux | grep train_v2_branch_book_cnn | grep -v grep | wc -l")
if [ "$BOOK_CNN_RUNNING" -gt 0 ]; then
  echo "  ERROR: book CNN still running on Neptune ($BOOK_CNN_RUNNING processes). Aborting."
  exit 1
fi
echo "  OK — Neptune is free."

# 2. Rsync salience parquets and dispatch script
echo "[2/4] Rsyncing salience parquets + dispatch script to Neptune..."
ssh -o ConnectTimeout=10 "$NEPTUNE_HOST" "mkdir -p $NEPTUNE_REPO/output/hc451_salience_tags/per_day $NEPTUNE_REPO/scripts/v3_4_research"
rsync -avh --progress \
  "$JUPITER_REPO/output/hc451_salience_tags/per_day/" \
  "$NEPTUNE_HOST:$NEPTUNE_REPO/output/hc451_salience_tags/per_day/"
rsync -avh \
  "$JUPITER_REPO/scripts/v3_4_research/dispatch_v34_2_salience.py" \
  "$NEPTUNE_HOST:$NEPTUNE_REPO/scripts/v3_4_research/dispatch_v34_2_salience.py"
echo "  Rsync complete."

# 3. Stage the launch script on Neptune (matches the working /tmp/launch_book_cnn_w0.sh pattern that survived SSH drops)
echo "[3/4] Staging Neptune-side launch script..."
ssh -o ConnectTimeout=10 "$NEPTUNE_HOST" "cat > /tmp/launch_hc451_salience.sh << 'EOF'
#!/bin/bash
cd $NEPTUNE_REPO || exit 1
LOG=$NEPTUNE_REPO/logs/hc451_salience_\$(date +%Y%m%d_%H%M%S).log
exec env V32_BATCH_SIZE=16 V32_WF_TRAIN_DAYS=10 V32_EPOCHS=3 PYTHONPATH=$NEPTUNE_REPO /home/nick/training-env/bin/python -u -X faulthandler scripts/v3_4_research/dispatch_v34_2_salience.py --device cuda --n-folds 1 --salience-tags-dir $NEPTUNE_REPO/output/hc451_salience_tags/per_day --warm-start-from $NEPTUNE_REPO/output/cnn_mamba_v3_4_2_fixedmtl/fold_00_intra_ckpt.pt --output-dir $NEPTUNE_REPO/output/hc451_cnn_mamba_v342_salience >\"\$LOG\" 2>&1
EOF
chmod +x /tmp/launch_hc451_salience.sh"
echo "  Staged."

# 4. Launch detached
echo "[4/4] Launching salience-augmented retrain..."
ssh -o ConnectTimeout=10 "$NEPTUNE_HOST" "setsid bash -c 'nohup /tmp/launch_hc451_salience.sh </dev/null >/tmp/launch_hc451_stdout.log 2>&1 &'"
sleep 10
ssh -o ConnectTimeout=10 "$NEPTUNE_HOST" "ps -ef | grep dispatch_v34_2_salience | grep -v grep | head -3; echo '---LOG---'; ls -lt $NEPTUNE_REPO/logs/hc451_salience_*.log 2>/dev/null | head -2; LATEST_LOG=\$(ls -t $NEPTUNE_REPO/logs/hc451_salience_*.log 2>/dev/null | head -1); if [ -n \"\$LATEST_LOG\" ]; then echo '---FIRST 20 LINES---'; head -20 \"\$LATEST_LOG\"; fi"

echo
echo "=== Launch complete. Monitor via mamba_monitor cron. ==="
