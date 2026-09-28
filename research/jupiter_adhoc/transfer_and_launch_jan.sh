#!/bin/bash
set -e
DATES="2025-12-30 2025-12-31 2026-01-02 2026-01-05 2026-01-06 2026-01-07 2026-01-08 2026-01-09 2026-01-12 2026-01-13 2026-01-14 2026-01-15"
PRED_SRC="/home/jupiter/Lvl3Quant/data/processed/cnn_wf_sim_predictions"
MBO_SRC="/home/jupiter/Lvl3Quant/data/raw/mbo"
SATURN="saturn@saturn"
PRED_DEST="$SATURN:/home/saturn/Lvl3Quant/data/processed/cnn_wf_sim_predictions/"
MBO_DEST="$SATURN:/home/saturn/Lvl3Quant/mbo_oot/"

echo "[$(date +%H:%M:%S)] Starting transfer of prediction files..."
FILES_TO_SEND=""
for d in $DATES; do
  for v in 50 60 70 80 90; do
    f="${PRED_SRC}/${d}_vol${v}_morning_afternoon.npz"
    if [ -f "$f" ]; then
      FILES_TO_SEND="$FILES_TO_SEND $f"
    fi
  done
done
echo "[$(date +%H:%M:%S)] Transferring 60 prediction files..."
rsync -av --progress $FILES_TO_SEND saturn@saturn:/home/saturn/Lvl3Quant/data/processed/cnn_wf_sim_predictions/ 2>&1
echo "[$(date +%H:%M:%S)] Prediction transfer DONE."

echo "[$(date +%H:%M:%S)] Starting transfer of MBO files..."
for d in $DATES; do
  d_nodash=$(echo $d | tr -d -)
  mbo="${MBO_SRC}/glbx-mdp3-${d_nodash}.mbo.dbn.zst"
  if [ -f "$mbo" ]; then
    echo "  Sending $mbo"
    rsync -av --progress "$mbo" saturn@saturn:/home/saturn/Lvl3Quant/mbo_oot/ 2>&1
  else
    echo "  MISSING: $mbo"
  fi
done
echo "[$(date +%H:%M:%S)] MBO transfer DONE."

echo "[$(date +%H:%M:%S)] Verifying files on Saturn..."
ssh saturn@saturn "ls /home/saturn/Lvl3Quant/data/processed/cnn_wf_sim_predictions/ | grep -E '2025-12-3|2026-01' | wc -l"
ssh saturn@saturn "ls /home/saturn/Lvl3Quant/mbo_oot/ | grep -E '20251230|20251231|20260102|20260105|20260106|20260107|20260108|20260109|20260112|20260113|20260114|20260115' | wc -l"

echo "[$(date +%H:%M:%S)] Launching sweep on Saturn..."
ssh saturn@saturn "tmux new-session -d -s wf_sweep_jan 'cd /home/saturn/Lvl3Quant && python3 alpha_discovery/saturn_wf_sweep_jan.py --workers 20 2>&1 | tee alpha_discovery/results/wf_sweep_jan/launch.log; exec bash'"
echo "[$(date +%H:%M:%S)] Sweep launched in tmux session wf_sweep_jan on Saturn."
echo "[$(date +%H:%M:%S)] ALL DONE."
