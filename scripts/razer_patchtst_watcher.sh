#!/bin/bash
# Watcher: poll Razer Tailscale every 2 min; when reachable, auto-pull PatchTST weights.
# Triggered by cron once Razer appears online. Self-disables after successful pull.
set -u
LOG=/home/jupiter/Lvl3Quant/logs/razer_patchtst_pull.log
DEST=/home/jupiter/Lvl3Quant/output/patchtst_razer_weights
DONE=/home/jupiter/Lvl3Quant/.razer_patchtst_pulled

mkdir -p "$DEST"
[ -f "$DONE" ] && exit 0

# Probe Razer
if ! ping -c 1 -W 2 razer >/dev/null 2>&1; then
  exit 0
fi

echo "[$(date '+%F %T')] Razer reachable; attempting PatchTST .pt weight pull..." >> "$LOG"

# Find and copy any *.pt under output/*patchtst* and predictions NPZs
ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no claude@razer \
  "powershell -Command \"Get-ChildItem -Path 'C:\\Users\\claude\\Lvl3Quant\\output' -Recurse -Filter '*patchtst*' -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName\"" \
  > /tmp/razer_patchtst_files.txt 2>>"$LOG"

if [ ! -s /tmp/razer_patchtst_files.txt ]; then
  echo "[$(date '+%F %T')] No PatchTST files found on Razer." >> "$LOG"
  exit 0
fi

# rsync entire patchtst output directories (use Cygwin-style path mapping)
rsync -avz --include='*.pt' --include='*predictions*.npz' --include='*.json' --include='*/' --exclude='*' \
  -e "ssh -o StrictHostKeyChecking=no" \
  claude@razer:/cygdrive/c/Users/claude/Lvl3Quant/output/patchtst_*/ "$DEST/" >> "$LOG" 2>&1 \
  || rsync -avz --include='*.pt' --include='*predictions*.npz' --include='*.json' --include='*/' --exclude='*' \
       -e "ssh -o StrictHostKeyChecking=no" \
       claude@razer:'C:/Users/claude/Lvl3Quant/output/patchtst_*' "$DEST/" >> "$LOG" 2>&1

PULLED=$(find "$DEST" -name '*.pt' | wc -l)
echo "[$(date '+%F %T')] pulled $PULLED .pt files into $DEST" >> "$LOG"
if [ "$PULLED" -gt 0 ]; then
  touch "$DONE"
  /home/jupiter/Lvl3Quant/scripts/autonomy_inject.sh "RAZER PATCHTST WEIGHTS PULLED: $PULLED .pt files now in $DEST. Time to relaunch fusion v2 with --warm-patchtst per HC #8."
fi
