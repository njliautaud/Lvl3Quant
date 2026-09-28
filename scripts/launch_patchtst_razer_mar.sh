#!/bin/bash
# Launch PatchTST on Razer with smart_v2 — targeting March/April OOT dates
# Uses n_folds=20 to cover ~20 recent trading days (Feb-Apr 2026)
# Waits for data sync to complete before launching

set -e

RAZER="claude@razer"
RAZER_ROOT="C:\\Users\\claude\\Lvl3Quant"
RAZER_ROOT_UNIX="C:/Users/claude/Lvl3Quant"
RAZER_PASS="${CLUSTER_SSH_PASSWORD}"

echo "=== PatchTST Razer Launch (March/Apr OOT dates) ==="

# Wait for data sync — check file count
LOCAL_COUNT=$(ls -1 /home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v2/*.npz 2>/dev/null | wc -l)
echo "Local files: $LOCAL_COUNT"

while true; do
    RAZER_COUNT=$(sshpass -p "$RAZER_PASS" ssh -o ConnectTimeout=10 $RAZER \
        "powershell -Command \"(Get-ChildItem $RAZER_ROOT\\data\\processed\\mbo_events_smart_v2\\*.npz -ErrorAction SilentlyContinue).Count\"" 2>/dev/null || echo "0")
    echo "$(date): Razer has $RAZER_COUNT / $LOCAL_COUNT files"
    if [ "$RAZER_COUNT" -ge "$LOCAL_COUNT" ]; then
        echo "Data sync complete!"
        break
    fi
    sleep 60
done

# Kill any existing Python training on Razer
echo "Checking for existing training..."
sshpass -p "$RAZER_PASS" ssh -o ConnectTimeout=10 $RAZER \
    "powershell -Command \"Get-Process python* -ErrorAction SilentlyContinue | Where-Object { \$_.CPU -gt 100 } | Stop-Process -Force\"" 2>/dev/null || true

# Launch PatchTST
echo "Launching PatchTST on Razer..."
sshpass -p "$RAZER_PASS" ssh -o ConnectTimeout=10 $RAZER "powershell -Command \"
    cd $RAZER_ROOT
    \$env:TST_FEATURE_SET='smart_v2'
    \$env:DISABLE_MLFLOW='1'
    \$env:EVENT_WINDOW_SIZE='500'
    \$env:EVENT_BATCH_SIZE='128'
    \$env:EVENT_STRIDE='250'
    \$env:TST_PATCH_SIZE='25'
    \$env:TST_D_MODEL='256'
    \$env:TST_N_HEADS='4'
    \$env:TST_HEAD_DIM='64'
    \$env:TST_N_LAYERS='4'
    \$env:TST_FFN_DIM='1024'
    \$env:TST_DROPOUT='0.1'
    \$env:EVENT_LR='3e-4'
    \$env:EVENT_EPOCHS='5'
    \$env:WF_WINDOW_MODE='sliding'
    \$env:WF_WINDOW_DAYS='60'
    \$env:PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
    Start-Process -NoNewWindow -FilePath 'C:\\Python311\\python.exe' -ArgumentList '-u', 'alpha_discovery\\deep_models\\train_event_patchtst.py', '--data-dir', 'data\\processed\\mbo_events_smart_v2', '--output-dir', 'output\\patchtst_smart_v2_mar', '--n-folds', '20', '--window-mode', 'sliding', '--train-days', '60', '--oot-days', '1' -RedirectStandardOutput 'logs\\patchtst_smart_v2_mar.log' -RedirectStandardError 'logs\\patchtst_smart_v2_mar_err.log'
    echo 'PatchTST launched on Razer — 20 folds, March/April OOT dates'
\""

echo ""
echo "Monitor: sshpass -p '$RAZER_PASS' ssh $RAZER 'powershell -Command \"Get-Content $RAZER_ROOT\\logs\\patchtst_smart_v2_mar.log -Tail 20\"'"
