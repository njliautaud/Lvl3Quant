#!/bin/bash
# Launch PatchTST transformer on Razer with smart_v2 (22 features)
# Prerequisites:
#   1. CNN must have finished all folds on Razer
#   2. smart_v2 data must be synced to Razer
#
# Architecture: PatchTST (patch_size=25, d_model=256, 4 heads×64, 4 layers, ALiBi)
# Data: smart_v2 (22 features, pre-normalized)
# Walk-forward: sliding 60d train, 1d OOT, 10 folds

set -e

RAZER="claude@razer"
RAZER_ROOT="C:\\Users\\claude\\Lvl3Quant"
RAZER_ROOT_UNIX="C:/Users/claude/Lvl3Quant"
RAZER_PASS="${CLUSTER_SSH_PASSWORD}"

echo "=== PatchTST Razer Launch ==="
echo ""

# Step 1: Check if CNN is still running
echo "Checking for running CNN processes on Razer..."
CNN_PID=$(sshpass -p "$RAZER_PASS" ssh -o ConnectTimeout=10 $RAZER \
    "powershell -Command \"Get-Process python* -ErrorAction SilentlyContinue | Where-Object { \$_.CPU -gt 100 } | Select-Object -ExpandProperty Id\"" 2>/dev/null || echo "")

if [ -n "$CNN_PID" ]; then
    echo "WARNING: Python process(es) still running on Razer: $CNN_PID"
    echo "CNN may still be training. Wait for it to finish or kill manually."
    echo "To kill: sshpass -p '$RAZER_PASS' ssh $RAZER 'taskkill /F /PID $CNN_PID'"
    exit 1
fi

# Step 2: Sync smart_v2 data to Razer
echo ""
echo "Checking smart_v2 data on Razer..."
RAZER_COUNT=$(sshpass -p "$RAZER_PASS" ssh -o ConnectTimeout=10 $RAZER \
    "powershell -Command \"(Get-ChildItem $RAZER_ROOT\\data\\processed\\mbo_events_smart_v2\\*.npz -ErrorAction SilentlyContinue).Count\"" 2>/dev/null || echo "0")

LOCAL_COUNT=$(ls -1 /home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v2/*.npz 2>/dev/null | wc -l)

echo "  Razer: $RAZER_COUNT files"
echo "  Local: $LOCAL_COUNT files"

if [ "$RAZER_COUNT" -lt "$LOCAL_COUNT" ]; then
    echo ""
    echo "Syncing smart_v2 data to Razer ($LOCAL_COUNT files)..."
    echo "This may take a while (211GB over network)..."

    # Create directory on Razer first
    sshpass -p "$RAZER_PASS" ssh -o ConnectTimeout=10 $RAZER \
        "powershell -Command \"New-Item -ItemType Directory -Force -Path '$RAZER_ROOT\\data\\processed\\mbo_events_smart_v2'\"" 2>/dev/null

    # Use scp with compression disabled (NPZ already compressed)
    sshpass -p "$RAZER_PASS" rsync -av --progress --no-compress \
        /home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v2/ \
        $RAZER:$RAZER_ROOT_UNIX/data/processed/mbo_events_smart_v2/

    echo "Sync complete!"
fi

# Step 3: Sync the PatchTST training script
echo ""
echo "Syncing PatchTST training script..."
sshpass -p "$RAZER_PASS" scp \
    /home/jupiter/Lvl3Quant/alpha_discovery/deep_models/train_event_patchtst.py \
    $RAZER:$RAZER_ROOT_UNIX/alpha_discovery/deep_models/train_event_patchtst.py

# Step 4: Launch training
echo ""
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
    \$env:PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
    Start-Process -NoNewWindow -FilePath 'C:\\Python311\\python.exe' -ArgumentList '-u', 'alpha_discovery\\deep_models\\train_event_patchtst.py', '--data-dir', 'data\\processed\\mbo_events_smart_v2', '--output-dir', 'output\\patchtst_sliding60d_smart_v2', '--n-folds', '10', '--window-mode', 'sliding', '--train-days', '60', '--oot-days', '1' -RedirectStandardOutput 'logs\\patchtst_smart_v2.log' -RedirectStandardError 'logs\\patchtst_smart_v2_err.log'
    echo 'PatchTST launched on Razer'
\""

echo ""
echo "Monitor with: sshpass -p '$RAZER_PASS' ssh $RAZER 'powershell -Command \"Get-Content $RAZER_ROOT\\output\\patchtst_sliding60d_smart_v2\\training.log -Tail 20 -Wait\"'"
echo "Or: sshpass -p '$RAZER_PASS' ssh $RAZER 'powershell -Command \"Get-Content $RAZER_ROOT\\logs\\patchtst_smart_v2.log -Tail 20\"'"
