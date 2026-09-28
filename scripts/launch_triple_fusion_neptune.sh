#!/bin/bash
# Launch Triple Fusion (CNN-Mamba-PatchTST) on Neptune
# Smart_v4_book: 29 event features + 20 book_shape + 10 book_dynamics = 59 total
# Sliding 60d window with decay, OOT from March 1st
# User directive: 2026-04-26 — Book reconstruction MANDATORY in smart_v4

set -e

NEPTUNE_HOST="nick@neptune"
NEPTUNE_LVL3="/home/nick/Lvl3Quant"
JUPITER_V4="/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v4"
NEPTUNE_V4="$NEPTUNE_LVL3/data/processed/mbo_events_smart_v4"
JUPITER_BOOK="/home/jupiter/Lvl3Quant/data/processed/mbo_book_normalized"
NEPTUNE_BOOK="$NEPTUNE_LVL3/data/processed/mbo_book_normalized"

echo "=== TRIPLE FUSION LAUNCH SCRIPT (smart_v4_book) ==="
echo "Date: $(date)"

# Step 1: Verify smart_v4 + book data complete on Jupiter
V4_COUNT=$(ls "$JUPITER_V4"/*.npz 2>/dev/null | wc -l)
BOOK_COUNT=$(ls "$JUPITER_BOOK"/*.npz 2>/dev/null | wc -l)
RAW_COUNT=$(ls /home/jupiter/Lvl3Quant/data/processed/mbo_events/*.npz 2>/dev/null | wc -l)
echo "Smart v4 files: $V4_COUNT / $RAW_COUNT"
echo "Book normalized files: $BOOK_COUNT / $RAW_COUNT"
if [ "$V4_COUNT" -lt 200 ]; then
    echo "ERROR: Smart v4 preprocessing not complete ($V4_COUNT files). Wait for it to finish."
    exit 1
fi
if [ "$BOOK_COUNT" -lt 200 ]; then
    echo "WARNING: Book normalization not complete ($BOOK_COUNT files). Will use smart_v4 without book."
    FEATURE_SET="smart_v4"
else
    FEATURE_SET="smart_v4_book"
fi

# Step 2: Kill any existing training on Neptune
echo ""
echo "=== Killing existing training on Neptune ==="
ssh -o ConnectTimeout=10 $NEPTUNE_HOST "pkill -f 'train_cnn_mamba' 2>/dev/null; pkill -f 'train_triple_fusion' 2>/dev/null; echo 'Cleaned up existing processes'"
sleep 2

# Step 3: Copy triple fusion script to Neptune
echo ""
echo "=== Syncing scripts to Neptune ==="
scp -o ConnectTimeout=10 \
    /home/jupiter/Lvl3Quant/alpha_discovery/deep_models/train_triple_fusion.py \
    $NEPTUNE_HOST:$NEPTUNE_LVL3/alpha_discovery/deep_models/

# Step 4: Rsync smart_v4 data to Neptune
echo ""
echo "=== Syncing smart_v4 data to Neptune ==="
ssh $NEPTUNE_HOST "mkdir -p $NEPTUNE_V4"
rsync -avz --progress \
    "$JUPITER_V4/" \
    "$NEPTUNE_HOST:$NEPTUNE_V4/"
echo "Smart v4 data sync complete."

# Step 5: Rsync book normalized data to Neptune (if using smart_v4_book)
if [ "$FEATURE_SET" = "smart_v4_book" ]; then
    echo ""
    echo "=== Syncing book normalized data to Neptune ==="
    ssh $NEPTUNE_HOST "mkdir -p $NEPTUNE_BOOK"
    rsync -avz --progress \
        "$JUPITER_BOOK/" \
        "$NEPTUNE_HOST:$NEPTUNE_BOOK/"
    echo "Book data sync complete."
fi

# Step 6: Launch Triple Fusion
echo ""
echo "=== Launching Triple Fusion on Neptune (FEATURE_SET=$FEATURE_SET) ==="
ssh $NEPTUNE_HOST "cd $NEPTUNE_LVL3/alpha_discovery/deep_models && \
    MAMBA_FEATURE_SET=$FEATURE_SET \
    MAMBA_D_MODEL=96 \
    MAMBA_D_STATE=32 \
    MAMBA_N_LAYERS=3 \
    MAMBA_EPOCHS=5 \
    EVENT_BATCH_SIZE=128 \
    EVENT_WINDOW_SIZE=1000 \
    EVENT_STRIDE=250 \
    EVENT_NUM_WORKERS=8 \
    ENABLE_MFE_MAE=1 \
    DECAY_HALFLIFE_DAYS=15 \
    DISABLE_MLFLOW=1 \
    WF_WINDOW_DAYS=60 \
    nohup /home/nick/miniconda3/envs/py311-train/bin/python -u \
        train_triple_fusion.py \
        --output-dir $NEPTUNE_LVL3/output/triple_fusion_v1_${FEATURE_SET}_mar \
        --window-mode sliding \
        --train-days 60 \
        --oot-days 1 \
        --oot-start-date 2026-03-01 \
        --n-folds 13 \
    > $NEPTUNE_LVL3/logs/triple_fusion_v1_${FEATURE_SET}_mar.log 2>&1 & \
    echo 'LAUNCHED PID='\$!"

echo ""
echo "=== Launch complete! ==="
echo "Feature set: $FEATURE_SET"
echo "  CNN branch: events (29) + book_shape (20) = spatial patterns at price levels"
echo "  PatchTST branch: events (29) + book_dynamics (10) = flow dynamics across time"
echo "  MLP branch: events (29) + book_shape (20) + book_dynamics (10) = all interactions"
echo "  Mamba backbone: gated fusion output"
echo ""
echo "Monitor with: ssh $NEPTUNE_HOST 'tail -f $NEPTUNE_LVL3/logs/triple_fusion_v1_${FEATURE_SET}_mar.log'"
