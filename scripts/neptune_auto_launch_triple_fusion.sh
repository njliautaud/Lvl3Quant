#!/bin/bash
# Auto-launch Triple Fusion with BOOK FEATURES on Neptune
# Waits for: (1) smart_v4 preprocessing done, (2) book_normalized data rsynced from Jupiter
# User directive Apr 26: book reconstruction MANDATORY in smart_v4

NEPTUNE_LVL3="/home/nick/Lvl3Quant"
V4_DIR="$NEPTUNE_LVL3/data/processed/mbo_events_smart_v4"
BOOK_DIR="$NEPTUNE_LVL3/data/processed/mbo_book_normalized"
TARGET_V4=200
TARGET_BOOK=200

echo "=== Waiting for smart_v4 + book_normalized data ==="
echo "V4 target: $TARGET_V4 in $V4_DIR"
echo "Book target: $TARGET_BOOK in $BOOK_DIR"

while true; do
    V4_COUNT=$(ls "$V4_DIR"/*.npz 2>/dev/null | wc -l)
    BOOK_COUNT=$(ls "$BOOK_DIR"/*.npz 2>/dev/null | wc -l)
    V4_WORKERS=$(ps aux | grep precompute_features_smart_v4 | grep -v grep | wc -l)
    echo "$(date '+%H:%M:%S') V4: $V4_COUNT/$TARGET_V4 (workers: $V4_WORKERS) | Book: $BOOK_COUNT/$TARGET_BOOK"
    
    # V4 must be done (workers=0 + enough files, OR target reached)
    V4_DONE=0
    if [ "$V4_WORKERS" -eq 0 ] && [ "$V4_COUNT" -ge 50 ]; then V4_DONE=1; fi
    if [ "$V4_COUNT" -ge "$TARGET_V4" ]; then V4_DONE=1; fi
    
    # Book must have enough files (rsynced from Jupiter)
    BOOK_DONE=0
    if [ "$BOOK_COUNT" -ge "$TARGET_BOOK" ]; then BOOK_DONE=1; fi
    
    if [ "$V4_DONE" -eq 1 ] && [ "$BOOK_DONE" -eq 1 ]; then
        echo "BOTH ready! V4=$V4_COUNT, Book=$BOOK_COUNT. Launching..."
        break
    fi
    
    # If V4 done but no book yet, just use smart_v4 (fallback after 30 min wait)
    if [ "$V4_DONE" -eq 1 ] && [ "$BOOK_COUNT" -lt "$TARGET_BOOK" ]; then
        echo "V4 ready but book not yet ($BOOK_COUNT). Waiting for book rsync from Jupiter..."
    fi
    
    sleep 30
done

# Determine feature set
BOOK_COUNT=$(ls "$BOOK_DIR"/*.npz 2>/dev/null | wc -l)
if [ "$BOOK_COUNT" -ge "$TARGET_BOOK" ]; then
    FEAT="smart_v4_book"
    echo "Using smart_v4_book (59 features: 29 event + 20 book_shape + 10 book_dynamics)"
else
    FEAT="smart_v4"
    echo "WARNING: Using smart_v4 without book ($BOOK_COUNT book files)"
fi

echo ""
echo "=== Launching Triple Fusion ($FEAT) ==="
cd "$NEPTUNE_LVL3/alpha_discovery/deep_models"

MAMBA_FEATURE_SET=$FEAT \
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
    --output-dir "$NEPTUNE_LVL3/output/triple_fusion_v1_${FEAT}_mar" \
    --window-mode sliding \
    --train-days 60 \
    --oot-days 1 \
    --oot-start-date 2026-03-01 \
    --n-folds 13 \
> "$NEPTUNE_LVL3/logs/triple_fusion_v1_${FEAT}_mar.log" 2>&1 &

echo "LAUNCHED PID=$!"
echo "Feature set: $FEAT"
echo "Monitor: tail -f $NEPTUNE_LVL3/logs/triple_fusion_v1_${FEAT}_mar.log"
