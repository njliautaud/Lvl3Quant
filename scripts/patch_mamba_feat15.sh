#!/bin/bash
# Patch Mamba training script on Neptune to:
# 1. Use feat15 data (15 features instead of 6)
# 2. Fix num_workers to 2 (not 8 which caused OOM)
# 3. Set 5-day OOT test windows
# Run on Neptune after recovery

SCRIPT="/home/nick/Lvl3Quant/alpha_discovery/deep_models/train_event_mamba_cuda.py"

# Fix input features: DEFAULT_DATA_DIR to feat15
sed -i "s|data/processed/mbo_events|data/processed/mbo_events_feat15|g" "$SCRIPT"

# Fix num_workers: 8 -> 2 (31GB RAM can't handle 8 workers)
sed -i 's/8 if isinstance(train_ds, LazyMboEventDataset)/2 if isinstance(train_ds, LazyMboEventDataset)/' "$SCRIPT"

# Fix OOT window size: make it configurable via env var, default 5 days
sed -i 's/oot_end   = oot_start + max(1, (n_files - min_train) \/\/ n_folds)/oot_days = int(os.environ.get("WF_OOT_DAYS", 5))\n        oot_end   = min(oot_start + oot_days, n_files)/' "$SCRIPT"

echo "Patched: feat15 data, num_workers=2, OOT=5d"
grep "mbo_events_feat15\|_num_workers\|oot_days" "$SCRIPT" | head -5
