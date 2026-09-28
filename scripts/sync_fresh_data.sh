#!/bin/bash
# Sync fresh MBO data and trigger LGBM walk-forward extension
# Run this after purchasing new April data

set -e

echo "=== Fresh Data Sync & Validation ==="
echo "Start time: $(date)"
echo ""

SOURCE_DIR="/home/jupiter/Lvl3Quant/data/processed/mbo_events"
NODES=("neptune:nick@neptune:/home/nick/Lvl3Quant/data/processed/mbo_events" \
       "saturn:saturn@saturn:/home/saturn/Lvl3Quant/data/processed/mbo_events" \
       "razer:claude@razer:/home/footb/Lvl3Quant/data/processed/mbo_events")

# 1. Detect new files (after March 12, 2026)
echo "Scanning for new files after 20260312..."
NEW_FILES=$(ls "$SOURCE_DIR"/202603[13-31]*.npz "$SOURCE_DIR"/202604*.npz 2>/dev/null | wc -l || echo 0)

if [ "$NEW_FILES" -eq 0 ]; then
    echo "⚠️  No new files found. Expected files: 20260313-20260419"
    echo "Did you copy the new data to $SOURCE_DIR?"
    exit 1
fi

echo "✅ Found $NEW_FILES new files"
echo ""

# 2. Validate data integrity
echo "Validating data integrity..."
python3 << 'PYEOF'
import sys
from pathlib import Path
import numpy as np

data_dir = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_events")
files = sorted(data_dir.glob("202603[13-31]*.npz")) + sorted(data_dir.glob("202604*.npz"))

errors = []
for f in files:
    try:
        data = np.load(f)
        # Check required keys
        required = ['events', 'labels_1s', 'labels_5s', 'labels_10s', 'timestamps']
        missing = [k for k in required if k not in data]
        if missing:
            errors.append(f"{f.name}: Missing keys {missing}")
            continue

        # Check shapes
        n_events = data['events'].shape[0]
        if n_events < 1000:
            errors.append(f"{f.name}: Only {n_events} events (suspiciously low)")

        print(f"✅ {f.name}: {n_events:,} events")
    except Exception as e:
        errors.append(f"{f.name}: {e}")

if errors:
    print("\n⚠️  ERRORS FOUND:")
    for err in errors:
        print(f"  {err}")
    sys.exit(1)
else:
    print(f"\n✅ All {len(files)} files validated successfully")
PYEOF

if [ $? -ne 0 ]; then
    echo "Data validation failed. Fix errors before syncing."
    exit 1
fi

echo ""

# 3. Sync to remote nodes
echo "Syncing to remote nodes..."
for node_spec in "${NODES[@]}"; do
    node_name=$(echo "$node_spec" | cut -d: -f1)
    node_dest=$(echo "$node_spec" | cut -d: -f2-)

    echo ""
    echo "→ Syncing to $node_name..."

    # Use rsync for efficient transfer (only new files)
    rsync -avz --progress \
        -e "ssh -o StrictHostKeyChecking=no" \
        "$SOURCE_DIR"/202603[13-31]*.npz "$SOURCE_DIR"/202604*.npz \
        "$node_dest/" 2>&1 | grep -E "sent|received|total size" || {
        echo "⚠️  Failed to sync to $node_name (continuing anyway)"
        continue
    }

    echo "✅ $node_name synced"
done

echo ""
echo "=== Data Sync Complete ==="
echo ""

# 4. Trigger LGBM walk-forward extension
echo "Ready to extend LGBM walk-forward training"
echo ""
echo "Next steps:"
echo "  1. Review data sync results above"
echo "  2. Launch LGBM training with extended date range:"
echo "     ssh saturn@saturn 'cd /home/saturn/Lvl3Quant && ./scripts/train_lgbm_extended.sh'"
echo ""
echo "This will train LGBM on ALL dates including fresh April data."
echo ""
