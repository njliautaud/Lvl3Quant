#!/usr/bin/env python3
"""
Leakage Audit: Re-run the integrated pipeline backtest with the fixed
train/val split to verify OOT results are not inflated.

The bug: train_data included val_data samples → early stopping was
evaluating on in-sample data → potential overfit (holdout IC was 0.9861).

This script runs the full walk-forward replay with the corrected code
and compares metrics before/after.

Author: Claude (leakage audit, 2026-06-29)
"""

import sys
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Back up existing state
state_dir = ROOT / "paper_engines" / "logs" / "integrated_pipeline_paper"
state_file = state_dir / "state.json"
backup_state = state_dir / "state_pre_audit.json"

if state_file.exists():
    import shutil
    shutil.copy2(state_file, backup_state)
    print(f"Backed up state to {backup_state}")

# Reset state for clean replay
clean_state = {
    "cash": 100000.0,
    "position": 0,
    "entry_price": 0,
    "entry_time": None,
    "entry_reason": "",
    "exit_target_time": None,
    "total_trades": 0,
    "total_pnl_ticks": 0,
    "total_pnl_dollars": 0,
    "wins": 0,
    "losses": 0,
    "last_signal_time": None,
    "last_retrain_date": None,
    "created": "2026-06-29T00:00:00+00:00",
}

with open(state_file, "w") as f:
    json.dump(clean_state, f, indent=2)

# Delete cached model to force retrain with fixed code
model_file = state_dir / "models" / "lgbm_model.txt"
scaler_file = state_dir / "models" / "scaler.npz"
meta_file = state_dir / "models" / "meta.json"
for f in [model_file, scaler_file, meta_file]:
    if f.exists():
        f.unlink()
        print(f"Deleted {f.name}")

print("\n=== Running replay with FIXED train/val split ===\n")

# Run replay mode (not --live)
start = time.time()
result = subprocess.run(
    [sys.executable, str(ROOT / "paper_engines" / "integrated_pipeline_paper.py"),
     "--replay", "0"],  # 0 = all available OOT days
    capture_output=True, text=True, timeout=600,
    cwd=str(ROOT),
)
elapsed = time.time() - start

print(f"Replay completed in {elapsed:.0f}s")
print("\n=== STDOUT (last 50 lines) ===")
for line in result.stdout.strip().split("\n")[-50:]:
    print(line)

if result.returncode != 0:
    print(f"\n=== STDERR ===")
    print(result.stderr[-2000:] if result.stderr else "(none)")

# Load results
if state_file.exists():
    with open(state_file) as f:
        new_state = json.load(f)

    print("\n" + "=" * 60)
    print("LEAKAGE AUDIT RESULTS (FIXED train/val split)")
    print("=" * 60)
    print(f"Trades:   {new_state.get('total_trades', 0)}")
    print(f"PnL:      {new_state.get('total_pnl_ticks', 0):.1f} ticks (${new_state.get('total_pnl_dollars', 0):,.0f})")
    print(f"Wins:     {new_state.get('wins', 0)}")
    print(f"Losses:   {new_state.get('losses', 0)}")
    wr = new_state['wins'] / max(new_state['total_trades'], 1) * 100
    print(f"Win Rate: {wr:.1f}%")
    print(f"Avg PnL:  {new_state.get('total_pnl_ticks', 0) / max(new_state.get('total_trades', 1), 1):.2f} ticks/trade")

    print("\n--- COMPARISON ---")
    print("BEFORE (leaked val): 314 trades, +1264t ($15,799), WR 49.7%")
    print(f"AFTER  (fixed val):  {new_state.get('total_trades', 0)} trades, "
          f"+{new_state.get('total_pnl_ticks', 0):.0f}t "
          f"(${new_state.get('total_pnl_dollars', 0):,.0f}), "
          f"WR {wr:.1f}%")

    pnl_before = 1264
    pnl_after = new_state.get('total_pnl_ticks', 0)
    pct_change = (pnl_after - pnl_before) / abs(pnl_before) * 100
    print(f"\nPnL change: {pct_change:+.1f}%")

    if pnl_after > pnl_before * 0.8:
        print("\n✅ PASS: Edge survives leakage fix (>80% of original)")
    elif pnl_after > 0:
        print("\n⚠️  DEGRADED: Still profitable but significantly worse")
    else:
        print("\n❌ FAIL: Edge was driven by leakage")

# Restore original state
if backup_state.exists():
    import shutil
    shutil.copy2(backup_state, state_file)
    print(f"\nRestored original state from backup")
