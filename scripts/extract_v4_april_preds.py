#!/usr/bin/env python3
"""
Extract V4 Multihead predictions for April 1-14 dates into tick replay format.
Maps fold predictions 1:1 to output files (matching existing format in v4_multihead_tick_replay_preds/).

Composite signal formula (from rebuild_v4_tick_replay_preds.py):
  composite = dir_1s * (1.0 + 0.3 * sign(dir_1s) * (eofi_1s - median(eofi_1s)))
"""

import numpy as np
import os

FOLD_DIR = '/home/jupiter/Lvl3Quant/output/v4_multihead_pressure_v1'
OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/v4_multihead_tick_replay_preds'

# Fold-to-date mapping (from scan of oot_files)
FOLD_DATE_MAP = {
    157: '20260401',
    158: '20260402',
    159: '20260403',
    160: '20260405',
    161: '20260406',
    162: '20260407',
    163: '20260408',
    164: '20260409',
    165: '20260410',
    166: '20260412',
    167: '20260413',
    168: '20260414',
}


def extract_fold(fold_num, date_str):
    fold_path = os.path.join(FOLD_DIR, f'fold_{fold_num}_oot_predictions.npz')
    if not os.path.exists(fold_path):
        return None, f"fold file not found"

    d = np.load(fold_path, allow_pickle=True)
    n_preds = d['preds_dir'].shape[0]

    dir_1s = d['preds_dir'][:, 0].astype(np.float32)
    eofi_1s = d['preds_eofi'][:, 0].astype(np.float32) if 'preds_eofi' in d else np.zeros(n_preds, dtype=np.float32)
    pdi_1s = d['preds_pdi'][:, 0].astype(np.float32) if 'preds_pdi' in d else np.zeros(n_preds, dtype=np.float32)

    # Composite signal: dir * (1 + 0.3 * sign(dir) * centered_eofi)
    eofi_centered = eofi_1s - np.median(eofi_1s)
    composite = dir_1s * (1.0 + 0.3 * np.sign(dir_1s) * eofi_centered)

    out_path = os.path.join(OUTPUT_DIR, f'oot_{date_str}.npz')
    np.savez(
        out_path,
        pred_log_ret_1s=dir_1s,
        composite_signal=composite.astype(np.float32),
        eofi_1s=eofi_1s,
        pdi_1s=pdi_1s,
        n_preds=n_preds,
    )
    return n_preds, None


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    success = []
    failed = []

    for fold_num, date_str in sorted(FOLD_DATE_MAP.items()):
        n_preds, err = extract_fold(fold_num, date_str)
        if err:
            print(f"  FAIL fold {fold_num} ({date_str}): {err}")
            failed.append(date_str)
        else:
            print(f"  OK   fold {fold_num} -> oot_{date_str}.npz  ({n_preds} preds)")
            success.append((date_str, n_preds))

    print(f"\nExtracted {len(success)}/{len(FOLD_DATE_MAP)} dates")
    if failed:
        print(f"Failed: {failed}")

    # Verify output structure matches existing files
    print("\n--- Verification ---")
    ref_path = os.path.join(OUTPUT_DIR, 'oot_20260318.npz')
    if os.path.exists(ref_path):
        ref = np.load(ref_path)
        ref_keys = set(ref.keys())
        for date_str, _ in success[:2]:
            new = np.load(os.path.join(OUTPUT_DIR, f'oot_{date_str}.npz'))
            new_keys = set(new.keys())
            match = ref_keys == new_keys
            print(f"  oot_{date_str}.npz keys match reference: {match}")
            if not match:
                print(f"    ref: {sorted(ref_keys)}, new: {sorted(new_keys)}")
            for k in sorted(new_keys):
                v = new[k]
                print(f"    {k}: dtype={v.dtype}, shape={v.shape if hasattr(v,'shape') and v.shape else 'scalar'}")


if __name__ == '__main__':
    main()
