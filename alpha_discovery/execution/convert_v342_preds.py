#!/usr/bin/env python3
"""Convert CNN-Mamba v3.4.2 per-date predictions to FIFOExecutionEnv format."""
import numpy as np
from pathlib import Path
import argparse

def convert_file(src: Path, dst: Path):
    d = np.load(src, allow_pickle=True)
    if 'pred_log_ret_1s' not in d:
        return 0  # Skip empty/holiday files
    preds = np.stack([d['pred_log_ret_1s'], d['pred_log_ret_5s'], d['pred_log_ret_10s']], axis=1).astype(np.float32)
    labels = np.stack([d['target_log_ret_1s'], d['target_log_ret_5s'], d['target_log_ret_10s']], axis=1).astype(np.float32)
    np.savez_compressed(dst, predictions=preds, labels=labels, oot_dates=d.get('oot_dates', np.array([])))
    return len(preds)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-dir', required=True)
    parser.add_argument('--dst-dir', required=True)
    args = parser.parse_args()
    src_dir, dst_dir = Path(args.src_dir), Path(args.dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    src_files = sorted(src_dir.glob('oot_*.npz'))
    print(f'Found {len(src_files)} source files')
    converted, skipped = 0, 0
    for sf in src_files:
        date_str = sf.stem.replace('oot_', '')
        dst_file = dst_dir / f'{date_str}_predictions.npz'
        n = convert_file(sf, dst_file)
        if n > 0:
            print(f'  {sf.name} -> {dst_file.name} ({n} samples)')
            converted += 1
        else:
            skipped += 1
    print(f'Done: {converted} converted, {skipped} skipped (empty)')

if __name__ == '__main__':
    main()
