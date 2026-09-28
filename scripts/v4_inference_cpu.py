#!/usr/bin/env python3
"""
V4 Multihead — CPU Inference for New Dates
==========================================
Uses a trained fold checkpoint to generate predictions on dates
that don't have walk-forward predictions yet.

This is NOT proper walk-forward (uses a single fold's weights for all dates),
but gives valid signal quality estimates since all test dates are AFTER
the training data.

Usage:
    python3 v4_inference_cpu.py                         # all expandable dates
    python3 v4_inference_cpu.py --dates 20260223 20260224  # specific dates
    python3 v4_inference_cpu.py --fold 141               # use specific fold weights
"""
import argparse
import sys
import time
import gc
import numpy as np
import torch
from pathlib import Path

# Import model classes from training script
sys.path.insert(0, str(Path(__file__).parent))
from train_v4_multihead import (
    CNNMambaV4MultiHead, V4MultiHeadDataset, DayGroupedSampler,
    run_oot_inference, compute_ic,
    N_EVENT_FEATURES, DIR_HORIZONS, PRESSURE_HORIZONS
)
from torch.utils.data import DataLoader

ROOT = Path("/home/jupiter/Lvl3Quant")
PRED_DIR = ROOT / "output" / "v4_multihead_pressure_v1"
MBO_DIR = ROOT / "data" / "processed" / "mbo_events_smart_v3"
PRESSURE_DIR = ROOT / "data" / "processed" / "smooth_pressure_targets"
TRADES_DIR = ROOT / "data" / "derived" / "mid_price_cache_hc439"
OUT_DIR = ROOT / "output" / "v4_inference_cpu"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Match training config
SEQ_LEN = 100
BATCH_SIZE = 512


def get_existing_pred_dates():
    """Get dates that already have V4 predictions."""
    dates = set()
    for f in PRED_DIR.glob("fold_*_oot_predictions.npz"):
        d = np.load(str(f), allow_pickle=True)
        date = str(d['oot_files'][0]).split('/')[-1].split('_')[0]
        dates.add(date)
    return dates


def get_expandable_dates():
    """Dates with tick data + MBO events but no V4 predictions."""
    existing = get_existing_pred_dates()
    trades_dates = {f.stem.replace('_trades', '') for f in TRADES_DIR.glob("*_trades.npz")}
    mbo_dates = {f.stem.replace('_mbo_events', '') for f in MBO_DIR.glob("*_mbo_events.npz")}
    return sorted(trades_dates & mbo_dates - existing)


def run_inference_on_dates(dates, fold_num, stride):
    """Run V4 inference on specified dates using given fold's weights."""
    # Load model
    ckpt_path = PRED_DIR / f"fold_{fold_num}_best.pt"
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        return []

    print(f"Loading fold {fold_num} checkpoint...")
    ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)

    model = CNNMambaV4MultiHead(
        n_features=N_EVENT_FEATURES,
        n_dir_horizons=len(DIR_HORIZONS),
        n_pressure_horizons=len(PRESSURE_HORIZONS),
    )
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    model = model.to('cpu')
    print(f"Model loaded (val_loss={ckpt.get('val_loss', '?')}, epoch={ckpt.get('epoch', '?')})")

    results = []

    for date_str in dates:
        out_path = OUT_DIR / f"infer_{date_str}_fold{fold_num}.npz"
        if out_path.exists():
            print(f"  {date_str}: already exists, skipping")
            results.append(str(out_path))
            continue

        event_file = MBO_DIR / f"{date_str}_mbo_events.npz"
        pressure_file = PRESSURE_DIR / f"{date_str}_pressure.npz"

        if not event_file.exists():
            print(f"  {date_str}: no MBO events, skipping")
            continue

        t0 = time.time()
        print(f"  {date_str}: loading data...", end='', flush=True)

        # Build dataset for single date
        pressure_map = {}
        if pressure_file.exists():
            pressure_map[date_str] = pressure_file

        try:
            dataset = V4MultiHeadDataset(
                event_files=[event_file],
                pressure_files=pressure_map,
                seq_len=SEQ_LEN,
                stride=stride,
                cache_days=1,
            )
        except Exception as e:
            print(f" ERROR: {e}")
            continue

        if len(dataset) == 0:
            print(f" no samples, skipping")
            continue

        print(f" {len(dataset)} samples...", end='', flush=True)

        loader = DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=0, pin_memory=False,
        )

        # Run inference
        preds_dict, labels_dict, embeddings = run_oot_inference(
            model, loader, torch.device('cpu'), use_amp=False
        )

        # Compute ICs
        ics = {}
        for head in ['dir', 'ntps', 'eofi', 'pdi', 'tia']:
            if head in preds_dict and head in labels_dict:
                p = preds_dict[head]
                l = labels_dict[head]
                horizons = DIR_HORIZONS if head == 'dir' else PRESSURE_HORIZONS
                for hi, h in enumerate(horizons):
                    if hi < p.shape[1]:
                        mask = ~np.isnan(l[:, hi])
                        if mask.sum() > 100:
                            ic = float(np.corrcoef(
                                scipy.stats.rankdata(p[mask, hi]),
                                scipy.stats.rankdata(l[mask, hi])
                            )[0, 1])
                            ics[f"ic_{head}_{h}"] = ic

        # Get MBO event count for stride detection in FIFO sim
        with np.load(str(event_file), allow_pickle=False) as edata:
            n_events = edata['events'].shape[0]

        # Save in same format as training predictions
        save_dict = {
            'fold': fold_num,
            'oot_files': np.array([str(event_file)]),
            'preds_dir': preds_dict.get('dir', np.array([])),
            'labels_dir': labels_dict.get('dir', np.array([])),
            'preds_ntps': preds_dict.get('ntps', np.array([])),
            'labels_ntps': labels_dict.get('ntps', np.array([])),
            'preds_eofi': preds_dict.get('eofi', np.array([])),
            'labels_eofi': labels_dict.get('eofi', np.array([])),
            'preds_pdi': preds_dict.get('pdi', np.array([])),
            'labels_pdi': labels_dict.get('pdi', np.array([])),
            'preds_tia': preds_dict.get('tia', np.array([])),
            'labels_tia': labels_dict.get('tia', np.array([])),
            'n_events': n_events,
            'stride': stride,
            'inference_fold': fold_num,  # flag that this is inference, not walk-forward
        }
        # Add ICs
        for k, v in ics.items():
            save_dict[k] = v

        np.savez_compressed(str(out_path), **save_dict)

        elapsed = time.time() - t0
        ic_1s = ics.get('ic_dir_1s', float('nan'))
        print(f" IC_1s={ic_1s:.3f}, {elapsed:.1f}s")

        results.append(str(out_path))

        # Free memory
        del dataset, loader, preds_dict, labels_dict, embeddings
        gc.collect()

    return results


if __name__ == "__main__":
    import scipy.stats  # needed for IC computation

    parser = argparse.ArgumentParser()
    parser.add_argument('--dates', nargs='+', help='Specific dates to process')
    parser.add_argument('--fold', type=int, default=142, help='Which fold checkpoint to use')
    parser.add_argument('--stride', type=int, default=500, help='Stride for inference')
    parser.add_argument('--max-dates', type=int, default=999, help='Max dates to process')
    args = parser.parse_args()

    if args.dates:
        dates = args.dates
    else:
        dates = get_expandable_dates()

    dates = dates[:args.max_dates]
    print(f"Running V4 inference on {len(dates)} dates using fold {args.fold} weights (stride={args.stride})")
    print(f"Output: {OUT_DIR}")
    print()

    results = run_inference_on_dates(dates, args.fold, args.stride)
    print(f"\nDone: {len(results)} dates processed")
