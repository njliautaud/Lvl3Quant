#!/usr/bin/env python3
"""
V4 Multihead Model Analysis
============================
Analyze the per-fold OOT predictions from the v4 multihead pressure model.
Check if the directional head produces a stronger tradeable signal than v3.4.2.

Author: Claude (autonomous research, 2026-07-02)
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    format="%(asctime)s [V4-MH] %(levelname)s %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("V4-MH")

ROOT = Path("/home/jupiter/Lvl3Quant")
V4_DIR = ROOT / "output" / "v4_multihead_pressure_v1"
SMART_V3_DIR = ROOT / "data" / "processed" / "mbo_events_smart_v3"
COST_PASSIVE = 0.376  # ticks


def main():
    # Load all fold predictions
    fold_files = sorted(V4_DIR.glob("fold_*_oot_predictions.npz"))
    log.info(f"Found {len(fold_files)} fold prediction files")

    all_preds_dir = []
    all_labels_dir = []
    all_preds_eofi = []
    all_labels_eofi = []
    all_preds_pdi = []
    all_labels_pdi = []
    all_oot_dates = []
    all_ics = {'dir_1s': [], 'dir_5s': [], 'dir_10s': [],
               'eofi_1s': [], 'pdi_1s': []}

    for fp in fold_files:
        try:
            data = np.load(str(fp), allow_pickle=True)
            fold = int(data['fold'])

            # Collect per-fold ICs
            for key in ['ic_dir_1s', 'ic_dir_5s', 'ic_dir_10s', 'ic_eofi_1s', 'ic_pdi_1s']:
                if key in data:
                    short_key = key.replace('ic_', '')
                    all_ics[short_key].append(float(data[key]))

            # Predictions and labels
            all_preds_dir.append(data['preds_dir'])
            all_labels_dir.append(data['labels_dir'])
            all_preds_eofi.append(data['preds_eofi'])
            all_labels_eofi.append(data['labels_eofi'])
            all_preds_pdi.append(data['preds_pdi'])
            all_labels_pdi.append(data['labels_pdi'])

            oot_file = str(data['oot_files'][0]) if 'oot_files' in data else 'unknown'
            all_oot_dates.append(oot_file)

            n = len(data['preds_dir'])
            log.info(f"  Fold {fold}: {n} predictions, OOT={Path(oot_file).stem}")

        except Exception as e:
            log.error(f"  {fp.name}: {e}")

    if not all_preds_dir:
        log.error("No predictions loaded!")
        return

    # Concatenate
    preds_dir = np.concatenate(all_preds_dir)  # shape (N, 3) → 3 horizons
    labels_dir = np.concatenate(all_labels_dir)  # shape (N, 3)
    preds_eofi = np.concatenate(all_preds_eofi)  # shape (N, 4)
    labels_eofi = np.concatenate(all_labels_eofi)  # shape (N, 4)
    preds_pdi = np.concatenate(all_preds_pdi)  # shape (N, 4)
    labels_pdi = np.concatenate(all_labels_pdi)  # shape (N, 4)

    N = len(preds_dir)
    log.info(f"\nTotal concatenated: {N:,} predictions across {len(fold_files)} folds")

    # ================================================================
    # ANALYSIS 1: Concat IC by head
    # ================================================================
    print(f"\n{'='*80}")
    print("ANALYSIS 1: PER-FOLD IC DISTRIBUTION")
    print(f"{'='*80}")
    for key, vals in all_ics.items():
        if vals:
            print(f"  {key:>10}: mean={np.mean(vals):.4f}, "
                  f"std={np.std(vals):.4f}, "
                  f"min={np.min(vals):.4f}, max={np.max(vals):.4f}")

    # ================================================================
    # ANALYSIS 2: Concat IC (proper cross-fold)
    # ================================================================
    print(f"\n{'='*80}")
    print("ANALYSIS 2: CONCATENATED IC (cross-fold)")
    print(f"{'='*80}")

    horizons_dir = ['1s', '5s', '10s']
    for h_idx, h_name in enumerate(horizons_dir):
        p = preds_dir[:, h_idx]
        l = labels_dir[:, h_idx]
        valid = ~np.isnan(p) & ~np.isnan(l) & (l != 0)
        if valid.sum() > 100:
            ic = np.corrcoef(p[valid], l[valid])[0, 1]
            print(f"  dir_{h_name}: IC = {ic:.4f} (n={valid.sum():,})")

    horizons_eofi = ['1s', '5s', '10s', '30s']
    for h_idx, h_name in enumerate(horizons_eofi):
        p = preds_eofi[:, h_idx]
        l = labels_eofi[:, h_idx]
        valid = ~np.isnan(p) & ~np.isnan(l)
        if valid.sum() > 100:
            ic = np.corrcoef(p[valid], l[valid])[0, 1]
            print(f"  eofi_{h_name}: IC = {ic:.4f} (n={valid.sum():,})")

    for h_idx, h_name in enumerate(horizons_eofi):
        p = preds_pdi[:, h_idx]
        l = labels_pdi[:, h_idx]
        valid = ~np.isnan(p) & ~np.isnan(l)
        if valid.sum() > 100:
            ic = np.corrcoef(p[valid], l[valid])[0, 1]
            print(f"  pdi_{h_name}: IC = {ic:.4f} (n={valid.sum():,})")

    # ================================================================
    # ANALYSIS 3: Directional trading edge (using labels_dir as PnL proxy)
    # ================================================================
    print(f"\n{'='*80}")
    print("ANALYSIS 3: DIRECTIONAL TRADING EDGE (labels_dir as PnL)")
    print(f"{'='*80}")

    for h_idx, h_name in enumerate(horizons_dir):
        p = preds_dir[:, h_idx]
        l = labels_dir[:, h_idx]
        valid = ~np.isnan(p) & ~np.isnan(l)

        if valid.sum() < 100:
            continue

        pv = p[valid]
        lv = l[valid]

        # Trade in predicted direction
        dirs = np.sign(pv)
        pnl = dirs * lv
        avg_pnl = np.mean(pnl)

        # Direction balance
        long_frac = np.mean(dirs > 0)
        wr = np.mean(pnl > 0)

        print(f"\n  Horizon {h_name}:")
        print(f"    N: {valid.sum():,}")
        print(f"    Avg gross PnL: {avg_pnl:+.4f} (units=label_dir)")
        print(f"    Long fraction: {long_frac:.1%}")
        print(f"    Win rate: {wr:.1%}")

        # Quantile breakdown
        abs_p = np.abs(pv)
        for pct_lo, pct_hi in [(0,50), (50,75), (75,90), (90,95), (95,100)]:
            lo = np.percentile(abs_p, pct_lo)
            hi = np.percentile(abs_p, pct_hi) if pct_hi < 100 else np.inf
            mask = (abs_p >= lo) & (abs_p < hi)
            if mask.sum() < 50:
                continue
            d = np.sign(pv[mask])
            pl = d * lv[mask]
            avg = np.mean(pl)
            w = np.mean(pl > 0)
            print(f"    {pct_lo}-{pct_hi}%: n={mask.sum():>6}, "
                  f"avg={avg:+.4f}, WR={w:.1%}, L%={np.mean(d>0):.0%}")

    # ================================================================
    # ANALYSIS 4: What are the label units?
    # ================================================================
    print(f"\n{'='*80}")
    print("ANALYSIS 4: LABEL DISTRIBUTIONS (understanding units)")
    print(f"{'='*80}")

    for name, arr in [('labels_dir', labels_dir), ('labels_eofi', labels_eofi),
                       ('labels_pdi', labels_pdi)]:
        print(f"\n  {name} (shape {arr.shape}):")
        for col in range(arr.shape[1]):
            v = arr[:, col]
            valid = ~np.isnan(v)
            if valid.sum() > 0:
                vv = v[valid]
                print(f"    col {col}: mean={np.mean(vv):+.3f}, std={np.std(vv):.3f}, "
                      f"min={np.min(vv):.2f}, max={np.max(vv):.2f}, "
                      f"p1={np.percentile(vv,1):.2f}, p99={np.percentile(vv,99):.2f}")

    # ================================================================
    # ANALYSIS 5: If labels_dir is in ticks, compute tradeable metrics
    # ================================================================
    print(f"\n{'='*80}")
    print("ANALYSIS 5: ECONOMIC VIABILITY (if labels_dir in ticks)")
    print(f"{'='*80}")

    for h_idx, h_name in enumerate(horizons_dir):
        p = preds_dir[:, h_idx]
        l = labels_dir[:, h_idx]
        valid = ~np.isnan(p) & ~np.isnan(l)

        if valid.sum() < 100:
            continue

        pv = p[valid]
        lv = l[valid]
        dirs = np.sign(pv)
        pnl = dirs * lv  # gross PnL per trade in label units

        avg_gross = np.mean(pnl)

        # Per-direction
        long_mask = dirs > 0
        short_mask = dirs < 0

        long_avg = np.mean(pnl[long_mask]) if long_mask.sum() > 0 else 0
        short_avg = np.mean(pnl[short_mask]) if short_mask.sum() > 0 else 0

        print(f"\n  Horizon {h_name}:")
        print(f"    Overall: gross={avg_gross:+.4f}/trade, "
              f"net_passive={avg_gross - COST_PASSIVE:+.4f}/trade")
        print(f"    Longs (n={long_mask.sum():,}): {long_avg:+.4f}/trade")
        print(f"    Shorts (n={short_mask.sum():,}): {short_avg:+.4f}/trade")

        if avg_gross > 0:
            trades_to_breakeven = COST_PASSIVE / avg_gross
            print(f"    Signal needs {trades_to_breakeven:.1f}x improvement to cover costs")

    print(f"\n{'='*80}")
    print("DONE")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
