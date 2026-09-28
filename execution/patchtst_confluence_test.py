#!/usr/bin/env python3
"""
PatchTST Confluence Test (HC #73)
==================================
Tests whether PatchTST agreement improves CNN-Mamba entries.

Strategy: When CNN-Mamba signals entry AND PatchTST agrees on direction → trade.
          When they disagree → skip.

Uses 7 overlapping OOS dates (Feb 26 - Mar 5, 2026).
"""

import numpy as np
import sys
from pathlib import Path
from collections import defaultdict

LVL3 = Path("/home/jupiter/Lvl3Quant")
CNN_DIR = LVL3 / "output" / "cnn_mamba_v2_smart_v3_mar"
PTST_DIR = LVL3 / "output" / "patchtst_razer_weights"

# Map dates to fold indices
OVERLAP_DATES = ['20260226', '20260227', '20260301', '20260302', '20260303', '20260304', '20260305']

def find_fold_for_date(model_dir, date_str, n_folds):
    """Find which fold contains predictions for a given date."""
    for i in range(n_folds):
        f = model_dir / f"fold_{i:02d}_oot_predictions.npz"
        if not f.exists():
            continue
        d = np.load(f, allow_pickle=True)
        for oot_file in d['oot_files']:
            fname = str(oot_file).replace('\\', '/').split('/')[-1]
            if fname.startswith(date_str):
                return i, d
    return None, None


def align_predictions(cnn_preds, ptst_preds):
    """Align predictions by subsampling the denser one."""
    n_cnn = len(cnn_preds)
    n_ptst = len(ptst_preds)

    if n_cnn == n_ptst:
        return cnn_preds, ptst_preds

    # Subsample the longer array to match the shorter
    ratio = n_ptst / n_cnn
    if ratio > 1.5:
        # PatchTST has more predictions, subsample it
        indices = np.round(np.linspace(0, n_ptst - 1, n_cnn)).astype(int)
        return cnn_preds, ptst_preds[indices]
    elif ratio < 0.67:
        # CNN has more predictions, subsample it
        indices = np.round(np.linspace(0, n_cnn - 1, n_ptst)).astype(int)
        return cnn_preds[indices], ptst_preds
    else:
        # Close enough, truncate to shorter
        n = min(n_cnn, n_ptst)
        return cnn_preds[:n], ptst_preds[:n]


def compute_metrics(preds, labels, tag=""):
    """Compute trading metrics for a set of predictions/labels."""
    if len(preds) == 0:
        return {'n': 0, 'da': 0, 'ic': 0, 'wr_sim': 0, 'avg_pnl': 0}

    # Directional accuracy
    correct = (np.sign(preds) == np.sign(labels))
    da = correct.mean()

    # IC (Pearson correlation)
    if np.std(preds) > 0 and np.std(labels) > 0:
        ic = np.corrcoef(preds, labels)[0, 1]
    else:
        ic = 0.0

    # Simple PnL sim (midpoint, no costs) — just for comparison
    # Long when pred > 0, short when pred < 0
    pnl_ticks = np.sign(preds) * labels  # labels are in ticks
    wr = (pnl_ticks > 0).mean()
    avg_pnl = pnl_ticks.mean()

    # Commission-adjusted
    comm = 0.376  # ticks RT
    net_pnl = pnl_ticks - comm
    net_wr = (net_pnl > 0).mean()

    return {
        'n': len(preds),
        'da': float(da),
        'ic': float(ic),
        'wr_gross': float(wr),
        'wr_net': float(net_wr),
        'avg_pnl_gross': float(avg_pnl),
        'avg_pnl_net': float(avg_pnl - comm),
        'sharpe': float(np.mean(net_pnl) / np.std(net_pnl)) if np.std(net_pnl) > 0 else 0,
    }


def run_confluence_test():
    print("=" * 70)
    print("PatchTST CONFLUENCE TEST — CNN-Mamba + PatchTST Agreement")
    print("=" * 70)
    print(f"Overlapping OOS dates: {len(OVERLAP_DATES)}")
    print()

    # Collect all aligned predictions across dates
    all_results = {h: defaultdict(list) for h in ['1s', '5s', '10s']}
    horizon_idx = {'1s': 0, '5s': 1, '10s': 2}

    for date in OVERLAP_DATES:
        cnn_fold, cnn_data = find_fold_for_date(CNN_DIR, date, 10)
        ptst_fold, ptst_data = find_fold_for_date(PTST_DIR, date, 17)

        if cnn_data is None or ptst_data is None:
            print(f"  {date}: SKIP (missing fold data)")
            continue

        cnn_preds_all = cnn_data['predictions']  # (N, 3)
        cnn_labels_all = cnn_data['labels']       # (N, 3)
        ptst_preds_all = ptst_data['predictions']  # (M, 3)
        ptst_labels_all = ptst_data['labels']      # (M, 3)

        print(f"  {date}: CNN fold {cnn_fold} ({len(cnn_preds_all)} preds) | "
              f"PatchTST fold {ptst_fold} ({len(ptst_preds_all)} preds)")

        for h_name, h_idx in horizon_idx.items():
            cnn_p = cnn_preds_all[:, h_idx]
            cnn_l = cnn_labels_all[:, h_idx]
            ptst_p = ptst_preds_all[:, h_idx]

            # Align
            cnn_p_aligned, ptst_p_aligned = align_predictions(cnn_p, ptst_p)
            cnn_l_aligned = cnn_l[:len(cnn_p_aligned)] if len(cnn_p) >= len(cnn_p_aligned) else cnn_l

            # Ensure same length
            n = min(len(cnn_p_aligned), len(ptst_p_aligned), len(cnn_l_aligned))
            cnn_p_aligned = cnn_p_aligned[:n]
            ptst_p_aligned = ptst_p_aligned[:n]
            cnn_l_aligned = cnn_l_aligned[:n]

            # Store for aggregated analysis
            all_results[h_name]['cnn_preds'].append(cnn_p_aligned)
            all_results[h_name]['ptst_preds'].append(ptst_p_aligned)
            all_results[h_name]['labels'].append(cnn_l_aligned)

    print()

    # Aggregate across all dates
    for h_name in ['1s', '5s', '10s']:
        if not all_results[h_name]['cnn_preds']:
            continue

        cnn_preds = np.concatenate(all_results[h_name]['cnn_preds'])
        ptst_preds = np.concatenate(all_results[h_name]['ptst_preds'])
        labels = np.concatenate(all_results[h_name]['labels'])

        print(f"\n{'='*70}")
        print(f"HORIZON: {h_name} — {len(cnn_preds):,} aligned predictions")
        print(f"{'='*70}")

        # Compute z-scores for gating
        cnn_std = np.std(cnn_preds)
        cnn_z = cnn_preds / cnn_std if cnn_std > 0 else cnn_preds
        ptst_std = np.std(ptst_preds)
        ptst_z = ptst_preds / ptst_std if ptst_std > 0 else ptst_preds

        # Agreement: both predict same direction
        agree = np.sign(cnn_preds) == np.sign(ptst_preds)
        disagree = ~agree

        print(f"\nAgreement rate: {agree.mean():.1%}")
        print(f"Disagree rate: {disagree.mean():.1%}")

        # Test configurations
        configs = [
            ("ALL CNN-Mamba (baseline)", np.ones(len(cnn_preds), dtype=bool)),
            ("CNN + PatchTST AGREE", agree),
            ("CNN + PatchTST DISAGREE", disagree),
        ]

        # Add z-score gated configs
        for z_thresh in [1.0, 1.5, 2.0, 3.0]:
            mask_cnn = np.abs(cnn_z) >= z_thresh
            mask_cnn_agree = mask_cnn & agree
            configs.append((f"CNN z>{z_thresh} (baseline)", mask_cnn))
            configs.append((f"CNN z>{z_thresh} + PTST agree", mask_cnn_agree))

        # Also test: CNN signal strength + PatchTST confidence
        for cnn_z_thresh in [1.0, 2.0]:
            for ptst_z_thresh in [0.5, 1.0]:
                mask = (np.abs(cnn_z) >= cnn_z_thresh) & (np.abs(ptst_z) >= ptst_z_thresh) & agree
                configs.append((f"CNN z>{cnn_z_thresh} + PTST z>{ptst_z_thresh} agree", mask))

        print(f"\n{'Strategy':<40} {'N':>7} {'DA':>7} {'IC':>7} {'WR_net':>7} {'PnL/trade':>10} {'Sharpe':>7}")
        print("-" * 90)

        for name, mask in configs:
            if mask.sum() == 0:
                print(f"{name:<40} {'0':>7} {'—':>7} {'—':>7} {'—':>7} {'—':>10} {'—':>7}")
                continue

            m = compute_metrics(cnn_preds[mask], labels[mask])
            pnl_str = f"{m['avg_pnl_net']:+.4f}t" if m['n'] > 0 else "—"
            print(f"{name:<40} {m['n']:>7,} {m['da']:>6.1%} {m['ic']:>7.4f} {m['wr_net']:>6.1%} {pnl_str:>10} {m['sharpe']:>7.3f}")

    print(f"\n{'='*70}")
    print("NOTE: All PnL is MIDPOINT-BASED (theoretical). FIFO validation needed.")
    print(f"{'='*70}")


if __name__ == "__main__":
    run_confluence_test()
