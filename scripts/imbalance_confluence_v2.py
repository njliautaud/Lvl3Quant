"""
Imbalance-Conditioned Confluence v2 (HC #514) — FIXED ALIGNMENT
=================================================================
v1 had a critical bug: predictions (81K, stride 250) were compared
1:1 with raw events (20M). This version properly maps each prediction
to its corresponding event via: event_idx = pred_idx * stride + window - 1

Tests: when models predict direction AND real-time imbalance confirms,
does accuracy lift above cost-clearing threshold?

Cost: 0.376 ticks only (HC #512).
"""

import os
import sys
import json
import glob
import time
import numpy as np
from collections import defaultdict
from scipy.stats import spearmanr

# Imbalance feature indices in 25-feature smart_v3
IMBALANCE_FEATS = {
    'cancel_asym': 6,       # cancel side asymmetry (50-event)
    'ofi_500': 7,           # rolling OFI z-score (500-event)
    'queue_replen': 15,     # queue replenishment ratio
    'ofi_x_spread': 17,     # OFI × spread interaction
    'buy_sell_ratio': 19,   # EWMA bid/ask event intensity [-1,+1]
    'sweep_intensity': 21,  # aggressive fill clustering
    'ofi_short': 22,        # short-term OFI (100-event)
    'ofi_long': 23,         # long-term OFI (2000-event)
    'ofi_accel': 24,        # OFI acceleration (rate of change)
}

COST_TICKS = 0.376


def align_predictions_to_events(pred_npz, event_npz, pressure_npz=None):
    """
    Properly align predictions with raw event features using stride/window metadata.
    Returns aligned: predictions, labels_real, imbalance_features, [pressure]
    """
    preds = pred_npz['predictions']       # (N_pred, 3)
    labels_pred = pred_npz['labels']       # (N_pred, 3) — model's training labels
    window = int(pred_npz['window_size'])  # 3000
    stride = int(pred_npz['stride'])       # 250

    events = event_npz['events']           # (N_events, 25)
    labels_1s = event_npz['labels_1s']
    labels_5s = event_npz['labels_5s']
    labels_10s = event_npz['labels_10s']
    timestamps = event_npz['timestamps']

    n_pred = preds.shape[0]
    n_events = events.shape[0]

    # Map each prediction to its reference event
    # Prediction i uses events[i*stride : i*stride+window]
    # The "current moment" for the prediction = last event in window
    ref_indices = np.arange(n_pred) * stride + (window - 1)

    # Filter to valid indices
    valid = ref_indices < n_events
    ref_indices = ref_indices[valid]
    preds = preds[valid]
    labels_pred = labels_pred[valid]

    # Extract imbalance features at each reference event
    imb_feats = {}
    for fname, fidx in IMBALANCE_FEATS.items():
        imb_feats[fname] = events[ref_indices, fidx]

    # Get real directional labels at reference events
    real_labels = np.stack([
        labels_1s[ref_indices],
        labels_5s[ref_indices],
        labels_10s[ref_indices],
    ], axis=1)

    # Timestamps for reference
    ts = timestamps[ref_indices]

    # Pressure targets if available
    pressure = None
    if pressure_npz is not None:
        n_pressure = pressure_npz['eofi'].shape[0]
        # Pressure is per-event, same alignment as events
        if n_pressure == n_events:
            pressure = {
                'eofi': pressure_npz['eofi'][ref_indices],
                'ntps': pressure_npz['ntps'][ref_indices],
                'pdi': pressure_npz['pdi'][ref_indices],
                'tia': pressure_npz['tia'][ref_indices],
            }

    return preds, labels_pred, real_labels, imb_feats, pressure, ts


def compute_composite_imbalance(imb_feats):
    """Weighted combination of imbalance features. All are directional (+ = buy pressure)."""
    return (
        0.25 * imb_feats['ofi_500'] +
        0.20 * imb_feats['buy_sell_ratio'] +
        0.15 * imb_feats['ofi_short'] +
        0.15 * imb_feats['ofi_accel'] +
        0.10 * imb_feats['sweep_intensity'] +
        0.10 * imb_feats['ofi_long'] +
        0.05 * imb_feats['cancel_asym']
    )


def analyze_date(preds_all, real_labels, imb_feats, pressure, horizons=['1s', '5s', '10s']):
    """
    For a single date: compute baseline accuracy, imbalance-conditioned accuracy,
    and P&L for various filter combinations.
    """
    N = preds_all.shape[0]
    composite = compute_composite_imbalance(imb_feats)

    results = {
        'n_samples': N,
        'baseline': {},
        'imb_conditioned': {},
        'eofi_conditioned': {},
        'grid': [],
    }

    for h_idx, h_name in enumerate(horizons):
        ens_pred = preds_all[:, h_idx]
        real = real_labels[:, h_idx]

        # === BASELINE: direction accuracy of ensemble ===
        # Real labels: continuous float. Positive = price went up.
        # Threshold at 0 for direction
        pred_dir = np.sign(ens_pred)

        # For real labels: > 0.5 = up (in binary 0/1 scheme). But let's check the distribution
        # Actually, labels might be continuous tick moves or binary. Let's check both.
        # Use > 0.5 for binary labels (0, 0.5, 1 encoding)
        real_up = (real > 0.5).astype(float)
        real_down = (real < 0.5).astype(float)
        # Fraction that's "flat" (== 0.5)
        flat_frac = np.mean(np.abs(real - 0.5) < 0.01)

        if flat_frac > 0.3:
            # Labels are mostly binary (0, 0.5, 1) — use 0.5 threshold
            true_dir = np.where(real > 0.5, 1.0, np.where(real < 0.5, -1.0, 0.0))
        else:
            # Labels are continuous — use 0 threshold
            true_dir = np.sign(real - 0.5)

        # Only count non-flat samples for accuracy
        non_flat = true_dir != 0
        if non_flat.sum() < 100:
            continue

        base_acc = np.mean(pred_dir[non_flat] == true_dir[non_flat])
        results['baseline'][h_name] = {
            'accuracy': float(base_acc),
            'n_non_flat': int(non_flat.sum()),
            'flat_frac': float(flat_frac),
        }

        # === IMBALANCE CONDITIONED ===
        # Trade only when composite imbalance confirms prediction direction
        for imb_thresh in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
            alignment = pred_dir * composite
            mask = (alignment > imb_thresh) & non_flat
            if mask.sum() < 50:
                continue

            cond_acc = np.mean(pred_dir[mask] == true_dir[mask])
            lift = cond_acc - base_acc

            # P&L approximation
            # Gross: +1 tick if correct direction, -1 tick if wrong (simplified)
            correct = (pred_dir[mask] == true_dir[mask]).astype(float)
            gross_ticks = correct.mean() * 2 - 1  # maps 50%→0, 60%→0.2, etc.
            net_ticks = gross_ticks - COST_TICKS

            results['grid'].append({
                'horizon': h_name,
                'filter': 'composite',
                'imb_thresh': imb_thresh,
                'n_trades': int(mask.sum()),
                'filter_rate': float(mask.sum() / N),
                'accuracy': float(cond_acc),
                'lift': float(lift),
                'gross_ticks': float(gross_ticks),
                'net_ticks': float(net_ticks),
                'profitable': net_ticks > 0,
            })

        # === INDIVIDUAL IMBALANCE FEATURES ===
        for fname in ['ofi_500', 'buy_sell_ratio', 'ofi_short', 'ofi_accel', 'sweep_intensity']:
            feat = imb_feats[fname]
            for imb_thresh in [0.0, 0.3, 0.5, 0.7]:
                alignment = pred_dir * feat
                mask = (alignment > imb_thresh) & non_flat
                if mask.sum() < 50:
                    continue

                cond_acc = np.mean(pred_dir[mask] == true_dir[mask])
                lift = cond_acc - base_acc
                correct = (pred_dir[mask] == true_dir[mask]).astype(float)
                gross = correct.mean() * 2 - 1
                net = gross - COST_TICKS

                results['grid'].append({
                    'horizon': h_name,
                    'filter': fname,
                    'imb_thresh': imb_thresh,
                    'n_trades': int(mask.sum()),
                    'filter_rate': float(mask.sum() / N),
                    'accuracy': float(cond_acc),
                    'lift': float(lift),
                    'gross_ticks': float(gross),
                    'net_ticks': float(net),
                    'profitable': net > 0,
                })

        # === EOFI CONDITIONED (if available) ===
        if pressure is not None:
            eofi = pressure['eofi']
            for imb_thresh in [0.0, 0.05, 0.1, 0.2, 0.3]:
                alignment = pred_dir * eofi
                mask = (alignment > imb_thresh) & non_flat
                if mask.sum() < 50:
                    continue

                cond_acc = np.mean(pred_dir[mask] == true_dir[mask])
                lift = cond_acc - base_acc
                correct = (pred_dir[mask] == true_dir[mask]).astype(float)
                gross = correct.mean() * 2 - 1
                net = gross - COST_TICKS

                results['grid'].append({
                    'horizon': h_name,
                    'filter': 'eofi',
                    'imb_thresh': imb_thresh,
                    'n_trades': int(mask.sum()),
                    'filter_rate': float(mask.sum() / N),
                    'accuracy': float(cond_acc),
                    'lift': float(lift),
                    'gross_ticks': float(gross),
                    'net_ticks': float(net),
                    'profitable': net > 0,
                })

            # NTPS conditioned
            ntps = pressure['ntps']
            for imb_thresh in [0.0, 0.1, 0.2, 0.3, 0.5]:
                alignment = pred_dir * ntps
                mask = (alignment > imb_thresh) & non_flat
                if mask.sum() < 50:
                    continue

                cond_acc = np.mean(pred_dir[mask] == true_dir[mask])
                lift = cond_acc - base_acc
                correct = (pred_dir[mask] == true_dir[mask]).astype(float)
                gross = correct.mean() * 2 - 1
                net = gross - COST_TICKS

                results['grid'].append({
                    'horizon': h_name,
                    'filter': 'ntps',
                    'imb_thresh': imb_thresh,
                    'n_trades': int(mask.sum()),
                    'filter_rate': float(mask.sum() / N),
                    'accuracy': float(cond_acc),
                    'lift': float(lift),
                    'gross_ticks': float(gross),
                    'net_ticks': float(net),
                    'profitable': net > 0,
                })

    return results


def main():
    base = '/home/jupiter/Lvl3Quant'
    out_dir = os.path.join(base, 'output/imbalance_confluence_v2')
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("IMBALANCE CONFLUENCE v2 — FIXED ALIGNMENT (HC #514)")
    print("=" * 70)
    print(f"Cost: {COST_TICKS} ticks RT")
    print()

    # Prediction dirs
    pred_dirs = {
        'cm_v2': os.path.join(base, 'output/cnn_mamba_v2_bulk_oot'),
        'cm_v2b': os.path.join(base, 'output/cnn_mamba_v2_bulk_oot_v2'),
        'ptst': os.path.join(base, 'output/patchtst_bulk_oot'),
    }
    events_dir = os.path.join(base, 'data/processed/mbo_events_smart_v3')
    pressure_dir = os.path.join(base, 'data/processed/smooth_pressure_targets')

    # Find dates with both predictions and events
    event_files = {os.path.basename(f).split('_')[0]: f
                   for f in glob.glob(os.path.join(events_dir, '*_mbo_events.npz'))}
    pressure_files = {os.path.basename(f).split('_')[0]: f
                      for f in glob.glob(os.path.join(pressure_dir, '*_pressure.npz'))}

    # Load prediction dates per model
    model_dates = {}
    for mname, mdir in pred_dirs.items():
        files = glob.glob(os.path.join(mdir, '*_predictions.npz'))
        model_dates[mname] = {os.path.basename(f).split('_')[0]: f for f in files}
        print(f"  {mname}: {len(model_dates[mname])} dates")

    # Find dates with events + at least 1 model
    all_pred_dates = set()
    for md in model_dates.values():
        all_pred_dates |= set(md.keys())
    common_dates = sorted(all_pred_dates & set(event_files.keys()))
    print(f"  events: {len(event_files)} dates")
    print(f"  pressure: {len(pressure_files)} dates")
    print(f"  Overlapping (events + any model): {len(common_dates)}")
    print()

    # Process each date
    all_date_results = {}
    t0 = time.time()

    header = f"{'Date':>10} {'N_pred':>8} {'Base_1s':>8} {'Cond_1s':>8} {'Lift_1s':>8} {'Best_Net':>9} {'Best_Filter':>15}"
    print(header)
    print("-" * 75)

    for date in common_dates:
        # Load events
        ev = np.load(event_files[date], allow_pickle=True)

        # Load pressure if available
        pr = None
        if date in pressure_files:
            pr = np.load(pressure_files[date], allow_pickle=True)

        # Load and ensemble predictions from available models
        model_preds = []
        for mname, mfiles in model_dates.items():
            if date in mfiles:
                p = np.load(mfiles[date], allow_pickle=True)
                model_preds.append(p)

        if not model_preds:
            continue

        # Use first model's metadata for alignment (all should match)
        ref = model_preds[0]
        window = int(ref['window_size'])
        stride = int(ref['stride'])
        n_pred = ref['predictions'].shape[0]

        # Align each model
        aligned_preds = []
        for mp in model_preds:
            p, lp, rl, imb, pressure, ts = align_predictions_to_events(mp, ev, pr)
            aligned_preds.append(p)

        # Ensemble: average predictions
        if len(aligned_preds) > 1:
            min_n = min(p.shape[0] for p in aligned_preds)
            ensemble = np.mean([p[:min_n] for p in aligned_preds], axis=0)
        else:
            ensemble = aligned_preds[0]
            min_n = ensemble.shape[0]

        # Get properly aligned features for ensemble size
        _, _, real_labels, imb_feats, pressure_aligned, timestamps = align_predictions_to_events(
            ref, ev, pr
        )
        real_labels = real_labels[:min_n]
        imb_feats = {k: v[:min_n] for k, v in imb_feats.items()}
        if pressure_aligned:
            pressure_aligned = {k: v[:min_n] for k, v in pressure_aligned.items()}
        ensemble = ensemble[:min_n]

        # Analyze
        date_results = analyze_date(ensemble, real_labels, imb_feats, pressure_aligned)
        all_date_results[date] = date_results

        # Print summary
        base_1s = date_results['baseline'].get('1s', {}).get('accuracy', float('nan'))
        best_net = -999
        best_filter = 'none'
        best_acc = 0
        for g in date_results['grid']:
            if g['horizon'] == '1s' and g['net_ticks'] > best_net:
                best_net = g['net_ticks']
                best_filter = f"{g['filter']}>{g['imb_thresh']}"
                best_acc = g['accuracy']

        lift = best_acc - base_1s if best_net > -999 else 0
        prof = "✅" if best_net > 0 else "❌"
        print(f"{date:>10} {min_n:8d} {base_1s:8.4f} {best_acc:8.4f} {lift:+8.4f} {best_net:+9.4f} {best_filter:>15} {prof}")

    elapsed = time.time() - t0

    # === AGGREGATE SUMMARY ===
    print("\n" + "=" * 70)
    print("AGGREGATE SUMMARY — ALL DATES")
    print("=" * 70)

    # Collect all grid results
    agg = defaultdict(lambda: defaultdict(list))
    for date, dr in all_date_results.items():
        for g in dr['grid']:
            key = (g['horizon'], g['filter'], g['imb_thresh'])
            agg[key]['accuracy'].append(g['accuracy'])
            agg[key]['net_ticks'].append(g['net_ticks'])
            agg[key]['n_trades'].append(g['n_trades'])
            agg[key]['lift'].append(g['lift'])

    # Baseline stats
    for h in ['1s', '5s', '10s']:
        base_accs = [dr['baseline'].get(h, {}).get('accuracy', 0) for dr in all_date_results.values()
                     if h in dr['baseline']]
        if base_accs:
            print(f"\n  {h} baseline accuracy: {np.mean(base_accs):.4f} ± {np.std(base_accs):.4f}")

    # Best configs by net ticks
    print(f"\n{'Horizon':>8} {'Filter':>18} {'Thresh':>7} {'Avg_Acc':>8} {'Avg_Lift':>9} {'Avg_Net':>8} {'Trades/d':>9} {'Days_Prof':>10}")
    print("-" * 90)

    sorted_configs = sorted(agg.items(),
                           key=lambda x: np.mean(x[1]['net_ticks']),
                           reverse=True)

    for (h, filt, thresh), metrics in sorted_configs[:30]:
        avg_acc = np.mean(metrics['accuracy'])
        avg_lift = np.mean(metrics['lift'])
        avg_net = np.mean(metrics['net_ticks'])
        avg_trades = np.mean(metrics['n_trades'])
        days_prof = np.mean([1 for n in metrics['net_ticks'] if n > 0])
        prof = "✅" if avg_net > 0 else "❌"
        print(f"{h:>8} {filt:>18} {thresh:7.2f} {avg_acc:8.4f} {avg_lift:+9.4f} {avg_net:+8.4f} {avg_trades:9.0f} {days_prof:10.1%} {prof}")

    # Save
    summary = {
        'n_dates': len(common_dates),
        'n_analyzed': len(all_date_results),
        'cost_ticks': COST_TICKS,
        'elapsed_s': elapsed,
        'baselines': {
            h: {
                'mean_accuracy': float(np.mean([dr['baseline'].get(h, {}).get('accuracy', 0)
                                                 for dr in all_date_results.values() if h in dr['baseline']])),
            } for h in ['1s', '5s', '10s']
        },
        'top_configs': [
            {
                'horizon': h, 'filter': filt, 'threshold': thresh,
                'avg_accuracy': float(np.mean(metrics['accuracy'])),
                'avg_net_ticks': float(np.mean(metrics['net_ticks'])),
                'avg_lift': float(np.mean(metrics['lift'])),
            }
            for (h, filt, thresh), metrics in sorted_configs[:20]
        ],
    }

    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(out_dir, 'per_date_results.json'), 'w') as f:
        json.dump(all_date_results, f, indent=2, default=str)

    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Results saved to {out_dir}")


if __name__ == '__main__':
    main()
