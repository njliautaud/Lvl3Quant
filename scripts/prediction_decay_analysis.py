"""
Prediction Decay Analysis (HC #514 follow-up)
================================================
The imbalance confluence v2 showed a striking pattern:
  - Feb 24 - Mar 5: 56-85% conditioned accuracy (PROFITABLE)
  - Mar 6 onward: drops to 48-54% (not profitable)

This script investigates WHY:
1. Is prediction FRESHNESS the key? (distance from training fold)
2. Do models decay over time?
3. Does the market regime shift explain the drop?
4. Per-date IC analysis to find the decay curve

If prediction freshness matters, the solution is clear:
retrain more frequently and only trade on fresh predictions.

Cost: 0.376 ticks (HC #512).
"""

import os
import sys
import json
import glob
import numpy as np
from collections import defaultdict
from scipy.stats import spearmanr
from datetime import datetime


def load_pred_date(pred_dir, date):
    f = os.path.join(pred_dir, f'{date}_predictions.npz')
    if not os.path.exists(f):
        return None
    d = np.load(f, allow_pickle=True)
    return {
        'predictions': d['predictions'],
        'labels': d['labels'],
        'window_size': int(d['window_size']),
        'stride': int(d['stride']),
    }


def load_events_date(events_dir, date):
    f = os.path.join(events_dir, f'{date}_mbo_events.npz')
    if not os.path.exists(f):
        return None
    d = np.load(f, allow_pickle=True)
    return {
        'events': d['events'],
        'labels_1s': d['labels_1s'],
        'labels_5s': d['labels_5s'],
        'labels_10s': d['labels_10s'],
    }


def compute_per_date_metrics(preds, labels, events, window, stride):
    """Compute comprehensive metrics for a single date."""
    n_pred = preds.shape[0]
    n_events = events.shape[0]

    # Align predictions to events
    ref_indices = np.arange(n_pred) * stride + (window - 1)
    valid = ref_indices < n_events
    ref_indices = ref_indices[valid]
    preds = preds[valid]
    labels = labels[valid]

    results = {}
    horizons = ['1s', '5s', '10s']

    for h_idx, h_name in enumerate(horizons):
        p = preds[:, h_idx]
        # Use prediction labels (from training) for IC
        l = labels[:, h_idx]

        # IC
        if np.std(p) > 1e-8 and np.std(l) > 1e-8:
            ic, _ = spearmanr(p, l)
        else:
            ic = 0.0

        # Direction accuracy against prediction labels
        pred_dir = np.sign(p)
        # For binary labels: > 0.5 = up, < 0.5 = down, == 0.5 = flat
        flat = np.abs(l - 0.5) < 0.01
        non_flat = ~flat
        true_dir = np.where(l > 0.5, 1.0, -1.0)  # always compute
        if non_flat.sum() > 100:
            acc = np.mean(pred_dir[non_flat] == true_dir[non_flat])
        else:
            acc = 0.5

        # Prediction magnitude stats
        mean_abs = np.mean(np.abs(p))
        std_pred = np.std(p)

        # Top 5% confidence accuracy
        abs_p = np.abs(p)
        top5_mask = abs_p >= np.percentile(abs_p, 95)
        if (top5_mask & non_flat).sum() > 20:
            top5_acc = np.mean(pred_dir[top5_mask & non_flat] == true_dir[top5_mask & non_flat])
        else:
            top5_acc = 0.5

        # Buy/sell ratio conditioned (feature 19)
        bsr = events[ref_indices, 19]  # buy_sell_intensity_ratio
        bsr_confirms = (pred_dir * bsr) > 0.5
        if (bsr_confirms & non_flat).sum() > 50:
            bsr_acc = np.mean(pred_dir[bsr_confirms & non_flat] == true_dir[bsr_confirms & non_flat])
        else:
            bsr_acc = 0.5

        # P&L estimate (1 tick per correct, -1 per wrong, - 0.376 commission)
        correct = pred_dir[non_flat] == true_dir[non_flat]
        gross = correct.astype(float).mean() * 2 - 1
        net = gross - 0.376

        results[h_name] = {
            'ic': float(ic),
            'accuracy': float(acc),
            'top5_accuracy': float(top5_acc),
            'bsr_accuracy': float(bsr_acc),
            'mean_abs_pred': float(mean_abs),
            'std_pred': float(std_pred),
            'n_samples': int(non_flat.sum()),
            'flat_frac': float(flat.mean()),
            'gross_ticks': float(gross),
            'net_ticks': float(net),
        }

    return results


def main():
    base = '/home/jupiter/Lvl3Quant'
    out_dir = os.path.join(base, 'output/prediction_decay_v1')
    os.makedirs(out_dir, exist_ok=True)

    # Model prediction directories
    pred_dirs = {
        'cm_v2': os.path.join(base, 'output/cnn_mamba_v2_bulk_oot'),
        'cm_v2b': os.path.join(base, 'output/cnn_mamba_v2_bulk_oot_v2'),
        'ptst': os.path.join(base, 'output/patchtst_bulk_oot'),
    }
    events_dir = os.path.join(base, 'data/processed/mbo_events_smart_v3')

    # Collect all dates
    all_dates = set()
    model_date_map = {}
    for mname, mdir in pred_dirs.items():
        files = glob.glob(os.path.join(mdir, '*_predictions.npz'))
        dates = {os.path.basename(f).split('_')[0] for f in files}
        model_date_map[mname] = dates
        all_dates |= dates
        print(f"  {mname}: {len(dates)} dates")

    event_dates = {os.path.basename(f).split('_')[0]
                   for f in glob.glob(os.path.join(events_dir, '*_mbo_events.npz'))}
    print(f"  events: {len(event_dates)} dates")

    # Use dates that have events
    analyze_dates = sorted(all_dates & event_dates)
    print(f"\n  Analyzing {len(analyze_dates)} dates")
    print()

    # Headers
    print(f"{'Date':>10} {'Model':>8} {'IC_1s':>7} {'Acc_1s':>7} {'Top5%':>7} {'BSR_Acc':>7} {'Net_1s':>8} {'IC_5s':>7} {'IC_10s':>7}")
    print("-" * 85)

    all_results = {}
    for date in analyze_dates:
        ev = load_events_date(events_dir, date)
        if ev is None:
            continue

        for mname, mdir in pred_dirs.items():
            if date not in model_date_map[mname]:
                continue

            p = load_pred_date(mdir, date)
            if p is None:
                continue

            metrics = compute_per_date_metrics(
                p['predictions'], p['labels'],
                ev['events'], p['window_size'], p['stride']
            )

            key = f"{date}_{mname}"
            all_results[key] = {
                'date': date,
                'model': mname,
                **metrics,
            }

            m1 = metrics.get('1s', {})
            m5 = metrics.get('5s', {})
            m10 = metrics.get('10s', {})
            print(f"{date:>10} {mname:>8} {m1.get('ic',0):7.4f} {m1.get('accuracy',0):7.4f} {m1.get('top5_accuracy',0):7.4f} {m1.get('bsr_accuracy',0):7.4f} {m1.get('net_ticks',0):+8.4f} {m5.get('ic',0):7.4f} {m10.get('ic',0):7.4f}")

    # === DECAY ANALYSIS ===
    print("\n" + "=" * 70)
    print("TEMPORAL DECAY ANALYSIS")
    print("=" * 70)

    # Group by date, average across models
    date_metrics = defaultdict(lambda: defaultdict(list))
    for key, r in all_results.items():
        date = r['date']
        for h in ['1s', '5s', '10s']:
            if h in r:
                for metric in ['ic', 'accuracy', 'top5_accuracy', 'bsr_accuracy', 'net_ticks']:
                    date_metrics[date][f'{h}_{metric}'].append(r[h][metric])

    # Print decay curve
    print(f"\n{'Date':>10} {'Avg_IC_1s':>10} {'Avg_Acc_1s':>11} {'Avg_BSR_1s':>11} {'Avg_Net_1s':>11} {'Models':>7}")
    print("-" * 65)
    dates_sorted = sorted(date_metrics.keys())
    for date in dates_sorted:
        m = date_metrics[date]
        ic = np.mean(m.get('1s_ic', [0]))
        acc = np.mean(m.get('1s_accuracy', [0]))
        bsr = np.mean(m.get('1s_bsr_accuracy', [0]))
        net = np.mean(m.get('1s_net_ticks', [0]))
        n_models = len(m.get('1s_ic', []))
        prof = "✅" if net > 0 else "❌"
        print(f"{date:>10} {ic:10.4f} {acc:11.4f} {bsr:11.4f} {net:+11.4f} {n_models:7d} {prof}")

    # Weekly aggregation
    print("\n\nWEEKLY AGGREGATION:")
    print(f"{'Week':>12} {'Avg_IC_1s':>10} {'Avg_Acc_1s':>11} {'Avg_BSR_1s':>11} {'Days_Prof':>10}")
    print("-" * 60)

    # Group by week
    weeks = defaultdict(lambda: defaultdict(list))
    for date in dates_sorted:
        dt = datetime.strptime(date, '%Y%m%d')
        week_key = dt.strftime('%Y-W%U')
        m = date_metrics[date]
        weeks[week_key]['ic'].append(np.mean(m.get('1s_ic', [0])))
        weeks[week_key]['acc'].append(np.mean(m.get('1s_accuracy', [0])))
        weeks[week_key]['bsr'].append(np.mean(m.get('1s_bsr_accuracy', [0])))
        weeks[week_key]['net'].append(np.mean(m.get('1s_net_ticks', [0])))

    for week in sorted(weeks.keys()):
        w = weeks[week]
        ic = np.mean(w['ic'])
        acc = np.mean(w['acc'])
        bsr = np.mean(w['bsr'])
        prof_days = np.mean([1 for n in w['net'] if n > 0])
        print(f"{week:>12} {ic:10.4f} {acc:11.4f} {bsr:11.4f} {prof_days:10.1%}")

    # Summary
    print("\n\nSUMMARY:")
    early_dates = [d for d in dates_sorted if d <= '20260305']
    late_dates = [d for d in dates_sorted if d > '20260305']

    if early_dates:
        early_acc = np.mean([np.mean(date_metrics[d].get('1s_accuracy', [0])) for d in early_dates])
        early_bsr = np.mean([np.mean(date_metrics[d].get('1s_bsr_accuracy', [0])) for d in early_dates])
        early_net = np.mean([np.mean(date_metrics[d].get('1s_net_ticks', [0])) for d in early_dates])
        print(f"  EARLY (<=Mar 5, {len(early_dates)} days): Acc={early_acc:.4f}, BSR_Acc={early_bsr:.4f}, Net={early_net:+.4f}")

    if late_dates:
        late_acc = np.mean([np.mean(date_metrics[d].get('1s_accuracy', [0])) for d in late_dates])
        late_bsr = np.mean([np.mean(date_metrics[d].get('1s_bsr_accuracy', [0])) for d in late_dates])
        late_net = np.mean([np.mean(date_metrics[d].get('1s_net_ticks', [0])) for d in late_dates])
        print(f"  LATE  (>Mar 5,  {len(late_dates)} days): Acc={late_acc:.4f}, BSR_Acc={late_bsr:.4f}, Net={late_net:+.4f}")

    if early_dates and late_dates:
        acc_drop = late_acc - early_acc
        print(f"  DECAY: {acc_drop:+.4f} accuracy ({acc_drop/early_acc*100:+.1f}%)")

    # Save
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nSaved to {out_dir}")
    print("DONE")


if __name__ == '__main__':
    main()
