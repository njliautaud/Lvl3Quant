"""
Imbalance-Conditioned Confluence Analyzer (HC #514)
=====================================================
Tests whether TRUE order flow imbalance, when combined with
directional model predictions as confluence, lifts accuracy
and P&L above the cost-clearing threshold.

Key idea: 51.6% directional accuracy alone doesn't clear costs.
But when models predict direction AND real-time imbalance confirms,
the signal should be much stronger.

Imbalance features from smart_v3 (25-feature set):
  - feat 7:  rolling_ofi_500 (order flow imbalance z-score)
  - feat 19: buy_sell_intensity_ratio (EWMA bid/ask event ratio)
  - feat 21: sweep_intensity (aggressive fill clustering)
  - feat 22: ofi_short_100 (short-term OFI)
  - feat 23: ofi_long_2000 (long-term OFI)
  - feat 24: ofi_acceleration (rate of change of OFI)

Also uses smooth pressure targets (NTPS, EOFI) when available.

Cost: 0.376 ticks only (HC #512).
"""

import argparse
import os
import sys
import json
import glob
import time
import numpy as np
from collections import defaultdict
from scipy.stats import spearmanr


# Imbalance feature indices in the 25-feature smart_v3 set
IMBALANCE_FEATURES = {
    'ofi_500': 7,
    'buy_sell_intensity': 19,
    'sweep_intensity': 21,
    'ofi_short_100': 22,
    'ofi_long_2000': 23,
    'ofi_acceleration': 24,
}

# Additional relevant features
EXTRA_FEATURES = {
    'cancel_side_asym': 6,
    'queue_replenishment': 15,
    'ofi_x_spread': 17,
}

COST_TICKS = 0.376  # HC #512


def load_predictions_for_date(pred_dir, date_str):
    """Load CNN-Mamba v2 predictions for a date."""
    f = os.path.join(pred_dir, f'{date_str}_predictions.npz')
    if not os.path.exists(f):
        return None
    d = np.load(f, allow_pickle=True)
    return {
        'predictions': d['predictions'],  # (N, 3) for 1s/5s/10s
        'labels': d['labels'],
    }


def load_events_for_date(events_dir, date_str):
    """Load raw event features for a date."""
    f = os.path.join(events_dir, f'{date_str}_mbo_events.npz')
    if not os.path.exists(f):
        return None
    d = np.load(f, allow_pickle=True)
    return {
        'events': d['events'],  # (N, 25)
        'labels_1s': d['labels_1s'],
        'labels_5s': d['labels_5s'],
        'labels_10s': d['labels_10s'],
        'timestamps': d['timestamps'],
    }


def load_pressure_for_date(pressure_dir, date_str):
    """Load smooth pressure targets for a date."""
    f = os.path.join(pressure_dir, f'{date_str}_pressure.npz')
    if not os.path.exists(f):
        return None
    d = np.load(f, allow_pickle=True)
    return {
        'ntps': d['ntps'],
        'eofi': d['eofi'],
        'pdi': d['pdi'],
        'tia': d['tia'],
    }


def compute_composite_imbalance(events, weights=None):
    """
    Compute a composite imbalance score from raw event features.
    Positive = buy pressure, negative = sell pressure.
    """
    if weights is None:
        weights = {
            'ofi_500': 0.3,
            'buy_sell_intensity': 0.25,
            'ofi_short_100': 0.2,
            'sweep_intensity': 0.15,
            'ofi_acceleration': 0.1,
        }

    composite = np.zeros(events.shape[0], dtype=np.float32)
    for fname, w in weights.items():
        idx = IMBALANCE_FEATURES[fname]
        composite += w * events[:, idx]

    return composite


def analyze_conditioned_signals(predictions, labels, imbalance, horizons=['1s', '5s', '10s'],
                                 confidence_tiers=[0.5, 0.8, 0.9, 0.95],
                                 imbalance_thresholds=[0.0, 0.3, 0.5, 0.7, 1.0]):
    """
    Core analysis: condition model predictions on imbalance alignment.

    For each combination of:
    - confidence tier (top X% of prediction magnitude)
    - imbalance threshold (min imbalance magnitude in predicted direction)
    - horizon (1s, 5s, 10s)

    Compute: accuracy, mean P&L, Sharpe, trade count.
    """
    results = []
    N = predictions.shape[0]

    for h_idx, h_name in enumerate(horizons):
        preds = predictions[:, h_idx]
        labs = labels[:, h_idx]

        for conf_pct in confidence_tiers:
            # Confidence filter: top X% by prediction magnitude
            abs_pred = np.abs(preds)
            conf_thresh = np.percentile(abs_pred, conf_pct * 100)
            conf_mask = abs_pred >= conf_thresh

            for imb_thresh in imbalance_thresholds:
                # Imbalance alignment: imbalance sign matches prediction sign
                # AND imbalance magnitude exceeds threshold
                pred_sign = np.sign(preds)
                imb_sign = np.sign(imbalance)
                alignment = pred_sign * imbalance  # positive when aligned

                imb_mask = alignment >= imb_thresh

                # Combined mask
                combined = conf_mask & imb_mask
                n_trades = combined.sum()

                if n_trades < 20:
                    continue

                # Direction of trades
                directions = np.sign(preds[combined])

                # Realized moves (labels are 0-1 scale: 0=down, 0.5=flat, 1=up)
                realized = (labs[combined] - 0.5) * 2  # scale to [-1, 1] direction

                # Gross ticks per trade (direction * realized)
                gross = directions * realized
                net = gross - COST_TICKS

                # Metrics
                accuracy = np.mean((directions > 0) == (realized > 0))
                mean_gross = np.mean(gross)
                mean_net = np.mean(net)
                std_net = np.std(net)
                sharpe = mean_net / std_net * np.sqrt(252) if std_net > 0 else 0
                pf = np.sum(net[net > 0]) / abs(np.sum(net[net < 0])) if np.any(net < 0) else float('inf')
                wr = np.mean(net > 0)

                results.append({
                    'horizon': h_name,
                    'confidence_pct': conf_pct,
                    'imbalance_thresh': imb_thresh,
                    'n_trades': int(n_trades),
                    'n_total': N,
                    'filter_rate': n_trades / N,
                    'accuracy': float(accuracy),
                    'mean_gross_ticks': float(mean_gross),
                    'mean_net_ticks': float(mean_net),
                    'sharpe': float(sharpe),
                    'profit_factor': float(pf),
                    'win_rate': float(wr),
                })

    return results


def analyze_multi_model_confluence(pred_dirs, events_dir, pressure_dir, dates):
    """
    Full analysis: multi-model predictions + imbalance conditioning.
    """
    all_results = defaultdict(list)
    per_date_results = {}

    print(f"\nAnalyzing {len(dates)} dates...")
    print(f"{'Date':>10} {'N_samp':>8} {'Imb_IC':>7} {'Base_Acc':>8} {'Cond_Acc':>8} {'Lift':>6}")
    print("-" * 55)

    for date in dates:
        # Load raw events
        ev = load_events_for_date(events_dir, date)
        if ev is None:
            continue

        # Load pressure
        pr = load_pressure_for_date(pressure_dir, date)

        # Load predictions from all available models
        model_preds = {}
        for mname, mdir in pred_dirs.items():
            p = load_predictions_for_date(mdir, date)
            if p is not None:
                model_preds[mname] = p

        if not model_preds:
            continue

        # Align sample counts
        n_events = ev['events'].shape[0]
        n_min = n_events
        for mp in model_preds.values():
            n_min = min(n_min, mp['predictions'].shape[0])
        if pr is not None:
            n_min = min(n_min, pr['ntps'].shape[0])

        events = ev['events'][:n_min]
        labels_1s = ev['labels_1s'][:n_min]
        labels_5s = ev['labels_5s'][:n_min]
        labels_10s = ev['labels_10s'][:n_min]
        labels = np.stack([labels_1s, labels_5s, labels_10s], axis=1)

        # Compute composite imbalance from raw features
        composite_imb = compute_composite_imbalance(events)

        # Also use EOFI if available (strongest pressure signal)
        if pr is not None:
            eofi = pr['eofi'][:n_min]
            ntps = pr['ntps'][:n_min]
        else:
            eofi = None
            ntps = None

        # Average predictions across models (simple ensemble)
        all_preds = np.stack([mp['predictions'][:n_min] for mp in model_preds.values()], axis=0)
        ensemble_pred = np.mean(all_preds, axis=0)  # (N, 3)

        # Model agreement: how many models agree on direction (per horizon)
        signs = np.sign(all_preds)  # (n_models, N, 3)
        agreement = np.abs(np.mean(signs, axis=0))  # 1.0 = all agree, 0.0 = split

        # Baseline accuracy (no imbalance conditioning)
        for h_idx, h_name in enumerate(['1s', '5s', '10s']):
            pred_dir = (ensemble_pred[:, h_idx] > 0).astype(int)
            true_dir = (labels[:, h_idx] > 0.5).astype(int)
            base_acc = np.mean(pred_dir == true_dir)
            all_results[f'base_acc_{h_name}'].append(base_acc)

        # Conditioned accuracy: only trade when imbalance confirms
        for h_idx, h_name in enumerate(['1s', '5s', '10s']):
            pred_sign = np.sign(ensemble_pred[:, h_idx])
            # Imbalance confirms: composite imbalance matches prediction direction
            imb_confirms = (pred_sign * composite_imb) > 0.3

            if imb_confirms.sum() > 100:
                pred_dir_c = (ensemble_pred[:, h_idx][imb_confirms] > 0).astype(int)
                true_dir_c = (labels[:, h_idx][imb_confirms] > 0.5).astype(int)
                cond_acc = np.mean(pred_dir_c == true_dir_c)
                all_results[f'cond_acc_{h_name}'].append(cond_acc)
                all_results[f'cond_n_{h_name}'].append(int(imb_confirms.sum()))
            else:
                cond_acc = float('nan')

            # EOFI conditioning (if available)
            if eofi is not None:
                eofi_confirms = (pred_sign * eofi) > 0.05
                if eofi_confirms.sum() > 100:
                    pred_dir_e = (ensemble_pred[:, h_idx][eofi_confirms] > 0).astype(int)
                    true_dir_e = (labels[:, h_idx][eofi_confirms] > 0.5).astype(int)
                    eofi_acc = np.mean(pred_dir_e == true_dir_e)
                    all_results[f'eofi_acc_{h_name}'].append(eofi_acc)

        # Per-date summary
        base_1s = all_results['base_acc_1s'][-1]
        cond_1s = all_results.get('cond_acc_1s', [float('nan')])[-1]
        lift = cond_1s - base_1s if not np.isnan(cond_1s) else 0

        # IC of composite imbalance vs future labels
        imb_ic, _ = spearmanr(composite_imb, labels_1s)

        print(f"{date:>10} {n_min:8d} {imb_ic:7.4f} {base_1s:8.4f} {cond_1s:8.4f} {lift:+6.3f}")

        # Full grid analysis for this date
        date_grid = analyze_conditioned_signals(
            ensemble_pred, labels, composite_imb,
            confidence_tiers=[0.5, 0.8, 0.9, 0.95],
            imbalance_thresholds=[0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
        )
        per_date_results[date] = date_grid

    return all_results, per_date_results


def print_summary(all_results, per_date_results):
    """Print aggregate results."""
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)

    horizons = ['1s', '5s', '10s']

    print(f"\n{'Horizon':>8} {'Base_Acc':>10} {'Cond_Acc':>10} {'EOFI_Acc':>10} {'Lift':>8} {'EOFI_Lift':>10}")
    print("-" * 60)
    for h in horizons:
        base = np.mean(all_results.get(f'base_acc_{h}', [0]))
        cond = np.mean(all_results.get(f'cond_acc_{h}', [0]))
        eofi = np.mean(all_results.get(f'eofi_acc_{h}', [0]))
        lift = cond - base
        eofi_lift = eofi - base
        print(f"{h:>8} {base:10.4f} {cond:10.4f} {eofi:10.4f} {lift:+8.4f} {eofi_lift:+10.4f}")

    # Best grid configs across all dates
    print("\n\nBEST CONFIGS (averaged across all dates):")
    print(f"{'Horizon':>8} {'Conf%':>6} {'Imb_Th':>7} {'Trades':>7} {'Accuracy':>9} {'Net_Ticks':>10} {'Sharpe':>8} {'PF':>6} {'WR':>6}")
    print("-" * 80)

    # Aggregate grid results
    grid_agg = defaultdict(lambda: defaultdict(list))
    for date, grid in per_date_results.items():
        for r in grid:
            key = (r['horizon'], r['confidence_pct'], r['imbalance_thresh'])
            for metric in ['accuracy', 'mean_net_ticks', 'sharpe', 'profit_factor', 'win_rate', 'n_trades']:
                grid_agg[key][metric].append(r[metric])

    # Sort by mean net ticks
    sorted_configs = sorted(grid_agg.items(),
                           key=lambda x: np.mean(x[1]['mean_net_ticks']),
                           reverse=True)

    for (h, conf, imb), metrics in sorted_configs[:20]:
        acc = np.mean(metrics['accuracy'])
        net = np.mean(metrics['mean_net_ticks'])
        sharpe = np.mean(metrics['sharpe'])
        pf = np.mean(metrics['profit_factor'])
        wr = np.mean(metrics['win_rate'])
        trades = np.mean(metrics['n_trades'])
        profitable = "✅" if net > 0 else "❌"
        print(f"{h:>8} {conf:6.0%} {imb:7.2f} {trades:7.0f} {acc:9.4f} {net:+10.4f} {sharpe:8.2f} {pf:6.2f} {wr:6.3f} {profitable}")


def main():
    parser = argparse.ArgumentParser(description='Imbalance-Conditioned Confluence Analyzer (HC #514)')
    parser.add_argument('--base-dir', default='/home/jupiter/Lvl3Quant')
    parser.add_argument('--output-dir', default='output/imbalance_confluence_v1')
    args = parser.parse_args()

    base = args.base_dir
    out_dir = os.path.join(base, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("IMBALANCE-CONDITIONED CONFLUENCE ANALYZER (HC #514)")
    print("=" * 70)
    print(f"Cost model: {COST_TICKS} ticks RT (HC #512)")
    print()

    # Prediction directories
    pred_dirs = {
        'cm_v2': os.path.join(base, 'output/cnn_mamba_v2_bulk_oot'),
        'cm_v2b': os.path.join(base, 'output/cnn_mamba_v2_bulk_oot_v2'),
        'ptst': os.path.join(base, 'output/patchtst_bulk_oot'),
    }

    events_dir = os.path.join(base, 'data/processed/mbo_events_smart_v3')
    pressure_dir = os.path.join(base, 'data/processed/smooth_pressure_targets')

    # Find overlapping dates across all sources
    pred_dates = {}
    for mname, mdir in pred_dirs.items():
        if os.path.isdir(mdir):
            files = glob.glob(os.path.join(mdir, '*_predictions.npz'))
            dates = set(os.path.basename(f).split('_')[0] for f in files)
            pred_dates[mname] = dates
            print(f"  {mname}: {len(dates)} dates")

    event_files = glob.glob(os.path.join(events_dir, '*_mbo_events.npz'))
    event_dates = set(os.path.basename(f).split('_')[0] for f in event_files)
    print(f"  events: {len(event_dates)} dates")

    pressure_files = glob.glob(os.path.join(pressure_dir, '*_pressure.npz'))
    pressure_dates = set(os.path.basename(f).split('_')[0] for f in pressure_files)
    print(f"  pressure: {len(pressure_dates)} dates")

    # Common dates: need events + at least 2 prediction models
    common = event_dates.copy()
    model_date_sets = list(pred_dates.values())
    if len(model_date_sets) >= 2:
        # Need at least 2 models
        pair_commons = []
        for i in range(len(model_date_sets)):
            for j in range(i+1, len(model_date_sets)):
                pair_commons.append(model_date_sets[i] & model_date_sets[j])
        # Union of all pair overlaps
        any_pair = set()
        for pc in pair_commons:
            any_pair |= pc
        common = common & any_pair
    else:
        common = common & model_date_sets[0]

    dates = sorted(common)
    print(f"\nOverlapping dates (events + ≥2 models): {len(dates)}")

    if len(dates) < 5:
        print("ERROR: Not enough overlapping dates")
        sys.exit(1)

    # Run analysis
    t0 = time.time()
    all_results, per_date_results = analyze_multi_model_confluence(
        pred_dirs, events_dir, pressure_dir, dates
    )
    elapsed = time.time() - t0

    # Print summary
    print_summary(all_results, per_date_results)

    # Save results
    summary = {
        'n_dates': len(dates),
        'dates': dates,
        'cost_ticks': COST_TICKS,
        'elapsed_s': elapsed,
        'aggregate': {k: float(np.mean(v)) for k, v in all_results.items()},
    }

    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # Save per-date grid
    with open(os.path.join(out_dir, 'grid_results.json'), 'w') as f:
        json.dump(per_date_results, f, indent=2, default=str)

    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Results saved to {out_dir}")
    print("\nDONE")


if __name__ == '__main__':
    main()
