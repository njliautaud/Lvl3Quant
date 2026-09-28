"""
Adversarial Validation v1 — Robustness tests for CNN-Mamba v2 top-2% short signals.

Tests whether the +0.32 ticks/trade edge is real or a statistical artifact.
Uses predictions+labels already paired in each OOT prediction file.

Predictions col 0 = return_1s prediction
Labels col 0 = return_1s actual (in ticks)
"""

import numpy as np
import json
import os
from pathlib import Path
from collections import defaultdict

# --- Config ---
PRED_DIR = Path("/home/jupiter/Lvl3Quant/output/cnn_mamba_v2_bulk_oot_v2")
OUT_DIR = Path("/home/jupiter/Lvl3Quant/output/adversarial_validation_v1")
COMMISSION_RT = 0.376  # ticks, passive entry
SHORT_PERCENTILE = 2   # bottom 2% of return_1s predictions = short signals
N_SHUFFLE = 100
N_BOOTSTRAP = 1000
NOISE_LEVELS = [0.10, 0.20, 0.50]  # fraction of prediction std
np.random.seed(42)


def load_all_dates():
    """Load predictions and labels for all OOT dates."""
    dates = {}
    for f in sorted(PRED_DIR.glob("*_predictions.npz")):
        date_str = f.name.split("_")[0]
        d = np.load(f)
        preds_1s = d['predictions'][:, 0]   # return_1s predictions
        labels_1s = d['labels'][:, 0]       # return_1s actual (ticks)
        if len(preds_1s) < 100:
            continue
            # Filter out NaN labels
        valid = ~np.isnan(labels_1s) & ~np.isnan(preds_1s)
        preds_1s = preds_1s[valid]
        labels_1s = labels_1s[valid]
        if len(preds_1s) < 100:
            continue
        dates[date_str] = {'preds': preds_1s, 'labels': labels_1s}
    return dates


def select_short_signals(preds, labels, percentile=SHORT_PERCENTILE):
    """Select bottom N percentile of predictions (most bearish = short signals)."""
    threshold = np.percentile(preds, percentile)
    mask = preds <= threshold
    return labels[mask]


def compute_gross_pnl_per_signal(selected_labels):
    """
    For short signals: P&L = -return_1s (we're short, so negative return = profit).
    Labels are already in ticks. Subtract commission.
    """
    gross = -selected_labels  # short: profit when price drops
    net = gross - COMMISSION_RT
    return net


def daily_pnl(dates):
    """Compute mean net P&L per signal for each date, and overall."""
    daily = {}
    all_net = []
    for date_str, d in dates.items():
        sel = select_short_signals(d['preds'], d['labels'])
        if len(sel) == 0:
            continue
        net = compute_gross_pnl_per_signal(sel)
        daily[date_str] = {'mean_net': float(np.mean(net)), 'n_signals': len(sel),
                           'net_values': net}
        all_net.append(net)
    all_net = np.concatenate(all_net) if all_net else np.array([])
    return daily, all_net


# ============================================================
# TEST 1: Shuffle test (permutation test)
# ============================================================
def shuffle_test(dates, n_iter=N_SHUFFLE):
    """Shuffle prediction-label pairing within each date. Compare real vs shuffled P&L."""
    _, real_all = daily_pnl(dates)
    real_mean = np.mean(real_all)

    shuffled_means = []
    for i in range(n_iter):
        all_shuffled_net = []
        for date_str, d in dates.items():
            preds = d['preds'].copy()
            labels = d['labels'].copy()
            # Shuffle labels relative to predictions
            np.random.shuffle(labels)
            sel = select_short_signals(preds, labels)
            if len(sel) == 0:
                continue
            net = compute_gross_pnl_per_signal(sel)
            all_shuffled_net.append(net)
        if all_shuffled_net:
            shuffled_means.append(float(np.mean(np.concatenate(all_shuffled_net))))

    shuffled_means = np.array(shuffled_means)
    pct_rank = np.mean(real_mean > shuffled_means) * 100
    passed = pct_rank >= 95.0

    return {
        'test': 'shuffle_test',
        'real_mean_pnl': round(real_mean, 4),
        'shuffled_mean': round(float(np.mean(shuffled_means)), 4),
        'shuffled_std': round(float(np.std(shuffled_means)), 4),
        'percentile_rank': round(pct_rank, 1),
        'pass': passed,
        'criterion': 'real P&L > 95th percentile of shuffled'
    }


# ============================================================
# TEST 2: Bootstrap confidence interval on dates
# ============================================================
def bootstrap_ci(dates, n_iter=N_BOOTSTRAP):
    """Resample dates with replacement, compute mean daily P&L each time."""
    daily_d, _ = daily_pnl(dates)
    daily_means = np.array([v['mean_net'] for v in daily_d.values()])
    n_dates = len(daily_means)

    boot_means = []
    for _ in range(n_iter):
        idx = np.random.choice(n_dates, size=n_dates, replace=True)
        boot_means.append(float(np.mean(daily_means[idx])))

    boot_means = np.array(boot_means)
    ci_5 = float(np.percentile(boot_means, 5))
    ci_95 = float(np.percentile(boot_means, 95))
    passed = ci_5 > 0

    return {
        'test': 'bootstrap_ci',
        'mean_daily_pnl': round(float(np.mean(daily_means)), 4),
        'ci_5': round(ci_5, 4),
        'ci_95': round(ci_95, 4),
        'n_dates': n_dates,
        'pass': passed,
        'criterion': '5th percentile of bootstrap > 0'
    }


# ============================================================
# TEST 3: Noise injection
# ============================================================
def noise_injection(dates, noise_levels=NOISE_LEVELS):
    """Add Gaussian noise to predictions, re-rank, compute P&L."""
    _, real_all = daily_pnl(dates)
    real_mean = np.mean(real_all)

    results_by_noise = {}
    for sigma_frac in noise_levels:
        all_noisy_net = []
        for date_str, d in dates.items():
            preds = d['preds'].copy()
            labels = d['labels'].copy()
            pred_std = np.std(preds)
            noise = np.random.normal(0, sigma_frac * pred_std, size=len(preds))
            noisy_preds = preds + noise
            sel = select_short_signals(noisy_preds, labels)
            if len(sel) == 0:
                continue
            net = compute_gross_pnl_per_signal(sel)
            all_noisy_net.append(net)
        if all_noisy_net:
            noisy_mean = float(np.mean(np.concatenate(all_noisy_net)))
        else:
            noisy_mean = 0.0
        decay_pct = (1 - noisy_mean / real_mean) * 100 if real_mean != 0 else 0
        results_by_noise[f'sigma_{int(sigma_frac*100)}pct'] = {
            'noisy_mean_pnl': round(noisy_mean, 4),
            'decay_pct': round(decay_pct, 1),
            'still_profitable': noisy_mean > 0
        }

    return {
        'test': 'noise_injection',
        'real_mean_pnl': round(real_mean, 4),
        'noise_results': results_by_noise,
        'pass': results_by_noise.get('sigma_20pct', {}).get('still_profitable', False),
        'criterion': 'still profitable at 20% noise'
    }


# ============================================================
# TEST 4: Half-sample stability
# ============================================================
def half_sample_stability(dates):
    """Split OOT dates in half chronologically. Both halves profitable?"""
    sorted_dates = sorted(dates.keys())
    mid = len(sorted_dates) // 2
    first_half = {d: dates[d] for d in sorted_dates[:mid]}
    second_half = {d: dates[d] for d in sorted_dates[mid:]}

    _, first_net = daily_pnl(first_half)
    _, second_net = daily_pnl(second_half)

    first_mean = float(np.mean(first_net)) if len(first_net) > 0 else 0
    second_mean = float(np.mean(second_net)) if len(second_net) > 0 else 0

    passed = first_mean > 0 and second_mean > 0

    return {
        'test': 'half_sample_stability',
        'first_half_dates': f'{sorted_dates[0]} to {sorted_dates[mid-1]}',
        'second_half_dates': f'{sorted_dates[mid]} to {sorted_dates[-1]}',
        'first_half_mean_pnl': round(first_mean, 4),
        'second_half_mean_pnl': round(second_mean, 4),
        'first_half_n_dates': mid,
        'second_half_n_dates': len(sorted_dates) - mid,
        'pass': passed,
        'criterion': 'both halves profitable'
    }


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("Loading data...")
    dates = load_all_dates()
    print(f"Loaded {len(dates)} OOT dates")

    # Baseline stats
    daily_d, all_net = daily_pnl(dates)
    print(f"\nBaseline: {len(all_net)} total short signals across {len(daily_d)} dates")
    print(f"Mean net P&L per signal: {np.mean(all_net):.4f} ticks")
    print(f"Median net P&L per signal: {np.median(all_net):.4f} ticks")
    print(f"Win rate: {np.mean(all_net > 0)*100:.1f}%")
    print(f"Mean daily mean P&L: {np.mean([v['mean_net'] for v in daily_d.values()]):.4f} ticks")
    print()

    # Run tests
    print("=" * 60)
    print("TEST 1: Shuffle Test (100 iterations)")
    print("=" * 60)
    r1 = shuffle_test(dates)
    print(f"  Real mean P&L:     {r1['real_mean_pnl']:.4f} ticks")
    print(f"  Shuffled mean:     {r1['shuffled_mean']:.4f} ticks")
    print(f"  Shuffled std:      {r1['shuffled_std']:.4f} ticks")
    print(f"  Percentile rank:   {r1['percentile_rank']}%")
    print(f"  RESULT: {'PASS' if r1['pass'] else 'FAIL'}")
    print()

    print("=" * 60)
    print("TEST 2: Bootstrap CI (1000 iterations, date-level)")
    print("=" * 60)
    r2 = bootstrap_ci(dates)
    print(f"  Mean daily P&L:    {r2['mean_daily_pnl']:.4f} ticks")
    print(f"  90% CI:            [{r2['ci_5']:.4f}, {r2['ci_95']:.4f}]")
    print(f"  N dates:           {r2['n_dates']}")
    print(f"  RESULT: {'PASS' if r2['pass'] else 'FAIL'}")
    print()

    print("=" * 60)
    print("TEST 3: Noise Injection")
    print("=" * 60)
    r3 = noise_injection(dates)
    print(f"  Real mean P&L:     {r3['real_mean_pnl']:.4f} ticks")
    for k, v in r3['noise_results'].items():
        print(f"  {k}: mean={v['noisy_mean_pnl']:.4f}, decay={v['decay_pct']:.1f}%, profitable={v['still_profitable']}")
    print(f"  RESULT: {'PASS' if r3['pass'] else 'FAIL'}")
    print()

    print("=" * 60)
    print("TEST 4: Half-Sample Stability")
    print("=" * 60)
    r4 = half_sample_stability(dates)
    print(f"  First half ({r4['first_half_dates']}):  {r4['first_half_mean_pnl']:.4f} ticks ({r4['first_half_n_dates']} dates)")
    print(f"  Second half ({r4['second_half_dates']}): {r4['second_half_mean_pnl']:.4f} ticks ({r4['second_half_n_dates']} dates)")
    print(f"  RESULT: {'PASS' if r4['pass'] else 'FAIL'}")
    print()

    # Summary
    results = {
        'baseline': {
            'total_signals': len(all_net),
            'n_dates': len(daily_d),
            'mean_net_pnl_per_signal': round(float(np.mean(all_net)), 4),
            'median_net_pnl': round(float(np.median(all_net)), 4),
            'win_rate': round(float(np.mean(all_net > 0)) * 100, 1),
            'commission_rt': COMMISSION_RT,
            'short_percentile': SHORT_PERCENTILE,
        },
        'tests': [r1, r2, r3, r4],
        'summary': {
            'tests_passed': sum([r1['pass'], r2['pass'], r3['pass'], r4['pass']]),
            'tests_total': 4,
            'verdict': 'ROBUST' if all([r1['pass'], r2['pass'], r3['pass'], r4['pass']]) else 'CONCERNS'
        }
    }

    # Convert numpy bools to Python bools for JSON
    def convert(obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj

    results = convert(results)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print("=" * 60)
    print(f"OVERALL: {results['summary']['tests_passed']}/{results['summary']['tests_total']} tests passed — {results['summary']['verdict']}")
    print(f"Results saved to {OUT_DIR / 'results.json'}")
    print("=" * 60)
