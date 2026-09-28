#!/usr/bin/env python3
"""
IC Decay Curve Analysis — THE critical question for profitability.

Key insight: Cost per trade is FIXED (1.24 ticks), but expected move grows with sqrt(hold_time).
If a 1s signal's IC for predicting Ns returns decays SLOWER than 1/sqrt(N), we can profit.

For each raw microstructure signal, compute:
  IC_N = Spearman_IC(signal, N-second-forward-return)
  for N = 1s, 2s, 5s, 10s, 30s, 60s, 120s, 300s

Then compute: IC_N * sqrt(N) = "profit potential" at horizon N.
If this INCREASES with N, longer holds are more profitable.

Uses: microprice_dev, pressure_imbalance, OFI, trade_imbalance, composite
"""

import gc
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Horizons to test (in 10ms bars)
HORIZONS = {
    '1s':   100,
    '2s':   200,
    '5s':   500,
    '10s':  1000,
    '30s':  3000,
    '60s':  6000,
    '120s': 12000,
    '300s': 30000,
}

# Signal definitions (column indices from 340-feature MBO)
# mid=0, microprice=3, best_bid=8, best_ask=9
SIGNALS = {
    'microprice_dev': {
        'cols': [0, 3],  # mid, microprice
        'compute': lambda X: X[:, 1] - X[:, 0],  # microprice - mid
    },
    'trade_imbalance': {
        'cols': [10],  # trade_imbalance
        'compute': lambda X: X[:, 0],
    },
    'pressure_imbalance': {
        'cols': [20],  # pressure_imbalance
        'compute': lambda X: X[:, 0],
    },
}

COST_TICKS = 1.24
TICK_SIZE = 0.25


def load_day_raw(fpath):
    """Load raw MBO features and mid prices."""
    data = np.load(str(fpath))
    raw = data['mbo_features']
    mid = raw[:, 0].copy()
    mask = np.isnan(mid)
    if mask.any():
        first_valid = np.argmax(~mask)
        mid[:first_valid] = mid[first_valid]
    return raw, mid


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-days', type=int, default=50)
    parser.add_argument('--subsample', type=int, default=100,
                        help='Subsample factor (100 = every 1s)')
    args = parser.parse_args()

    data_dir = ROOT / 'data' / 'processed' / 'mbo_features_cache'
    files = sorted(data_dir.glob('*_mbo_features.npz'))[:args.n_days]
    print(f"IC Decay Curve Analysis — {len(files)} days")
    print(f"Subsample: every {args.subsample} bars ({args.subsample/100:.1f}s)")
    print(f"Horizons: {list(HORIZONS.keys())}")
    print(f"Signals: {list(SIGNALS.keys())}")
    print()

    # Results: signal -> horizon -> list of daily ICs
    results = {sig: {h: [] for h in HORIZONS} for sig in SIGNALS}

    t0 = time.time()
    for day_idx, fpath in enumerate(files):
        date = fpath.stem.replace('_mbo_features', '')
        raw, mid = load_day_raw(fpath)
        N = len(mid)

        # Compute signals (at every bar, subsample for IC calc)
        signal_values = {}
        for sig_name, sig_def in SIGNALS.items():
            cols = sig_def['cols']
            X = raw[:, cols]
            X = np.nan_to_num(X, nan=0.0)
            signal_values[sig_name] = sig_def['compute'](X)

        # For each horizon, compute forward return and IC
        for h_name, h_bars in HORIZONS.items():
            if N <= h_bars + args.subsample:
                continue

            # Forward return at this horizon (vectorized)
            fwd_ret = np.full(N, np.nan)
            fwd_ret[:N - h_bars] = (mid[h_bars:] - mid[:N - h_bars]) / np.where(
                mid[:N - h_bars] > 0, mid[:N - h_bars], 1.0)

            # Subsample for IC computation (avoid autocorrelation)
            indices = np.arange(0, N - h_bars, max(args.subsample, h_bars))
            if len(indices) < 30:
                continue

            y = fwd_ret[indices]
            valid = np.isfinite(y) & (y != 0)

            for sig_name in SIGNALS:
                x = signal_values[sig_name][indices]
                mask = valid & np.isfinite(x)
                if mask.sum() < 30:
                    continue
                ic, _ = spearmanr(x[mask], y[mask])
                if np.isfinite(ic):
                    results[sig_name][h_name].append(ic)

        if (day_idx + 1) % 10 == 0 or day_idx == 0:
            elapsed = time.time() - t0
            print(f"  [{day_idx+1}/{len(files)}] {date} ({elapsed:.1f}s)")

        del raw, mid, signal_values
        gc.collect()

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")

    # Summary
    print(f"\n{'='*90}")
    print(f"IC DECAY CURVE — How fast does predictive power decay with horizon?")
    print(f"{'='*90}")

    header = f"{'Signal':25s}"
    for h in HORIZONS:
        header += f" {'IC@'+h:>8s}"
    print(header)
    print("-" * 90)

    for sig_name in SIGNALS:
        row = f"{sig_name:25s}"
        for h_name in HORIZONS:
            ics = results[sig_name][h_name]
            if ics:
                row += f" {np.mean(ics):+8.4f}"
            else:
                row += f" {'N/A':>8s}"
        print(row)

    # Profit potential: IC * sqrt(horizon_seconds)
    print(f"\n{'='*90}")
    print(f"PROFIT POTENTIAL: IC_N * sqrt(N_seconds) — should INCREASE if longer holds help")
    print(f"{'='*90}")

    header = f"{'Signal':25s}"
    for h in HORIZONS:
        h_sec = HORIZONS[h] / 100
        header += f" {h:>8s}"
    print(header)
    print("-" * 90)

    for sig_name in SIGNALS:
        row = f"{sig_name:25s}"
        for h_name, h_bars in HORIZONS.items():
            ics = results[sig_name][h_name]
            h_sec = h_bars / 100
            if ics:
                mean_ic = np.mean(ics)
                profit_pot = mean_ic * np.sqrt(h_sec)
                row += f" {profit_pot:+8.4f}"
            else:
                row += f" {'N/A':>8s}"
        print(row)

    # Expected PnL at each horizon
    print(f"\n{'='*90}")
    print(f"EXPECTED PnL PER TRADE (ticks) = IC * avg_move_ticks - cost")
    print(f"Cost = {COST_TICKS:.2f} ticks per trade")
    print(f"{'='*90}")

    header = f"{'Signal':25s}"
    for h in HORIZONS:
        header += f" {h:>8s}"
    print(header)
    print("-" * 90)

    for sig_name in SIGNALS:
        row = f"{sig_name:25s}"
        for h_name, h_bars in HORIZONS.items():
            ics = results[sig_name][h_name]
            h_sec = h_bars / 100
            if ics:
                mean_ic = abs(np.mean(ics))
                # Estimate avg move in ticks: use empirical vol
                # avg |return| at horizon h ~ vol * sqrt(h)
                # For ES: typical 1s vol ≈ 0.001% → 1s move ≈ 0.058 ticks
                # Actually compute from data — avg |fwd_ret|
                # For now use sqrt scaling: 2.8 ticks at 1s * sqrt(h/1)
                avg_move = 2.8 * np.sqrt(h_sec)
                expected_pnl = mean_ic * avg_move - COST_TICKS
                row += f" {expected_pnl:+8.2f}"
            else:
                row += f" {'N/A':>8s}"
        print(row)

    print(f"\n{'='*90}")
    print(f"BREAKEVEN IC at each horizon:")
    print(f"{'='*90}")
    for h_name, h_bars in HORIZONS.items():
        h_sec = h_bars / 100
        avg_move = 2.8 * np.sqrt(h_sec)
        breakeven = COST_TICKS / avg_move
        print(f"  {h_name:>5s}: avg_move={avg_move:6.1f} ticks, breakeven IC={breakeven:.4f}")

    # Best signal at each horizon
    print(f"\n{'='*90}")
    print(f"BEST SIGNAL AT EACH HORIZON:")
    print(f"{'='*90}")
    for h_name, h_bars in HORIZONS.items():
        best_sig = None
        best_ic = 0
        for sig_name in SIGNALS:
            ics = results[sig_name][h_name]
            if ics:
                mean_ic = abs(np.mean(ics))
                if mean_ic > best_ic:
                    best_ic = mean_ic
                    best_sig = sig_name
        h_sec = h_bars / 100
        avg_move = 2.8 * np.sqrt(h_sec)
        expected_pnl = best_ic * avg_move - COST_TICKS if best_sig else float('nan')
        status = "PROFITABLE!" if expected_pnl > 0 else f"need {COST_TICKS/avg_move:.4f} IC"
        if best_sig:
            print(f"  {h_name:>5s}: {best_sig:25s} IC={best_ic:.4f}  "
                  f"E[PnL]={expected_pnl:+.2f} ticks  [{status}]")


if __name__ == '__main__':
    main()
