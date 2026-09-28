#!/usr/bin/env python3
"""
Signal Decay Analysis (#1)
==========================
Measures how our prediction signal persists over time.

Key questions answered:
1. Signal autocorrelation — how correlated is signal(t) with signal(t+Δ)?
2. IC at different forward windows — does IC persist beyond 10s?
3. Signal half-life — at what lag does IC drop to 50% of peak?
4. Conditional IC — does high-confidence signal persist longer?
5. Long vs Short signal decay — does one side persist more?

Uses CNN-Mamba v2 predictions on 39 OOT dates.
Multi-core for speed (HC #62).
"""

import os
import sys
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
import json
import time

LVL3 = Path('/home/jupiter/Lvl3Quant')
PRED_DIR = LVL3 / 'output' / 'decay_v4_comprehensive' / 'CNN-Mamba_v2'
MBO_DIR = LVL3 / 'data' / 'processed' / 'mbo_events'
OUT_DIR = LVL3 / 'execution' / 'results' / 'signal_decay'
OUT_DIR.mkdir(parents=True, exist_ok=True)

TICK_USD = 12.50


def load_date(date_str):
    """Load predictions + MBO data for a date."""
    pred_path = PRED_DIR / date_str / 'predictions.npz'
    mbo_path = MBO_DIR / f'{date_str}_mbo_events.npz'

    if not pred_path.exists() or not mbo_path.exists():
        return None

    pred = np.load(pred_path)
    mbo = np.load(mbo_path)

    preds = pred['preds']  # (N, 3) — 1s, 5s, 10s predictions
    valid_idx = pred['valid_indices']
    timestamps = mbo['timestamps']  # nanoseconds
    labels_1s = mbo['labels_1s']
    labels_5s = mbo['labels_5s']
    labels_10s = mbo['labels_10s']
    labels_30s = mbo['labels_30s'] if 'labels_30s' in mbo else None

    # Filter valid_idx to be within bounds
    max_idx = len(timestamps) - 1
    in_bounds = valid_idx <= max_idx
    valid_idx = valid_idx[in_bounds]
    preds = preds[in_bounds]

    # Get timestamps and labels at prediction points
    pred_ts = timestamps[valid_idx]
    pred_labels_1s = labels_1s[valid_idx]
    pred_labels_5s = labels_5s[valid_idx]
    pred_labels_10s = labels_10s[valid_idx]
    pred_labels_30s = labels_30s[valid_idx] if labels_30s is not None else None

    return {
        'date': date_str,
        'preds': preds,
        'timestamps': pred_ts,
        'labels_1s': pred_labels_1s,
        'labels_5s': pred_labels_5s,
        'labels_10s': pred_labels_10s,
        'labels_30s': pred_labels_30s,
        'valid_idx': valid_idx,
        'all_timestamps': timestamps,
        'all_labels_1s': labels_1s,
        'all_labels_5s': labels_5s,
        'all_labels_10s': labels_10s,
    }


def analyze_date(date_str):
    """Full signal decay analysis for one date."""
    data = load_date(date_str)
    if data is None:
        return None

    preds = data['preds']
    ts = data['timestamps']
    N = len(preds)

    if N < 50:
        return None

    # Composite signal: average of 1s, 5s, 10s predictions
    signal = preds.mean(axis=1)
    # Also per-horizon signals
    sig_1s = preds[:, 0]
    sig_5s = preds[:, 1]
    sig_10s = preds[:, 2]

    results = {'date': date_str, 'n_signals': N}

    # ── 1. Signal Autocorrelation ──────────────────────────────────
    # How correlated is signal(t) with signal(t+k)?
    # Use time-based lags since events are irregularly spaced
    lag_windows_ms = [100, 250, 500, 1000, 2000, 5000, 10000, 30000, 60000]
    autocorrs = {}

    for lag_ms in lag_windows_ms:
        lag_ns = lag_ms * 1_000_000
        corrs = []
        for i in range(0, N - 1, max(1, N // 2000)):  # Sample for speed
            # Find next signal after lag
            target_ts = ts[i] + lag_ns
            j = np.searchsorted(ts, target_ts)
            if j < N:
                corrs.append((signal[i], signal[j]))

        if len(corrs) > 30:
            corrs = np.array(corrs)
            r = np.corrcoef(corrs[:, 0], corrs[:, 1])[0, 1]
            autocorrs[f'{lag_ms}ms'] = float(r)
        else:
            autocorrs[f'{lag_ms}ms'] = None

    results['signal_autocorrelation'] = autocorrs

    # ── 2. IC at Different Horizons ────────────────────────────────
    # Direct IC measurement using available labels
    horizons = {
        '1s': data['labels_1s'],
        '5s': data['labels_5s'],
        '10s': data['labels_10s'],
    }
    if data['labels_30s'] is not None:
        horizons['30s'] = data['labels_30s']

    ic_by_horizon = {}
    for hz_name, labels in horizons.items():
        valid = ~np.isnan(labels) & ~np.isnan(signal)
        if valid.sum() > 30:
            ic = np.corrcoef(signal[valid], labels[valid])[0, 1]
            ic_by_horizon[hz_name] = float(ic)

    results['ic_by_horizon'] = ic_by_horizon

    # ── 3. Per-Horizon IC (each prediction head vs its target) ─────
    per_head_ic = {}
    for idx, hz_name in enumerate(['1s', '5s', '10s']):
        labels = horizons[hz_name]
        sig = preds[:, idx]
        valid = ~np.isnan(labels) & ~np.isnan(sig)
        if valid.sum() > 30:
            ic = np.corrcoef(sig[valid], labels[valid])[0, 1]
            per_head_ic[hz_name] = float(ic)
    results['per_head_ic'] = per_head_ic

    # ── 4. Signal Half-Life ────────────────────────────────────────
    # Estimate from autocorrelation: at what lag does corr drop to 50%?
    peak_corr = autocorrs.get('100ms', 1.0)
    if peak_corr and peak_corr > 0:
        half_target = peak_corr * 0.5
        half_life_ms = None
        for lag_ms in lag_windows_ms:
            key = f'{lag_ms}ms'
            if autocorrs.get(key) is not None and autocorrs[key] < half_target:
                half_life_ms = lag_ms
                break
        results['signal_half_life_ms'] = half_life_ms
    else:
        results['signal_half_life_ms'] = None

    # ── 5. Confidence-Conditional IC ───────────────────────────────
    # Does high confidence signal have BETTER IC?
    abs_signal = np.abs(signal)
    percentiles = [50, 75, 90, 95, 99]
    thresholds = np.percentile(abs_signal, percentiles)

    conditional_ic = {}
    for pct, thr in zip(percentiles, thresholds):
        mask = abs_signal >= thr
        n_above = mask.sum()
        if n_above > 20:
            for hz_name, labels in horizons.items():
                valid = mask & ~np.isnan(labels) & ~np.isnan(signal)
                if valid.sum() > 15:
                    ic = np.corrcoef(signal[valid], labels[valid])[0, 1]
                    conditional_ic[f'top_{100-pct}pct_{hz_name}'] = {
                        'ic': float(ic),
                        'n': int(valid.sum()),
                        'threshold': float(thr),
                    }

    results['conditional_ic'] = conditional_ic

    # ── 6. LONG vs SHORT Decay ─────────────────────────────────────
    long_mask = signal > 0
    short_mask = signal < 0

    for side, mask, side_name in [(long_mask, long_mask, 'long'), (short_mask, short_mask, 'short')]:
        side_ic = {}
        for hz_name, labels in horizons.items():
            valid = mask & ~np.isnan(labels) & ~np.isnan(signal)
            if valid.sum() > 20:
                ic = np.corrcoef(signal[valid], labels[valid])[0, 1]
                side_ic[hz_name] = float(ic)
        results[f'{side_name}_ic'] = side_ic
        results[f'{side_name}_n'] = int(mask.sum())

    # ── 7. Expected Move in Ticks by Confidence Bucket ──────────────
    # This directly answers: "how many ticks does the signal predict?"
    buckets = {
        'all': np.ones(N, dtype=bool),
        'top_50pct': abs_signal >= np.percentile(abs_signal, 50),
        'top_20pct': abs_signal >= np.percentile(abs_signal, 80),
        'top_10pct': abs_signal >= np.percentile(abs_signal, 90),
        'top_5pct': abs_signal >= np.percentile(abs_signal, 95),
        'top_1pct': abs_signal >= np.percentile(abs_signal, 99),
    }

    move_analysis = {}
    for bucket_name, bucket_mask in buckets.items():
        for side_name, side_fn in [('long', lambda s: s > 0), ('short', lambda s: s < 0), ('both', lambda s: np.ones_like(s, dtype=bool))]:
            combined_mask = bucket_mask & side_fn(signal)
            n = combined_mask.sum()
            if n < 5:
                continue

            for hz_name, labels in horizons.items():
                valid = combined_mask & ~np.isnan(labels)
                if valid.sum() < 5:
                    continue

                directed_labels = labels[valid].copy()
                # For shorts, we profit when price goes DOWN
                if side_name == 'short':
                    directed_labels = -directed_labels
                elif side_name == 'both':
                    # Direction-adjusted: positive prediction → label as-is, negative → flip
                    signs = np.sign(signal[valid])
                    directed_labels = directed_labels * signs

                avg_move = float(np.mean(directed_labels))
                median_move = float(np.median(directed_labels))
                win_rate = float(np.mean(directed_labels > 0))
                avg_winner = float(np.mean(directed_labels[directed_labels > 0])) if (directed_labels > 0).any() else 0
                avg_loser = float(np.mean(directed_labels[directed_labels < 0])) if (directed_labels < 0).any() else 0

                key = f'{bucket_name}_{side_name}_{hz_name}'
                move_analysis[key] = {
                    'n': int(valid.sum()),
                    'avg_move_ticks': avg_move,
                    'median_move_ticks': median_move,
                    'win_rate': win_rate,
                    'avg_winner_ticks': avg_winner,
                    'avg_loser_ticks': avg_loser,
                    'net_after_passive_commission': avg_move - 0.376,  # passive limit
                    'net_after_market_order': avg_move - 0.376,  # HC #231(A): commission only
                }

    results['move_by_confidence'] = move_analysis

    # ── 8. Signal Persistence After Entry ──────────────────────────
    # After a high-confidence signal fires, how many ms does it stay high?
    top_10_mask = abs_signal >= np.percentile(abs_signal, 90)
    top_10_indices = np.where(top_10_mask)[0]

    persistence_windows = [500, 1000, 2000, 5000, 10000]
    persistence = {}

    for window_ms in persistence_windows:
        window_ns = window_ms * 1_000_000
        still_strong = []

        for i in top_10_indices[::max(1, len(top_10_indices) // 500)]:
            target_ts = ts[i] + window_ns
            j = np.searchsorted(ts, target_ts)
            if j < N:
                # Is signal still in same direction and still strong?
                same_dir = np.sign(signal[i]) == np.sign(signal[j])
                still_top = abs_signal[j] >= np.percentile(abs_signal, 75)
                still_strong.append(same_dir and still_top)

        if still_strong:
            persistence[f'{window_ms}ms'] = float(np.mean(still_strong))

    results['signal_persistence_rate'] = persistence

    return results


def main():
    print("=" * 70)
    print("SIGNAL DECAY ANALYSIS — CNN-Mamba v2")
    print("=" * 70)

    # Find all dates with predictions
    dates = sorted([d.name for d in PRED_DIR.iterdir() if d.is_dir()])
    print(f"\nFound {len(dates)} OOT dates with predictions")

    # Run in parallel
    n_workers = min(14, os.cpu_count() or 4)
    print(f"Running on {n_workers} cores...")

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(analyze_date, dates))

    results = [r for r in results if r is not None]
    elapsed = time.time() - t0
    print(f"\nProcessed {len(results)} dates in {elapsed:.1f}s")

    # ── Aggregate across all dates ─────────────────────────────────
    print("\n" + "=" * 70)
    print("AGGREGATED RESULTS")
    print("=" * 70)

    # 1. Average IC by horizon
    print("\n📊 IC by Horizon (averaged across dates):")
    for hz in ['1s', '5s', '10s', '30s']:
        ics = [r['ic_by_horizon'].get(hz) for r in results if r['ic_by_horizon'].get(hz) is not None]
        if ics:
            print(f"  {hz:>4s}: IC = {np.mean(ics):.4f} ± {np.std(ics):.4f} (n={len(ics)} dates)")

    # 2. Per-head IC
    print("\n📊 Per-Head IC (prediction head vs its own target):")
    for hz in ['1s', '5s', '10s']:
        ics = [r['per_head_ic'].get(hz) for r in results if r['per_head_ic'].get(hz) is not None]
        if ics:
            print(f"  {hz:>4s} head: IC = {np.mean(ics):.4f} ± {np.std(ics):.4f}")

    # 3. Signal autocorrelation
    print("\n📊 Signal Autocorrelation (avg across dates):")
    lag_keys = [f'{ms}ms' for ms in [100, 250, 500, 1000, 2000, 5000, 10000, 30000, 60000]]
    for key in lag_keys:
        vals = [r['signal_autocorrelation'].get(key) for r in results if r['signal_autocorrelation'].get(key) is not None]
        if vals:
            print(f"  {key:>8s}: {np.mean(vals):.4f}")

    # 4. Signal half-life
    half_lives = [r['signal_half_life_ms'] for r in results if r['signal_half_life_ms'] is not None]
    if half_lives:
        print(f"\n📊 Signal Half-Life: median = {np.median(half_lives):.0f}ms, mean = {np.mean(half_lives):.0f}ms")

    # 5. LONG vs SHORT IC
    print("\n📊 Long vs Short IC:")
    for side in ['long', 'short']:
        for hz in ['1s', '5s', '10s']:
            ics = [r[f'{side}_ic'].get(hz) for r in results if r[f'{side}_ic'].get(hz) is not None]
            if ics:
                print(f"  {side:>5s} {hz:>4s}: IC = {np.mean(ics):.4f} ± {np.std(ics):.4f}")

    # 6. Expected Move by Confidence (THE KEY TABLE)
    print("\n" + "=" * 70)
    print("💰 EXPECTED MOVE IN TICKS BY CONFIDENCE BUCKET")
    print("   (Direction-adjusted: positive = trade would have profited)")
    print("   Commission: 0.376 ticks (HC #231(A): same for passive + market — no spread cost)")
    print("=" * 70)

    for side in ['long', 'short', 'both']:
        print(f"\n  === {side.upper()} TRADES ===")
        print(f"  {'Bucket':>12s} | {'Hz':>4s} | {'N':>6s} | {'AvgMove':>8s} | {'WR':>6s} | {'NetPassive':>10s} | {'NetMarket':>10s}")
        print(f"  {'-'*12}-+-{'-'*4}-+-{'-'*6}-+-{'-'*8}-+-{'-'*6}-+-{'-'*10}-+-{'-'*10}")

        for bucket in ['all', 'top_50pct', 'top_20pct', 'top_10pct', 'top_5pct', 'top_1pct']:
            for hz in ['1s', '5s', '10s']:
                key = f'{bucket}_{side}_{hz}'
                moves = []
                wrs = []
                for r in results:
                    m = r['move_by_confidence'].get(key)
                    if m:
                        moves.append(m['avg_move_ticks'])
                        wrs.append(m['win_rate'])

                if moves:
                    avg_move = np.mean(moves)
                    avg_wr = np.mean(wrs)
                    net_passive = avg_move - 0.376
                    net_market = avg_move - 0.376  # HC #231(A): commission only

                    passive_flag = "✅" if net_passive > 0 else "❌"
                    market_flag = "✅" if net_market > 0 else "❌"

                    print(f"  {bucket:>12s} | {hz:>4s} | {len(moves):>6d} | {avg_move:>+8.3f} | {avg_wr:>5.1%} | {net_passive:>+8.3f} {passive_flag} | {net_market:>+8.3f} {market_flag}")

    # 7. Signal Persistence
    print("\n📊 Signal Persistence (% of top-10% signals still strong after delay):")
    for window_ms in [500, 1000, 2000, 5000, 10000]:
        key = f'{window_ms}ms'
        rates = [r['signal_persistence_rate'].get(key) for r in results if r['signal_persistence_rate'].get(key) is not None]
        if rates:
            print(f"  After {window_ms:>6d}ms: {np.mean(rates):.1%} still strong → {'CHASE viable' if np.mean(rates) > 0.5 else 'Signal decayed'}")

    # 8. Conditional IC at high confidence
    print("\n📊 IC at High Confidence Levels:")
    for pct in ['top_50pct', 'top_25pct', 'top_10pct', 'top_5pct', 'top_1pct']:
        for hz in ['1s', '5s', '10s']:
            key = f'{pct}_{hz}'
            ics = [r['conditional_ic'].get(key, {}).get('ic') for r in results if r['conditional_ic'].get(key, {}).get('ic') is not None]
            if ics:
                print(f"  {pct:>12s} {hz:>4s}: IC = {np.mean(ics):.4f} ± {np.std(ics):.4f}")

    # ── Save full results ──────────────────────────────────────────
    out_path = OUT_DIR / 'signal_decay_full_results.json'

    # Make JSON serializable
    serializable = []
    for r in results:
        sr = {}
        for k, v in r.items():
            if isinstance(v, dict):
                sr[k] = {kk: (float(vv) if isinstance(vv, (np.floating, float)) else vv) for kk, vv in v.items()}
            elif isinstance(v, (np.floating, float)):
                sr[k] = float(v)
            elif isinstance(v, (np.integer, int)):
                sr[k] = int(v)
            else:
                sr[k] = v
        serializable.append(sr)

    with open(out_path, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\n💾 Full results saved to {out_path}")

    print("\n" + "=" * 70)
    print("DONE — Signal Decay Analysis Complete")
    print("=" * 70)


if __name__ == '__main__':
    main()
