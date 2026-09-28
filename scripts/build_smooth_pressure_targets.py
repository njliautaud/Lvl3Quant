#!/usr/bin/env python3
"""
build_smooth_pressure_targets.py — Build smooth pressure targets for v4 multi-head model.

Reads smart_v3 event files and computes continuous, smooth pressure scores
that capture *sustained* directional pressure rather than noisy tick-by-tick returns.

Targets computed:
  1. NTPS (Net Taker Pressure Score)  — bounded [-1,+1], trade-only rolling buy/sell ratio + EMA
  2. EOFI (Exp Order Flow Imbalance)  — continuous, EMA of signed event flow
  3. PDI  (Pressure Duration Index)   — bounded [-1,+1], consecutive same-sign OFI streaks
  4. TIA  (Trade Intensity Asymmetry) — bounded [-1,+1], rolling buy/sell trade count ratio + EMA

For each, forward-looking labels at 1s/5s/10s/30s horizons are also computed
(what the model will predict = future pressure state).

Output: one .npz per day in data/processed/smooth_pressure_targets/
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

# ============================================================
# Constants from streaming_features_smart_v3.py
# ============================================================
# Event features (25 cols) — normalized at precompute time
COL_SIDE_ID = 2        # already mapped to -1/+1
COL_QTY_LOG = 4        # normalized: (raw_qty_log - 0.693) / 3.0

# event_type_raw: 0=add, 1=cancel, 2=modify, 3=trade, 4=fill
EVENT_TRADE = 3
EVENT_FILL  = 4

# Denormalize qty_log: raw = col * 3.0 + 0.693, then qty = exp(raw)
def denorm_qty(col_val: np.ndarray) -> np.ndarray:
    """Convert normalized qty_log back to approximate contract count."""
    raw_log = col_val * 3.0 + 0.693
    return np.clip(np.exp(raw_log), 1.0, 1e4)

EPS = 1e-8

# ============================================================
# Core computation
# ============================================================

def ema_1d(x: np.ndarray, alpha: float) -> np.ndarray:
    """Causal EMA using scipy lfilter (vectorized, no Python loop)."""
    from scipy.signal import lfilter
    # EMA: y[i] = alpha*x[i] + (1-alpha)*y[i-1]
    # Transfer function: b=[alpha], a=[1, -(1-alpha)]
    b = np.array([alpha], dtype=np.float64)
    a = np.array([1.0, -(1.0 - alpha)], dtype=np.float64)
    return lfilter(b, a, x.astype(np.float64)).astype(np.float32)


def rolling_sum(x: np.ndarray, w: int) -> np.ndarray:
    """Causal rolling sum over window w. Uses cumsum for speed."""
    cs = np.cumsum(x)
    out = np.empty_like(cs)
    out[:w] = cs[:w]
    out[w:] = cs[w:] - cs[:-w]
    return out


def compute_ntps(is_trade: np.ndarray, signed_vol: np.ndarray,
                 window: int = 100, ema_alpha: float = 0.01) -> np.ndarray:
    """Net Taker Pressure Score: rolling buy/sell volume ratio among trades, EMA-smoothed."""
    # Only count trades
    trade_signed = signed_vol * is_trade
    buy_vol  = rolling_sum(np.maximum(trade_signed, 0.0), window)
    sell_vol = rolling_sum(np.maximum(-trade_signed, 0.0), window)
    raw = (buy_vol - sell_vol) / (buy_vol + sell_vol + EPS)
    return ema_1d(raw, ema_alpha)


def compute_eofi(signed_flow: np.ndarray, ema_alpha: float = 0.02) -> np.ndarray:
    """Exponential Order Flow Imbalance: EMA of signed event flow."""
    return ema_1d(signed_flow, ema_alpha)


def compute_pdi(ofi_sign: np.ndarray) -> np.ndarray:
    """Pressure Duration Index: tanh of consecutive same-sign OFI streak length.
    Vectorized: detect sign-change boundaries, compute run lengths, apply tanh."""
    n = len(ofi_sign)
    # Find where sign changes
    sign_change = np.empty(n, dtype=bool)
    sign_change[0] = True
    sign_change[1:] = (ofi_sign[1:] != ofi_sign[:-1]) | (ofi_sign[1:] == 0)

    # Compute run lengths using cumsum of sign_change boundaries
    group_ids = np.cumsum(sign_change)
    # Position within each group
    positions = np.arange(n, dtype=np.float32)
    # Start position of each group
    group_starts = np.empty(n, dtype=np.float32)
    group_starts[sign_change] = positions[sign_change]
    # Forward-fill group starts
    idx = np.arange(n)
    mask = sign_change
    starts_idx = idx[mask]
    starts_vals = positions[mask]
    # Use searchsorted to map each position to its group start
    group_idx = np.searchsorted(starts_idx, idx, side='right') - 1
    group_starts = starts_vals[group_idx]
    run_length = positions - group_starts + 1  # 1-based

    pdi = np.tanh(run_length / 10.0) * ofi_sign
    pdi[ofi_sign == 0] = 0.0
    return pdi.astype(np.float32)


def compute_tia(is_trade: np.ndarray, side_sign: np.ndarray,
                window: int = 100, ema_alpha: float = 0.015) -> np.ndarray:
    """Trade Intensity Asymmetry: rolling buy/sell trade count ratio, EMA-smoothed."""
    is_buy_trade  = ((side_sign > 0) & (is_trade > 0.5)).astype(np.float32)
    is_sell_trade = ((side_sign < 0) & (is_trade > 0.5)).astype(np.float32)
    buy_count  = rolling_sum(is_buy_trade, window)
    sell_count = rolling_sum(is_sell_trade, window)
    raw = (buy_count - sell_count) / (buy_count + sell_count + EPS)
    return ema_1d(raw, ema_alpha)


def forward_shift_by_time(values: np.ndarray, timestamps: np.ndarray,
                          horizon_ns: int) -> np.ndarray:
    """For each event i, find the value at the first event with ts >= ts[i] + horizon_ns."""
    n = len(values)
    out = np.full(n, np.nan, dtype=np.float32)
    j = 0
    for i in range(n):
        target_ts = timestamps[i] + horizon_ns
        while j < n and timestamps[j] < target_ts:
            j += 1
        if j < n:
            out[i] = values[j]
        # Reset j to at least i+1 for next iteration? No — timestamps are sorted,
        # so j only advances. But we need to NOT reset j between iterations.
        # Actually j should start from current position since ts is sorted.
    # Fix: j needs to be local tracking per i. Use searchsorted for correctness.
    return out


def forward_shift_by_time_fast(values: np.ndarray, timestamps: np.ndarray,
                               horizon_ns: int) -> np.ndarray:
    """Vectorized forward shift using searchsorted."""
    target_ts = timestamps + horizon_ns
    indices = np.searchsorted(timestamps, target_ts, side='left')
    valid = indices < len(values)
    out = np.full(len(values), np.nan, dtype=np.float32)
    out[valid] = values[indices[valid]]
    return out


def autocorr(x: np.ndarray, lag: int) -> float:
    """Pearson autocorrelation at given lag, ignoring NaN."""
    if lag >= len(x):
        return 0.0
    a = x[:-lag] if lag > 0 else x
    b = x[lag:] if lag > 0 else x
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 100:
        return 0.0
    a, b = a[mask], b[mask]
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a**2).sum() * (b**2).sum())
    if denom < 1e-12:
        return 0.0
    return float((a * b).sum() / denom)


def rank_ic(pred: np.ndarray, actual: np.ndarray) -> float:
    """Spearman rank IC, ignoring NaN."""
    mask = np.isfinite(pred) & np.isfinite(actual)
    if mask.sum() < 100:
        return 0.0
    return float(spearmanr(pred[mask], actual[mask]).statistic)


# ============================================================
# Process one day-file
# ============================================================

def process_file(args_tuple):
    fpath, out_dir = args_tuple
    fname = os.path.basename(fpath)
    date_str = fname[:8]  # YYYYMMDD

    try:
        data = np.load(fpath)
        events         = data['events']           # (N, 25) float32
        event_type_raw = data['event_type_raw']    # (N,) int8
        timestamps     = data['timestamps']        # (N,) int64 nanoseconds

        n = len(events)
        if n < 1000:
            print(f"  {date_str}: only {n} events, skipping")
            return None

        # --- Extract raw signals ---
        side_sign = events[:, COL_SIDE_ID].astype(np.float32)  # already -1/+1
        qty = denorm_qty(events[:, COL_QTY_LOG]).astype(np.float32)

        is_trade = ((event_type_raw == EVENT_TRADE) |
                    (event_type_raw == EVENT_FILL)).astype(np.float32)

        signed_vol = qty * side_sign   # for all events
        # For OFI sign tracking, use the rolling OFI feature (col 7) or compute from signed_vol
        # Use a quick EMA of signed_vol as proxy for OFI sign
        quick_ofi = ema_1d(signed_vol, 0.05)
        ofi_sign = np.sign(quick_ofi).astype(np.float32)

        # --- Compute targets ---
        ntps = compute_ntps(is_trade, signed_vol, window=100, ema_alpha=0.01)
        eofi = compute_eofi(signed_vol, ema_alpha=0.02)
        pdi  = compute_pdi(ofi_sign)
        tia  = compute_tia(is_trade, side_sign, window=100, ema_alpha=0.015)

        # --- Forward-looking labels ---
        horizons_ns = {
            '1s':  1_000_000_000,
            '5s':  5_000_000_000,
            '10s': 10_000_000_000,
            '30s': 30_000_000_000,
        }

        labels = {}
        for name, target in [('ntps', ntps), ('eofi', eofi), ('pdi', pdi), ('tia', tia)]:
            for h_name, h_ns in horizons_ns.items():
                lbl = forward_shift_by_time_fast(target, timestamps, h_ns)
                labels[f'{name}_label_{h_name}'] = lbl.astype(np.float32)

        # --- Save ---
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'{date_str}_pressure.npz')
        save_dict = {
            'ntps': ntps.astype(np.float32),
            'eofi': eofi.astype(np.float32),
            'pdi':  pdi.astype(np.float32),
            'tia':  tia.astype(np.float32),
            'timestamps': timestamps,
        }
        save_dict.update(labels)
        np.savez_compressed(out_path, **save_dict)

        # --- Stats ---
        # Load raw labels for correlation
        raw_labels = {}
        for h in ['1s', '5s', '10s', '30s']:
            key = f'labels_{h}'
            if key in data:
                raw_labels[h] = data[key]

        stats = {'date': date_str, 'n_events': n,
                 'n_trades': int(is_trade.sum())}

        for name, arr in [('ntps', ntps), ('eofi', eofi), ('pdi', pdi), ('tia', tia)]:
            finite = arr[np.isfinite(arr)]
            stats[f'{name}_mean'] = float(finite.mean()) if len(finite) else 0
            stats[f'{name}_std']  = float(finite.std())  if len(finite) else 0
            stats[f'{name}_ac1']   = autocorr(arr, 1)
            stats[f'{name}_ac10']  = autocorr(arr, 10)
            stats[f'{name}_ac100'] = autocorr(arr, 100)

            # IC vs raw price labels
            for h in ['1s', '5s', '10s']:
                if h in raw_labels:
                    ic = rank_ic(arr, raw_labels[h])
                    stats[f'{name}_ic_vs_raw_{h}'] = ic

            # Autocorrelation of the FORWARD label
            lbl_key = f'{name}_label_5s'
            if lbl_key in labels:
                stats[f'{name}_label5s_ac1'] = autocorr(labels[lbl_key], 1)
                stats[f'{name}_label5s_ac100'] = autocorr(labels[lbl_key], 100)

        return stats

    except Exception as e:
        print(f"  ERROR processing {fname}: {e}")
        traceback.print_exc()
        return None


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Build smooth pressure targets for v4 multi-head')
    parser.add_argument('--workers', type=int, default=2,
                        help='Number of parallel workers (default 2)')
    parser.add_argument('--year-min', type=int, default=2025,
                        help='Minimum year to process (default 2025)')
    parser.add_argument('--limit', type=int, default=0,
                        help='Process only N files (0=all)')
    parser.add_argument('--input-dir', type=str,
                        default='/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3',
                        help='Input directory')
    parser.add_argument('--output-dir', type=str,
                        default='/home/jupiter/Lvl3Quant/data/processed/smooth_pressure_targets',
                        help='Output directory')
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    files = sorted(glob.glob(os.path.join(input_dir, '*_mbo_events.npz')))
    if args.year_min:
        files = [f for f in files if int(os.path.basename(f)[:4]) >= args.year_min]
    if args.limit > 0:
        files = files[:args.limit]

    print(f"Found {len(files)} files to process (workers={args.workers})")
    if not files:
        print("No files found.")
        return

    os.makedirs(output_dir, exist_ok=True)

    task_args = [(f, output_dir) for f in files]
    all_stats = []

    t0 = time.time()

    if args.workers <= 1:
        for i, ta in enumerate(task_args):
            print(f"[{i+1}/{len(files)}] Processing {os.path.basename(ta[0])}...")
            stats = process_file(ta)
            if stats:
                all_stats.append(stats)
    else:
        with Pool(args.workers) as pool:
            for i, stats in enumerate(pool.imap(process_file, task_args)):
                fname = os.path.basename(files[i])
                print(f"[{i+1}/{len(files)}] Done {fname}")
                if stats:
                    all_stats.append(stats)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Completed {len(all_stats)}/{len(files)} files in {elapsed:.1f}s")
    print(f"{'='*70}\n")

    if not all_stats:
        print("No stats collected.")
        return

    # --- Aggregate stats ---
    targets = ['ntps', 'eofi', 'pdi', 'tia']

    print(f"{'Target':<8} {'Mean':>8} {'Std':>8} {'AC-1':>7} {'AC-10':>7} {'AC-100':>7}")
    print('-' * 55)
    for t in targets:
        means = [s[f'{t}_mean'] for s in all_stats]
        stds  = [s[f'{t}_std']  for s in all_stats]
        ac1s  = [s[f'{t}_ac1']  for s in all_stats]
        ac10s = [s[f'{t}_ac10'] for s in all_stats]
        ac100 = [s[f'{t}_ac100'] for s in all_stats]
        print(f"{t:<8} {np.mean(means):>8.4f} {np.mean(stds):>8.4f} "
              f"{np.mean(ac1s):>7.4f} {np.mean(ac10s):>7.4f} {np.mean(ac100):>7.4f}")

    print(f"\n{'Target':<8} {'IC_1s':>8} {'IC_5s':>8} {'IC_10s':>8}")
    print('-' * 40)
    for t in targets:
        ics = {}
        for h in ['1s', '5s', '10s']:
            key = f'{t}_ic_vs_raw_{h}'
            vals = [s[key] for s in all_stats if key in s]
            ics[h] = np.mean(vals) if vals else 0.0
        print(f"{t:<8} {ics['1s']:>8.4f} {ics['5s']:>8.4f} {ics['10s']:>8.4f}")

    print(f"\n{'Target':<8} {'Lbl5s_AC1':>10} {'Lbl5s_AC100':>12}")
    print('-' * 35)
    for t in targets:
        ac1_key = f'{t}_label5s_ac1'
        ac100_key = f'{t}_label5s_ac100'
        ac1 = np.mean([s[ac1_key] for s in all_stats if ac1_key in s]) if any(ac1_key in s for s in all_stats) else 0
        ac100 = np.mean([s[ac100_key] for s in all_stats if ac100_key in s]) if any(ac100_key in s for s in all_stats) else 0
        print(f"{t:<8} {ac1:>10.4f} {ac100:>12.4f}")

    # Per-day trade counts
    total_events = sum(s['n_events'] for s in all_stats)
    total_trades = sum(s['n_trades'] for s in all_stats)
    print(f"\nTotal events: {total_events:,}  |  Total trades: {total_trades:,}  "
          f"({100*total_trades/total_events:.1f}%)")
    print(f"Output dir: {output_dir}")


if __name__ == '__main__':
    main()
