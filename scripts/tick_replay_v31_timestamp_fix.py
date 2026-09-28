#!/usr/bin/env python3
"""
Tick Replay v31 — TIMESTAMP-FIXED Long-Only Regime-Stratified Backtest
======================================================================
Fixes the CATASTROPHIC stride mismatch found in deep audit (HC #722):
- Old: PRED_STRIDE=250 on raw MBO events (WRONG — up to 5hr temporal drift)
- New: Map predictions to raw MBO events by TIMESTAMP matching (CORRECT)

Uses prediction files from output/v4_tick_replay_preds_ts/ which include
pred_timestamps_ns — the actual nanosecond timestamps when each prediction
was generated (from processed events at STRIDE=500).

Same simulation engine as v29 (passive fill sim, BBO reconstruction).
Tests top 6 configs from prior runs that passed permutation.
"""

import numpy as np
import os
import sys
import json
import time
from pathlib import Path
from collections import defaultdict

PRED_DIR = '/home/jupiter/Lvl3Quant/output/v4_tick_replay_preds_ts'
MBO_DIR = '/home/jupiter/Lvl3Quant/data/preprocessed_mbo'
OUTPUT = Path('/home/jupiter/Lvl3Quant/output/tick_replay_v31_ts_fixed')
OUTPUT.mkdir(parents=True, exist_ok=True)

N_PERMS = 200
TICK = 0.25

# Top configs from v28/v29 that previously passed permutation
CONFIGS = [
    # Champion configs
    {'q': 0.10, 'tp': 16, 'sl': 4, 'hold': 30, 'cancel': 10, 'label': 'q10_tp16_sl4'},
    {'q': 0.10, 'tp': 16, 'sl': 6, 'hold': 30, 'cancel': 10, 'label': 'q10_tp16_sl6'},
    {'q': 0.10, 'tp': 16, 'sl': 8, 'hold': 30, 'cancel': 10, 'label': 'q10_tp16_sl8'},
    {'q': 0.10, 'tp': 12, 'sl': 4, 'hold': 30, 'cancel': 10, 'label': 'q10_tp12_sl4'},
    # Horizon-aligned: shorter holds closer to 1s prediction
    {'q': 0.10, 'tp': 4, 'sl': 2, 'hold': 3, 'cancel': 3, 'label': 'q10_tp4_sl2_h3'},
    {'q': 0.10, 'tp': 6, 'sl': 3, 'hold': 5, 'cancel': 5, 'label': 'q10_tp6_sl3_h5'},
    {'q': 0.10, 'tp': 8, 'sl': 4, 'hold': 10, 'cancel': 8, 'label': 'q10_tp8_sl4_h10'},
    # Ultra-short for horizon test
    {'q': 0.10, 'tp': 2, 'sl': 1, 'hold': 2, 'cancel': 2, 'label': 'q10_tp2_sl1_h2'},
    {'q': 0.10, 'tp': 3, 'sl': 2, 'hold': 3, 'cancel': 3, 'label': 'q10_tp3_sl2_h3'},
]


def classify_date_regime(mbo_data):
    """Classify a date as green/red/flat from MBO price data.
    Uses first 5% vs last 5% of session prices with wider threshold (20 ticks = 5 pts)."""
    prices = mbo_data['price']
    valid = prices[prices > 0]
    if len(valid) < 100:
        return 'flat'
    n = len(valid)
    pct5 = max(50, n // 20)
    open_price = np.median(valid[:pct5])
    close_price = np.median(valid[-pct5:])
    change_ticks = (close_price - open_price) / TICK
    if change_ticks > 20:
        return 'green'
    elif change_ticks < -20:
        return 'red'
    else:
        return 'flat'


def is_rth_ns(ts_ns):
    """Check if a nanosecond timestamp falls within RTH (9:30-16:00 ET)."""
    from datetime import datetime, timezone, timedelta
    dt = datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc)
    month = dt.month
    if 3 <= month <= 10:
        et_offset = timedelta(hours=-4)  # EDT
    else:
        et_offset = timedelta(hours=-5)  # EST
    et_dt = dt + et_offset
    h, m = et_dt.hour, et_dt.minute
    return (h > 9 or (h == 9 and m >= 30)) and h < 16


def filter_rth(pred_timestamps_ns, *arrays):
    """Filter predictions to RTH only. Returns mask and filtered arrays."""
    # Vectorized: convert to hour of day in ET
    # For speed, use the first timestamp to determine EDT/EST offset
    from datetime import datetime, timezone, timedelta
    dt0 = datetime.fromtimestamp(pred_timestamps_ns[0] / 1e9, tz=timezone.utc)
    if 3 <= dt0.month <= 10:
        offset_ns = -4 * 3600 * 10**9
    else:
        offset_ns = -5 * 3600 * 10**9

    et_ns = pred_timestamps_ns.astype(np.int64) + offset_ns
    # Seconds since midnight ET
    secs_since_midnight = (et_ns % (86400 * 10**9)) / 10**9
    rth_open = 9 * 3600 + 30 * 60   # 9:30 AM = 34200s
    rth_close = 16 * 3600            # 4:00 PM = 57600s
    mask = (secs_since_midnight >= rth_open) & (secs_since_midnight < rth_close)

    filtered = [a[mask] for a in arrays]
    return mask, pred_timestamps_ns[mask], filtered


def build_timestamp_index(pred_timestamps_ns, mbo_timestamps_ns):
    """Map each prediction timestamp to the nearest raw MBO event index.

    Uses binary search (np.searchsorted) for O(n log m) mapping.
    Returns array of raw MBO indices, one per prediction.
    """
    # searchsorted finds insertion points — each pred maps to the nearest MBO event at or after
    indices = np.searchsorted(mbo_timestamps_ns, pred_timestamps_ns, side='left')
    # Clamp to valid range
    indices = np.clip(indices, 0, len(mbo_timestamps_ns) - 1)
    return indices


def run_long_only(data, head, q, tp, sl, hold_s, cancel_s, randomize_timing=False, seed=None):
    """Timestamp-fixed simulation. Maps predictions to correct raw MBO events by timestamp."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = None

    trades = []

    for date, ddata in data.items():
        preds = ddata['preds']
        mbo = ddata['mbo']
        pred_event_indices = ddata['pred_event_indices']  # NEW: correct MBO indices

        if head not in preds:
            continue

        signal = preds[head]
        n_preds = len(signal)

        try:
            timestamps = mbo['ts_ns'] if 'ts_ns' in mbo else mbo['ts_event']
            prices = mbo['price']
            sizes = mbo['size']
            sides_arr = mbo['side']
        except KeyError:
            continue

        n_events = len(timestamps)

        # Long-only: positive signals above quantile threshold
        pos_signals = signal.copy()
        pos_signals[pos_signals <= 0] = 0
        if np.max(pos_signals) <= 0:
            continue
        pos_nonzero = pos_signals[pos_signals > 0]
        if len(pos_nonzero) == 0:
            continue
        threshold = np.quantile(pos_nonzero, 1 - q)

        candidates = []
        for pi in range(n_preds):
            if signal[pi] > 0 and signal[pi] >= threshold:
                # Only include if the mapped MBO index is valid
                if pi < len(pred_event_indices) and pred_event_indices[pi] < n_events - 100:
                    candidates.append(pi)

        if not candidates:
            continue

        # Randomize timing for permutation test
        if randomize_timing and rng is not None:
            n_cands = len(candidates)
            valid_indices = [pi for pi in range(n_preds)
                           if pi < len(pred_event_indices) and pred_event_indices[pi] < n_events - 100]
            if len(valid_indices) >= n_cands:
                candidates = sorted(rng.choice(valid_indices, size=n_cands, replace=False).tolist())

        # Execute trades
        for pi in candidates:
            event_idx = int(pred_event_indices[pi])  # FIXED: use timestamp-mapped index
            if event_idx >= n_events - 100:
                continue

            entry_time = timestamps[event_idx]

            # Find BBO from recent events
            bid_price = 0.0
            ask_price = 0.0
            for ei in range(max(0, event_idx - 200), event_idx):
                p = prices[ei]
                if p <= 0 or np.isnan(p):
                    continue
                s = sides_arr[ei]
                if s == 0:
                    bid_price = max(bid_price, p)
                elif s == 1:
                    ask_price = min(ask_price, p) if ask_price > 0 else p

            if bid_price <= 0 or ask_price <= 0 or ask_price <= bid_price:
                continue

            entry_price = bid_price  # long = buy at bid (passive)

            tp_price = entry_price + tp * TICK
            sl_price = entry_price - sl * TICK
            cancel_deadline = entry_time + cancel_s * 10**9

            # Fill sim
            filled = False
            fill_time = 0
            fill_idx = event_idx
            for ei in range(event_idx, min(event_idx + 5000, n_events)):
                t = timestamps[ei]
                p = prices[ei]
                if t > cancel_deadline:
                    break
                if p <= entry_price and sizes[ei] > 0:
                    filled = True; fill_time = t; fill_idx = ei; break

            if not filled:
                continue

            # Exit sim
            exit_deadline = fill_time + hold_s * 10**9
            exit_price = entry_price
            exit_reason = 'time_stop'
            last_p = entry_price
            mfe = 0.0
            mae = 0.0

            for ei in range(fill_idx + 1, min(fill_idx + 50000, n_events)):
                t = timestamps[ei]
                p = prices[ei]
                if p <= 0 or np.isnan(p):
                    continue
                last_p = p

                unrealized = (p - entry_price) / TICK
                mfe = max(mfe, unrealized)
                mae = min(mae, unrealized)

                if p >= tp_price:
                    exit_price = tp_price; exit_reason = 'tp'; break
                if p <= sl_price:
                    exit_price = sl_price; exit_reason = 'sl'; break
                if t >= exit_deadline:
                    exit_price = p; exit_reason = 'time_stop'; break

            if exit_price <= 0:
                exit_price = last_p

            gross = (exit_price - entry_price) / TICK
            cost = 0.376 if exit_reason == 'tp' else 1.376
            net = gross - cost

            trades.append({
                'date': date,
                'gross': float(gross),
                'net': float(net),
                'exit_reason': exit_reason,
                'mfe': float(mfe),
                'mae': float(mae),
                'entry_price': float(entry_price),
            })

    return trades


def regime_stratify(trades, date_regimes):
    """Stratify by regime. Returns per-regime stats + regime gap check."""
    regime_trades = {'green': [], 'red': [], 'flat': []}
    for t in trades:
        r = date_regimes.get(t['date'], 'flat')
        regime_trades[r].append(t)

    stats = {}
    for regime, rtrades in regime_trades.items():
        if len(rtrades) < 3:
            stats[regime] = {'n_trades': len(rtrades), 'sharpe': 0, 'net_per_trade': 0, 'n_days': 0}
            continue
        nets = np.array([t['net'] for t in rtrades])
        day_data = defaultdict(float)
        for t in rtrades:
            day_data[t['date']] += t['net']
        day_vals = list(day_data.values())
        day_sharpe = float(np.mean(day_vals) / np.std(day_vals) * np.sqrt(252)) if len(day_vals) > 1 and np.std(day_vals) > 0 else 0
        stats[regime] = {
            'n_trades': len(rtrades),
            'n_days': len(day_data),
            'net_per_trade': round(float(np.mean(nets)), 4),
            'total_net': round(float(np.sum(nets)), 1),
            'win_rate': round(float(np.mean(nets > 0)), 3),
            'day_sharpe': round(day_sharpe, 2),
            'green_days': sum(1 for v in day_vals if v > 0),
        }

    g_days = stats.get('green', {}).get('n_days', 0)
    r_days = stats.get('red', {}).get('n_days', 0)
    g_sharpe = abs(stats.get('green', {}).get('day_sharpe', 0))
    r_sharpe = abs(stats.get('red', {}).get('day_sharpe', 0))
    max_sharpe = max(g_sharpe, r_sharpe, 0.001)
    regime_gap = abs(g_sharpe - r_sharpe) / max_sharpe

    if g_days < 3 or r_days < 3:
        stats['regime_gap'] = round(regime_gap, 3)
        stats['regime_gate'] = 'UNDERPOWERED'
        stats['regime_note'] = f'green={g_days}d, red={r_days}d — need >=3 each'
    else:
        stats['regime_gap'] = round(regime_gap, 3)
        stats['regime_gate'] = 'PASS' if regime_gap <= 0.50 else 'FAIL'

    return stats


def mfe_horizon_check(trades, hold_s):
    """Check MFE within horizon."""
    mfes = [t['mfe'] for t in trades]
    if len(mfes) < 10:
        return {'gate': 'SKIP', 'reason': 'too few trades'}
    p90_mfe = np.percentile(mfes, 90)
    median_mfe = np.median(mfes)
    return {
        'p90_mfe': round(float(p90_mfe), 1),
        'median_mfe': round(float(median_mfe), 1),
        'mean_mfe': round(float(np.mean(mfes)), 1),
        'p10_mfe': round(float(np.percentile(mfes, 10)), 1),
        'gate': 'INFO',
    }


# =============================================================================
# MAIN
# =============================================================================

print(f"{'='*70}")
print(f"TICK REPLAY v31 — TIMESTAMP-FIXED LONG-ONLY BACKTEST")
print(f"{'='*70}")
print(f"FIX: Predictions mapped to raw MBO by TIMESTAMP, not positional stride.")
print(f"Testing {len(CONFIGS)} configs ({len([c for c in CONFIGS if c['hold'] <= 5])} horizon-aligned).\n")

# Check if timestamp-fixed prediction files exist
if not os.path.exists(PRED_DIR):
    print(f"ERROR: Prediction directory {PRED_DIR} not found.")
    print("Run rebuild_preds_with_timestamps.py first.")
    sys.exit(1)

pred_files = sorted(os.listdir(PRED_DIR))
pred_dates = [f.replace('oot_', '').replace('.npz', '') for f in pred_files if f.startswith('oot_')]

if not pred_dates:
    print(f"ERROR: No prediction files found in {PRED_DIR}")
    sys.exit(1)

mbo_files = {f.replace('mbo_', '').replace('.npz', ''): os.path.join(MBO_DIR, f)
             for f in os.listdir(MBO_DIR) if f.endswith('.npz')}
common_dates = sorted([d for d in pred_dates if d in mbo_files])
print(f"Prediction dates: {len(pred_dates)}, MBO dates: {len(mbo_files)}, Common: {len(common_dates)}")

# Load data with TIMESTAMP-BASED index mapping
all_data = {}
skipped = 0
for date in common_dates:
    try:
        pred_data = np.load(os.path.join(PRED_DIR, f'oot_{date}.npz'))
        mbo_data = np.load(mbo_files[date])

        # Must have timestamps
        if 'pred_timestamps_ns' not in pred_data:
            print(f"  Skip {date}: no pred_timestamps_ns (use rebuild_preds_with_timestamps.py)")
            skipped += 1
            continue

        pred_ts_raw = pred_data['pred_timestamps_ns']

        # CRITICAL: Filter to RTH only (9:30-16:00 ET)
        # Overnight predictions cause garbage exits (timestamp gaps, no liquidity)
        rth_mask, pred_ts, (sig_lr, sig_comp, sig_eofi, sig_pdi) = filter_rth(
            pred_ts_raw,
            pred_data['pred_log_ret_1s'],
            pred_data['composite_signal'],
            pred_data['eofi_1s'],
            pred_data['pdi_1s'],
        )
        n_rth = len(pred_ts)
        n_total = len(pred_ts_raw)
        if n_rth < 50:
            print(f"  Skip {date}: only {n_rth}/{n_total} preds in RTH")
            skipped += 1
            continue

        # Get raw MBO timestamps
        mbo_ts = mbo_data['ts_ns'] if 'ts_ns' in mbo_data else mbo_data['ts_event']

        # Map RTH-filtered predictions to raw MBO events by timestamp
        pred_event_indices = build_timestamp_index(pred_ts, mbo_ts)

        # Verify mapping quality (should be sub-second for RTH)
        mapped_ts = mbo_ts[pred_event_indices]
        drift_ns = np.abs(mapped_ts.astype(np.int64) - pred_ts.astype(np.int64))
        max_drift_ms = np.max(drift_ns) / 1e6
        mean_drift_ms = np.mean(drift_ns) / 1e6

        if max_drift_ms > 1000:  # >1s drift after RTH filter = real problem
            print(f"  WARN {date}: RTH max drift = {max_drift_ms:.0f}ms (mean={mean_drift_ms:.1f}ms)")

        # Rebuild filtered prediction dict
        filtered_preds = {
            'pred_log_ret_1s': sig_lr,
            'composite_signal': sig_comp,
            'eofi_1s': sig_eofi,
            'pdi_1s': sig_pdi,
            'pred_timestamps_ns': pred_ts,
            'n_preds': n_rth,
        }

        all_data[date] = {
            'preds': filtered_preds,
            'mbo': dict(mbo_data),
            'pred_event_indices': pred_event_indices,
        }
    except Exception as e:
        print(f"  Skip {date}: {e}")
        skipped += 1

print(f"Loaded {len(all_data)} dates (skipped {skipped})")

# Classify regimes
date_regimes = {}
for date, ddata in all_data.items():
    date_regimes[date] = classify_date_regime(ddata['mbo'])
regime_counts = defaultdict(int)
for r in date_regimes.values():
    regime_counts[r] += 1
print(f"Regimes: {dict(regime_counts)}")

# Show timestamp mapping stats for first date
first_date = sorted(all_data.keys())[0]
fd = all_data[first_date]
fd_pred_ts = fd['preds']['pred_timestamps_ns']
fd_mbo_ts = fd['mbo']['ts_ns'] if 'ts_ns' in fd['mbo'] else fd['mbo']['ts_event']
fd_mapped = fd['pred_event_indices']
print(f"\nTimestamp mapping check ({first_date}):")
print(f"  Predictions: {len(fd_pred_ts)}")
print(f"  Raw MBO events: {len(fd_mbo_ts)}")
print(f"  Mapped index range: {fd_mapped[0]} to {fd_mapped[-1]} ({fd_mapped[-1]/len(fd_mbo_ts)*100:.1f}% of day)")
mapped_ts = fd_mbo_ts[fd_mapped]
drift = np.abs(mapped_ts.astype(np.int64) - fd_pred_ts.astype(np.int64))
print(f"  Max timestamp drift: {np.max(drift)/1e6:.1f}ms, Mean: {np.mean(drift)/1e6:.1f}ms")
print()

# === MAIN SWEEP ===
results = []
t0 = time.time()

for ci, cfg in enumerate(CONFIGS):
    label = cfg['label']
    elapsed_so_far = time.time() - t0
    print(f"\n[{ci+1}/{len(CONFIGS)}] {label} (q={cfg['q']}, TP={cfg['tp']}, SL={cfg['sl']}, hold={cfg['hold']}s) [{elapsed_so_far/60:.0f}m elapsed]")

    # Phase 1: Real trades
    trades = run_long_only(all_data, 'composite_signal', cfg['q'], cfg['tp'], cfg['sl'], cfg['hold'], cfg['cancel'])

    if len(trades) < 10:
        print(f"  Too few trades ({len(trades)}) — skip")
        continue

    nets = np.array([t['net'] for t in trades])
    day_data = defaultdict(float)
    for t in trades:
        day_data[t['date']] += t['net']
    day_vals = list(day_data.values())
    n_days = len(day_vals)
    day_sharpe = float(np.mean(day_vals) / np.std(day_vals) * np.sqrt(252)) if n_days > 1 and np.std(day_vals) > 0 else 0

    # Day concentration
    sorted_dv = sorted(day_vals, reverse=True)
    total_net = float(np.sum(nets))
    top2_conc = (sum(sorted_dv[:2]) / total_net * 100) if total_net > 0 else 999

    exits = defaultdict(int)
    for t in trades:
        exits[t['exit_reason']] += 1

    stats = {
        'label': label,
        'params': cfg,
        'n_trades': len(trades),
        'trades_per_day': round(len(trades) / max(n_days, 1), 1),
        'net_per_trade': round(float(np.mean(nets)), 4),
        'total_net_ticks': round(total_net, 1),
        'win_rate': round(float(np.mean(nets > 0)), 3),
        'day_sharpe': round(day_sharpe, 2),
        'n_days': n_days,
        'green_days': sum(1 for v in day_vals if v > 0),
        'red_days': sum(1 for v in day_vals if v <= 0),
        'day_wr': round(sum(1 for v in day_vals if v > 0) / max(n_days, 1), 3),
        'top2_day_concentration': round(top2_conc, 1),
        'exits': dict(exits),
        'version': 'v31_timestamp_fixed',
    }

    # Sortino
    neg_returns = [v for v in day_vals if v < 0]
    downside_std = np.std(neg_returns) if len(neg_returns) > 1 else 1.0
    sortino = float(np.mean(day_vals) / downside_std * np.sqrt(252)) if downside_std > 0 else 0
    stats['day_sortino'] = round(sortino, 2)

    # Profit factor
    gross_wins = sum(n for n in nets if n > 0)
    gross_losses = abs(sum(n for n in nets if n < 0))
    stats['profit_factor'] = round(gross_wins / max(gross_losses, 0.001), 2)

    # MFE check
    mfe_info = mfe_horizon_check(trades, cfg['hold'])
    stats['mfe'] = mfe_info

    # Regime stratification
    regime_stats = regime_stratify(trades, date_regimes)
    stats['regime'] = regime_stats

    # Original vs new date split
    orig_trades = [t for t in trades if t['date'] < '20260320']
    new_trades = [t for t in trades if t['date'] >= '20260401']
    stats['orig_net_per_trade'] = round(float(np.mean([t['net'] for t in orig_trades])), 4) if orig_trades else 0
    stats['new_net_per_trade'] = round(float(np.mean([t['net'] for t in new_trades])), 4) if new_trades else 0
    stats['orig_n_days'] = len(set(t['date'] for t in orig_trades))
    stats['new_n_days'] = len(set(t['date'] for t in new_trades))

    print(f"  {len(trades)} trades, {n_days} days, net/trade={stats['net_per_trade']:+.3f}, Day Sharpe={day_sharpe:.2f}, PF={stats['profit_factor']:.2f}")
    print(f"  Regime: {regime_stats['regime_gate']} (gap={regime_stats['regime_gap']:.2f}) | Green={regime_stats.get('green', {}).get('day_sharpe', 0):.1f}, Red={regime_stats.get('red', {}).get('day_sharpe', 0):.1f}")
    print(f"  DayConc={top2_conc:.0f}% | Orig={stats['orig_net_per_trade']:+.3f}/t ({stats['orig_n_days']}d), New={stats['new_net_per_trade']:+.3f}/t ({stats['new_n_days']}d)")
    print(f"  MFE: median={mfe_info.get('median_mfe', 0):.1f}, p90={mfe_info.get('p90_mfe', 0):.1f}")
    print(f"  Exits: {dict(exits)}")

    # Quick reject on day concentration
    if top2_conc > 70:
        stats['perm_verdict'] = 'SKIP_CONC'
        print(f"  Day concentration {top2_conc:.0f}% > 70% cap — skipping permtest")
        results.append(stats)
        continue

    # Phase 2: Permutation test
    print(f"  Running permutation test ({N_PERMS} perms)...", flush=True)
    real_mean = stats['net_per_trade']
    perm_means = []
    for p in range(N_PERMS):
        if (p + 1) % 50 == 0:
            print(f"    Perm {p+1}/{N_PERMS}...", flush=True)
        perm_trades = run_long_only(all_data, 'composite_signal', cfg['q'], cfg['tp'], cfg['sl'],
                                     cfg['hold'], cfg['cancel'], randomize_timing=True, seed=p*137)
        if len(perm_trades) > 0:
            perm_means.append(float(np.mean([t['net'] for t in perm_trades])))

    if perm_means:
        perm_arr = np.array(perm_means)
        p_value = float(np.mean(perm_arr >= real_mean))
        stats['perm_real_mean'] = real_mean
        stats['perm_random_mean'] = round(float(np.mean(perm_arr)), 4)
        stats['perm_edge'] = round(real_mean - float(np.mean(perm_arr)), 4)
        stats['perm_p_value'] = p_value
        stats['perm_verdict'] = 'PASS' if p_value < 0.05 else 'FAIL'
        print(f"  Perm: real={real_mean:+.3f}, random={stats['perm_random_mean']:+.3f}, edge={stats['perm_edge']:+.3f}, p={p_value:.3f} → {stats['perm_verdict']}")

    # Overall verdict
    perm_pass = stats.get('perm_verdict') == 'PASS'
    regime_ok = regime_stats['regime_gate'] in ('PASS', 'UNDERPOWERED')
    conc_pass = top2_conc <= 70
    stats['overall_verdict'] = 'PRODUCTION_CANDIDATE' if (perm_pass and regime_ok and conc_pass) else 'REJECT'
    if regime_stats['regime_gate'] == 'UNDERPOWERED' and perm_pass and conc_pass:
        stats['overall_verdict'] = 'CANDIDATE_REGIME_TBD'
    print(f"  → {stats['overall_verdict']}")

    results.append(stats)

elapsed = time.time() - t0

# === SUMMARY ===
print(f"\n{'='*70}")
print(f"v31 TIMESTAMP-FIXED RESULTS SUMMARY ({elapsed/60:.0f} minutes)")
print(f"{'='*70}")

candidates = [r for r in results if r.get('overall_verdict') in ('PRODUCTION_CANDIDATE', 'CANDIDATE_REGIME_TBD')]
rejects = [r for r in results if r.get('overall_verdict') not in ('PRODUCTION_CANDIDATE', 'CANDIDATE_REGIME_TBD')]

print(f"\nPRODUCTION CANDIDATES: {len(candidates)}")
for r in sorted(candidates, key=lambda x: x.get('day_sharpe', 0), reverse=True):
    print(f"  {r['label']}: net/trade={r['net_per_trade']:+.3f}, Sharpe={r['day_sharpe']:.2f}, "
          f"PF={r['profit_factor']:.2f}, WR={r['win_rate']:.1%}, trades={r['n_trades']}, "
          f"perm_edge={r.get('perm_edge', 0):+.3f}, p={r.get('perm_p_value', 1):.3f}")

print(f"\nREJECTED: {len(rejects)}")
for r in sorted(rejects, key=lambda x: x.get('day_sharpe', 0), reverse=True):
    reason = r.get('perm_verdict', 'unknown')
    if r.get('regime', {}).get('regime_gate') == 'FAIL':
        reason += '+REGIME'
    print(f"  {r['label']}: net/trade={r['net_per_trade']:+.3f}, Sharpe={r['day_sharpe']:.2f}, "
          f"reason={reason}")

# Compare horizon-aligned vs original configs
horizon_configs = [r for r in results if r['params']['hold'] <= 10]
original_configs = [r for r in results if r['params']['hold'] >= 30]

if horizon_configs and original_configs:
    print(f"\n{'='*70}")
    print(f"HORIZON ALIGNMENT COMPARISON")
    print(f"{'='*70}")
    print(f"\nOriginal (30s hold, 16-tick TP):")
    for r in original_configs:
        print(f"  {r['label']}: net/trade={r['net_per_trade']:+.3f}, Sharpe={r['day_sharpe']:.2f}, "
              f"trades={r['n_trades']}, MFE_p90={r.get('mfe', {}).get('p90_mfe', 0):.1f}")
    print(f"\nHorizon-Aligned (≤10s hold, ≤8 tick TP):")
    for r in horizon_configs:
        print(f"  {r['label']}: net/trade={r['net_per_trade']:+.3f}, Sharpe={r['day_sharpe']:.2f}, "
              f"trades={r['n_trades']}, MFE_p90={r.get('mfe', {}).get('p90_mfe', 0):.1f}")

# Save all results
with open(OUTPUT / 'results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to {OUTPUT / 'results.json'}")
print(f"Total runtime: {elapsed/60:.1f} minutes")
