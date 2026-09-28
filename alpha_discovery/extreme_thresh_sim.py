"""
Extreme Threshold Dynamic Exit Sim — Tests ALL exit strategies at HIGH thresholds.

The market order sweep showed profitability at thresh >= 2.0.
This script focuses on 1.5-4.0 range with all 5 dynamic exit strategies.

Usage:
    python alpha_discovery/extreme_thresh_sim.py --n-days 50
"""

import sys
import time
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np

LVL3_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(LVL3_ROOT))

RESULTS_DIR = LVL3_ROOT / "alpha_discovery" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(str(RESULTS_DIR / f"extreme_thresh_{_ts}.log"), mode='w'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger("ext_thresh")

TICK = 0.25
TICK_VAL = 12.50
COMM_TICKS = 3.00 / TICK_VAL  # 0.24
TOTAL_COST = 1.0 + COMM_TICKS  # 1.24
BARS_SEC = 10

FEAT_CACHE = LVL3_ROOT / "data" / "processed" / "mbo_features_cache"
SIG_DIR = LVL3_ROOT / "data" / "processed" / "signal_predictions"


def preload_days(n_days):
    days = {}
    for f in sorted(FEAT_CACHE.glob("*_mbo_features.npz"))[:n_days]:
        date = f.stem.replace("_mbo_features", "")
        data = np.load(str(f))
        feats = data['mbo_features']
        days[date] = {'mid': feats[:, 0].copy(), 'spread': feats[:, 1].copy()}
        del feats, data
    return days


def load_signal(sig_name, date):
    path = SIG_DIR / f"{sig_name}_{date}.npz"
    if not path.exists():
        return None
    return np.load(str(path))['predictions']


def sim_static(mid, spread, preds, thresh, hold_bars, trail=0, cooldown=10):
    """Baseline static hold/trail market order sim."""
    n = min(len(mid), len(preds))
    total_pnl = 0.0; trades = 0; wins = 0
    last_exit = -cooldown; in_pos = False
    entry_price = 0.0; direction = 0; entry_bar = 0; peak = 0.0

    for i in range(n):
        if in_pos:
            unrealized = ((mid[i] - entry_price) if direction == 1 else (entry_price - mid[i])) / TICK
            peak = max(peak, unrealized)
            bars = i - entry_bar
            do_exit = bars >= hold_bars or (trail > 0 and (peak - unrealized) >= trail)
            if do_exit:
                pnl = unrealized - spread[i] / 2.0 / TICK - COMM_TICKS
                total_pnl += pnl; trades += 1
                if pnl > 0: wins += 1
                in_pos = False; last_exit = i
        elif i - last_exit >= cooldown:
            if abs(preds[i]) > thresh:
                if preds[i] > 0:
                    entry_price = mid[i] + spread[i] / 2.0; direction = 1
                else:
                    entry_price = mid[i] - spread[i] / 2.0; direction = -1
                in_pos = True; entry_bar = i; peak = 0.0

    if in_pos:
        i = n - 1
        unrealized = ((mid[i] - entry_price) if direction == 1 else (entry_price - mid[i])) / TICK
        pnl = unrealized - spread[i] / 2.0 / TICK - COMM_TICKS
        total_pnl += pnl; trades += 1
        if pnl > 0: wins += 1
    return total_pnl * TICK_VAL, trades, wins


def sim_signal_flip(mid, spread, preds, thresh, max_hold=3000, cooldown=10):
    n = min(len(mid), len(preds))
    total_pnl = 0.0; trades = 0; wins = 0
    last_exit = -cooldown; in_pos = False
    entry_price = 0.0; direction = 0; entry_bar = 0

    for i in range(n):
        if in_pos:
            pred_sign = 1 if preds[i] > 0 else (-1 if preds[i] < 0 else 0)
            do_exit = (direction == 1 and pred_sign == -1) or \
                      (direction == -1 and pred_sign == 1) or \
                      (i - entry_bar >= max_hold)
            if do_exit:
                unrealized = ((mid[i] - entry_price) if direction == 1 else (entry_price - mid[i])) / TICK
                pnl = unrealized - spread[i] / 2.0 / TICK - COMM_TICKS
                total_pnl += pnl; trades += 1
                if pnl > 0: wins += 1
                in_pos = False; last_exit = i
        elif i - last_exit >= cooldown:
            if abs(preds[i]) > thresh:
                if preds[i] > 0:
                    entry_price = mid[i] + spread[i] / 2.0; direction = 1
                else:
                    entry_price = mid[i] - spread[i] / 2.0; direction = -1
                in_pos = True; entry_bar = i

    if in_pos:
        i = n - 1
        unrealized = ((mid[i] - entry_price) if direction == 1 else (entry_price - mid[i])) / TICK
        pnl = unrealized - spread[i] / 2.0 / TICK - COMM_TICKS
        total_pnl += pnl; trades += 1
        if pnl > 0: wins += 1
    return total_pnl * TICK_VAL, trades, wins


def sim_signal_weaken(mid, spread, preds, thresh, decay_factor=0.3, max_hold=3000, cooldown=10):
    n = min(len(mid), len(preds))
    total_pnl = 0.0; trades = 0; wins = 0
    last_exit = -cooldown; in_pos = False
    entry_price = 0.0; direction = 0; entry_bar = 0; entry_strength = 0.0

    for i in range(n):
        if in_pos:
            do_exit = abs(preds[i]) < entry_strength * decay_factor or (i - entry_bar >= max_hold)
            if do_exit:
                unrealized = ((mid[i] - entry_price) if direction == 1 else (entry_price - mid[i])) / TICK
                pnl = unrealized - spread[i] / 2.0 / TICK - COMM_TICKS
                total_pnl += pnl; trades += 1
                if pnl > 0: wins += 1
                in_pos = False; last_exit = i
        elif i - last_exit >= cooldown:
            if abs(preds[i]) > thresh:
                if preds[i] > 0:
                    entry_price = mid[i] + spread[i] / 2.0; direction = 1
                else:
                    entry_price = mid[i] - spread[i] / 2.0; direction = -1
                in_pos = True; entry_bar = i; entry_strength = abs(preds[i])

    if in_pos:
        i = n - 1
        unrealized = ((mid[i] - entry_price) if direction == 1 else (entry_price - mid[i])) / TICK
        pnl = unrealized - spread[i] / 2.0 / TICK - COMM_TICKS
        total_pnl += pnl; trades += 1
        if pnl > 0: wins += 1
    return total_pnl * TICK_VAL, trades, wins


def sim_dynamic_trail(mid, spread, preds, thresh, base_trail=8.0, max_hold=3000, cooldown=10):
    n = min(len(mid), len(preds))
    total_pnl = 0.0; trades = 0; wins = 0
    last_exit = -cooldown; in_pos = False
    entry_price = 0.0; direction = 0; entry_bar = 0
    entry_strength = 0.0; peak = 0.0

    for i in range(n):
        if in_pos:
            unrealized = ((mid[i] - entry_price) if direction == 1 else (entry_price - mid[i])) / TICK
            peak = max(peak, unrealized)
            ratio = min(abs(preds[i]) / max(entry_strength, 1e-10), 1.0)
            adaptive_trail = max(base_trail * ratio, 2.0)
            do_exit = (peak - unrealized) >= adaptive_trail or (i - entry_bar >= max_hold)
            if do_exit:
                pnl = unrealized - spread[i] / 2.0 / TICK - COMM_TICKS
                total_pnl += pnl; trades += 1
                if pnl > 0: wins += 1
                in_pos = False; last_exit = i
        elif i - last_exit >= cooldown:
            if abs(preds[i]) > thresh:
                if preds[i] > 0:
                    entry_price = mid[i] + spread[i] / 2.0; direction = 1
                else:
                    entry_price = mid[i] - spread[i] / 2.0; direction = -1
                in_pos = True; entry_bar = i
                entry_strength = abs(preds[i]); peak = 0.0

    if in_pos:
        i = n - 1
        unrealized = ((mid[i] - entry_price) if direction == 1 else (entry_price - mid[i])) / TICK
        pnl = unrealized - spread[i] / 2.0 / TICK - COMM_TICKS
        total_pnl += pnl; trades += 1
        if pnl > 0: wins += 1
    return total_pnl * TICK_VAL, trades, wins


def sim_multi_signal(mid, spread, entry_preds, exit_preds,
                     entry_thresh, exit_thresh, max_hold=3000, cooldown=10):
    n = min(len(mid), len(entry_preds), len(exit_preds))
    total_pnl = 0.0; trades = 0; wins = 0
    last_exit = -cooldown; in_pos = False
    entry_price = 0.0; direction = 0; entry_bar = 0

    for i in range(n):
        if in_pos:
            do_exit = (direction == 1 and exit_preds[i] < -exit_thresh) or \
                      (direction == -1 and exit_preds[i] > exit_thresh) or \
                      (i - entry_bar >= max_hold)
            if do_exit:
                unrealized = ((mid[i] - entry_price) if direction == 1 else (entry_price - mid[i])) / TICK
                pnl = unrealized - spread[i] / 2.0 / TICK - COMM_TICKS
                total_pnl += pnl; trades += 1
                if pnl > 0: wins += 1
                in_pos = False; last_exit = i
        elif i - last_exit >= cooldown:
            if abs(entry_preds[i]) > entry_thresh:
                if entry_preds[i] > 0:
                    entry_price = mid[i] + spread[i] / 2.0; direction = 1
                else:
                    entry_price = mid[i] - spread[i] / 2.0; direction = -1
                in_pos = True; entry_bar = i

    if in_pos:
        i = n - 1
        unrealized = ((mid[i] - entry_price) if direction == 1 else (entry_price - mid[i])) / TICK
        pnl = unrealized - spread[i] / 2.0 / TICK - COMM_TICKS
        total_pnl += pnl; trades += 1
        if pnl > 0: wins += 1
    return total_pnl * TICK_VAL, trades, wins


def run_sweep(days, signals, sig_name, strategy_fn, params_list, strategy_name):
    """Generic sweep function. Returns list of result dicts."""
    if sig_name not in signals:
        return []
    sig_data = signals[sig_name]
    results = []
    for params in params_list:
        day_pnls = []
        total_trades = 0
        total_wins = 0
        for date in sorted(sig_data.keys()):
            pnl, tr, w = strategy_fn(
                days[date]['mid'], days[date]['spread'],
                sig_data[date], **params
            )
            day_pnls.append(pnl)
            total_trades += tr
            total_wins += w
        n = len(day_pnls)
        active = sum(1 for t, p in zip([total_trades]*n, day_pnls) if True)
        pos = sum(1 for p in day_pnls if p > 0)
        total = sum(day_pnls)
        results.append({
            'strategy': strategy_name,
            'signal': sig_name,
            **params,
            'total_pnl': total,
            'mean_daily': total / max(n, 1),
            'trades': total_trades,
            'trades_per_day': total_trades / max(n, 1),
            'wr': total_wins / max(total_trades, 1),
            'pos_days': pos,
            'n_days': n,
            'pct_pos': 100 * pos / max(n, 1),
            'sharpe': float((np.mean(day_pnls) / max(np.std(day_pnls), 1e-10)) * np.sqrt(252))
                      if n > 1 else 0.0,
        })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-days', type=int, default=50)
    args = parser.parse_args()

    log.info("Extreme Threshold Dynamic Exit Simulator")
    log.info(f"  Market orders: {TOTAL_COST:.2f} ticks/RT")
    log.info(f"  Focus: thresholds 1.5-4.0 where edge lives")

    t0 = time.time()
    log.info(f"\nPreloading {args.n_days} days...")
    days = preload_days(args.n_days)
    log.info(f"  Loaded {len(days)} days in {time.time()-t0:.1f}s")

    # Top signals
    signal_names = [
        'cancel_asym', 'cancel_asym_chain', 'volgated',
        'depth_ratio', 'depth_ratio_z', 'top5_ensemble',
        'order_frag', 'ask_orders', 'book_imb_z',
        'slow_decay_combo', 'novel_risk_adjusted_return',
    ]

    signals = {}
    for sig in signal_names:
        sig_data = {}
        for date in days:
            s = load_signal(sig, date)
            if s is not None:
                sig_data[date] = s
        if sig_data:
            signals[sig] = sig_data
            log.info(f"  {sig}: {len(sig_data)} days")

    # EXTREME thresholds
    thresholds = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

    all_results = []
    combo_count = 0

    # ── Strategy 0: Static baseline (for comparison) ───────────────
    log.info(f"\n{'='*80}")
    log.info("STRATEGY: Static Baseline (market orders)")
    log.info(f"{'='*80}")

    for sig_name in signals:
        params_list = []
        for thresh in thresholds:
            for hold in [100, 300, 600, 1200, 3000]:
                for trail in [0, 4, 8, 12]:
                    params_list.append({'thresh': thresh, 'hold_bars': hold, 'trail': trail})
        results = run_sweep(days, signals, sig_name, sim_static, params_list, 'static')
        all_results.extend(results)
        combo_count += len(params_list)
        log.info(f"  {sig_name}: {len(params_list)} combos done")

    # ── Strategy 1: Signal-Flip ────────────────────────────────────
    log.info(f"\n{'='*80}")
    log.info("STRATEGY: Signal-Flip Exit")
    log.info(f"{'='*80}")

    for sig_name in signals:
        params_list = []
        for thresh in thresholds:
            for max_hold in [300, 600, 1800, 3000, 6000]:
                params_list.append({'thresh': thresh, 'max_hold': max_hold})
        results = run_sweep(days, signals, sig_name, sim_signal_flip, params_list, 'signal_flip')
        all_results.extend(results)
        combo_count += len(params_list)
        log.info(f"  {sig_name}: {len(params_list)} combos done")

    # ── Strategy 2: Signal-Weaken ──────────────────────────────────
    log.info(f"\n{'='*80}")
    log.info("STRATEGY: Signal-Weaken Exit")
    log.info(f"{'='*80}")

    for sig_name in signals:
        params_list = []
        for thresh in thresholds:
            for decay_factor in [0.1, 0.3, 0.5, 0.7]:
                for max_hold in [600, 1800, 3000]:
                    params_list.append({'thresh': thresh, 'decay_factor': decay_factor, 'max_hold': max_hold})
        results = run_sweep(days, signals, sig_name, sim_signal_weaken, params_list, 'signal_weaken')
        all_results.extend(results)
        combo_count += len(params_list)
        log.info(f"  {sig_name}: {len(params_list)} combos done")

    # ── Strategy 3: Dynamic Trail ──────────────────────────────────
    log.info(f"\n{'='*80}")
    log.info("STRATEGY: Dynamic Trailing Stop")
    log.info(f"{'='*80}")

    for sig_name in signals:
        params_list = []
        for thresh in thresholds:
            for base_trail in [4, 8, 12, 16, 24]:
                for max_hold in [600, 1800, 3000]:
                    params_list.append({'thresh': thresh, 'base_trail': base_trail, 'max_hold': max_hold})
        results = run_sweep(days, signals, sig_name, sim_dynamic_trail, params_list, 'dynamic_trail')
        all_results.extend(results)
        combo_count += len(params_list)
        log.info(f"  {sig_name}: {len(params_list)} combos done")

    # ── Strategy 4: Multi-Signal ───────────────────────────────────
    log.info(f"\n{'='*80}")
    log.info("STRATEGY: Multi-Signal Entry/Exit")
    log.info(f"{'='*80}")

    multi_pairs = [
        ('cancel_asym', 'depth_ratio'),
        ('cancel_asym', 'order_frag'),
        ('cancel_asym_chain', 'depth_ratio'),
        ('cancel_asym_chain', 'order_frag'),
        ('depth_ratio', 'cancel_asym'),
        ('depth_ratio', 'order_frag'),
        ('order_frag', 'depth_ratio'),
        ('volgated', 'depth_ratio'),
        ('novel_risk_adjusted_return', 'cancel_asym'),
    ]

    for entry_sig, exit_sig in multi_pairs:
        if entry_sig not in signals or exit_sig not in signals:
            continue
        dates = sorted(set(signals[entry_sig].keys()) & set(signals[exit_sig].keys()) & set(days.keys()))
        if not dates:
            continue

        for entry_thresh in thresholds:
            for exit_thresh in [0.3, 0.5, 1.0, 1.5]:
                day_pnls = []
                total_trades = 0
                total_wins = 0
                for date in dates:
                    pnl, tr, w = sim_multi_signal(
                        days[date]['mid'], days[date]['spread'],
                        signals[entry_sig][date], signals[exit_sig][date],
                        entry_thresh, exit_thresh
                    )
                    day_pnls.append(pnl)
                    total_trades += tr
                    total_wins += w
                n = len(day_pnls)
                pos = sum(1 for p in day_pnls if p > 0)
                total = sum(day_pnls)
                all_results.append({
                    'strategy': 'multi_signal',
                    'entry_signal': entry_sig,
                    'exit_signal': exit_sig,
                    'signal': f"{entry_sig}>{exit_sig}",
                    'entry_thresh': entry_thresh,
                    'exit_thresh': exit_thresh,
                    'total_pnl': total,
                    'mean_daily': total / max(n, 1),
                    'trades': total_trades,
                    'trades_per_day': total_trades / max(n, 1),
                    'wr': total_wins / max(total_trades, 1),
                    'pos_days': pos,
                    'n_days': n,
                    'pct_pos': 100 * pos / max(n, 1),
                    'sharpe': float((np.mean(day_pnls) / max(np.std(day_pnls), 1e-10)) * np.sqrt(252))
                              if n > 1 else 0.0,
                })
                combo_count += 1
        log.info(f"  {entry_sig} > {exit_sig}: done")

    elapsed = time.time() - t0

    # ── GRAND RANKING ──────────────────────────────────────────────
    log.info(f"\n\n{'='*80}")
    log.info(f"GRAND RANKING — {len(all_results)} configs, {combo_count} combos, {elapsed:.0f}s")
    log.info(f"{'='*80}")

    # Filter to configs with at least 5 active days
    active = [r for r in all_results if r['trades'] > 0]
    active.sort(key=lambda x: x['mean_daily'], reverse=True)

    log.info(f"\n  TOP 50 (all strategies, extreme thresholds):")
    log.info(f"  {'strategy':>14} {'signal':>25} {'mean_pnl':>10} {'total':>10} {'wr':>6} {'tr/d':>6} {'%pos':>6} {'sharpe':>7} {'tr':>5} {'n':>3}")
    log.info("  " + "-" * 120)

    for r in active[:50]:
        sig = r.get('signal', '')
        log.info(f"  {r['strategy']:>14} {sig:>25} {r['mean_daily']:>+10.2f} "
                f"{r['total_pnl']:>+10.0f} {r['wr']*100:>5.1f}% "
                f"{r['trades_per_day']:>5.1f} {r['pct_pos']:>5.1f}% "
                f"{r['sharpe']:>+6.2f} {r['trades']:>5} {r['n_days']:>3}")

    # By strategy summary
    log.info(f"\n  BY STRATEGY (mean of top 10 per strategy):")
    by_strat = defaultdict(list)
    for r in active:
        by_strat[r['strategy']].append(r)
    for strat, results in sorted(by_strat.items()):
        top10 = sorted(results, key=lambda x: x['mean_daily'], reverse=True)[:10]
        mp = np.mean([r['mean_daily'] for r in top10])
        wr = np.mean([r['wr'] for r in top10])
        pp = np.mean([r['pct_pos'] for r in top10])
        log.info(f"    {strat:>14}: avg_top10_pnl={mp:>+8.2f}  wr={wr*100:>5.1f}%  pct_pos={pp:>5.1f}%")

    # By signal summary
    log.info(f"\n  BY SIGNAL (mean of top 10 per signal):")
    by_sig = defaultdict(list)
    for r in active:
        by_sig[r.get('signal', '')].append(r)
    for sig, results in sorted(by_sig.items()):
        top10 = sorted(results, key=lambda x: x['mean_daily'], reverse=True)[:10]
        mp = np.mean([r['mean_daily'] for r in top10])
        wr = np.mean([r['wr'] for r in top10])
        pp = np.mean([r['pct_pos'] for r in top10])
        log.info(f"    {sig:>25}: avg_top10_pnl={mp:>+8.2f}  wr={wr*100:>5.1f}%  pct_pos={pp:>5.1f}%")

    # Save JSON
    def to_native(obj):
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        return obj

    out = RESULTS_DIR / f"extreme_thresh_{_ts}.json"
    clean = [{k: to_native(v) for k, v in r.items()} for r in active[:100]]
    with open(str(out), 'w') as f:
        json.dump({
            'timestamp': _ts,
            'thresholds': thresholds,
            'total_configs': len(all_results),
            'top_100': clean,
        }, f, indent=2)
    log.info(f"\nSaved: {out}")
    log.info(f"Total time: {elapsed:.0f}s")


if __name__ == '__main__':
    main()
