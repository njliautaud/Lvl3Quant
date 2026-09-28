"""
TRUE Out-of-Sample Composite Holdout Test (LOCAL — all signals)
================================================================
Uses the EXACT same composite methodology as multi_signal_composite.py
but with a STRICT train/test split:
  - Days 1-40: parameter optimization (threshold, hold time)
  - Days 41-100: TRUE OOS holdout with FIXED best params

This addresses the OOS audit finding that threshold=3.5 was picked
by searching over ALL 100 days (in-sample optimization).
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

TICK = 0.25
TICK_VAL = 12.50
COMM_TICKS = 3.00 / TICK_VAL
BARS_SEC = 10

LVL3_ROOT = Path(__file__).parent.parent
FEAT_CACHE = LVL3_ROOT / 'data' / 'processed' / 'mbo_features_cache'
SIG_DIR = LVL3_ROOT / 'data' / 'processed' / 'signal_predictions'
RESULTS_DIR = LVL3_ROOT / 'alpha_discovery' / 'results'

THRESHOLDS = [2.0, 2.5, 3.0, 3.5, 4.0]
HOLDS = [300, 600, 900, 1200]  # seconds
COOLDOWN = 50
TRAIL = 0

TRAIN_DAYS = 40  # First 40 days for optimization
# Days 41-100 are TRUE OOS holdout


def sim_day(mid, spread, preds, thresh, hold_bars, trail=0, cooldown=50):
    """Market order sim — identical to multi_signal_composite.py"""
    n = min(len(mid), len(preds))
    total_pnl = 0.0
    trades = 0
    wins = 0
    last_exit = -cooldown
    in_pos = False
    entry_price = 0.0
    direction = 0
    entry_bar = 0
    peak = 0.0

    for i in range(n):
        if in_pos:
            elapsed = i - entry_bar
            curr_pnl = direction * (mid[i] - entry_price) / TICK

            if trail > 0:
                peak = max(peak, curr_pnl)
                if curr_pnl < peak - trail:
                    # Trailing stop exit
                    pnl = curr_pnl - COMM_TICKS - spread[i] / (2 * TICK)
                    total_pnl += pnl
                    trades += 1
                    if pnl > 0:
                        wins += 1
                    in_pos = False
                    last_exit = i
                    continue

            if elapsed >= hold_bars:
                # Time-based exit
                pnl = curr_pnl - COMM_TICKS - spread[i] / (2 * TICK)
                total_pnl += pnl
                trades += 1
                if pnl > 0:
                    wins += 1
                in_pos = False
                last_exit = i

        elif i - last_exit >= cooldown and abs(preds[i]) > thresh:
            # Entry
            entry_price = mid[i]
            direction = 1 if preds[i] > 0 else -1
            entry_bar = i
            peak = 0.0
            in_pos = True
            entry_price += direction * spread[i] / 2  # cross spread

    return total_pnl * TICK_VAL, trades, wins


def main():
    t0 = time.time()
    print("=" * 70)
    print("TRUE OOS COMPOSITE HOLDOUT TEST (LOCAL — ALL SIGNALS)")
    print("=" * 70)

    # Load features
    files = sorted(FEAT_CACHE.glob('*_mbo_features.npz'))[:100]
    print(f"\nLoading {len(files)} days of MBO data...")

    days = {}
    for f in files:
        date = f.stem.replace('_mbo_features', '').replace('mbo_features_', '')
        try:
            data = np.load(str(f))
            feats = data['mbo_features']
            days[date] = {
                'mid': feats[:, 0].astype(np.float32).copy(),
                'spread': feats[:, 1].astype(np.float32).copy(),
                'n_bars': len(feats),
            }
            del feats, data
        except Exception as e:
            print(f"  Skip {date}: {e}")

    all_dates = sorted(days.keys())
    print(f"Loaded {len(all_dates)} days")

    # Parse signal names
    sig_names_map = defaultdict(set)
    for f in SIG_DIR.glob('*.npz'):
        parts = f.stem.split('_')
        for i in range(len(parts)):
            if parts[i].startswith('2025-'):
                sig_name = '_'.join(parts[:i])
                date = '_'.join(parts[i:])
                sig_names_map[sig_name].add(date)
                break

    available_sigs = {k: v for k, v in sig_names_map.items() if len(v) >= 80}
    print(f"\nSignals with >=80 days: {len(available_sigs)}")
    for name, dates in sorted(available_sigs.items()):
        print(f"  {name}: {len(dates)} days")

    # Build composite signals
    print("\nBuilding composite signals (mean z-score, equal weight)...")
    composite = {}
    for date in all_dates:
        n_bars = days[date]['n_bars']
        sig_sum = np.zeros(n_bars, dtype=np.float64)
        n_sigs = 0

        for sig_name in available_sigs:
            path = SIG_DIR / f'{sig_name}_{date}.npz'
            if not path.exists():
                continue
            try:
                preds = np.load(str(path))['predictions']
            except Exception:
                continue
            if len(preds) != n_bars:
                preds = preds[:n_bars] if len(preds) > n_bars else np.pad(preds, (0, n_bars - len(preds)))
            sig_sum += preds.astype(np.float64)
            n_sigs += 1

        if n_sigs > 0:
            composite[date] = (sig_sum / n_sigs).astype(np.float32)

    print(f"Composite signals built for {len(composite)} days")

    # Split into train and test
    train_dates = all_dates[:TRAIN_DAYS]
    test_dates = all_dates[TRAIN_DAYS:]
    print(f"\nTRAIN: {len(train_dates)} days ({train_dates[0]} to {train_dates[-1]})")
    print(f"TEST:  {len(test_dates)} days ({test_dates[0]} to {test_dates[-1]})")

    # Phase 1: Optimize on TRAIN ONLY
    print(f"\n{'='*70}")
    print(f"PHASE 1: Parameter Optimization on Days 1-{TRAIN_DAYS} ONLY")
    print(f"{'='*70}")

    best_config = None
    best_train_pnl = -np.inf
    all_configs = []

    for thresh in THRESHOLDS:
        for hold_s in HOLDS:
            hold_bars = hold_s * BARS_SEC
            total_pnl = 0.0
            total_trades = 0
            total_wins = 0
            day_pnls = []

            for date in train_dates:
                if date not in composite:
                    day_pnls.append(0.0)
                    continue
                pnl, trades, wins = sim_day(
                    days[date]['mid'], days[date]['spread'],
                    composite[date], thresh, hold_bars, TRAIL, COOLDOWN
                )
                total_pnl += pnl
                total_trades += trades
                total_wins += wins
                day_pnls.append(pnl)

            avg_daily = np.mean(day_pnls) if day_pnls else 0
            std_daily = np.std(day_pnls) if day_pnls else 1
            sharpe = (avg_daily / std_daily * np.sqrt(252)) if std_daily > 0 else 0
            win_rate = total_wins / total_trades if total_trades > 0 else 0
            pct_pos = np.mean([p > 0 for p in day_pnls if p != 0]) * 100 if any(p != 0 for p in day_pnls) else 0

            config = {
                'thresh': thresh, 'hold_s': hold_s,
                'train_pnl': round(total_pnl), 'trades': total_trades,
                'sharpe': round(sharpe, 2), 'win_rate': round(win_rate, 3),
                'pct_pos_days': round(pct_pos, 1),
            }
            all_configs.append(config)

            if total_pnl > best_train_pnl:
                best_train_pnl = total_pnl
                best_config = config

            print(f"  t={thresh:.1f} h={hold_s:5d}s: ${total_pnl:+9,.0f} | "
                  f"{total_trades:4d} trades | Sharpe={sharpe:+.2f} | "
                  f"WR={win_rate:.1%} | {pct_pos:.0f}% days +")

    print(f"\n  BEST TRAIN CONFIG: t={best_config['thresh']}, h={best_config['hold_s']}s")
    print(f"  TRAIN PnL: ${best_config['train_pnl']:+,}")
    print(f"  TRAIN Sharpe: {best_config['sharpe']:+.2f}")

    # Count how many configs are positive
    pos_configs = sum(1 for c in all_configs if c['train_pnl'] > 0)
    print(f"\n  Positive configs: {pos_configs}/{len(all_configs)}")

    # Phase 2: TRUE OOS TEST with FIXED best params
    print(f"\n{'='*70}")
    print(f"PHASE 2: TRUE OOS TEST on Days {TRAIN_DAYS+1}-{len(all_dates)}")
    print(f"  Config FIXED: t={best_config['thresh']}, h={best_config['hold_s']}s")
    print(f"{'='*70}")

    thresh = best_config['thresh']
    hold_bars = best_config['hold_s'] * BARS_SEC
    oos_total_pnl = 0.0
    oos_total_trades = 0
    oos_total_wins = 0
    oos_day_pnls = []

    for date in test_dates:
        if date not in composite:
            oos_day_pnls.append(0.0)
            continue
        pnl, trades, wins = sim_day(
            days[date]['mid'], days[date]['spread'],
            composite[date], thresh, hold_bars, TRAIL, COOLDOWN
        )
        oos_total_pnl += pnl
        oos_total_trades += trades
        oos_total_wins += wins
        oos_day_pnls.append(pnl)

        if trades > 0:
            print(f"  {date}: ${pnl:+8,.0f} | {trades:3d} trades | WR={wins/trades:.0%}")

    avg_oos = np.mean(oos_day_pnls) if oos_day_pnls else 0
    std_oos = np.std(oos_day_pnls) if oos_day_pnls else 1
    sharpe_oos = (avg_oos / std_oos * np.sqrt(252)) if std_oos > 0 else 0
    wr_oos = oos_total_wins / oos_total_trades if oos_total_trades > 0 else 0
    pct_pos_oos = np.mean([p > 0 for p in oos_day_pnls if p != 0]) * 100 if any(p != 0 for p in oos_day_pnls) else 0

    # Monthly breakdown
    monthly = defaultdict(lambda: {'pnl': 0, 'trades': 0, 'days': 0})
    for date, pnl in zip(test_dates, oos_day_pnls):
        m = date[:7]
        monthly[m]['pnl'] += pnl
        monthly[m]['trades'] += 1
        monthly[m]['days'] += 1

    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"\n  TRAIN (days 1-{TRAIN_DAYS}):")
    print(f"    PnL: ${best_train_pnl:+,.0f}")
    print(f"    Sharpe: {best_config['sharpe']:+.2f}")
    print(f"    Trades: {best_config['trades']}")
    print(f"    Config: t={best_config['thresh']}, h={best_config['hold_s']}s")

    print(f"\n  TRUE OOS (days {TRAIN_DAYS+1}-{len(all_dates)}):")
    print(f"    PnL: ${oos_total_pnl:+,.0f}")
    print(f"    Sharpe: {sharpe_oos:+.2f}")
    print(f"    Trades: {oos_total_trades}")
    print(f"    Win rate: {wr_oos:.1%}")
    print(f"    Pct positive days: {pct_pos_oos:.0f}%")

    print(f"\n  OOS Monthly:")
    for m in sorted(monthly.keys()):
        d = monthly[m]
        print(f"    {m}: ${d['pnl']:+8,.0f} ({d['days']} days)")

    # Verdict
    if oos_total_pnl > 0 and sharpe_oos > 1.0:
        print(f"\n  >>> STRATEGY SURVIVES OOS! Sharpe={sharpe_oos:.2f}")
    elif oos_total_pnl > 0:
        print(f"\n  >>> MARGINALLY POSITIVE OOS. Sharpe={sharpe_oos:.2f} — weak but alive.")
    else:
        print(f"\n  >>> STRATEGY DEAD IN TRUE OOS. The prior results were in-sample optimized.")

    # Save results
    results = {
        'methodology': 'true_oos_holdout',
        'train_days': TRAIN_DAYS,
        'test_days': len(test_dates),
        'total_signals': len(available_sigs),
        'signal_names': sorted(available_sigs.keys()),
        'train': {
            'pnl': best_train_pnl,
            'sharpe': best_config['sharpe'],
            'trades': best_config['trades'],
            'best_config': best_config,
        },
        'oos': {
            'pnl': oos_total_pnl,
            'sharpe': round(sharpe_oos, 2),
            'trades': oos_total_trades,
            'win_rate': round(wr_oos, 3),
            'pct_positive_days': round(pct_pos_oos, 1),
            'monthly': {m: {'pnl': round(d['pnl']), 'days': d['days']} for m, d in monthly.items()},
        },
        'all_train_configs': all_configs,
        'oos_day_pnls': [round(p) for p in oos_day_pnls],
        'elapsed_s': round(time.time() - t0, 1),
    }

    out_path = RESULTS_DIR / f'true_oos_holdout_local_{len(available_sigs)}sig.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved: {out_path}")
    print(f"  Elapsed: {time.time() - t0:.0f}s")


if __name__ == '__main__':
    main()
