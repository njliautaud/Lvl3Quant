"""
TRUE Out-of-Sample Composite Strategy Holdout Test
====================================================
Proper OOS protocol:
  - Days 1-40: parameter optimization (threshold + hold time grid search)
  - Days 41-100: TRUE OOS test with fixed best config

This test avoids the in-sample bias of the prior composite validation
which searched over ALL 100 days to pick threshold=3.5 / hold=900s.

Signal: composite_mean_zscore_9sig (mean of 9+ signals, equal weight)
Execution: Python market-order sim (Rust fill_sim called separately if needed)

Grid search (days 1-40 ONLY):
  thresholds: [2.0, 2.5, 3.0, 3.5, 4.0]
  hold times: [300s, 600s, 900s, 1200s] (matching prior test)

Output:
  1. Best config from training set (days 1-40)
  2. TRUE OOS result on holdout set (days 41-100)
  3. Comparison to highlight any performance gap
"""

import sys
import os
import json
import time
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(line_buffering=True)

TICK = 0.25
TICK_VAL = 12.50
COMM_TICKS = 3.00 / TICK_VAL   # 0.24 ticks
BARS_SEC = 10                   # 100ms bars -> 10 bars per second

LVL3_ROOT = Path('/home/jupiter/lvl3quant')
FEAT_CACHE = LVL3_ROOT / 'data' / 'processed' / 'mbo_features_cache'
SIG_DIR    = LVL3_ROOT / 'data' / 'processed' / 'signal_predictions'
RESULTS_DIR = LVL3_ROOT / 'alpha_discovery' / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [2.0, 2.5, 3.0, 3.5, 4.0]
HOLDS_SEC  = [300, 600, 900, 1200]   # seconds
COOLDOWN   = 50   # bars (5 seconds) — fixed, not searched
TRAIN_DAYS = 40   # days 1-40 are training set
SPLIT_IDX  = TRAIN_DAYS  # index: [0..39] train, [40..99] OOS


def sim_day(mid, spread, preds, thresh, hold_bars, cooldown=COOLDOWN):
    """Simulate market orders for one day. Returns (pnl_dollars, trades, wins)."""
    n = min(len(mid), len(preds))
    total_pnl = 0.0
    trades = 0
    wins = 0
    last_exit = -cooldown
    in_pos = False
    entry_price = 0.0
    direction = 0
    entry_bar = 0

    for i in range(n):
        if in_pos:
            if i - entry_bar >= hold_bars:
                if direction == 1:
                    unr = (mid[i] - entry_price) / TICK
                else:
                    unr = (entry_price - mid[i]) / TICK
                pnl = unr - spread[i] / 2.0 / TICK - COMM_TICKS
                total_pnl += pnl
                trades += 1
                wins += (1 if pnl > 0 else 0)
                in_pos = False
                last_exit = i
        elif (i - last_exit >= cooldown) and abs(preds[i]) > thresh:
            d = 1 if preds[i] > 0 else -1
            entry_price = mid[i] + d * spread[i] / 2.0
            direction = d
            in_pos = True
            entry_bar = i

    # Close any open position at end of day
    if in_pos:
        i = n - 1
        if direction == 1:
            unr = (mid[i] - entry_price) / TICK
        else:
            unr = (entry_price - mid[i]) / TICK
        pnl = unr - spread[i] / 2.0 / TICK - COMM_TICKS
        total_pnl += pnl
        trades += 1
        wins += (1 if pnl > 0 else 0)

    return total_pnl * TICK_VAL, trades, wins


def get_signal_names():
    """Find all signal names with >=80 days of predictions."""
    sig_names_map = defaultdict(set)
    for f in SIG_DIR.glob('*.npz'):
        parts = f.stem.split('_')
        for i in range(len(parts)):
            if parts[i].startswith('2025-'):
                sig_name = '_'.join(parts[:i])
                date = '_'.join(parts[i:])
                sig_names_map[sig_name].add(date)
                break
    available = {k: v for k, v in sig_names_map.items() if len(v) >= 80}
    return available


def load_mid_spread(date):
    """Load mid and spread arrays from mbo_features_cache."""
    path = FEAT_CACHE / f'{date}_mbo_features.npz'
    if not path.exists():
        return None, None
    data = np.load(str(path))
    feats = data['mbo_features']
    mid    = feats[:, 0].astype(np.float32)
    spread = feats[:, 1].astype(np.float32)
    del feats, data
    return mid, spread


def build_composite(date, sig_list, n_bars_hint=None):
    """Build mean z-score composite for one date. Returns array or None."""
    # First pass: determine n_bars
    if n_bars_hint is not None:
        n_bars = n_bars_hint
    else:
        for sig_name in sig_list:
            path = SIG_DIR / f'{sig_name}_{date}.npz'
            if path.exists():
                try:
                    p = np.load(str(path))['predictions']
                    n_bars = len(p)
                    del p
                    break
                except Exception:
                    continue
        else:
            return None

    sig_sum = np.zeros(n_bars, dtype=np.float64)
    n_sigs = 0

    for sig_name in sig_list:
        path = SIG_DIR / f'{sig_name}_{date}.npz'
        if not path.exists():
            continue
        try:
            preds = np.load(str(path))['predictions']
        except Exception:
            continue

        if len(preds) != n_bars:
            if len(preds) > n_bars:
                preds = preds[:n_bars]
            else:
                preds = np.pad(preds, (0, n_bars - len(preds)))

        sig_sum += preds.astype(np.float64)
        n_sigs += 1

    if n_sigs == 0:
        return None

    return (sig_sum / n_sigs).astype(np.float32)


def run_grid_search(days_subset, day_data):
    """
    Grid search over thresholds x hold_times on days_subset.
    day_data: dict {date -> {'mid': arr, 'spread': arr, 'composite': arr}}

    Returns list of results sorted by total_pnl DESC.
    """
    results = []

    for thresh in THRESHOLDS:
        for hold_sec in HOLDS_SEC:
            hold_bars = hold_sec * BARS_SEC

            total_pnl = 0.0
            total_trades = 0
            total_wins = 0
            day_pnls = []

            for date in days_subset:
                if date not in day_data:
                    day_pnls.append(0.0)
                    continue
                dd = day_data[date]
                p, t, w = sim_day(
                    dd['mid'], dd['spread'], dd['composite'],
                    thresh, hold_bars
                )
                day_pnls.append(p)
                total_pnl += p
                total_trades += t
                total_wins += w

            arr = np.array(day_pnls)
            std = float(arr.std()) if len(arr) > 1 else 1.0
            sharpe = float(arr.mean() / max(std, 0.01) * np.sqrt(252))

            results.append({
                'threshold': thresh,
                'hold_sec': hold_sec,
                'config': f't{thresh}_h{hold_sec}s',
                'total_pnl': round(total_pnl, 2),
                'sharpe': round(sharpe, 3),
                'trades': total_trades,
                'win_rate': round(total_wins / max(total_trades, 1), 4),
                'profit_days': int((arr > 0).sum()),
                'loss_days': int((arr < 0).sum()),
                'n_days': len(days_subset),
                'day_pnls': [round(x, 2) for x in day_pnls],
            })

    results.sort(key=lambda x: x['total_pnl'], reverse=True)
    return results


def main():
    t0 = time.time()
    print('=' * 70)
    print('TRUE OOS COMPOSITE HOLDOUT TEST')
    print(f'Training: days 1-{TRAIN_DAYS} | OOS: days {TRAIN_DAYS+1}-100')
    print('=' * 70)

    # Step 1: Load signal inventory
    print('\n[1/5] Loading signal inventory...')
    signal_names = get_signal_names()
    sig_list = sorted(signal_names.keys())
    print(f'  Signals with >=80 days: {len(sig_list)}')
    for name in sig_list:
        print(f'    {name}: {len(signal_names[name])} days')

    # Step 2: Get sorted dates from feature cache
    print('\n[2/5] Discovering dates...')
    feat_dates = sorted(
        f.stem.replace('_mbo_features', '')
        for f in FEAT_CACHE.glob('*_mbo_features.npz')
    )
    print(f'  Feature files: {len(feat_dates)} days ({feat_dates[0]} to {feat_dates[-1]})')

    if len(feat_dates) < TRAIN_DAYS + 1:
        print(f'ERROR: Need at least {TRAIN_DAYS + 1} days, got {len(feat_dates)}')
        sys.exit(1)

    train_dates = feat_dates[:TRAIN_DAYS]
    oos_dates   = feat_dates[TRAIN_DAYS:]
    all_dates   = feat_dates

    print(f'  Training set: {len(train_dates)} days ({train_dates[0]} to {train_dates[-1]})')
    print(f'  OOS holdout:  {len(oos_dates)} days ({oos_dates[0]} to {oos_dates[-1]})')

    # Step 3: Build composite signals for ALL dates
    print('\n[3/5] Building composite signals (all dates)...')
    day_data = {}
    skipped = 0

    for i, date in enumerate(all_dates):
        mid, spread = load_mid_spread(date)
        if mid is None:
            skipped += 1
            continue

        composite = build_composite(date, sig_list, n_bars_hint=len(mid))
        if composite is None:
            skipped += 1
            continue

        day_data[date] = {
            'mid': mid,
            'spread': spread,
            'composite': composite,
        }

        if (i + 1) % 20 == 0 or i == len(all_dates) - 1:
            print(f'  {i+1}/{len(all_dates)} dates loaded')

    print(f'  Loaded: {len(day_data)} days, skipped: {skipped}')

    # Sample composite stats
    sample_dates = [d for d in train_dates if d in day_data]
    if sample_dates:
        sample = day_data[sample_dates[len(sample_dates)//2]]['composite']
        nz = sample[sample != 0]
        if len(nz) > 0:
            print(f'  Composite sample stats: mean={nz.mean():.4f} std={nz.std():.4f} '
                  f'range=[{nz.min():.3f}, {nz.max():.3f}]')

    # Step 4: Grid search on TRAINING DAYS ONLY (1-40)
    print(f'\n[4/5] Grid search on training days (1-{TRAIN_DAYS})...')
    valid_train = [d for d in train_dates if d in day_data]
    print(f'  Valid training days: {len(valid_train)}/{len(train_dates)}')

    train_results = run_grid_search(valid_train, day_data)

    print(f'\n  TOP 10 CONFIGS (training days 1-{TRAIN_DAYS}):')
    print(f'  {"Config":<20} {"PnL":>10} {"Sharpe":>8} {"Trades":>7} {"Win%":>7} {"Days+":>7}')
    print('  ' + '-' * 65)
    for r in train_results[:10]:
        print(f'  {r["config"]:<20} ${r["total_pnl"]:>+9,.0f} {r["sharpe"]:>+8.3f} '
              f'{r["trades"]:>7} {r["win_rate"]:>7.1%} '
              f'{r["profit_days"]}/{r["n_days"]}')

    best = train_results[0]
    print(f'\n  BEST CONFIG (training): {best["config"]}')
    print(f'    Training PnL: ${best["total_pnl"]:+,.0f} over {len(valid_train)} days')
    print(f'    Training Sharpe: {best["sharpe"]:+.3f}')
    print(f'    Threshold: {best["threshold"]}, Hold: {best["hold_sec"]}s')

    # Step 5: Run BEST config on OOS days (41-100) — FIXED parameters
    print(f'\n[5/5] Running best config on OOS holdout (days {TRAIN_DAYS+1}-100)...')
    print(f'  Config: threshold={best["threshold"]}, hold={best["hold_sec"]}s')
    print(f'  Parameters are LOCKED from training set — no peeking at OOS data')

    valid_oos = [d for d in oos_dates if d in day_data]
    print(f'  Valid OOS days: {len(valid_oos)}/{len(oos_dates)}')

    hold_bars_best = best['hold_sec'] * BARS_SEC
    thresh_best = best['threshold']

    oos_total_pnl = 0.0
    oos_trades = 0
    oos_wins = 0
    oos_day_pnls = []
    oos_details = []

    for date in valid_oos:
        dd = day_data[date]
        p, t, w = sim_day(
            dd['mid'], dd['spread'], dd['composite'],
            thresh_best, hold_bars_best
        )
        oos_total_pnl += p
        oos_trades += t
        oos_wins += w
        oos_day_pnls.append(p)
        oos_details.append({'date': date, 'pnl': round(p, 2), 'trades': t, 'wins': w})

    oos_arr = np.array(oos_day_pnls)
    oos_sharpe = (
        float(oos_arr.mean() / oos_arr.std() * np.sqrt(252))
        if len(oos_arr) > 1 and oos_arr.std() > 0 else 0.0
    )
    oos_profit_days = int((oos_arr > 0).sum())
    oos_win_rate = oos_wins / max(oos_trades, 1)

    # Also run the best config on training set (for comparison)
    train_total_pnl = best['total_pnl']
    train_sharpe = best['sharpe']
    train_profit_days = best['profit_days']
    train_trades = best['trades']

    # Results output
    print('\n' + '=' * 70)
    print('TRUE OOS COMPOSITE HOLDOUT RESULTS')
    print('=' * 70)
    print(f'Best config (from training days 1-{TRAIN_DAYS}): {best["config"]}')
    print(f'  Threshold: {thresh_best}  |  Hold: {best["hold_sec"]}s')
    print()
    print(f'{"Metric":<30} {"Training (d1-40)":>18} {"TRUE OOS (d41-100)":>20}')
    print('-' * 70)
    print(f'{"Total PnL":<30} ${train_total_pnl:>+16,.0f} ${oos_total_pnl:>+18,.0f}')
    print(f'{"Annualized Sharpe":<30} {train_sharpe:>+17.3f} {oos_sharpe:>+19.3f}')
    print(f'{"Total Trades":<30} {train_trades:>18} {oos_trades:>20}')
    print(f'{"Win Rate":<30} {best["win_rate"]:>17.1%} {oos_win_rate:>19.1%}')
    print(f'{"Profitable Days":<30} {train_profit_days}/{len(valid_train):>14} '
          f'{oos_profit_days}/{len(valid_oos):>18}')
    print()

    verdict = ''
    if oos_total_pnl > 0 and oos_sharpe > 0.5:
        verdict = 'PASSES OOS — Strategy has real edge'
    elif oos_total_pnl > 0:
        verdict = 'MARGINAL — OOS positive but low Sharpe'
    else:
        verdict = 'DEAD — Strategy fails true OOS test (likely overfitted)'

    print(f'VERDICT: {verdict}')
    print()

    # OOS daily breakdown
    print(f'OOS Daily PnL breakdown (days {TRAIN_DAYS+1}-{TRAIN_DAYS+len(valid_oos)}):')
    cumulative = 0.0
    for i, row in enumerate(oos_details):
        cumulative += row['pnl']
        flag = '+' if row['pnl'] > 0 else ('-' if row['pnl'] < 0 else '.')
        print(f'  [{flag}] day{TRAIN_DAYS+i+1:>3} {row["date"]}: ${row["pnl"]:>+8,.0f}  '
              f'trades={row["trades"]}  cum=${cumulative:>+9,.0f}')

    # Compare all training configs for context
    print(f'\nAll training configs for context:')
    print(f'{"Config":<20} {"Train PnL":>12} {"Train Sharpe":>13}')
    print('-' * 47)
    for r in train_results:
        print(f'{r["config"]:<20} ${r["total_pnl"]:>+10,.0f} {r["sharpe"]:>+12.3f}')

    elapsed = time.time() - t0
    print(f'\nElapsed: {elapsed:.1f}s')

    # Save full results
    ts = time.strftime('%Y%m%d_%H%M%S')
    out_path = RESULTS_DIR / f'true_oos_composite_{ts}.json'
    result_data = {
        'timestamp': ts,
        'elapsed_sec': round(elapsed, 1),
        'protocol': {
            'train_days': len(valid_train),
            'oos_days': len(valid_oos),
            'train_range': [train_dates[0], train_dates[TRAIN_DAYS-1]],
            'oos_range': [oos_dates[0], oos_dates[-1]],
            'n_signals': len(sig_list),
            'signal_names': sig_list,
        },
        'best_config': {
            'threshold': thresh_best,
            'hold_sec': best['hold_sec'],
            'config_name': best['config'],
        },
        'training_results': {
            'total_pnl': train_total_pnl,
            'sharpe': train_sharpe,
            'trades': train_trades,
            'win_rate': best['win_rate'],
            'profit_days': train_profit_days,
            'n_days': len(valid_train),
        },
        'oos_results': {
            'total_pnl': round(oos_total_pnl, 2),
            'sharpe': round(oos_sharpe, 3),
            'trades': oos_trades,
            'win_rate': round(oos_win_rate, 4),
            'profit_days': oos_profit_days,
            'n_days': len(valid_oos),
            'daily_pnls': oos_day_pnls,
            'daily_details': oos_details,
        },
        'verdict': verdict,
        'all_train_configs': [
            {k: v for k, v in r.items() if k != 'day_pnls'}
            for r in train_results
        ],
    }

    with open(str(out_path), 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f'\nFull results saved: {out_path}')

    return result_data


if __name__ == '__main__':
    main()
