"""
Cross-Venue Backtest: ES Signal -> SPY Execution
=================================================
Uses ES microstructure signals but with SPY cost assumptions.

Cost comparison:
  ES:  spread=1tick ($12.50) + commission=$3.00 RT = $15.50/RT
  SPY: spread=$0.01 on ~$550 + commission=$0 (PFOF brokers) = ~$1.00/RT per 100 shares

We normalize to equivalent notional: 1 ES contract ≈ 5 SPY lots (500 shares)
  ES cost:  $15.50 per RT
  SPY cost: $5.00 per RT (5 × $1.00)  -> 3.1x cheaper

Also tests SPY options cost structure.

Runs BOTH the full-sample and TRUE OOS (days 1-40 train, 41-100 test).
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

BARS_SEC = 10

LVL3_ROOT = Path(__file__).parent.parent
FEAT_CACHE = LVL3_ROOT / 'data' / 'processed' / 'mbo_features_cache'
SIG_DIR = LVL3_ROOT / 'data' / 'processed' / 'signal_predictions'
RESULTS_DIR = LVL3_ROOT / 'alpha_discovery' / 'results'

# Cost structures (expressed in ES ticks for comparison)
TICK = 0.25
TICK_VAL = 12.50  # $ per tick for ES

COST_PROFILES = {
    'es_futures': {
        'name': 'ES Futures',
        'spread_ticks': 1.0,       # 1 tick spread
        'comm_ticks': 3.00 / TICK_VAL,  # $3.00 RT = 0.24 ticks
        'desc': 'spread=$12.50 + comm=$3.00 = $15.50/RT',
    },
    'spy_stock': {
        'name': 'SPY Stock (500 shares)',
        'spread_ticks': 0.01 / (TICK * 5),  # $0.01 SPY spread / ($1.25 per ES tick equivalent for 500 shares)
        'comm_ticks': 0.0,  # $0 commission at PFOF brokers
        'desc': 'spread=$0.01×500sh=$5.00, comm=$0 -> $5.00/RT equiv',
    },
    'spy_stock_100sh': {
        'name': 'SPY Stock (100 shares)',
        'spread_ticks': 0.01 / (TICK * 1),  # much smaller notional
        'comm_ticks': 0.0,
        'desc': 'spread=$0.01×100sh=$1.00, comm=$0 -> $1.00/RT equiv',
    },
    'spy_zero_cost': {
        'name': 'SPY Zero Cost (theoretical)',
        'spread_ticks': 0.0,
        'comm_ticks': 0.0,
        'desc': 'No costs — raw signal P&L',
    },
}

THRESHOLDS = [2.0, 2.5, 3.0, 3.5, 4.0]
HOLDS = [300, 600, 900, 1200]
COOLDOWN = 50
TRAIL = 0
TRAIN_DAYS = 40


def sim_day(mid, spread, preds, thresh, hold_bars, cost_spread_ticks, cost_comm_ticks, cooldown=50):
    """Market order sim with configurable costs."""
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
            elapsed = i - entry_bar
            if elapsed >= hold_bars:
                curr_pnl = direction * (mid[i] - entry_price) / TICK
                # Apply configurable costs
                pnl = curr_pnl - cost_comm_ticks - cost_spread_ticks
                total_pnl += pnl
                trades += 1
                if pnl > 0:
                    wins += 1
                in_pos = False
                last_exit = i

        elif i - last_exit >= cooldown and abs(preds[i]) > thresh:
            entry_price = mid[i]
            direction = 1 if preds[i] > 0 else -1
            entry_bar = i
            in_pos = True
            # Entry cost: cross the spread
            entry_price += direction * spread[i] / 2

    return total_pnl * TICK_VAL, trades, wins


def run_test(days, all_dates, composite, cost_profile, label=''):
    """Run full grid search and OOS test for a given cost profile."""
    cp = COST_PROFILES[cost_profile]
    spread_t = cp['spread_ticks']
    comm_t = cp['comm_ticks']

    train_dates = all_dates[:TRAIN_DAYS]
    test_dates = all_dates[TRAIN_DAYS:]

    # Phase 1: Optimize on train
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
                    composite[date], thresh, hold_bars, spread_t, comm_t, COOLDOWN
                )
                total_pnl += pnl
                total_trades += trades
                total_wins += wins
                day_pnls.append(pnl)

            std_daily = np.std(day_pnls) if day_pnls else 1
            avg_daily = np.mean(day_pnls)
            sharpe = (avg_daily / std_daily * np.sqrt(252)) if std_daily > 0 else 0

            config = {
                'thresh': thresh, 'hold_s': hold_s,
                'train_pnl': round(total_pnl), 'trades': total_trades,
                'sharpe': round(sharpe, 2),
            }
            all_configs.append(config)

            if total_pnl > best_train_pnl:
                best_train_pnl = total_pnl
                best_config = config

    # Phase 2: TRUE OOS with fixed best params
    thresh = best_config['thresh']
    hold_bars = best_config['hold_s'] * BARS_SEC
    oos_total_pnl = 0.0
    oos_total_trades = 0
    oos_total_wins = 0
    oos_day_pnls = []
    monthly = defaultdict(lambda: {'pnl': 0, 'trades': 0})

    for date in test_dates:
        if date not in composite:
            oos_day_pnls.append(0.0)
            continue
        pnl, trades, wins = sim_day(
            days[date]['mid'], days[date]['spread'],
            composite[date], thresh, hold_bars, spread_t, comm_t, COOLDOWN
        )
        oos_total_pnl += pnl
        oos_total_trades += trades
        oos_total_wins += wins
        oos_day_pnls.append(pnl)
        m = date[:7]
        monthly[m]['pnl'] += pnl
        monthly[m]['trades'] += trades

    std_oos = np.std(oos_day_pnls) if oos_day_pnls else 1
    avg_oos = np.mean(oos_day_pnls)
    sharpe_oos = (avg_oos / std_oos * np.sqrt(252)) if std_oos > 0 else 0
    wr_oos = oos_total_wins / oos_total_trades if oos_total_trades > 0 else 0

    # Also run full 100-day with best config (for comparison)
    full_pnl = 0.0
    full_trades = 0
    for date in all_dates:
        if date not in composite:
            continue
        pnl, trades, _ = sim_day(
            days[date]['mid'], days[date]['spread'],
            composite[date], thresh, hold_bars, spread_t, comm_t, COOLDOWN
        )
        full_pnl += pnl
        full_trades += trades

    pos_configs = sum(1 for c in all_configs if c['train_pnl'] > 0)

    return {
        'cost_profile': cost_profile,
        'cost_name': cp['name'],
        'cost_desc': cp['desc'],
        'best_config': best_config,
        'train_pnl': round(best_train_pnl),
        'train_sharpe': best_config['sharpe'],
        'oos_pnl': round(oos_total_pnl),
        'oos_sharpe': round(sharpe_oos, 2),
        'oos_trades': oos_total_trades,
        'oos_win_rate': round(wr_oos, 3),
        'oos_monthly': {m: round(d['pnl']) for m, d in sorted(monthly.items())},
        'full_100d_pnl': round(full_pnl),
        'full_100d_trades': full_trades,
        'pos_train_configs': f'{pos_configs}/{len(all_configs)}',
    }


def main():
    t0 = time.time()
    print("=" * 70)
    print("CROSS-VENUE BACKTEST: ES Signal -> Multiple Cost Structures")
    print("=" * 70)

    # Load features
    files = sorted(FEAT_CACHE.glob('*_mbo_features.npz'))[:100]
    print(f"\nLoading {len(files)} days...")

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

    # Build composite
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
    print(f"Signals: {len(available_sigs)} with >=80 days")

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

    print(f"Composite built for {len(composite)} days")

    # Run all cost profiles
    print(f"\n{'='*70}")
    print("RUNNING ALL COST PROFILES (Train days 1-40, OOS days 41-100)")
    print(f"{'='*70}\n")

    results = []
    for profile in COST_PROFILES:
        print(f"\n--- {COST_PROFILES[profile]['name']} ---")
        print(f"    {COST_PROFILES[profile]['desc']}")
        r = run_test(days, all_dates, composite, profile)
        results.append(r)

        print(f"    Best train config: t={r['best_config']['thresh']}, h={r['best_config']['hold_s']}s")
        print(f"    TRAIN: ${r['train_pnl']:+,} (Sharpe {r['train_sharpe']:+.2f}) [{r['pos_train_configs']} positive]")
        print(f"    OOS:   ${r['oos_pnl']:+,} (Sharpe {r['oos_sharpe']:+.2f}, {r['oos_trades']} trades, WR={r['oos_win_rate']:.1%})")
        print(f"    Full 100d: ${r['full_100d_pnl']:+,} ({r['full_100d_trades']} trades)")
        for m, pnl in sorted(r['oos_monthly'].items()):
            print(f"      {m}: ${pnl:+,}")

    # Summary table
    print(f"\n\n{'='*70}")
    print("SUMMARY COMPARISON")
    print(f"{'='*70}")
    print(f"{'Cost Profile':<30} {'Train':>10} {'OOS':>10} {'OOS Sharpe':>12} {'Config':<15}")
    print("-" * 80)
    for r in results:
        cfg = f"t={r['best_config']['thresh']}/h={r['best_config']['hold_s']}s"
        print(f"{r['cost_name']:<30} ${r['train_pnl']:>+8,} ${r['oos_pnl']:>+8,} {r['oos_sharpe']:>+10.2f}   {cfg:<15}")

    # Key question
    print(f"\n{'='*70}")
    spy_oos = next((r for r in results if r['cost_profile'] == 'spy_stock'), None)
    zero_oos = next((r for r in results if r['cost_profile'] == 'spy_zero_cost'), None)
    es_oos = next((r for r in results if r['cost_profile'] == 'es_futures'), None)

    if spy_oos and spy_oos['oos_pnl'] > 0:
        print(f">>> SPY EXECUTION MAKES THE STRATEGY PROFITABLE!")
        print(f">>> ES OOS: ${es_oos['oos_pnl']:+,} -> SPY OOS: ${spy_oos['oos_pnl']:+,}")
        print(f">>> The signal is real, ES costs just eat it. SPY is the answer.")
    elif zero_oos and zero_oos['oos_pnl'] > 0:
        print(f">>> Raw signal IS profitable (${zero_oos['oos_pnl']:+,} at zero cost)")
        print(f">>> But even SPY costs eat it (${spy_oos['oos_pnl']:+,})")
        print(f">>> Need even cheaper execution or bigger signal.")
    else:
        print(f">>> Signal is NOT profitable even at zero cost (${zero_oos['oos_pnl']:+,})")
        print(f">>> The edge doesn't exist in OOS regardless of execution venue.")

    # Save
    out_path = RESULTS_DIR / 'spy_cost_backtest_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {out_path}")
    print(f"Elapsed: {time.time() - t0:.0f}s")


if __name__ == '__main__':
    main()
