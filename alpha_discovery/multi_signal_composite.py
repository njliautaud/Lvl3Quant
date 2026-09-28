"""
Multi-Signal Composite Market Order Test
=========================================
Combines ALL raw microstructure signals into a single composite signal.
If individual signals have weak, uncorrelated edges, the combination should be stronger.

Two approaches:
1. Mean z-score: average all signals (equal weight, no optimization)
2. Vote: count how many signals agree on direction (sign agreement)

Tests with market orders on 100 OOS days with first/last half split.
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

THRESHOLDS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
HOLDS = [50, 100, 300, 600, 1200, 3000]  # 5s to 300s
TRAILS = [0, 4, 8]
COOLDOWNS = [10, 50]


def sim_day(mid, spread, preds, thresh, hold_bars, trail, cooldown=10):
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
            if direction == 1:
                unr = (mid[i] - entry_price) / TICK
            else:
                unr = (entry_price - mid[i]) / TICK
            peak = max(peak, unr)
            if (i - entry_bar >= hold_bars) or (trail > 0 and (peak - unr) >= trail):
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
            peak = 0.0

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


def main():
    print("=" * 70)
    print("Multi-Signal Composite Market Order Test")
    print("=" * 70)

    # Load mid/spread
    print("\nLoading mid/spread data...")
    days = {}
    for f in sorted(FEAT_CACHE.glob('*_mbo_features.npz')):
        date = f.stem.replace('_mbo_features', '')
        data = np.load(str(f))
        feats = data['mbo_features']
        days[date] = {
            'mid': feats[:, 0].astype(np.float32).copy(),
            'spread': feats[:, 1].astype(np.float32).copy(),
            'n_bars': len(feats),
        }
        del feats, data

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

    # Filter to signals available for most dates
    available_sigs = {k: v for k, v in sig_names_map.items() if len(v) >= 80}
    print(f"Signals with >=80 days: {len(available_sigs)}")
    for name, dates in sorted(available_sigs.items()):
        print(f"  {name}: {len(dates)} days")

    # Build composite signals for each date
    print("\nBuilding composite signals...")
    composite_mean = {}  # mean z-score
    composite_vote = {}  # sign agreement count

    for date in all_dates:
        n_bars = days[date]['n_bars']
        sig_sum = np.zeros(n_bars, dtype=np.float64)
        vote_sum = np.zeros(n_bars, dtype=np.float64)
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
            vote_sum += np.sign(preds).astype(np.float64)
            n_sigs += 1

        if n_sigs > 0:
            composite_mean[date] = (sig_sum / n_sigs).astype(np.float32)
            composite_vote[date] = vote_sum.astype(np.float32)

    print(f"Composite signals built for {len(composite_mean)} days using {len(available_sigs)} signals")

    # Signal statistics
    for name, composite in [('MEAN', composite_mean), ('VOTE', composite_vote)]:
        sample = composite[all_dates[50]]
        nz = sample[sample != 0]
        print(f"\n{name} signal stats (sample day): "
              f"mean={nz.mean():.4f}, std={nz.std():.4f}, "
              f"range=[{nz.min():.3f}, {nz.max():.3f}]")

    # Test both composites
    mid_idx = len(all_dates) // 2

    for comp_name, composite in [('MEAN_ZSCORE', composite_mean), ('VOTE_COUNT', composite_vote)]:
        print(f"\n{'='*70}")
        print(f"TESTING: {comp_name}")
        print(f"{'='*70}")

        best_pnl = -999999
        best_results = None
        all_results = []

        for thresh in THRESHOLDS:
            for hold in HOLDS:
                for trail in TRAILS:
                    for cd in COOLDOWNS:
                        h1_pnl = h2_pnl = 0
                        tot_tr = tot_w = 0
                        day_pnls = []

                        for i, date in enumerate(all_dates):
                            if date not in composite:
                                day_pnls.append(0)
                                continue
                            p, t, w = sim_day(
                                days[date]['mid'], days[date]['spread'],
                                composite[date], thresh, hold, trail, cd
                            )
                            day_pnls.append(p)
                            if i < mid_idx:
                                h1_pnl += p
                            else:
                                h2_pnl += p
                            tot_tr += t
                            tot_w += w

                        total = h1_pnl + h2_pnl
                        both_pos = h1_pnl > 0 and h2_pnl > 0
                        arr = np.array(day_pnls)
                        std = float(arr.std()) if len(arr) > 1 else 1.0
                        sharpe = float(arr.mean() / max(std, 0.01) * np.sqrt(252))

                        result = {
                            'config': f"t{thresh}_h{hold//BARS_SEC}s_tr{trail}_cd{cd}",
                            'total_pnl': total, 'h1': h1_pnl, 'h2': h2_pnl,
                            'sharpe': sharpe, 'trades': tot_tr,
                            'win': tot_w / max(tot_tr, 1),
                            'both_positive': both_pos,
                            'profit_days': int((arr > 0).sum()),
                        }
                        all_results.append(result)

                        if total > best_pnl:
                            best_pnl = total
                            best_results = result

        # Sort and report
        all_results.sort(key=lambda x: x['total_pnl'], reverse=True)

        print(f"\nTOP 15 COMBOS:")
        print(f"{'Config':<30} {'Sharpe':>7} {'PnL':>10} {'H1':>8} {'H2':>8} "
              f"{'Trades':>7} {'Win%':>6} {'Days+':>6} {'Both':>5}")
        print("-" * 100)
        for r in all_results[:15]:
            flag = "YES" if r['both_positive'] else ""
            print(f"{r['config']:<30} {r['sharpe']:>+7.2f} ${r['total_pnl']:>+9,.0f} "
                  f"${r['h1']:>+7,.0f} ${r['h2']:>+7,.0f} {r['trades']:>7} "
                  f"{r['win']:>5.1%} {r['profit_days']:>3}/100 {flag:>5}")

        both_pos_count = sum(1 for r in all_results if r['both_positive'] and r['total_pnl'] > 0)
        profitable = sum(1 for r in all_results if r['total_pnl'] > 0)
        print(f"\nProfitable: {profitable}/{len(all_results)}")
        print(f"Both halves positive: {both_pos_count}/{len(all_results)}")

        if both_pos_count > 0:
            print("\nBOTH-POSITIVE COMBOS:")
            for r in all_results:
                if r['both_positive'] and r['total_pnl'] > 0:
                    print(f"  {r['config']:<30} +${r['total_pnl']:>9,.0f}  "
                          f"H1=+${r['h1']:>7,.0f}  H2=+${r['h2']:>7,.0f}  "
                          f"Sharpe={r['sharpe']:+.2f}  trades={r['trades']}")

    # Save
    ts = time.strftime('%Y%m%d_%H%M%S')
    out = RESULTS_DIR / f'multi_signal_composite_{ts}.json'
    with open(out, 'w') as f:
        json.dump({
            'n_signals': len(available_sigs),
            'signal_names': list(available_sigs.keys()),
            'n_days': len(all_dates),
        }, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
