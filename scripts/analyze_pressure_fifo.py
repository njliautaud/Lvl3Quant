#!/usr/bin/env python3
"""
Analyze pressure FIFO sweep results — BUY-only filter.
The fill_sim_cli runs both sides, but our profitable config is BUY-only afternoon.
This script filters to buy trades only and computes proper metrics.
"""

import json, glob, sys
import numpy as np
from pathlib import Path
from collections import defaultdict

BASE = Path("/home/nick/Lvl3Quant/output/pressure_fifo_sweep")

def analyze_config(config_dir, buy_only=True):
    """Analyze a single config's results across all dates."""
    files = sorted(glob.glob(str(config_dir / "*.json")))
    if not files:
        return None

    total_pnl = 0
    total_trades = 0
    total_wins = 0
    gross_win = 0
    gross_loss = 0
    daily_pnl = {}
    exit_reasons = defaultdict(int)
    hold_secs = []

    for f in files:
        try:
            with open(f) as fh:
                d = json.load(fh)
        except:
            continue

        date = Path(f).stem
        trades = d.get('trades', [])

        if buy_only:
            trades = [t for t in trades if t.get('side') == 'BUY']

        day_pnl = 0
        for t in trades:
            pnl = t['pnl_ticks']
            total_pnl += pnl
            total_trades += 1
            day_pnl += pnl

            if pnl > 0:
                total_wins += 1
                gross_win += pnl
            else:
                gross_loss += abs(pnl)

            reason = t.get('exit_reason', 'Unknown')
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

            if 'hold_duration_ns' in t:
                hold_secs.append(t['hold_duration_ns'] / 1e9)

        if trades:  # only count days with trades
            daily_pnl[date] = day_pnl

    if total_trades == 0:
        return None

    pf = gross_win / gross_loss if gross_loss > 0 else 999
    wr = total_wins / total_trades

    daily_vals = list(daily_pnl.values())
    sharpe = np.mean(daily_vals) / np.std(daily_vals) * np.sqrt(252) if len(daily_vals) > 1 and np.std(daily_vals) > 0 else 0
    neg = [v for v in daily_vals if v < 0]
    sortino = np.mean(daily_vals) / np.std(neg) * np.sqrt(252) if neg and np.std(neg) > 0 else 0

    march = sum(v for k,v in daily_pnl.items() if k.startswith('202603'))
    april = sum(v for k,v in daily_pnl.items() if k.startswith('202604'))
    green = sum(1 for v in daily_vals if v > 0)
    red = sum(1 for v in daily_vals if v <= 0)

    avg_hold = np.mean(hold_secs) if hold_secs else 0

    return {
        'pf': round(pf, 4),
        'wr': round(wr, 4),
        'net_ticks': round(total_pnl, 1),
        'sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2),
        'trades': total_trades,
        'trades_per_day': round(total_trades / max(len(daily_pnl), 1), 1),
        'green_days': green,
        'red_days': red,
        'march_ticks': round(march, 1),
        'april_ticks': round(april, 1),
        'exit_reasons': dict(exit_reasons),
        'avg_hold_sec': round(avg_hold, 1),
        'daily_pnl': {k: round(v, 1) for k, v in sorted(daily_pnl.items())},
    }


def main():
    configs = sorted([d for d in BASE.iterdir() if d.is_dir()])
    if not configs:
        print("No config directories found")
        return

    print(f"Found {len(configs)} configs\n")
    print("=" * 80)
    print("BUY-ONLY AFTERNOON RESULTS (FIFO-validated)")
    print("=" * 80)

    results = []
    for config_dir in configs:
        name = config_dir.name
        r = analyze_config(config_dir, buy_only=True)
        if r:
            r['name'] = name
            results.append(r)
            print(f"\n  {name}:")
            print(f"    PF={r['pf']:.3f}  WR={r['wr']:.1%}  Net={r['net_ticks']:.0f}t  "
                  f"Sharpe={r['sharpe']:.1f}  Sortino={r['sortino']:.1f}")
            print(f"    Trades={r['trades']} ({r['trades_per_day']}/day)  "
                  f"Green={r['green_days']}/Red={r['red_days']}  "
                  f"March={r['march_ticks']:.0f}t/April={r['april_ticks']:.0f}t")
            print(f"    AvgHold={r['avg_hold_sec']:.0f}s  Exits={r['exit_reasons']}")

    # Sort by PF
    results.sort(key=lambda x: x['pf'], reverse=True)

    print("\n" + "=" * 80)
    print("RANKING BY PF (buy-only)")
    print("=" * 80)
    for i, r in enumerate(results):
        marker = " ★" if r['name'] == 'conv0_mag0.0' else ""
        print(f"  {i+1}. {r['name']}: PF={r['pf']:.3f} WR={r['wr']:.1%} "
              f"Net={r['net_ticks']:.0f}t Sharpe={r['sharpe']:.1f} "
              f"March={r['march_ticks']:.0f}/April={r['april_ticks']:.0f}{marker}")

    # Save
    outfile = BASE / "buy_only_analysis.json"
    with open(outfile, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outfile}")


if __name__ == '__main__':
    main()
