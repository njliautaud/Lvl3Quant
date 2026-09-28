#!/usr/bin/env python3
"""Aggregate Card 3 screening results from existing JSON files."""
import json
import os
import sys
import statistics
from pathlib import Path

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/data/processed/card3_screening'

TESTED = {
    'book_predstdExit_conv2.5_vol70',
    'book_predstdExit_conv1.5_vol50',
    'mom_emaExit_conv0.3_ethr0.0_vol70',
}

DATES = [
    '2025-12-05', '2025-12-10', '2025-12-16',
    '2026-01-02', '2026-01-07', '2026-01-14', '2026-01-22',
    '2026-02-03', '2026-02-10', '2026-02-18',
]

def main():
    # Read all result files
    results_by_type = {}
    for f in os.listdir(OUTPUT_DIR):
        if not f.endswith('.json') or f == 'screening_summary.json':
            continue
        # filename: {pred_type}_{date}.json
        # But pred_type itself contains underscores, so parse from the end
        base = f.replace('.json', '')
        # Date is last 10 chars: YYYY-MM-DD
        date = base[-10:]
        pred_type = base[:-11]  # remove _YYYY-MM-DD

        if pred_type in TESTED:
            continue

        try:
            with open(os.path.join(OUTPUT_DIR, f)) as fh:
                data = json.load(fh)
        except:
            continue

        if pred_type not in results_by_type:
            results_by_type[pred_type] = {}
        results_by_type[pred_type][date] = data

    # Aggregate
    agg = {}
    for pred_type, date_results in results_by_type.items():
        total_pnl = 0
        total_trades = 0
        total_wins = 0
        total_gross_profit = 0
        total_gross_loss = 0
        dates_with_data = 0
        pnls = []

        for date, res in date_results.items():
            if res is None:
                continue
            pnl_dollars = res.get('total_pnl_dollars', 0) or 0
            pnl = pnl_dollars / 12.5
            trades = res.get('total_trades', 0) or 0
            wr_day = res.get('win_rate', 0) or 0
            wins = int(round(wr_day * trades)) if trades > 0 else 0
            pf_day = res.get('profit_factor', None)
            if pf_day is None:
                pf_day = 0

            if pf_day > 1 and pnl > 0:
                gross_loss = pnl / (pf_day - 1)
                gross_profit = pf_day * gross_loss
            elif 0 < pf_day < 1 and pnl < 0:
                gross_loss = abs(pnl) / (1 - pf_day)
                gross_profit = pf_day * gross_loss
            else:
                gross_profit = max(0, pnl)
                gross_loss = max(0, -pnl)

            total_pnl += pnl
            total_trades += trades
            total_wins += wins
            total_gross_profit += gross_profit
            total_gross_loss += gross_loss
            dates_with_data += 1
            pnls.append(pnl)

        if dates_with_data == 0 or total_trades == 0:
            continue

        wr = total_wins / total_trades * 100 if total_trades > 0 else 0
        pf = total_gross_profit / total_gross_loss if total_gross_loss > 0 else float('inf')

        if len(pnls) > 1:
            mean_pnl = statistics.mean(pnls)
            std_pnl = statistics.stdev(pnls)
            sharpe = (mean_pnl / std_pnl * (252**0.5)) if std_pnl > 0 else 0
        else:
            mean_pnl = pnls[0] if pnls else 0
            sharpe = 0

        agg[pred_type] = {
            'total_pnl_ticks': round(total_pnl, 1),
            'total_pnl_usd': round(total_pnl * 12.5, 2),
            'total_trades': total_trades,
            'trades_per_day': round(total_trades / dates_with_data, 1),
            'win_rate': round(wr, 1),
            'profit_factor': round(pf, 3),
            'sharpe_proxy': round(sharpe, 2),
            'dates_with_data': dates_with_data,
            'mean_pnl_per_day': round(mean_pnl, 1),
        }

    # Save
    with open(os.path.join(OUTPUT_DIR, 'screening_summary.json'), 'w') as f:
        json.dump(agg, f, indent=2)

    ranked = sorted(agg.items(), key=lambda x: x[1]['sharpe_proxy'], reverse=True)

    print(f'===== CARD 3 SCREENING RESULTS =====')
    print(f'Types with ANY trades: {len(agg)}')

    viable = [(k, v) for k, v in ranked if v['total_pnl_ticks'] > 0 and v['total_trades'] > 20 and v['profit_factor'] > 1.2]
    print(f'Viable (PnL>0, trades>20, PF>1.2): {len(viable)}')

    print(f'\n--- TOP 25 by Sharpe (all with trades) ---')
    header = f'{"Rank":>4} {"Prediction Type":<50} {"PnL(t)":>8} {"PnL($)":>10} {"Trades":>7} {"T/Day":>6} {"WR%":>6} {"PF":>7} {"Sharpe":>7}'
    print(header)
    print('-' * len(header))
    for i, (pred_type, stats) in enumerate(ranked[:25]):
        print(f'{i+1:>4} {pred_type:<50} {stats["total_pnl_ticks"]:>8.1f} {stats["total_pnl_usd"]:>10.0f} {stats["total_trades"]:>7} {stats["trades_per_day"]:>6.1f} {stats["win_rate"]:>6.1f} {stats["profit_factor"]:>7.3f} {stats["sharpe_proxy"]:>7.2f}')

    print(f'\n--- VIABLE CANDIDATES (PnL>0, trades>20, PF>1.2) ---')
    print(header)
    print('-' * len(header))
    for i, (pred_type, stats) in enumerate(viable[:25]):
        print(f'{i+1:>4} {pred_type:<50} {stats["total_pnl_ticks"]:>8.1f} {stats["total_pnl_usd"]:>10.0f} {stats["total_trades"]:>7} {stats["trades_per_day"]:>6.1f} {stats["win_rate"]:>6.1f} {stats["profit_factor"]:>7.3f} {stats["sharpe_proxy"]:>7.2f}')

    # Diversity focus: show best from each non-book family
    print(f'\n--- BEST PER FAMILY (Card 3 diversity focus) ---')
    families = {}
    for pred_type, stats in ranked:
        parts = pred_type.split('_')
        family = parts[0] + '_' + parts[1]
        if family not in families:
            families[family] = []
        families[family].append((pred_type, stats))

    for family, members in sorted(families.items()):
        profitable = [m for m in members if m[1]['total_pnl_ticks'] > 0]
        best = max(members, key=lambda x: x[1]['sharpe_proxy']) if members else None
        if best:
            s = best[1]
            marker = ' ***' if s['total_pnl_ticks'] > 0 and s['profit_factor'] > 1.2 and s['total_trades'] > 20 else ''
            print(f'  {family}: {len(profitable)}/{len(members)} profitable | Best: {best[0]}')
            print(f'    PnL: {s["total_pnl_ticks"]:.1f}t (${s["total_pnl_usd"]:.0f}) | {s["total_trades"]} trades ({s["trades_per_day"]:.1f}/d) | WR {s["win_rate"]:.1f}% | PF {s["profit_factor"]:.3f} | Sharpe {s["sharpe_proxy"]:.2f}{marker}')

    # Bottom 10 (worst performers)
    print(f'\n--- BOTTOM 10 (worst) ---')
    for i, (pred_type, stats) in enumerate(ranked[-10:]):
        print(f'{len(ranked)-9+i:>4} {pred_type:<50} {stats["total_pnl_ticks"]:>8.1f} {stats["total_pnl_usd"]:>10.0f} {stats["total_trades"]:>7} {stats["trades_per_day"]:>6.1f} {stats["win_rate"]:>6.1f} {stats["profit_factor"]:>7.3f} {stats["sharpe_proxy"]:>7.2f}')

if __name__ == '__main__':
    main()
