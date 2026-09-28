#!/usr/bin/env python3
"""Card 3 Screening Sweep - test all 102 untested prediction types on 10 OOT dates."""
import subprocess
import json
import os
import sys
import time
import statistics
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

PRED_DIR = '/home/jupiter/Lvl3Quant/data/processed/cnn_wf_stacked_predictions'
MBO_DIR = '/home/jupiter/Lvl3Quant/data/raw/mbo'
BINARY = '/home/jupiter/Lvl3Quant/rust_cache_builder/target/release/fill_sim_cli'
OUTPUT_DIR = '/home/jupiter/Lvl3Quant/data/processed/card3_screening'

# Already tested
TESTED = {
    'book_predstdExit_conv2.5_vol70',
    'book_predstdExit_conv1.5_vol50',
    'mom_emaExit_conv0.3_ethr0.0_vol70',
}

# 10 diverse OOT dates (weekdays with MBO data)
DATES = [
    '2025-12-05', '2025-12-10', '2025-12-16',
    '2026-01-02', '2026-01-07', '2026-01-14', '2026-01-22',
    '2026-02-03', '2026-02-10', '2026-02-18',
]

def get_all_pred_types():
    types = set()
    for f in os.listdir(PRED_DIR):
        if f.endswith('.npz'):
            # date is YYYY-MM-DD_, so skip first 11 chars
            rest = f[11:]
            pred_type = rest.replace('.npz', '')
            types.add(pred_type)
    return sorted(types)

def run_sim(pred_type, date):
    """Run fill_sim for one pred_type x date combo."""
    date_nodash = date.replace('-', '')
    pred_file = os.path.join(PRED_DIR, f'{date}_{pred_type}.npz')
    mbo_file = os.path.join(MBO_DIR, f'glbx-mdp3-{date_nodash}.mbo.dbn.zst')
    out_file = os.path.join(OUTPUT_DIR, f'{pred_type}_{date}.json')

    if not os.path.exists(pred_file):
        return None
    if not os.path.exists(mbo_file):
        return None
    if os.path.exists(out_file):
        try:
            with open(out_file) as f:
                return json.load(f)
        except:
            pass

    cmd = [
        BINARY,
        '--predictions', pred_file,
        '--mbo-file', mbo_file,
        '--output', out_file,
        '--signal-threshold', '0.1',
        '--hold-ms', '3600000',
        '--max-wait-bars', '50',
        '--latency-ms', '50',
        '--chase-entry',
        '--chase-max-ticks', '1',
        '--chase-max-reprices', '3',
        '--take-profit-ticks', '8',
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if os.path.exists(out_file):
            with open(out_file) as f:
                return json.load(f)
        else:
            return None
    except Exception as e:
        return None

def aggregate_results(results_by_type):
    """Aggregate per prediction type."""
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
            # fill_sim_cli output format: root level keys
            pnl_dollars = res.get('total_pnl_dollars', 0)
            pnl = pnl_dollars / 12.5  # convert to ticks
            trades = res.get('total_trades', 0)
            wr_day = res.get('win_rate', 0)
            wins = int(round(wr_day * trades)) if trades > 0 else 0
            pf_day = res.get('profit_factor', 0)
            # Reconstruct gross profit/loss from PF
            # PF = GP / GL, and PnL = GP - GL
            # GP = PF * GL, PnL = PF*GL - GL = GL*(PF-1), so GL = PnL/(PF-1) if PF != 1
            if pf_day > 1 and pnl > 0:
                gross_loss = pnl / (pf_day - 1)
                gross_profit = pf_day * gross_loss
            elif pf_day < 1 and pf_day > 0 and pnl < 0:
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
            'daily_pnls': pnls,
        }
    return agg

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_types = get_all_pred_types()
    untested = [t for t in all_types if t not in TESTED]
    print(f'Total prediction types: {len(all_types)}')
    print(f'Already tested: {len(TESTED)}')
    print(f'Untested to screen: {len(untested)}')

    # Build job list
    jobs = []
    for pred_type in untested:
        for date in DATES:
            jobs.append((pred_type, date))

    print(f'Total jobs: {len(jobs)}')
    print(f'Using 14 workers...')
    sys.stdout.flush()

    results_by_type = {t: {} for t in untested}
    completed = 0
    start = time.time()

    with ProcessPoolExecutor(max_workers=14) as executor:
        futures = {}
        for pred_type, date in jobs:
            f = executor.submit(run_sim, pred_type, date)
            futures[f] = (pred_type, date)

        for f in as_completed(futures):
            pred_type, date = futures[f]
            try:
                result = f.result()
                results_by_type[pred_type][date] = result
            except Exception as e:
                results_by_type[pred_type][date] = None

            completed += 1
            if completed % 100 == 0:
                elapsed = time.time() - start
                rate = completed / elapsed
                remaining = (len(jobs) - completed) / rate if rate > 0 else 0
                print(f'  {completed}/{len(jobs)} done ({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)')
                sys.stdout.flush()

    elapsed = time.time() - start
    print(f'All {len(jobs)} jobs completed in {elapsed:.0f}s')
    sys.stdout.flush()

    agg = aggregate_results(results_by_type)

    # Save full results
    agg_clean = {}
    for k, v in agg.items():
        agg_clean[k] = {kk: vv for kk, vv in v.items() if kk != 'daily_pnls'}

    with open(os.path.join(OUTPUT_DIR, 'screening_summary.json'), 'w') as f:
        json.dump(agg_clean, f, indent=2)

    ranked = sorted(agg.items(), key=lambda x: x[1]['sharpe_proxy'], reverse=True)

    print(f'\n===== CARD 3 SCREENING RESULTS =====')
    print(f'Tested {len(untested)} prediction types x {len(DATES)} dates')
    print(f'Types with ANY trades: {len(agg)}')

    viable = [(k, v) for k, v in ranked if v['total_pnl_ticks'] > 0 and v['total_trades'] > 20 and v['profit_factor'] > 1.2]
    print(f'Viable (PnL>0, trades>20, PF>1.2): {len(viable)}')

    print(f'\n--- TOP 20 by Sharpe (all with trades) ---')
    print(f'{"Rank":>4} {"Prediction Type":<50} {"PnL(t)":>8} {"PnL($)":>10} {"Trades":>7} {"T/Day":>6} {"WR%":>6} {"PF":>7} {"Sharpe":>7}')
    print('-' * 110)
    for i, (pred_type, stats) in enumerate(ranked[:20]):
        print(f'{i+1:>4} {pred_type:<50} {stats["total_pnl_ticks"]:>8.1f} {stats["total_pnl_usd"]:>10.0f} {stats["total_trades"]:>7} {stats["trades_per_day"]:>6.1f} {stats["win_rate"]:>6.1f} {stats["profit_factor"]:>7.3f} {stats["sharpe_proxy"]:>7.2f}')

    print(f'\n--- VIABLE CANDIDATES (PnL>0, trades>20, PF>1.2) ---')
    print(f'{"Rank":>4} {"Prediction Type":<50} {"PnL(t)":>8} {"PnL($)":>10} {"Trades":>7} {"T/Day":>6} {"WR%":>6} {"PF":>7} {"Sharpe":>7}')
    print('-' * 110)
    for i, (pred_type, stats) in enumerate(viable[:20]):
        print(f'{i+1:>4} {pred_type:<50} {stats["total_pnl_ticks"]:>8.1f} {stats["total_pnl_usd"]:>10.0f} {stats["total_trades"]:>7} {stats["trades_per_day"]:>6.1f} {stats["win_rate"]:>6.1f} {stats["profit_factor"]:>7.3f} {stats["sharpe_proxy"]:>7.2f}')

    print(f'\n--- FAMILY BREAKDOWN ---')
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
            print(f'  {family}: {len(profitable)}/{len(members)} profitable | Best: {best[0]} (Sharpe {best[1]["sharpe_proxy"]:.2f}, PF {best[1]["profit_factor"]:.3f})')

if __name__ == '__main__':
    main()
