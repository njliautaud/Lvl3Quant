
#!/usr/bin/env python3
"""
iceberg_regime_analysis.py
Analyze iceberg fill sim day-by-day results to find regime filters that salvage edge.
Looks at: vol regime, day of week, time-of-day patterns, best queue position days.
"""
import json, os, glob, collections, statistics

RESULTS_DIR = '/home/jupiter/Lvl3Quant/data/processed/iceberg_fillsim'
DATA_DIR = '/home/jupiter/Lvl3Quant/data/processed'
OUTPUT = '/home/jupiter/Lvl3Quant/data/processed/iceberg_regime_analysis.json'

# Focus on best config: hold10s_prime
CONFIG = 'hold10s_prime'

# Load all day files for this config
day_results = []
pattern = os.path.join(RESULTS_DIR, f'*_{CONFIG}.json')
for f in sorted(glob.glob(pattern)):
    try:
        with open(f) as fh:
            d = json.load(fh)
        if 'error' not in d or d.get('error') != 'no_mbo':
            day_results.append(d)
    except:
        pass

print(f'Loaded {len(day_results)} day results for {CONFIG}')

# Sort by pnl
day_results_sorted = sorted(day_results, key=lambda x: x.get('total_pnl_dollars', -999999), reverse=True)

# Top 20 best days
print('
=== TOP 20 BEST DAYS ===')
print(f"{'Date':<12} {'PnL$':>8} {'Trades':>7} {'WR%':>6} {'FillR%':>7} {'QueuePos':>9}")
for d in day_results_sorted[:20]:
    date = d.get('date','?')[:10]
    pnl = d.get('total_pnl_dollars', 0)
    trades = d.get('total_trades', 0)
    wr = d.get('win_rate', 0) * 100
    filled = d.get('total_filled', 0)
    posted = d.get('total_posted', 1)
    fill_r = filled / max(posted, 1) * 100
    queue = d.get('avg_queue_position', 0)
    print(f'{date:<12} {pnl:>8.0f} {trades:>7} {wr:>6.1f} {fill_r:>7.1f} {queue:>9.1f}')

# Bottom 20 worst days
print('
=== TOP 20 WORST DAYS ===')
for d in day_results_sorted[-20:]:
    date = d.get('date','?')[:10]
    pnl = d.get('total_pnl_dollars', 0)
    trades = d.get('total_trades', 0)
    wr = d.get('win_rate', 0) * 100
    queue = d.get('avg_queue_position', 0)
    print(f'{date:<12} {pnl:>8.0f} {trades:>7} {wr:>6.1f} {queue:>9.1f}')

# Low queue position days (queue < 15 on average)
low_queue = [d for d in day_results if d.get('avg_queue_position', 999) < 15]
print(f'
=== LOW QUEUE DAYS (avg_queue<15): {len(low_queue)} days ===')
if low_queue:
    tot_pnl = sum(d.get('total_pnl_dollars',0) for d in low_queue)
    tot_trades = sum(d.get('total_trades',0) for d in low_queue)
    avg_wr = statistics.mean(d.get('win_rate',0) for d in low_queue) * 100
    print(f'  Total PnL: ${tot_pnl:.0f}, $/day: {tot_pnl/len(low_queue):.0f}, trades: {tot_trades}, WR: {avg_wr:.1f}%')

# High WR days (wr > 0.5)
high_wr = [d for d in day_results if d.get('win_rate', 0) > 0.5]
print(f'
=== HIGH WR DAYS (wr>50%): {len(high_wr)} days ===')
if high_wr:
    tot_pnl = sum(d.get('total_pnl_dollars',0) for d in high_wr)
    print(f'  Total PnL: ${tot_pnl:.0f}, $/day: {tot_pnl/len(high_wr):.0f}')

# Positive PnL days
pos_days = [d for d in day_results if d.get('total_pnl_dollars', 0) > 0]
print(f'
=== POSITIVE DAYS: {len(pos_days)}/{len(day_results)} ({len(pos_days)/len(day_results)*100:.1f}%) ===')
if pos_days:
    tot_pnl = sum(d.get('total_pnl_dollars',0) for d in pos_days)
    avg_q = statistics.mean(d.get('avg_queue_position',0) for d in pos_days)
    print(f'  Avg queue on positive days: {avg_q:.1f}')
    print(f'  Avg PnL on positive days: ${tot_pnl/len(pos_days):.0f}')

# Queue position distribution on positive vs negative days
pos_queue = [d.get('avg_queue_position',0) for d in day_results if d.get('total_pnl_dollars',0) > 0]
neg_queue = [d.get('avg_queue_position',0) for d in day_results if d.get('total_pnl_dollars',0) <= 0]
if pos_queue: print(f'  Mean queue (pos days): {statistics.mean(pos_queue):.1f}')
if neg_queue: print(f'  Mean queue (neg days): {statistics.mean(neg_queue):.1f}')

print('
Done.')
