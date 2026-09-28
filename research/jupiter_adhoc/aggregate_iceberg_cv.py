import json, os, glob
from collections import defaultdict
import statistics

FILLSIM_DIR = '/home/jupiter/Lvl3Quant/data/processed/iceberg_highcv_fillsim'
OUTPUT = '/home/jupiter/iceberg_highcv_aggregate.json'

files = glob.glob(os.path.join(FILLSIM_DIR, '*.json'))
print(f'Found {len(files)} files')

# Group by cv_tier and config_name
by_config = defaultdict(list)
for fpath in files:
    fname = os.path.basename(fpath)
    parts = fname.replace('.json','').split('_')
    # format: DATE_cvXpY_CONFIG...
    cv_part = [p for p in parts if p.startswith('cv')]
    if not cv_part:
        continue
    cv_tier = cv_part[0]  # e.g. cv5p0, cv6p0, cv8p0
    # find index of cv part
    cv_idx = next(i for i, p in enumerate(parts) if p.startswith('cv'))
    config_name = '_'.join(parts[cv_idx+1:])
    key = (cv_tier, config_name)
    try:
        with open(fpath) as f:
            d = json.load(f)
        by_config[key].append(d)
    except Exception as e:
        pass

print(f'Loaded {sum(len(v) for v in by_config.values())} records across {len(by_config)} config combos')

results = []
for (cv_tier, config_name), days in by_config.items():
    n_days = len(days)
    total_pnl = sum(d.get('total_pnl_dollars', 0) for d in days)
    total_trades = sum(d.get('total_trades', 0) for d in days)
    total_signals = sum(d.get('total_signals', 0) for d in days)
    total_filled = sum(d.get('total_filled', 0) for d in days)
    fill_rates = [d.get('fill_rate', 0) for d in days if d.get('total_signals', 0) > 0]
    avg_fill_rate = statistics.mean(fill_rates) if fill_rates else 0
    queue_positions = [d.get('avg_queue_position', 0) for d in days if d.get('total_trades', 0) > 0]
    avg_queue_pos = statistics.mean(queue_positions) if queue_positions else 0
    daily_pnls = [d.get('total_pnl_dollars', 0) for d in days]
    avg_daily_pnl = statistics.mean(daily_pnls)
    if len(daily_pnls) > 1:
        std_daily = statistics.stdev(daily_pnls)
        sortino_neg = [p for p in daily_pnls if p < 0]
        downside_std = statistics.stdev(sortino_neg) if len(sortino_neg) > 1 else (std_daily if std_daily > 0 else 1)
        sortino = (avg_daily_pnl / downside_std) if downside_std > 0 else 0
    else:
        sortino = 0
    win_rates = [d.get('win_rate', 0) for d in days if d.get('total_trades', 0) > 0]
    avg_wr = statistics.mean(win_rates) if win_rates else 0
    results.append({
        'cv_tier': cv_tier,
        'config': config_name,
        'n_days': n_days,
        'total_pnl': round(total_pnl, 2),
        'avg_daily_pnl': round(avg_daily_pnl, 2),
        'sortino': round(sortino, 3),
        'total_trades': total_trades,
        'trades_per_day': round(total_trades / n_days, 1),
        'avg_fill_rate': round(avg_fill_rate, 3),
        'avg_queue_pos': round(avg_queue_pos, 1),
        'avg_wr': round(avg_wr, 3),
    })

results.sort(key=lambda x: x['sortino'], reverse=True)

with open(OUTPUT, 'w') as f:
    json.dump(results, f, indent=2)

print('\nTop 20 by Sortino:')
for r in results[:20]:
    print('  ' + r['cv_tier'] + ' ' + r['config'][:30].ljust(30) +
          ' Sortino=' + str(r['sortino']) +
          ' PnL/day=$' + str(r['avg_daily_pnl']) +
          ' Trades/day=' + str(r['trades_per_day']) +
          ' WR=' + str(round(r['avg_wr']*100,1)) + '%' +
          ' FillRate=' + str(round(r['avg_fill_rate']*100,1)) + '%' +
          ' QueuePos=' + str(r['avg_queue_pos']))

print('\nSummary by CV tier (positive Sortino configs):')
for cv in ['cv5p0', 'cv6p0', 'cv8p0']:
    total_cv = [r for r in results if r['cv_tier'] == cv]
    pos_cv = [r for r in total_cv if r['sortino'] > 0]
    print('  ' + cv + ': ' + str(len(pos_cv)) + ' positive-Sortino configs out of ' + str(len(total_cv)))

print('\nSaved to ' + OUTPUT)
