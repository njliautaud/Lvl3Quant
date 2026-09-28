#!/usr/bin/env python3
import json, glob, os
from collections import defaultdict
dir_ = '/home/jupiter/Lvl3Quant/data/processed/iceberg_highcv_fillsim/'
files = glob.glob(dir_ + '*.json')
configs = defaultdict(lambda: {'pnl': 0.0, 'trades': 0, 'wins': 0, 'days': 0})
for fp in files:
    try:
        with open(fp) as f:
            d = json.load(f)
        base = os.path.basename(fp)
        parts = base.replace('.json','').split('_')
        cv = parts[2]
        style = '_'.join(parts[3:])
        key = cv + '|' + style
        pnl = d.get('total_pnl', d.get('pnl', 0)) or 0
        trades = d.get('total_trades', d.get('n_trades', 0)) or 0
        wins = d.get('wins', 0) or 0
        configs[key]['pnl'] += pnl
        configs[key]['trades'] += trades
        configs[key]['wins'] += wins
        configs[key]['days'] += 1
    except:
        pass
rows = []
for key, v in configs.items():
    if v['days'] > 0 and v['trades'] > 0:
        pnl_day = v['pnl'] / v['days']
        wr = v['wins'] / v['trades'] if v['trades'] > 0 else 0
        rows.append((pnl_day, key, v['pnl'], v['trades'], v['days'], wr))
rows.sort(reverse=True)
for row in rows[:20]:
    pnl_day, key, total_pnl, trades, days, wr = row
    print("{:40s} pnl/day={:+.0f} total={:+.0f} trades={} days={} WR={:.1%}".format(key, pnl_day, total_pnl, trades, days, wr))
print("Total files: {}".format(len(files)))
