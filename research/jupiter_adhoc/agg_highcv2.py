#!/usr/bin/env python3
import json, glob, os, math
from collections import defaultdict
dir_ = '/home/jupiter/Lvl3Quant/data/processed/iceberg_highcv_fillsim/'
files = glob.glob(dir_ + '*.json')
configs = defaultdict(lambda: {'pnl': 0.0, 'trades': 0, 'wins': 0, 'days': 0, 'pnl_list': []})
for fp in files:
    try:
        with open(fp) as f:
            d = json.load(f)
        base = os.path.basename(fp)
        parts = base.replace('.json','').split('_')
        cv = parts[2]
        style = '_'.join(parts[3:])
        key = cv + '|' + style
        pnl = d.get('total_pnl_dollars', 0) or 0
        trades = d.get('total_trades', 0) or 0
        wr = d.get('win_rate', 0) or 0
        fill_rate = d.get('fill_rate', 0) or 0
        avg_q = d.get('avg_queue_position', 0) or 0
        configs[key]['pnl'] += pnl
        configs[key]['trades'] += trades
        configs[key]['wins'] += int(trades * wr)
        configs[key]['days'] += 1
        configs[key]['pnl_list'].append(pnl)
        if 'fill_rate_sum' not in configs[key]:
            configs[key]['fill_rate_sum'] = 0
            configs[key]['avg_q_sum'] = 0
        configs[key]['fill_rate_sum'] += fill_rate
        configs[key]['avg_q_sum'] += avg_q
    except Exception as e:
        pass

rows = []
for key, v in configs.items():
    if v['days'] > 0 and v['trades'] > 0:
        pnl_day = v['pnl'] / v['days']
        wr = v['wins'] / v['trades']
        fill_rate = v['fill_rate_sum'] / v['days']
        avg_q = v['avg_q_sum'] / v['days']
        # Sortino: daily pnl list
        plist = v['pnl_list']
        mean_d = sum(plist)/len(plist)
        neg = [x for x in plist if x < 0]
        if len(neg) > 1:
            import math
            dsd = math.sqrt(sum(x**2 for x in neg)/len(neg))
            sortino = mean_d / dsd if dsd > 0 else 0
        else:
            sortino = 99.0
        rows.append((sortino, key, pnl_day, v['pnl'], v['trades'], v['days'], wr, fill_rate, avg_q))

rows.sort(reverse=True)
print("{:35s} {:>7s} {:>9s} {:>7s} {:>6s} {:>5s} {:>5s} {:>6s}".format(
    "Config","Sortino","PnL/day","Total$","Trades","Days","WR","FillR"))
for row in rows[:20]:
    sortino, key, pnl_day, total_pnl, trades, days, wr, fill_rate, avg_q = row
    print("{:35s} {:>7.3f} {:>+9.0f} {:>+7.0f} {:>6d} {:>5d} {:>5.1%} {:>6.1%}".format(
        key, sortino, pnl_day, total_pnl, trades, days, wr, fill_rate))
print("Total files: {}".format(len(files)))
