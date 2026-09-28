
import json, glob, os, math, re
from collections import defaultdict

DIR = '/home/jupiter/Lvl3Quant/data/processed/iceberg_fillsim/'
files = glob.glob(DIR + '2025-*.json')
files = [f for f in files if 'raw' not in os.path.basename(f) and 'summary' not in os.path.basename(f)]

by_config = defaultdict(list)
for f in files:
    try:
        d = json.load(open(f))
        bn = os.path.basename(f).replace('.json','')
        # Format: 2025-07-22_hold10s_chase => strip date (YYYY-MM-DD_)
        c = re.sub(r'^\d{4}-\d{2}-\d{2}_', '', bn)
        by_config[c].append(d)
    except:
        pass

print('Configs found:', sorted(by_config.keys()))
results = []
for cfg, days in sorted(by_config.items()):
    nd = len(days)
    nt = sum(d.get('total_trades', 0) for d in days)
    pnl = sum(d.get('total_pnl_dollars', 0) for d in days)
    dpnls = [d.get('total_pnl_dollars', 0) for d in days]
    sigs = sum(d.get('total_signals', 0) for d in days)
    filled = sum(d.get('total_filled', 0) for d in days)
    wins = sum(d.get('total_trades', 0) * d.get('win_rate', 0) for d in days)
    q_sum = sum(d.get('avg_queue_position', 0) * max(d.get('total_filled', 1), 1) for d in days)
    lat_sum = sum(d.get('avg_fill_latency_ms', 0) * max(d.get('total_filled', 1), 1) for d in days)
    neg = [p for p in dpnls if p < 0]
    ds = math.sqrt(sum(p**2 for p in neg)/len(neg)) if neg else 1e-9
    mu = pnl / nd
    sortino = mu / ds
    fr = filled / max(sigs, 1)
    wr = wins / max(nt, 1)
    aq = q_sum / max(filled, 1)
    lat = lat_sum / max(filled, 1)
    results.append((sortino, cfg, nd, nt, pnl, mu, wr, fr, aq, lat))

results.sort(reverse=True)
print()
print(f"{'Config':<28} {'Days':>5} {'Trades':>7} {'PnL$':>12} {'$/day':>9} {'WR':>6} {'FillR':>6} {'AvgQ':>6} {'Lat(ms)':>8} {'Sortino':>9}")
print('-' * 98)
for s, c, nd, nt, pnl, dpnl, wr, fr, aq, lat in results:
    print(f"{c:<28} {nd:>5} {nt:>7} {pnl:>12,.0f} {dpnl:>9,.0f} {wr:>5.1%} {fr:>5.1%} {aq:>6.1f} {lat:>8.0f} {s:>9.3f}")
print()
print(f'N configs: {len(results)}, days/config ~{nd}')
