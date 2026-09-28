
import json, glob, os, math
from collections import defaultdict

DIR = '/home/jupiter/Lvl3Quant/data/processed/iceberg_fillsim/'
files = glob.glob(DIR + '2025-*.json')
files = [f for f in files if 'raw' not in os.path.basename(f) and 'summary' not in os.path.basename(f)]

by_config = defaultdict(list)
for f in files:
    try:
        d = json.load(open(f))
        cfg = os.path.basename(f).replace('.json','')
        parts = cfg.split('_')
        c = '_'.join(parts[3:])
        by_config[c].append(d)
    except:
        pass

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
print(f"{'Config':<25} {'Days':>5} {'Trades':>7} {'PnL$':>12} {'$/day':>9} {'WR':>6} {'FillR':>6} {'AvgQ':>6} {'Lat':>8} {'Sortino':>9}")
print('-' * 95)
for s, cfg, nd, nt, pnl, dpnl, wr, fr, aq, lat in results:
    print(f"{cfg:<25} {nd:>5} {nt:>7} {pnl:>12,.0f} {dpnl:>9,.0f} {wr:>5.1%} {fr:>5.1%} {aq:>6.1f} {lat:>8.0f} {s:>9.3f}")
print()
print('Total configs:', len(results), ' Days/config:', results[0][2] if results else 0)
