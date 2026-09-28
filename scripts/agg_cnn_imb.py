import json, os, glob
import numpy as np
from pathlib import Path

results_dir = Path('/home/jupiter/Lvl3Quant/data/processed/cnn_imb_filter')
files = sorted(glob.glob(str(results_dir / '*.json')))
files = [f for f in files if 'summary' not in f]
print('Found %d result files' % len(files))

by_cfg = {}
for f in files:
    try:
        with open(f) as fh:
            d = json.load(fh)
        basename = os.path.basename(f)
        date = basename[:10]
        cfg = basename[11:].replace('.json', '')
        if cfg not in by_cfg:
            by_cfg[cfg] = []
        by_cfg[cfg].append({
            'n': d.get('total_trades', 0),
            'pnl_usd': d.get('total_pnl_dollars', 0),
            'wr': d.get('win_rate', 0),
            'fill_rate': d.get('fill_rate', 0),
            'mean_pnl': d.get('mean_pnl_per_trade', 0)
        })
    except:
        pass

print('Config                                     days  n    fill  WR    mpnl   total  sort')
print('-' * 85)
summaries = []
for cfg, recs in sorted(by_cfg.items()):
    recs = [r for r in recs if r['n'] > 0]
    if not recs: continue
    n = sum(r['n'] for r in recs)
    total_pnl = sum(r['pnl_usd'] for r in recs)
    avg_wr = float(np.mean([r['wr'] for r in recs]))
    avg_fill = float(np.mean([r['fill_rate'] for r in recs]))
    mean_pnl = total_pnl / n if n > 0 else 0
    arr = np.array([r['pnl_usd'] for r in recs])
    neg = arr[arr < 0]
    ds = float(np.sqrt(np.mean(neg**2))) if len(neg) else 1e-6
    sortino = float(np.mean(arr) / ds)
    print('%-42s  %3d %4d  %.1f  %.1f  %7.2f  %7.0f  %.3f' % (
        cfg, len(recs), n, avg_fill*100, avg_wr*100, mean_pnl, total_pnl, sortino))
    summaries.append({'cfg': cfg, 'days': len(recs), 'n': n, 'sortino': sortino, 'wr': avg_wr, 'fill': avg_fill, 'total_pnl': total_pnl, 'mean_pnl': mean_pnl})

summaries.sort(key=lambda x: x['sortino'], reverse=True)
print()
if summaries:
    b = summaries[0]
    print('BEST: %s  Sort=%.3f  WR=%.1f  n=%d  total=$%.0f' % (b['cfg'], b['sortino'], b['wr']*100, b['n'], b['total_pnl']))
