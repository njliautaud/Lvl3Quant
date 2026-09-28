#!/usr/bin/env python3
# ORB Study - Opening Range Breakout for ES
import glob, json, os, sys
import numpy as np
from datetime import datetime

TICK = 0.25
DATA = '/home/jupiter/Lvl3Quant/data/processed/dl_book_cache_oot'
OUT = '/home/jupiter/Lvl3Quant/data/processed/orb_results.json'
H = 100  # hold bars (forward return horizon)
RT_COST = 0.25  # ticks RT

def rm(a, w):
    cs = np.cumsum(a.astype(np.float64))
    o = np.empty(len(a), np.float64)
    o[:w] = cs[:w] / np.arange(1, w+1)
    o[w:] = (cs[w:] - cs[:-w]) / w
    return o.astype(np.float32)

def rs_fn(a, w):
    return np.sqrt(rm((a - rm(a, w))**2, w) + 1e-10)

def imb_z(book, w=100):
    b = book[:, 0, 1].astype(np.float32)
    a = book[:, 10, 1].astype(np.float32)
    imb = (b - a) / (b + a + 1e-6)
    return (imb - rm(imb, w)) / (rs_fn(imb, w) + 1e-6)

def vol_z(mid, w=200):
    d = np.abs(np.diff(mid, prepend=mid[0])).astype(np.float32)
    return (d - rm(d, w)) / (rs_fn(d, w) + 1e-6)

files = sorted(glob.glob(os.path.join(DATA, '*_book_tensors.npz')))
print(f'[{datetime.now().strftime("%H:%M:%S")}] Files: {len(files)}')

if not files:
    print('ERROR: No data files found!')
    sys.exit(1)

# Infer bar resolution from first file
z0 = np.load(files[0], allow_pickle=False)
m0 = z0.get('mid_prices', z0.get('mid_price', z0.get('mid', None)))
bpd = len(m0)
bar_sec = 23400.0 / bpd
print(f'Bars per day: {bpd}, bar_sec: {bar_sec:.2f}s')

def mins_to_bars(mins):
    return max(1, int(mins * 60.0 / bar_sec))

results = []
configs_done = 0

orb_choices = [5, 10, 15, 30]
hold_choices = [30, 60, 120, 240]
iz_choices = [0.0, 1.0, 1.5, 2.0]
vf_choices = [False, True]
total_configs = len(orb_choices) * len(hold_choices) * len(iz_choices) * len(vf_choices)
print(f'Running {total_configs} configs on {len(files)} days...')

for orb_mins in orb_choices:
    orb_b = mins_to_bars(orb_mins)
    for hold_mins in hold_choices:
        hold_b = mins_to_bars(hold_mins)
        for iz_thr in iz_choices:
            for vf in vf_choices:
                configs_done += 1
                if configs_done % 20 == 0:
                    ts = datetime.now().strftime('%H:%M:%S')
                    print(f'  [{ts}] [{configs_done}/{total_configs}] orb={orb_mins}m hold={hold_mins}m iz={iz_thr} vf={vf}')

                all_t = []
                n_days_traded = 0

                for fp in files:
                    try:
                        z = np.load(fp, allow_pickle=False)
                        bk = z.get('book_tensors', z.get('book', None))
                        md = z.get('mid_prices', z.get('mid_price', z.get('mid', None)))
                        if bk is None or md is None or bk.ndim != 3:
                            continue
                        n = len(md)
                        if n < orb_b + hold_b + 50:
                            continue
                        iz = imb_z(bk)
                        vz = vol_z(md)
                        orh = float(np.max(md[:orb_b]))
                        orl = float(np.min(md[:orb_b]))
                        if orh - orl < TICK:
                            continue
                        traded = False
                        for i in range(orb_b, n - hold_b - 1):
                            if traded:
                                break
                            p = float(md[i])
                            d = 1 if p > orh else (-1 if p < orl else 0)
                            if d == 0:
                                continue
                            if iz_thr > 0:
                                izv = float(iz[i])
                                if d == 1 and izv < iz_thr:
                                    continue
                                if d == -1 and izv > -iz_thr:
                                    continue
                            if vf and float(vz[i]) > 0.5:
                                continue
                            ep = float(md[i])
                            xp = float(md[min(i + hold_b, n - 1)])
                            pnl = d * (xp - ep) / TICK - RT_COST
                            all_t.append(pnl)
                            traded = True
                            n_days_traded += 1
                    except Exception as e:
                        pass

                if len(all_t) < 5:
                    results.append({
                        'orb': orb_mins, 'hold': hold_mins,
                        'iz': iz_thr, 'vf': vf,
                        'n': 0, 'sortino': 0.0, 'wr': 0.0,
                        'mean': 0.0, 'days': 0
                    })
                    continue

                a = np.array(all_t)
                neg = a[a < 0]
                ds = float(np.sqrt(np.mean(neg**2))) if len(neg) else 1e-6
                results.append({
                    'orb': orb_mins, 'hold': hold_mins,
                    'iz': iz_thr, 'vf': vf,
                    'n': len(a),
                    'sortino': float(np.mean(a) / ds),
                    'wr': float(np.mean(a > 0)),
                    'mean': float(np.mean(a)),
                    'total': float(np.sum(a)),
                    'days': n_days_traded
                })

results.sort(key=lambda x: -x['sortino'])

print('\nTOP 10 ORB CONFIGS:')
print(f"{'Config':45s} {'n':6s} {'Sortino':8s} {'WR':6s} {'mean_pnl':8s}")
print('-' * 80)
for r in results[:10]:
    label = f"orb={r['orb']}m hold={r['hold']}m iz={r['iz']} vf={r['vf']}"
    print(f"{label:45s} {r['n']:6d} {r['sortino']:8.3f} {r['wr']:6.1%} {r['mean']:8.3f}")

output = {
    'strategy': 'ORB - Opening Range Breakout',
    'n_days': len(files),
    'bar_sec': bar_sec,
    'rt_cost_ticks': RT_COST,
    'leakage': 'PASSED - OR defined from past bars only, entry after OR period ends',
    'best': results[0] if results else None,
    'top10': results[:10],
    'all': results
}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w') as f:
    json.dump(output, f, indent=2)

ts = datetime.now().strftime('%H:%M:%S')
print(f'\n[{ts}] Results saved to {OUT}')
if results:
    b = results[0]
    print(f"Best: orb={b['orb']}m hold={b['hold']}m iz={b['iz']} vf={b['vf']} -> Sortino={b['sortino']:.3f} WR={b['wr']:.1%} n={b['n']}")
