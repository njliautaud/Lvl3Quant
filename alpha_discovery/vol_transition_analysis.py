#!/usr/bin/env python3
"""Analyze ET fold-level IC by vol regime — does ET predict better during vol transitions?"""

import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Load ET 70d fold results
with open(ROOT / 'alpha_discovery/deep_models/results/walkforward_event_20260225_225006.json') as f:
    et = json.load(f)

date_ics = {}
for fold in et['fold_details']:
    date_ics[fold['test_date']] = fold['ic']

print(f"ET 70d: {len(date_ics)} folds")
print(f"Date range: {min(date_ics.keys())} to {max(date_ics.keys())}")
print(f"Mean IC: {np.mean(list(date_ics.values())):+.4f}")

# Compute daily vol for each date
data_dir = ROOT / 'data' / 'processed' / 'mbo_features_cache'
files = sorted(data_dir.glob('*_mbo_features.npz'))

daily_vol = {}
for fpath in files:
    stem = fpath.stem.replace('_mbo_features', '')
    date = stem  # already in YYYY-MM-DD format
    data = np.load(str(fpath))
    mid = data['mbo_features'][:, 0]
    mask = ~np.isnan(mid) & (mid > 0)
    if mask.sum() > 100:
        returns = np.diff(mid[mask]) / mid[mask][:-1]
        daily_vol[date] = np.std(returns) * 100

print(f"Daily vol data: {len(daily_vol)} days")

# Match ET fold dates with vol data
matched = []
for date, ic in date_ics.items():
    if date in daily_vol:
        matched.append({'date': date, 'ic': ic, 'vol': daily_vol[date]})

matched.sort(key=lambda x: x['date'])
print(f"Matched: {len(matched)} dates")

# Classify vol regime relative to recent history
for i in range(len(matched)):
    if i >= 5:
        prev_vols = [matched[j]['vol'] for j in range(i-5, i)]
        matched[i]['prev_vol_mean'] = np.mean(prev_vols)
        matched[i]['vol_ratio'] = matched[i]['vol'] / matched[i]['prev_vol_mean']
    else:
        matched[i]['prev_vol_mean'] = matched[i]['vol']
        matched[i]['vol_ratio'] = 1.0

# Identify transitions: calm previous, volatile current (ratio > 1.3)
transitions = [m for m in matched if m['vol_ratio'] > 1.3]
calm = [m for m in matched if m['vol_ratio'] < 0.8]
normal = [m for m in matched if 0.8 <= m['vol_ratio'] <= 1.3]

print(f"\n{'='*60}")
print("VOL REGIME ANALYSIS — ET 70d IC by Vol Regime")
print(f"{'='*60}")

print(f"\nTransition days (vol spike >30%): {len(transitions)}")
if transitions:
    t_ics = [t['ic'] for t in transitions]
    print(f"  Mean IC: {np.mean(t_ics):+.4f}")
    print(f"  Median IC: {np.median(t_ics):+.4f}")
    for t in sorted(transitions, key=lambda x: x['ic'], reverse=True)[:8]:
        print(f"    {t['date']} IC={t['ic']:+.4f} vol_ratio={t['vol_ratio']:.2f}")

print(f"\nCalm days (vol drop >20%): {len(calm)}")
if calm:
    c_ics = [c['ic'] for c in calm]
    print(f"  Mean IC: {np.mean(c_ics):+.4f}")

print(f"\nNormal days: {len(normal)}")
if normal:
    n_ics = [n['ic'] for n in normal]
    print(f"  Mean IC: {np.mean(n_ics):+.4f}")

# High/Low absolute vol
vols = [m['vol'] for m in matched]
med_vol = np.median(vols)
high_vol = [m for m in matched if m['vol'] > med_vol]
low_vol = [m for m in matched if m['vol'] <= med_vol]
print(f"\nMedian daily vol: {med_vol:.6f}")
print(f"High vol days ({len(high_vol)}): Mean IC = {np.mean([m['ic'] for m in high_vol]):+.4f}")
print(f"Low vol days ({len(low_vol)}):  Mean IC = {np.mean([m['ic'] for m in low_vol]):+.4f}")

# Quartile analysis
q25, q75 = np.percentile(vols, [25, 75])
q1 = [m for m in matched if m['vol'] <= q25]
q4 = [m for m in matched if m['vol'] > q75]
print(f"\nQ1 (lowest vol, {len(q1)} days): IC = {np.mean([m['ic'] for m in q1]):+.4f}")
print(f"Q4 (highest vol, {len(q4)} days): IC = {np.mean([m['ic'] for m in q4]):+.4f}")

# Top/bottom IC days
top10 = sorted(matched, key=lambda x: x['ic'], reverse=True)[:10]
print(f"\nTop 10 IC days:")
for t in top10:
    print(f"  {t['date']} IC={t['ic']:+.4f} vol={t['vol']:.6f} ratio={t['vol_ratio']:.2f}")

bottom10 = sorted(matched, key=lambda x: x['ic'])[:10]
print(f"\nBottom 10 IC days:")
for t in bottom10:
    print(f"  {t['date']} IC={t['ic']:+.4f} vol={t['vol']:.6f} ratio={t['vol_ratio']:.2f}")

# Expected PnL analysis
COST_TICKS = 1.24
TICK_VALUE = 12.50

print(f"\n{'='*60}")
print("EXPECTED PnL ANALYSIS")
print(f"{'='*60}")

for label, group in [("All days", matched), ("Transitions", transitions),
                      ("High vol Q4", q4), ("Calm", calm)]:
    if not group:
        continue
    ics_arr = np.array([g['ic'] for g in group])
    vols_arr = np.array([g['vol'] for g in group])
    mean_ic = np.mean(ics_arr)
    # Average move size estimation: vol in bps * sqrt(100 bars) * price/tick
    # More direct: use daily vol to estimate typical 1s move in ticks
    # vol is std(returns)*100 in bps, for 1-bar returns
    # 1-bar = 10ms. 100 bars = 1s.
    # Expected |1s return| ~ vol * sqrt(100) * mean_price / tick_size
    # We need move in ticks, not percentage
    avg_vol_bps = np.mean(vols_arr)
    # typical ES price ~5800, tick=0.25, so 1 tick = 0.25/5800 = 0.0043%
    # avg 1-bar |ret| ~ vol_bps, so 100-bar ~ vol_bps * sqrt(100) = vol*10
    est_1s_move_bps = avg_vol_bps * 10  # rough sqrt(100) scaling
    est_1s_move_ticks = est_1s_move_bps * 5800 / (0.25 * 100)  # convert bps to ticks
    expected_ret = mean_ic * est_1s_move_ticks
    profit = expected_ret - COST_TICKS

    print(f"\n  {label} ({len(group)} days):")
    print(f"    Mean IC: {mean_ic:+.4f}")
    print(f"    Avg vol (bps): {avg_vol_bps:.6f}")
    print(f"    Est 1s move: {est_1s_move_ticks:.2f} ticks")
    print(f"    Expected return: {expected_ret:.2f} ticks")
    print(f"    Cost: {COST_TICKS:.2f} ticks")
    print(f"    Net PnL/trade: {profit:.2f} ticks ({profit * TICK_VALUE:.2f} USD)")
    if profit > 0:
        print(f"    >>> PROFITABLE!")
    else:
        print(f"    >>> Need IC > {COST_TICKS / est_1s_move_ticks:.4f} to break even")
