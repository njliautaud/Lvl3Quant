"""Aggregate Jupiter aggressive_sweep results into summary."""
import json, os, re, sys
from collections import defaultdict

DIR = '/home/jupiter/lvl3quant/data/processed/aggressive_sweep'
results = defaultdict(lambda: {'pnl': 0, 'trades': 0, 'wins': 0, 'fills': 0, 'signals': 0, 'days': 0, 'pos_days': 0})

count = 0
for fname in os.listdir(DIR):
    if not fname.endswith('.json'):
        continue
    try:
        with open(os.path.join(DIR, fname)) as f:
            d = json.load(f)

        parts = fname.replace('.json', '').rsplit('_', 1)
        date = parts[1] if len(parts) > 1 else 'unknown'
        config = parts[0] if len(parts) > 1 else fname

        m = re.match(r'(.+?)_st([\d.]+)_hold(\d+)_trail(\d+)_mw(\d+)', config)
        if not m:
            continue
        signal = m.group(1)
        thresh = float(m.group(2))
        hold = int(m.group(3))
        trail = int(m.group(4))
        mw = int(m.group(5))

        key = "%s_st%.1f_hold%d_trail%d_mw%d" % (signal, thresh, hold, trail, mw)

        pnl = d.get('total_pnl_dollars', 0)
        trades = d.get('total_trades', 0)
        wr = d.get('win_rate', 0)
        wins = int(trades * wr)
        fills = d.get('total_filled', 0)
        signals_count = d.get('total_signals', 0)

        r = results[key]
        r['pnl'] += pnl
        r['trades'] += trades
        r['wins'] += wins
        r['fills'] += fills
        r['signals'] += signals_count
        r['days'] += 1
        if pnl > 0:
            r['pos_days'] += 1
        r['signal'] = signal
        r['thresh'] = thresh
        r['hold'] = hold
        r['trail'] = trail
        r['mw'] = mw
        count += 1
    except Exception as e:
        pass

print("Parsed %d result files into %d configs" % (count, len(results)))

# Sort by mean daily PnL
sorted_results = sorted(results.items(), key=lambda x: -x[1]['pnl']/max(x[1]['days'],1))

print("\nTOP 30 BY PnL:")
print("%15s %6s %8s %5s %4s %8s %8s %5s %6s %6s %5s %4s" % (
    "Signal", "Thresh", "Hold", "Trail", "MW", "PnL/d", "Total", "Tr", "WR", "Fill%", "Pos%", "Days"))
print("-" * 100)
for key, r in sorted_results[:30]:
    nd = max(r['days'], 1)
    mean_pnl = r['pnl'] / nd
    wr = r['wins'] / max(r['trades'], 1) * 100
    fill_pct = r['fills'] / max(r['signals'], 1) * 100
    pos_pct = r['pos_days'] / nd * 100
    print("%15s %6.1f %8d %5d %4d %+8.1f %+8.0f %5d %5.1f%% %5.1f%% %4.0f%% %4d" % (
        r['signal'], r['thresh'], r['hold'], r['trail'], r['mw'],
        mean_pnl, r['pnl'], r['trades'], wr, fill_pct, pos_pct, r['days']))

print("\nBOTTOM 5:")
for key, r in sorted_results[-5:]:
    nd = max(r['days'], 1)
    mean_pnl = r['pnl'] / nd
    wr = r['wins'] / max(r['trades'], 1) * 100
    fill_pct = r['fills'] / max(r['signals'], 1) * 100
    pos_pct = r['pos_days'] / nd * 100
    print("%15s %6.1f %8d %5d %4d %+8.1f %+8.0f %5d %5.1f%% %5.1f%% %4.0f%% %4d" % (
        r['signal'], r['thresh'], r['hold'], r['trail'], r['mw'],
        mean_pnl, r['pnl'], r['trades'], wr, fill_pct, pos_pct, r['days']))

# Summary by signal
print("\nBY SIGNAL:")
sig_data = defaultdict(lambda: {'total_pnl': 0, 'n_configs': 0, 'n_positive': 0, 'best_pnl': -999999, 'best_key': ''})
for key, r in sorted_results:
    nd = max(r['days'], 1)
    mean_pnl = r['pnl'] / nd
    sd = sig_data[r['signal']]
    sd['total_pnl'] += mean_pnl
    sd['n_configs'] += 1
    if mean_pnl > 0:
        sd['n_positive'] += 1
    if mean_pnl > sd['best_pnl']:
        sd['best_pnl'] = mean_pnl
        sd['best_key'] = key

for sig in sorted(sig_data.keys()):
    sd = sig_data[sig]
    mean = sd['total_pnl'] / sd['n_configs']
    print("  %15s: mean=%+8.1f  best=%+8.1f  pos=%d/%d  best_cfg=%s" % (
        sig, mean, sd['best_pnl'], sd['n_positive'], sd['n_configs'], sd['best_key']))

# Summary by threshold
print("\nBY THRESHOLD:")
thresh_data = defaultdict(lambda: {'total_pnl': 0, 'n': 0, 'n_pos': 0, 'best': -999999})
for key, r in sorted_results:
    nd = max(r['days'], 1)
    mean_pnl = r['pnl'] / nd
    td = thresh_data[r['thresh']]
    td['total_pnl'] += mean_pnl
    td['n'] += 1
    if mean_pnl > 0:
        td['n_pos'] += 1
    td['best'] = max(td['best'], mean_pnl)

for t in sorted(thresh_data.keys()):
    td = thresh_data[t]
    mean = td['total_pnl'] / td['n']
    print("  t=%4.1f: mean=%+8.1f  best=%+8.1f  pos=%d/%d" % (t, mean, td['best'], td['n_pos'], td['n']))
