#!/usr/bin/env python3
"""
Deep-dive: SELL trade performance analysis
Why are SELL trades destroying value?
"""

import os
import json
import re
from collections import defaultdict

RESULTS_DIR = os.path.expanduser('~/Lvl3Quant/alpha_discovery/results/wf_sweep/')

print("=" * 70)
print("SELL TRADE DEEP DIVE ANALYSIS")
print("=" * 70)
print()

files = os.listdir(RESULTS_DIR)
json_files = [f for f in files if f.endswith('.json') and re.search(r'\d{4}-\d{2}-\d{2}', f)]

all_buy_pnl = []
all_sell_pnl = []
config_sell_pnl = defaultdict(list)  # config_type -> [sell pnls]
config_buy_pnl = defaultdict(list)

# Parse config name from filename: wf_v{vol}_{conv}_{hold}_{entry}_{chase}_{lat}_{date}.json
def parse_config(fname):
    # e.g. wf_v50_c1.5_h10m_chase_ct1r1_lat0_2025-12-01.json
    m = re.match(r'wf_(v\d+)_(c[\d.]+)_(h\d+m)_(\w+)_(\w+)_(lat\d+)_', fname)
    if m:
        return f"{m.group(1)}_{m.group(2)}_{m.group(3)}_{m.group(4)}_{m.group(5)}_{m.group(6)}"
    return "unknown"

total_configs = 0
for fname in json_files:
    fpath = os.path.join(RESULTS_DIR, fname)
    try:
        with open(fpath) as f:
            result = json.load(f)
        trades = result.get('trades', [])
        cfg = parse_config(fname)
        for t in trades:
            if 'side' not in t or 'pnl_dollars' not in t:
                continue
            if t['side'] == 'BUY':
                all_buy_pnl.append(t['pnl_dollars'])
                config_buy_pnl[cfg].append(t['pnl_dollars'])
            else:
                all_sell_pnl.append(t['pnl_dollars'])
                config_sell_pnl[cfg].append(t['pnl_dollars'])
        total_configs += 1
    except Exception:
        pass

print(f"Analyzed {total_configs} configs")
print()

# Overall BUY vs SELL
print("OVERALL BUY vs SELL PERFORMANCE:")
print(f"  BUY  trades: {len(all_buy_pnl):,}  Total: ${sum(all_buy_pnl):+,.0f}  Avg: ${sum(all_buy_pnl)/len(all_buy_pnl):+.2f}  WR: {sum(1 for x in all_buy_pnl if x>0)/len(all_buy_pnl)*100:.1f}%" if all_buy_pnl else "  No BUY trades")
print(f"  SELL trades: {len(all_sell_pnl):,}  Total: ${sum(all_sell_pnl):+,.0f}  Avg: ${sum(all_sell_pnl)/len(all_sell_pnl):+.2f}  WR: {sum(1 for x in all_sell_pnl if x>0)/len(all_sell_pnl)*100:.1f}%" if all_sell_pnl else "  No SELL trades")
print()

# Are there ANY configs where SELL trades are profitable?
sell_profitable_configs = {}
sell_loss_configs = {}
for cfg, pnls in config_sell_pnl.items():
    total = sum(pnls)
    if total > 0:
        sell_profitable_configs[cfg] = (total, len(pnls))
    else:
        sell_loss_configs[cfg] = (total, len(pnls))

print(f"Configs with profitable SELL trades: {len(sell_profitable_configs)}/{len(config_sell_pnl)}")
print(f"Configs with losing SELL trades: {len(sell_loss_configs)}/{len(config_sell_pnl)}")
print()

if sell_profitable_configs:
    top_sell = sorted(sell_profitable_configs.items(), key=lambda x: -x[1][0])[:10]
    print("TOP 10 configs by SELL P&L:")
    for cfg, (total, n) in top_sell:
        print(f"  {cfg}: ${total:+,.0f} over {n} trades (avg ${total/n:+.2f})")
    print()

if sell_loss_configs:
    worst_sell = sorted(sell_loss_configs.items(), key=lambda x: x[1][0])[:10]
    print("WORST 10 configs by SELL P&L:")
    for cfg, (total, n) in worst_sell:
        print(f"  {cfg}: ${total:+,.0f} over {n} trades (avg ${total/n:+.2f})")
    print()

# Breakdown by vol threshold and hold period for SELL trades
print("SELL P&L by vol threshold:")
for vol in ['v50', 'v60', 'v70', 'v80', 'v90']:
    sells = [p for cfg, pnls in config_sell_pnl.items() if vol in cfg for p in pnls]
    if sells:
        print(f"  {vol}: {len(sells):,} trades, Total: ${sum(sells):+,.0f}, Avg: ${sum(sells)/len(sells):+.2f}, WR: {sum(1 for x in sells if x>0)/len(sells)*100:.1f}%")

print()
print("SELL P&L by hold period:")
for hold in ['h10m', 'h20m', 'h30m', 'h60m']:
    sells = [p for cfg, pnls in config_sell_pnl.items() if hold in cfg for p in pnls]
    if sells:
        print(f"  {hold}: {len(sells):,} trades, Total: ${sum(sells):+,.0f}, Avg: ${sum(sells)/len(sells):+.2f}, WR: {sum(1 for x in sells if x>0)/len(sells)*100:.1f}%")

print()
print("SELL P&L by conv threshold:")
for conv in ['c1.5', 'c2.0', 'c2.5', 'c3.0']:
    sells = [p for cfg, pnls in config_sell_pnl.items() if conv in cfg for p in pnls]
    if sells:
        print(f"  {conv}: {len(sells):,} trades, Total: ${sum(sells):+,.0f}, Avg: ${sum(sells)/len(sells):+.2f}, WR: {sum(1 for x in sells if x>0)/len(sells)*100:.1f}%")

print()

# Day-by-day BUY vs SELL
print("DAY-BY-DAY BUY vs SELL:")
date_groups = defaultdict(lambda: {'buy': [], 'sell': []})

for fname in json_files:
    m = re.search(r'(\d{4}-\d{2}-\d{2})', fname)
    if not m:
        continue
    date = m.group(1)
    fpath = os.path.join(RESULTS_DIR, fname)
    try:
        with open(fpath) as f:
            result = json.load(f)
        for t in result.get('trades', []):
            if 'side' not in t or 'pnl_dollars' not in t:
                continue
            if t['side'] == 'BUY':
                date_groups[date]['buy'].append(t['pnl_dollars'])
            else:
                date_groups[date]['sell'].append(t['pnl_dollars'])
    except Exception:
        pass

print(f"{'Date':<12} {'BuyN':>6} {'BuyPnL':>12} {'BuyAvg':>10} {'SellN':>6} {'SellPnL':>12} {'SellAvg':>10}")
print("-" * 70)
for date in sorted(date_groups.keys()):
    buys = date_groups[date]['buy']
    sells = date_groups[date]['sell']
    b_total = sum(buys)
    s_total = sum(sells)
    b_avg = b_total / len(buys) if buys else 0
    s_avg = s_total / len(sells) if sells else 0
    print(f"{date:<12} {len(buys):>6} {b_total:>+12,.0f} {b_avg:>+10.2f} {len(sells):>6} {s_total:>+12,.0f} {s_avg:>+10.2f}")
print()

# Signal strength analysis: do SELLs have weaker signals?
print("SIGNAL STRENGTH: BUY vs SELL")
all_trades_flat = []
for fname in json_files[:1000]:  # sample first 1000 files for speed
    fpath = os.path.join(RESULTS_DIR, fname)
    try:
        with open(fpath) as f:
            result = json.load(f)
        all_trades_flat.extend(result.get('trades', []))
    except Exception:
        pass

buy_signals = [abs(t.get('signal_strength', 0)) for t in all_trades_flat if t.get('side') == 'BUY' and 'signal_strength' in t]
sell_signals = [abs(t.get('signal_strength', 0)) for t in all_trades_flat if t.get('side') == 'SELL' and 'signal_strength' in t]

if buy_signals and sell_signals:
    print(f"  BUY  avg |signal strength|: {sum(buy_signals)/len(buy_signals):.4f} (n={len(buy_signals)})")
    print(f"  SELL avg |signal strength|: {sum(sell_signals)/len(sell_signals):.4f} (n={len(sell_signals)})")
    print()

    # Queue position analysis
    buy_queue = [t.get('queue_position_at_post', 0) for t in all_trades_flat if t.get('side') == 'BUY']
    sell_queue = [t.get('queue_position_at_post', 0) for t in all_trades_flat if t.get('side') == 'SELL']
    if buy_queue and sell_queue:
        print(f"  BUY  avg queue position: {sum(buy_queue)/len(buy_queue):.1f}")
        print(f"  SELL avg queue position: {sum(sell_queue)/len(sell_queue):.1f}")
        print("  (Higher queue = worse fill = more adverse selection)")

print()
print("=" * 70)
print("Analysis complete.")
