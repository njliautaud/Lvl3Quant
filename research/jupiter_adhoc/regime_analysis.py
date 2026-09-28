import json
import os
import math
from collections import defaultdict

results_dir = '/home/jupiter/lvl3quant/results/composite_mbo/t3.5_h900s'

monthly_data = defaultdict(lambda: {
    'pnl': 0.0,
    'trades': 0,
    'signals': 0,
    'wins': 0,
    'losses': 0,
    'daily_pnls': [],
    'days': []
})

all_files = sorted(os.listdir(results_dir))
print("Total files:", len(all_files))

for fname in all_files:
    if not fname.endswith('.json'):
        continue
    date_str = fname.replace('.json', '')
    month_key = date_str[:7]
    
    fpath = os.path.join(results_dir, fname)
    with open(fpath) as f:
        d = json.load(f)
    
    pnl = d.get('total_pnl_dollars', 0)
    trades = d.get('total_trades', 0)
    signals = d.get('total_signals', 0)
    win_rate = d.get('win_rate', 0)
    wins = round(trades * win_rate) if trades > 0 else 0
    losses = trades - wins
    
    monthly_data[month_key]['pnl'] += pnl
    monthly_data[month_key]['trades'] += trades
    monthly_data[month_key]['signals'] += signals
    monthly_data[month_key]['wins'] += wins
    monthly_data[month_key]['losses'] += losses
    monthly_data[month_key]['daily_pnls'].append(pnl)
    monthly_data[month_key]['days'].append(date_str)

def sharpe(daily_pnls):
    n = len(daily_pnls)
    if n < 2:
        return None
    mean = sum(daily_pnls) / n
    var = sum((x - mean)**2 for x in daily_pnls) / (n - 1)
    std = math.sqrt(var) if var > 0 else 0
    if std == 0:
        return None
    return (mean / std) * math.sqrt(252)

print("")
print("=== MONTHLY BREAKDOWN: Original 9-Signal MEAN_ZSCORE Composite (t3.5, h900s) ===")
print("")
print("{:<12} {:>5} {:>10} {:>7} {:>8} {:>6} {:>12}".format('Month','Days','PnL $','Trades','Signals','Win%','Sharpe(ann)'))
print("-" * 65)

total_pnl = 0
total_trades = 0
all_daily_pnls = []

for month in sorted(monthly_data.keys()):
    m = monthly_data[month]
    daily_pnls = m['daily_pnls']
    all_daily_pnls.extend(daily_pnls)
    
    n_days = len(daily_pnls)
    pnl = m['pnl']
    trades = m['trades']
    signals = m['signals']
    wins = m['wins']
    win_pct = (wins / trades * 100) if trades > 0 else 0
    sh = sharpe(daily_pnls)
    sh_str = "{:.2f}".format(sh) if sh is not None else "N/A"
    
    total_pnl += pnl
    total_trades += trades
    
    print("{:<12} {:>5} {:>10.2f} {:>7} {:>8} {:>5.1f}% {:>12}".format(
        month, n_days, pnl, trades, signals, win_pct, sh_str))
    
    # Print daily breakdown
    for day, dpnl in zip(m['days'], daily_pnls):
        print("  {} {:>8.2f}".format(day, dpnl))

print("-" * 65)
overall_sharpe = sharpe(all_daily_pnls)
sh_str = "{:.2f}".format(overall_sharpe) if overall_sharpe is not None else "N/A"
print("{:<12} {:>5} {:>10.2f} {:>7}  {:>8}         {:>12}".format(
    'TOTAL', len(all_daily_pnls), total_pnl, total_trades, '', sh_str))

print("")
print("=== REGIME SPLIT: Jul-Sep vs Oct-Nov ===")
early_months = [m for m in sorted(monthly_data.keys()) if m <= '2025-09']
late_months = [m for m in sorted(monthly_data.keys()) if m >= '2025-10']

early_pnl = sum(monthly_data[m]['pnl'] for m in early_months)
late_pnl = sum(monthly_data[m]['pnl'] for m in late_months)
early_days = sum(len(monthly_data[m]['daily_pnls']) for m in early_months)
late_days = sum(len(monthly_data[m]['daily_pnls']) for m in late_months)
early_trades = sum(monthly_data[m]['trades'] for m in early_months)
late_trades = sum(monthly_data[m]['trades'] for m in late_months)
early_signals = sum(monthly_data[m]['signals'] for m in early_months)
late_signals = sum(monthly_data[m]['signals'] for m in late_months)

early_daily = [p for m in early_months for p in monthly_data[m]['daily_pnls']]
late_daily = [p for m in late_months for p in monthly_data[m]['daily_pnls']]

early_sh = sharpe(early_daily)
late_sh = sharpe(late_daily)
early_sh_str = "{:.2f}".format(early_sh) if early_sh is not None else "N/A"
late_sh_str = "{:.2f}".format(late_sh) if late_sh is not None else "N/A"

early_wins = sum(monthly_data[m]['wins'] for m in early_months)
late_wins = sum(monthly_data[m]['wins'] for m in late_months)
early_win_pct = (early_wins / early_trades * 100) if early_trades > 0 else 0
late_win_pct = (late_wins / late_trades * 100) if late_trades > 0 else 0

print("Jul-Sep ({} days): {} trades, {} signals, PnL=${:.2f}, Win={:.1f}%, Sharpe={}".format(
    early_days, early_trades, early_signals, early_pnl, early_win_pct, early_sh_str))
print("Oct-Nov ({} days): {} trades, {} signals, PnL=${:.2f}, Win={:.1f}%, Sharpe={}".format(
    late_days, late_trades, late_signals, late_pnl, late_win_pct, late_sh_str))

# Avg PnL per day
print("")
print("Avg daily PnL Jul-Sep: ${:.2f}".format(early_pnl/early_days if early_days else 0))
print("Avg daily PnL Oct-Nov: ${:.2f}".format(late_pnl/late_days if late_days else 0))
print("Avg PnL per trade Jul-Sep: ${:.2f}".format(early_pnl/early_trades if early_trades else 0))
print("Avg PnL per trade Oct-Nov: ${:.2f}".format(late_pnl/late_trades if late_trades else 0))
