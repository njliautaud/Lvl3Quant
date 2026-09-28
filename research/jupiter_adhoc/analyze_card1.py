import json, os, sys
from collections import defaultdict
from datetime import datetime
import math

DATA_DIR = '/home/jupiter/Lvl3Quant/data/processed/best_config_sweep'

# =====================================================================
# LOAD ALL DATA
# =====================================================================
configs = {}  # config_name -> {date -> data}
all_dates = set()

for fname in sorted(os.listdir(DATA_DIR)):
    if not fname.endswith('.json'):
        continue
    parts = fname.replace('.json', '').split('_')
    date = parts[0]
    config_name = '_'.join(parts[1:])

    fpath = os.path.join(DATA_DIR, fname)
    with open(fpath) as f:
        data = json.load(f)

    if config_name not in configs:
        configs[config_name] = {}
    configs[config_name][date] = data
    all_dates.add(date)

all_dates = sorted(all_dates)
print(f'Loaded {len(configs)} configs across {len(all_dates)} dates')
print(f'Configs: {sorted(configs.keys())}')
print()

# =====================================================================
# SECTION 1: HEAD-TO-HEAD COMPARISON OF ALL 12 CONFIGS
# =====================================================================
print('=' * 100)
print('SECTION 1: HEAD-TO-HEAD COMPARISON - ALL 12 CONFIGS')
print('=' * 100)

config_stats = []
for cfg_name in sorted(configs.keys()):
    cfg_data = configs[cfg_name]
    total_pnl = 0
    total_trades = 0
    total_signals = 0
    total_filled = 0
    daily_pnls = []
    all_trades_cfg = []
    wins = 0
    losses = 0
    gross_win = 0
    gross_loss = 0

    for date in all_dates:
        if date not in cfg_data:
            daily_pnls.append(0)
            continue
        d = cfg_data[date]
        day_pnl = d['total_pnl_dollars']
        total_pnl += day_pnl
        total_trades += d['total_trades']
        total_signals += d.get('total_signals', 0)
        total_filled += d.get('total_filled', 0)
        daily_pnls.append(day_pnl)

        for t in d.get('trades', []):
            all_trades_cfg.append(t)
            if t['pnl_dollars'] > 0:
                wins += 1
                gross_win += t['pnl_dollars']
            else:
                losses += 1
                gross_loss += abs(t['pnl_dollars'])

    wr = wins / max(1, wins + losses) * 100
    pf = gross_win / max(0.01, gross_loss)
    n_days = len(daily_pnls)
    winning_days = sum(1 for p in daily_pnls if p > 0)
    zero_days = sum(1 for p in daily_pnls if p == 0)
    active_days = n_days - zero_days
    avg_daily = total_pnl / max(1, n_days)
    med_daily = sorted(daily_pnls)[len(daily_pnls)//2] if daily_pnls else 0

    if len(daily_pnls) > 1:
        mean_d = sum(daily_pnls) / len(daily_pnls)
        var_d = sum((x - mean_d)**2 for x in daily_pnls) / (len(daily_pnls) - 1)
        std_d = math.sqrt(var_d) if var_d > 0 else 0.001
        sharpe = (mean_d / std_d) * math.sqrt(252) if std_d > 0 else 0
    else:
        sharpe = 0
        std_d = 0

    cum = 0
    peak = 0
    max_dd = 0
    for p in daily_pnls:
        cum += p
        peak = max(peak, cum)
        dd = peak - cum
        max_dd = max(max_dd, dd)

    fill_rate = total_filled / max(1, total_signals) * 100 if total_signals > 0 else 0

    config_stats.append({
        'name': cfg_name, 'total_pnl': total_pnl, 'trades': total_trades,
        'wr': wr, 'pf': pf, 'sharpe': sharpe, 'avg_daily': avg_daily,
        'med_daily': med_daily, 'max_dd': max_dd, 'winning_day_pct': winning_days/max(1,active_days)*100,
        'winning_days': winning_days, 'zero_days': zero_days, 'n_days': n_days,
        'active_days': active_days,
        'daily_std': std_d, 'fill_rate': fill_rate, 'total_signals': total_signals,
        'gross_win': gross_win, 'gross_loss': gross_loss
    })

config_stats.sort(key=lambda x: -x['total_pnl'])

print(f'\n{"Config":<25} {"PnL($)":>9} {"Trades":>7} {"WR%":>6} {"PF":>6} {"Sharpe":>7} {"AvgD($)":>9} {"MedD($)":>9} {"MaxDD":>9} {"WinD":>7} {"Fill%":>7}')
print('-' * 120)
for s in config_stats:
    print(f'{s["name"]:<25} {s["total_pnl"]:>9.1f} {s["trades"]:>7} {s["wr"]:>5.1f}% {s["pf"]:>6.2f} {s["sharpe"]:>7.2f} {s["avg_daily"]:>9.2f} {s["med_daily"]:>9.2f} {s["max_dd"]:>9.1f} {s["winning_days"]}/{s["active_days"]:<2} {s["fill_rate"]:>6.1f}%')

# =====================================================================
# SECTION 2: DEEP DIVE - CARD 1 (tp3_trail25_wb50)
# =====================================================================
CARD1 = 'trail25_tp3_wb50'
print(f'\n\n{"="*100}')
print(f'SECTION 2: DEEP DIVE - CARD 1 ({CARD1})')
print(f'{"="*100}')

c1 = configs[CARD1]
c1_trades = []
c1_daily = []

for date in all_dates:
    if date not in c1:
        c1_daily.append({'date': date, 'pnl': 0, 'trades': 0, 'signals': 0, 'filled': 0, 'trades_list': []})
        continue
    d = c1[date]
    c1_daily.append({'date': date, 'pnl': d['total_pnl_dollars'], 'trades': d['total_trades'],
                     'signals': d.get('total_signals',0), 'filled': d.get('total_filled',0),
                     'trades_list': d.get('trades', [])})
    for t in d.get('trades', []):
        t['date'] = date
        c1_trades.append(t)

print(f'\n--- Per-Day PnL Breakdown ---')
print(f'{"Date":<12} {"PnL($)":>9} {"Trades":>7} {"Signals":>8} {"CumPnL":>10}')
print('-' * 55)
cum = 0
for day in c1_daily:
    cum += day['pnl']
    sig = day.get('signals', 0)
    print(f'{day["date"]:<12} {day["pnl"]:>9.2f} {day["trades"]:>7} {sig:>8} {cum:>10.2f}')

# L/S split
print(f'\n--- Long vs Short Split ---')
for side_label, side_val in [('LONG (BUY)', 'BUY'), ('SHORT (SELL)', 'SELL')]:
    side_trades = [t for t in c1_trades if t['side'] == side_val]
    if not side_trades:
        print(f'{side_label}: 0 trades')
        continue
    w = sum(1 for t in side_trades if t['pnl_dollars'] > 0)
    l = len(side_trades) - w
    gw = sum(t['pnl_dollars'] for t in side_trades if t['pnl_dollars'] > 0)
    gl = sum(abs(t['pnl_dollars']) for t in side_trades if t['pnl_dollars'] <= 0)
    pnl_list = [t['pnl_dollars'] for t in side_trades]
    avg_pnl = sum(pnl_list) / len(pnl_list)
    if len(pnl_list) > 1:
        m = sum(pnl_list)/len(pnl_list)
        v = sum((x-m)**2 for x in pnl_list)/(len(pnl_list)-1)
        s = math.sqrt(v) if v > 0 else 0.001
        sh = m / s * math.sqrt(252) if s > 0 else 0
    else:
        sh = 0
    wr = w / max(1, w+l) * 100
    pf = gw / max(0.01, gl)
    print(f'{side_label}: {len(side_trades)} trades | WR={wr:.1f}% | PF={pf:.2f} | Sharpe={sh:.2f} | PnL=${sum(pnl_list):.2f} | AvgPnL=${avg_pnl:.2f}')

# Daily PnL distribution
print(f'\n--- Daily PnL Distribution ---')
daily_pnls = [d['pnl'] for d in c1_daily]
buckets = [(-999,-200), (-200,-100), (-100,-50), (-50,0), (0,0.001), (0.001,50), (50,100), (100,200), (200,500), (500,999)]
for lo, hi in buckets:
    count = sum(1 for p in daily_pnls if lo <= p < hi)
    if count > 0:
        label = f'${lo:.0f} to ${hi:.0f}'
        if lo == 0 and hi == 0.001:
            label = '$0 (flat)'
        elif lo == 0.001:
            label = '$0 to $50'
        print(f'  {label:<20}: {count} days {"#"*count}')

# Consistency metrics
print(f'\n--- Consistency Metrics ---')
winning_days = sum(1 for p in daily_pnls if p > 0)
losing_days = sum(1 for p in daily_pnls if p < 0)
zero_days = sum(1 for p in daily_pnls if p == 0)
non_zero_pnls = [p for p in daily_pnls if p != 0]
print(f'Total days: {len(daily_pnls)}')
print(f'Winning days: {winning_days} ({winning_days/len(daily_pnls)*100:.1f}%)')
print(f'Losing days: {losing_days} ({losing_days/len(daily_pnls)*100:.1f}%)')
print(f'Zero days (no trades): {zero_days} ({zero_days/len(daily_pnls)*100:.1f}%)')
if non_zero_pnls:
    print(f'Of days with trades: {winning_days}/{winning_days+losing_days} winning = {winning_days/(winning_days+losing_days)*100:.1f}%')
    print(f'Median daily PnL (all): ${sorted(daily_pnls)[len(daily_pnls)//2]:.2f}')
    print(f'Median daily PnL (non-zero): ${sorted(non_zero_pnls)[len(non_zero_pnls)//2]:.2f}')
    print(f'Mean daily PnL: ${sum(daily_pnls)/len(daily_pnls):.2f}')
    var_d = sum((x - sum(daily_pnls)/len(daily_pnls))**2 for x in daily_pnls) / (len(daily_pnls)-1)
    print(f'Daily PnL Std: ${math.sqrt(var_d):.2f}')
    print(f'Best day: ${max(daily_pnls):.2f}')
    print(f'Worst day: ${min(non_zero_pnls):.2f}')

# Streak analysis
print(f'\n--- Streak Analysis ---')
max_consec_lose = 0
cur_lose = 0
max_consec_win = 0
cur_win = 0
for p in daily_pnls:
    if p < 0:
        cur_lose += 1
        cur_win = 0
        max_consec_lose = max(max_consec_lose, cur_lose)
    elif p > 0:
        cur_win += 1
        cur_lose = 0
        max_consec_win = max(max_consec_win, cur_win)
    else:
        cur_lose = 0
        cur_win = 0
print(f'Max consecutive losing days: {max_consec_lose}')
print(f'Max consecutive winning days: {max_consec_win}')

max_intra_lose = 0
cur_il = 0
for t in c1_trades:
    if t['pnl_dollars'] < 0:
        cur_il += 1
        max_intra_lose = max(max_intra_lose, cur_il)
    else:
        cur_il = 0
print(f'Max consecutive losing TRADES: {max_intra_lose}')

# Drawdown
print(f'\n--- Drawdown Analysis ---')
cum = 0
peak = 0
max_dd = 0
dd_start = None
dd_end = None
cur_dd_start = None
for i, day in enumerate(c1_daily):
    cum += day['pnl']
    if cum > peak:
        peak = cum
        cur_dd_start = all_dates[i]
    dd = peak - cum
    if dd > max_dd:
        max_dd = dd
        dd_start = cur_dd_start
        dd_end = all_dates[i]
print(f'Max Drawdown: ${max_dd:.2f}')
print(f'DD period: {dd_start} to {dd_end}')
print(f'Peak equity: ${peak:.2f}')

# Regime analysis
print(f'\n--- Regime Analysis (Monthly) ---')
months = defaultdict(list)
for day in c1_daily:
    month = day['date'][:7]
    months[month].append(day)

print(f'{"Month":<10} {"PnL($)":>9} {"Trades":>7} {"Days":>5} {"WinDays":>8} {"AvgDaily":>9}')
print('-' * 55)
for month in sorted(months.keys()):
    days = months[month]
    pnl = sum(d['pnl'] for d in days)
    trades = sum(d['trades'] for d in days)
    n = len(days)
    w = sum(1 for d in days if d['pnl'] > 0)
    avg = pnl / n
    print(f'{month:<10} {pnl:>9.2f} {trades:>7} {n:>5} {w}/{n:<5} {avg:>9.2f}')

# Fill rate
print(f'\n--- Fill Rate ---')
total_sig = sum(d.get('signals',0) for d in c1_daily)
total_fill = sum(d.get('filled',0) for d in c1_daily)
total_trades_c1 = sum(d['trades'] for d in c1_daily)
print(f'Total signals: {total_sig}')
print(f'Total fills: {total_fill}')
print(f'Total trades: {total_trades_c1}')
if total_sig > 0:
    print(f'Fill rate: {total_fill/total_sig*100:.1f}%')

# Exit reasons
print(f'\n--- Exit Reasons ---')
exit_reasons = defaultdict(int)
for t in c1_trades:
    exit_reasons[t.get('exit_reason','unknown')] += 1
for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
    pnl_for_reason = sum(t['pnl_dollars'] for t in c1_trades if t.get('exit_reason') == reason)
    wr_r = sum(1 for t in c1_trades if t.get('exit_reason') == reason and t['pnl_dollars'] > 0) / max(1, count) * 100
    print(f'  {reason:<20}: {count:>4} trades | PnL=${pnl_for_reason:>8.2f} | WR={wr_r:.1f}%')

# MAE/MFE
print(f'\n--- Per-Trade MAE/MFE Distribution ---')
maes = [t['mae_ticks'] for t in c1_trades if 'mae_ticks' in t]
mfes = [t['mfe_ticks'] for t in c1_trades if 'mfe_ticks' in t]
if maes:
    print(f'MAE (all): min={min(maes):.1f}, p25={sorted(maes)[int(len(maes)*0.25)]:.1f}, med={sorted(maes)[len(maes)//2]:.1f}, mean={sum(maes)/len(maes):.1f}, p75={sorted(maes)[int(len(maes)*0.75)]:.1f}, p90={sorted(maes)[int(len(maes)*0.9)]:.1f}, max={max(maes):.1f}')
if mfes:
    print(f'MFE (all): min={min(mfes):.1f}, p25={sorted(mfes)[int(len(mfes)*0.25)]:.1f}, med={sorted(mfes)[len(mfes)//2]:.1f}, mean={sum(mfes)/len(mfes):.1f}, p75={sorted(mfes)[int(len(mfes)*0.75)]:.1f}, p90={sorted(mfes)[int(len(mfes)*0.9)]:.1f}, max={max(mfes):.1f}')

# =====================================================================
# SECTION 3: WINNER vs LOSER ANALYSIS
# =====================================================================
print(f'\n\n{"="*100}')
print(f'SECTION 3: WINNER vs LOSER DEEP ANALYSIS')
print(f'{"="*100}')

winners = [t for t in c1_trades if t['pnl_dollars'] > 0]
losers = [t for t in c1_trades if t['pnl_dollars'] <= 0]

print(f'\nWinners: {len(winners)} | Losers: {len(losers)}')

def ns_to_sec(ns):
    return ns / 1e9

if winners:
    w_holds = [ns_to_sec(t['hold_duration_ns']) for t in winners if 'hold_duration_ns' in t]
    print(f'\n--- Winner Hold Time (seconds) ---')
    if w_holds:
        print(f'  min={min(w_holds):.1f}s, med={sorted(w_holds)[len(w_holds)//2]:.1f}s, mean={sum(w_holds)/len(w_holds):.1f}s, max={max(w_holds):.1f}s')
        for lo, hi, label in [(0,30,'0-30s'), (30,60,'30-60s'), (60,120,'1-2min'), (120,300,'2-5min'), (300,600,'5-10min'), (600,1800,'10-30min'), (1800,9999,'30min+')]:
            c = sum(1 for h in w_holds if lo <= h < hi)
            if c > 0:
                print(f'    {label:<12}: {c:>3} ({c/len(w_holds)*100:.0f}%)')

if losers:
    l_holds = [ns_to_sec(t['hold_duration_ns']) for t in losers if 'hold_duration_ns' in t]
    print(f'\n--- Loser Hold Time (seconds) ---')
    if l_holds:
        print(f'  min={min(l_holds):.1f}s, med={sorted(l_holds)[len(l_holds)//2]:.1f}s, mean={sum(l_holds)/len(l_holds):.1f}s, max={max(l_holds):.1f}s')
        for lo, hi, label in [(0,30,'0-30s'), (30,60,'30-60s'), (60,120,'1-2min'), (120,300,'2-5min'), (300,600,'5-10min'), (600,1800,'10-30min'), (1800,9999,'30min+')]:
            c = sum(1 for h in l_holds if lo <= h < hi)
            if c > 0:
                print(f'    {label:<12}: {c:>3} ({c/len(l_holds)*100:.0f}%)')

# CRITICAL: Winner MAE
print(f'\n--- CRITICAL: Winner MAE (how much do winners dip?) ---')
if winners:
    w_maes = [t['mae_ticks'] for t in winners if 'mae_ticks' in t]
    if w_maes:
        print(f'  Winner MAE: min={min(w_maes):.1f}, med={sorted(w_maes)[len(w_maes)//2]:.1f}, mean={sum(w_maes)/len(w_maes):.1f}, max={max(w_maes):.1f}')
        print(f'  p10={sorted(w_maes)[int(len(w_maes)*0.1)]:.1f}, p25={sorted(w_maes)[int(len(w_maes)*0.25)]:.1f}, p75={sorted(w_maes)[int(len(w_maes)*0.75)]:.1f}, p90={sorted(w_maes)[int(len(w_maes)*0.9)]:.1f}')
        for lo, hi, label in [(0,0.5,'0t'), (0.5,1.5,'1t'), (1.5,2.5,'2t'), (2.5,3.5,'3t'), (3.5,5.5,'4-5t'), (5.5,10.5,'6-10t'), (10.5,15.5,'11-15t'), (15.5,20.5,'16-20t'), (20.5,999,'20+t')]:
            c = sum(1 for m in w_maes if lo <= m < hi)
            if c > 0:
                print(f'    MAE {label:<8}: {c:>3} ({c/len(w_maes)*100:.1f}%)')

        print(f'\n  Winners surviving different SL levels:')
        for sl in [3, 5, 7, 10, 15, 20, 25]:
            survive = sum(1 for m in w_maes if m < sl)
            print(f'    SL{sl:<3}: {survive}/{len(w_maes)} survive ({survive/len(w_maes)*100:.1f}%)')

# Loser MFE
print(f'\n--- Loser MFE (how close do losers get to profit?) ---')
if losers:
    l_mfes = [t['mfe_ticks'] for t in losers if 'mfe_ticks' in t]
    if l_mfes:
        print(f'  Loser MFE: min={min(l_mfes):.1f}, med={sorted(l_mfes)[len(l_mfes)//2]:.1f}, mean={sum(l_mfes)/len(l_mfes):.1f}, max={max(l_mfes):.1f}')
        for lo, hi, label in [(0,0.5,'0t (never green)'), (0.5,1.5,'1t'), (1.5,2.5,'2t'), (2.5,3.5,'3t (near TP!)'), (3.5,5.5,'4-5t'), (5.5,10.5,'6-10t'), (10.5,999,'10+t')]:
            c = sum(1 for m in l_mfes if lo <= m < hi)
            if c > 0:
                print(f'    MFE {label:<20}: {c:>3} ({c/len(l_mfes)*100:.1f}%)')

# Loser MAE
print(f'\n--- Loser MAE ---')
if losers:
    l_maes = [t['mae_ticks'] for t in losers if 'mae_ticks' in t]
    if l_maes:
        print(f'  Loser MAE: min={min(l_maes):.1f}, med={sorted(l_maes)[len(l_maes)//2]:.1f}, mean={sum(l_maes)/len(l_maes):.1f}, max={max(l_maes):.1f}')

# Signal strength
print(f'\n--- Signal Strength: Winners vs Losers ---')
w_sig = [t['signal_strength'] for t in winners if 'signal_strength' in t]
l_sig = [t['signal_strength'] for t in losers if 'signal_strength' in t]
if w_sig:
    print(f'  Winner signal: mean={sum(w_sig)/len(w_sig):.3f}, med={sorted(w_sig)[len(w_sig)//2]:.3f}, min={min(w_sig):.3f}, max={max(w_sig):.3f}')
if l_sig:
    print(f'  Loser signal:  mean={sum(l_sig)/len(l_sig):.3f}, med={sorted(l_sig)[len(l_sig)//2]:.3f}, min={min(l_sig):.3f}, max={max(l_sig):.3f}')

# Signal strength buckets with WR
print(f'\n--- Win Rate by Signal Strength Bucket ---')
all_sigs = [(t['signal_strength'], t['pnl_dollars'] > 0) for t in c1_trades if 'signal_strength' in t]
if all_sigs:
    for lo, hi in [(0,1), (1,2), (2,3), (3,5), (5,10), (10,999)]:
        bucket = [(s, w) for s, w in all_sigs if lo <= abs(s) < hi]
        if bucket:
            n = len(bucket)
            w = sum(1 for _, win in bucket if win)
            avg_s = sum(abs(s) for s, _ in bucket) / n
            print(f'  |sig| {lo}-{hi}: {n} trades, WR={w/n*100:.1f}%, avg_sig={avg_s:.2f}')

# Exit reason breakdown
print(f'\n--- Exit Reason: Winners vs Losers ---')
for reason in sorted(exit_reasons.keys()):
    w_count = sum(1 for t in winners if t.get('exit_reason') == reason)
    l_count = sum(1 for t in losers if t.get('exit_reason') == reason)
    w_pnl = sum(t['pnl_dollars'] for t in winners if t.get('exit_reason') == reason)
    l_pnl = sum(t['pnl_dollars'] for t in losers if t.get('exit_reason') == reason)
    print(f'  {reason:<20}: W={w_count:>3} (${w_pnl:>8.2f}) | L={l_count:>3} (${l_pnl:>8.2f})')

# Queue position
print(f'\n--- Queue Position at Entry ---')
w_queue = [t['queue_position_at_post'] for t in winners if 'queue_position_at_post' in t and t['queue_position_at_post'] > 0]
l_queue = [t['queue_position_at_post'] for t in losers if 'queue_position_at_post' in t and t['queue_position_at_post'] > 0]
if w_queue:
    print(f'  Winner queue pos: mean={sum(w_queue)/len(w_queue):.1f}, med={sorted(w_queue)[len(w_queue)//2]:.1f}')
if l_queue:
    print(f'  Loser queue pos:  mean={sum(l_queue)/len(l_queue):.1f}, med={sorted(l_queue)[len(l_queue)//2]:.1f}')

# Fill latency
w_lat = [t['fill_latency_ns']/1e6 for t in winners if 'fill_latency_ns' in t]
l_lat = [t['fill_latency_ns']/1e6 for t in losers if 'fill_latency_ns' in t]
if w_lat:
    print(f'  Winner fill latency: mean={sum(w_lat)/len(w_lat):.0f}ms, med={sorted(w_lat)[len(w_lat)//2]:.0f}ms')
if l_lat:
    print(f'  Loser fill latency:  mean={sum(l_lat)/len(l_lat):.0f}ms, med={sorted(l_lat)[len(l_lat)//2]:.0f}ms')

# =====================================================================
# SECTION 4: OPTIMIZATION - DIRECT CONFIG COMPARISONS
# =====================================================================
print(f'\n\n{"="*100}')
print(f'SECTION 4: OPTIMIZATION - DIRECT CONFIG COMPARISONS')
print(f'{"="*100}')

def summarize_cfg(name):
    for s in config_stats:
        if s['name'] == name:
            return s
    return None

# Trail comparison
print(f'\n--- Trailing Stop: trail15 vs trail20 vs trail25 (TP3, WB50) ---')
print(f'{"Config":<25} {"PnL($)":>9} {"Trades":>7} {"WR%":>6} {"PF":>6} {"Sharpe":>7} {"MaxDD":>9} {"WinDays":>8}')
for trail in ['trail15_tp3_wb50', 'trail20_tp3_wb50', 'trail25_tp3_wb50']:
    s = summarize_cfg(trail)
    if s:
        print(f'{s["name"]:<25} {s["total_pnl"]:>9.1f} {s["trades"]:>7} {s["wr"]:>5.1f}% {s["pf"]:>6.2f} {s["sharpe"]:>7.2f} {s["max_dd"]:>9.1f} {s["winning_days"]}/{s["active_days"]}')

print(f'\n--- Trailing Stop: trail15 vs trail20 vs trail25 (TP3, WB30) ---')
for trail in ['trail15_tp3_wb30', 'trail20_tp3_wb30', 'trail25_tp3_wb30']:
    s = summarize_cfg(trail)
    if s:
        print(f'{s["name"]:<25} {s["total_pnl"]:>9.1f} {s["trades"]:>7} {s["wr"]:>5.1f}% {s["pf"]:>6.2f} {s["sharpe"]:>7.2f} {s["max_dd"]:>9.1f} {s["winning_days"]}/{s["active_days"]}')

# TP comparison
print(f'\n--- Take Profit: TP3 vs TP5 (all trail/wb combos) ---')
print(f'{"Config":<25} {"PnL($)":>9} {"Trades":>7} {"WR%":>6} {"PF":>6} {"Sharpe":>7} {"MaxDD":>9} {"WinDays":>8}')
for trail in [15, 20, 25]:
    for wb in [30, 50]:
        for tp in [3, 5]:
            name = f'trail{trail}_tp{tp}_wb{wb}'
            s = summarize_cfg(name)
            if s:
                print(f'{s["name"]:<25} {s["total_pnl"]:>9.1f} {s["trades"]:>7} {s["wr"]:>5.1f}% {s["pf"]:>6.2f} {s["sharpe"]:>7.2f} {s["max_dd"]:>9.1f} {s["winning_days"]}/{s["active_days"]}')
    print()

# WB comparison
print(f'\n--- Wait Bars: WB30 vs WB50 (TP3 configs) ---')
print(f'{"Config":<25} {"PnL($)":>9} {"Trades":>7} {"WR%":>6} {"PF":>6} {"Sharpe":>7} {"MaxDD":>9} {"WinDays":>8}')
for trail in [15, 20, 25]:
    for wb in [30, 50]:
        name = f'trail{trail}_tp3_wb{wb}'
        s = summarize_cfg(name)
        if s:
            print(f'{s["name"]:<25} {s["total_pnl"]:>9.1f} {s["trades"]:>7} {s["wr"]:>5.1f}% {s["pf"]:>6.2f} {s["sharpe"]:>7.2f} {s["max_dd"]:>9.1f} {s["winning_days"]}/{s["active_days"]}')

# Ranking
print(f'\n\n--- FINAL RANKING (all 12 configs, sorted by Sharpe) ---')
by_sharpe = sorted(config_stats, key=lambda x: -x['sharpe'])
print(f'{"Rank":<5} {"Config":<25} {"PnL($)":>9} {"Sharpe":>7} {"WR%":>6} {"PF":>6} {"Trades":>7} {"MaxDD":>9}')
print('-' * 85)
for i, s in enumerate(by_sharpe):
    marker = ' <-- CARD 1' if s['name'] == CARD1 else ''
    print(f'{i+1:<5} {s["name"]:<25} {s["total_pnl"]:>9.1f} {s["sharpe"]:>7.2f} {s["wr"]:>5.1f}% {s["pf"]:>6.2f} {s["trades"]:>7} {s["max_dd"]:>9.1f}{marker}')

# Per-trade PnL distribution
print(f'\n--- Card 1 Per-Trade PnL Distribution ---')
trade_pnls = [t['pnl_dollars'] for t in c1_trades]
if trade_pnls:
    for lo, hi, label in [(-500,-200,'-$500 to -$200'), (-200,-100,'-$200 to -$100'), (-100,-50,'-$100 to -$50'), (-50,0,'-$50 to $0'), (0,0.01,'$0 (BE)'), (0.01,25,'$0-$25'), (25,50,'$25-$50'), (50,100,'$50-$100'), (100,200,'$100-$200')]:
        c = sum(1 for p in trade_pnls if lo <= p < hi)
        if c > 0:
            print(f'    {label:<18}: {c:>3} trades')

    avg_win_pnl = sum(t['pnl_dollars'] for t in winners) / max(1,len(winners)) if winners else 0
    avg_loss_pnl = sum(t['pnl_dollars'] for t in losers) / max(1,len(losers)) if losers else 0
    print(f'\n  Avg winner: ${avg_win_pnl:.2f}')
    print(f'  Avg loser: ${avg_loss_pnl:.2f}')
    if avg_win_pnl > 0:
        print(f'  Loss/Win ratio: {abs(avg_loss_pnl)/avg_win_pnl:.1f}:1')

# Time-of-day
print(f'\n--- Time-of-Day Analysis (Card 1) ---')
hour_stats = defaultdict(lambda: {'trades': 0, 'pnl': 0, 'wins': 0})
for t in c1_trades:
    if 'fill_time_ns' in t and t['fill_time_ns'] > 0:
        ts = t['fill_time_ns'] / 1e9
        try:
            dt = datetime.utcfromtimestamp(ts)
            et_hour = (dt.hour - 5) % 24
            hour_stats[et_hour]['trades'] += 1
            hour_stats[et_hour]['pnl'] += t['pnl_dollars']
            if t['pnl_dollars'] > 0:
                hour_stats[et_hour]['wins'] += 1
        except:
            pass

if hour_stats:
    print(f'{"Hour(ET)":<10} {"Trades":>7} {"PnL($)":>9} {"WR%":>6} {"AvgPnL":>9}')
    for h in sorted(hour_stats.keys()):
        s = hour_stats[h]
        wr = s['wins']/max(1,s['trades'])*100
        avg = s['pnl']/max(1,s['trades'])
        print(f'  {h:>2}:00     {s["trades"]:>7} {s["pnl"]:>9.2f} {wr:>5.1f}% {avg:>9.2f}')

print(f'\n\n{"="*100}')
print('ANALYSIS COMPLETE')
print(f'{"="*100}')
