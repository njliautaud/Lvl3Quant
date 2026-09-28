#!/usr/bin/env python3
"""
Daily P&L Time Series Analysis
================================
Best config: CNN-Mamba v2, top 3% short, meta-filtered (top 50%), passive entry, 1s hold.
Uses confluence_meta_v2 OOT predictions (already top-3% short, with realized net P&L).
Applies meta_score > 0 filter (top ~50% of meta predictions).
Reconstructs labels_1s P&L from raw MBO data (single pass).

Cost: 0.376 ticks (passive RT commission).
"""

import json
import sys
from pathlib import Path

import numpy as np

# Force unbuffered output
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)

# === PATHS ===
META_FILE = Path('/home/jupiter/Lvl3Quant/output/confluence_meta_v2/oot_predictions.npz')
RESULTS_FILE = Path('/home/jupiter/Lvl3Quant/output/confluence_meta_v2/results.json')
PRED_DIR = Path('/home/jupiter/Lvl3Quant/output/cnn_mamba_v2_bulk_oot_v2')
MBO_DIR = Path('/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3')
OUT_DIR = Path('/home/jupiter/Lvl3Quant/output/daily_pnl_analysis')
OUT_DIR.mkdir(parents=True, exist_ok=True)

COMMISSION = 0.376
SHORT_PERCENTILE = 3

# === LOAD META DATA ===
print("Loading meta predictions...")
meta = np.load(META_FILE, allow_pickle=True)
meta_preds = meta['predictions']
meta_actuals = meta['actuals']

with open(RESULTS_FILE) as f:
    results = json.load(f)

folds = results['per_fold']

# Build date-to-index mapping
date_map = []
offset = 0
for fold in folds:
    n = fold['n_test']
    date_map.append({'date': fold['date'], 'start': offset, 'end': offset + n, 'n': n})
    offset += n
assert offset == len(meta_preds)

# === SINGLE PASS: Load each day, get labels_1s, apply meta filter ===
print(f"Processing {len(date_map)} OOT dates (single pass)...")

daily_data = []  # list of dicts with per-trade P&L arrays

for dm in date_map:
    date_str = dm['date']
    start, end, n_total = dm['start'], dm['end'], dm['n']
    day_meta = meta_preds[start:end]

    print(f"  {date_str}: {n_total} top-3% trades...", end=" ")

    # Load raw data
    pred_file = PRED_DIR / f'{date_str}_predictions.npz'
    mbo_file = MBO_DIR / f'{date_str}_mbo_events.npz'

    if not pred_file.exists() or not mbo_file.exists():
        print("SKIP (missing files)")
        continue

    pred_data = np.load(pred_file, allow_pickle=True)
    preds = pred_data['predictions']
    n_windows = int(pred_data['n_windows'])
    window_size = int(pred_data['window_size'])
    stride = int(pred_data['stride'])

    mbo = np.load(mbo_file, allow_pickle=True)
    events = mbo['events']
    l1s = mbo['labels_1s']
    l5s = mbo['labels_5s']

    indices = np.arange(n_windows) * stride + (window_size - 1)
    max_idx = min(len(events), len(l1s), len(l5s)) - 1
    valid = indices <= max_idx
    indices = indices[valid]
    preds_aligned = preds[:len(indices)]

    label_1s = l1s[indices]
    label_5s = l5s[indices]
    feat = events[indices]

    valid_mask = ~(np.isnan(label_1s) | np.isnan(label_5s) | np.any(np.isnan(feat), axis=1))
    label_1s = label_1s[valid_mask]
    preds_aligned = preds_aligned[valid_mask]

    pred_5s = preds_aligned[:, 1]
    threshold = np.percentile(pred_5s, SHORT_PERCENTILE)
    short_mask = pred_5s <= threshold

    l1s_short = label_1s[short_mask]

    if len(l1s_short) != n_total:
        print(f"SKIP (count mismatch: {len(l1s_short)} vs {n_total})")
        continue

    # Apply meta filter: score > 0
    meta_mask = day_meta > 0
    n_filtered = int(meta_mask.sum())

    if n_filtered == 0:
        print(f"0 after meta filter")
        daily_data.append({'date': date_str, 'trades': np.array([]), 'n_pre_meta': n_total})
        continue

    # P&L for 1s hold short: -label_1s - commission
    trade_pnl = -l1s_short[meta_mask] - COMMISSION
    print(f"{n_filtered} after meta, mean={trade_pnl.mean():+.3f}")

    daily_data.append({
        'date': date_str,
        'trades': trade_pnl,
        'n_pre_meta': n_total,
    })

    # Free memory
    del mbo, events, l1s, l5s, pred_data, preds

# === COMPUTE DAILY STATS ===
active_days = [d for d in daily_data if len(d['trades']) > 0]

daily_records = []
for d in active_days:
    t = d['trades']
    wins = t[t > 0]
    losses = t[t <= 0]
    gw = float(wins.sum()) if len(wins) > 0 else 0.0
    gl = float(abs(losses.sum())) if len(losses) > 0 else 1e-9

    # Intraday drawdown
    cum = np.cumsum(t)
    peak = np.maximum.accumulate(cum)
    intraday_dd = float((cum - peak).min())

    daily_records.append({
        'date': d['date'],
        'n_trades': len(t),
        'total_pnl': float(t.sum()),
        'mean_pnl': float(t.mean()),
        'wr': float(np.mean(t > 0)),
        'pf': gw / gl if gl > 0 else float('inf'),
        'gross_wins': gw,
        'gross_losses': gl,
        'std_pnl': float(t.std()) if len(t) > 1 else 0.0,
        'max_trade': float(t.max()),
        'min_trade': float(t.min()),
        'median_pnl': float(np.median(t)),
        'intraday_dd': intraday_dd,
    })

# Arrays for aggregate calcs
daily_pnls = np.array([d['total_pnl'] for d in daily_records])
daily_counts = np.array([d['n_trades'] for d in daily_records])
n_active = len(daily_records)

# All trades concatenated
all_trades = np.concatenate([d['trades'] for d in active_days])

# === PRINT RESULTS ===
print(f"\n{'='*90}")
print(f"DAILY P&L ANALYSIS — CNN-Mamba v2, Top 3% Short, Meta>0, Passive, 1s Hold")
print(f"{'='*90}")
print(f"Total OOT dates: {len(daily_data)}")
print(f"Active trading days: {n_active}")
print(f"Zero-trade days: {len(daily_data) - n_active}")

# Per-day table
print(f"\n{'Date':>10s} {'Trades':>7s} {'Total PnL':>10s} {'Mean PnL':>9s} {'WR':>6s} {'PF':>7s} {'Best':>7s} {'Worst':>7s} {'IntraDD':>9s}")
print("-" * 85)
for d in daily_records:
    pf_str = f"{d['pf']:.2f}" if d['pf'] < 999 else "inf"
    print(f"{d['date']:>10s} {d['n_trades']:>7d} {d['total_pnl']:>+10.2f} {d['mean_pnl']:>+9.4f} {d['wr']:>5.1%} {pf_str:>7s} {d['max_trade']:>+7.2f} {d['min_trade']:>+7.2f} {d['intraday_dd']:>+9.2f}")

# === AGGREGATE ===
total_trades = len(all_trades)
total_pnl = float(all_trades.sum())
per_trade_mean = float(all_trades.mean())
per_trade_wr = float(np.mean(all_trades > 0))

# Aggregate PF
total_wins = sum(d['gross_wins'] for d in daily_records)
total_losses = sum(d['gross_losses'] for d in daily_records)
agg_pf = total_wins / total_losses if total_losses > 0 else float('inf')

# Daily Sharpe
daily_mean = daily_pnls.mean()
daily_std = daily_pnls.std(ddof=1)
daily_sharpe = (daily_mean / daily_std) * np.sqrt(252) if daily_std > 0 else 0

# Sortino
downside = daily_pnls[daily_pnls < 0]
downside_dev = np.sqrt(np.mean(downside**2)) if len(downside) > 0 else 1e-9
sortino = (daily_mean / downside_dev) * np.sqrt(252)

# Drawdown
cum_pnl = np.cumsum(daily_pnls)
running_max = np.maximum.accumulate(cum_pnl)
drawdown = cum_pnl - running_max
max_dd = float(drawdown.min())
max_dd_idx = int(drawdown.argmin())
peak_idx = int(np.argmax(cum_pnl[:max_dd_idx+1])) if max_dd_idx > 0 else 0

# Calmar
ann_return = daily_mean * 252
calmar = abs(ann_return / max_dd) if max_dd != 0 else float('inf')

# Green/Red/Flat
green = int(np.sum(daily_pnls > 0))
red = int(np.sum(daily_pnls < 0))
flat = int(np.sum(daily_pnls == 0))

# Best/Worst
best_idx = int(np.argmax(daily_pnls))
worst_idx = int(np.argmin(daily_pnls))

# Max intraday DD across all days
all_intraday_dds = [(d['date'], d['intraday_dd']) for d in daily_records]
worst_intraday = min(all_intraday_dds, key=lambda x: x[1])

# Streaks
def compute_streaks(pnls):
    win_s, lose_s = [], []
    cur, is_w = 0, None
    for p in pnls:
        if p > 0:
            if is_w:
                cur += 1
            else:
                if is_w is not None and cur > 0:
                    lose_s.append(cur)
                cur, is_w = 1, True
        elif p < 0:
            if is_w == False:
                cur += 1
            else:
                if is_w is not None and cur > 0:
                    win_s.append(cur)
                cur, is_w = 1, False
        else:
            if is_w is not None and cur > 0:
                (win_s if is_w else lose_s).append(cur)
            cur, is_w = 0, None
    if is_w is not None and cur > 0:
        (win_s if is_w else lose_s).append(cur)
    return win_s, lose_s

win_streaks, lose_streaks = compute_streaks(daily_pnls)
max_win = max(win_streaks) if win_streaks else 0
max_lose = max(lose_streaks) if lose_streaks else 0
avg_win = np.mean(win_streaks) if win_streaks else 0
avg_lose = np.mean(lose_streaks) if lose_streaks else 0

print(f"\n{'='*90}")
print(f"AGGREGATE STATISTICS")
print(f"{'='*90}")
print(f"Total trades:              {total_trades:,}")
print(f"Total P&L (ticks):         {total_pnl:+.2f}")
print(f"Total P&L ($, 1-lot ES):   ${total_pnl * 12.50:+,.2f}")
print(f"Mean P&L/trade (ticks):    {per_trade_mean:+.4f}")
print(f"Median P&L/trade (ticks):  {float(np.median(all_trades)):+.4f}")
print(f"Per-trade WR:              {per_trade_wr:.1%}")
print(f"Aggregate PF:              {agg_pf:.2f}")
print(f"")
print(f"--- Daily Metrics ---")
print(f"Mean daily P&L (ticks):    {daily_mean:+.2f}")
print(f"Std daily P&L (ticks):     {daily_std:.2f}")
print(f"Mean daily trades:         {daily_counts.mean():.1f}")
print(f"")
print(f"--- Risk-Adjusted Returns ---")
print(f"Daily Sharpe (ann.):       {daily_sharpe:.2f}")
print(f"Sortino (ann.):            {sortino:.2f}")
print(f"Calmar:                    {calmar:.2f}")
print(f"")
print(f"--- Day Classification ---")
print(f"Green days (P&L > 0):      {green} ({green/n_active:.1%})")
print(f"Red days (P&L < 0):        {red} ({red/n_active:.1%})")
print(f"Flat days (P&L = 0):       {flat} ({flat/n_active:.1%})")
print(f"")
print(f"--- Drawdown ---")
print(f"Max drawdown (daily cum):  {max_dd:+.2f} ticks (${max_dd * 12.50:+,.2f})")
print(f"DD peak date:              {daily_records[peak_idx]['date']}")
print(f"DD trough date:            {daily_records[max_dd_idx]['date']}")
print(f"Max intraday DD:           {worst_intraday[1]:+.2f} ticks (${worst_intraday[1] * 12.50:+,.2f}) on {worst_intraday[0]}")
print(f"")
print(f"--- Best / Worst ---")
print(f"Best day:                  {daily_records[best_idx]['date']} — {daily_pnls[best_idx]:+.2f} ticks ({daily_records[best_idx]['n_trades']} trades, WR {daily_records[best_idx]['wr']:.1%})")
print(f"Worst day:                 {daily_records[worst_idx]['date']} — {daily_pnls[worst_idx]:+.2f} ticks ({daily_records[worst_idx]['n_trades']} trades, WR {daily_records[worst_idx]['wr']:.1%})")
print(f"")
print(f"--- Streak Analysis ---")
print(f"Longest winning streak:    {max_win} days")
print(f"Longest losing streak:     {max_lose} days")
print(f"Avg winning streak:        {avg_win:.1f} days")
print(f"Avg losing streak:         {avg_lose:.1f} days")

# === CUMULATIVE EQUITY CURVE ===
print(f"\n{'='*90}")
print(f"CUMULATIVE EQUITY CURVE")
print(f"{'='*90}")
print(f"{'Date':>10s} {'Day PnL':>9s} {'Cum PnL':>10s} {'Trades':>7s} {'DD':>8s}")
print("-" * 50)
for i, d in enumerate(daily_records):
    print(f"{d['date']:>10s} {d['total_pnl']:>+9.2f} {cum_pnl[i]:>+10.2f} {d['n_trades']:>7d} {drawdown[i]:>+8.2f}")

# === SAVE ===
output = {
    'config': {
        'model': 'CNN-Mamba v2',
        'signal_filter': 'top 3% short (pred_5s)',
        'meta_filter': 'score > 0 (top ~50%)',
        'hold': '1s (labels_1s)',
        'cost': '0.376 ticks (passive RT commission)',
        'n_oot_dates': len(daily_data),
        'n_active_dates': n_active,
    },
    'aggregate': {
        'total_trades': total_trades,
        'total_pnl_ticks': round(total_pnl, 4),
        'total_pnl_usd': round(total_pnl * 12.50, 2),
        'mean_pnl_per_trade': round(per_trade_mean, 4),
        'median_pnl_per_trade': round(float(np.median(all_trades)), 4),
        'per_trade_wr': round(per_trade_wr, 4),
        'aggregate_pf': round(agg_pf, 4),
        'daily_sharpe_ann': round(daily_sharpe, 4),
        'sortino_ann': round(sortino, 4),
        'calmar': round(calmar, 4),
        'green_days': green,
        'red_days': red,
        'flat_days': flat,
        'green_pct': round(green / n_active, 4),
        'max_drawdown_ticks': round(max_dd, 4),
        'max_drawdown_usd': round(max_dd * 12.50, 2),
        'max_intraday_dd_ticks': round(worst_intraday[1], 4),
        'max_intraday_dd_date': worst_intraday[0],
        'best_day': daily_records[best_idx]['date'],
        'best_day_pnl': round(daily_pnls[best_idx], 4),
        'worst_day': daily_records[worst_idx]['date'],
        'worst_day_pnl': round(daily_pnls[worst_idx], 4),
        'max_win_streak': max_win,
        'max_lose_streak': max_lose,
        'avg_win_streak': round(avg_win, 1),
        'avg_lose_streak': round(avg_lose, 1),
    },
    'per_day': [
        {
            'date': d['date'],
            'n_trades': d['n_trades'],
            'total_pnl': round(d['total_pnl'], 4),
            'mean_pnl': round(d['mean_pnl'], 4),
            'wr': round(d['wr'], 4),
            'pf': round(d['pf'], 4) if d['pf'] < 999 else None,
            'cum_pnl': round(float(cum_pnl[i]), 4),
            'drawdown': round(float(drawdown[i]), 4),
            'intraday_dd': round(d['intraday_dd'], 4),
        }
        for i, d in enumerate(daily_records)
    ],
}

with open(OUT_DIR / 'daily_pnl_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to {OUT_DIR / 'daily_pnl_results.json'}")
print("DONE.")
