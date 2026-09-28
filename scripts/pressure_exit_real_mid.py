#!/usr/bin/env python3
"""
Pressure Exit v2 validation with REAL mid prices (v2-extracted, correct instrument).
Tests early exit when prediction signal fades, using actual tick-by-tick mid prices
instead of linear interpolation.
"""
import json, glob, sys
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta

FILLSIM_DIR = Path("/home/nick/Lvl3Quant/output/extended_oot_validation/fillsim_results")
PRED_DIR = Path("/home/nick/Lvl3Quant/output/extended_oot_validation/pred_npzs")
MID_DIR = Path("/home/nick/Lvl3Quant/data/derived/mid_price_bars")

N_BARS = 234_000
BAR_NS = 100_000_000  # 100ms

def compute_rth_open_ns(date_str):
    d = datetime.strptime(date_str, "%Y%m%d")
    midnight_utc = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
    month = d.month
    if 3 <= month <= 10:
        rth_open_utc = midnight_utc + timedelta(hours=13, minutes=30)
    else:
        rth_open_utc = midnight_utc + timedelta(hours=14, minutes=30)
    return int(rth_open_utc.timestamp() * 1_000_000_000)

def ns_to_bar(target_ns, rth_open_ns):
    offset_ns = target_ns - rth_open_ns
    bar = int(offset_ns / BAR_NS)
    return max(0, min(bar, N_BARS - 1))

def load_trades(date_str):
    f = FILLSIM_DIR / f"buy_afternoon_{date_str}.json"
    if not f.exists():
        return []
    with open(f) as fh:
        d = json.load(fh)
    trades = d.get('trades', [])
    return [t for t in trades if t.get('side') == 'BUY']

def run_pressure_exit(date_str, trades, preds, mid_prices, cfg):
    """Apply pressure exit to trades using real mid prices."""
    rth_open_ns = compute_rth_open_ns(date_str)
    fade_thresh = cfg['fade_thresh']
    fade_n = cfg['fade_n']
    rev_thresh = cfg['rev_thresh']
    rev_n = cfg['rev_n']
    
    results = []
    for t in trades:
        entry_ns = t['fill_time_ns']
        entry_px = t['entry_price']
        orig_pnl = t['pnl_ticks']
        orig_exit_reason = t.get('exit_reason', 'Unknown')
        
        entry_bar = ns_to_bar(entry_ns, rth_open_ns)
        
        # Original exit time
        exit_ns = t.get('exit_time_ns', entry_ns + 1800_000_000_000)  # default 30min
        exit_bar = ns_to_bar(exit_ns, rth_open_ns)
        
        # TP/SL in price
        tp_px = entry_px + 8 * 0.25  # 8 ticks = 2.0 pts
        sl_px = entry_px - 16 * 0.25  # 16 ticks = 4.0 pts
        
        # Walk bar by bar from entry
        fade_count = 0
        rev_count = 0
        pressure_exit = False
        pressure_exit_bar = None
        
        for bar in range(entry_bar + 1, min(exit_bar + 1, N_BARS)):
            mid = mid_prices[bar]
            if mid <= 0:
                continue
            
            # Check TP/SL first (these override pressure exit)
            if mid >= tp_px:
                # TP hit before pressure exit
                break
            if mid <= sl_px:
                # SL hit before pressure exit
                break
            
            # Check prediction signal
            pred = preds[bar]
            
            # For BUY trades: signal < fade_thresh = fading
            if pred < fade_thresh:
                fade_count += 1
            else:
                fade_count = 0
            
            # Reversal: strong negative signal
            if pred < rev_thresh:
                rev_count += 1
            else:
                rev_count = 0
            
            # Trigger pressure exit
            if (fade_n > 0 and fade_count >= fade_n) or (rev_n > 0 and rev_count >= rev_n):
                pressure_exit = True
                pressure_exit_bar = bar
                break
        
        if pressure_exit and pressure_exit_bar is not None:
            # Exit at mid price at pressure exit bar
            exit_mid = mid_prices[pressure_exit_bar]
            if exit_mid > 0:
                # P&L in ticks (0.25 pts per tick)
                new_pnl_pts = exit_mid - entry_px
                new_pnl_ticks = new_pnl_pts / 0.25
                # Add market exit cost (crossing spread to exit = 0.5 ticks)
                # But original trade already includes exit cost, so compare apples to apples
                results.append({
                    'orig_pnl': orig_pnl,
                    'new_pnl': new_pnl_ticks,
                    'exit_type': 'pressure',
                    'orig_exit': orig_exit_reason,
                    'bars_held': pressure_exit_bar - entry_bar,
                })
            else:
                results.append({
                    'orig_pnl': orig_pnl,
                    'new_pnl': orig_pnl,
                    'exit_type': 'original',
                    'orig_exit': orig_exit_reason,
                })
        else:
            results.append({
                'orig_pnl': orig_pnl,
                'new_pnl': orig_pnl,
                'exit_type': 'original',
                'orig_exit': orig_exit_reason,
            })
    
    return results

# Configs to test (focused around best from v2 interpolated results)
configs = []
for fade_thresh in [-0.3, -0.1, 0.0, 0.1]:
    for fade_n in [20, 40, 80, 160]:
        for rev_thresh in [-0.5, -0.3]:
            for rev_n in [5, 8, 15]:
                configs.append({
                    'fade_thresh': fade_thresh,
                    'fade_n': fade_n,
                    'rev_thresh': rev_thresh,
                    'rev_n': rev_n,
                })

# Also add baseline (no pressure exit)
configs.append({'fade_thresh': -999, 'fade_n': 999999, 'rev_thresh': -999, 'rev_n': 999999})

print(f"Testing {len(configs)} configs")

# Load all data
all_data = {}
trade_dates = sorted(glob.glob(str(FILLSIM_DIR / "buy_afternoon_*.json")))
for f in trade_dates:
    date_str = Path(f).stem.replace("buy_afternoon_", "")
    
    pred_file = PRED_DIR / f"{date_str}_unfiltered.npz"
    mid_file = MID_DIR / f"{date_str}.npz"
    
    if not pred_file.exists() or not mid_file.exists():
        continue
    
    trades = load_trades(date_str)
    if not trades:
        continue
    
    preds = np.load(pred_file)['predictions']
    mid_prices = np.load(mid_file)['mid_prices']
    
    # Sanity: check mid prices are reasonable
    nonzero = mid_prices[mid_prices > 0]
    if len(nonzero) < N_BARS * 0.5:
        print(f"  SKIP {date_str}: only {len(nonzero)} nonzero mid bars")
        continue
    
    all_data[date_str] = {
        'trades': trades,
        'preds': preds,
        'mid_prices': mid_prices,
    }

print(f"Loaded {len(all_data)} dates with trades + predictions + mid prices")
total_trades = sum(len(d['trades']) for d in all_data.values())
print(f"Total buy trades: {total_trades}")

# Run all configs
best_pf = 0
best_cfg = None
best_results = None

all_config_results = []

for ci, cfg in enumerate(configs):
    all_results = []
    daily_pnl = {}
    
    for date_str, data in sorted(all_data.items()):
        results = run_pressure_exit(date_str, data['trades'], data['preds'], data['mid_prices'], cfg)
        all_results.extend(results)
        day_pnl = sum(r['new_pnl'] for r in results)
        daily_pnl[date_str] = day_pnl
    
    if not all_results:
        continue
    
    total_new = sum(r['new_pnl'] for r in all_results)
    total_orig = sum(r['orig_pnl'] for r in all_results)
    
    gross_win = sum(r['new_pnl'] for r in all_results if r['new_pnl'] > 0)
    gross_loss = sum(abs(r['new_pnl']) for r in all_results if r['new_pnl'] < 0)
    wins = sum(1 for r in all_results if r['new_pnl'] > 0)
    n_pressure = sum(1 for r in all_results if r['exit_type'] == 'pressure')
    
    pf = gross_win / gross_loss if gross_loss > 0 else 999
    wr = wins / len(all_results)
    
    daily_vals = list(daily_pnl.values())
    sharpe = np.mean(daily_vals) / np.std(daily_vals) * np.sqrt(252) if len(daily_vals) > 1 and np.std(daily_vals) > 0 else 0
    neg = [v for v in daily_vals if v < 0]
    sortino = np.mean(daily_vals) / np.std(neg) * np.sqrt(252) if neg and np.std(neg) > 0 else 0
    green = sum(1 for v in daily_vals if v > 0)
    red = sum(1 for v in daily_vals if v <= 0)
    
    improvement = total_new - total_orig
    
    config_result = {
        'cfg': cfg,
        'pf': pf, 'wr': wr, 'sharpe': sharpe, 'sortino': sortino,
        'net_ticks': total_new, 'orig_ticks': total_orig,
        'improvement': improvement,
        'n_pressure': n_pressure, 'n_total': len(all_results),
        'pct_pressure': n_pressure / len(all_results) * 100,
        'green': green, 'red': red,
    }
    all_config_results.append(config_result)
    
    if pf > best_pf:
        best_pf = pf
        best_cfg = cfg
        best_results = config_result

# Sort by PF
all_config_results.sort(key=lambda x: x['pf'], reverse=True)

print("\n" + "=" * 100)
print("TOP 15 CONFIGS BY PROFIT FACTOR (real mid prices)")
print("=" * 100)

for i, r in enumerate(all_config_results[:15]):
    c = r['cfg']
    is_baseline = c['fade_n'] > 100000
    label = "BASELINE" if is_baseline else f"fade={c['fade_thresh']}/n={c['fade_n']} rev={c['rev_thresh']}/n={c['rev_n']}"
    print(f"  {i+1:2d}. PF={r['pf']:.3f} WR={r['wr']:.1%} Sharpe={r['sharpe']:.1f} Sortino={r['sortino']:.1f} "
          f"Net={r['net_ticks']:.0f}t Δ={r['improvement']:+.0f}t "
          f"Pressure={r['pct_pressure']:.0f}% G/R={r['green']}/{r['red']} "
          f"| {label}")

# Find baseline
baseline = [r for r in all_config_results if r['cfg']['fade_n'] > 100000]
if baseline:
    b = baseline[0]
    print(f"\nBASELINE: PF={b['pf']:.3f} WR={b['wr']:.1%} Sharpe={b['sharpe']:.1f} Net={b['net_ticks']:.0f}t G/R={b['green']}/{b['red']}")

# Show improvement distribution
print("\n" + "=" * 100)
print("CONFIGS THAT BEAT BASELINE (by PF)")
print("=" * 100)
if baseline:
    b_pf = baseline[0]['pf']
    beating = [r for r in all_config_results if r['pf'] > b_pf and r['cfg']['fade_n'] < 100000]
    print(f"  {len(beating)} configs beat baseline PF of {b_pf:.3f}")
    for r in beating[:10]:
        c = r['cfg']
        print(f"    PF={r['pf']:.3f} (+{r['pf']-b_pf:.3f}) Sharpe={r['sharpe']:.1f} "
              f"Net={r['net_ticks']:.0f}t Δ={r['improvement']:+.0f}t "
              f"fade={c['fade_thresh']}/n={c['fade_n']} rev={c['rev_thresh']}/n={c['rev_n']}")

# Save full results
outfile = "/home/nick/Lvl3Quant/output/pressure_exit_real_mid_results.json"
save_data = []
for r in all_config_results:
    save_data.append({
        'fade_thresh': r['cfg']['fade_thresh'],
        'fade_n': r['cfg']['fade_n'],
        'rev_thresh': r['cfg']['rev_thresh'],
        'rev_n': r['cfg']['rev_n'],
        'pf': float(round(r['pf'], 4)),
        'wr': float(round(r['wr'], 4)),
        'sharpe': float(round(r['sharpe'], 2)),
        'sortino': float(round(r['sortino'], 2)),
        'net_ticks': float(round(r['net_ticks'], 1)),
        'improvement': float(round(r['improvement'], 1)),
        'pct_pressure': float(round(r['pct_pressure'], 1)),
        'green': r['green'],
        'red': r['red'],
    })
with open(outfile, 'w') as f:
    json.dump(save_data, f, indent=2)
print(f"\nSaved {len(save_data)} results to {outfile}")
