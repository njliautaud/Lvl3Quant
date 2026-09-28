import json, glob, sys
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta

FILLSIM_DIR = Path('/home/nick/Lvl3Quant/output/extended_oot_validation/fillsim_results')
PRED_DIR = Path('/home/nick/Lvl3Quant/output/extended_oot_validation/pred_npzs')
MID_DIR = Path('/home/nick/Lvl3Quant/data/derived/mid_price_bars')
N_BARS = 234_000
BAR_NS = 100_000_000

def compute_rth_open_ns(date_str):
    d = datetime.strptime(date_str, '%Y%m%d')
    midnight_utc = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
    if 3 <= d.month <= 10:
        rth_open_utc = midnight_utc + timedelta(hours=13, minutes=30)
    else:
        rth_open_utc = midnight_utc + timedelta(hours=14, minutes=30)
    return int(rth_open_utc.timestamp() * 1_000_000_000)

def ns_to_bar(target_ns, rth_open_ns):
    offset_ns = target_ns - rth_open_ns
    bar = int(offset_ns / BAR_NS)
    return max(0, min(bar, N_BARS - 1))

configs = [
    {'fade_thresh': -0.3, 'fade_n': 20, 'rev_thresh': -0.5, 'rev_n': 5, 'label': 'best_buy_cfg'},
    {'fade_thresh': -0.1, 'fade_n': 40, 'rev_thresh': -0.5, 'rev_n': 8, 'label': 'v2'},
    {'fade_thresh': 999, 'fade_n': 999999, 'rev_thresh': 999, 'rev_n': 999999, 'label': 'BASELINE'},
]

for side_filter in ['BUY', 'SELL']:
    print(f'\n{"="*80}')
    print(f'{side_filter} SIDE — AFTERNOON (both_afternoon files)')
    print(f'{"="*80}')
    
    for cfg in configs:
        daily_pnl = {}
        total_new = 0
        total_orig = 0
        n_trades = 0
        n_pressure = 0
        
        for f in sorted(glob.glob(str(FILLSIM_DIR / 'both_afternoon_*.json'))):
            date_str = Path(f).stem.replace('both_afternoon_', '')
            pred_file = PRED_DIR / f'{date_str}_unfiltered.npz'
            mid_file = MID_DIR / f'{date_str}.npz'
            if not pred_file.exists() or not mid_file.exists():
                continue
            
            with open(f) as fh:
                d = json.load(fh)
            trades = [t for t in d.get('trades', []) if t.get('side') == side_filter]
            if not trades:
                continue
            
            preds = np.load(pred_file)['predictions']
            mid_prices = np.load(mid_file)['mid_prices']
            if np.count_nonzero(mid_prices) < N_BARS * 0.5:
                continue
            
            rth_open_ns = compute_rth_open_ns(date_str)
            day_orig = 0
            day_new = 0
            
            for t in trades:
                entry_ns = t['fill_time_ns']
                entry_px = t['entry_price']
                orig_pnl = t['pnl_ticks']
                exit_ns = t.get('exit_time_ns', entry_ns + 1800_000_000_000)
                
                entry_bar = ns_to_bar(entry_ns, rth_open_ns)
                exit_bar = ns_to_bar(exit_ns, rth_open_ns)
                
                if side_filter == 'BUY':
                    tp_px = entry_px + 8 * 0.25
                    sl_px = entry_px - 16 * 0.25
                else:
                    tp_px = entry_px - 8 * 0.25
                    sl_px = entry_px + 16 * 0.25
                
                fade_count = 0
                rev_count = 0
                pressure_exit = False
                pressure_bar = None
                
                for bar in range(entry_bar + 1, min(exit_bar + 1, N_BARS)):
                    mid = mid_prices[bar]
                    if mid <= 0:
                        continue
                    
                    if side_filter == 'BUY':
                        if mid >= tp_px or mid <= sl_px:
                            break
                    else:
                        if mid <= tp_px or mid >= sl_px:
                            break
                    
                    pred = preds[bar]
                    
                    # For BUY: negative pred = fading buying pressure
                    # For SELL: positive pred = fading selling pressure (buying pressure = adverse)
                    if side_filter == 'BUY':
                        fade_trigger = pred < cfg['fade_thresh']
                        rev_trigger = pred < cfg['rev_thresh']
                    else:
                        # Mirror: for sells, use abs thresholds but check positive direction
                        fade_trigger = pred > abs(cfg['fade_thresh'])
                        rev_trigger = pred > abs(cfg['rev_thresh'])
                    
                    if fade_trigger:
                        fade_count += 1
                    else:
                        fade_count = 0
                    if rev_trigger:
                        rev_count += 1
                    else:
                        rev_count = 0
                    
                    if (cfg['fade_n'] > 0 and fade_count >= cfg['fade_n']) or \
                       (cfg['rev_n'] > 0 and rev_count >= cfg['rev_n']):
                        pressure_exit = True
                        pressure_bar = bar
                        break
                
                if pressure_exit and pressure_bar is not None:
                    exit_mid = mid_prices[pressure_bar]
                    if exit_mid > 0:
                        if side_filter == 'BUY':
                            new_pnl = (exit_mid - entry_px) / 0.25
                        else:
                            new_pnl = (entry_px - exit_mid) / 0.25
                        day_new += new_pnl
                        n_pressure += 1
                    else:
                        day_new += orig_pnl
                else:
                    day_new += orig_pnl
                
                day_orig += orig_pnl
                n_trades += 1
            
            daily_pnl[date_str] = day_new
            total_new += day_new
            total_orig += day_orig
        
        daily_vals = list(daily_pnl.values())
        sharpe = np.mean(daily_vals) / np.std(daily_vals) * np.sqrt(252) if len(daily_vals) > 1 and np.std(daily_vals) > 0 else 0
        green = sum(1 for v in daily_vals if v > 0)
        red = sum(1 for v in daily_vals if v <= 0)
        
        march_t = sum(v for k,v in daily_pnl.items() if k.startswith('202603'))
        april_t = sum(v for k,v in daily_pnl.items() if k.startswith('202604'))
        
        print(f'  {cfg["label"]}: Net={total_new:.0f}t (orig={total_orig:.0f}t D={total_new-total_orig:+.0f}t) '
              f'Sharpe={sharpe:.1f} G/R={green}/{red} Trades={n_trades} Pressure={n_pressure} '
              f'Mar={march_t:.0f}/Apr={april_t:.0f}')
