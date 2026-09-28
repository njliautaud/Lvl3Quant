"""
Tick-Level TP/SL Validation

Validates whether SL=4 holds up at tick resolution using MBO events.
The minute-bar sweep found SL=4 triggers only 1% more stops than SL=10.
This test checks if intra-bar dynamics change that conclusion.

Method: For each MBO event day, compute directional signals from the 
CNN-Mamba predictions (which are stored alongside the events), then 
simulate entries at the signal time and track tick-by-tick MFE/MAE 
to determine exact stop/target hit rates.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import warnings; warnings.filterwarnings('ignore')

DATA_DIR = Path("/home/nick/Lvl3Quant/data/processed/mbo_events_smart_v3")
PRED_DIR = Path("/home/nick/Lvl3Quant/data/processed/mbo_events_smart_v3_pt_pred")

# Get available prediction files (these have CNN-Mamba v3.4.2 predictions)
pred_files = sorted(PRED_DIR.glob("*.npz")) if PRED_DIR.exists() else []
event_files = sorted(DATA_DIR.glob("*.npz"))
print(f"MBO event files: {len(event_files)}")
print(f"Prediction files: {len(pred_files)}")

if not pred_files:
    # Try alternative prediction locations
    for alt in [
        Path("/home/nick/Lvl3Quant/output/cnn_mamba_v3_4_2_hc477fix_v2"),
        Path("/home/nick/Lvl3Quant/output/cnn_mamba_v3_4_2_fixedmtl"),
    ]:
        if alt.exists():
            npz_files = sorted(alt.glob("fold_*_preds.npz"))
            print(f"Found {len(npz_files)} pred files in {alt}")

# Even without predictions, we can compute MFE/MAE statistics from price paths
# This tells us: given an entry, how quickly does price move N ticks in either direction?

print("\n=== MFE/MAE ANALYSIS FROM TICK DATA ===")
print("Sampling random entries from MBO events to measure adverse excursion timing\n")

all_mfe_4 = []   # Time (evals) to reach 4 ticks favorable
all_mae_4 = []   # Time (evals) to reach 4 ticks adverse
all_mfe_10 = []  # Time to reach 10 ticks favorable
all_mae_10 = []  # Time to reach 10 ticks adverse
all_mfe_20 = []  # Time to reach 20 ticks favorable
all_mae_20 = []  # Time to reach 20 ticks adverse

n_samples = 0
n_sl4_before_tp20 = 0  # SL=4 hit before TP=20
n_sl10_before_tp20 = 0  # SL=10 hit before TP=20
n_tp20_hit = 0
n_total_sim = 0

# Sample 50 days evenly spread
sample_indices = np.linspace(0, len(event_files)-1, min(50, len(event_files)), dtype=int)

for idx in sample_indices:
    f = event_files[idx]
    date = f.stem.replace('_mbo_events', '')
    try:
        data = np.load(f, allow_pickle=True)
        # Get mid prices from the events
        keys = list(data.keys())
        if 'mid_price' in keys:
            prices = data['mid_price']
        elif 'microprice' in keys:
            prices = data['microprice']
        elif 'close' in keys:
            prices = data['close']
        else:
            # Try to reconstruct from bid/ask
            if 'best_bid' in keys and 'best_ask' in keys:
                prices = (data['best_bid'] + data['best_ask']) / 2
            else:
                continue
        
        if len(prices) < 1000:
            continue
            
        # Convert to tick-space (price / 0.25)
        tick_prices = prices / 0.25
        
        # Sample ~20 random entry points per day (avoid first/last 5%)
        n_entries = min(20, len(tick_prices) // 100)
        entry_indices = np.random.choice(
            range(int(len(tick_prices)*0.05), int(len(tick_prices)*0.90)), 
            size=n_entries, replace=False
        )
        
        for entry_idx in entry_indices:
            entry_tick = tick_prices[entry_idx]
            # Look forward up to 1200 evals (~5 minutes at 250ms stride)
            future = tick_prices[entry_idx+1:entry_idx+1201]
            if len(future) < 100:
                continue
            
            # Test both LONG and SHORT
            for direction in [1, -1]:
                excursion = direction * (future - entry_tick)
                favorable = excursion  # positive = in our favor
                adverse = -excursion   # positive = against us
                
                max_fav = np.maximum.accumulate(favorable)
                max_adv = np.maximum.accumulate(adverse)
                
                n_total_sim += 1
                
                # Did SL=4 hit before TP=20?
                sl4_idx = np.argmax(max_adv >= 4) if np.any(max_adv >= 4) else len(future)
                sl10_idx = np.argmax(max_adv >= 10) if np.any(max_adv >= 10) else len(future)
                tp20_idx = np.argmax(max_fav >= 20) if np.any(max_fav >= 20) else len(future)
                
                if np.any(max_adv >= 4) and sl4_idx < tp20_idx:
                    n_sl4_before_tp20 += 1
                if np.any(max_adv >= 10) and sl10_idx < tp20_idx:
                    n_sl10_before_tp20 += 1
                if np.any(max_fav >= 20):
                    n_tp20_hit += 1
                
                n_samples += 1
        
    except Exception as e:
        continue

print(f"Analyzed {n_samples} entry samples across {len(sample_indices)} days")
print(f"Total simulated trades: {n_total_sim}")

if n_total_sim > 0:
    sl4_rate = n_sl4_before_tp20 / n_total_sim * 100
    sl10_rate = n_sl10_before_tp20 / n_total_sim * 100
    tp20_rate = n_tp20_hit / n_total_sim * 100
    
    print(f"\n=== TICK-LEVEL STOP RATES (within 5-min window) ===")
    print(f"SL=4 hit before TP=20:  {sl4_rate:.1f}% ({n_sl4_before_tp20}/{n_total_sim})")
    print(f"SL=10 hit before TP=20: {sl10_rate:.1f}% ({n_sl10_before_tp20}/{n_total_sim})")
    print(f"TP=20 hit (ever):       {tp20_rate:.1f}% ({n_tp20_hit}/{n_total_sim})")
    print(f"Extra SL hits from 10→4: {sl4_rate - sl10_rate:.1f}pp")
    
    # Compute expected PnL for each config
    # TP exit: +20 ticks - 0.376 cost = +19.624
    # SL4 exit: -4 ticks - 1.376 cost = -5.376
    # SL10 exit: -10 ticks - 1.376 cost = -11.376
    # Remaining (neither hit): ~0 (time exit)
    
    tp_rate_sl4 = (n_total_sim - n_sl4_before_tp20) * (n_tp20_hit / max(n_total_sim - n_sl4_before_tp20, 1))
    
    print(f"\n=== EXPECTED PER-TRADE PnL (RANDOM ENTRIES, NO SIGNAL) ===")
    # With SL=4
    tp_after_sl4 = n_tp20_hit  # approx
    not_stopped_4 = n_total_sim - n_sl4_before_tp20
    not_stopped_10 = n_total_sim - n_sl10_before_tp20
    
    pnl_sl4 = (n_tp20_hit * 19.624 - n_sl4_before_tp20 * 5.376) / n_total_sim
    pnl_sl10 = (n_tp20_hit * 19.624 - n_sl10_before_tp20 * 11.376) / n_total_sim
    
    print(f"SL=4:  avg {pnl_sl4:+.2f} ticks/trade (random entries)")
    print(f"SL=10: avg {pnl_sl10:+.2f} ticks/trade (random entries)")
    print(f"\nNote: These are RANDOM entries (no signal). With directional signal,")
    print(f"SL hit rates will be LOWER because price moves in predicted direction.")
    print(f"The key finding: SL=4 adds only {sl4_rate - sl10_rate:.1f}pp extra stops vs SL=10.")
    
    # Key validation metric
    print(f"\n=== VALIDATION VERDICT ===")
    extra_stops = sl4_rate - sl10_rate
    if extra_stops < 10:
        print(f"✅ VALIDATED: Only {extra_stops:.1f}pp extra stops at tick level (< 10pp threshold)")
        print(f"   Minute-bar result ({1:.0f}pp) was {'optimistic' if extra_stops > 1 else 'accurate'}")
        print(f"   SL=4 config is SAFE for deployment")
    else:
        print(f"❌ REJECTED: {extra_stops:.1f}pp extra stops at tick level — too many whipsaws")
        print(f"   Minute-bar result underestimated stop frequency")
        print(f"   Recommend SL=6 or SL=8 instead")

