"""
IMB Z-Score Leakage Audit + Fill Sim
Signal: L1 bid-ask imbalance z-score (iz >= 2.0, very-low-vol regime)
NOTE: Book tensor layout = [bid_price_offset, bid_size, ask_size, ask_price_offset]
      bid_sz = col1, ask_sz = col2 (FIXED from original which used col3)
"""
import numpy as np
import glob, json, time, os

t0 = time.time()
print('[START] iz2_leakage_audit.py', flush=True)

TICK = 0.25
DATA_DIR = "/home/jupiter/Lvl3Quant/data/processed/dl_book_cache_oot"
OUTPUT = "/home/jupiter/iz2_leakage_audit_results.json"

EXEC_CONFIGS = [
    {'name': 'market_order', 'slippage': 2.0, 'fill_rate': 1.0},
    {'name': 'limit_chase',  'slippage': 0.5, 'fill_rate': 0.15},
    {'name': 'passive_limit','slippage': 0.0, 'fill_rate': 0.07},
]
COMMISSION_RT = 0.376
HOLD_BARS = 120
IZ_THRESH = 2.0
VOL_WINDOW = 300
VOL_LOW_PCT = 33

files = sorted(glob.glob(DATA_DIR + "/*_book_tensors.npz"))
print(f"Found {len(files)} days", flush=True)

if not files:
    print("ERROR: No files found in " + DATA_DIR, flush=True)
    exit(1)

n_is = int(len(files) * 0.6)
is_files = files[:n_is]
oos_files = files[n_is:]
print(f"IS: {len(is_files)} days | OOS: {len(oos_files)} days", flush=True)

def load_day(f):
    d = np.load(f, allow_pickle=True)
    bt = d['book_tensors'].astype(np.float32)
    mid = d['mid_prices'].astype(np.float64)
    # Layout: [bid_price_offset, bid_size, ask_size, ask_price_offset]
    bid_sz = bt[:, 0, 1].astype(np.float64)
    ask_sz = bt[:, 0, 2].astype(np.float64)  # FIXED: was col3, must be col2
    total = bid_sz + ask_sz
    imb = np.where(total > 0, (bid_sz - ask_sz) / total, 0.0)
    return imb, mid

def compute_rolling_vol(mid, window=VOL_WINDOW):
    returns = np.diff(mid, prepend=mid[0])
    vol = np.full(len(mid), np.nan)
    for i in range(window, len(mid)):
        vol[i] = np.std(returns[i-window:i])
    return vol

def compute_iz_fast(imb, window=300):
    iz = np.zeros(len(imb))
    arr = np.array(imb, dtype=np.float64)
    n = len(arr)
    if n < 60:
        return iz
    cumsum = np.cumsum(arr)
    cumsum2 = np.cumsum(arr**2)
    for i in range(60, n):
        w_start = max(0, i - window)
        w_len = i - w_start
        s = cumsum[i-1] - (cumsum[w_start-1] if w_start > 0 else 0)
        s2 = cumsum2[i-1] - (cumsum2[w_start-1] if w_start > 0 else 0)
        mu = s / w_len
        var = s2/w_len - mu**2
        sd = np.sqrt(max(var, 1e-16))
        iz[i] = (arr[i] - mu) / sd
    return iz

def compute_ic(iz_masked, mid, iz_thresh, horizon=HOLD_BARS):
    if len(iz_masked) < horizon + 100:
        return np.nan, np.array([])
    sig_idx = np.where(np.abs(iz_masked) >= iz_thresh)[0]
    sig_idx = sig_idx[sig_idx < len(mid) - horizon]
    if len(sig_idx) < 10:
        return np.nan, np.array([])
    fwd = (mid[sig_idx + horizon] - mid[sig_idx]) / TICK
    dirs = np.sign(iz_masked[sig_idx])
    dir_fwd = fwd * dirs
    sig_mag = np.abs(iz_masked[sig_idx])
    if len(sig_mag) < 2 or np.std(sig_mag) < 1e-10:
        return np.nan, dir_fwd
    ic = np.corrcoef(sig_mag, dir_fwd)[0, 1]
    return float(ic), dir_fwd

print("\nPhase 1: IS vol threshold...", flush=True)
is_vol_samples = []
for f in is_files[:30]:
    try:
        imb, mid = load_day(f)
        vol = compute_rolling_vol(mid)
        is_vol_samples.extend(vol[~np.isnan(vol)].tolist())
    except Exception as e:
        print(f"  ERR {f}: {e}", flush=True)

if not is_vol_samples:
    print("ERROR: no vol samples", flush=True)
    exit(1)

vol_low_thresh = np.percentile(is_vol_samples, VOL_LOW_PCT)
print(f"  Vol p33: {vol_low_thresh:.6f}", flush=True)

print("\nPhase 2: IS IC...", flush=True)
is_ics = []
for f in is_files:
    try:
        imb, mid = load_day(f)
        vol = compute_rolling_vol(mid)
        iz = compute_iz_fast(imb)
        low_vol = vol < vol_low_thresh
        iz_masked = np.where(low_vol, iz, 0.0)
        ic, _ = compute_ic(iz_masked, mid, IZ_THRESH)
        if not np.isnan(ic):
            is_ics.append(ic)
    except Exception as e:
        pass

is_ic_mean = float(np.nanmean(is_ics)) if is_ics else float('nan')
is_ic_pos = float(np.mean([x > 0 for x in is_ics])) if is_ics else float('nan')
print(f"  IS: mean={is_ic_mean:.4f}, pos%={is_ic_pos:.1%}, n={len(is_ics)}", flush=True)

print("\nPhase 3: OOS IC...", flush=True)
oos_ics = []
oos_sig_count = 0
for f in oos_files:
    try:
        imb, mid = load_day(f)
        vol = compute_rolling_vol(mid)
        iz = compute_iz_fast(imb)
        low_vol = vol < vol_low_thresh
        iz_masked = np.where(low_vol, iz, 0.0)
        ic, _ = compute_ic(iz_masked, mid, IZ_THRESH)
        if not np.isnan(ic):
            oos_ics.append(ic)
        oos_sig_count += int(np.sum(np.abs(iz_masked) >= IZ_THRESH))
    except Exception as e:
        pass

oos_ic_mean = float(np.nanmean(oos_ics)) if oos_ics else float('nan')
oos_ic_pos = float(np.mean([x > 0 for x in oos_ics])) if oos_ics else float('nan')
print(f"  OOS: mean={oos_ic_mean:.4f}, pos%={oos_ic_pos:.1%}, n={len(oos_ics)}", flush=True)
print(f"  OOS signals: {oos_sig_count} total, {oos_sig_count/max(1,len(oos_files)):.1f}/day", flush=True)

print("\nPhase 4: Fill sim...", flush=True)
exec_results = {}

for cfg in EXEC_CONFIGS:
    name = cfg['name']
    fill_rate = cfg['fill_rate']
    slip = cfg['slippage']
    pnls = []
    wins = 0
    rng2 = np.random.default_rng(42)
    for f in oos_files:
        try:
            imb, mid = load_day(f)
            vol = compute_rolling_vol(mid)
            iz = compute_iz_fast(imb)
            low_vol = vol < vol_low_thresh
            iz_masked = np.where(low_vol, iz, 0.0)
            n = len(mid)
            i = VOL_WINDOW + 10
            while i < n - HOLD_BARS - 5:
                if abs(iz_masked[i]) < IZ_THRESH:
                    i += 1
                    continue
                if rng2.random() > fill_rate:
                    i += 1
                    continue
                direction = 1 if iz_masked[i] > 0 else -1
                entry_px = mid[i] + direction * slip * TICK
                exit_px = mid[i + HOLD_BARS]
                raw = direction * (exit_px - entry_px) / TICK
                comm = COMMISSION_RT / (TICK * 50)
                net = raw - comm - slip
                pnls.append(net)
                if net > 0:
                    wins += 1
                i += HOLD_BARS + 1
        except Exception as e:
            pass
    if len(pnls) >= 5:
        arr = np.array(pnls)
        mean_p = float(np.mean(arr))
        wr = wins / len(pnls)
        down = arr[arr < 0]
        sortino = (mean_p / np.std(down)) * np.sqrt(252 * len(pnls) / max(1, len(oos_files))) if len(down) > 2 else float('nan')
        n_day = len(pnls) / max(1, len(oos_files))
        exec_results[name] = {
            'n': len(pnls), 'n_per_day': round(n_day, 1),
            'fill_rate': fill_rate, 'slippage_ticks': slip,
            'mean_pnl_ticks': round(mean_p, 4),
            'win_rate': round(wr, 4),
            'sortino': round(sortino, 4) if not np.isnan(sortino) else None,
        }
        print(f"  {name}: n={len(pnls)}, WR={wr:.1%}, mean={mean_p:.4f}tk, Sortino={sortino:.3f}", flush=True)
    else:
        exec_results[name] = {'n': len(pnls), 'error': 'too_few'}
        print(f"  {name}: too few ({len(pnls)})", flush=True)

leakage_ok = abs(oos_ic_mean - is_ic_mean) < 0.05 if not (np.isnan(is_ic_mean) or np.isnan(oos_ic_mean)) else False
results = {
    'signal': 'imbalance_z_score_iz2_very_low_vol',
    'iz_thresh': IZ_THRESH,
    'vol_low_pct': VOL_LOW_PCT,
    'hold_bars': HOLD_BARS,
    'is_days': len(is_files), 'oos_days': len(oos_files),
    'is_ic_mean': round(is_ic_mean, 4) if not np.isnan(is_ic_mean) else None,
    'is_ic_pct_pos': round(is_ic_pos, 4) if not np.isnan(is_ic_pos) else None,
    'oos_ic_mean': round(oos_ic_mean, 4) if not np.isnan(oos_ic_mean) else None,
    'oos_ic_pct_pos': round(oos_ic_pos, 4) if not np.isnan(oos_ic_pos) else None,
    'oos_signals_per_day': round(oos_sig_count / max(1, len(oos_files)), 1),
    'leakage_verdict': 'CLEAN' if leakage_ok else 'SUSPECT',
    'exec_results': exec_results,
    'elapsed_sec': round(time.time() - t0, 1),
}

print("\n=== FINAL RESULTS ===", flush=True)
print(f"IS IC: {is_ic_mean:.4f} ({is_ic_pos:.0%} pos)", flush=True)
print(f"OOS IC: {oos_ic_mean:.4f} ({oos_ic_pos:.0%} pos)", flush=True)
print(f"Leakage verdict: {results['leakage_verdict']}", flush=True)

with open(OUTPUT, 'w') as fp:
    json.dump(results, fp, indent=2)
print(f"[DONE] {time.time()-t0:.1f}s => {OUTPUT}", flush=True)
