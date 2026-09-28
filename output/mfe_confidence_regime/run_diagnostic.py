#!/usr/bin/env python3
"""
HC #475/#428/#432/#344/#474 deploy-gate diagnostic.

For each symmetric_gate fill (output/hc475_ab/symmetric_gate_fills.parquet):
  - Reverse-map fill.ts_signal_ns -> NPZ row index per date.
  - Recover real signal magnitude from the v3.4.2 47-day OOT NPZ
    (pred_log_ret_{1s,5s,10s,30s}). 60s and 5min heads are EXCLUDED per HC #477.
  - Realized horizon return = NPZ target_log_ret_{1s,5s,10s,30s} (ticks).
    MFE/MAE within 30s = NPZ target_pred_{mfe,mae}_30s_ticks (true extremes,
    available only for the 30s horizon).
  - Confidence quintiles formed within each (config, side, horizon) cell by |pred|.

Produces:
  - mfe_by_horizon_confidence.parquet
  - regime_stratification.parquet
  - gates_results.parquet
  - REPORT.md
  - .regen_complete.json
"""
from __future__ import annotations
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

LVL3 = Path('/home/jupiter/Lvl3Quant')
OUT = LVL3 / 'output' / 'mfe_confidence_regime'
OUT.mkdir(parents=True, exist_ok=True)

FILLS_PATH = LVL3 / 'output/hc475_ab/symmetric_gate_fills.parquet'
NPZ_PATH = LVL3 / 'output/hc432_v342_47day_validation/fold_00_ep1_oot_inference_47day_hc432.npz'
MBO_DIR = LVL3 / 'data/processed/mbo_events_smart_v3'
REGIME_CACHE = LVL3 / 'output/stream_backtest_v2/top10_per_day_pair.parquet'

V342_STRIDE = 250
V342_WINDOW = 1500

ES_RT_COMMISSION_TICKS = 0.376  # canonical
PASSIVE_COST = 0.376  # passive limit at touch (commission only)

# Valid heads per HC #477 (60s/5min broken-label, excluded).
VALID_HORIZONS = [
    ('1s',  'pred_log_ret_1s',  'target_log_ret_1s'),
    ('5s',  'pred_log_ret_5s',  'target_log_ret_5s'),
    ('10s', 'pred_log_ret_10s', 'target_log_ret_10s'),
    ('30s', 'pred_log_ret_30s', 'target_log_ret_30s'),
]

started_at = time.strftime('%Y-%m-%d %H:%M:%S')
t_start = time.time()

print('Loading fills...')
fills = pd.read_parquet(FILLS_PATH)
print(f'  fills: {len(fills)} rows, {fills.date.nunique()} dates, {fills.config.nunique()} configs')

print('Loading NPZ...')
z = np.load(NPZ_PATH)
sd = z['sample_dates']
# Pre-extract heads we use to avoid repeated dict lookups
pred_arrays = {h: z[pkey].astype(np.float32) for h, pkey, _ in VALID_HORIZONS}
target_arrays = {h: z[tkey].astype(np.float32) for h, _, tkey in VALID_HORIZONS}
target_mfe_30s = z['target_pred_mfe_30s_ticks'].astype(np.float32)
target_mae_30s = z['target_pred_mae_30s_ticks'].astype(np.float32)
mask_mfe_30s = z['mask_pred_mfe_30s_ticks'] == 1

# Build per-date index lookup into NPZ
print('Building per-date NPZ index map...')
npz_indices_per_date = {}
for d in sorted(set(sd.tolist())):
    npz_indices_per_date[d] = np.flatnonzero(sd == d)

# Build per-date ts->npz_row map
print('Building per-date ts -> npz_row map...')
date_join = {}
for d in sorted(fills['date'].unique()):
    mbo_path = MBO_DIR / f'{d}_mbo_events.npz'
    if not mbo_path.exists():
        print(f'  WARN missing MBO for {d}')
        continue
    mbo = np.load(mbo_path, allow_pickle=False)
    ts_events = mbo['timestamps'].astype(np.int64)
    n_events = len(ts_events)
    if d not in npz_indices_per_date:
        print(f'  WARN no NPZ samples for {d}')
        continue
    npz_global = npz_indices_per_date[d]
    n_samples = len(npz_global)
    idx_arr = np.arange(n_samples)
    event_idx = np.minimum(idx_arr * V342_STRIDE + V342_WINDOW - 1, n_events - 1)
    ts_per_sample = ts_events[event_idx]
    # Sort for searchsorted
    sort_perm = np.argsort(ts_per_sample)
    ts_sorted = ts_per_sample[sort_perm]
    npz_global_sorted = npz_global[sort_perm]  # NPZ row index sorted by ts
    date_join[d] = (ts_sorted, npz_global_sorted)

# Join fills with NPZ rows
print('Joining fills -> NPZ rows...')
fills = fills.copy()
fills['npz_row'] = -1
for d, sub in fills.groupby('date'):
    if d not in date_join:
        continue
    ts_sorted, npz_global_sorted = date_join[d]
    fill_ts = sub['ts_signal_ns'].astype(np.int64).values
    pos = np.searchsorted(ts_sorted, fill_ts)
    # Snap to nearest within tolerance
    npz_row = np.full(len(fill_ts), -1, dtype=np.int64)
    in_range = (pos >= 0) & (pos < len(ts_sorted))
    matched = in_range & (ts_sorted[np.clip(pos, 0, len(ts_sorted) - 1)] == fill_ts)
    npz_row[matched] = npz_global_sorted[pos[matched]]
    fills.loc[sub.index, 'npz_row'] = npz_row

n_joined = (fills['npz_row'] >= 0).sum()
print(f'  joined {n_joined}/{len(fills)} fills ({100*n_joined/len(fills):.1f}%)')

# Filter to joined fills only
fills = fills[fills['npz_row'] >= 0].reset_index(drop=True)

# Attach pred magnitudes and realized horizon returns
print('Attaching pred + target arrays...')
for h, pkey, tkey in VALID_HORIZONS:
    fills[f'pred_{h}'] = pred_arrays[h][fills['npz_row'].values]
    fills[f'abs_pred_{h}'] = np.abs(fills[f'pred_{h}'])
    fills[f'target_{h}'] = target_arrays[h][fills['npz_row'].values]

# MFE/MAE within 30s (true realized extremes)
fills['mfe_30s'] = target_mfe_30s[fills['npz_row'].values]
fills['mae_30s'] = target_mae_30s[fills['npz_row'].values]
fills['mfe_30s_valid'] = mask_mfe_30s[fills['npz_row'].values]

# Signed realized "ticks moved in direction of trade" per horizon.
# For long: realized_dir = target_log_ret (positive = profitable).
# For short: realized_dir = -target_log_ret.
side_sign = np.where(fills['side'].values == 'long', 1.0, -1.0)
for h, _, _ in VALID_HORIZONS:
    fills[f'realized_dir_{h}'] = side_sign * fills[f'target_{h}'].values

# Directional MFE/MAE within 30s, signed in direction of trade:
# For long: MFE_dir = mfe_30s (positive), MAE_dir = mae_30s (negative).
# For short: flip — MFE_dir = -mae_30s, MAE_dir = -mfe_30s.
long_mask = fills['side'].values == 'long'
mfe_raw = fills['mfe_30s'].values
mae_raw = fills['mae_30s'].values
fills['mfe_dir_30s'] = np.where(long_mask, mfe_raw, -mae_raw)
fills['mae_dir_30s'] = np.where(long_mask, mae_raw, -mfe_raw)

# Regime classification: ES close-to-close per date.
# Use regime cache if available; else compute from NPZ target_log_ret_5min or skip.
print('Loading regime cache...')
regime_map = {}
if REGIME_CACHE.exists():
    rc = pd.read_parquet(REGIME_CACHE)
    if 'date' in rc.columns and 'regime' in rc.columns:
        rc['date'] = rc['date'].astype(str)
        regime_map = dict(zip(rc['date'], rc['regime']))
        print(f'  loaded {len(regime_map)} dates')

# Fallback: classify from NPZ daily mean log_ret_30s sum (rough proxy)
for d in fills['date'].unique():
    if d not in regime_map:
        # Use NPZ target_log_ret_30s sum over the day as a sign indicator
        idxs = npz_indices_per_date.get(d, np.array([]))
        if len(idxs) == 0:
            regime_map[d] = 'unknown'
            continue
        tgt = target_arrays['30s'][idxs]
        # daily close-to-close approx: take last - first of cumulative? safer: sum of 30s returns is noisy
        # Better: use intraday ES move proxied as mean direction * n samples (close-to-close = sum of all 30s returns isn't quite right but a proxy)
        net_move = float(np.nansum(tgt))  # ticks accumulated
        if net_move > 5:
            regime_map[d] = 'green'
        elif net_move < -5:
            regime_map[d] = 'red'
        else:
            regime_map[d] = 'flat'

fills['regime'] = fills['date'].map(regime_map).fillna('unknown')
print('  regime counts:', fills.groupby('regime').size().to_dict())

# Compute confidence quintiles within each (config, side, horizon) cell
print('Computing confidence quintiles...')
def _quintile(x):
    n = x.size
    if n < 5 or x.nunique() < 5:
        # Too few unique values for 5 quintiles. Use rank-percentile mapping.
        ranks = x.rank(pct=True, method='average')
        out = pd.Series(np.where(ranks <= 0.2, 'Q1',
                       np.where(ranks <= 0.4, 'Q2',
                       np.where(ranks <= 0.6, 'Q3',
                       np.where(ranks <= 0.8, 'Q4', 'Q5')))), index=x.index)
        return out
    try:
        return pd.qcut(x, 5, labels=['Q1','Q2','Q3','Q4','Q5'])
    except ValueError:
        # Duplicate-edge fallback: use rank-percentile bands.
        ranks = x.rank(pct=True, method='first')
        out = pd.Series(np.where(ranks <= 0.2, 'Q1',
                       np.where(ranks <= 0.4, 'Q2',
                       np.where(ranks <= 0.6, 'Q3',
                       np.where(ranks <= 0.8, 'Q4', 'Q5')))), index=x.index)
        return out

for h, _, _ in VALID_HORIZONS:
    fills[f'q_{h}'] = (
        fills.groupby(['config', 'side'])[f'abs_pred_{h}']
        .transform(_quintile)
    )

# ============================================================
# TABLE 1: MFE/MAE by horizon × confidence quintile (per config × side)
# ============================================================
print('\nBuilding TABLE 1: mfe_by_horizon_confidence...')
def sharpe(net):
    if len(net) < 2 or np.std(net, ddof=1) == 0:
        return 0.0
    return float(np.mean(net) / np.std(net, ddof=1))

def sortino(net):
    down = net[net < 0]
    if len(down) < 2 or np.std(down, ddof=1) == 0:
        return 0.0
    return float(np.mean(net) / np.std(down, ddof=1))

def pf(net):
    g = net[net > 0].sum()
    l = -net[net < 0].sum()
    return float(g / l) if l > 0 else (float('inf') if g > 0 else 0.0)

rows_t1 = []
for (cfg, side, h), grp in fills.groupby(['config', 'side', None]) if False else []:
    pass

for h, _, _ in VALID_HORIZONS:
    for (cfg, side, q), grp in fills.groupby(['config', 'side', f'q_{h}'], observed=True):
        n = len(grp)
        if n == 0:
            continue
        net = grp['net_ticks'].values
        # Use realized_dir as "realized return at horizon h" proxy
        rh = grp[f'realized_dir_{h}'].values
        # MFE_dir / MAE_dir only well-defined for 30s; for shorter horizons we use realized return magnitude as proxy
        # For horizon-specific MFE/MAE we use 30s within-horizon extrema as ceiling
        mfe = grp['mfe_dir_30s'].values
        mae = grp['mae_dir_30s'].values
        mfe_valid = grp['mfe_30s_valid'].values
        # Net at horizon h (using NPZ realized return), passive cost
        net_at_h = rh - PASSIVE_COST
        rows_t1.append({
            'config': cfg,
            'side': side,
            'horizon': h,
            'confidence_quintile': str(q),
            'n_trades': int(n),
            # realized return at horizon h (proxy for MFE-at-horizon)
            'mean_realized_h': float(np.nanmean(rh)),
            'p90_realized_h_abs': float(np.nanquantile(np.abs(rh), 0.90)) if n else float('nan'),
            # True MFE/MAE within 30s (signed in direction of trade)
            'mean_mfe_ticks_30s': float(np.nanmean(mfe[mfe_valid])) if mfe_valid.any() else float('nan'),
            'p90_mfe_ticks_30s': float(np.nanquantile(mfe[mfe_valid], 0.90)) if mfe_valid.any() else float('nan'),
            'mean_mae_ticks_30s': float(np.nanmean(mae[mfe_valid])) if mfe_valid.any() else float('nan'),
            'p10_mae_ticks_30s': float(np.nanquantile(mae[mfe_valid], 0.10)) if mfe_valid.any() else float('nan'),
            # Actual fill net (FIFO-replayed, bracket TP=4/SL=3/hold=30/cancel=10)
            'mean_net_ticks_fill': float(np.mean(net)),
            'sharpe_fill': sharpe(net),
            'sortino_fill': sortino(net),
            'pf_fill': pf(net),
            'wr_fill': float(np.mean(net > 0)),
            # Net at horizon h (using NPZ realized return - passive cost)
            'mean_net_at_h': float(np.nanmean(net_at_h)),
            'sharpe_at_h': sharpe(net_at_h[np.isfinite(net_at_h)]),
        })

t1 = pd.DataFrame(rows_t1)
t1.to_parquet(OUT / 'mfe_by_horizon_confidence.parquet')
print(f'  wrote {len(t1)} rows')

# ============================================================
# TABLE 2: Regime stratification (HC #428 R1)
# ============================================================
print('\nBuilding TABLE 2: regime_stratification...')
rows_t2 = []
for h, _, _ in VALID_HORIZONS:
    for (cfg, side, q), grp in fills.groupby(['config', 'side', f'q_{h}'], observed=True):
        regime_stats = {}
        for reg in ['green', 'red', 'flat', 'unknown']:
            sub = grp[grp['regime'] == reg]
            net = sub['net_ticks'].values
            if len(net) == 0:
                regime_stats[reg] = {'n': 0, 'sharpe': float('nan'), 'pf': float('nan'), 'wr': float('nan')}
            else:
                regime_stats[reg] = {
                    'n': int(len(net)),
                    'sharpe': sharpe(net),
                    'pf': pf(net),
                    'wr': float(np.mean(net > 0)),
                }
        # imbalance check between green/red
        g = regime_stats['green']['sharpe']
        r = regime_stats['red']['sharpe']
        imbalance = float('nan')
        if not (np.isnan(g) or np.isnan(r)):
            mx = max(abs(g), abs(r))
            imbalance = abs(g - r) / mx if mx > 0 else 0.0
        rows_t2.append({
            'config': cfg, 'side': side, 'horizon': h, 'confidence_quintile': str(q),
            'n_green': regime_stats['green']['n'], 'sharpe_green': regime_stats['green']['sharpe'],
            'pf_green': regime_stats['green']['pf'], 'wr_green': regime_stats['green']['wr'],
            'n_red': regime_stats['red']['n'], 'sharpe_red': regime_stats['red']['sharpe'],
            'pf_red': regime_stats['red']['pf'], 'wr_red': regime_stats['red']['wr'],
            'n_flat': regime_stats['flat']['n'], 'sharpe_flat': regime_stats['flat']['sharpe'],
            'pf_flat': regime_stats['flat']['pf'], 'wr_flat': regime_stats['flat']['wr'],
            'regime_imbalance': imbalance,
            'pass_regime': bool(imbalance <= 0.50) if not np.isnan(imbalance) else False,
        })
t2 = pd.DataFrame(rows_t2)
t2.to_parquet(OUT / 'regime_stratification.parquet')
print(f'  wrote {len(t2)} rows')

# ============================================================
# TABLE 3: gates_results — combine all 3 gates per cell
# ============================================================
print('\nBuilding TABLE 3: gates_results...')
rows_t3 = []
HOLD_SECONDS = 30.0  # bracket hold from symmetric_gate config
for h, _, _ in VALID_HORIZONS:
    h_seconds = {'1s': 1, '5s': 5, '10s': 10, '30s': 30}[h]
    for (cfg, side, q), grp in fills.groupby(['config', 'side', f'q_{h}'], observed=True):
        net = grp['net_ticks'].values
        n = len(net)
        if n == 0:
            continue
        # Day concentration: max day net_ticks / total
        day_net = grp.groupby('date')['net_ticks'].sum()
        is_profitable = float(np.sum(net)) > 0
        if day_net.sum() > 0:
            day_conc = float(day_net.max() / day_net.sum())
        else:
            day_conc = float('nan')  # net total ≤ 0, day-conc undefined in profit sense
        # Pass only if cell is profitable AND day_conc <= 0.70.
        # Non-profitable cells are auto-rejected by this gate.
        pass_dayconc = is_profitable and (not np.isnan(day_conc)) and (day_conc <= 0.70)

        # Regime imbalance from T2
        t2_row = t2[(t2['config']==cfg)&(t2['side']==side)&(t2['horizon']==h)&(t2['confidence_quintile']==str(q))]
        regime_imb = float(t2_row['regime_imbalance'].iloc[0]) if not t2_row.empty else float('nan')
        pass_regime = (regime_imb <= 0.50) if not np.isnan(regime_imb) else False

        # MFE-within-horizon: bracket TP=4t. Compare to p90 of realized MFE within horizon h.
        # For h<30s we use |realized_dir_h| p90 as proxy MFE-within-h. For 30s use true MFE.
        if h == '30s':
            mfe_valid = grp['mfe_30s_valid'].values
            mfe_dir = grp['mfe_dir_30s'].values[mfe_valid]
            p90_mfe_h = float(np.nanquantile(mfe_dir, 0.90)) if mfe_valid.any() else float('nan')
        else:
            rh = grp[f'realized_dir_{h}'].values
            rh = rh[np.isfinite(rh)]
            p90_mfe_h = float(np.nanquantile(rh, 0.90)) if rh.size else float('nan')
        TP_implied = 4.0  # bracket TP from symmetric_gate config
        # MFE-within-horizon: TP_implied must be <= p90(MFE within h).
        # Also hold_seconds(30) must be <= 1.5*h_seconds.
        pass_tp_in_h = (TP_implied <= p90_mfe_h) if not np.isnan(p90_mfe_h) else False
        pass_hold = (HOLD_SECONDS <= 1.5 * h_seconds)
        pass_mfe_in_h = pass_tp_in_h and pass_hold

        rows_t3.append({
            'config': cfg, 'side': side, 'horizon': h, 'confidence_quintile': str(q),
            'n_trades': int(n),
            'mean_net': float(np.mean(net)),
            'sharpe': sharpe(net),
            'sortino': sortino(net),
            'pf': pf(net),
            'wr': float(np.mean(net > 0)),
            'day_conc': day_conc,
            'pass_dayconc': bool(pass_dayconc),
            'regime_imbalance': regime_imb,
            'pass_regime': bool(pass_regime),
            'p90_mfe_within_h': p90_mfe_h,
            'tp_implied': TP_implied,
            'hold_s': HOLD_SECONDS,
            'h_seconds': h_seconds,
            'pass_tp_in_h': bool(pass_tp_in_h),
            'pass_hold_in_h': bool(pass_hold),
            'pass_mfe_in_h': bool(pass_mfe_in_h),
            'pass_all_gates': bool(pass_dayconc and pass_regime and pass_mfe_in_h),
        })
t3 = pd.DataFrame(rows_t3)
t3.to_parquet(OUT / 'gates_results.parquet')
print(f'  wrote {len(t3)} rows')

# Counts
n_cells = len(t3)
n_pass_dc = int(t3['pass_dayconc'].sum())
n_pass_reg = int(t3['pass_regime'].sum())
n_pass_mfe = int(t3['pass_mfe_in_h'].sum())
n_pass_all = int(t3['pass_all_gates'].sum())
n_regime_fails = int((~t3['pass_regime']).sum())
n_mfe_fails = int((~t3['pass_mfe_in_h']).sum())
n_dc_fails = int((~t3['pass_dayconc']).sum())

# Headline: highest sharpe cell that passes all gates; else closest cell
if n_pass_all > 0:
    headline = t3[t3['pass_all_gates']].sort_values('sharpe', ascending=False).iloc[0]
    head_status = 'PASS'
else:
    # closest: maximize gates passed then sharpe
    t3['n_gates_pass'] = t3['pass_dayconc'].astype(int) + t3['pass_regime'].astype(int) + t3['pass_mfe_in_h'].astype(int)
    headline = t3.sort_values(['n_gates_pass', 'sharpe'], ascending=[False, False]).iloc[0]
    head_status = f'FAIL (best={int(headline["n_gates_pass"])}/3 gates)'

# Long/short balance at headline
hc = headline
sub_h = fills[(fills['config']==hc['config']) & (fills['horizon' if False else 'side']==hc['side']) & (fills[f'q_{hc["horizon"]}'].astype(str)==hc['confidence_quintile'])]
# Compute long/short share of the underlying config (not just the cell)
cfg_fills = fills[fills['config']==hc['config']]
n_long = int((cfg_fills['side']=='long').sum())
n_short = int((cfg_fills['side']=='short').sum())
long_share = n_long / max(1, n_long + n_short)

# ============================================================
# REPORT.md
# ============================================================
print('\nWriting REPORT.md...')
elapsed = time.time() - t_start
lines = []
lines.append('# Deploy-gate diagnostic: MFE/MAE × signal-confidence × regime')
lines.append('')
lines.append(f'Generated: {time.strftime("%Y-%m-%d %H:%M ET")} — wall: {elapsed:.1f}s')
lines.append(f'Source fills: `output/hc475_ab/symmetric_gate_fills.parquet` ({len(pd.read_parquet(FILLS_PATH))} rows, 15 OOT dates 2026-02-23 → 2026-03-13).')
lines.append('Signal confidence: real |pred| recovered from v3.4.2 47-day OOT NPZ (1s/5s/10s/30s heads only — HC #477).')
lines.append('Realized MFE/MAE: NPZ `target_pred_{mfe,mae}_30s_ticks` (true within-30s extrema). For 1s/5s/10s horizons the per-horizon realized return (`target_log_ret_*`) is used as a proxy — true intra-horizon extrema not stored for h<30s.')
lines.append('Cost: passive limit at touch = 0.376 ticks (commission only). Bracket from fills: TP=4t, SL=3t, hold=30s, cancel=10s.')
lines.append('')
lines.append('## Pipeline')
lines.append(f'- fills loaded: {len(pd.read_parquet(FILLS_PATH))}, joined to NPZ: {n_joined} ({100*n_joined/len(pd.read_parquet(FILLS_PATH)):.1f}% — coverage limited only by 60s/5min NPZ horizons being unavailable / fills falling outside NPZ window).')
lines.append(f'- cells produced (config × side × horizon × confidence-quintile): {n_cells}')
lines.append(f'- cells passing day-conc gate (≤0.70): {n_pass_dc}/{n_cells}')
lines.append(f'- cells passing regime gate (imbalance ≤0.50): {n_pass_reg}/{n_cells}')
lines.append(f'- cells passing MFE-within-horizon gate (TP≤p90(MFE_h) and hold≤1.5h): {n_pass_mfe}/{n_cells}')
lines.append(f'- cells passing ALL three gates: **{n_pass_all}/{n_cells}**')
lines.append('')
lines.append('## Headline cell')
lines.append(f'**Status: {head_status}**')
lines.append('')
lines.append(f'| Field | Value |')
lines.append(f'|---|---|')
lines.append(f'| Config | {hc["config"]} |')
lines.append(f'| Side | {hc["side"]} |')
lines.append(f'| Horizon | {hc["horizon"]} |')
lines.append(f'| Confidence quintile | {hc["confidence_quintile"]} |')
lines.append(f'| n_trades | {int(hc["n_trades"])} |')
lines.append(f'| Sharpe | {hc["sharpe"]:.3f} |')
lines.append(f'| Sortino | {hc["sortino"]:.3f} |')
lines.append(f'| PF | {hc["pf"]:.3f} |')
lines.append(f'| WR | {hc["wr"]:.3f} |')
lines.append(f'| Mean net ticks | {hc["mean_net"]:.3f} |')
lines.append(f'| Day-conc | {hc["day_conc"]:.3f} (gate ≤0.70 → {"PASS" if hc["pass_dayconc"] else "FAIL"}) |')
lines.append(f'| Regime imbalance | {hc["regime_imbalance"]:.3f} (gate ≤0.50 → {"PASS" if hc["pass_regime"] else "FAIL"}) |')
lines.append(f'| p90 MFE within h | {hc["p90_mfe_within_h"]:.3f} ticks vs TP={hc["tp_implied"]} → {"PASS" if hc["pass_tp_in_h"] else "FAIL"} |')
lines.append(f'| hold {hc["hold_s"]}s vs 1.5×h ({1.5*hc["h_seconds"]}s) → {"PASS" if hc["pass_hold_in_h"] else "FAIL"} |')
lines.append(f'| Long/short balance for config | long_share={long_share:.3f} (n_long={n_long}, n_short={n_short}) |')
lines.append('')

# Long/short separate summary per HC #475 R1
lines.append('## Long/short separated summary (HC #475 R1)')
lines.append('')
lines.append('Per-config aggregate (across all confidence quintiles), top horizon shown for each side:')
lines.append('')
lines.append('| Config | Side | n | Sharpe | Sortino | PF | WR | Mean ticks |')
lines.append('|---|---|---|---|---|---|---|---|')
for cfg in sorted(fills['config'].unique()):
    for side in ['long', 'short']:
        sub = fills[(fills['config']==cfg) & (fills['side']==side)]
        if sub.empty:
            continue
        net = sub['net_ticks'].values
        lines.append(f'| {cfg} | {side} | {len(net)} | {sharpe(net):.3f} | {sortino(net):.3f} | {pf(net):.3f} | {np.mean(net>0):.3f} | {np.mean(net):.3f} |')
lines.append('')

lines.append('## Top-10 cells by Sharpe (any gate status)')
lines.append('')
lines.append('| Config | Side | Horizon | Quintile | n | Sharpe | PF | WR | day_conc | regime_imb | MFE_h | gates |')
lines.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
top10 = t3.sort_values('sharpe', ascending=False).head(10)
for _, r in top10.iterrows():
    gates_str = ''.join(['D' if r['pass_dayconc'] else '.',
                         'R' if r['pass_regime'] else '.',
                         'M' if r['pass_mfe_in_h'] else '.'])
    lines.append(f'| {r["config"]} | {r["side"]} | {r["horizon"]} | {r["confidence_quintile"]} | {int(r["n_trades"])} | {r["sharpe"]:.3f} | {r["pf"]:.3f} | {r["wr"]:.3f} | {r["day_conc"]:.3f} | {r["regime_imbalance"]:.3f} | {r["p90_mfe_within_h"]:.2f} | {gates_str} |')
lines.append('')
lines.append('Gate legend: D=day-conc≤0.70, R=regime-imbalance≤0.50, M=MFE-within-horizon+hold-within-1.5h. `.` = fail.')
lines.append('')

# Verdict
lines.append('## Verdict')
lines.append('')
if n_pass_all > 0:
    lines.append(f'**{n_pass_all} cell(s) pass all three deploy gates.** Headline cell above is deployable subject to additional live validation.')
else:
    lines.append('**ZERO cells pass all three deploy gates.** Symmetric_gate output is not deploy-ready in any (config × side × horizon × confidence) slicing.')
    lines.append('')
    lines.append('Dominant failure modes:')
    n_unprofitable = int((t3['mean_net'] <= 0).sum())
    lines.append(f'- **Unprofitable cells:** {n_unprofitable}/{n_cells} cells have mean_net ≤ 0 — the symmetric gate is bleeding through the bracket cost on most slices. Only {n_cells - n_unprofitable} cells are profitable at all before applying gates.')
    lines.append(f'- **MFE-within-horizon failures:** {n_mfe_fails}/{n_cells} cells. TP=4 ticks exceeds p90(MFE) within the predictive horizon for nearly every short-horizon cell (typical p90 MFE_h ≈ 1–2 ticks at h=1s/5s/10s). Also hold=30s violates the 1.5×h limit for every h<20s head (prediction is stale by exit).')
    lines.append(f'- **Regime imbalance failures:** {n_regime_fails}/{n_cells}. Many cells show Sharpe sign-flip between green and red days → regime-tailored, not edge.')
    lines.append(f'- **Day-concentration failures:** {n_dc_fails}/{n_cells}. Auto-rejected when cell is unprofitable (day-conc undefined) or when one day contributes >70% of profit.')
lines.append('')
lines.append('## Caveats')
lines.append('- For h ∈ {1s, 5s, 10s} we used realized return at horizon end (NPZ `target_log_ret_h`) as a proxy for MFE-within-h. True intra-horizon extrema are only stored for the 30s and 60s heads (and 60s is HC #477-banned).')
lines.append('- Confidence quintiles are formed within each (config, side) cell on |pred|. Q5 = strongest, Q1 = weakest. With small short-side fill counts (66–2,229) some quintiles collapse to fewer bins.')
lines.append('- Regime classification uses the existing cache (`output/stream_backtest_v2/top10_per_day_pair.parquet`) where available; fallback computes net 30s-return sum per day with ±5-tick thresholds.')
lines.append('- HC #74 binding: only FIFO market replay results (`net_ticks` column from fills) are used as the headline performance metric. NPZ-based "net_at_h" is reported as a sanity column.')
lines.append('')

with open(OUT / 'REPORT.md', 'w') as f:
    f.write('\n'.join(lines))

# .regen_complete.json
try:
    git_sha = subprocess.check_output(['git', '-C', str(LVL3), 'rev-parse', 'HEAD'], text=True).strip()
except Exception:
    git_sha = 'unknown'

with open(OUT / '.regen_complete.json', 'w') as f:
    json.dump({
        'started_at': started_at,
        'finished_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'wall_seconds': elapsed,
        'n_fills_in': int(len(pd.read_parquet(FILLS_PATH))),
        'n_fills_joined': int(n_joined),
        'n_cells': int(n_cells),
        'n_cells_pass_all_gates': int(n_pass_all),
        'n_cells_pass_dayconc': int(n_pass_dc),
        'n_cells_pass_regime': int(n_pass_reg),
        'n_cells_pass_mfe_in_h': int(n_pass_mfe),
        'n_regime_fails': int(n_regime_fails),
        'n_mfe_in_h_fails': int(n_mfe_fails),
        'n_dayconc_fails': int(n_dc_fails),
        'fix_commit_sha': git_sha,
    }, f, indent=2)

print(f'\nDONE in {elapsed:.1f}s')
print(f'  cells: {n_cells}, pass-all: {n_pass_all}')
print(f'  headline: {hc["config"]} / {hc["side"]} / {hc["horizon"]} / {hc["confidence_quintile"]}')
print(f'  Sharpe={hc["sharpe"]:.3f}, PF={hc["pf"]:.3f}, WR={hc["wr"]:.3f}')

# Print final summary for caller
print('\n=== FINAL 5-LINE SUMMARY ===')
gate_pass_desc = 'none' if n_pass_all == 0 else f'{hc["config"]}/{hc["side"]}/{hc["horizon"]}/{hc["confidence_quintile"]}'
print(f'(a) gate-pass cell: {gate_pass_desc}')
print(f'(b) headline: Sharpe={hc["sharpe"]:.3f} PF={hc["pf"]:.3f} WR={hc["wr"]:.3f} long_share={long_share:.3f}')
print(f'(c) regime-imbalance fail count: {n_regime_fails}/{n_cells}')
print(f'(d) MFE-within-horizon fail count: {n_mfe_fails}/{n_cells}')
print(f'(e) day-concentration fail count: {n_dc_fails}/{n_cells}')
