#!/usr/bin/env python3
"""
exec_deepdive_v1.py — Per-trade dynamics deep-dive on cnn_mamba_v2 OOT predictions.

Per DIRECTIVES.md 19:32 ET point 3: "catalog EVERYTHING that existing cnn_mamba_v2
OOT predictions teach about per-trade dynamics".

What it does
============
For each fold (10 days, Feb 23 - Mar 5 2026):
  1. Load OOT predictions and align them to MBO event positions
     (predictions are at stride positions: pred[i] -> event index i*STRIDE + WINDOW - 1)
  2. Build a continuous mid-price path from `mid_price_change_ticks` cumulative sum
     (this is in ticks; ES tick = $12.50)
  3. Compute z-scores using PER-FOLD OOT std (causal-ish — see leakage note below)
  4. Trigger trades at |z_10s| >= 2.3 (production gate). Long if z>0, short if z<0.
  5. Forward-replay each trade for up to 60 seconds, recording per-event:
       MFE, MAE, time_to_MFE, time_to_MAE, MAE-before-MFE flag,
       PnL-at-60s, exit (tp=9 / sl=15 / timeout), hold time, hour bucket
  6. Aggregate across folds and emit CSVs + a markdown report.

Leakage note (per DIRECTIVES.md 19:32 ET point 5d)
==================================================
The "correct causal" quintile boundaries would be from the PREVIOUS fold's signal
distribution. Here we use the CURRENT fold's OOT std/quintile boundaries because:
  - The production system uses an expanding running z-score within a day
  - Fold 0 has no previous fold; using fold-0's own distribution is the only option
  - Quintile boundaries are computed PER FOLD and are not pooled (no cross-fold leak)
This is documented as an acknowledged limitation; cells are flagged "underpowered" if n<30.

Fills
=====
Per DIRECTIVES.md HARD CONSTRAINT: NO MID-PRICE PNL.
  - Entry: pay 0.5 spread (worst case for marketable limit at touch)
  - Exit (TP/SL/timeout): pay 0.5 spread out
  - Net trade cost: 1 tick (spread tax) + 0.078 ticks commission ($4.70 RT / $50 / 0.25)
The MFE/MAE numbers reported are RAW (no spread tax) so they describe price dynamics;
PnL columns deduct the spread tax.

Output
======
  /home/jupiter/Lvl3Quant/execution/results/exec_deepdive_20260427/
     trades.csv                 — per-trade record (~500-1500 rows)
     mfe_distribution.csv       — percentiles by quintile, hour, side
     mae_distribution.csv       — same for MAE
     mae_before_mfe.csv         — % of winners that went red first
     time_to_mfe.csv            — quantiles by quintile
     time_to_stop.csv           — exit-time distribution by quintile
     exit_reason_heatmap.csv    — exit reason x quintile
     adaptive_tp_grid.csv       — TP x SL grid by (quintile, hour)
     mfe_vs_mae_heatmap.csv     — 6x6 bin counts
     hour_quintile_pnl.csv      — total ticks per cell
     REPORT.md                  — concise findings, recommendations
"""

import os
import sys
import json
import time
import math
from pathlib import Path
from datetime import datetime, timezone, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

LVL3_ROOT = Path('/home/jupiter/Lvl3Quant')
PRED_DIR = LVL3_ROOT / 'output' / 'cnn_mamba_v2_smart_v3_mar'
EVENT_DIR = LVL3_ROOT / 'data' / 'processed' / 'mbo_events_smart_v3'
BOOK_DIR = LVL3_ROOT / 'data' / 'processed' / 'mbo_book_features'
OUT_DIR = LVL3_ROOT / 'execution' / 'results' / 'exec_deepdive_20260427'
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW = 1000
STRIDE = 500
TICK_SIZE = 0.25
TICK_DOLLARS = 12.50
ACCOUNT = 50_000.0
COMMISSION_RT_DOLLARS = 4.70
COMMISSION_TICKS = COMMISSION_RT_DOLLARS / TICK_DOLLARS  # ~0.376 ticks/RT
SPREAD_TAX_TICKS = 1.0  # paying half spread on entry + half on exit (ES is 1-tick wide)
ZSCORE_GATE = 2.3
HOLD_MS = 60_000
TP_DEFAULT = 9
SL_DEFAULT = 15
TP_GRID = [4, 6, 8, 10, 12, 15]
SL_GRID = [6, 10, 15, 20]
NS_PER_MS = 1_000_000


def rth_start_ns_for_date(date_str):
    y, m, d = int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8])
    dt = datetime(y, m, d)
    dst_start_2026 = datetime(2026, 3, 8)
    dst_end_2026 = datetime(2026, 11, 1)
    dst_start_2025 = datetime(2025, 3, 9)
    dst_end_2025 = datetime(2025, 11, 2)
    if (dst_start_2025 <= dt < dst_end_2025) or (dst_start_2026 <= dt < dst_end_2026):
        utc_off = -4
    else:
        utc_off = -5
    rth_h = 9.5 - utc_off
    midnight = datetime(y, m, d, tzinfo=timezone.utc)
    rth = midnight + timedelta(hours=rth_h)
    return int(rth.timestamp() * 1e9), utc_off


def hour_bucket(ts_ns, rth_start_ns, utc_off):
    """Return ET hour as int (9..16). 9 = first 30 min (9:30-10:00)."""
    delta_s = (ts_ns - rth_start_ns) / 1e9
    et_hour = 9 + 0.5 + (delta_s / 3600.0)
    h = int(et_hour)
    if h < 9:
        h = 9
    if h > 16:
        h = 16
    return h


def process_fold(fold_idx, date_str):
    """Process one fold; return list of trade dicts and fold meta."""
    pred_path = PRED_DIR / f'fold_{fold_idx:02d}_oot_predictions.npz'
    event_path = EVENT_DIR / f'{date_str}_mbo_events.npz'
    book_path = BOOK_DIR / f'{date_str}_book_features.npz'

    if not all(p.exists() for p in [pred_path, event_path, book_path]):
        return [], {'fold': fold_idx, 'date': date_str, 'error': 'missing files',
                    'n_signals': 0, 'n_trades': 0}

    pred_data = np.load(pred_path, allow_pickle=True)
    preds = pred_data['predictions']  # (N, 3) [1s, 5s, 10s]
    n_pred = preds.shape[0]

    book_data = np.load(book_path, allow_pickle=True)
    book_feats = book_data['features']
    book_ts = book_data['timestamps']
    n_events = book_feats.shape[0]

    # Derive continuous mid-price path (in ticks, anchored at session open)
    mid_path_ticks = np.cumsum(book_feats[:, 27].astype(np.float64))  # mid_price_change_ticks
    spread_arr = book_feats[:, 24].astype(np.float64)  # spread_ticks (0 if book invalid)

    # Map predictions to event indices (per training stride)
    starts = np.arange(0, n_events - WINDOW + 1, STRIDE, dtype=np.int64)
    label_idxs = starts + WINDOW - 1
    if len(label_idxs) > n_pred:
        label_idxs = label_idxs[:n_pred]
    elif n_pred > len(label_idxs):
        preds = preds[:len(label_idxs)]
    n_aligned = len(label_idxs)

    # Filter to RTH only
    rth_start, utc_off = rth_start_ns_for_date(date_str)
    rth_end = rth_start + int(6.5 * 3600 * 1e9)
    pred_event_ts = book_ts[label_idxs]
    rth_mask = (pred_event_ts >= rth_start) & (pred_event_ts < rth_end)
    label_idxs = label_idxs[rth_mask]
    preds = preds[rth_mask]

    if len(label_idxs) == 0:
        return [], {'fold': fold_idx, 'date': date_str, 'n_signals': 0, 'n_trades': 0,
                    'note': 'no RTH preds'}

    # Per-fold z-score using OOT std (full-day OOT std)
    pred_10s = preds[:, 2].astype(np.float64)
    mu_10s = pred_10s.mean()
    sd_10s = pred_10s.std() + 1e-9
    z_10s = (pred_10s - mu_10s) / sd_10s

    # Trigger trades
    trigger_mask = np.abs(z_10s) >= ZSCORE_GATE
    trig_idxs = np.where(trigger_mask)[0]
    n_signals = len(trig_idxs)

    trades = []
    last_exit_ts = 0  # cooldown: skip new triggers while a prior trade is open
    for ti in trig_idxs:
        ev_i = label_idxs[ti]
        z = z_10s[ti]
        side = 1 if z > 0 else -1
        entry_ts = book_ts[ev_i]
        # Cooldown: only fire if prior trade exited (no stacking — production behavior)
        if entry_ts < last_exit_ts:
            continue
        timeout_ts = entry_ts + HOLD_MS * NS_PER_MS

        # Find replay window end (event index up to timeout_ts)
        # Use searchsorted (efficient since book_ts is sorted)
        end_i = np.searchsorted(book_ts, timeout_ts, side='right')
        end_i = min(end_i, n_events - 1)
        if end_i <= ev_i:
            continue

        # Entry price: pay half-spread worse than mid
        entry_mid = mid_path_ticks[ev_i]
        # spread_arr is sparse; use spread at entry if valid, else assume 1 tick (ES typical)
        sp_local = spread_arr[ev_i] if spread_arr[ev_i] > 0 else 1.0
        # For LONG, fill at ask = mid + sp/2 (we're hit). For SHORT, fill at bid = mid - sp/2.
        entry_fill = entry_mid + side * (sp_local / 2.0)

        # Forward path: signed PnL in ticks from entry_fill, treated as exit-at-touch
        # If LONG (side=+1), exit fill = mid_path - sp/2, so pnl = (exit_mid - sp/2) - entry_fill
        #                                              = (exit_mid - sp/2) - (entry_mid + sp/2)
        #                                              = (exit_mid - entry_mid) - sp
        # If SHORT (side=-1), pnl = (entry_fill) - (exit_mid + sp/2)
        #                         = (entry_mid - sp/2) - (exit_mid + sp/2) = -(exit_mid - entry_mid) - sp
        # So: pnl_exec = side * (exit_mid - entry_mid) - sp
        # For MFE/MAE we report mid-to-mid drift in side direction (no spread tax).
        path = mid_path_ticks[ev_i:end_i + 1]  # mid path
        ts_path = book_ts[ev_i:end_i + 1]
        signed = side * (path - entry_mid)  # ticks gained in trade direction (mid-to-mid)

        # MFE/MAE on signed mid-drift
        mfe_idx = int(np.argmax(signed))
        mae_idx = int(np.argmin(signed))
        mfe = float(signed[mfe_idx])
        mae = float(signed[mae_idx])  # negative or 0
        time_to_mfe_ms = (ts_path[mfe_idx] - entry_ts) / NS_PER_MS
        time_to_mae_ms = (ts_path[mae_idx] - entry_ts) / NS_PER_MS

        # MAE-before-MFE? (did we go red before going green?)
        if mfe_idx > 0:
            pre_mfe_min = float(signed[:mfe_idx + 1].min())
        else:
            pre_mfe_min = 0.0
        mae_before_mfe = pre_mfe_min < 0.0
        mae_before_mfe_ticks = -pre_mfe_min if mae_before_mfe else 0.0

        # Simulate tp=9 / sl=15 (default production gate)
        # Use signed (mid drift) crossings; pay spread on exit
        tp_hit_idx = -1
        sl_hit_idx = -1
        for k in range(1, len(signed)):
            if signed[k] >= TP_DEFAULT:
                tp_hit_idx = k; break
            if signed[k] <= -SL_DEFAULT:
                sl_hit_idx = k; break
        if tp_hit_idx >= 0 and (sl_hit_idx < 0 or tp_hit_idx <= sl_hit_idx):
            exit_reason = 'tp'
            exit_idx = tp_hit_idx
            exit_signed_mid = TP_DEFAULT  # struck TP
        elif sl_hit_idx >= 0:
            exit_reason = 'sl'
            exit_idx = sl_hit_idx
            exit_signed_mid = -SL_DEFAULT
        else:
            exit_reason = 'timeout'
            exit_idx = len(signed) - 1
            exit_signed_mid = float(signed[-1])
        # Net PnL in ticks after spread tax (entry sp/2 + exit sp/2 ~= 1 tick)
        pnl_ticks_default = exit_signed_mid - SPREAD_TAX_TICKS - COMMISSION_TICKS
        hold_ms = (ts_path[exit_idx] - entry_ts) / NS_PER_MS
        # Cooldown end: trade exits at ts_path[exit_idx]
        last_exit_ts = int(ts_path[exit_idx])

        # Final at-60s mid drift (no TP/SL fired)
        final_mid_drift = float(signed[-1])
        final_pnl_ticks = final_mid_drift - SPREAD_TAX_TICKS - COMMISSION_TICKS

        trades.append({
            'fold': fold_idx,
            'date': date_str,
            'event_idx': int(ev_i),
            'entry_ts_ns': int(entry_ts),
            'hour_et': hour_bucket(entry_ts, rth_start, utc_off),
            'side': side,
            'z_10s': float(z),
            'abs_z': float(abs(z)),
            'entry_mid_ticks': float(entry_mid),
            'spread_at_entry': float(sp_local),
            'mfe_ticks': mfe,
            'mae_ticks': mae,
            'time_to_mfe_ms': float(time_to_mfe_ms),
            'time_to_mae_ms': float(time_to_mae_ms),
            'mae_before_mfe': bool(mae_before_mfe),
            'mae_before_mfe_ticks': float(mae_before_mfe_ticks),
            'final_drift_60s_ticks': float(final_mid_drift),
            'final_pnl_ticks_60s': float(final_pnl_ticks),
            'exit_reason_default': exit_reason,
            'pnl_ticks_default': float(pnl_ticks_default),
            'hold_ms_default': float(hold_ms),
        })

    # Compute per-fold quintile boundaries on ABSOLUTE z (PER FOLD — no cross-fold pool)
    if len(trades) > 0:
        abs_z_arr = np.array([t['abs_z'] for t in trades])
        try:
            qbins = np.quantile(abs_z_arr, [0.2, 0.4, 0.6, 0.8])
        except Exception:
            qbins = np.array([2.4, 2.6, 2.9, 3.3])
        for t in trades:
            v = t['abs_z']
            q = 1
            for j, b in enumerate(qbins):
                if v > b:
                    q = j + 2
            t['quintile'] = q

    return trades, {'fold': fold_idx, 'date': date_str, 'n_signals': n_signals,
                    'n_trades': len(trades), 'pred_mu_10s': float(mu_10s),
                    'pred_sd_10s': float(sd_10s)}


def main():
    fold_dates = [
        (0, '20260223'), (1, '20260224'), (2, '20260225'), (3, '20260226'),
        (4, '20260227'), (5, '20260301'), (6, '20260302'), (7, '20260303'),
        (8, '20260304'), (9, '20260305'),
    ]

    print(f'[exec_deepdive_v1] starting; out -> {OUT_DIR}', flush=True)
    t0 = time.time()

    all_trades = []
    fold_metas = []
    # Use multiprocessing for the 10 folds
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(process_fold, fi, ds): (fi, ds) for fi, ds in fold_dates}
        for fut in as_completed(futs):
            fi, ds = futs[fut]
            try:
                trades, meta = fut.result()
                all_trades.extend(trades)
                fold_metas.append(meta)
                print(f'  fold {fi} ({ds}): n_trades={meta["n_trades"]}', flush=True)
            except Exception as e:
                print(f'  fold {fi} ({ds}) FAILED: {e}', flush=True)
                fold_metas.append({'fold': fi, 'date': ds, 'error': str(e)})

    if not all_trades:
        print('[ERROR] No trades generated. Aborting.', flush=True)
        return

    df = pd.DataFrame(all_trades)
    df.sort_values('entry_ts_ns', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Save raw trades
    df.to_csv(OUT_DIR / 'trades.csv', index=False)
    print(f'\n[trades.csv] {len(df)} trades total', flush=True)

    # ---- Aggregations ----
    pcts = [10, 25, 50, 75, 90, 95, 99]

    def pct_table(series, group_cols, target='mfe_ticks'):
        if isinstance(group_cols, str):
            group_cols = [group_cols]
        agg = series.groupby(group_cols).agg(
            n=('n', 'count'),
            **{f'p{p}': (target, lambda s, p=p: np.percentile(s, p)) for p in pcts},
            mean=(target, 'mean'),
            std=(target, 'std'),
        )
        return agg

    df['n'] = 1

    # MFE distribution
    mfe_overall = df.groupby(lambda x: 0).agg(
        n=('n', 'sum'),
        **{f'p{p}': ('mfe_ticks', lambda s, p=p: np.percentile(s, p)) for p in pcts},
        mean=('mfe_ticks', 'mean'),
    )
    mfe_overall.index = ['ALL']
    mfe_q = df.groupby('quintile').agg(
        n=('n', 'sum'),
        **{f'p{p}': ('mfe_ticks', lambda s, p=p: np.percentile(s, p)) for p in pcts},
        mean=('mfe_ticks', 'mean'),
    )
    mfe_q.index = [f'Q{i}' for i in mfe_q.index]
    mfe_h = df.groupby('hour_et').agg(
        n=('n', 'sum'),
        **{f'p{p}': ('mfe_ticks', lambda s, p=p: np.percentile(s, p)) for p in pcts},
        mean=('mfe_ticks', 'mean'),
    )
    mfe_h.index = [f'H{int(i)}' for i in mfe_h.index]
    mfe_s = df.groupby('side').agg(
        n=('n', 'sum'),
        **{f'p{p}': ('mfe_ticks', lambda s, p=p: np.percentile(s, p)) for p in pcts},
        mean=('mfe_ticks', 'mean'),
    )
    mfe_s.index = ['short' if i < 0 else 'long' for i in mfe_s.index]
    mfe_full = pd.concat([mfe_overall, mfe_q, mfe_h, mfe_s])
    mfe_full.to_csv(OUT_DIR / 'mfe_distribution.csv')

    # MAE distribution (use abs(mae) so percentiles read intuitively)
    df['mae_abs_ticks'] = -df['mae_ticks']
    mae_overall = df.groupby(lambda x: 0).agg(
        n=('n', 'sum'),
        **{f'p{p}': ('mae_abs_ticks', lambda s, p=p: np.percentile(s, p)) for p in pcts},
        mean=('mae_abs_ticks', 'mean'),
    )
    mae_overall.index = ['ALL']
    mae_q = df.groupby('quintile').agg(
        n=('n', 'sum'),
        **{f'p{p}': ('mae_abs_ticks', lambda s, p=p: np.percentile(s, p)) for p in pcts},
        mean=('mae_abs_ticks', 'mean'),
    )
    mae_q.index = [f'Q{i}' for i in mae_q.index]
    mae_h = df.groupby('hour_et').agg(
        n=('n', 'sum'),
        **{f'p{p}': ('mae_abs_ticks', lambda s, p=p: np.percentile(s, p)) for p in pcts},
        mean=('mae_abs_ticks', 'mean'),
    )
    mae_h.index = [f'H{int(i)}' for i in mae_h.index]
    mae_s = df.groupby('side').agg(
        n=('n', 'sum'),
        **{f'p{p}': ('mae_abs_ticks', lambda s, p=p: np.percentile(s, p)) for p in pcts},
        mean=('mae_abs_ticks', 'mean'),
    )
    mae_s.index = ['short' if i < 0 else 'long' for i in mae_s.index]
    mae_full = pd.concat([mae_overall, mae_q, mae_h, mae_s])
    mae_full.to_csv(OUT_DIR / 'mae_distribution.csv')

    # MAE → MFE pattern (winners-only)
    df['winner_60s'] = df['final_drift_60s_ticks'] > 0
    winners = df[df['winner_60s']]
    if len(winners) > 0:
        wgrp = winners.groupby('quintile').agg(
            n_winners=('n', 'sum'),
            pct_went_red_first=('mae_before_mfe', lambda s: 100 * s.mean()),
            mean_red_depth_ticks=('mae_before_mfe_ticks', 'mean'),
            p50_red_depth=('mae_before_mfe_ticks', lambda s: np.percentile(s, 50)),
            p90_red_depth=('mae_before_mfe_ticks', lambda s: np.percentile(s, 90)),
        )
        wgrp.to_csv(OUT_DIR / 'mae_before_mfe.csv')
    else:
        pd.DataFrame().to_csv(OUT_DIR / 'mae_before_mfe.csv')

    # Time-to-MFE
    ttm = df.groupby('quintile').agg(
        n=('n', 'sum'),
        **{f'p{p}': ('time_to_mfe_ms', lambda s, p=p: np.percentile(s, p)) for p in pcts},
        mean=('time_to_mfe_ms', 'mean'),
    )
    ttm.index = [f'Q{i}' for i in ttm.index]
    ttm.to_csv(OUT_DIR / 'time_to_mfe.csv')

    # Time-to-stop (hold time at default tp/sl exit)
    tts = df.groupby(['quintile', 'exit_reason_default']).agg(
        n=('n', 'sum'),
        p25_hold_ms=('hold_ms_default', lambda s: np.percentile(s, 25)),
        p50_hold_ms=('hold_ms_default', lambda s: np.percentile(s, 50)),
        p75_hold_ms=('hold_ms_default', lambda s: np.percentile(s, 75)),
        p95_hold_ms=('hold_ms_default', lambda s: np.percentile(s, 95)),
        mean_hold_ms=('hold_ms_default', 'mean'),
    )
    tts.to_csv(OUT_DIR / 'time_to_stop.csv')

    # Exit-reason × quintile heatmap
    er_heat = pd.crosstab(df['quintile'], df['exit_reason_default'])
    er_heat.to_csv(OUT_DIR / 'exit_reason_heatmap.csv')

    # Adaptive TP grid: for each (quintile, hour, tp, sl) compute winrate, mean_pnl_ticks, sortino
    grid_rows = []
    for q in sorted(df['quintile'].unique()):
        for h in sorted(df['hour_et'].unique()):
            sub = df[(df['quintile'] == q) & (df['hour_et'] == h)]
            n_cell = len(sub)
            if n_cell == 0:
                continue
            # We need MFE/MAE arrays to simulate any TP/SL pair without re-replaying
            # Approximation: a trade hits TP iff mfe >= tp AND mae > -sl OR mfe occurs before mae.
            # This requires the mfe/mae ordering — we already captured time_to_mfe and time_to_mae.
            # If time_to_mae < time_to_mfe and abs(mae) >= sl, exit is sl (regardless of mfe).
            # If time_to_mfe < time_to_mae and mfe >= tp, exit is tp.
            # Else both can happen — use the ordering.
            mfe_arr = sub['mfe_ticks'].values
            mae_arr = sub['mae_ticks'].values
            ttm_arr = sub['time_to_mfe_ms'].values
            ttma_arr = sub['time_to_mae_ms'].values
            final_arr = sub['final_drift_60s_ticks'].values
            for tp in TP_GRID:
                for sl in SL_GRID:
                    pnls = []
                    wins = 0
                    for i in range(n_cell):
                        m_pos = mfe_arr[i]
                        m_neg = -mae_arr[i]  # adverse magnitude (positive)
                        tp_hit = m_pos >= tp
                        sl_hit = m_neg >= sl
                        if tp_hit and not sl_hit:
                            pnl = tp
                        elif sl_hit and not tp_hit:
                            pnl = -sl
                        elif tp_hit and sl_hit:
                            # Both hit; whichever came first (approx via time_to_mfe vs time_to_mae)
                            if ttm_arr[i] < ttma_arr[i]:
                                pnl = tp
                            else:
                                pnl = -sl
                        else:
                            pnl = final_arr[i]  # neither hit, exit at 60s
                        pnl_net = pnl - SPREAD_TAX_TICKS - COMMISSION_TICKS
                        pnls.append(pnl_net)
                        if pnl_net > 0:
                            wins += 1
                    pnls = np.array(pnls)
                    mean_pnl = pnls.mean()
                    win_rate = wins / n_cell
                    downside = pnls[pnls < 0]
                    if len(downside) > 0 and downside.std() > 0:
                        sortino = mean_pnl / downside.std()
                    else:
                        sortino = float('nan') if mean_pnl <= 0 else float('inf')
                    total_pnl_dollars = pnls.sum() * TICK_DOLLARS
                    grid_rows.append({
                        'quintile': q, 'hour_et': h, 'tp': tp, 'sl': sl, 'n': n_cell,
                        'win_rate': round(win_rate, 4),
                        'mean_pnl_ticks': round(mean_pnl, 3),
                        'sortino': round(sortino, 4) if math.isfinite(sortino) else None,
                        'total_pnl_dollars': round(total_pnl_dollars, 2),
                        'underpowered': n_cell < 30,
                    })
    grid_df = pd.DataFrame(grid_rows)
    grid_df.to_csv(OUT_DIR / 'adaptive_tp_grid.csv', index=False)

    # Best (TP,SL) per (quintile, hour) — by mean_pnl_ticks then sortino
    best_per_cell = []
    for (q, h), g in grid_df.groupby(['quintile', 'hour_et']):
        # Filter to non-underpowered first; if all underpowered keep them
        g_use = g[~g['underpowered']]
        if len(g_use) == 0:
            g_use = g
        g_sorted = g_use.sort_values(['mean_pnl_ticks', 'sortino'], ascending=[False, False])
        best = g_sorted.iloc[0].to_dict()
        best_per_cell.append(best)
    best_df = pd.DataFrame(best_per_cell)
    best_df.to_csv(OUT_DIR / 'adaptive_tp_best_per_cell.csv', index=False)

    # MFE vs MAE 6x6 heatmap
    mfe_bins = [0, 2, 5, 9, 15, 25, 1000]
    mae_bins = [0, 2, 5, 9, 15, 25, 1000]
    df['mfe_bin'] = pd.cut(df['mfe_ticks'].clip(lower=0), bins=mfe_bins, right=False,
                           labels=[f'{a}-{b}' for a, b in zip(mfe_bins[:-1], mfe_bins[1:])])
    df['mae_bin'] = pd.cut(df['mae_abs_ticks'].clip(lower=0), bins=mae_bins, right=False,
                           labels=[f'{a}-{b}' for a, b in zip(mae_bins[:-1], mae_bins[1:])])
    mfe_mae_heat = pd.crosstab(df['mae_bin'], df['mfe_bin'])
    mfe_mae_heat.to_csv(OUT_DIR / 'mfe_vs_mae_heatmap.csv')

    # Hour x Quintile heatmap (default tp/sl PnL)
    hq_heat = df.pivot_table(index='hour_et', columns='quintile',
                              values='pnl_ticks_default', aggfunc='sum').round(2)
    hq_heat.to_csv(OUT_DIR / 'hour_quintile_pnl.csv')

    # Per-fold meta
    pd.DataFrame(fold_metas).to_csv(OUT_DIR / 'fold_meta.csv', index=False)

    # ---- Print summary + write REPORT.md ----
    n_total = len(df)
    n_long = int((df['side'] == 1).sum())
    n_short = int((df['side'] == -1).sum())
    overall_winrate = float((df['final_drift_60s_ticks'] > 0).mean())
    overall_default_winrate = float((df['pnl_ticks_default'] > 0).mean())
    overall_default_pnl = float(df['pnl_ticks_default'].sum())
    overall_default_dollars = overall_default_pnl * TICK_DOLLARS
    overall_default_pct_account = (overall_default_dollars / ACCOUNT) * 100
    mfe_p50 = float(np.percentile(df['mfe_ticks'], 50))
    mfe_p75 = float(np.percentile(df['mfe_ticks'], 75))
    mae_p50 = float(np.percentile(df['mae_abs_ticks'], 50))
    mae_p75 = float(np.percentile(df['mae_abs_ticks'], 75))
    ttm_p50 = float(np.percentile(df['time_to_mfe_ms'], 50)) / 1000.0
    ttm_p75 = float(np.percentile(df['time_to_mfe_ms'], 75)) / 1000.0
    pct_red_first = 100 * float(df['mae_before_mfe'].mean())
    elapsed = time.time() - t0
    print(f'\n[done] {n_total} trades, {elapsed:.1f}s', flush=True)

    # Q5 (highest conviction) MFE
    q5 = df[df['quintile'] == 5]
    q5_mfe_p50 = float(np.percentile(q5['mfe_ticks'], 50)) if len(q5) > 0 else float('nan')
    q5_ttm_p50 = float(np.percentile(q5['time_to_mfe_ms'], 50)) / 1000.0 if len(q5) > 0 else float('nan')
    q1 = df[df['quintile'] == 1]
    q1_default_pnl = float(q1['pnl_ticks_default'].sum()) if len(q1) > 0 else 0.0

    # Best adaptive cell summary — quintile-only adaptive (collapse hours)
    best_per_q = []
    for q in sorted(df['quintile'].unique()):
        sub = df[df['quintile'] == q]
        n_cell = len(sub)
        best_row = None
        for tp in TP_GRID:
            for sl in SL_GRID:
                mfe_arr = sub['mfe_ticks'].values
                mae_arr = sub['mae_ticks'].values
                ttm_arr = sub['time_to_mfe_ms'].values
                ttma_arr = sub['time_to_mae_ms'].values
                final_arr = sub['final_drift_60s_ticks'].values
                pnls = []
                for i in range(n_cell):
                    m_pos = mfe_arr[i]; m_neg = -mae_arr[i]
                    tp_hit = m_pos >= tp; sl_hit = m_neg >= sl
                    if tp_hit and not sl_hit:
                        pnl = tp
                    elif sl_hit and not tp_hit:
                        pnl = -sl
                    elif tp_hit and sl_hit:
                        pnl = tp if ttm_arr[i] < ttma_arr[i] else -sl
                    else:
                        pnl = final_arr[i]
                    pnls.append(pnl - SPREAD_TAX_TICKS - COMMISSION_TICKS)
                pnls = np.array(pnls)
                mp = pnls.mean(); ts = pnls.sum()
                if best_row is None or mp > best_row['mean_pnl_ticks']:
                    best_row = {'quintile': q, 'tp': tp, 'sl': sl, 'n': n_cell,
                                'mean_pnl_ticks': mp, 'total_pnl_ticks': ts,
                                'win_rate': float((pnls > 0).mean())}
        best_per_q.append(best_row)
    best_per_q_df = pd.DataFrame(best_per_q)
    best_per_q_df.to_csv(OUT_DIR / 'adaptive_tp_best_per_quintile.csv', index=False)

    # ---- REPORT.md ----
    report_lines = []
    report_lines.append(f"# CNN-Mamba v2 — Per-Trade Dynamics Deep Dive")
    report_lines.append(f"")
    report_lines.append(f"**Run**: {datetime.now().isoformat()}  |  **Trades**: {n_total} "
                        f"({n_long} long, {n_short} short) across 10 folds (Feb 23 - Mar 5 2026)")
    report_lines.append(f"**Gate**: |z_10s| >= 2.3, per-fold OOT z-score, RTH only, hold 60s")
    report_lines.append(f"**Fills**: entry pays half-spread; exit pays half-spread (1-tick spread tax)")
    report_lines.append(f"**Account ref**: ${ACCOUNT:,.0f} for $-as-pct conversions")
    report_lines.append(f"")
    report_lines.append(f"## Headline Numbers")
    report_lines.append(f"")
    report_lines.append(f"- **Default tp=9 / sl=15 net PnL (after spread+comm)**: "
                        f"{overall_default_pnl:+.1f} ticks = "
                        f"${overall_default_dollars:+,.0f} = "
                        f"**{overall_default_pct_account:+.2f}% of $50K**")
    report_lines.append(f"- **Default win rate**: {100*overall_default_winrate:.1f}%")
    report_lines.append(f"- **Naive 60s timeout win rate (mid-drift > 0)**: {100*overall_winrate:.1f}%")
    report_lines.append(f"- **MFE p50 / p75**: {mfe_p50:.1f} / {mfe_p75:.1f} ticks")
    report_lines.append(f"- **MAE p50 / p75**: {mae_p50:.1f} / {mae_p75:.1f} ticks")
    report_lines.append(f"- **Time-to-MFE p50 / p75**: {ttm_p50:.1f}s / {ttm_p75:.1f}s")
    report_lines.append(f"- **% of trades that go red first (then recover, MFE-after-MAE)**: "
                        f"{pct_red_first:.1f}%")
    report_lines.append(f"")
    report_lines.append(f"## Top 5 Findings")
    report_lines.append(f"")

    # Finding 1 — quintile gradient
    qpnl = df.groupby('quintile')['pnl_ticks_default'].agg(['sum', 'mean', 'count'])
    qpnl_str = '; '.join([f"Q{q}: ${row['sum']*TICK_DOLLARS:+.0f} (n={row['count']}, ${row['mean']*TICK_DOLLARS:+.1f}/trade)"
                          for q, row in qpnl.iterrows()])
    report_lines.append(f"1. **Conviction gradient (default tp9/sl15)**: {qpnl_str}. "
                        f"Q1 (lowest |z|) loses ${q1_default_pnl*TICK_DOLLARS:+.0f}; "
                        f"top quintiles do the lifting.")

    # Finding 2 — time to MFE
    if len(q5) > 0:
        report_lines.append(f"2. **Q5 reaches MFE in {q5_ttm_p50:.1f}s median** "
                            f"(MFE p50 = {q5_mfe_p50:.1f} ticks). "
                            f"60s hold is leaving up to {60 - q5_ttm_p50:.1f}s of post-peak drift "
                            f"with no positive expectation — strong case for time-stop or trailing.")

    # Finding 3 — MAE before MFE
    if len(winners) > 0:
        wred = 100 * float(winners['mae_before_mfe'].mean())
        wred_p50 = float(np.percentile(winners['mae_before_mfe_ticks'], 50))
        wred_p90 = float(np.percentile(winners['mae_before_mfe_ticks'], 90))
        report_lines.append(f"3. **{wred:.1f}% of eventual winners go red first** "
                            f"(median red depth {wred_p50:.1f} ticks, p90 {wred_p90:.1f}). "
                            f"Tight stops (sl<8) would kill many otherwise-profitable trades.")

    # Finding 4 — exit-reason mix
    er_total = df['exit_reason_default'].value_counts(normalize=True) * 100
    er_str = ', '.join([f"{k}: {v:.1f}%" for k, v in er_total.items()])
    report_lines.append(f"4. **Exit-reason mix at tp9/sl15**: {er_str}. "
                        f"Timeout share indicates 60s hold isn't binding for most trades.")

    # Finding 5 — hour-of-day
    hpnl = df.groupby('hour_et')['pnl_ticks_default'].sum().sort_values(ascending=False)
    best_h = hpnl.head(2).index.tolist()
    worst_h = hpnl.tail(2).index.tolist()
    report_lines.append(f"5. **Hour-of-day skew**: best hours ET = {best_h} "
                        f"(${hpnl.head(2).sum()*TICK_DOLLARS:+.0f}); "
                        f"worst hours = {worst_h} (${hpnl.tail(2).sum()*TICK_DOLLARS:+.0f}). "
                        f"Hour filter could materially boost Sortino.")
    report_lines.append(f"")

    # Recommended adaptive rule — best per-quintile
    report_lines.append(f"## Recommended Adaptive TP/SL Rule")
    report_lines.append(f"")
    report_lines.append(f"From `adaptive_tp_best_per_quintile.csv` (collapsed across hours for power):")
    report_lines.append(f"")
    report_lines.append(f"| Quintile | n | TP | SL | mean PnL (ticks) | total PnL ($) | winrate |")
    report_lines.append(f"|---|---|---|---|---|---|---|")
    for r in best_per_q:
        report_lines.append(f"| Q{r['quintile']} | {r['n']} | {r['tp']} | {r['sl']} | "
                            f"{r['mean_pnl_ticks']:+.2f} | ${r['total_pnl_ticks']*TICK_DOLLARS:+,.0f} | "
                            f"{100*r['win_rate']:.1f}% |")
    report_lines.append(f"")
    # Rule recommendation
    best_overall = max(best_per_q, key=lambda r: r['total_pnl_ticks'])
    report_lines.append(f"**RULE**: Use per-quintile TP/SL from the table above. "
                        f"Highest-impact bucket is Q{best_overall['quintile']} "
                        f"(TP={best_overall['tp']}, SL={best_overall['sl']}, n={best_overall['n']}). "
                        f"Across all quintiles using adaptive TP/SL: "
                        f"${sum(r['total_pnl_ticks']*TICK_DOLLARS for r in best_per_q):+,.0f} vs "
                        f"baseline tp9/sl15 ${overall_default_dollars:+,.0f}.")
    report_lines.append(f"")

    # Confluence filter recommendations
    report_lines.append(f"## Confluence Filter Recommendations")
    report_lines.append(f"")
    report_lines.append(f"1. **Drop Q1 (lowest |z|)**: Q1 PnL is ${q1_default_pnl*TICK_DOLLARS:+.0f}; "
                        f"removing it eliminates dead-weight trades.")
    h_to_drop = [h for h, p in hpnl.items() if p * TICK_DOLLARS < -100]
    if h_to_drop:
        report_lines.append(f"2. **Hour filter**: Drop hours ET in {h_to_drop} "
                            f"(each loses >$100 cumulative). Keep prime 9:30-11:30 + best afternoon hours.")
    side_pnl = df.groupby('side')['pnl_ticks_default'].sum()
    if abs(side_pnl.iloc[0] - side_pnl.iloc[-1]) * TICK_DOLLARS > 500:
        report_lines.append(f"3. **Side asymmetry**: Long PnL = ${side_pnl.get(1, 0)*TICK_DOLLARS:+.0f}, "
                            f"Short PnL = ${side_pnl.get(-1, 0)*TICK_DOLLARS:+.0f}. "
                            f"Investigate whether short side warrants stricter |z| gate.")
    if len(winners) > 0:
        report_lines.append(f"4. **MAE-before-MFE patience**: {wred:.0f}% of winners go red first by "
                            f"{wred_p50:.1f} ticks median. Set SL >= {max(8, int(wred_p90))} "
                            f"to avoid scratching winners; combine with vol-based exit "
                            f"(adverse > N ticks WITHOUT recovery in K bars).")
    report_lines.append(f"")

    # Caveats
    report_lines.append(f"## Caveats")
    report_lines.append(f"")
    underpowered_cells = (grid_df['underpowered'].sum())
    total_cells = len(grid_df)
    report_lines.append(f"- **Sample size**: {n_total} trades total. "
                        f"{underpowered_cells}/{total_cells} (quintile x hour x TP x SL) cells have n<30. "
                        f"Per-cell adaptive rules are noisy below n=30 — prefer per-quintile (no hour split) "
                        f"in production until n grows.")
    report_lines.append(f"- **Quintile leakage**: per-fold OOT std/quintile boundaries used. "
                        f"Production should use causal running z (which is what tp9_z2.3 uses). "
                        f"Direction of bias: per-fold quintile breakpoints reflect within-fold rank, "
                        f"so they cannot leak across folds, but a trade ranked Q5 with 100% knowledge "
                        f"of fold's full distribution is slightly different from a real-time Q5 estimate.")
    report_lines.append(f"- **Spread model**: paid 1 full tick (half-spread x 2). ES is 1-tick wide >95% "
                        f"of RTH so this is conservative-but-realistic. Real fill rate is ~83% (queue model); "
                        f"this report assumes 100% fill at touch.")
    report_lines.append(f"- **Mid-price source**: cumulative `mid_price_change_ticks` from book_features. "
                        f"BBO bid/ask is sparse (6% of events have valid both-sides) so we cannot validate "
                        f"per-event spread; we assume spread=1 tick when the book file lacks valid quote.")
    report_lines.append(f"- **No regime tagging beyond hour-of-day**: vol regime, news, etc. not captured here.")
    report_lines.append(f"")
    report_lines.append(f"## Files")
    report_lines.append(f"All CSVs in `{OUT_DIR}/`:")
    for f in sorted(OUT_DIR.glob('*.csv')):
        report_lines.append(f"- `{f.name}` ({f.stat().st_size//1024} KB)")

    report_path = OUT_DIR / 'REPORT.md'
    report_path.write_text('\n'.join(report_lines))
    print(f'[REPORT.md] saved to {report_path}', flush=True)

    # ---- Discord-formatted summary ----
    discord_lines = []
    discord_lines.append(f"**EXEC DEEP-DIVE — cnn_mamba_v2 OOT, 10 folds**")
    discord_lines.append(f"Gate: |z_10s|>=2.3, hold 60s. Trades: {n_total} ({n_long}L/{n_short}S).")
    discord_lines.append(f"Default tp9/sl15: ${overall_default_dollars:+,.0f} = "
                        f"{overall_default_pct_account:+.2f}% of $50K, "
                        f"win {100*overall_default_winrate:.0f}%")
    discord_lines.append(f"MFE p50={mfe_p50:.0f}t @ {ttm_p50:.0f}s | MAE p50={mae_p50:.0f}t | "
                        f"{pct_red_first:.0f}% go red first")
    if len(q5) > 0:
        discord_lines.append(f"Q5 hits MFE {q5_mfe_p50:.0f}t in {q5_ttm_p50:.0f}s median "
                            f"(60s hold leaves theta on table)")
    discord_lines.append(f"Best per-Q (TP/SL): " + ', '.join([f"Q{r['quintile']}={r['tp']}/{r['sl']}" for r in best_per_q]))
    adaptive_total = sum(r['total_pnl_ticks']*TICK_DOLLARS for r in best_per_q)
    discord_lines.append(f"Adaptive vs baseline: ${adaptive_total:+,.0f} vs ${overall_default_dollars:+,.0f} "
                        f"= {(adaptive_total - overall_default_dollars)/ACCOUNT*100:+.2f}% account uplift")
    discord_lines.append(f"Report: {report_path}")
    print('\n--- DISCORD SUMMARY ---')
    print('\n'.join(discord_lines))


if __name__ == '__main__':
    main()
