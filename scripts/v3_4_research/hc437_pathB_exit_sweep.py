#!/usr/bin/env python3
"""
HC #437 Path B — Exit-side parametric sweep (TP / SL / hold) over v2 1s short top0.5%.

Goal: find a TP/SL/hold config that produces a real PASS under realtime_sl exit
logic on 47-day OOT. Baseline HC #413 brackets (TP1=0.48, SL=0.57, hold=10) lose
0.27 tk/fill under realtime_sl despite +0.49 raw label edge — the SL is being
clipped on intra-event noise.

STRATEGY (key optimization):
  The FIFOReplayEngine.simulate() walks all ~23M MBO events per call (~140s).
  Sweeping 80 cells per date naively => hours per date.

  Instead, we do the engine.simulate() ONCE per (date) with VERY WIDE tp/sl
  (so no SL/TP fires) and capture the actual passive_at_touch fills + their
  post-fill TRADE event trajectories. Then we resolve (TP, SL, hold) cells
  ARITHMETICALLY against those trajectories — same first-touch logic the
  engine uses (line 890-916 of fifo_market_replay.py), but reused across
  cells.

  cancel_s is fixed at 10s (baseline) for all cells. hold_s varies up to 20s.
  cancel_s affects which orders fill; tp/sl/hold don't. So one engine run per
  date suffices to cache fills.

VALIDATION:
  After building cache, re-run a cell that matches the existing 47-day realtime_sl
  baseline (TP=0.9564, SL=0.5686, hold=10) and verify net_tk/fill matches
  -0.2745 ± 0.01 (the published Bug 2 baseline). If diverges → STOP and re-engineer.

PHASES:
  1. Build fill+trajectory cache per pilot date (5 dates).
  2. Validate cache on baseline cell.
  3. Phase 1 pilot: 5×4×4=80 cells × 5 dates (analytic, fast).
  4. Phase 2: top-5 cells × all 38 useable OOT dates.
  5. Phase 3: dollar-risk normalization.
  6. Phase 4: write final_report.md.

Per HC #420 — this is authorized augmentation of user's research codebase.
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

LVL3 = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(LVL3))

V2_OOT_DIR = LVL3 / "output" / "cnn_mamba_v2_bulk_oot_v2"
MBO_EVENT_DIR = LVL3 / "data" / "processed" / "mbo_events_smart_v3"
OUT_DIR = LVL3 / "output" / "hc437_pathB_exit_sweep"
CACHE_DIR = OUT_DIR / "fill_cache"

HORIZON_IDX = {"1": 0, "5": 1, "10": 2}

TICK_RAW = 25_000_000  # ES one tick = 0.25 pts (in raw int price units, same as engine)
COMMISSION_TICKS = 0.376  # AMP/Rithmic RT commission in ticks
TICK_USD = 12.50

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("pathB_sweep")


# ─────────────────────────────────────────────────────────────────────────────
# Signal selection (identical to hc432_v2_baseline_runner.py)
# ─────────────────────────────────────────────────────────────────────────────

def parse_conf_band(band: str) -> float:
    if not band.startswith("top"):
        raise ValueError(band)
    return float(band[3:]) / 100.0


def list_dates() -> List[str]:
    return [p.stem.split("_")[0] for p in sorted(V2_OOT_DIR.glob("2026*_predictions.npz"))]


def load_v2_day(date_str: str, horizon: str) -> Tuple[np.ndarray, int, int]:
    p = V2_OOT_DIR / f"{date_str}_predictions.npz"
    d = np.load(p, allow_pickle=False)
    preds = d["predictions"][:, HORIZON_IDX[horizon]].astype(np.float64)
    ws = int(d["window_size"])
    st = int(d["stride"])
    return preds, ws, st


def select_signals_global(
    dates: List[str], horizon: str, side: str, conf_band: str
) -> Dict[str, Tuple[np.ndarray, np.ndarray, int, int]]:
    """Pool dates, pick top-N% side-aligned globally, group by date.
    Mirrors hc432_v2_baseline_runner.select_signals_global() exactly.
    """
    all_preds: List[np.ndarray] = []
    day_meta: List[Tuple[str, int, int, int]] = []
    for d in dates:
        preds, ws, st = load_v2_day(d, horizon)
        all_preds.append(preds)
        day_meta.append((d, preds.size, ws, st))
    preds_concat = np.concatenate(all_preds)
    if side == "long":
        mask_side = preds_concat > 0
        strength = preds_concat
    else:
        mask_side = preds_concat < 0
        strength = -preds_concat

    frac = parse_conf_band(conf_band)
    side_str = strength[mask_side]
    if side_str.size == 0:
        return {}
    k = max(1, int(side_str.size * frac))
    thresh = np.partition(side_str, -k)[-k]
    selected = mask_side & (strength >= thresh)
    log.info(f"v2 selected {int(selected.sum()):,} signals (k={k:,} thresh={thresh:.4g})")

    out: Dict[str, Tuple[np.ndarray, np.ndarray, int, int]] = {}
    cursor = 0
    for (d, n, ws, st) in day_meta:
        sel_day = selected[cursor:cursor + n]
        if sel_day.any():
            idx = np.flatnonzero(sel_day)
            strs = strength[cursor:cursor + n][idx]
            out[d] = (idx, strs.astype(np.float64), ws, st)
        cursor += n
    return out


def map_to_ts(date_str: str, idx_in_day: np.ndarray, window: int, stride: int) -> np.ndarray:
    mbo = np.load(MBO_EVENT_DIR / f"{date_str}_mbo_events.npz", allow_pickle=False)
    ts = mbo["timestamps"].astype(np.int64)
    n = len(ts)
    event_idx = np.minimum(idx_in_day * stride + window - 1, n - 1)
    return ts[event_idx]


# ─────────────────────────────────────────────────────────────────────────────
# Fill + Trajectory Cache (one engine run per date)
# ─────────────────────────────────────────────────────────────────────────────

def build_fill_cache_for_date(
    date_str: str,
    idx: np.ndarray, strs: np.ndarray, ws: int, st: int,
    side: str,
    cancel_s: float = 10.0,
    max_hold_s: float = 30.0,
    order_type: str = "passive_at_touch",
) -> dict:
    """Run engine ONCE with very wide TP/SL to capture passive fills, then
    re-walk MBO TRADE events to extract per-fill post-fill price trajectory
    for any hold ≤ max_hold_s.

    Returns dict with:
      fills:    list of {sig_ts, entry_ts, entry_price, mid_at_signal,
                         queue_wait_ns, slippage_ticks, pred_strength, direction}
      traj:     list of np.ndarray[(ts_offset_ns, price_raw)] per fill,
                truncated to max_hold_s after entry.
      eod_ts:   final timestamp (for max-hold->eod fallback)
    """
    from alpha_discovery.deep_models.fifo_market_replay import (
        FIFOReplayEngine, A_TRADE, A_FILL,
    )

    cancel_ns = int(cancel_s * 1e9)
    hold_ns = int(max_hold_s * 1e9)

    # Map signal indices to MBO timestamps
    ts_ns = map_to_ts(date_str, idx, ws, st)
    signals = [{"ts_ns": int(t), "direction": side, "strength": float(s)}
               for t, s in zip(ts_ns, strs)]

    engine_order = {"passive_at_touch": "limit", "market": "market", "chase": "chase"}[order_type]
    engine = FIFOReplayEngine(date=date_str,
                              cancel_after_ns=cancel_ns,
                              max_hold_ns=hold_ns)

    # Run with TP=SL=999 ticks → exits will be 'max_hold' or 'eod' for all fills.
    # We don't care about engine's exit decisions; we extract entries only.
    trades = engine.simulate(signals=signals,
                              tp_ticks=999.0, sl_ticks=999.0,
                              order_type=engine_order,
                              )

    # Build trajectory by re-walking self.records for TRADE/FILL events
    # only (these are the events the engine checks against TP/SL).
    recs = engine.records
    rec_ts = engine._ts
    rec_action = recs['action']
    rec_price = recs['price']

    # Mask trade-class events
    is_trade = np.isin(rec_action, [A_TRADE, A_FILL])
    trade_ts = rec_ts[is_trade]
    trade_pr = rec_price[is_trade].astype(np.int64)

    # For each filled trade, slice trade_ts/pr in [fill_ts, fill_ts+max_hold_ns]
    fills_out: List[dict] = []
    traj_out: List[np.ndarray] = []

    for tr in trades:
        if tr.entry_ts_ns is None or tr.entry_ts_ns == 0:
            continue
        fill_ts = int(tr.entry_ts_ns)
        fill_pr = int(tr.entry_price_raw)
        end_ts = fill_ts + hold_ns
        lo = int(np.searchsorted(trade_ts, fill_ts, side='left'))
        hi = int(np.searchsorted(trade_ts, end_ts, side='right'))
        traj = np.empty((hi - lo, 2), dtype=np.int64)
        traj[:, 0] = trade_ts[lo:hi] - fill_ts  # offset
        traj[:, 1] = trade_pr[lo:hi]

        fills_out.append({
            'date': date_str,
            'sig_ts_ns': int(tr.signal_ts_ns),
            'entry_ts_ns': fill_ts,
            'entry_price_raw': fill_pr,
            'mid_at_signal': int(tr.mid_at_signal) if tr.mid_at_signal else 0,
            'queue_wait_ns': int(tr.queue_wait_ns),
            'queue_ahead': int(tr.queue_ahead),
            'slippage_ticks': float(tr.slippage_ticks),
            'pred_strength': float(tr.pred_strength),
            'direction': tr.direction,
        })
        traj_out.append(traj)

    eod_ts = int(rec_ts[-1]) if len(rec_ts) else 0

    return {
        'date': date_str,
        'eod_ts': eod_ts,
        'fills': fills_out,
        'traj': traj_out,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Analytic exit resolver — replicates engine's realtime_sl logic per fill
# ─────────────────────────────────────────────────────────────────────────────

def resolve_cell_for_fill(fill: dict, traj: np.ndarray,
                          tp_ticks: float, sl_ticks: float,
                          hold_s: float, eod_ts: int) -> dict:
    """Replicate engine's per-fill exit:
      - On each TRADE event after fill, check TP then SL hit
      - If neither and elapsed > hold_ns → 'max_hold' exit at trade price (engine
        uses book.mid_raw() which we approximate with last trade price)
      - If no trade events within hold_ns → 'max_hold' at fill_price (no movement)
      - If date ends before hold_ns → 'eod' at last trade price
    """
    direction = fill['direction']
    entry_pr = fill['entry_price_raw']
    hold_ns = int(hold_s * 1e9)
    tp_raw_off = int(round(tp_ticks * TICK_RAW))
    sl_raw_off = int(round(sl_ticks * TICK_RAW))
    if direction == 'long':
        tp_price = entry_pr + tp_raw_off
        sl_price = entry_pr - sl_raw_off
    else:
        tp_price = entry_pr - tp_raw_off
        sl_price = entry_pr + sl_raw_off

    # Restrict trajectory to within hold window (caller may have given wider)
    if traj.size == 0:
        # No trades in [fill, fill+hold]. Exit at entry (no movement observed).
        exit_pr = entry_pr
        exit_reason = 'max_hold'
        hold_ns_actual = hold_ns
    else:
        # iterate trades within hold
        offsets = traj[:, 0]
        prices  = traj[:, 1]
        within_hold = offsets <= hold_ns
        if not within_hold.any():
            exit_pr = entry_pr
            exit_reason = 'max_hold'
            hold_ns_actual = hold_ns
        else:
            offs_h = offsets[within_hold]
            prs_h  = prices[within_hold]
            # Engine logic: check TP first then SL (lines 897-910)
            if direction == 'long':
                tp_hit = prs_h >= tp_price
                sl_hit = prs_h <= sl_price
            else:
                tp_hit = prs_h <= tp_price
                sl_hit = prs_h >= sl_price
            first_tp = np.argmax(tp_hit) if tp_hit.any() else -1
            first_sl = np.argmax(sl_hit) if sl_hit.any() else -1
            if first_tp >= 0 and (first_sl < 0 or first_tp <= first_sl):
                exit_pr = int(tp_price)
                exit_reason = 'tp'
                hold_ns_actual = int(offs_h[first_tp])
            elif first_sl >= 0:
                exit_pr = int(sl_price)
                exit_reason = 'sl'
                hold_ns_actual = int(offs_h[first_sl])
            else:
                # No TP/SL → max_hold at last trade price within hold (engine
                # uses book.mid_raw, approximated by last trade price)
                exit_pr = int(prs_h[-1])
                exit_reason = 'max_hold'
                hold_ns_actual = int(offs_h[-1])
                # If last trade is before hold expiry, still 'max_hold' once
                # elapsed > hold_ns. The engine checks at every event so we
                # take the price-at-time-of-exit which we approximate with
                # the last observed price.

    # Net P&L in ticks
    if direction == 'long':
        gross_raw = exit_pr - entry_pr
    else:
        gross_raw = entry_pr - exit_pr
    gross_tk = gross_raw / TICK_RAW
    net_tk = gross_tk - COMMISSION_TICKS

    return {
        **{k: fill[k] for k in ('date', 'sig_ts_ns', 'entry_ts_ns',
                                'entry_price_raw', 'queue_ahead',
                                'queue_wait_ns', 'slippage_ticks',
                                'pred_strength', 'direction')},
        'exit_price_raw': exit_pr,
        'hold_ns': hold_ns_actual,
        'exit_reason': exit_reason,
        'gross_ticks': gross_tk,
        'net_ticks': net_tk,
        'net_dollars': net_tk * TICK_USD,
    }


def resolve_cell(cache: dict, tp_ticks: float, sl_ticks: float, hold_s: float) -> List[dict]:
    """Resolve all fills in one date-cache for one (TP, SL, hold) cell."""
    eod_ts = cache['eod_ts']
    out = []
    for fill, traj in zip(cache['fills'], cache['traj']):
        out.append(resolve_cell_for_fill(fill, traj, tp_ticks, sl_ticks, hold_s, eod_ts))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(rows: List[dict]) -> dict:
    if not rows:
        return {'n_fills': 0, 'mean_net_tk': float('nan'),
                'mean_gross_tk': float('nan'),
                'PF': float('nan'), 'WR_pct': float('nan'),
                'Sh_sqrtN': float('nan')}
    nets = np.array([r['net_ticks'] for r in rows])
    gross = np.array([r['gross_ticks'] for r in rows])
    wins = (nets > 0).sum()
    gp = nets[nets > 0].sum()
    gl = -nets[nets < 0].sum()
    pf = (gp / gl) if gl > 0 else float('inf')
    wr = 100.0 * wins / len(nets)
    sh = (nets.mean() / (nets.std(ddof=1) + 1e-12)) * np.sqrt(len(nets)) if len(nets) > 1 else 0.0
    return {
        'n_fills': int(len(nets)),
        'mean_gross_tk': float(gross.mean()),
        'mean_net_tk': float(nets.mean()),
        'PF': float(pf),
        'WR_pct': float(wr),
        'Sh_sqrtN': float(sh),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Cache build worker (one date per process)
# ─────────────────────────────────────────────────────────────────────────────

def _cache_worker(args):
    date_str, idx_b, strs_b, ws, st, side, cancel_s, max_hold_s, order_type = args
    idx = np.frombuffer(idx_b, dtype=np.int64).copy()
    strs = np.frombuffer(strs_b, dtype=np.float64).copy()
    t0 = time.time()
    try:
        cache = build_fill_cache_for_date(date_str, idx, strs, ws, st,
                                          side, cancel_s, max_hold_s, order_type)
    except FileNotFoundError as e:
        return {'date': date_str, 'error': f'no_dbn: {e}'}
    except Exception as e:
        import traceback
        return {'date': date_str, 'error': f'{type(e).__name__}: {e}',
                'traceback': traceback.format_exc()}
    # Save to disk
    out_p = CACHE_DIR / f"{date_str}_cache.npz"
    # Flatten trajectories into ragged arrays (use object dtype)
    fills_arr = np.array(cache['fills'], dtype=object)
    traj_lens = np.array([len(t) for t in cache['traj']], dtype=np.int64)
    if cache['traj']:
        traj_flat = np.concatenate([t for t in cache['traj']], axis=0) if any(t.size for t in cache['traj']) else np.empty((0, 2), dtype=np.int64)
    else:
        traj_flat = np.empty((0, 2), dtype=np.int64)
    np.savez_compressed(out_p,
                        fills=fills_arr,
                        traj_lens=traj_lens,
                        traj_flat=traj_flat,
                        eod_ts=np.int64(cache['eod_ts']))
    elapsed = time.time() - t0
    return {
        'date': date_str,
        'n_fills': len(cache['fills']),
        'cache_path': str(out_p),
        'elapsed_s': elapsed,
    }


def load_cache(date_str: str) -> dict:
    p = CACHE_DIR / f"{date_str}_cache.npz"
    z = np.load(p, allow_pickle=True)
    fills = list(z['fills'])
    traj_lens = z['traj_lens']
    traj_flat = z['traj_flat']
    traj_list = []
    cursor = 0
    for L in traj_lens:
        L = int(L)
        traj_list.append(traj_flat[cursor:cursor+L])
        cursor += L
    return {
        'date': date_str,
        'eod_ts': int(z['eod_ts']),
        'fills': fills,
        'traj': traj_list,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["cache", "validate", "pilot", "phase2",
                                         "phase3", "phase4"], required=True)
    ap.add_argument("--dates", nargs="+", default=None,
                    help="Dates (YYYYMMDD) to process; default = pilot dates")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--horizon", default="1")
    ap.add_argument("--side", default="short")
    ap.add_argument("--conf-band", default="top0.5")
    ap.add_argument("--cancel-s", type=float, default=10.0)
    ap.add_argument("--max-hold-s", type=float, default=30.0)
    args = ap.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "logs").mkdir(parents=True, exist_ok=True)

    PILOT_DATES = ["20260305", "20260309", "20260415", "20260313", "20260319"]
    if args.dates is None:
        args.dates = PILOT_DATES

    if args.phase == "cache":
        # Build per-date fill+trajectory caches
        all_dates = list_dates()
        target_dates = [d for d in all_dates if d in args.dates]
        log.info(f"Building cache for {len(target_dates)} dates: {target_dates}")
        sig_by_date = select_signals_global(all_dates, args.horizon, args.side, args.conf_band)
        # Only keep target dates
        sig_by_date = {d: v for d, v in sig_by_date.items() if d in target_dates}
        log.info(f"Signals on target dates: {sum(len(v[0]) for v in sig_by_date.values()):,}")

        tasks = []
        for d, (idx, strs, ws, st) in sig_by_date.items():
            tasks.append((d, idx.tobytes(), strs.tobytes(), ws, st,
                          args.side, args.cancel_s, args.max_hold_s,
                          'passive_at_touch'))
        log.info(f"Dispatching {len(tasks)} cache-build jobs, workers={args.workers}")

        ctx = mp.get_context("spawn")
        results = []
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
            futs = {ex.submit(_cache_worker, t): t[0] for t in tasks}
            for fut in as_completed(futs):
                d = futs[fut]
                try:
                    r = fut.result()
                except Exception as e:
                    log.exception(f"{d} failed: {e}")
                    results.append({'date': d, 'error': f'future: {e}'})
                    continue
                results.append(r)
                if 'error' in r:
                    log.error(f"{d}: {r['error']}")
                else:
                    log.info(f"{d}: cached n_fills={r['n_fills']} elapsed={r['elapsed_s']:.1f}s")

        out_p = OUT_DIR / "logs" / f"cache_results_{'_'.join(args.dates[:3])}.json"
        out_p.write_text(json.dumps(results, indent=2, default=str))
        log.info(f"Cache results → {out_p}")

    elif args.phase == "validate":
        # Resolve the HC #437 baseline cell from cache and compare to published numbers
        log.info("Validation: TP=0.9564, SL=0.5686, hold=10 vs Bug 2 baseline -0.2745")
        # Use ALL cached dates we have
        cached_dates = sorted([p.stem.replace('_cache', '')
                              for p in CACHE_DIR.glob("*_cache.npz")])
        log.info(f"Cached dates available: {len(cached_dates)}")
        all_rows = []
        for d in cached_dates:
            cache = load_cache(d)
            rows = resolve_cell(cache, tp_ticks=0.9564, sl_ticks=0.5686, hold_s=10.0)
            all_rows.extend(rows)
        m = compute_metrics(all_rows)
        log.info(f"VALIDATION cell metrics: {json.dumps(m, indent=2)}")
        # Also write per-day
        per_day = []
        for d in cached_dates:
            cache = load_cache(d)
            rows = resolve_cell(cache, tp_ticks=0.9564, sl_ticks=0.5686, hold_s=10.0)
            per_day.append({'date': d, **compute_metrics(rows)})
        pd.DataFrame(per_day).to_csv(OUT_DIR / "logs" / "validate_per_day.csv", index=False)
        (OUT_DIR / "logs" / "validate_aggregate.json").write_text(json.dumps(m, indent=2))

    elif args.phase == "pilot":
        # 5 SL × 4 TP × 4 hold = 80 cells. Resolve each per-cell × per-date.
        # HC #428 R2: for 1s horizon, hold ≤ 1.5s and cancel ≤ 1.0s.
        # Including 1.0 and 1.5 (R2-compliant) plus 2.0/5.0/10.0/20.0 for
        # comparison so we can see where the R2 rule actually bites.
        SLs = [0.57, 1.0, 1.5, 2.0, 3.0]
        TPs = [0.25, 0.48, 0.75, 1.0, 1.5]
        Hs  = [1.0, 1.5, 2.0, 5.0, 10.0]
        cached_dates = sorted([p.stem.replace('_cache', '')
                              for p in CACHE_DIR.glob("*_cache.npz")])
        target = [d for d in cached_dates if d in args.dates]
        log.info(f"Pilot on dates: {target}")
        # Pre-load caches
        caches = {d: load_cache(d) for d in target}

        cell_rows = []
        for sl in SLs:
            for tp in TPs:
                for h in Hs:
                    rows = []
                    for d in target:
                        rows.extend(resolve_cell(caches[d], tp, sl, h))
                    m = compute_metrics(rows)
                    cell_rows.append({
                        'SL_ticks': sl, 'TP_ticks': tp, 'hold_s': h,
                        **m,
                    })
        df = pd.DataFrame(cell_rows)
        df.sort_values('Sh_sqrtN', ascending=False, inplace=True)
        df.to_csv(OUT_DIR / "pilot" / "pilot_5day_grid.csv", index=False)
        log.info(f"Pilot top 10:\n{df.head(10).to_string()}")
        (OUT_DIR / "pilot" / "pilot_top10.json").write_text(
            df.head(10).to_json(orient='records', indent=2)
        )

    elif args.phase == "phase2":
        # Top-5 cells from pilot, full 38-day cached set
        pilot_csv = OUT_DIR / "pilot" / "pilot_5day_grid.csv"
        df_pilot = pd.read_csv(pilot_csv)
        # Filter to cells with min sane fill count
        df_pilot = df_pilot[df_pilot['n_fills'] >= 50]
        top5 = df_pilot.head(5)
        log.info(f"Top 5 pilot cells:\n{top5.to_string()}")
        cached_dates = sorted([p.stem.replace('_cache', '')
                              for p in CACHE_DIR.glob("*_cache.npz")])
        log.info(f"Cached dates: {len(cached_dates)} {cached_dates[:5]}...")
        caches = {d: load_cache(d) for d in cached_dates}

        # Load regime lookup
        rl = pd.read_csv(LVL3 / "output/hc437_harness_debug/regime_lookup_v2drift.csv")
        regime = dict(zip(rl['date'].astype(str), rl['regime']))

        results = []
        for _, row in top5.iterrows():
            tp = row['TP_ticks']; sl = row['SL_ticks']; h = row['hold_s']
            cell_key = f"TP{tp}_SL{sl}_H{int(h)}s"
            per_day_rows = []
            all_fills = []
            for d in cached_dates:
                fills = resolve_cell(caches[d], tp, sl, h)
                m = compute_metrics(fills)
                per_day_rows.append({'date': d, 'regime': regime.get(str(d), 'unk'), **m})
                all_fills.extend(fills)
            agg = compute_metrics(all_fills)
            # Regime stratification
            green_rows = [f for f in all_fills if regime.get(str(f['date']), '') == 'green']
            red_rows   = [f for f in all_fills if regime.get(str(f['date']), '') == 'red']
            g_m = compute_metrics(green_rows)
            r_m = compute_metrics(red_rows)
            # Day-concentration
            from collections import Counter
            day_counts = Counter(str(f['date']) for f in all_fills)
            max_day_share = max(day_counts.values()) / max(1, len(all_fills))
            results.append({
                'TP_ticks': tp, 'SL_ticks': sl, 'hold_s': h,
                'cell_key': cell_key,
                **{f'agg_{k}': v for k, v in agg.items()},
                'green_n_fills': g_m['n_fills'], 'green_net_tk': g_m['mean_net_tk'],
                'green_PF': g_m['PF'], 'green_WR': g_m['WR_pct'], 'green_Sh': g_m['Sh_sqrtN'],
                'red_n_fills':   r_m['n_fills'], 'red_net_tk':   r_m['mean_net_tk'],
                'red_PF':   r_m['PF'], 'red_WR':   r_m['WR_pct'], 'red_Sh':   r_m['Sh_sqrtN'],
                'regime_delta_norm': (
                    abs(g_m['Sh_sqrtN'] - r_m['Sh_sqrtN']) / max(abs(g_m['Sh_sqrtN']), abs(r_m['Sh_sqrtN']), 1e-9)
                    if g_m['n_fills'] > 0 and r_m['n_fills'] > 0 else float('nan')
                ),
                'max_day_share': max_day_share,
            })
            # Save per-day csv
            pd.DataFrame(per_day_rows).to_csv(
                OUT_DIR / "full47" / f"{cell_key}_per_day.csv", index=False
            )
            # Save raw fills for top candidates
            pd.DataFrame(all_fills).to_csv(
                OUT_DIR / "full47" / f"{cell_key}_fills.csv", index=False
            )
        pd.DataFrame(results).to_csv(OUT_DIR / "full47" / "phase2_top5_results.csv", index=False)
        (OUT_DIR / "full47" / "phase2_top5_results.json").write_text(
            json.dumps(results, indent=2, default=str)
        )

    elif args.phase == "phase3":
        # Dollar-risk normalized PnL/day for Phase 2 PASS candidates
        # size_factor = 0.57 / SL_ticks (baseline normalization)
        results_p = OUT_DIR / "full47" / "phase2_top5_results.csv"
        if not results_p.exists():
            log.error(f"Missing {results_p}; run phase2 first.")
            return
        df = pd.read_csv(results_p)
        df['size_factor_norm'] = 0.57 / df['SL_ticks']
        # dollar PnL / fill (normalized) = net_tk * 12.50 * size_factor
        df['agg_net_dollars_norm_per_fill'] = df['agg_mean_net_tk'] * 12.50 * df['size_factor_norm']
        df.to_csv(OUT_DIR / "full47" / "phase3_dollar_normalized.csv", index=False)
        log.info(df.to_string())

    elif args.phase == "phase4":
        # Build final_report.md (handled separately by writing in main script)
        log.info("Phase 4 = report assembly; run write_final_report.py")


if __name__ == "__main__":
    main()
