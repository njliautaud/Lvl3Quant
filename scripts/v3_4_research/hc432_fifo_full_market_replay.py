#!/usr/bin/env python3
"""
HC #432 / HC #74 — Full-FIFO market replay for v3.4.2 47-day OOT concat.

Loads the concatenated fold_00 NPZ, generates signals for a target config
(side / horizon / confidence band), maps each prediction to its nanosecond
timestamp via the MBO event file for that date, and runs
alpha_discovery.deep_models.fifo_market_replay.FIFOReplayEngine.simulate()
day-by-day for full FIFO realism (NEVER midpoint exits — HC #74).

Cost model (HC #52 / CLAUDE.md):
  tick           = $12.50
  RT commission  = $4.70  (0.376 ticks)
  passive limit  = entry at touch (best bid for long / best ask for short)
                   no crossing cost. realized exits via TP/SL/cancel/hold.
  market order   = +1.0 tick crossing cost on top of commission.

CLI:
  --horizon {1,5,10,30}      Which pred head (default 1s)
  --side {long,short}        Direction filter
  --conf-band top0.5         Top-percent confidence band, e.g. top0.5, top1, top5
  --tp-ticks                 Take-profit in ticks
  --sl-ticks                 Stop-loss in ticks
  --hold-s                   Max hold seconds after fill
  --cancel-s                 Cancel pending limit after N seconds
  --order-type {passive_at_touch, market}
  --workers                  CPU workers for per-day parallelism (default 12)
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

LVL3_ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(LVL3_ROOT))

OUT_DIR = LVL3_ROOT / "output" / "hc432_v342_47day_validation"
CONCAT_NPZ = OUT_DIR / "fold_00_ep1_oot_inference_47day_hc432.npz"
MBO_EVENT_DIR = LVL3_ROOT / "data" / "processed" / "mbo_events_smart_v3"

# v3.4.2 inference settings (from v342_run_oot_inference.py)
V342_STRIDE = 250
V342_WINDOW = 1500  # primary t1 window

# horizon name → pred key in NPZ
HORIZON_PRED_KEY = {
    "1": "pred_log_ret_1s",
    "5": "pred_log_ret_5s",
    "10": "pred_log_ret_10s",
    "30": "pred_log_ret_30s",
    "60": "pred_log_ret_60s",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("hc432_fifo")


def parse_conf_band(band: str) -> float:
    """
    'top0.5' → 0.005, 'top1' → 0.01, 'top5' → 0.05.
    Returns the fraction (e.g. 0.005 means top 0.5%).
    """
    if not band.startswith("top"):
        raise ValueError(f"unrecognized conf-band: {band}")
    pct_str = band[3:]
    return float(pct_str) / 100.0


def load_concat() -> Dict[str, np.ndarray]:
    if not CONCAT_NPZ.exists():
        log.error(f"concat NPZ not found: {CONCAT_NPZ}")
        sys.exit(2)
    d = np.load(CONCAT_NPZ, allow_pickle=False)
    return {k: d[k] for k in d.files}


def per_date_signals(
    arrs: Dict[str, np.ndarray],
    horizon: str,
    side: str,
    conf_band: str,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Build per-date signals.

    Returns: { date_str: (sample_idx_in_day, direction_array, strength_array) }
    where sample_idx_in_day is the index within that date's MBO-aligned stream.

    Confidence band is applied GLOBALLY (per-day rank would be inconsistent
    with the global confidence band semantics in HC #429).
    """
    pred_key = HORIZON_PRED_KEY[horizon]
    if pred_key not in arrs:
        raise KeyError(f"NPZ missing {pred_key}")

    preds = arrs[pred_key]
    sample_dates = arrs["sample_dates"]
    n_total = preds.shape[0]
    log.info(f"  total samples in concat: {n_total:,}")

    # signed signal for direction selection
    if side == "long":
        signed = preds  # take only positives
        mask_side = preds > 0
        strength = preds
    elif side == "short":
        signed = -preds
        mask_side = preds < 0
        strength = -preds  # positive strength
    else:
        raise ValueError(f"side {side}")

    # confidence band: top-N% by strength among side-aligned predictions
    frac = parse_conf_band(conf_band)
    side_strengths = strength[mask_side]
    if side_strengths.size == 0:
        raise RuntimeError("no signals on this side")
    k = max(1, int(side_strengths.size * frac))
    # threshold = the k-th largest strength among side-aligned preds
    thresh = np.partition(side_strengths, -k)[-k]
    selected = mask_side & (strength >= thresh)
    log.info(f"  selected {int(selected.sum()):,} signals "
             f"(side={side}, band={conf_band}, thresh={thresh:.6g}, k={k:,})")

    direction_global = np.where(selected, side, "")
    # group by date
    out: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    unique_dates = np.unique(sample_dates)
    for d in unique_dates:
        in_day = np.flatnonzero(sample_dates == d)
        sel_in_day = selected[in_day]
        if not sel_in_day.any():
            continue
        idx_in_day = np.where(sel_in_day)[0]  # 0..n_day-1
        out[str(d)] = (idx_in_day,
                       np.array([side] * idx_in_day.size, dtype=object),
                       strength[in_day][idx_in_day].astype(np.float64))
    log.info(f"  signal days: {len(out)} / {len(unique_dates)}")
    return out


def map_signals_to_timestamps(date_str: str, idx_in_day: np.ndarray) -> np.ndarray:
    """
    Map per-day sample indices to nanosecond timestamps via MBO event file.

    Each sample i corresponds to MBO event index min(i*stride + window - 1, n-1).
    """
    mbo_path = MBO_EVENT_DIR / f"{date_str}_mbo_events.npz"
    if not mbo_path.exists():
        return None
    mbo = np.load(mbo_path, allow_pickle=False)
    ts_events = mbo["timestamps"].astype(np.int64)
    n_events = len(ts_events)
    event_idx = np.minimum(idx_in_day * V342_STRIDE + V342_WINDOW - 1, n_events - 1)
    return ts_events[event_idx]


def run_one_date(
    date_str: str,
    idx_in_day: np.ndarray,
    direction: str,
    strength: np.ndarray,
    tp_ticks: float,
    sl_ticks: float,
    hold_s: float,
    cancel_s: float,
    order_type: str,
) -> List[dict]:
    """
    Run FIFOReplayEngine for one trading day. Returns list of fill records.
    Imports inside the function so worker processes get clean imports.
    """
    from alpha_discovery.deep_models.fifo_market_replay import FIFOReplayEngine

    ts_ns = map_signals_to_timestamps(date_str, idx_in_day)
    if ts_ns is None:
        return [{"date": date_str, "error": "missing_mbo_events"}]

    signals = []
    for i, t in enumerate(ts_ns):
        signals.append({
            "ts_ns":     int(t),
            "direction": direction,
            "strength":  float(strength[i]),
        })
    if not signals:
        return []

    # Map order_type to engine arg
    if order_type == "passive_at_touch":
        engine_order_type = "limit"
    elif order_type == "market":
        engine_order_type = "market"
    elif order_type == "chase":
        engine_order_type = "chase"
    else:
        raise ValueError(f"order_type {order_type}")

    cancel_ns = int(cancel_s * 1_000_000_000)
    hold_ns = int(hold_s * 1_000_000_000)

    try:
        engine = FIFOReplayEngine(
            date=date_str,
            cancel_after_ns=cancel_ns,
            max_hold_ns=hold_ns,
        )
    except FileNotFoundError as e:
        return [{"date": date_str, "error": f"no_dbn: {e}"}]
    except Exception as e:
        return [{"date": date_str, "error": f"engine_init: {e}"}]

    try:
        trades = engine.simulate(
            signals=signals,
            tp_ticks=tp_ticks,
            sl_ticks=sl_ticks,
            order_type=engine_order_type,
        )
    except Exception as e:
        return [{"date": date_str, "error": f"simulate: {e}"}]

    fills: List[dict] = []
    for t in trades:
        # extract fields safely (dataclass)
        fill_type = t.exit_reason
        entry_ts = t.entry_ts_ns
        exit_ts = t.exit_ts_ns
        hold_seconds = (exit_ts - entry_ts) / 1e9 if entry_ts and exit_ts else 0.0
        net_ticks = float(t.pnl_ticks_net)
        net_dollars = float(t.pnl_dollars)
        fills.append({
            "date": date_str,
            "ts_signal_ns": int(t.signal_ts_ns),
            "ts_entry_ns":  int(entry_ts) if entry_ts else 0,
            "ts_exit_ns":   int(exit_ts) if exit_ts else 0,
            "direction":    t.direction,
            "order_type":   t.order_type,
            "entry_raw":    int(t.entry_price_raw) if t.entry_price_raw else 0,
            "exit_raw":     int(t.exit_price_raw) if t.exit_price_raw else 0,
            "hold_s":       hold_seconds,
            "fill_type":    fill_type,
            "net_ticks":    net_ticks,
            "net_dollars":  net_dollars,
            "queue_ahead":  int(t.queue_ahead),
            "queue_wait_ns": int(t.queue_wait_ns),
            "slippage_ticks": float(t.slippage_ticks),
            "pred_strength":  float(t.pred_strength),
        })
    return fills


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", default="1", choices=list(HORIZON_PRED_KEY.keys()))
    ap.add_argument("--side", default="long", choices=["long", "short"])
    ap.add_argument("--conf-band", default="top0.5")
    ap.add_argument("--tp-ticks", type=float, default=1.0)
    ap.add_argument("--sl-ticks", type=float, default=0.5)
    ap.add_argument("--hold-s", type=float, default=1.5)
    ap.add_argument("--cancel-s", type=float, default=1.0)
    ap.add_argument("--order-type", default="passive_at_touch",
                    choices=["passive_at_touch", "market", "chase"])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--config-name", default=None,
                    help="output CSV name prefix; auto-generated if absent")
    args = ap.parse_args()

    if args.config_name is None:
        args.config_name = (f"v342_{args.side}_{args.horizon}s_{args.conf_band}"
                            f"_tp{args.tp_ticks}_sl{args.sl_ticks}"
                            f"_h{args.hold_s}_c{args.cancel_s}_{args.order_type}")

    log.info(f"Loading concat NPZ: {CONCAT_NPZ}")
    arrs = load_concat()
    n_dates = arrs["oot_dates"].shape[0]
    log.info(f"  unique dates: {n_dates}")

    signals_by_date = per_date_signals(arrs, args.horizon, args.side, args.conf_band)

    log.info(f"Running FIFO replay across {len(signals_by_date)} days "
             f"with {args.workers} workers...")

    fills_all: List[dict] = []
    errors: List[dict] = []
    tasks = [(d, idx, dir_arr[0] if len(dir_arr) else args.side, st)
             for d, (idx, dir_arr, st) in signals_by_date.items()]

    # workers limited by mp context; 12 is reasonable
    workers = max(1, min(args.workers, len(tasks)))
    if workers == 1 or len(tasks) <= 1:
        # serial path (useful for first-test smoke run)
        for d, idx, direction, st in tasks:
            log.info(f"  [serial] {d}: {idx.size} signals")
            fills = run_one_date(d, idx, direction, st,
                                 args.tp_ticks, args.sl_ticks,
                                 args.hold_s, args.cancel_s, args.order_type)
            for f in fills:
                if "error" in f:
                    errors.append(f)
                else:
                    fills_all.append(f)
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futs = {ex.submit(run_one_date, d, idx, direction, st,
                              args.tp_ticks, args.sl_ticks,
                              args.hold_s, args.cancel_s, args.order_type): d
                    for d, idx, direction, st in tasks}
            for fut in as_completed(futs):
                d = futs[fut]
                try:
                    fills = fut.result()
                except Exception as e:
                    errors.append({"date": d, "error": f"future: {e}"})
                    log.error(f"  {d}: {e}")
                    continue
                for f in fills:
                    if "error" in f:
                        errors.append(f)
                    else:
                        fills_all.append(f)
                log.info(f"  {d}: {len([f for f in fills if 'error' not in f])} fills")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / f"{args.config_name}_fifo_fills.csv"
    if fills_all:
        import csv as _csv
        cols = list(fills_all[0].keys())
        with csv_path.open("w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in fills_all:
                w.writerow(r)
        log.info(f"wrote {csv_path}  ({len(fills_all)} fills)")
    else:
        log.warning("no fills produced")
        csv_path.write_text("")  # empty marker

    # first-pass metrics
    if fills_all:
        nets = np.array([f["net_ticks"] for f in fills_all])
        wins = (nets > 0).sum()
        gross_p = nets[nets > 0].sum()
        gross_l = -nets[nets < 0].sum()
        pf = gross_p / gross_l if gross_l > 0 else float("inf")
        wr = 100.0 * wins / len(nets)
        s = nets.mean() / (nets.std(ddof=1) + 1e-12) * np.sqrt(len(nets)) if len(nets) > 1 else 0.0
        log.info(f"  FIRST-PASS: n={len(nets)}  net_tk_mean={nets.mean():.4f}  "
                 f"PF={pf:.3f}  WR={wr:.2f}%  Sharpe(sqrt-N)={s:.3f}  "
                 f"sum_net={nets.sum():.2f} tk  errors={len(errors)}")

    # write summary json
    summary = {
        "config_name": args.config_name,
        "config": {
            "horizon": args.horizon, "side": args.side, "conf_band": args.conf_band,
            "tp_ticks": args.tp_ticks, "sl_ticks": args.sl_ticks,
            "hold_s": args.hold_s, "cancel_s": args.cancel_s,
            "order_type": args.order_type,
        },
        "n_dates": int(n_dates),
        "n_signal_days": len(signals_by_date),
        "n_fills": len(fills_all),
        "n_errors": len(errors),
        "errors": errors[:20],
        "csv_path": str(csv_path),
    }
    summary_path = OUT_DIR / f"{args.config_name}_fifo_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log.info(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
