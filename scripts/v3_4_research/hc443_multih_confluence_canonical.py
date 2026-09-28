"""
HC #443 — Multi-horizon confluence canonical replay.

Loads CNN-Mamba v2 per-date predictions for horizons {1s, 5s, 10s}, requires the
signal to be in the top-{conf-band} SHORT (or LONG) at ALL horizons, then runs
the canonical realtime_sl FIFO engine on the intersection.

This is the "Layer-B confluence" remediation per HC #442 R2 — fewer trades, each
one survives multi-horizon agreement.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("hc443")

LVL3 = Path("/home/jupiter/Lvl3Quant")
V2_OOT_DIR = LVL3 / "output" / "cnn_mamba_v2_bulk_oot_v2"
MBO_EVENT_DIR = LVL3 / "data" / "processed" / "mbo_events_smart_v3"
OUT_DIR = LVL3 / "output" / "hc432_v342_47day_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZON_IDX = {"1": 0, "5": 1, "10": 2}


def parse_conf_band(band: str) -> float:
    # "top0.5" -> 0.005, "top30" -> 0.30, etc.
    band = band.strip().lower().replace("top", "")
    val = float(band)
    return val / 100.0 if val >= 1.0 else val


def load_v2_day_all(date_str: str) -> Tuple[np.ndarray, int, int]:
    """Return (preds[:,3-horizon], window_size, stride)."""
    p = V2_OOT_DIR / f"{date_str}_predictions.npz"
    d = np.load(p, allow_pickle=False)
    preds = d["predictions"][:, :3].astype(np.float64)  # 1s, 5s, 10s
    return preds, int(d["window_size"]), int(d["stride"])


def select_confluence_signals(
    dates: List[str],
    side: str,
    conf_band: str,
    horizons: List[str],
) -> Dict[str, Tuple[np.ndarray, np.ndarray, int, int]]:
    """
    Return {date: (idx_in_day, strengths_at_h1, window, stride)} where signals
    are in the top-{conf_band}% side-aligned independently at EACH horizon, then
    intersected.

    Strength used downstream = horizon-1 strength (so existing engine code that
    takes strengths plays nicely; meta-info is in 'pred_strength' column).
    """
    per_day_preds: List[np.ndarray] = []
    day_meta: List[Tuple[str, int, int, int]] = []
    for d in dates:
        preds, ws, st = load_v2_day_all(d)
        per_day_preds.append(preds)
        day_meta.append((d, preds.shape[0], ws, st))
    all_preds = np.concatenate(per_day_preds, axis=0)  # (N, 3)
    log.info(f"  loaded {all_preds.shape[0]:,} samples across {len(dates)} dates, 3 horizons")

    frac = parse_conf_band(conf_band)
    intersected = np.ones(all_preds.shape[0], dtype=bool)
    for h in horizons:
        h_idx = HORIZON_IDX[h]
        col = all_preds[:, h_idx]
        if side == "long":
            side_mask = col > 0
            strength = col
        else:
            side_mask = col < 0
            strength = -col
        side_str = strength[side_mask]
        if side_str.size == 0:
            log.warning(f"  no side-{side} preds at h={h}s, returning empty")
            return {}
        k = max(1, int(side_str.size * frac))
        thresh = np.partition(side_str, -k)[-k]
        sel_h = side_mask & (strength >= thresh)
        log.info(f"  h={h}s: {int(sel_h.sum()):,} signals (k={k:,} thresh={thresh:.4g})")
        intersected &= sel_h
    log.info(f"  intersection: {int(intersected.sum()):,} signals")

    # Strength for downstream = h=1s strength (negated for short to keep positive)
    if side == "long":
        h1_strength = all_preds[:, HORIZON_IDX["1"]]
    else:
        h1_strength = -all_preds[:, HORIZON_IDX["1"]]

    out: Dict[str, Tuple[np.ndarray, np.ndarray, int, int]] = {}
    cursor = 0
    for (d, n, ws, st) in day_meta:
        sel_day = intersected[cursor:cursor + n]
        if sel_day.any():
            idx = np.flatnonzero(sel_day)
            strs = h1_strength[cursor:cursor + n][idx]
            out[d] = (idx, strs.astype(np.float64), ws, st)
        cursor += n
    return out


def map_to_ts(date_str: str, idx_in_day: np.ndarray, window: int, stride: int) -> np.ndarray:
    mbo = np.load(MBO_EVENT_DIR / f"{date_str}_mbo_events.npz", allow_pickle=False)
    ts = mbo["timestamps"].astype(np.int64)
    n = len(ts)
    event_idx = np.minimum(idx_in_day * stride + window - 1, n - 1)
    return ts[event_idx]


def run_one_date(
    date_str: str, idx: np.ndarray, strengths: np.ndarray,
    window: int, stride: int, side: str,
    tp: float, sl: float, hold_s: float, cancel_s: float, order: str,
) -> List[dict]:
    from alpha_discovery.deep_models.fifo_market_replay import FIFOReplayEngine

    mbo_p = MBO_EVENT_DIR / f"{date_str}_mbo_events.npz"
    if not mbo_p.exists():
        return [{"date": date_str, "error": "missing_mbo_events"}]

    ts_ns = map_to_ts(date_str, idx, window, stride)
    signals = [{"ts_ns": int(t), "direction": side, "strength": float(s)}
               for t, s in zip(ts_ns, strengths)]

    engine_order = {"passive_at_touch": "limit", "market": "market", "chase": "chase"}[order]
    cancel_ns = int(cancel_s * 1e9)
    hold_ns = int(hold_s * 1e9)
    try:
        engine = FIFOReplayEngine(date=date_str,
                                  cancel_after_ns=cancel_ns,
                                  max_hold_ns=hold_ns)
    except FileNotFoundError as e:
        return [{"date": date_str, "error": f"no_dbn: {e}"}]
    except Exception as e:
        return [{"date": date_str, "error": f"engine_init: {e}"}]
    try:
        trades = engine.simulate(signals=signals, tp_ticks=tp, sl_ticks=sl,
                                  order_type=engine_order)
    except Exception as e:
        return [{"date": date_str, "error": f"simulate: {e}"}]
    out = []
    for t in trades:
        out.append({
            "date": date_str,
            "ts_signal_ns": int(t.signal_ts_ns),
            "ts_entry_ns": int(t.entry_ts_ns or 0),
            "ts_exit_ns": int(t.exit_ts_ns or 0),
            "direction": t.direction,
            "order_type": t.order_type,
            "entry_raw": int(t.entry_price_raw or 0),
            "exit_raw": int(t.exit_price_raw or 0),
            "hold_s": (t.exit_ts_ns - t.entry_ts_ns) / 1e9 if t.entry_ts_ns and t.exit_ts_ns else 0.0,
            "fill_type": t.exit_reason,
            "net_ticks": float(t.pnl_ticks_net),
            "net_dollars": float(t.pnl_dollars),
            "queue_ahead": int(t.queue_ahead),
            "queue_wait_ns": int(t.queue_wait_ns),
            "slippage_ticks": float(t.slippage_ticks),
            "pred_strength": float(t.pred_strength),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", default="short", choices=["long", "short"])
    ap.add_argument("--conf-band", default="top30",
                    help="top-X% INSIDE EACH HORIZON. Use top30 for confluence (not top0.5).")
    ap.add_argument("--horizons", nargs="+", default=["1", "5", "10"],
                    help="Horizons to intersect, all must agree on side+band.")
    ap.add_argument("--tp-ticks", type=float, default=3.0)
    ap.add_argument("--sl-ticks", type=float, default=0.5)
    ap.add_argument("--hold-s", type=float, default=1.5)
    ap.add_argument("--cancel-s", type=float, default=1.0)
    ap.add_argument("--order-type", default="passive_at_touch")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--config-name", default=None)
    args = ap.parse_args()

    if args.config_name is None:
        args.config_name = (f"hc443_multih_{args.side}_"
                            f"{'_'.join(args.horizons)}_band{args.conf_band}_"
                            f"sl{args.sl_ticks}_tp{args.tp_ticks}_h{args.hold_s}")

    dates = sorted([p.stem.split("_")[0]
                    for p in V2_OOT_DIR.glob("2026*_predictions.npz")])
    log.info(f"hc443 multi-h confluence: {len(dates)} dates  cfg={args.config_name}")
    log.info(f"  horizons={args.horizons}  side={args.side}  band={args.conf_band}")

    sig_by_date = select_confluence_signals(
        dates, args.side, args.conf_band, args.horizons,
    )
    log.info(f"  signal days: {len(sig_by_date)}")
    if not sig_by_date:
        log.error("no signals after confluence — exiting")
        sys.exit(0)

    fills_all: List[dict] = []
    errors: List[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {}
        for d, (idx, strs, ws, st) in sig_by_date.items():
            fut = ex.submit(
                run_one_date, d, idx, strs, ws, st, args.side,
                args.tp_ticks, args.sl_ticks, args.hold_s, args.cancel_s, args.order_type,
            )
            futures[fut] = d
        for fut in as_completed(futures):
            d = futures[fut]
            try:
                r = fut.result()
            except Exception as e:
                errors.append({"date": d, "error": f"future: {e}"})
                continue
            for f in r:
                (errors if "error" in f else fills_all).append(f)
            log.info(f"  {d}: {len([f for f in r if 'error' not in f])} fills")

    import pandas as pd
    df = pd.DataFrame(fills_all)
    csv_path = OUT_DIR / f"{args.config_name}_fifo_fills.csv"
    df.to_csv(csv_path, index=False)
    log.info(f"wrote {csv_path}  fills={len(fills_all)}  errors={len(errors)}")

    if len(df) > 0:
        net = df["net_ticks"]
        gains = net[net > 0].sum()
        losses = -net[net < 0].sum()
        pf = gains / losses if losses > 0 else float("inf")
        sharpe = (net.mean() / net.std() * np.sqrt(len(net))) if net.std() > 0 else 0.0
        log.info(f"FIRST-PASS: n={len(df)} mean={net.mean():.4f} PF={pf:.3f} "
                 f"WR={(net>0).mean()*100:.2f}% Sharpe(sqrt-N)={sharpe:.3f}")
    else:
        log.warning("no fills produced")

    summary = {
        "config_name": args.config_name,
        "config": {
            "side": args.side, "horizons": args.horizons, "conf_band": args.conf_band,
            "tp_ticks": args.tp_ticks, "sl_ticks": args.sl_ticks,
            "hold_s": args.hold_s, "cancel_s": args.cancel_s,
            "order_type": args.order_type,
        },
        "n_dates_present": len(sig_by_date),
        "n_dates_target": len(dates),
        "n_fills": len(fills_all),
        "n_errors": len(errors),
        "csv_path": str(csv_path),
        "overall": (
            {"n": len(df), "mean_tk": float(net.mean()),
             "PF": float(pf), "WR_pct": float((net>0).mean()*100),
             "Sharpe_sqrtN": float(sharpe), "sum_tk": float(net.sum())}
            if len(df) > 0 else {}
        ),
        "errors_sample": errors[:20],
    }
    sj = OUT_DIR / f"{args.config_name}_summary.json"
    with open(sj, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"summary: {sj}")


if __name__ == "__main__":
    main()
