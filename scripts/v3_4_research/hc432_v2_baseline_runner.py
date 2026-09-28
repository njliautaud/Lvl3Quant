#!/usr/bin/env python3
"""
HC #432 — v2 (cnn_mamba_v2) FIFO sanity baseline.

Runs the same FIFOReplayEngine harness used for v3.4.2 against the v2 per-date
predictions (cnn_mamba_v2_all_oot/YYYYMMDD_predictions.npz). The HC #413
known-good result is short / 1s / top0.5 / passive_at_touch ≈ +0.274 tk/fill.

If this run reproduces the HC #413 result within ±0.05 tk, the harness is
trusted. If not, the FIFO/timestamp mapping has a bug and v3.4.2 verdicts
must be reviewed before they're treated as actionable.

CLI mirrors hc432_fifo_full_market_replay.py + writes the same
*_summary.json / *_verdict.md so the leaderboard builder can ingest it.
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

LVL3 = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(LVL3))

# HC #437 FIX (2026-05-19): the previous V2_OOT_DIR pointed at
# `cnn_mamba_v2_all_oot/` which contains predictions generated with
# EVENT_WINDOW_SIZE=3000 (trainer default), even though
# fold_10_best.pt's `arch.window_size` is 1000. The May-3 audit
# (`scripts/v3_3_research/regenerate_v2_bulk_oot.py` header) documented
# this as a corrupt-inference issue: IC_1s collapses from the canonical
# 0.22 to ~0.01. HC #432 was therefore unknowingly evaluating a no-edge
# signal, producing -0.17 tk/fill ≈ noise + costs.
# `cnn_mamba_v2_bulk_oot_v2/` is the correct-window=1000 inference
# directory used by HC #413/HC #417. Switching to it restores edge.
V2_OOT_DIR = LVL3 / "output" / "cnn_mamba_v2_bulk_oot_v2"
OUT_DIR = LVL3 / "output" / "hc432_v342_47day_validation"
MBO_EVENT_DIR = LVL3 / "data" / "processed" / "mbo_events_smart_v3"

# v2 inference: window=3000, stride=250 (read from each NPZ to be safe)
HORIZON_IDX = {"1": 0, "5": 1, "10": 2}
HORIZON_SEC = {"1": 1.0, "5": 5.0, "10": 10.0}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("hc432_v2_baseline")


def parse_conf_band(band: str) -> float:
    if not band.startswith("top"):
        raise ValueError(band)
    return float(band[3:]) / 100.0


def list_dates() -> List[str]:
    dates = []
    for p in sorted(V2_OOT_DIR.glob("2026*_predictions.npz")):
        dates.append(p.stem.split("_")[0])
    return dates


def load_v2_day(date_str: str, horizon: str) -> Tuple[np.ndarray, int, int]:
    """Return (preds_at_horizon, window_size, stride) for one date."""
    p = V2_OOT_DIR / f"{date_str}_predictions.npz"
    d = np.load(p, allow_pickle=False)
    preds = d["predictions"][:, HORIZON_IDX[horizon]].astype(np.float64)
    ws = int(d["window_size"])
    st = int(d["stride"])
    return preds, ws, st


def select_signals_global(
    dates: List[str], horizon: str, side: str, conf_band: str
) -> Dict[str, Tuple[np.ndarray, np.ndarray, int, int]]:
    """
    Pool all dates, pick top-N% of side-aligned preds GLOBALLY, then group by date.

    Returns: { date: (idx_in_day, strengths, window_size, stride) }
    """
    all_preds: List[np.ndarray] = []
    day_meta: List[Tuple[str, int, int, int]] = []  # (date, n, window, stride)
    cursor = 0
    for d in dates:
        preds, ws, st = load_v2_day(d, horizon)
        all_preds.append(preds)
        day_meta.append((d, preds.size, ws, st))
        cursor += preds.size
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
    log.info(f"  v2 selected {int(selected.sum()):,} signals (k={k:,} thresh={thresh:.4g})")

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
    ap.add_argument("--horizon", default="1", choices=list(HORIZON_IDX.keys()))
    ap.add_argument("--side", default="short", choices=["long", "short"])
    ap.add_argument("--conf-band", default="top0.5")
    ap.add_argument("--tp-ticks", type=float, default=1.0)
    ap.add_argument("--sl-ticks", type=float, default=0.5)
    ap.add_argument("--hold-s", type=float, default=1.5)
    ap.add_argument("--cancel-s", type=float, default=1.0)
    ap.add_argument("--order-type", default="passive_at_touch")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--config-name", default="v2_short_1s_top0.5_baseline")
    ap.add_argument("--limit-dates", type=int, default=None,
                    help="for quick smoke tests; runs on first N dates only")
    args = ap.parse_args()

    dates = list_dates()
    if args.limit_dates:
        dates = dates[:args.limit_dates]
    log.info(f"v2 baseline: {len(dates)} dates  cfg={args.config_name}")

    sig_by_date = select_signals_global(dates, args.horizon, args.side, args.conf_band)
    log.info(f"  signal days: {len(sig_by_date)}")

    fills_all: List[dict] = []
    errors: List[dict] = []
    tasks = [(d, idx, strs, ws, st) for d, (idx, strs, ws, st) in sig_by_date.items()]
    workers = max(1, min(args.workers, len(tasks)))

    if workers == 1 or len(tasks) <= 1:
        for d, idx, strs, ws, st in tasks:
            r = run_one_date(d, idx, strs, ws, st, args.side, args.tp_ticks,
                             args.sl_ticks, args.hold_s, args.cancel_s,
                             args.order_type)
            for f in r:
                (errors if "error" in f else fills_all).append(f)
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futs = {ex.submit(run_one_date, d, idx, strs, ws, st, args.side,
                              args.tp_ticks, args.sl_ticks, args.hold_s,
                              args.cancel_s, args.order_type): d
                    for d, idx, strs, ws, st in tasks}
            for fut in as_completed(futs):
                d = futs[fut]
                try:
                    r = fut.result()
                except Exception as e:
                    errors.append({"date": d, "error": f"future: {e}"})
                    continue
                for f in r:
                    (errors if "error" in f else fills_all).append(f)
                log.info(f"  {d}: {len([f for f in r if 'error' not in f])} fills")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / f"{args.config_name}_fifo_fills.csv"
    if fills_all:
        pd.DataFrame(fills_all).to_csv(csv_path, index=False)
    else:
        csv_path.write_text("")
    log.info(f"wrote {csv_path}  fills={len(fills_all)}  errors={len(errors)}")

    # First-pass metrics
    if fills_all:
        nets = np.array([f["net_ticks"] for f in fills_all])
        wins = (nets > 0).sum()
        gp, gl = nets[nets > 0].sum(), -nets[nets < 0].sum()
        pf = gp / gl if gl > 0 else float("inf")
        wr = 100.0 * wins / len(nets)
        s = (nets.mean() / (nets.std(ddof=1) + 1e-12)) * np.sqrt(len(nets)) if len(nets) > 1 else 0.0
        log.info(f"FIRST-PASS: n={len(nets)} mean={nets.mean():+.4f} PF={pf:.3f} "
                 f"WR={wr:.2f}% Sharpe(sqrt-N)={s:.3f}")

    # Write fifo_summary.json (matches v3.4.2 shape for downstream consumers)
    (OUT_DIR / f"{args.config_name}_fifo_summary.json").write_text(json.dumps({
        "config_name": args.config_name,
        "config": {
            "horizon": args.horizon, "side": args.side, "conf_band": args.conf_band,
            "tp_ticks": args.tp_ticks, "sl_ticks": args.sl_ticks,
            "hold_s": args.hold_s, "cancel_s": args.cancel_s,
            "order_type": args.order_type,
        },
        "n_dates": len(dates),
        "n_signal_days": len(sig_by_date),
        "n_fills": len(fills_all),
        "n_errors": len(errors),
        "errors": errors[:20],
        "csv_path": str(csv_path),
    }, indent=2))

    # Run the full validator so a *_summary.json + *_verdict.md exist for the leaderboard
    if fills_all:
        import subprocess
        cmd = [
            "python3", str(LVL3 / "scripts/v3_4_research/hc432_validate_full.py"),
            "--fills-csv", str(csv_path),
            "--config-name", args.config_name,
            "--horizon", args.horizon, "--side", args.side, "--conf-band", args.conf_band,
            "--tp-ticks", str(args.tp_ticks), "--sl-ticks", str(args.sl_ticks),
            "--hold-s", str(args.hold_s), "--cancel-s", str(args.cancel_s),
            "--order-type", args.order_type,
            "--total-oot-dates", str(len(dates)),
        ]
        log.info("running validator on v2 baseline fills...")
        subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
