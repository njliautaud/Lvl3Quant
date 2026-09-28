#!/usr/bin/env python3
"""HC #425 R4 — Alternative TP/SL label-geometry generator (FIFO market replay).

HC #424 regime-audit verdict (c) found tp4sl3 has STRUCTURAL negative EV by
construction (~-0.19t/fill: 42% WR × +3.5t vs 58% × -2.9t). This script generates
alternative label geometries to find one with positive (or less-negative) base
rate.

Geometries (HC #74 FIFO market replay, passive limit at touch, 2s cancel):
  - tp6sl2   (favor letting winners run, cut losers fast)
  - tp8sl3   (more asymmetric)
  - tp4sl4   (symmetric baseline)
  - tp10sl3  (extreme asymmetric)
  - time-60s (pure time exit at 60s, no TP/SL)
  - time-120s (pure time exit at 120s, no TP/SL)

Vol-adaptive geometry (TP=2*sigma_30s, SL=1*sigma_30s) is NOT implemented here
because FIFOReplayEngine does not support per-signal TP/SL parameters. Would
require engine modification — out of scope for the 90-min budget.

Inputs:
  - data/processed/mbo_events_smart_v3/<date>_mbo_events.npz (for ts_ns)
  - data/processed/mbo_events_smart_v3_fifo_labels/<date>_fifo_labels.npz
    (window_k / ts_ns alignment — match existing schema)
  - data/raw/mbo/glbx-mdp3-<date>.mbo.dbn.zst (DBN replay)

Outputs:
  - output/hc425_alternative_labels/<geom>/<date>_alt_labels.npz
    keys: window_k, ts_ns, side, fill_price, exit_price, exit_ts,
          gross_ticks, net_ticks, hit_tp, hit_sl, hit_time, exit_reason,
          filled

OOT dates: 20260223-20260227 (5 days, ~241k windows).
Cost: 0.376 ticks per fill (HC #392).

HC #420 binding: user's authorized quant-trading research codebase.
"""
from __future__ import annotations

import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

LVL3 = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(LVL3))

MBO_EVENT_DIR = LVL3 / "data/processed/mbo_events_smart_v3"
EXISTING_LABELS_DIR = LVL3 / "data/processed/mbo_events_smart_v3_fifo_labels"
OUT_ROOT = LVL3 / "output/hc425_alternative_labels"

OOT_DATES = ["20260223", "20260224", "20260225", "20260226", "20260227"]

# HC #279(A) live config: 2s cancel
CANCEL_NS = 2_000_000_000

# Geometries: tp/sl/max_hold_ns + "is_time_exit" flag.
# For pure-time exits we set TP/SL to a very large value so the engine never
# triggers them, and the trade exits at max_hold (mid-price exit).
LARGE_TP = 1000.0
LARGE_SL = 1000.0

GEOMETRIES = {
    "tp6sl2":   {"tp": 6.0,  "sl": 2.0,  "max_hold_ns": 30_000_000_000, "is_time": False},
    "tp8sl3":   {"tp": 8.0,  "sl": 3.0,  "max_hold_ns": 30_000_000_000, "is_time": False},
    "tp4sl4":   {"tp": 4.0,  "sl": 4.0,  "max_hold_ns": 30_000_000_000, "is_time": False},
    "tp10sl3":  {"tp": 10.0, "sl": 3.0,  "max_hold_ns": 30_000_000_000, "is_time": False},
    "time60":   {"tp": LARGE_TP, "sl": LARGE_SL, "max_hold_ns":  60_000_000_000, "is_time": True},
    "time120":  {"tp": LARGE_TP, "sl": LARGE_SL, "max_hold_ns": 120_000_000_000, "is_time": True},
}

SHORT_OFFSET_NS = 1
CAP_TICKS = 20.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(processName)s] %(message)s",
)
log = logging.getLogger("hc425_alt_labels")


def _label_one_date(date_str: str, geoms_to_run: list[str]) -> str:
    """Worker: replays DBN once for date, simulates each geometry, writes per-geom NPZ."""
    from alpha_discovery.deep_models.fifo_market_replay import FIFOReplayEngine

    # Use the existing FIFO labels file's window_k/ts_ns to keep alignment
    existing = EXISTING_LABELS_DIR / f"{date_str}_fifo_labels.npz"
    if not existing.exists():
        return f"skip:{date_str}:no_existing_labels"
    z = np.load(existing)
    window_k = z["window_k"].astype(np.int64)
    window_ts = z["ts_ns"].astype(np.int64)
    n_win = len(window_k)
    log.info(f"[{date_str}] n_windows={n_win:,}")

    # Map (ts_ns, direction) -> window_k for trade -> window assignment
    ts2k = {int(t): int(k) for k, t in enumerate(window_ts)}

    # Build common signals list (long + short per window)
    signals = []
    for k, t in enumerate(window_ts):
        signals.append({"ts_ns": int(t), "direction": "long", "strength": 1.0})
        signals.append({"ts_ns": int(t) + SHORT_OFFSET_NS,
                        "direction": "short", "strength": 1.0})

    # We need a fresh engine per (date, max_hold) since max_hold_ns is set on init.
    # Group geometries by max_hold_ns to reduce engine inits.
    by_max_hold: dict[int, list[str]] = {}
    for g in geoms_to_run:
        cfg = GEOMETRIES[g]
        by_max_hold.setdefault(cfg["max_hold_ns"], []).append(g)

    for max_hold_ns, geom_group in by_max_hold.items():
        try:
            t0 = time.time()
            eng = FIFOReplayEngine(
                date=date_str,
                cancel_after_ns=CANCEL_NS,
                max_hold_ns=max_hold_ns,
            )
            log.info(f"[{date_str}] engine init max_hold={max_hold_ns/1e9:.0f}s "
                     f"in {time.time()-t0:.1f}s")
        except FileNotFoundError as e:
            return f"fail:{date_str}:no_dbn ({e})"

        for geom in geom_group:
            out_dir = OUT_ROOT / geom
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{date_str}_alt_labels.npz"
            if out_path.exists():
                log.info(f"[{date_str}] {geom} already exists, skip")
                continue

            cfg = GEOMETRIES[geom]
            t1 = time.time()
            try:
                trades = eng.simulate(
                    signals=signals,
                    tp_ticks=cfg["tp"],
                    sl_ticks=cfg["sl"],
                    order_type="limit",
                )
            except Exception as e:
                log.exception(f"[{date_str}] {geom} simulate FAILED")
                continue
            log.info(f"[{date_str}] {geom}: {len(trades):,} fills "
                     f"in {time.time()-t1:.1f}s")

            # Allocate output arrays — separate long/short flattened
            # Schema requested: entry_ts, side, fill_price, exit_price, exit_ts,
            # gross_ticks, net_ticks, hit_tp, hit_sl, hit_time
            # We also keep window_k for alignment back to predictions NPZ.
            n_trades = len(trades)
            out = {
                "window_k_long": np.full(n_win, -1, dtype=np.int64),
                "window_k_short": np.full(n_win, -1, dtype=np.int64),
                "ts_ns": window_ts,
                "long_filled": np.zeros(n_win, dtype=bool),
                "long_fill_price_raw": np.zeros(n_win, dtype=np.int64),
                "long_exit_price_raw": np.zeros(n_win, dtype=np.int64),
                "long_exit_ts_ns": np.zeros(n_win, dtype=np.int64),
                "long_gross_ticks": np.zeros(n_win, dtype=np.float32),
                "long_net_ticks": np.zeros(n_win, dtype=np.float32),
                "long_hit_tp": np.zeros(n_win, dtype=bool),
                "long_hit_sl": np.zeros(n_win, dtype=bool),
                "long_hit_time": np.zeros(n_win, dtype=bool),
                "long_exit_reason": np.full(n_win, "unfilled", dtype="<U16"),
                "short_filled": np.zeros(n_win, dtype=bool),
                "short_fill_price_raw": np.zeros(n_win, dtype=np.int64),
                "short_exit_price_raw": np.zeros(n_win, dtype=np.int64),
                "short_exit_ts_ns": np.zeros(n_win, dtype=np.int64),
                "short_gross_ticks": np.zeros(n_win, dtype=np.float32),
                "short_net_ticks": np.zeros(n_win, dtype=np.float32),
                "short_hit_tp": np.zeros(n_win, dtype=bool),
                "short_hit_sl": np.zeros(n_win, dtype=bool),
                "short_hit_time": np.zeros(n_win, dtype=bool),
                "short_exit_reason": np.full(n_win, "unfilled", dtype="<U16"),
            }

            for t in trades:
                ts_signal = int(t.signal_ts_ns)
                if t.direction == "short":
                    ts_lookup = ts_signal - SHORT_OFFSET_NS
                else:
                    ts_lookup = ts_signal
                k = ts2k.get(ts_lookup, -1)
                if k < 0:
                    continue

                gross = max(-CAP_TICKS, min(CAP_TICKS, float(t.pnl_ticks)))
                net = max(-CAP_TICKS, min(CAP_TICKS, float(t.pnl_ticks_net)))
                exit_reason = str(t.exit_reason)
                is_tp = (exit_reason == "tp")
                is_sl = (exit_reason == "sl")
                # max_hold / eod / timeout count as "time" exit
                is_time = exit_reason in ("max_hold", "eod", "timeout")

                if t.direction == "long":
                    out["window_k_long"][k] = k
                    out["long_filled"][k] = True
                    out["long_fill_price_raw"][k] = int(t.entry_price_raw)
                    out["long_exit_price_raw"][k] = int(t.exit_price_raw)
                    out["long_exit_ts_ns"][k] = int(t.exit_ts_ns)
                    out["long_gross_ticks"][k] = gross
                    out["long_net_ticks"][k] = net
                    out["long_hit_tp"][k] = is_tp
                    out["long_hit_sl"][k] = is_sl
                    out["long_hit_time"][k] = is_time
                    out["long_exit_reason"][k] = exit_reason
                else:
                    out["window_k_short"][k] = k
                    out["short_filled"][k] = True
                    out["short_fill_price_raw"][k] = int(t.entry_price_raw)
                    out["short_exit_price_raw"][k] = int(t.exit_price_raw)
                    out["short_exit_ts_ns"][k] = int(t.exit_ts_ns)
                    out["short_gross_ticks"][k] = gross
                    out["short_net_ticks"][k] = net
                    out["short_hit_tp"][k] = is_tp
                    out["short_hit_sl"][k] = is_sl
                    out["short_hit_time"][k] = is_time
                    out["short_exit_reason"][k] = exit_reason

            np.savez_compressed(out_path, **out)
            lf = out["long_filled"]
            sf = out["short_filled"]
            log.info(
                f"[{date_str}] WROTE {geom} | "
                f"long: {lf.sum()}/{n_win} ({100*lf.mean():.0f}%) "
                f"meanNet={out['long_net_ticks'][lf].mean():+.3f} | "
                f"short: {sf.sum()}/{n_win} ({100*sf.mean():.0f}%) "
                f"meanNet={out['short_net_ticks'][sf].mean():+.3f}"
            )

        # Free engine memory before next max_hold group
        del eng

    return f"ok:{date_str}"


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--geoms", type=str, default="all",
                   help="Comma-separated geom names or 'all'")
    p.add_argument("--dates", type=str, default=",".join(OOT_DATES))
    args = p.parse_args()

    if args.geoms == "all":
        geoms = list(GEOMETRIES.keys())
    else:
        geoms = args.geoms.split(",")
        for g in geoms:
            if g not in GEOMETRIES:
                raise ValueError(f"Unknown geom: {g}")

    dates = args.dates.split(",")

    log.info(f"Geometries: {geoms}")
    log.info(f"Dates: {dates}")
    log.info(f"Workers: {args.workers}")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    statuses: dict[str, str] = {}

    if args.workers <= 1:
        for d in dates:
            statuses[d] = _label_one_date(d, geoms)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(_label_one_date, d, geoms): d for d in dates}
            for fut in as_completed(futures):
                d = futures[fut]
                try:
                    statuses[d] = fut.result()
                except Exception as e:
                    statuses[d] = f"fail:{e}"

    log.info(f"DONE in {time.time()-t_start:.0f}s")
    for d, s in sorted(statuses.items()):
        log.info(f"  {d}: {s}")

    n_fail = sum(1 for s in statuses.values() if s.startswith("fail"))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
