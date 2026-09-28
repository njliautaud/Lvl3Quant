#!/usr/bin/env python3
"""
FIFO label generator for CNN-Mamba v3 retrain (HC #270/#271(D)/#278/#281).

For each date in mbo_events_smart_v3, walk through every training window
position (window_end_idx = WINDOW_SIZE - 1 + k * STRIDE), submit BOTH a
long and short signal at that decision time, simulate fills under BOTH
deployed live configs (tp4sl3 and tp8sl5), and emit per-window labels.

Output: data/processed/mbo_events_smart_v3_fifo_labels/<date>_fifo_labels.npz
Each NPZ contains arrays of length N_windows = (n_events - WINDOW + 1) // STRIDE + 1
indexed by window_k:
  window_k       : (N,)   int64 — k index of the window (matches v2 trainer windowing)
  ts_ns          : (N,)   int64 — timestamp of last event in window (decision time)

For each of the 4 (config, direction) combos:
  <config>_<dir>_filled       : (N,) bool
  <config>_<dir>_net_ticks    : (N,) float32 — pnl_ticks_net (incl. commission). 0.0 if unfilled
  <config>_<dir>_hit_tp       : (N,) bool — True iff exit_reason == 'tp'
  <config>_<dir>_exit_reason  : (N,) <U16 — 'tp'/'sl'/'max_hold'/'eod'/'timeout'/'unfilled'

Configs: A = tp4sl3 (TP=4 SL=3), B = tp8sl5 (TP=8 SL=5)
Directions: long, short
=> 16 outcome arrays per file.

Live config inherited from HC #279(A): cancel=2000ms, max_hold=30000ms, order_type='limit'.

Authority: HC #281 — user "go ahead and begin training it". User-owned trading research code.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------
# Paths + constants
# ----------------------------------------------------------------------
LVL3_ROOT = Path(os.environ.get("LVL3_ROOT", "/home/jupiter/Lvl3Quant"))
sys.path.insert(0, str(LVL3_ROOT))

MBO_EVENT_DIR = LVL3_ROOT / "data" / "processed" / "mbo_events_smart_v3"
DEFAULT_OUT_DIR = LVL3_ROOT / "data" / "processed" / "mbo_events_smart_v3_fifo_labels"

WINDOW_SIZE = 3000   # matches v2 trainer
STRIDE = 250         # matches v2 trainer

# Per HC #279(A): 2-second cancel, 30-second max hold, passive limit
CANCEL_AFTER_NS = 2_000_000_000
MAX_HOLD_NS     = 30_000_000_000

CONFIGS = {
    "tp4sl3": {"tp_ticks": 4.0, "sl_ticks": 3.0},
    "tp8sl5": {"tp_ticks": 8.0, "sl_ticks": 5.0},
}
DIRECTIONS = ("long", "short")

# Cap labels to prevent regression head gradient blowup
LABEL_CAP_TICKS = 20.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(processName)s] %(message)s",
)
log = logging.getLogger("fifo_label_v3")


# ----------------------------------------------------------------------
# Per-date worker
# ----------------------------------------------------------------------

def _label_one_date(date_str: str, out_dir: Path, force: bool = False) -> str:
    """
    Generate FIFO labels for a single date. Run inside a worker process.

    Returns a status string ("ok:<N> windows filled_AB=<rate>" or "skip:<reason>"
    or "fail:<exception>").
    """
    out_path = out_dir / f"{date_str}_fifo_labels.npz"
    if out_path.exists() and not force:
        return f"skip:already-labeled→{out_path.name}"

    mbo_npz = MBO_EVENT_DIR / f"{date_str}_mbo_events.npz"
    if not mbo_npz.exists():
        return f"skip:no-smart_v3-events"

    try:
        # Lazy import inside worker (avoids pickling FIFOReplayEngine class state)
        try:
            from alpha_discovery.deep_models.fifo_market_replay import FIFOReplayEngine
        except (ImportError, ModuleNotFoundError):
            # Fallback: direct import if __init__.py chain fails
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "fifo_market_replay",
                str(LVL3_ROOT / "alpha_discovery" / "deep_models" / "fifo_market_replay.py"))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            FIFOReplayEngine = mod.FIFOReplayEngine

        data = np.load(mbo_npz, allow_pickle=True)
        ts_events = data["timestamps"].astype(np.int64)
        n_events = len(ts_events)
        if n_events <= WINDOW_SIZE:
            return f"skip:only-{n_events}-events"

        # Compute window-end indices (matches v2 trainer)
        # k indexes windows: window k covers events [k*STRIDE, k*STRIDE + WINDOW)
        # window-end event = k*STRIDE + WINDOW - 1 (the LAST event in the window — decision time)
        end_idx = np.arange(WINDOW_SIZE - 1, n_events, STRIDE, dtype=np.int64)
        n_win = len(end_idx)
        window_ts = ts_events[end_idx]
        log.info(f"[{date_str}] {n_events:,} events → {n_win:,} windows")

        # Init engine ONCE per date (loads DBN under the hood — expensive)
        try:
            engine = FIFOReplayEngine(
                date=date_str,
                instrument_id=None,        # auto-detect (HC: ESH6 / ESM6 contract roll)
                cancel_after_ns=CANCEL_AFTER_NS,
                max_hold_ns=MAX_HOLD_NS,
                max_reprices=3,
                reprice_after_ns=1_000_000_000,
            )
        except FileNotFoundError as e:
            return f"skip:no-DBN ({e})"

        # Result arrays — index by window_k
        out: dict = {
            "window_k": np.arange(n_win, dtype=np.int64),
            "ts_ns": window_ts.astype(np.int64),
        }
        for cfg_name in CONFIGS:
            for direction in DIRECTIONS:
                tag = f"{cfg_name}_{direction}"
                out[f"{tag}_filled"]      = np.zeros(n_win, dtype=bool)
                out[f"{tag}_net_ticks"]   = np.zeros(n_win, dtype=np.float32)
                out[f"{tag}_gross_ticks"] = np.zeros(n_win, dtype=np.float32)
                out[f"{tag}_hit_tp"]      = np.zeros(n_win, dtype=bool)
                out[f"{tag}_exit_reason"] = np.full(n_win, "unfilled", dtype="<U16")
                out[f"{tag}_hold_time_ns"] = np.zeros(n_win, dtype=np.int64)

        # Pre-build ts_ns -> window_k lookup (engine returns trades indexed by signal_ts_ns)
        # Note: many windows may share the same ts_ns if events are dense — we'll
        # use a (ts_ns, direction) key instead to disambiguate; offset the short
        # signal by +1ns to avoid collision with the long signal at the same window.
        ts2k = {int(t): int(k) for k, t in enumerate(window_ts)}

        # For each config, submit both directions in ONE simulate call.
        # Offset short by +1ns from long so trade_by_ts disambiguates the two.
        SHORT_OFFSET_NS = 1
        for cfg_name, cfg in CONFIGS.items():
            tag_long = f"{cfg_name}_long"
            tag_short = f"{cfg_name}_short"

            signals = []
            for k, ts in enumerate(window_ts):
                signals.append(
                    {"ts_ns": int(ts), "direction": "long", "strength": 1.0}
                )
                signals.append(
                    {"ts_ns": int(ts) + SHORT_OFFSET_NS,
                     "direction": "short", "strength": 1.0}
                )

            t0 = time.time()
            try:
                trades = engine.simulate(
                    signals=signals,
                    tp_ticks=cfg["tp_ticks"],
                    sl_ticks=cfg["sl_ticks"],
                    order_type="limit",
                )
            except Exception as e:
                log.exception(f"[{date_str}] {cfg_name} simulate FAILED")
                return f"fail:simulate {cfg_name}: {e}"
            log.info(
                f"[{date_str}] {cfg_name}: {len(trades):,} fills out of "
                f"{len(signals):,} signals "
                f"({100.0*len(trades)/max(1,len(signals)):.1f}%) "
                f"in {time.time()-t0:.1f}s"
            )

            # Map each trade back to (window_k, direction) via signal_ts_ns
            for t in trades:
                ts_signal = int(t.signal_ts_ns)
                if t.direction == "short":
                    # Reverse the +1ns offset
                    ts_lookup = ts_signal - SHORT_OFFSET_NS
                else:
                    ts_lookup = ts_signal
                k = ts2k.get(ts_lookup, -1)
                if k < 0:
                    continue

                tag = f"{cfg_name}_{t.direction}"
                net = float(t.pnl_ticks_net)
                gross = float(t.pnl_ticks)
                # Cap to prevent gradient blowup
                net = max(-LABEL_CAP_TICKS, min(LABEL_CAP_TICKS, net))
                gross = max(-LABEL_CAP_TICKS, min(LABEL_CAP_TICKS, gross))

                out[f"{tag}_filled"][k] = True
                out[f"{tag}_net_ticks"][k] = net
                out[f"{tag}_gross_ticks"][k] = gross
                out[f"{tag}_hit_tp"][k] = (str(t.exit_reason) == "tp")
                out[f"{tag}_exit_reason"][k] = str(t.exit_reason)
                out[f"{tag}_hold_time_ns"][k] = int(t.hold_time_ns)

        # Compute summary fill / TP rates
        summary = {}
        for cfg_name in CONFIGS:
            for direction in DIRECTIONS:
                tag = f"{cfg_name}_{direction}"
                f = out[f"{tag}_filled"]
                summary[tag] = {
                    "filled": int(f.sum()),
                    "fill_rate": float(f.mean()) if n_win else 0.0,
                    "mean_net": float(out[f"{tag}_net_ticks"][f].mean()) if f.any() else 0.0,
                    "tp_rate": float(out[f"{tag}_hit_tp"][f].mean()) if f.any() else 0.0,
                }

        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, **out)
        log.info(
            f"[{date_str}] WROTE {out_path.name} | "
            + " ".join(
                f"{k}={v['filled']}/{n_win}({100*v['fill_rate']:.0f}%) "
                f"meanNet={v['mean_net']:+.2f}t TPr={100*v['tp_rate']:.0f}%"
                for k, v in summary.items()
            )
        )
        return f"ok:{n_win} windows"

    except Exception as e:
        log.exception(f"[{date_str}] FAILED")
        return f"fail:{e}"


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description="Generate FIFO labels for v3 retrain.")
    p.add_argument("--date", type=str, default=None,
                   help="Single date YYYYMMDD; if omitted, run all smart_v3 dates "
                        "in the configured range.")
    p.add_argument("--start-date", type=str, default="20251101",
                   help="Inclusive start date (default 2025-11-01 — ~60d before "
                        "first v3 OOT week).")
    p.add_argument("--end-date", type=str, default="20260430",
                   help="Inclusive end date (default 2026-04-30).")
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel processes (FIFO replay is CPU-bound).")
    p.add_argument("--force", action="store_true",
                   help="Re-label even if output exists.")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build date list
    if args.date:
        dates = [args.date]
    else:
        all_npz = sorted(MBO_EVENT_DIR.glob("*_mbo_events.npz"))
        dates = []
        for f in all_npz:
            d = f.name[:8]
            if args.start_date <= d <= args.end_date:
                dates.append(d)

    log.info(f"Labeling {len(dates)} dates → {out_dir}")
    log.info(f"Configs: {list(CONFIGS.keys())}; directions: {DIRECTIONS}")
    log.info(f"Window={WINDOW_SIZE} stride={STRIDE} cancel={CANCEL_AFTER_NS//1_000_000}ms "
             f"hold={MAX_HOLD_NS//1_000_000}ms")
    log.info(f"Workers: {args.workers}")

    statuses: dict[str, str] = {}
    t_start = time.time()

    if args.workers <= 1:
        for d in dates:
            statuses[d] = _label_one_date(d, out_dir, force=args.force)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {
                ex.submit(_label_one_date, d, out_dir, args.force): d for d in dates
            }
            for fut in as_completed(futures):
                d = futures[fut]
                try:
                    statuses[d] = fut.result()
                except Exception as e:
                    statuses[d] = f"fail:{e}"

    n_ok = sum(1 for s in statuses.values() if s.startswith("ok"))
    n_skip = sum(1 for s in statuses.values() if s.startswith("skip"))
    n_fail = sum(1 for s in statuses.values() if s.startswith("fail"))
    log.info(f"DONE in {time.time()-t_start:.0f}s — ok={n_ok} skip={n_skip} fail={n_fail}")
    if n_fail:
        log.error("Failed dates:")
        for d, s in sorted(statuses.items()):
            if s.startswith("fail"):
                log.error(f"  {d}: {s}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
