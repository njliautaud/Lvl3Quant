#!/usr/bin/env python3
"""HC #494 R3 — confidence-tightening at the best TP/SL cell.

Best cell so far: TP=1.0 / SL=0.25 / top-5% → FIFO net -0.055 t/trade.

Test if tightening to top-1% (higher pred-strength) lifts gross above
the commission floor (0.376). Also test short-only at top-1% since
short side has been consistently the better edge.
"""
from __future__ import annotations
import sys, time, json, logging, multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd

LVL3_ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(LVL3_ROOT))
sys.path.insert(0, str(LVL3_ROOT / "output" / "v7_fifo_tpsl_sweep_2026-05-28"))
from run_sweep import (load_v7_for_dates, select_topk, MBO_EVENT_DIR,
                       HOLD_S, CANCEL_S, DATES, summarize)

OUT_DIR = LVL3_ROOT / "output" / "v7_fifo_tpsl_sweep_2026-05-28"
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.FileHandler(OUT_DIR / "run_top1.log"),
                              logging.StreamHandler()])
log = logging.getLogger("top1")

TP, SL = 1.0, 0.25


def run_one(date_str, idx_in_day, direction, strength, window_size, stride):
    from alpha_discovery.deep_models.fifo_market_replay import FIFOReplayEngine
    mbo_path = MBO_EVENT_DIR / f"{date_str}_mbo_events.npz"
    if not mbo_path.exists():
        return []
    mbo = np.load(mbo_path, allow_pickle=False)
    ts_events = mbo["timestamps"].astype(np.int64)
    n_events = len(ts_events)
    evt_idx = np.minimum(idx_in_day * stride + window_size - 1, n_events - 1)
    ts_ns = ts_events[evt_idx]
    signals = [{"ts_ns": int(t), "direction": direction, "strength": float(strength[i])}
               for i, t in enumerate(ts_ns)]
    if not signals:
        return []
    engine = FIFOReplayEngine(
        date=date_str,
        cancel_after_ns=int(CANCEL_S * 1_000_000_000),
        max_hold_ns=int(HOLD_S * 1_000_000_000),
    )
    trades = engine.simulate(
        signals=signals, tp_ticks=TP, sl_ticks=SL, order_type="limit",
    )
    rows = []
    for t in trades:
        rows.append({
            "date": date_str,
            "direction": t.direction,
            "fill_type": t.exit_reason,
            "net_ticks": float(t.pnl_ticks_net),
            "pred_strength": float(t.pred_strength),
        })
    return rows


def grade_cell(perday, pct, sides, tag, workers=6):
    jobs = []
    for date_str, rec in perday.items():
        for side in sides:
            sel = select_topk(rec, side, pct)
            if sel is None:
                continue
            idx, strength = sel
            jobs.append((date_str, idx, side, strength, rec["window_size"], rec["stride"]))
    log.info(f"[{tag}] jobs={len(jobs)} pct={pct} sides={sides}")
    all_rows = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        futures = {ex.submit(run_one, *j): (j[0], j[2]) for j in jobs}
        for fut in as_completed(futures):
            try:
                rows = fut.result()
            except Exception as e:
                log.error(f"[{tag}] worker crashed: {e}")
                continue
            all_rows.extend(rows)
    if not all_rows:
        return None
    df = pd.DataFrame(all_rows)
    df.to_parquet(OUT_DIR / f"fills_top1_{tag}.parquet")
    return df


def main():
    t0 = time.time()
    log.info("=== HC #494 R3 top-1% sweep (3-day, TP=1.0/SL=0.25) ===")
    perday = load_v7_for_dates(DATES)
    if not perday:
        log.error("no dates aligned"); sys.exit(1)

    cells = [
        ("TOP01_BOTH",  0.01, ["short", "long"]),
        ("TOP01_SHORT", 0.01, ["short"]),
        ("TOP02_BOTH",  0.02, ["short", "long"]),
        ("TOP02_SHORT", 0.02, ["short"]),
        ("TOP05_SHORT", 0.05, ["short"]),  # control: short-only at best cell
    ]
    summaries = []
    for tag, pct, sides in cells:
        df = grade_cell(perday, pct, sides, tag, workers=6)
        if df is None:
            log.warning(f"[{tag}] no fills")
            continue
        s = summarize(df, tag, TP, SL, pct)
        n_sl = int((df["fill_type"] == "sl").sum())
        n_tp = int((df["fill_type"] == "tp").sum())
        n_mh = int((df["fill_type"] == "max_hold").sum())
        log.info(f"[{tag}] NET={s['fifo_net_t_per_trade']:+.4f} gross={s['fifo_gross_t_per_trade']:+.4f} "
                 f"n={s['n_filled']} wr={s['wr']:.3f} skew={s['regime_skew']:.3f} "
                 f"sl={n_sl} tp={n_tp} mh={n_mh}")
        s["n_sl"] = n_sl
        s["n_tp"] = n_tp
        s["n_max_hold"] = n_mh
        summaries.append(s)

    if summaries:
        sdf = pd.DataFrame(summaries)
        sdf = sdf.sort_values("fifo_net_t_per_trade", ascending=False)
        sdf.to_csv(OUT_DIR / "summary_top1.csv", index=False)
        log.info("FINAL RANKING:\n" + sdf.to_string(index=False))

    log.info(f"DONE in {(time.time()-t0):.0f}s")


if __name__ == "__main__":
    main()
