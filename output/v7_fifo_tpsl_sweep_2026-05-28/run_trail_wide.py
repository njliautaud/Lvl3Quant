#!/usr/bin/env python3
"""HC #494 R3 — trailing-stop exit-axis test with WIDER initial SL.

First trail sweep showed trail logic is correct but tight SL=0.25 (1 tick)
leaves no room for MFE to anchor before SL fires. Pivot to wider SL where
trail has room to operate mid-flight. Trail re-anchors SL to ENTRY (anchor=0)
once MFE crosses trigger.

Bounded by HC #428 R2: TP ≤ p90 MFE within h=1s = 1.5 ticks.
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
                    handlers=[logging.FileHandler(OUT_DIR / "run_trail_wide.log"),
                              logging.StreamHandler()])
log = logging.getLogger("trail_wide")

PCT = 0.05


def run_one(date_str, idx_in_day, direction, strength, window_size, stride,
            tp, sl, trail_trigger, trail_anchor):
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
        signals=signals, tp_ticks=tp, sl_ticks=sl, order_type="limit",
        trail_trigger_ticks=trail_trigger,
        trail_anchor_offset_ticks=trail_anchor,
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


def grade_cell(perday, tp, sl, trail_trigger, trail_anchor, tag, workers=6):
    jobs = []
    for date_str, rec in perday.items():
        for side in ["short", "long"]:
            sel = select_topk(rec, side, PCT)
            if sel is None:
                continue
            idx, strength = sel
            jobs.append((date_str, idx, side, strength, rec["window_size"],
                         rec["stride"], tp, sl, trail_trigger, trail_anchor))
    log.info(f"[{tag}] jobs={len(jobs)} tp={tp} sl={sl} trail_trig={trail_trigger} anchor={trail_anchor}")
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
    df.to_parquet(OUT_DIR / f"fills_trailwide_{tag}.parquet")
    return df


def main():
    t0 = time.time()
    log.info("=== HC #494 R3 trail-wide sweep (3-day) ===")
    perday = load_v7_for_dates(DATES)
    if not perday:
        log.error("no dates aligned"); sys.exit(1)

    # (tag, tp, sl, trigger, anchor)
    cells = [
        # SL=0.5 — 2 ticks. Trail anchors at +1tick MFE, locks at entry.
        ("BASE_SL050",         1.0, 0.5,  0.0,  0.0),
        ("TRAIL025_SL050",     1.0, 0.5,  0.25, 0.0),
        ("TRAIL050_SL050",     1.0, 0.5,  0.5,  0.0),
        # SL=0.75 — 3 ticks. More room for trail to fire mid-flight.
        ("BASE_SL075",         1.0, 0.75, 0.0,  0.0),
        ("TRAIL025_SL075",     1.0, 0.75, 0.25, 0.0),
        ("TRAIL050_SL075",     1.0, 0.75, 0.5,  0.0),
        # TP=1.5 — at HC #428 R2 ceiling. SL=0.75. Trail at +0.5t.
        ("BASE_TP15_SL075",    1.5, 0.75, 0.0,  0.0),
        ("TRAIL050_TP15_SL075",1.5, 0.75, 0.5,  0.0),
    ]
    summaries = []
    for tag, tp, sl, trigger, anchor in cells:
        elapsed = time.time() - t0
        if elapsed > 1700:
            log.warning(f"budget hit, skip {tag}")
            continue
        df = grade_cell(perday, tp, sl, trigger, anchor, tag, workers=6)
        if df is None:
            log.warning(f"[{tag}] no fills")
            continue
        nz = int((df["net_ticks"] != 0).sum())
        s = summarize(df, tag, tp, sl, PCT)
        n_sl = int((df["fill_type"] == "sl").sum())
        n_trail = int((df["fill_type"] == "trail_sl").sum())
        n_tp = int((df["fill_type"] == "tp").sum())
        n_mh = int((df["fill_type"] == "max_hold").sum())
        log.info(f"[{tag}] NET={s['fifo_net_t_per_trade']:+.4f} gross={s['fifo_gross_t_per_trade']:+.4f} "
                 f"n={s['n_filled']} wr={s['wr']:.3f} skew={s['regime_skew']:.3f} "
                 f"sl={n_sl} trail_sl={n_trail} tp={n_tp} mh={n_mh} nz={nz}")
        s["n_sl"] = n_sl
        s["n_trail_sl"] = n_trail
        s["n_tp"] = n_tp
        s["n_max_hold"] = n_mh
        summaries.append(s)

    if summaries:
        sdf = pd.DataFrame(summaries)
        sdf = sdf.sort_values("fifo_net_t_per_trade", ascending=False)
        sdf.to_csv(OUT_DIR / "summary_trail_wide.csv", index=False)
        log.info("FINAL RANKING:\n" + sdf.to_string(index=False))

    log.info(f"DONE in {(time.time()-t0):.0f}s")


if __name__ == "__main__":
    main()
