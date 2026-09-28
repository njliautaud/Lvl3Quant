#!/usr/bin/env python3
"""HC #494 R3 — trailing-breakeven exit-axis test.

Reuses the v7 1s-horizon sweep infrastructure (load_v7_for_dates,
select_topk, run_one) but calls the harness's new trail kwargs added
today: trail_trigger_ticks + trail_anchor_offset_ticks.

Base cell (winning): pct=5%, TP=1.0, SL=0.25. Same 3 OOT days.

Cells:
  BASE              trail_trigger=0     (control, reproduces -0.055)
  TRAIL_0.25        trigger=0.25 anchor=0     (re-anchor SL to entry on +0.25t)
  TRAIL_0.5         trigger=0.5  anchor=0     (re-anchor to entry on +0.5t)
  TRAIL_0.5_LOCK    trigger=0.5  anchor=0.4   (re-anchor 0.4t above entry on +0.5t)
  TRAIL_0.75        trigger=0.75 anchor=0     (more conservative)
"""
from __future__ import annotations
import sys, time, json, logging, multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd

LVL3_ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(LVL3_ROOT))

# Reuse the sweep helpers
sys.path.insert(0, str(LVL3_ROOT / "output" / "v7_fifo_tpsl_sweep_2026-05-28"))
from run_sweep import (load_v7_for_dates, select_topk, MBO_EVENT_DIR,
                       HOLD_S, CANCEL_S, DATES, summarize)

OUT_DIR = LVL3_ROOT / "output" / "v7_fifo_tpsl_sweep_2026-05-28"
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.FileHandler(OUT_DIR / "run_trail.log"),
                              logging.StreamHandler()])
log = logging.getLogger("trail_sweep")

PCT = 0.05
TP = 1.0
SL = 0.25


def run_one_trail(date_str, idx_in_day, direction, strength, window_size, stride,
                  trail_trigger, trail_anchor):
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


def grade_trail_cell(perday, trail_trigger, trail_anchor, tag, workers=6):
    jobs = []
    for date_str, rec in perday.items():
        for side in ["short", "long"]:
            sel = select_topk(rec, side, PCT)
            if sel is None:
                continue
            idx, strength = sel
            jobs.append((date_str, idx, side, strength, rec["window_size"],
                         rec["stride"], trail_trigger, trail_anchor))
    log.info(f"[{tag}] jobs={len(jobs)} trail_trigger={trail_trigger} trail_anchor={trail_anchor}")
    all_rows = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        futures = {ex.submit(run_one_trail, *j): (j[0], j[2]) for j in jobs}
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
    df.to_parquet(OUT_DIR / f"fills_trail_{tag}.parquet")
    return df


def main():
    t0 = time.time()
    log.info("=== HC #494 R3 trailing-stop sweep (3-day) ===")
    perday = load_v7_for_dates(DATES)
    if not perday:
        log.error("no dates aligned"); sys.exit(1)

    cells = [
        ("BASE",            0.0,  0.0),
        ("TRAIL_0.25",      0.25, 0.0),
        ("TRAIL_0.5",       0.5,  0.0),
        ("TRAIL_0.5_LOCK",  0.5,  0.4),
        ("TRAIL_0.75",      0.75, 0.0),
    ]
    summaries = []
    for tag, trigger, anchor in cells:
        elapsed = time.time() - t0
        if elapsed > 1800:  # 30min hard budget
            log.warning(f"budget hit, skip {tag}")
            continue
        df = grade_trail_cell(perday, trigger, anchor, tag, workers=6)
        if df is None:
            log.warning(f"[{tag}] no fills")
            continue
        # verify-then-report
        first3 = df.head(3).to_dict("records")
        nz_net = int((df["net_ticks"] != 0).sum())
        log.info(f"[{tag}] first3={first3}  nonzero_net={nz_net}/{len(df)}")

        s = summarize(df, tag, TP, SL, PCT)
        # fraction of SL-type exits with non-trail
        n_sl = int((df["fill_type"] == "sl").sum())
        n_trail = int((df["fill_type"] == "trail_sl").sum())
        n_tp = int((df["fill_type"] == "tp").sum())
        n_mh = int((df["fill_type"] == "max_hold").sum())
        log.info(f"[{tag}] FIFO NET={s['fifo_net_t_per_trade']:.4f}t/trade "
                 f"gross={s['fifo_gross_t_per_trade']:.4f} n={s['n_filled']} "
                 f"wr={s['wr']:.3f} skew={s['regime_skew']:.3f} "
                 f"sl={n_sl} trail_sl={n_trail} tp={n_tp} mh={n_mh}")
        s["n_sl"] = n_sl
        s["n_trail_sl"] = n_trail
        s["n_tp"] = n_tp
        s["n_max_hold"] = n_mh
        summaries.append(s)

    if summaries:
        sdf = pd.DataFrame(summaries)
        sdf = sdf.sort_values("fifo_net_t_per_trade", ascending=False)
        sdf.to_csv(OUT_DIR / "summary_trail.csv", index=False)
        log.info("FINAL RANKING:\n" + sdf.to_string(index=False))

    log.info(f"DONE in {(time.time()-t0):.0f}s")


if __name__ == "__main__":
    main()
