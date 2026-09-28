#!/usr/bin/env python3
"""HC #494 R1 candidate test — TOP02_SHORT @ TP=1.0/SL=0.25 across all 27 OOT days.

3-day result was FIFO net -0.030 t/trade with regime skew 1.80 (red-tailored).
Need full-OOT-set regrade to determine if (a) the -0.030 holds, (b) whether
the red-tailoring is sample-specific or systematic.

If full-set FIFO net ≥ 0 AND regime skew ≤ 0.50: first HC #494 R1 candidate.
If still negative but regime-stable: report honestly, axis-rotate.
If still red-tailored: confirm regime-dependent, reject.
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
                       HOLD_S, CANCEL_S, summarize)

OUT_DIR = LVL3_ROOT / "output" / "v7_fifo_tpsl_sweep_2026-05-28"
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.FileHandler(OUT_DIR / "run_top2_27d.log"),
                              logging.StreamHandler()])
log = logging.getLogger("top2_27d")

TP, SL, PCT = 1.0, 0.25, 0.02
DATES_27 = ['20260320', '20260322', '20260323', '20260324', '20260325',
            '20260326', '20260327', '20260329', '20260330', '20260331',
            '20260401', '20260402', '20260403', '20260405', '20260406',
            '20260407', '20260408', '20260409', '20260410', '20260412',
            '20260413', '20260414', '20260415', '20260416', '20260417',
            '20260419', '20260420']


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


def main():
    t0 = time.time()
    log.info(f"=== HC #494 R1 — TOP02_SHORT 27-day regrade ===")
    log.info(f"TP={TP} SL={SL} PCT={PCT} dates={len(DATES_27)}")
    perday = load_v7_for_dates(DATES_27)
    log.info(f"Loaded {len(perday)} dates with v7+v2 preds")

    jobs = []
    for date_str, rec in perday.items():
        sel = select_topk(rec, "short", PCT)
        if sel is None:
            continue
        idx, strength = sel
        jobs.append((date_str, idx, "short", strength, rec["window_size"], rec["stride"]))
    log.info(f"Total jobs: {len(jobs)}")

    all_rows = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=8, mp_context=ctx) as ex:
        futures = {ex.submit(run_one, *j): j[0] for j in jobs}
        for fut in as_completed(futures):
            d = futures[fut]
            try:
                rows = fut.result()
                log.info(f"  {d}: {len(rows)} trades")
            except Exception as e:
                log.error(f"  {d}: worker crashed {e}")
                continue
            all_rows.extend(rows)

    if not all_rows:
        log.error("no fills total"); return

    df = pd.DataFrame(all_rows)
    df.to_parquet(OUT_DIR / "fills_top2short_27d.parquet")

    # Per-day breakdown
    per_day = df.groupby("date").agg(
        n=("net_ticks", "size"),
        net_mean=("net_ticks", "mean"),
        gross_mean=("fill_type", lambda x: 0),  # placeholder
    )
    per_day["net_mean"] = df.groupby("date")["net_ticks"].mean()
    per_day["wr"] = df.groupby("date").apply(lambda g: (g["net_ticks"] > 0).mean())
    log.info("PER-DAY NET (t/trade):\n" + per_day.to_string())

    # Overall stats
    n = len(df)
    fifo_net = df["net_ticks"].mean()
    fifo_gross = fifo_net + 0.376  # COMMISSION_TICKS
    wr = (df["net_ticks"] > 0).mean()
    daily_mean = df.groupby("date")["net_ticks"].mean()
    skew = (daily_mean.max() - daily_mean.min()) / max(abs(daily_mean.max()), abs(daily_mean.min()), 1e-9)
    n_sl = int((df["fill_type"] == "sl").sum())
    n_tp = int((df["fill_type"] == "tp").sum())
    n_mh = int((df["fill_type"] == "max_hold").sum())
    log.info(f"=== HEADLINE: FIFO NET={fifo_net:+.4f}t/trade gross={fifo_gross:+.4f} n={n} "
             f"wr={wr:.3f} skew={skew:.3f} sl={n_sl} tp={n_tp} mh={n_mh} ===")

    # HC #494 R1 verdict
    if fifo_net > 0 and skew <= 0.50 and n >= 100:
        log.info("HC #494 R1 CANDIDATE — VIABLE")
    elif fifo_net > 0:
        log.info(f"FIFO-positive but regime-skew {skew:.2f} > 0.50 — fails R1")
    else:
        log.info(f"FIFO-net negative ({fifo_net:.4f}) — fails R1")

    log.info(f"DONE in {(time.time()-t0):.0f}s")


if __name__ == "__main__":
    main()
