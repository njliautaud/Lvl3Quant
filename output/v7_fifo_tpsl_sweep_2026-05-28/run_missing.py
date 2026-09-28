#!/usr/bin/env python3
"""Run missing TP/SL cells from the 2026-05-28 v7 sweep.

Self-contained (logic copied from run_sweep.py so multiprocessing pickling
works correctly — spawned workers re-import this module).
Cells to fill:
  pct=5%: tp ∈ {1.0, 1.25, 1.5} × sl ∈ {0.25, 0.5}     (6 cells)
  pct=1%: (tp=0.75, sl=0.25), (tp=1.0, sl=0.25)         (2 cells)
  pct=2%: (tp=0.75, sl=0.25)                            (1 cell)
"""
from __future__ import annotations

import sys, time, json, logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd

LVL3_ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(LVL3_ROOT))

OUT_DIR = LVL3_ROOT / "output" / "v7_fifo_tpsl_sweep_2026-05-28"
V7_PRED_NPZ = LVL3_ROOT / "output" / "meta_v7_prod" / "concat_oot_predictions.npz"
V2_DIR = LVL3_ROOT / "output" / "cnn_mamba_v2_bulk_oot_v2"
MBO_EVENT_DIR = LVL3_ROOT / "data" / "processed" / "mbo_events_smart_v3"

H_SEC = 1.0
HOLD_S = 1.5
CANCEL_S = 1.0
DATES = ["20260403", "20260413", "20260420"]

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.FileHandler(OUT_DIR / "run.log"),
                              logging.StreamHandler()])
log = logging.getLogger("tpsl_missing")


def load_v7_for_dates(target_dates):
    d = np.load(V7_PRED_NPZ, allow_pickle=False)
    preds = d["predictions"].astype(np.float32)
    dates = d["dates"].astype(str)
    perday = {}
    for date_str in target_dates:
        mask = dates == date_str
        v7_preds_d = preds[mask]
        if v7_preds_d.size == 0:
            continue
        v2_npz = V2_DIR / f"{date_str}_predictions.npz"
        if not v2_npz.exists():
            continue
        v2 = np.load(v2_npz, allow_pickle=False)
        ws = int(v2["window_size"])
        st = int(v2["stride"])
        nw = int(v2["n_windows"])
        if v7_preds_d.size > nw:
            continue
        perday[date_str] = {"preds": v7_preds_d, "window_size": ws, "stride": st}
        log.info(f"{date_str}: preds={v7_preds_d.size} ws={ws} st={st}")
    return perday


def select_topk(rec, side, pct):
    x = rec["preds"]
    if side == "long":
        mask = x > 0
        strength = x
    else:
        mask = x < 0
        strength = -x
    side_s = strength[mask]
    if side_s.size == 0:
        return None
    k = max(1, int(side_s.size * pct))
    thresh = np.partition(side_s, -k)[-k]
    selected = mask & (strength >= thresh)
    idx = np.where(selected)[0]
    if idx.size == 0:
        return None
    return idx, strength[idx]


def run_one(date_str, idx_in_day, direction, strength, window_size, stride, tp_ticks, sl_ticks):
    from alpha_discovery.deep_models.fifo_market_replay import FIFOReplayEngine
    mbo_path = MBO_EVENT_DIR / f"{date_str}_mbo_events.npz"
    if not mbo_path.exists():
        return [{"date": date_str, "error": "missing_mbo_events"}]
    mbo = np.load(mbo_path, allow_pickle=False)
    ts_events = mbo["timestamps"].astype(np.int64)
    n_events = len(ts_events)
    evt_idx = np.minimum(idx_in_day * stride + window_size - 1, n_events - 1)
    ts_ns = ts_events[evt_idx]
    signals = [{"ts_ns": int(t), "direction": direction, "strength": float(strength[i])}
               for i, t in enumerate(ts_ns)]
    if not signals:
        return []
    try:
        engine = FIFOReplayEngine(
            date=date_str,
            cancel_after_ns=int(CANCEL_S * 1_000_000_000),
            max_hold_ns=int(HOLD_S * 1_000_000_000),
        )
    except Exception as e:
        return [{"date": date_str, "direction": direction, "error": f"engine_init: {e}"}]
    try:
        trades = engine.simulate(signals=signals, tp_ticks=tp_ticks, sl_ticks=sl_ticks, order_type="limit")
    except Exception as e:
        return [{"date": date_str, "direction": direction, "error": f"simulate: {e}"}]
    rows = []
    for t in trades:
        hold_s = (t.exit_ts_ns - t.entry_ts_ns) / 1e9 if (t.entry_ts_ns and t.exit_ts_ns) else 0.0
        rows.append({
            "date": date_str,
            "direction": t.direction,
            "hold_s": hold_s,
            "fill_type": t.exit_reason,
            "net_ticks": float(t.pnl_ticks_net),
            "net_dollars": float(t.pnl_dollars),
            "pred_strength": float(t.pred_strength),
        })
    return rows


def grade_cell(perday, pct, tp, sl, tag, workers=6):
    jobs = []
    for date_str, rec in perday.items():
        for side in ["short", "long"]:
            sel = select_topk(rec, side, pct)
            if sel is None:
                continue
            idx, strength = sel
            jobs.append((date_str, idx, side, strength, rec["window_size"], rec["stride"], tp, sl))
    log.info(f"[{tag}] jobs={len(jobs)} TP={tp} SL={sl} pct={pct}")
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
    if "error" in df.columns:
        df = df[df["error"].isna()].drop(columns=["error"])
    if df.empty:
        return None
    df.to_parquet(OUT_DIR / f"fills_{tag}.parquet")
    return df


def summarize(df, tag, tp, sl, pct):
    n = len(df)
    fills = df[df["fill_type"].isin(["tp", "sl", "hold_expire", "horizon", "max_hold"])]
    n_filled = len(fills)
    if n_filled == 0:
        return {"tag": tag, "tp": tp, "sl": sl, "pct": pct, "n": n, "n_filled": 0,
                "fifo_net_t_per_trade": float("nan"), "fifo_gross_t_per_trade": float("nan"),
                "wr": float("nan"), "n_days": df["date"].nunique(),
                "by_date_net_mean": "{}", "regime_spread": float("nan")}
    nets = df["net_ticks"].values
    mean_net = float(nets.mean())
    mean_gross = mean_net + 0.376
    wr = float((nets > 0).mean())
    by_date = df.groupby("date")["net_ticks"].mean().to_dict()
    vals = np.array(list(by_date.values()))
    spread = float(vals.max() - vals.min()) if len(vals) >= 2 else float("nan")
    return {
        "tag": tag, "tp": tp, "sl": sl, "pct": pct, "n": n, "n_filled": n_filled,
        "fifo_net_t_per_trade": mean_net,
        "fifo_gross_t_per_trade": mean_gross,
        "wr": wr,
        "n_days": df["date"].nunique(),
        "by_date_net_mean": json.dumps({k: round(float(v), 4) for k, v in by_date.items()}),
        "regime_spread": spread,
    }


def main():
    t0 = time.time()
    log.info("=== v7 FIFO TP/SL MISSING CELLS (additive, self-contained) ===")
    perday = load_v7_for_dates(DATES)
    if not perday:
        log.error("No dates aligned, abort.")
        sys.exit(1)

    cells = []
    for tp in [1.0, 1.25, 1.5]:
        for sl in [0.25, 0.5]:
            cells.append((0.05, tp, sl))
    cells.append((0.01, 0.75, 0.25))
    cells.append((0.02, 0.75, 0.25))
    cells.append((0.01, 1.0,  0.25))

    summaries = []
    for pct, tp, sl in cells:
        tag = f"pct{int(pct*100)}_tp{tp}_sl{sl}"
        existing = OUT_DIR / f"fills_{tag}.parquet"
        if existing.exists():
            log.info(f"[{tag}] EXISTS, loading")
            df = pd.read_parquet(existing)
        else:
            elapsed = time.time() - t0
            if elapsed > 2400:
                log.warning(f"Budget exceeded at {elapsed:.0f}s, skip {tag}")
                continue
            log.info(f"[{tag}] grading pct={pct} tp={tp} sl={sl}")
            df = grade_cell(perday, pct, tp, sl, tag, workers=6)
            if df is None or df.empty:
                log.warning(f"[{tag}] no fills")
                continue
        nz = int((df["net_ticks"] != 0).sum())
        log.info(f"[{tag}] VERIFY first 3 rows:\n{df.head(3).to_string()}")
        log.info(f"[{tag}] nonzero net_ticks count = {nz}/{len(df)}")
        s = summarize(df, tag, tp, sl, pct)
        log.info(f"[{tag}] FIFO NET={s['fifo_net_t_per_trade']:.4f}t/trade "
                 f"(gross={s['fifo_gross_t_per_trade']:.4f}) n={s['n_filled']} "
                 f"wr={s['wr']:.3f} spread={s['regime_spread']:.4f} "
                 f"by_date={s['by_date_net_mean']}")
        summaries.append(s)

    sdf_new = pd.DataFrame(summaries)
    out_csv = OUT_DIR / "summary_missing.csv"
    sdf_new.to_csv(out_csv, index=False)
    if "fifo_net_t_per_trade" in sdf_new.columns and len(sdf_new):
        log.info("MISSING-CELL SUMMARY:\n" +
                 sdf_new.sort_values("fifo_net_t_per_trade", ascending=False).to_string(index=False))
    else:
        log.warning("No cells produced summaries.")
    log.info(f"DONE in {(time.time()-t0):.0f}s")


if __name__ == "__main__":
    main()
