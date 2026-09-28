#!/usr/bin/env python3
"""
HC #428 R2 + HC #494 R1 — v7 1s-horizon TP/SL sweep on 3 OOT days.

STEP 1: Measure MFE distribution within h=1s for top-5% v7 signals using
        labels_1s (= mid(t+1s) - mid(t) in ticks). Report p25/p50/p75/p90 by side.

STEP 2: Grid sweep TP x SL on top-5% signals, capped at p90 of MFE per side.
        Then test top-1% at best 2 cells. Same 3 OOT dates.

Outputs: fills_*.parquet per cell, summary.csv, mfe_dist.json.
"""
from __future__ import annotations

import sys
import time
import json
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

LVL3_ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(LVL3_ROOT))

OUT_DIR = LVL3_ROOT / "output" / "v7_fifo_tpsl_sweep_2026-05-28"
OUT_DIR.mkdir(parents=True, exist_ok=True)

V7_PRED_NPZ = LVL3_ROOT / "output" / "meta_v7_prod" / "concat_oot_predictions.npz"
V2_DIR = LVL3_ROOT / "output" / "cnn_mamba_v2_bulk_oot_v2"
MBO_EVENT_DIR = LVL3_ROOT / "data" / "processed" / "mbo_events_smart_v3"

H_SEC = 1.0
HOLD_S = 1.5
CANCEL_S = 1.0

DATES = ["20260403", "20260413", "20260420"]  # red, green, flat

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.FileHandler(OUT_DIR / "run.log"), logging.StreamHandler()])
log = logging.getLogger("tpsl_sweep")


# ──────────────────────────────────────────────────────────────────────
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


def map_idx_to_event_idx(idx_in_day, window_size, stride, n_events):
    return np.minimum(idx_in_day * stride + window_size - 1, n_events - 1)


# ──────────────────────────────────────────────────────────────────────
# STEP 1: Measure MFE within 1s using labels_1s (= mid(t+1s) - mid(t))
# ──────────────────────────────────────────────────────────────────────
def measure_mfe_distribution(perday, pct=0.05):
    """For top-`pct` signals per date per side, return realized
    DIRECTIONAL excursion at h=1s in ticks.
    long signal: realized = labels_1s[evt]
    short signal: realized = -labels_1s[evt]
    Reported absolute "MFE-at-1s" is the directional realized in ticks.
    """
    out = {"long": [], "short": []}
    for date_str, rec in perday.items():
        mbo_path = MBO_EVENT_DIR / f"{date_str}_mbo_events.npz"
        if not mbo_path.exists():
            log.warning(f"{date_str}: no MBO npz, skip MFE measurement")
            continue
        mbo = np.load(mbo_path, allow_pickle=False)
        labels_1s = mbo["labels_1s"].astype(np.float32)
        n_events = labels_1s.size

        for side in ["long", "short"]:
            sel = select_topk(rec, side, pct)
            if sel is None:
                continue
            idx, strength = sel
            evt_idx = map_idx_to_event_idx(idx, rec["window_size"], rec["stride"], n_events)
            lab = labels_1s[evt_idx]
            # directional realized
            real = lab if side == "long" else -lab
            real = real[~np.isnan(real)]
            out[side].append(real)
            log.info(f"  {date_str}/{side}: n={real.size} mean={real.mean():.3f}t std={real.std():.3f}t")

    dist = {}
    for side in ["long", "short"]:
        if not out[side]:
            dist[side] = {"n": 0}
            continue
        all_real = np.concatenate(out[side])
        # Realized at 1s is a noisy proxy for MFE-within-1s. To be conservative,
        # use |realized| as a lower bound on MFE (true MFE >= |terminal|).
        # But for the TP ceiling we care about the *positive* tail (favorable side).
        favorable = all_real[all_real > 0]
        dist[side] = {
            "n_total": int(all_real.size),
            "n_favorable": int(favorable.size),
            "frac_favorable": float((all_real > 0).mean()),
            "p25_realized": float(np.percentile(all_real, 25)),
            "p50_realized": float(np.percentile(all_real, 50)),
            "p75_realized": float(np.percentile(all_real, 75)),
            "p90_realized": float(np.percentile(all_real, 90)),
            "p25_favorable": float(np.percentile(favorable, 25)) if favorable.size else float("nan"),
            "p50_favorable": float(np.percentile(favorable, 50)) if favorable.size else float("nan"),
            "p75_favorable": float(np.percentile(favorable, 75)) if favorable.size else float("nan"),
            "p90_favorable": float(np.percentile(favorable, 90)) if favorable.size else float("nan"),
            "mean": float(all_real.mean()),
            "std": float(all_real.std()),
        }
    return dist


# ──────────────────────────────────────────────────────────────────────
# STEP 2: TP/SL sweep — worker
# ──────────────────────────────────────────────────────────────────────
def run_one(date_str, idx_in_day, direction, strength, window_size, stride,
            tp_ticks, sl_ticks):
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
                "by_date_net_mean": "{}", "regime_skew": float("nan")}
    nets = df["net_ticks"].values
    mean_net = float(nets.mean())
    # gross = net + commission (0.376 ticks RT)
    mean_gross = mean_net + 0.376
    wr = float((nets > 0).mean())
    by_date = df.groupby("date")["net_ticks"].mean().to_dict()
    # regime_skew = (max - min) / max(|max|, |min|) across the 3 days
    vals = np.array(list(by_date.values()))
    if len(vals) >= 2 and max(abs(vals.max()), abs(vals.min())) > 1e-9:
        skew = float((vals.max() - vals.min()) / max(abs(vals.max()), abs(vals.min())))
    else:
        skew = float("nan")
    return {
        "tag": tag, "tp": tp, "sl": sl, "pct": pct, "n": n, "n_filled": n_filled,
        "fifo_net_t_per_trade": mean_net,
        "fifo_gross_t_per_trade": mean_gross,
        "wr": wr,
        "n_days": df["date"].nunique(),
        "by_date_net_mean": json.dumps({k: round(float(v), 4) for k, v in by_date.items()}),
        "regime_skew": skew,
    }


# ──────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    log.info("=== v7 FIFO TP/SL sweep (3-day) ===")
    perday = load_v7_for_dates(DATES)
    if not perday:
        log.error("No dates aligned, abort.")
        sys.exit(1)

    # ── STEP 1: MFE distribution
    log.info("--- STEP 1: MFE distribution at h=1s (top-5%) ---")
    dist = measure_mfe_distribution(perday, pct=0.05)
    with open(OUT_DIR / "mfe_dist.json", "w") as f:
        json.dump(dist, f, indent=2)
    log.info(f"MFE dist: {json.dumps(dist, indent=2)}")

    # Determine TP ceiling per side
    tp_ceil_long = dist["long"].get("p90_favorable", 1.0) if dist["long"].get("n_total", 0) else 1.0
    tp_ceil_short = dist["short"].get("p90_favorable", 1.0) if dist["short"].get("n_total", 0) else 1.0
    tp_ceil = max(tp_ceil_long, tp_ceil_short)  # use the more permissive for grid
    log.info(f"TP ceiling: long p90={tp_ceil_long:.3f}t short p90={tp_ceil_short:.3f}t -> grid cap {tp_ceil:.3f}t")

    # ── STEP 2: Grid sweep top-5%
    log.info("--- STEP 2: TP/SL grid sweep, top-5% ---")
    tp_grid_full = [0.25, 0.5, 0.75, 1.0]
    sl_grid = [0.25, 0.5, 0.75, 1.0]
    # cap TP grid by p90 (round up slightly)
    tp_grid = [tp for tp in tp_grid_full if tp <= tp_ceil + 0.25]
    if not tp_grid:
        tp_grid = [min(tp_grid_full)]
    log.info(f"TP grid (after cap): {tp_grid}")

    summaries = []
    pct = 0.05
    for tp in tp_grid:
        for sl in sl_grid:
            tag = f"pct{int(pct*100)}_tp{tp}_sl{sl}"
            elapsed = (time.time() - t0)
            if elapsed > 2400:  # 40min budget for grid
                log.warning(f"Budget exceeded at {elapsed:.0f}s, skipping {tag}")
                continue
            df = grade_cell(perday, pct, tp, sl, tag, workers=6)
            if df is None:
                log.warning(f"[{tag}] no fills")
                continue
            s = summarize(df, tag, tp, sl, pct)
            log.info(f"[{tag}] FIFO NET={s['fifo_net_t_per_trade']:.4f}t/trade "
                     f"(gross={s['fifo_gross_t_per_trade']:.4f}) n={s['n_filled']} wr={s['wr']:.3f} "
                     f"skew={s['regime_skew']:.3f}")
            summaries.append(s)

    # ── STEP 2b: Top-1% at best 2 (TP,SL) cells from top-5%
    log.info("--- STEP 2b: top-1% at best 2 (TP,SL) cells ---")
    if summaries:
        # rank by FIFO net
        ranked = sorted([s for s in summaries if not np.isnan(s["fifo_net_t_per_trade"])],
                        key=lambda x: -x["fifo_net_t_per_trade"])
        top2 = ranked[:2]
        for s in top2:
            tp, sl = s["tp"], s["sl"]
            pct1 = 0.01
            tag = f"pct1_tp{tp}_sl{sl}"
            elapsed = (time.time() - t0)
            if elapsed > 2700:
                log.warning(f"Budget exceeded, skip {tag}")
                continue
            df = grade_cell(perday, pct1, tp, sl, tag, workers=6)
            if df is None:
                continue
            s2 = summarize(df, tag, tp, sl, pct1)
            log.info(f"[{tag}] FIFO NET={s2['fifo_net_t_per_trade']:.4f}t/trade "
                     f"n={s2['n_filled']} wr={s2['wr']:.3f} skew={s2['regime_skew']:.3f}")
            summaries.append(s2)

    # ── Write summary
    if summaries:
        sdf = pd.DataFrame(summaries)
        sdf = sdf.sort_values("fifo_net_t_per_trade", ascending=False)
        sdf.to_csv(OUT_DIR / "summary.csv", index=False)
        log.info("\n" + sdf.to_string(index=False))

    log.info(f"DONE in {(time.time()-t0):.0f}s")


if __name__ == "__main__":
    main()
