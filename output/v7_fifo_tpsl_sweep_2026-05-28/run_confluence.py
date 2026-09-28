#!/usr/bin/env python3
"""
HC #490 R3 confluence test on top of v7 base cell (pct=5, TP=1.0, SL=0.25).
3 OOT days. FIFO graded. Same harness as run_sweep.py.

Confluence axes:
  1) PatchTST sign-agreement at h=1s
  2) Vol regime filter (rolling 5-min realized vol; high vs low)
  3) OFI sign-agreement (ofi_short_100 z-score sign vs v7 dir)
  4) Stacked best two of (1)-(3)

Verify-then-summarize: print 3 sample rows + nonzero counts per filter
before grading any cell.
"""
from __future__ import annotations
import sys, time, json, logging, multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd

LVL3 = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(LVL3))

OUT = LVL3 / "output" / "v7_fifo_tpsl_sweep_2026-05-28"
V7  = LVL3 / "output" / "meta_v7_prod" / "concat_oot_predictions.npz"
V2D = LVL3 / "output" / "cnn_mamba_v2_bulk_oot_v2"
PT  = LVL3 / "output" / "patchtst_bulk_oot"
MBO = LVL3 / "data" / "processed" / "mbo_events_smart_v3"

DATES = ["20260403", "20260413", "20260420"]
PCT, TP, SL = 0.05, 1.0, 0.25
HOLD_S, CANCEL_S = 1.5, 1.0
COMM = 0.376  # AMP RT commission in ticks

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.FileHandler(OUT / "confluence.log"),
                              logging.StreamHandler()])
log = logging.getLogger("conf")


# ── data loading ─────────────────────────────────────────────────────
def load_v7(target):
    d = np.load(V7, allow_pickle=False)
    preds = d["predictions"].astype(np.float32)
    dates = d["dates"].astype(str)
    out = {}
    for ds in target:
        mask = dates == ds
        v7p = preds[mask]
        if v7p.size == 0: continue
        v2 = np.load(V2D / f"{ds}_predictions.npz", allow_pickle=False)
        out[ds] = {"preds": v7p,
                   "window_size": int(v2["window_size"]),
                   "stride": int(v2["stride"])}
    return out


def load_patchtst(target):
    out = {}
    for ds in target:
        f = PT / f"{ds}_predictions.npz"
        if not f.exists():
            log.warning(f"patchtst missing: {ds}")
            continue
        z = np.load(f, allow_pickle=False)
        # horizons = [1s,5s,10s]; col 0 = 1s prediction
        out[ds] = {"preds_1s": z["predictions"][:, 0].astype(np.float32),
                   "window_size": int(z["window_size"]),
                   "stride": int(z["stride"])}
    return out


def load_mbo(ds):
    z = np.load(MBO / f"{ds}_mbo_events.npz", allow_pickle=False)
    return {"timestamps": z["timestamps"].astype(np.int64),
            "labels_1s": z["labels_1s"].astype(np.float32),
            "ofi_short": z["events"][:, 22].astype(np.float32),
            "n_events": z["timestamps"].size}


# ── selection / mapping ──────────────────────────────────────────────
def select_topk(preds, side, pct):
    if side == "long":
        mask = preds > 0; strength = preds
    else:
        mask = preds < 0; strength = -preds
    s = strength[mask]
    if s.size == 0: return None
    k = max(1, int(s.size * pct))
    thresh = np.partition(s, -k)[-k]
    sel = mask & (strength >= thresh)
    idx = np.where(sel)[0]
    return idx, strength[idx]


def evt_idx_v7(idx_v7, ws, st, n_events):
    return np.minimum(idx_v7 * st + ws - 1, n_events - 1)


# ── confluence masks ─────────────────────────────────────────────────
def patchtst_sign_mask(idx_v7, side, pt_rec):
    """PatchTST window index = v2 (= v7) index + 2 since v2 ws=1000, patch ws=500, stride=250."""
    pt_idx = idx_v7 + 2
    pt_idx = np.clip(pt_idx, 0, pt_rec["preds_1s"].size - 1)
    pt_pred = pt_rec["preds_1s"][pt_idx]
    if side == "long":
        return pt_pred > 0
    else:
        return pt_pred < 0


def rolling_vol_mask(idx_v7, side, mbo, ws, st, n_events, window_ns=300_000_000_000):
    """Rolling 5-min realized vol = std of labels_1s over prior 5 min by ts.
    HIGH = top 40% across the day's selected entries; LOW = bottom 40%.
    Returns dict {'high': mask, 'low': mask}.
    """
    eidx = evt_idx_v7(idx_v7, ws, st, n_events)
    ts = mbo["timestamps"][eidx]
    lab = mbo["labels_1s"]
    # Compute rolling vol at each entry: std of labels_1s in [t-5min, t]
    # Use binary search on global timestamps
    gts = mbo["timestamps"]
    vols = np.empty(eidx.size, dtype=np.float32)
    for i, (t, e) in enumerate(zip(ts, eidx)):
        lo = np.searchsorted(gts, t - window_ns, side="left")
        seg = lab[lo:e+1]
        seg = seg[~np.isnan(seg)]
        vols[i] = float(seg.std()) if seg.size > 10 else np.nan
    valid = ~np.isnan(vols)
    if valid.sum() < 10:
        return {"high": np.zeros_like(valid), "low": np.zeros_like(valid)}
    p60 = np.percentile(vols[valid], 60)
    p40 = np.percentile(vols[valid], 40)
    high = valid & (vols >= p60)
    low  = valid & (vols <= p40)
    return {"high": high, "low": low}


def ofi_sign_mask(idx_v7, side, mbo, ws, st, n_events):
    """OFI sign agreement: ofi_short_100 z-score at entry event."""
    eidx = evt_idx_v7(idx_v7, ws, st, n_events)
    ofi = mbo["ofi_short"][eidx]
    if side == "long":
        return ofi > 0
    else:
        return ofi < 0


# ── FIFO worker ──────────────────────────────────────────────────────
def fifo_run(date_str, idx_in_day, direction, strength, ws, st, tp, sl):
    from alpha_discovery.deep_models.fifo_market_replay import FIFOReplayEngine
    z = np.load(MBO / f"{date_str}_mbo_events.npz", allow_pickle=False)
    ts_events = z["timestamps"].astype(np.int64)
    n_events = len(ts_events)
    eidx = np.minimum(idx_in_day * st + ws - 1, n_events - 1)
    ts_ns = ts_events[eidx]
    sigs = [{"ts_ns": int(t), "direction": direction, "strength": float(strength[i])}
            for i, t in enumerate(ts_ns)]
    if not sigs: return []
    try:
        eng = FIFOReplayEngine(
            date=date_str,
            cancel_after_ns=int(CANCEL_S * 1e9),
            max_hold_ns=int(HOLD_S * 1e9),
        )
        trades = eng.simulate(signals=sigs, tp_ticks=tp, sl_ticks=sl, order_type="limit")
    except Exception as e:
        return [{"date": date_str, "direction": direction, "error": str(e)}]
    rows = []
    for t in trades:
        hold = (t.exit_ts_ns - t.entry_ts_ns) / 1e9 if (t.entry_ts_ns and t.exit_ts_ns) else 0.0
        rows.append({"date": date_str, "direction": t.direction, "hold_s": hold,
                     "fill_type": t.exit_reason, "net_ticks": float(t.pnl_ticks_net),
                     "pred_strength": float(t.pred_strength)})
    return rows


def grade(perday, filtered_idx_strength_by_date_side, tag, workers=6):
    """filtered = {date: {'long': (idx, strength), 'short': (...)}}"""
    jobs = []
    for ds, rec in perday.items():
        side_d = filtered_idx_strength_by_date_side.get(ds, {})
        for side in ["short", "long"]:
            if side not in side_d: continue
            idx, strength = side_d[side]
            if idx.size == 0: continue
            jobs.append((ds, idx, side, strength, rec["window_size"], rec["stride"], TP, SL))
    log.info(f"[{tag}] jobs={len(jobs)}")
    if not jobs: return None
    all_rows = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        futs = {ex.submit(fifo_run, *j): (j[0], j[2]) for j in jobs}
        for f in as_completed(futs):
            try:
                rows = f.result()
                all_rows.extend(rows)
            except Exception as e:
                log.error(f"worker {futs[f]} crashed: {e}")
    if not all_rows: return None
    df = pd.DataFrame(all_rows)
    if "error" in df.columns:
        df = df[df["error"].isna()].drop(columns=["error"])
    if df.empty: return None
    df.to_parquet(OUT / f"conf_{tag}.parquet")
    return df


def summarize(df, tag):
    fills = df[df["fill_type"].isin(["tp", "sl", "hold_expire", "horizon", "max_hold"])]
    n = len(fills)
    if n == 0:
        return {"tag": tag, "n": 0, "fifo_net": float("nan"), "fifo_gross": float("nan"),
                "wr": float("nan"), "skew": float("nan"), "by_date": "{}"}
    nets = fills["net_ticks"].values
    mean_net = float(nets.mean())
    by_date = fills.groupby("date")["net_ticks"].mean().to_dict()
    vals = np.array(list(by_date.values()))
    skew = float((vals.max() - vals.min()) / max(abs(vals.max()), abs(vals.min()))) \
           if max(abs(vals.max()), abs(vals.min())) > 1e-9 else float("nan")
    return {"tag": tag, "n": n,
            "fifo_net": mean_net,
            "fifo_gross": mean_net + COMM,
            "wr": float((nets > 0).mean()),
            "n_days": fills["date"].nunique(),
            "skew": skew,
            "by_date": json.dumps({k: round(float(v), 4) for k, v in by_date.items()})}


# ── main ─────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    log.info("=== CONFLUENCE TEST ===")
    perday = load_v7(DATES)
    if not perday:
        log.error("no v7 alignment"); sys.exit(1)
    pt = load_patchtst(DATES)
    pt_avail = sorted(pt.keys())
    log.info(f"PatchTST availability: {pt_avail}")

    # Build base top-5% selections per date per side
    base = {}
    mbo_cache = {}
    for ds, rec in perday.items():
        mbo_cache[ds] = load_mbo(ds)
        base[ds] = {}
        for side in ["long", "short"]:
            sel = select_topk(rec["preds"], side, PCT)
            if sel is None: continue
            base[ds][side] = sel  # (idx, strength)

    # Verify-then-report: counts per date per side, then a few sample masks
    log.info("--- BASE COUNTS (top-5%) ---")
    for ds, sd in base.items():
        for side, (idx, s) in sd.items():
            log.info(f"  {ds}/{side}: n_base={idx.size}")

    # Build confluence masks per date per side
    masks_pt = {}; masks_vol_high = {}; masks_vol_low = {}; masks_ofi = {}
    for ds, sd in base.items():
        mb = mbo_cache[ds]
        ws, st, ne = perday[ds]["window_size"], perday[ds]["stride"], mb["n_events"]
        masks_pt[ds] = {}; masks_vol_high[ds] = {}; masks_vol_low[ds] = {}; masks_ofi[ds] = {}
        for side, (idx, strength) in sd.items():
            # PatchTST
            if ds in pt:
                m = patchtst_sign_mask(idx, side, pt[ds])
            else:
                m = np.zeros(idx.size, dtype=bool)
            masks_pt[ds][side] = m
            # Vol
            vm = rolling_vol_mask(idx, side, mb, ws, st, ne)
            masks_vol_high[ds][side] = vm["high"]
            masks_vol_low[ds][side]  = vm["low"]
            # OFI
            masks_ofi[ds][side] = ofi_sign_mask(idx, side, mb, ws, st, ne)
            log.info(f"  {ds}/{side}: base={idx.size} "
                     f"pt_keep={int(masks_pt[ds][side].sum())} "
                     f"vol_hi={int(masks_vol_high[ds][side].sum())} "
                     f"vol_lo={int(masks_vol_low[ds][side].sum())} "
                     f"ofi={int(masks_ofi[ds][side].sum())}")

    # Print 3 sample rows for verification (first date)
    ds0 = list(base.keys())[0]
    if "short" in base[ds0]:
        idx0, s0 = base[ds0]["short"]
        log.info(f"SAMPLE {ds0}/short first 3 idx={idx0[:3]} strength={s0[:3]} "
                 f"pt_mask={masks_pt[ds0]['short'][:3]} "
                 f"volH={masks_vol_high[ds0]['short'][:3]} "
                 f"ofi={masks_ofi[ds0]['short'][:3]}")

    # Helper: apply mask dict to base
    def apply_mask(mask_by_date_side):
        out = {}
        for ds, sd in base.items():
            out[ds] = {}
            for side, (idx, strength) in sd.items():
                m = mask_by_date_side[ds][side]
                out[ds][side] = (idx[m], strength[m])
        return out

    # Grade each confluence cell
    summaries = []

    # 1) BASE (sanity reproduction of best cell)
    log.info("--- GRADING BASE ---")
    df = grade(perday, base, "base_pct5_tp1.0_sl0.25")
    if df is not None: summaries.append(summarize(df, "BASE"))

    # 2) PatchTST agreement
    log.info("--- GRADING PatchTST ---")
    df = grade(perday, apply_mask(masks_pt), "patchtst_agree")
    if df is not None: summaries.append(summarize(df, "PATCHTST"))

    # 3) Vol high / low
    log.info("--- GRADING VOL HIGH ---")
    df = grade(perday, apply_mask(masks_vol_high), "vol_high")
    if df is not None: summaries.append(summarize(df, "VOL_HIGH"))
    log.info("--- GRADING VOL LOW ---")
    df = grade(perday, apply_mask(masks_vol_low), "vol_low")
    if df is not None: summaries.append(summarize(df, "VOL_LOW"))

    # 4) OFI
    log.info("--- GRADING OFI ---")
    df = grade(perday, apply_mask(masks_ofi), "ofi_agree")
    if df is not None: summaries.append(summarize(df, "OFI"))

    # Decide best two single-confluence (excluding BASE) by FIFO net for stack
    singles = [s for s in summaries if s["tag"] not in ("BASE",)
               and not np.isnan(s["fifo_net"]) and s["n"] >= 50]
    singles_sorted = sorted(singles, key=lambda x: -x["fifo_net"])
    log.info(f"Singles ranked: {[(s['tag'], round(s['fifo_net'],4), s['n']) for s in singles_sorted]}")

    if len(singles_sorted) >= 2:
        top2 = singles_sorted[:2]
        def get_mask_for(tag):
            if tag == "PATCHTST": return masks_pt
            if tag == "VOL_HIGH": return masks_vol_high
            if tag == "VOL_LOW":  return masks_vol_low
            if tag == "OFI":      return masks_ofi
            return None
        m1 = get_mask_for(top2[0]["tag"])
        m2 = get_mask_for(top2[1]["tag"])
        stacked = {}
        for ds, sd in base.items():
            stacked[ds] = {}
            for side, (idx, strength) in sd.items():
                comb = m1[ds][side] & m2[ds][side]
                stacked[ds][side] = comb
        stack_tag = f"STACK_{top2[0]['tag']}_AND_{top2[1]['tag']}"
        log.info(f"--- GRADING {stack_tag} ---")
        df = grade(perday, apply_mask(stacked), stack_tag.lower())
        if df is not None: summaries.append(summarize(df, stack_tag))

    # Write summary
    sdf = pd.DataFrame(summaries)
    if not sdf.empty:
        sdf = sdf.sort_values("fifo_net", ascending=False)
        sdf.to_csv(OUT / "confluence_summary.csv", index=False)
        log.info("\n" + sdf.to_string(index=False))
    log.info(f"DONE in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
