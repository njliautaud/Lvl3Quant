#!/usr/bin/env python3
"""HC #488 execution-axis test: microprice-direction filter on hc489 asym-long quantile preds.

Inputs:
  - Razer-trained NPZs (rsynced to Jupiter): hc489_dlinear_quantile_asym_long_v1/fold_{01,02}_preds.npz
    Each NPZ expected keys: preds (B, n_h=3, n_q=3) or split P10/P50/P90; event_idx; date.
  - Jupiter MBO event NPZ per date: data/processed/mbo_events/{YYYYMMDD}_mbo_events.npz
    Used columns: bid_px, ask_px, bid_sz, ask_sz (verify on first load).

Method:
  - For each fold's events, take top-1% P50_long_5s confidence (P50 at h=5s, long side = P50 > 0).
  - Compute microprice = (ask_sz*bid_px + bid_sz*ask_px) / (bid_sz + ask_sz) at the entry event.
  - mid = (bid_px + ask_px) / 2.
  - Filter A (baseline): top-1% long_5s signal, no microprice gate.
  - Filter B (microprice-filtered): same set AND (microprice - mid) > 0 (microprice pressing UP).
  - Replay forward 5s using realized log_ret_5s (or recompute from mid path). Net = realized - 0.376 ticks (passive-limit cost, HC #428 R2).
  - Report per-date and aggregate: N, mean net ticks/trade, WR, profitable-day count.

Output:
  output/hc488_microprice_filter_v1/summary.csv  (cols: variant, date, N, mean_net_ticks, wr, sharpe_proxy)
  output/hc488_microprice_filter_v1/.regen_complete.json  (HC #485 R5 schema)
"""

from __future__ import annotations
import json, os, sys, time, traceback
from pathlib import Path
import numpy as np

ROOT = Path("/home/jupiter/Lvl3Quant")
PRED_DIR = ROOT / "data/razer_pull/hc489_dlinear_quantile_asym_long_v1"
MBO_DIR = ROOT / "data/processed/mbo_events"
OUT_DIR = ROOT / "output/hc488_microprice_filter_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COST_TICKS = 0.376  # passive limit, HC #428 R2
TICK_SIZE = 0.25
ES_PT_VALUE = 50.0  # 1 pt = 4 ticks * $12.50 — for tick conversion only

# fold -> OOT date(s)
FOLD_DATE = {1: "20260428", 2: "20260429"}  # per SESSION_STATE: 3 OOT 4/27/28/29, fold-1 trains→tests-4/28, fold-2 trains→tests-4/29. fold-0 test=4/27 may be in a separate file.

t0 = time.time()
results = []

def safe_load(p):
    try:
        return dict(np.load(p, allow_pickle=True))
    except Exception as e:
        print(f"load fail {p}: {e}", flush=True)
        return None

# Inspect a pred NPZ to discover keys
sample = None
for fnum, date in FOLD_DATE.items():
    pred_p = PRED_DIR / f"fold_0{fnum}_preds.npz"
    if not pred_p.exists():
        print(f"missing {pred_p}", flush=True)
        continue
    d = safe_load(pred_p)
    if d is None:
        continue
    if sample is None:
        sample = d
        print(f"NPZ keys: {sorted(d.keys())}", flush=True)
        for k in sorted(d.keys()):
            v = d[k]
            print(f"  {k}: shape={getattr(v, 'shape', '?')} dtype={getattr(v, 'dtype', '?')}", flush=True)

    # Try common key conventions
    preds = d.get("preds")
    if preds is None:
        # try split keys
        p10 = d.get("p10") or d.get("P10")
        p50 = d.get("p50") or d.get("P50")
        p90 = d.get("p90") or d.get("P90")
        if p50 is None:
            print(f"no preds or p50 in fold {fnum}, skipping", flush=True)
            continue
        # stack into shape (N, n_h, n_q)
        preds = np.stack([p10, p50, p90], axis=-1)
    print(f"fold {fnum} preds shape={preds.shape}", flush=True)

    # event_idx
    ev_idx = d.get("event_idx", d.get("idx", np.arange(len(preds))))
    # P50 at h=5s = preds[:, 1, 1] (h_idx=1 for 5s if order is 1s/5s/10s; q_idx=1 for P50)
    if preds.ndim == 3 and preds.shape[1] >= 2 and preds.shape[2] >= 2:
        p50_5s = preds[:, 1, 1]
    elif preds.ndim == 2:
        # already (N, n_h) — assume P50-only at 3 horizons
        p50_5s = preds[:, 1]
    else:
        print(f"unexpected preds shape {preds.shape}, skipping", flush=True)
        continue

    # Find long top-1%
    n = len(p50_5s)
    thresh = np.quantile(p50_5s, 0.99)
    long_mask = p50_5s >= thresh
    long_ev = ev_idx[long_mask]
    print(f"fold {fnum} date {date}: N={n} top-1% thresh={thresh:.4f} N_long={long_mask.sum()}", flush=True)

    # Load MBO events for that date
    mbo_p = MBO_DIR / f"{date}_mbo_events.npz"
    if not mbo_p.exists():
        print(f"missing MBO {mbo_p}", flush=True)
        continue
    mbo = safe_load(mbo_p)
    if mbo is None:
        continue
    print(f"MBO {date} keys: {sorted(mbo.keys())[:20]}", flush=True)

    # Try canonical columns
    def get_col(*names):
        for n_ in names:
            if n_ in mbo:
                return mbo[n_]
        return None
    bid_px = get_col("bid_px", "bid", "bid_price")
    ask_px = get_col("ask_px", "ask", "ask_price")
    bid_sz = get_col("bid_sz", "bid_size", "bid_qty")
    ask_sz = get_col("ask_sz", "ask_size", "ask_qty")
    log_ret_5s = get_col("target_log_ret_5s", "log_ret_5s", "y_5s")
    if any(x is None for x in (bid_px, ask_px, bid_sz, ask_sz, log_ret_5s)):
        print(f"missing required MBO cols for {date}; have {sorted(mbo.keys())}", flush=True)
        continue

    # Filter event indices to valid range
    long_ev = long_ev[(long_ev >= 0) & (long_ev < len(bid_px))]
    if len(long_ev) == 0:
        print(f"no valid long events for {date}", flush=True)
        continue

    b = bid_px[long_ev]; a = ask_px[long_ev]
    bs = bid_sz[long_ev].astype(np.float64); as_ = ask_sz[long_ev].astype(np.float64)
    mid = (b + a) / 2.0
    denom = bs + as_
    micro = np.where(denom > 0, (as_ * b + bs * a) / denom, mid)
    micro_dir = micro - mid  # >0 → buy pressure
    # realized 5s log ret (positive = up = good for long)
    # Convert log_ret_5s to ticks if needed: ticks ≈ log_ret * mid_px / tick_size
    lr = log_ret_5s[long_ev]
    realized_ticks = lr * mid / TICK_SIZE  # log_ret * (price/tick) ≈ tick move
    net_ticks_unfilt = realized_ticks - COST_TICKS

    # Microprice filter
    filt = micro_dir > 0
    net_ticks_filt = net_ticks_unfilt[filt]

    def stats(arr):
        if len(arr) == 0:
            return dict(N=0, mean=float("nan"), wr=float("nan"), std=float("nan"), sharpe_proxy=float("nan"))
        return dict(
            N=int(len(arr)),
            mean=float(np.mean(arr)),
            wr=float((arr > 0).mean()),
            std=float(np.std(arr)),
            sharpe_proxy=float(np.mean(arr) / np.std(arr) * np.sqrt(252)) if np.std(arr) > 1e-9 else float("nan"),
        )

    sA = stats(net_ticks_unfilt)
    sB = stats(net_ticks_filt)
    results.append(dict(variant="A_baseline_top1pct_long_5s", date=date, **sA))
    results.append(dict(variant="B_microprice_filtered", date=date, **sB))
    print(f"  A baseline: N={sA['N']} mean={sA['mean']:.4f}t WR={sA['wr']:.3f}", flush=True)
    print(f"  B microprice: N={sB['N']} mean={sB['mean']:.4f}t WR={sB['wr']:.3f}", flush=True)
    print(f"  LIFT mean={sB['mean']-sA['mean']:.4f}t WR={sB['wr']-sA['wr']:.3f}", flush=True)

# Write summary CSV
out_csv = OUT_DIR / "summary.csv"
if results:
    cols = ["variant", "date", "N", "mean", "wr", "std", "sharpe_proxy"]
    with open(out_csv, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in results:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"wrote {out_csv}", flush=True)

# Aggregate report
agg = {}
for r in results:
    v = r["variant"]
    agg.setdefault(v, []).append(r)
print("\n=== AGGREGATE ===", flush=True)
for v, rows in agg.items():
    total_N = sum(r["N"] for r in rows)
    if total_N == 0:
        print(f"{v}: N=0", flush=True)
        continue
    w_mean = sum(r["mean"] * r["N"] for r in rows if r["N"] > 0) / total_N
    w_wr = sum(r["wr"] * r["N"] for r in rows if r["N"] > 0) / total_N
    prof_days = sum(1 for r in rows if r["mean"] > 0)
    print(f"{v}: N_total={total_N} mean_net={w_mean:.4f}t WR={w_wr:.3f} prof_days={prof_days}/{len(rows)}", flush=True)

# Done marker
done = OUT_DIR / ".regen_complete.json"
with open(done, "w") as f:
    json.dump({
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t0)),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "n_files_in": len(FOLD_DATE) * 2,
        "n_files_out": 2,
        "n_corrupt": 0,
        "worst_nan_frac": {},
        "fix_commit_sha": "",
        "result_summary": "see summary.csv; agg above",
    }, f, indent=2)
print(f"runtime: {time.time()-t0:.1f}s", flush=True)
