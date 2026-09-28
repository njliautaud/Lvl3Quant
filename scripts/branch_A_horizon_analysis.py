#!/usr/bin/env python3
"""
Branch A: Does the wide_ofi_pos_vol_q3_flow_pos confluence bucket carry edge
at other prediction horizons (1s, 10s, 30s) besides the verified 5s?

Data:
  - OOT preds: output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate/oot_YYYYMMDD.npz
    Fields: pred_log_ret_{1s,5s,10s,30s}, target_log_ret_{1s,5s,10s,30s}
  - OFI features: data/processed/mbo_events_smart_v3_ofi_features/YYYYMMDD_ofi.npz
    Fields: ofi_book_{1s,5s,10s,30s}, trade_signed_flow_{1s,5s,10s,30s},
            ofi_aggressive_*, spread_ticks_now

Bucketing (wide_ofi_pos_vol_q3_flow_pos):
  - wide: |ofi_book_1s| >= per-day median (top-50% absolute OFI magnitude = "wide imbalance")
  - ofi_pos: ofi_book_1s > 0
  - vol_q3: |ofi_book_30s| in 3rd quartile [q50, q75) per-day
  - flow_pos: trade_signed_flow_1s > 0

Selection:
  - Top-1% SHORT per horizon H: bottom 1st percentile of pred_log_ret_H per day

Cost: 0.376 ticks (passive both ways)

For SHORT side: net_ticks = -target_log_ret_H - 0.376

Regime: daily close-to-close sign inferred from mean(target_log_ret_30s) per day sign
"""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/jupiter/Lvl3Quant")
OOT_DIR = ROOT / "output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate"
OFI_DIR = ROOT / "data/processed/mbo_events_smart_v3_ofi_features"
OUT_DIR = ROOT / "output/branch_A_horizon"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OFFSET = 1499
STRIDE = 250
COMM = 0.376
HORIZONS = ["1s", "5s", "10s", "30s"]


def get_valid_dates():
    oot_dates = {p.stem.replace("oot_", "") for p in OOT_DIR.glob("oot_*.npz")}
    ofi_dates = {p.name.replace("_ofi.npz", "") for p in OFI_DIR.glob("*_ofi.npz")}
    overlap = sorted(oot_dates & ofi_dates)
    valid = []
    for date in overlap:
        try:
            d = np.load(OOT_DIR / f"oot_{date}.npz")
            if "pred_log_ret_5s" in d.files and d["pred_log_ret_5s"].shape[0] > 100:
                valid.append(date)
        except Exception:
            pass
    return valid


def load_day(date: str):
    """Load and align OOT predictions with OFI features at stride-250 grid."""
    oot_path = OOT_DIR / f"oot_{date}.npz"
    ofi_path = OFI_DIR / f"{date}_ofi.npz"
    if not (oot_path.exists() and ofi_path.exists()):
        return None

    oot = np.load(oot_path)
    ofi = np.load(ofi_path)

    n_pred = oot["pred_log_ret_5s"].shape[0]
    n_ofi = ofi["spread_ticks_now"].shape[0]
    v4_idx = OFFSET + np.arange(n_pred) * STRIDE
    cap = v4_idx < n_ofi
    v4_idx = v4_idx[cap]
    n = len(v4_idx)

    out = {"date": date, "n": n}
    # Predictions and targets for each horizon
    for h in HORIZONS:
        pk = f"pred_log_ret_{h}"
        tk = f"target_log_ret_{h}"
        if pk in oot.files and tk in oot.files:
            out[f"pred_{h}"] = oot[pk][:n].astype(np.float64)
            out[f"target_{h}"] = oot[tk][:n].astype(np.float64)
        else:
            out[f"pred_{h}"] = np.full(n, np.nan)
            out[f"target_{h}"] = np.full(n, np.nan)

    # OFI features at stride grid
    out["ofi_book_1s"] = ofi["ofi_book_1s"][v4_idx].astype(np.float64)
    out["trade_signed_flow_1s"] = ofi["trade_signed_flow_1s"][v4_idx].astype(np.float64)
    out["ofi_book_30s"] = ofi["ofi_book_30s"][v4_idx].astype(np.float64)

    return out


def apply_bucket(day: dict):
    """
    Apply the wide_ofi_pos_vol_q3_flow_pos bucket to the loaded day data.
    Returns a boolean mask.

    Steps (print N remaining at each step for pre-flight trace):
      1. wide: |ofi_book_1s| >= per-day median
      2. ofi_pos: ofi_book_1s > 0
      3. vol_q3: |ofi_book_30s| in [q50, q75) per-day
      4. flow_pos: trade_signed_flow_1s > 0
    """
    n = day["n"]
    ofi1 = day["ofi_book_1s"]
    tsf1 = day["trade_signed_flow_1s"]
    ofi30 = np.abs(day["ofi_book_30s"])

    # Step 1: wide = |ofi1| >= median
    ofi1_abs = np.abs(ofi1)
    med_ofi1 = np.median(ofi1_abs)
    wide_mask = ofi1_abs >= med_ofi1

    # Step 2: ofi_pos
    ofi_pos_mask = ofi1 > 0

    # Step 3: vol_q3 = |ofi30| in [q50, q75)
    q50_30, q75_30 = np.percentile(ofi30, [50, 75])
    vol_q3_mask = (ofi30 >= q50_30) & (ofi30 < q75_30)

    # Step 4: flow_pos
    flow_pos_mask = tsf1 > 0

    # Combined bucket
    bucket_mask = wide_mask & ofi_pos_mask & vol_q3_mask & flow_pos_mask
    return bucket_mask, {
        "n_wide": int(wide_mask.sum()),
        "n_ofi_pos": int((wide_mask & ofi_pos_mask).sum()),
        "n_vol_q3": int((wide_mask & ofi_pos_mask & vol_q3_mask).sum()),
        "n_bucket": int(bucket_mask.sum()),
    }


def regime_from_day(day: dict) -> str:
    """Estimate daily regime from mean of 30s returns: positive=green, negative=red."""
    t30 = day["target_30s"]
    valid = t30[np.isfinite(t30)]
    if len(valid) == 0:
        return "flat"
    m = float(np.mean(valid))
    if m > 0.05:
        return "green"
    elif m < -0.05:
        return "red"
    return "flat"


def analyze_horizon(days: list, horizon: str):
    """
    For each day: select top-1% SHORT by pred_log_ret_H, intersect with bucket,
    compute net ticks = -target - 0.376.
    Return per-trade list of dicts.
    """
    pred_key = f"pred_{horizon}"
    tgt_key = f"target_{horizon}"

    trades = []
    for day in days:
        p = day[pred_key]
        t = day[tgt_key]
        bucket_mask = day["bucket_mask"]
        date = day["date"]
        regime = day["regime"]

        # Valid = finite target
        valid = np.isfinite(t) & np.isfinite(p)

        # Per-day top-1% short by pred (most negative predictions = strongest short signal)
        k = max(1, int(0.01 * valid.sum()))
        valid_preds = p[valid]
        if len(valid_preds) < 10:
            continue
        thr = np.partition(valid_preds, k - 1)[k - 1]  # k-th smallest
        top1_short = valid & (p <= thr)

        # Intersect with bucket
        sel = top1_short & bucket_mask

        if sel.sum() == 0:
            continue

        tgt_sel = t[sel]
        net = -tgt_sel - COMM
        for val, nt in zip(tgt_sel, net):
            trades.append({
                "date": date,
                "regime": regime,
                "horizon": horizon,
                "target_ticks": float(val),
                "net_ticks": float(nt),
                "win": int(nt > 0),
            })
    return trades


def compute_horizon_stats(trades: list, horizon: str):
    if not trades:
        return None
    arr = pd.DataFrame(trades)
    net = arr["net_ticks"].values
    dates = arr["date"].values

    n_trades = len(net)
    mean_net = float(np.mean(net))
    wr = float(np.mean(net > 0))

    # Regime split
    green = arr[arr["regime"] == "green"]["net_ticks"].values
    red = arr[arr["regime"] == "red"]["net_ticks"].values
    green_net = float(np.mean(green)) if len(green) > 0 else float("nan")
    red_net = float(np.mean(red)) if len(red) > 0 else float("nan")
    n_green = len(green)
    n_red = len(red)

    # Per-date stats for Sharpe
    unique_dates = sorted(set(dates))
    n_days = len(unique_dates)

    return {
        "horizon": horizon,
        "n_trades": n_trades,
        "n_days": n_days,
        "mean_net_ticks": mean_net,
        "win_rate": wr,
        "green_net": green_net,
        "red_net": red_net,
        "n_green_trades": n_green,
        "n_red_trades": n_red,
        "trades": trades,
    }


def main():
    t0 = time.time()
    print("=" * 70, flush=True)
    print("Branch A: Horizon Generalizability of wide_ofi_pos_vol_q3_flow_pos", flush=True)
    print("=" * 70, flush=True)

    # --- PRE-FLIGHT: Directory listing ---
    print("\n[PRE-FLIGHT 1] Directory listings:", flush=True)
    oot_files = sorted(OOT_DIR.glob("oot_*.npz"))
    ofi_files = sorted(OFI_DIR.glob("*_ofi.npz"))
    print(f"OOT dir: {len(oot_files)} files")
    for f in oot_files[:3]:
        import os
        st = os.stat(f)
        print(f"  {f.name}: {st.st_size:,} bytes, mtime {time.strftime('%Y-%m-%d %H:%M', time.localtime(st.st_mtime))}")
    print(f"OFI dir: {len(ofi_files)} files")
    for f in ofi_files[:3]:
        st = os.stat(f)
        print(f"  {f.name}: {st.st_size:,} bytes, mtime {time.strftime('%Y-%m-%d %H:%M', time.localtime(st.st_mtime))}")

    # --- PRE-FLIGHT 2: Inspect one OOT and one OFI file ---
    print("\n[PRE-FLIGHT 2] Sample file inspection:", flush=True)
    sample_oot = np.load(OOT_DIR / "oot_20260302.npz")
    print(f"OOT file (20260302): {len(sample_oot.files)} arrays")
    for k in ["pred_log_ret_1s", "pred_log_ret_5s", "pred_log_ret_10s", "pred_log_ret_30s",
              "target_log_ret_1s", "target_log_ret_5s", "target_log_ret_10s", "target_log_ret_30s"]:
        if k in sample_oot.files:
            v = sample_oot[k]
            nz = v[np.isfinite(v)]
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}, first3={v[:3].tolist()}")
        else:
            print(f"  {k}: MISSING ← ABORT RISK")

    sample_ofi = np.load(OFI_DIR / "20260302_ofi.npz")
    print(f"\nOFI file (20260302): {len(sample_ofi.files)} arrays")
    for k in ["ofi_book_1s", "trade_signed_flow_1s", "ofi_book_30s", "spread_ticks_now"]:
        if k in sample_ofi.files:
            v = sample_ofi[k]
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}, first3={v[:3].tolist()}")
        else:
            print(f"  {k}: MISSING ← ABORT")
            raise RuntimeError(f"Required OFI field missing: {k}")

    # --- Get valid dates ---
    dates = get_valid_dates()
    print(f"\n[DATA] Valid dates with both OOT+OFI: {len(dates)}", flush=True)
    print(f"  {dates[0]} → {dates[-1]}", flush=True)

    # --- Load all days ---
    print("\n[LOAD] Loading and aligning all days...", flush=True)
    days = []
    for date in dates:
        day = load_day(date)
        if day is None:
            print(f"  SKIP {date}: load failed", flush=True)
            continue
        # Apply bucket filter (regardless of horizon)
        bucket_mask, bucket_steps = apply_bucket(day)
        day["bucket_mask"] = bucket_mask

        # Add regime from 30s target
        day["target_30s"] = day["target_30s"] if "target_30s" in day else day.get("target_30s", np.full(day["n"], np.nan))
        day["regime"] = regime_from_day(day)

        days.append(day)

    print(f"  Loaded {len(days)} days", flush=True)

    # --- PRE-FLIGHT 3: Bucket filter trace on sample day ---
    print("\n[PRE-FLIGHT 3] Bucket filter trace on 20260302:", flush=True)
    sample_day = next((d for d in days if d["date"] == "20260302"), days[0])
    _, bsteps = apply_bucket(sample_day)
    n_total = sample_day["n"]
    print(f"  Total events: {n_total}")
    print(f"  After wide (|ofi1| >= median): N={bsteps['n_wide']} (was {n_total})")
    print(f"  After ofi_pos (ofi1 > 0): N={bsteps['n_ofi_pos']} (was {bsteps['n_wide']})")
    print(f"  After vol_q3 (|ofi30| in [q50,q75)): N={bsteps['n_vol_q3']} (was {bsteps['n_ofi_pos']})")
    print(f"  After flow_pos (tsf1 > 0): N={bsteps['n_bucket']} (was {bsteps['n_vol_q3']})")
    if bsteps["n_bucket"] == 0:
        print("  ABORT: bucket is empty on sample day — bucketing is wrong")
        raise RuntimeError("Bucket empty on sample day")

    # --- Check bucket non-empty across all days ---
    total_bucket = sum(d["bucket_mask"].sum() for d in days)
    print(f"\n  Total bucket events across all {len(days)} days: {total_bucket}", flush=True)

    # --- STEP 1: Reproduce 5s baseline ---
    print("\n" + "=" * 50, flush=True)
    print("[STEP 1] Reproducing 5s baseline (must hit +0.768 ± 0.15)", flush=True)
    print("=" * 50, flush=True)

    trades_5s = analyze_horizon(days, "5s")
    stats_5s = compute_horizon_stats(trades_5s, "5s")

    if stats_5s:
        repro_net = stats_5s["mean_net_ticks"]
        repro_n = stats_5s["n_trades"]
        repro_wr = stats_5s["win_rate"]
        within_tolerance = abs(repro_net - 0.768) <= 0.15
        print(f"  5s bucket: n_trades={repro_n}, net={repro_net:+.4f}, WR={repro_wr*100:.1f}%")
        print(f"  Target: +0.768 ± 0.15")
        print(f"  Diff from target: {repro_net - 0.768:+.4f}")
        print(f"  REPRODUCTION CHECK: {'PASS' if within_tolerance else 'FAIL'}", flush=True)

        if not within_tolerance:
            print(f"\n  WARN: {repro_net:+.4f} is outside ±0.15 of +0.768.")
            print(f"  This means our bucketing differs from the prior run.")
            print(f"  Proceeding with our best-match bucketing for horizon comparison.")
    else:
        print("  ERROR: No 5s bucket trades found", flush=True)
        repro_net = float("nan")
        within_tolerance = False

    # --- STEP 2: All horizons ---
    print("\n" + "=" * 50, flush=True)
    print("[STEP 2] Analyzing all horizons with identical bucketing", flush=True)
    print("=" * 50, flush=True)

    all_results = {}
    all_trades = []

    for h in HORIZONS:
        print(f"\n--- Horizon {h} ---", flush=True)
        trades_h = analyze_horizon(days, h)
        stats_h = compute_horizon_stats(trades_h, h)
        if stats_h:
            all_results[h] = stats_h
            all_trades.extend(trades_h)
            print(f"  n_trades={stats_h['n_trades']}, n_days={stats_h['n_days']}")
            print(f"  net_ticks={stats_h['mean_net_ticks']:+.4f}, WR={stats_h['win_rate']*100:.1f}%")
            print(f"  green_net={stats_h['green_net']:+.4f} (n={stats_h['n_green_trades']}), "
                  f"red_net={stats_h['red_net']:+.4f} (n={stats_h['n_red_trades']})")
        else:
            print(f"  No trades found for {h}", flush=True)

    # --- Build summary table ---
    print("\n" + "=" * 70, flush=True)
    print("RESULTS TABLE", flush=True)
    print("=" * 70, flush=True)
    print(f"{'horizon':<10} {'n_trades':<10} {'net_ticks':<12} {'WR':>6} {'green_net':<12} {'red_net':<12}", flush=True)
    print("-" * 70, flush=True)
    for h in HORIZONS:
        if h in all_results:
            r = all_results[h]
            print(f"{h:<10} {r['n_trades']:<10} {r['mean_net_ticks']:>+10.4f}   "
                  f"{r['win_rate']*100:>5.1f}%  {r['green_net']:>+10.4f}   {r['red_net']:>+10.4f}")
        else:
            print(f"{h:<10} {'N/A':<10}")

    # --- Specific data points ---
    print("\n[SPECIFIC DATA POINTS]", flush=True)
    for h in HORIZONS:
        if h not in all_results:
            continue
        trades_df = pd.DataFrame(all_results[h]["trades"])
        # Pick 3 dates with non-trivial trades
        date_counts = trades_df.groupby("date").size().sort_values(ascending=False)
        dates_with_trades = date_counts.index[:5]
        print(f"\n  Horizon {h} — sample dates:")
        for d in dates_with_trades[:3]:
            sub = trades_df[trades_df.date == d]
            print(f"    {d}: n={len(sub)}, regime={sub.iloc[0].regime}, net={sub.net_ticks.mean():+.4f}")

    # --- Verdict ---
    print("\n[VERDICT]", flush=True)
    positive_horizons = [h for h in HORIZONS if h in all_results and all_results[h]["mean_net_ticks"] > 0]
    n_positive = len(positive_horizons)
    print(f"  Positive net ticks at {n_positive}/4 horizons: {positive_horizons}", flush=True)
    if n_positive >= 3:
        print("  VERDICT: Edge persists across horizons (≥3/4 positive) — NOT 5s-specific overfit", flush=True)
    elif n_positive == 1 or n_positive == 2:
        print(f"  VERDICT: Edge partially generalizes ({n_positive}/4) — MIXED SIGNAL", flush=True)
    else:
        print("  VERDICT: Edge does NOT persist — 5s bucket may be overfit or 5s-specific", flush=True)

    # --- Save outputs ---
    print("\n[SAVE] Writing outputs...", flush=True)

    # summary.csv
    rows = []
    for h in HORIZONS:
        if h in all_results:
            r = all_results[h]
            rows.append({
                "horizon": h,
                "n_trades": r["n_trades"],
                "n_days": r["n_days"],
                "net_ticks": round(r["mean_net_ticks"], 4),
                "win_rate_pct": round(r["win_rate"] * 100, 1),
                "green_net_ticks": round(r["green_net"], 4) if not np.isnan(r["green_net"]) else "nan",
                "red_net_ticks": round(r["red_net"], 4) if not np.isnan(r["red_net"]) else "nan",
                "n_green_trades": r["n_green_trades"],
                "n_red_trades": r["n_red_trades"],
            })
        else:
            rows.append({"horizon": h, "n_trades": 0, "n_days": 0,
                         "net_ticks": "nan", "win_rate_pct": "nan",
                         "green_net_ticks": "nan", "red_net_ticks": "nan",
                         "n_green_trades": 0, "n_red_trades": 0})
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(OUT_DIR / "summary.csv", index=False)
    print(f"  Wrote summary.csv", flush=True)

    # headline.txt
    headline_lines = [
        f"Branch A — Horizon Generalizability of wide_ofi_pos_vol_q3_flow_pos bucket",
        f"CNN-Mamba v3.4.2 SHORT top-1% | OOT dates: {len(days)} | Cost: 0.376t passive",
        "",
        f"5s REPRODUCTION CHECK: target +0.768, got {repro_net:+.4f} ({'PASS' if within_tolerance else 'FAIL'} ±0.15 tolerance)",
        "",
        "horizon | n_trades | net_ticks | WR% | green_net | red_net",
        "-" * 65,
    ]
    for h in HORIZONS:
        if h in all_results:
            r = all_results[h]
            headline_lines.append(
                f"{h:<8} | {r['n_trades']:<8} | {r['mean_net_ticks']:>+9.4f} | "
                f"{r['win_rate']*100:>5.1f} | {r['green_net']:>+9.4f} | {r['red_net']:>+9.4f}"
            )
    headline_lines.append("")
    headline_lines.append(f"Positive horizons: {n_positive}/4 — {positive_horizons}")
    if n_positive >= 3:
        headline_lines.append("VERDICT: Edge generalizes across horizons (not 5s-specific).")
    elif n_positive >= 2:
        headline_lines.append("VERDICT: Partial generalization — mixed signal.")
    else:
        headline_lines.append("VERDICT: Edge does NOT generalize — 5s-specific or noise.")
    headline_lines.append("")
    headline_lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M ET')}")

    (OUT_DIR / "headline.txt").write_text("\n".join(headline_lines))
    print(f"  Wrote headline.txt", flush=True)

    print(f"\n[DONE] Elapsed: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
