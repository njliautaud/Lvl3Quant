#!/usr/bin/env python3
"""
ofi_confluence_v1.py — OFI confluence analysis for CNN-Mamba v2 short signals.

Merges pre-computed OFI features (per-event, smart_v3 aligned) with CNN-Mamba v2
OOT predictions (stride-250 windows). Tests whether OFI agreement with short
signal direction improves performance.

OFI features used:
  - ofi_aggressive_{1,5,10}s: net aggressive trade flow (buys - sells)
  - ofi_book_{1,5,10}s: net passive book flow (bid adds - ask adds)
  - trade_signed_flow_{1,5,10}s: alias for aggressive

Queue imbalance: computed from smart_v3 event features where possible.

For short signals: OFI negative = selling pressure = agrees with short direction.

Output: /home/jupiter/Lvl3Quant/output/ofi_confluence_v1/
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MBO_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3")
OFI_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3_ofi_features")
PRED_DIR = Path("/home/jupiter/Lvl3Quant/output/cnn_mamba_v2_bulk_oot_v2")
OUT_DIR = Path("/home/jupiter/Lvl3Quant/output/ofi_confluence_v1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Cost
COMMISSION_TICKS = 0.376

# OFI feature names to test (1s, 5s, 10s windows only per task spec)
OFI_FEATURES = [
    "ofi_aggressive_1s", "ofi_aggressive_5s", "ofi_aggressive_10s",
    "ofi_book_1s", "ofi_book_5s", "ofi_book_10s",
    "trade_signed_flow_1s", "trade_signed_flow_5s", "trade_signed_flow_10s",
]

# Prediction horizons (columns in predictions array)
HORIZONS = ["1s", "5s", "10s"]


def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def find_overlap_dates():
    ofi_dates = {f[:8] for f in os.listdir(OFI_DIR) if f.endswith("_ofi.npz")}
    pred_dates = {f[:8] for f in os.listdir(PRED_DIR) if f.endswith("_predictions.npz")}
    mbo_dates = {f[:8] for f in os.listdir(MBO_DIR) if f.endswith("_mbo_events.npz")}
    return sorted(ofi_dates & pred_dates & mbo_dates)


def load_day(date: str):
    """Load predictions + labels + OFI features for one date, aligned at stride grid."""
    pred_data = np.load(PRED_DIR / f"{date}_predictions.npz", allow_pickle=True)
    ofi_data = np.load(OFI_DIR / f"{date}_ofi.npz", allow_pickle=True)
    mbo_data = np.load(MBO_DIR / f"{date}_mbo_events.npz", allow_pickle=True)

    preds = pred_data["predictions"]  # (N_pred, 3) for 1s/5s/10s
    labels = pred_data["labels"]      # (N_pred, 3) for 1s/5s/10s
    ws = int(pred_data["window_size"])
    stride = int(pred_data["stride"])
    n_pred = preds.shape[0]

    # Event indices for each prediction
    event_idx = ws - 1 + np.arange(n_pred) * stride

    # Verify bounds
    n_events = mbo_data["events"].shape[0]
    valid = event_idx < n_events
    if not valid.all():
        n_pred = int(valid.sum())
        event_idx = event_idx[:n_pred]
        preds = preds[:n_pred]
        labels = labels[:n_pred]

    # Extract OFI features at prediction points
    ofi_vals = {}
    for fn in OFI_FEATURES:
        if fn in ofi_data.files:
            ofi_vals[fn] = ofi_data[fn][event_idx].astype(np.float32)
        else:
            ofi_vals[fn] = np.full(n_pred, np.nan, dtype=np.float32)

    # Compute queue imbalance from smart_v3 events if possible
    # Events have 25 columns. Without column names, we can't reliably extract
    # bid/ask sizes. Instead we use the OFI book features as a proxy for
    # queue imbalance (net bid-side vs ask-side activity).
    # We'll compute a normalized version: ofi_book / (|ofi_book| + epsilon)
    for w in ["1s", "5s", "10s"]:
        book_key = f"ofi_book_{w}"
        if book_key in ofi_vals:
            bk = ofi_vals[book_key]
            ofi_vals[f"queue_imbalance_proxy_{w}"] = bk / (np.abs(bk) + 1.0)

    # Timestamps for the prediction events
    timestamps = mbo_data["timestamps"][event_idx]

    return {
        "date": date,
        "preds": preds,
        "labels": labels,
        "ofi": ofi_vals,
        "timestamps": timestamps,
        "n": n_pred,
    }


def compute_trade_tape_velocity(ofi_vals):
    """Compute trade tape velocity: net signed trades normalized by window size."""
    velocities = {}
    for w, secs in [("1s", 1.0), ("5s", 5.0), ("10s", 10.0)]:
        key = f"trade_signed_flow_{w}"
        if key in ofi_vals:
            velocities[f"tape_velocity_{w}"] = ofi_vals[key] / secs
    return velocities


def analyze_confluence(days: list[dict]):
    """Main confluence analysis: OFI agreement vs disagreement for top short signals."""
    results = []

    for h_idx, horizon in enumerate(HORIZONS):
        log(f"  Analyzing horizon {horizon}...")

        # Collect all predictions and OFI across days
        all_preds = []
        all_labels = []
        all_ofi = {fn: [] for fn in OFI_FEATURES}
        all_velocities = {f"tape_velocity_{w}": [] for w in ["1s", "5s", "10s"]}
        all_proxy = {f"queue_imbalance_proxy_{w}": [] for w in ["1s", "5s", "10s"]}
        all_dates = []

        for day in days:
            p = day["preds"][:, h_idx]
            l = day["labels"][:, h_idx]
            n = day["n"]
            all_preds.append(p)
            all_labels.append(l)
            for fn in OFI_FEATURES:
                all_ofi[fn].append(day["ofi"][fn])
            # Tape velocity
            vel = compute_trade_tape_velocity(day["ofi"])
            for vk in all_velocities:
                if vk in vel:
                    all_velocities[vk].append(vel[vk])
                else:
                    all_velocities[vk].append(np.full(n, np.nan, dtype=np.float32))
            # Queue imbalance proxy
            for pk in all_proxy:
                if pk in day["ofi"]:
                    all_proxy[pk].append(day["ofi"][pk])
                else:
                    all_proxy[pk].append(np.full(n, np.nan, dtype=np.float32))

            all_dates.append(np.full(n, day["date"], dtype=object))

        preds = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)
        dates = np.concatenate(all_dates)
        ofi = {fn: np.concatenate(all_ofi[fn]) for fn in OFI_FEATURES}
        velocities = {k: np.concatenate(all_velocities[k]) for k in all_velocities}
        proxy = {k: np.concatenate(all_proxy[k]) for k in all_proxy}

        # Valid mask
        valid = np.isfinite(preds) & np.isfinite(labels)
        n_valid = valid.sum()
        log(f"    Total valid predictions: {n_valid:,}")

        # Short signals: pred < 0 (model predicts price will drop)
        short_mask = valid & (preds < 0)
        n_short = short_mask.sum()
        log(f"    Short signals: {n_short:,}")

        if n_short < 100:
            continue

        # Top 5% short signals by magnitude (most negative pred)
        short_preds = preds[short_mask]
        thresh_5pct = np.percentile(short_preds, 5)  # 5th percentile = most negative
        top5_in_short = short_preds <= thresh_5pct
        top5_global = short_mask.copy()
        top5_global[short_mask] = top5_in_short
        n_top5 = top5_global.sum()
        log(f"    Top 5% short signals (pred <= {thresh_5pct:.4f}): {n_top5:,}")

        # Top 10% short signals
        thresh_10pct = np.percentile(short_preds, 10)
        top10_in_short = short_preds <= thresh_10pct
        top10_global = short_mask.copy()
        top10_global[short_mask] = top10_in_short
        n_top10 = top10_global.sum()

        # Labels for short side: positive label = price dropped (good for short)
        # labels are in ticks. For shorts, profit = -label (if price drops, label negative, short profits)
        short_pnl = -labels  # for short positions

        # Analyze each OFI feature as confluence gate
        all_features = {}
        all_features.update(ofi)
        all_features.update(velocities)
        all_features.update(proxy)

        for fn, fvals in all_features.items():
            if not np.any(np.isfinite(fvals)):
                continue

            for bucket_name, bucket_mask, n_bucket in [
                ("top_5pct", top5_global, n_top5),
                ("top_10pct", top10_global, n_top10),
            ]:
                bm = bucket_mask.copy()
                bm_finite = bm & np.isfinite(fvals)
                n_bm = bm_finite.sum()
                if n_bm < 50:
                    continue

                fvals_bucket = fvals[bm_finite]
                pnl_bucket = short_pnl[bm_finite]
                dates_bucket = dates[bm_finite]

                # OFI agreeing with short: OFI negative (selling pressure)
                ofi_agrees = fvals_bucket < 0
                ofi_disagrees = fvals_bucket >= 0

                n_agree = ofi_agrees.sum()
                n_disagree = ofi_disagrees.sum()

                if n_agree < 20 or n_disagree < 20:
                    continue

                # PnL metrics for agreeing group
                pnl_agree = pnl_bucket[ofi_agrees] - COMMISSION_TICKS
                pnl_disagree = pnl_bucket[ofi_disagrees] - COMMISSION_TICKS

                avg_agree = float(np.mean(pnl_agree))
                avg_disagree = float(np.mean(pnl_disagree))
                wr_agree = float(np.mean(pnl_agree > 0))
                wr_disagree = float(np.mean(pnl_disagree > 0))

                # PF (profit factor)
                wins_a = pnl_agree[pnl_agree > 0].sum()
                losses_a = -pnl_agree[pnl_agree < 0].sum()
                pf_agree = float(wins_a / max(losses_a, 1e-9))

                wins_d = pnl_disagree[pnl_disagree > 0].sum()
                losses_d = -pnl_disagree[pnl_disagree < 0].sum()
                pf_disagree = float(wins_d / max(losses_d, 1e-9))

                # Per-day stats for agreeing group
                unique_days_a = np.unique(dates_bucket[ofi_agrees])
                day_pnls_a = []
                for ud in unique_days_a:
                    dm = dates_bucket[ofi_agrees] == ud
                    day_pnls_a.append(float(np.mean(pnl_agree[dm])))
                day_pnls_a = np.array(day_pnls_a)
                profitable_days_a = int((day_pnls_a > 0).sum())
                total_days_a = len(day_pnls_a)

                # Sharpe (annualized from daily)
                if len(day_pnls_a) > 1 and np.std(day_pnls_a) > 0:
                    sharpe_a = float(np.mean(day_pnls_a) / np.std(day_pnls_a, ddof=1) * np.sqrt(252))
                else:
                    sharpe_a = 0.0

                # Lift: how much better is agreeing vs disagreeing
                lift = avg_agree - avg_disagree

                results.append({
                    "horizon": horizon,
                    "ofi_feature": fn,
                    "bucket": bucket_name,
                    "n_total": int(n_bm),
                    "n_agree": int(n_agree),
                    "n_disagree": int(n_disagree),
                    "agree_frac": round(n_agree / n_bm, 3),
                    "avg_ticks_agree": round(avg_agree, 4),
                    "avg_ticks_disagree": round(avg_disagree, 4),
                    "lift_ticks": round(lift, 4),
                    "wr_agree": round(wr_agree, 4),
                    "wr_disagree": round(wr_disagree, 4),
                    "pf_agree": round(pf_agree, 3),
                    "pf_disagree": round(pf_disagree, 3),
                    "sharpe_agree": round(sharpe_a, 3),
                    "profitable_days_agree": profitable_days_a,
                    "total_days_agree": total_days_a,
                })

    return pd.DataFrame(results)


def main():
    t0 = time.time()
    log("OFI Confluence v1 — starting")

    dates = find_overlap_dates()
    log(f"Found {len(dates)} overlapping dates (OFI + predictions + MBO)")

    if len(dates) < 5:
        log("FATAL: not enough overlapping dates")
        sys.exit(1)

    # Load all days
    days = []
    for d in dates:
        try:
            day = load_day(d)
            days.append(day)
            log(f"  Loaded {d}: {day['n']:,} predictions")
        except Exception as e:
            log(f"  FAILED {d}: {e}")

    log(f"Loaded {len(days)} days, total predictions: {sum(d['n'] for d in days):,}")

    # Run confluence analysis
    log("Running confluence analysis...")
    df = analyze_confluence(days)

    # Save full results
    csv_path = OUT_DIR / "confluence_results.csv"
    df.to_csv(csv_path, index=False)
    log(f"Saved {len(df)} rows to {csv_path.name}")

    # Summary: best OFI features by lift
    log("\n" + "=" * 80)
    log("CONFLUENCE RESULTS SUMMARY")
    log("=" * 80)

    if len(df) == 0:
        log("No results - insufficient data")
        return

    # Sort by lift
    df_sorted = df.sort_values("lift_ticks", ascending=False)

    # Top results
    log("\nTop 10 OFI features by lift (agree - disagree):")
    for _, row in df_sorted.head(10).iterrows():
        log(f"  {row['ofi_feature']:30s} {row['horizon']:3s} {row['bucket']:10s} "
            f"lift={row['lift_ticks']:+.4f}t  "
            f"agree={row['avg_ticks_agree']:+.4f}t WR={row['wr_agree']:.1%} PF={row['pf_agree']:.2f}  "
            f"disagree={row['avg_ticks_disagree']:+.4f}t WR={row['wr_disagree']:.1%}")

    # Check if any agree group is net positive after costs
    profitable = df[df["avg_ticks_agree"] > 0]
    log(f"\nOFI-agree groups that are NET POSITIVE after costs: {len(profitable)}/{len(df)}")
    if len(profitable) > 0:
        best = profitable.sort_values("avg_ticks_agree", ascending=False).head(5)
        for _, row in best.iterrows():
            log(f"  {row['ofi_feature']:30s} {row['horizon']:3s} {row['bucket']:10s} "
                f"net={row['avg_ticks_agree']:+.4f}t WR={row['wr_agree']:.1%} PF={row['pf_agree']:.2f} "
                f"Sharpe={row['sharpe_agree']:.2f} "
                f"profdays={row['profitable_days_agree']}/{row['total_days_agree']}")

    # Worst (most negative lift = OFI hurts)
    log("\nWorst 5 (OFI agreement HURTS):")
    for _, row in df_sorted.tail(5).iterrows():
        log(f"  {row['ofi_feature']:30s} {row['horizon']:3s} {row['bucket']:10s} "
            f"lift={row['lift_ticks']:+.4f}t")

    # Aggregate: average lift across all features per horizon
    log("\nAverage lift by horizon:")
    for h in HORIZONS:
        hdf = df[df["horizon"] == h]
        if len(hdf) > 0:
            log(f"  {h}: mean_lift={hdf['lift_ticks'].mean():+.4f}t "
                f"mean_agree_net={hdf['avg_ticks_agree'].mean():+.4f}t "
                f"mean_disagree_net={hdf['avg_ticks_disagree'].mean():+.4f}t")

    # Save summary JSON
    summary = {
        "task": "ofi_confluence_v1",
        "dates_used": len(days),
        "total_predictions": int(sum(d["n"] for d in days)),
        "n_results": len(df),
        "n_profitable_agree": int(len(profitable)),
        "best_lift_ticks": float(df_sorted["lift_ticks"].iloc[0]) if len(df_sorted) > 0 else 0,
        "best_feature": str(df_sorted["ofi_feature"].iloc[0]) if len(df_sorted) > 0 else "none",
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    if len(profitable) > 0:
        best_row = profitable.sort_values("avg_ticks_agree", ascending=False).iloc[0]
        summary["best_profitable"] = {
            "feature": str(best_row["ofi_feature"]),
            "horizon": str(best_row["horizon"]),
            "bucket": str(best_row["bucket"]),
            "net_ticks": float(best_row["avg_ticks_agree"]),
            "wr": float(best_row["wr_agree"]),
            "pf": float(best_row["pf_agree"]),
            "sharpe": float(best_row["sharpe_agree"]),
        }

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
