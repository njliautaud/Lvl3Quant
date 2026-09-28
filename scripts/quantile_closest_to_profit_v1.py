#!/usr/bin/env python3
"""
quantile_closest_to_profit_v1.py — HC #489 BASE QUANTILE ALPHA SURVIVAL TEST

Adapter for DLinear quantile predictions (P10/P50/P90 at 1s/5s/10s horizons).
Uses realized signed moves as proxy MFE/MAE (same as closest_to_profit_v4 for 1s/5s/10s).

Confidence ordering: P50 (median prediction).
  - Long: bucket by highest P50 (most bullish)
  - Short: bucket by lowest P50 (most bearish)

Net ticks = realized_move - 0.376 (passive limit cost per HC cost canon).

DEPLOY GATES (HC #428 R1 + R2):
  - net_ticks_per_event > +0.10
  - win_rate >= 0.52
  - BOTH days profitable (2/2, not 1/2)
  - |Sharpe_green - Sharpe_red| / max(|.,.|) <= 0.50

Data source: /home/jupiter/Lvl3Quant/data/razer_pull/hc489_dlinear_quantile_asym_long_v1/
  - fold_01: 2026-04-27 (green, WR=57.78%)
  - fold_02: 2026-04-28 (red, WR=48.64%)

Output: /home/jupiter/Lvl3Quant/output/quantile_closest_to_profit_v1/
  - summary.csv
  - per_day.csv
  - findings.md
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------- constants ----------------------------------
COMMISSION_TICKS = 0.376  # passive fill round-trip (HC)
HORIZONS = ["1s", "5s", "10s"]  # quantile preds only have these
SIDES = ["long", "short"]
BUCKETS = {
    "top_1pct":   (0.99, 1.00),
    "top_2pct":   (0.98, 1.00),
    "top_5pct":   (0.95, 1.00),
    "top_10pct":  (0.90, 1.00),
    "top_20pct":  (0.80, 1.00),
    "bottom_50pct": (0.00, 0.50),
}

DATA_DIR = Path("/home/jupiter/Lvl3Quant/data/razer_pull/hc489_dlinear_quantile_asym_long_v1")
OUT_DIR = Path("/home/jupiter/Lvl3Quant/output/quantile_closest_to_profit_v1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------- data load ----------------------------------
def load_quantile_fold(npz_path: Path):
    """Load quantile DLinear predictions from NPZ.

    Returns dict with:
      - date: YYYYMMDD
      - labels (N,3): [1s, 5s, 10s] realized moves in ticks (signed)
      - preds (N,3,3): [1s, 5s, 10s] x [P10, P50, P90] quantile predictions
      - P10/P50/P90_{1s,5s,10s}: (N,) individual quantile arrays
      - mask: boolean array indicating valid (non-outlier) events
    """
    d = np.load(npz_path, allow_pickle=True)
    out = {}
    out["date"] = d["date"].item() if d["date"].shape == () else str(d["date"])

    labels_raw = d["labels"].astype(np.float32)  # (N, 3)

    # Filter: keep only rows where all 3 labels are reasonable (abs < 50 ticks)
    # Outliers (val=764) are data corruption/sentinel values
    valid_mask = np.abs(labels_raw).max(axis=1) < 50
    n_dropped = (~valid_mask).sum()

    out["labels"] = labels_raw[valid_mask]
    out["preds"] = d["preds"].astype(np.float32)[valid_mask]
    out["mask"] = valid_mask
    out["n_dropped"] = int(n_dropped)

    # Store individual quantile arrays (filtered)
    for h_idx, h in enumerate(["1s", "5s", "10s"]):
        out[f"P10_{h}"] = d[f"P10_{h}"].astype(np.float32)[valid_mask]
        out[f"P50_{h}"] = d[f"P50_{h}"].astype(np.float32)[valid_mask]
        out[f"P90_{h}"] = d[f"P90_{h}"].astype(np.float32)[valid_mask]

    return out

# ----------------------- helper functions --------------------------
def realized_mfe_mae_for_side(fold_data: dict, h: str, side: str):
    """
    Returns (mfe_ticks, mae_ticks, signed_move_ticks) for the given side.
    Uses realized signed move as proxy MFE/MAE (same as v4 for 1s/5s/10s).

    For long: realized_move>0 = favorable MFE, <0 = adverse MAE
    For short: realized_move>0 = adverse MAE, <0 = favorable MFE
    """
    h_idx = {"1s": 0, "5s": 1, "10s": 2}[h]
    signed = fold_data["labels"][:, h_idx]  # (N,) realized moves in ticks

    if side == "short":
        signed = -signed  # flip for short

    # Proxy: mfe = max(0, signed); mae = max(0, -signed)
    mfe_proxy = np.where(signed > 0, signed, 0.0).astype(np.float32)
    mae_proxy = np.where(signed < 0, -signed, 0.0).astype(np.float32)

    return mfe_proxy, mae_proxy, signed.astype(np.float32)

def get_confidence_pred(fold_data: dict, h: str, side: str):
    """
    For long side: return P50 (higher = more bullish = higher confidence long)
    For short side: return -P50 (lower P50 = more bearish, so negate to rank by |pred|)

    Returns: (confidence_score, directional_pred)
    confidence_score is used for bucketing |pred|; directional_pred is pred before |.|
    """
    p50 = fold_data[f"P50_{h}"]
    if side == "long":
        return np.abs(p50), p50
    else:
        return np.abs(p50), -p50  # flip for short: bearish = low p50, so negate

def bucket_indices(abs_pred: np.ndarray, lo_pct: float, hi_pct: float):
    """Return boolean mask selecting items whose |pred| percentile falls in [lo, hi)."""
    if len(abs_pred) == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(abs_pred)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(order)) / max(1, len(order) - 1)
    return (ranks >= lo_pct) & (ranks <= hi_pct)

def cell_stats(mfe, mae, signed, bucket_mask):
    """Compute stats for a bucket."""
    sel = bucket_mask & ~np.isnan(mfe) & ~np.isnan(mae) & ~np.isnan(signed)
    n = int(sel.sum())
    if n == 0:
        return None

    mfe_s = mfe[sel]
    mae_s = mae[sel]
    signed_s = signed[sel]

    # net ticks per event = realized move - commission
    net_ticks = signed_s - COMMISSION_TICKS

    win_rate = float(np.mean(mfe_s > mae_s))
    mfe_p50 = float(np.percentile(mfe_s, 50))
    mae_p50 = float(np.percentile(mae_s, 50))

    return dict(
        n=n,
        mfe_p50=mfe_p50,
        mae_p50=mae_p50,
        net_ticks_per_event=float(np.nanmean(net_ticks)),
        net_ticks_median=float(np.nanmedian(net_ticks)),
        signed_move_mean=float(np.mean(signed_s)),
        win_rate=win_rate,
    )

def sharpe(net_ticks_array):
    """Annualized Sharpe (252 trading days)."""
    if len(net_ticks_array) < 2:
        return 0.0
    s = np.std(net_ticks_array, ddof=1)
    if s == 0:
        return 0.0
    return float(np.mean(net_ticks_array) / s * np.sqrt(252))

# ----------------------------- main ---------------------------------------
def main():
    t0 = time.time()
    print(f"[start] {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    # Load folds
    folds_paths = sorted(DATA_DIR.glob("fold_*.npz"))
    print(f"[load] found {len(folds_paths)} fold files in {DATA_DIR}", flush=True)

    if not folds_paths:
        print("[fatal] no fold files found; aborting.", file=sys.stderr)
        sys.exit(2)

    folds = []
    for fpath in folds_paths:
        try:
            fold = load_quantile_fold(fpath)
            folds.append(fold)
            n_dropped = fold["n_dropped"]
            print(f"  loaded {fpath.name}: date={fold['date']}, N={fold['labels'].shape[0]:,} (dropped {n_dropped} outliers)", flush=True)
        except Exception as e:
            print(f"  FAILED {fpath.name}: {e}", file=sys.stderr)

    total_events = sum(f["labels"].shape[0] for f in folds)
    print(f"[load] total events across all folds: {total_events:,}", flush=True)

    summary_rows = []
    per_day_rows = []

    for h in HORIZONS:
        for side in SIDES:
            print(f"\n[analyze] h={h} side={side}", flush=True)

            # Collect data across folds
            all_conf_pred = []
            all_dir_pred = []
            all_mfe = []
            all_mae = []
            all_signed = []
            all_date = []

            for fold in folds:
                conf_pred, dir_pred = get_confidence_pred(fold, h, side)
                mfe, mae, signed = realized_mfe_mae_for_side(fold, h, side)

                # Directional filter: only consider side-aligned predictions
                if side == "long":
                    side_mask = dir_pred > 0
                else:
                    side_mask = dir_pred < 0

                all_conf_pred.append(conf_pred)
                all_dir_pred.append(dir_pred)
                all_mfe.append(mfe)
                all_mae.append(mae)
                all_signed.append(signed)
                all_date.append(np.full(len(dir_pred), fold["date"], dtype=object))

            conf_pred_all = np.concatenate(all_conf_pred)
            dir_pred_all = np.concatenate(all_dir_pred)
            mfe_all = np.concatenate(all_mfe)
            mae_all = np.concatenate(all_mae)
            signed_all = np.concatenate(all_signed)
            date_all = np.concatenate(all_date)

            # Base mask: directional alignment
            side_mask_all = (dir_pred_all > 0) if side == "long" else (dir_pred_all < 0)
            base_mask = side_mask_all & ~np.isnan(conf_pred_all) & ~np.isnan(signed_all)

            n_base = int(base_mask.sum())
            if n_base == 0:
                print(f"  WARNING: 0 valid events for h={h} side={side}")
                continue

            print(f"  base mask: {n_base:,} events", flush=True)

            # Rank over base mask
            masked_idx = np.where(base_mask)[0]
            order = np.argsort(conf_pred_all[masked_idx])
            ranks_dense = np.empty(len(masked_idx), dtype=np.float64)
            ranks_dense[order] = np.arange(len(masked_idx)) / max(1, len(masked_idx) - 1)
            ranks_full = np.full(len(conf_pred_all), -1.0, dtype=np.float64)
            ranks_full[masked_idx] = ranks_dense

            # Buckets
            for bname, (lo, hi) in BUCKETS.items():
                bucket_mask = (ranks_full >= lo) & (ranks_full <= hi) & base_mask
                stats = cell_stats(mfe_all, mae_all, signed_all, bucket_mask)

                if stats is None:
                    continue

                # Per-day breakdown
                unique_dates = np.unique(date_all[bucket_mask])
                day_net_means = []
                day_regimes = {}
                profitable_days = 0

                for ud in unique_dates:
                    day_mask = bucket_mask & (date_all == ud) & ~np.isnan(signed_all)
                    if day_mask.sum() == 0:
                        continue

                    day_net = signed_all[day_mask] - COMMISSION_TICKS
                    mean_net = float(np.nanmean(day_net))
                    day_net_means.append(mean_net)

                    # Infer regime from realized 30s direction (proxy)
                    # For simplicity: if mean_net > 0 on that day, it's "profitable"
                    # Regime: green = bullish day, red = bearish day (based on day date)
                    # 4/27 = green, 4/28 = red per context
                    if ud == "20260427":
                        regime = "green"
                    elif ud == "20260428":
                        regime = "red"
                    else:
                        regime = "flat"

                    day_regimes[ud] = regime

                    if mean_net > 0:
                        profitable_days += 1

                    per_day_rows.append(dict(
                        horizon=h, side=side, bucket=bname, date=ud,
                        regime=regime, n=int(day_mask.sum()),
                        mean_net_ticks=mean_net,
                        mean_mfe=float(np.mean(mfe_all[day_mask])),
                        mean_mae=float(np.mean(mae_all[day_mask])),
                    ))

                # Sharpe per regime
                day_net_arr = np.array(day_net_means)
                regimes_list = np.array([day_regimes.get(d, "flat") for d in unique_dates])

                s_green = sharpe(day_net_arr[regimes_list == "green"])
                s_red = sharpe(day_net_arr[regimes_list == "red"])
                s_flat = sharpe(day_net_arr[regimes_list == "flat"])
                s_all = sharpe(day_net_arr)

                denom = max(abs(s_green), abs(s_red), 1e-9)
                regime_imbalance = abs(s_green - s_red) / denom

                row = dict(
                    horizon=h, side=side, bucket=bname,
                    n=stats["n"],
                    net_ticks_per_event=stats["net_ticks_per_event"],
                    net_ticks_median=stats["net_ticks_median"],
                    win_rate=stats["win_rate"],
                    signed_move_mean=stats["signed_move_mean"],
                    mfe_p50=stats["mfe_p50"],
                    mae_p50=stats["mae_p50"],
                    sharpe_all=s_all,
                    sharpe_green=s_green,
                    sharpe_red=s_red,
                    sharpe_flat=s_flat,
                    regime_imbalance=regime_imbalance,
                    profitable_days=profitable_days,
                    total_days=len(unique_dates),
                    data_type="asymmetric_long",
                )
                summary_rows.append(row)

                print(f"  {bname}: n={stats['n']:,} net={stats['net_ticks_per_event']:+.4f} "
                      f"wr={stats['win_rate']:.3f} prof_days={profitable_days}/{len(unique_dates)}",
                      flush=True)

    # ----------------------- write outputs --------------------------------
    print(f"\n[write] outputs...", flush=True)
    summary_df = pd.DataFrame(summary_rows)
    per_day_df = pd.DataFrame(per_day_rows)

    summary_csv = OUT_DIR / "summary.csv"
    per_day_csv = OUT_DIR / "per_day.csv"

    summary_df.to_csv(summary_csv, index=False)
    per_day_df.to_csv(per_day_csv, index=False)

    print(f"[write] summary -> {summary_csv}", flush=True)
    print(f"[write] per_day -> {per_day_csv}", flush=True)

    # ----------------------- deploy gate check ----------------------------------
    if len(summary_df) > 0:
        # Deploy gates: net_ticks > +0.10, wr >= 0.52, BOTH days profitable, regime_imbalance <= 0.50
        deploy_gate = summary_df[
            (summary_df["net_ticks_per_event"] > 0.10) &
            (summary_df["win_rate"] >= 0.52) &
            (summary_df["profitable_days"] == 2) &  # BOTH days (2/2)
            (summary_df["regime_imbalance"] <= 0.50)
        ].copy()
    else:
        deploy_gate = pd.DataFrame()

    n_deploy_pass = len(deploy_gate)
    n_total = len(summary_df)

    # ----------------------- findings --------------------------------
    findings_path = OUT_DIR / "findings.md"
    with open(findings_path, "w") as f:
        f.write("# Quantile DLinear — Closest-to-Profit Analysis v1\n\n")
        f.write("## Context\n")
        f.write(f"- Data source: {DATA_DIR.name}\n")
        f.write(f"- Variant: asymmetric_long (P10/P50/P90 quantile preds optimized for LONG side)\n")
        f.write(f"- Dates: 2026-04-27 (green, N=2.37M), 2026-04-28 (red, N=2.67M)\n")
        f.write(f"- Total events: {total_events:,}\n")
        f.write(f"- Cost model: passive limit fill, {COMMISSION_TICKS} ticks round-trip\n\n")

        f.write("## Deploy Gates (HC #428 R1 + R2)\n")
        f.write("1. net_ticks_per_event > +0.10\n")
        f.write("2. win_rate >= 0.52\n")
        f.write("3. Profitable on BOTH days (2/2, not 1/2)\n")
        f.write("4. |Sharpe_green - Sharpe_red| / max <= 0.50\n\n")

        f.write(f"## Results Summary\n")
        f.write(f"- Total cells analyzed: {n_total}\n")
        f.write(f"- Cells passing ALL deploy gates: **{n_deploy_pass}**\n\n")

        if n_deploy_pass == 0:
            f.write("## FINDING: BASE QUANTILE ALPHA DOES NOT SURVIVE COSTS\n\n")

            if n_total > 0:
                ranked = summary_df.sort_values("net_ticks_per_event", ascending=False)
                best = ranked.iloc[0]

                f.write("### Closest Miss (Top Cell by Net Ticks)\n")
                f.write(f"- horizon={best['horizon']} side={best['side']} bucket={best['bucket']}\n")
                f.write(f"- n={int(best['n']):,} events\n")
                f.write(f"- net_ticks_per_event={best['net_ticks_per_event']:+.4f} ")
                f.write(f"(need > +0.10, gap={0.10 - best['net_ticks_per_event']:+.4f})\n")
                f.write(f"- win_rate={best['win_rate']:.4f} (need >= 0.52)\n")
                f.write(f"- profitable_days={int(best['profitable_days'])}/{int(best['total_days'])} ")
                f.write(f"(need 2/2)\n")
                f.write(f"- regime_imbalance={best['regime_imbalance']:.4f} (need <= 0.50)\n\n")

                f.write("### Diagnostic: Which Gate Fails Most Often?\n")
                gate_fail = {}
                for col, thresh, comp in [
                    ("net_ticks_per_event", 0.10, ">"),
                    ("win_rate", 0.52, ">="),
                    ("profitable_days", 2, "=="),
                    ("regime_imbalance", 0.50, "<="),
                ]:
                    if comp == ">":
                        pass_mask = summary_df[col] > thresh
                    elif comp == ">=":
                        pass_mask = summary_df[col] >= thresh
                    elif comp == "==":
                        pass_mask = summary_df[col] == thresh
                    elif comp == "<=":
                        pass_mask = summary_df[col] <= thresh
                    fail_count = int((~pass_mask).sum())
                    gate_fail[col] = fail_count

                sorted_gates = sorted(gate_fail.items(), key=lambda x: -x[1])
                for col, count in sorted_gates:
                    f.write(f"- {col}: {count}/{n_total} cells fail\n")
        else:
            f.write(f"## FINDING: {n_deploy_pass} CELLS PASS ALL GATES\n\n")
            f.write("### Winning Cells\n")
            best_by_net = deploy_gate.sort_values("net_ticks_per_event", ascending=False).iloc[0]
            f.write(f"Best cell (net_ticks):\n")
            f.write(f"  horizon={best_by_net['horizon']} side={best_by_net['side']} bucket={best_by_net['bucket']}\n")
            f.write(f"  net_ticks={best_by_net['net_ticks_per_event']:+.4f}\n")
            f.write(f"  wr={best_by_net['win_rate']:.4f}\n")
            f.write(f"  sharpe_green={best_by_net['sharpe_green']:.3f}, sharpe_red={best_by_net['sharpe_red']:.3f}\n")
            f.write(f"  regime_imbalance={best_by_net['regime_imbalance']:.4f}\n\n")
            f.write(deploy_gate[["horizon","side","bucket","n","net_ticks_per_event","win_rate",
                                 "profitable_days","regime_imbalance"]].to_string(index=False))

        f.write("\n\n## Regime Asymmetry Check\n")
        # Count cells where sign flips green-to-red (per_day data)
        if len(per_day_df) > 0:
            green_days = per_day_df[per_day_df["regime"] == "green"]
            red_days = per_day_df[per_day_df["regime"] == "red"]

            f.write(f"- Green day (4/27) cells: {len(green_days)}\n")
            f.write(f"- Red day (4/28) cells: {len(red_days)}\n")

            # Find sign flips
            sign_flip_count = 0
            for h in HORIZONS:
                for side in SIDES:
                    for bucket in BUCKETS.keys():
                        gf = green_days[(green_days["horizon"]==h) & (green_days["side"]==side)
                                       & (green_days["bucket"]==bucket)]
                        rf = red_days[(red_days["horizon"]==h) & (red_days["side"]==side)
                                     & (red_days["bucket"]==bucket)]
                        if len(gf) > 0 and len(rf) > 0:
                            g_net = gf["mean_net_ticks"].values[0]
                            r_net = rf["mean_net_ticks"].values[0]
                            if (g_net > 0 and r_net < 0) or (g_net < 0 and r_net > 0):
                                sign_flip_count += 1

            f.write(f"- Sign flips (profit↔loss) green-to-red: {sign_flip_count}\n\n")

        f.write("## Data Caveats\n")
        f.write("- Data type: ASYMMETRIC-LONG variant (P50 predictions optimized for long alpha)\n")
        f.write("- Missing fold_00 (no 4/29 data)\n")
        f.write("- MFE/MAE are PROXY (realized signed moves) for 1s/5s/10s horizons\n")
        f.write("  (true MFE/MAE not available in quantile predictions)\n")
        f.write("- Confidence ordering: P50 (median quantile prediction)\n")

    print(f"[write] findings -> {findings_path}", flush=True)

    # ----------------------- sentinel ---------------------
    regen = {
        "task": "quantile_closest_to_profit_v1",
        "hc_refs": ["HC#489", "HC#428R1", "HC#428R2"],
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "predictions_source": str(DATA_DIR),
        "variant": "asymmetric_long",
        "n_folds": len(folds),
        "n_dates": len(set(f["date"] for f in folds)),
        "total_events": int(total_events),
        "n_summary_cells": int(len(summary_df)),
        "n_deploy_pass": int(n_deploy_pass),
        "elapsed_seconds": round(time.time() - t0, 1),
        "outputs": {
            "summary_csv": str(summary_csv),
            "per_day_csv": str(per_day_csv),
            "findings_md": str(findings_path),
        },
    }

    regen_path = OUT_DIR / ".regen_complete.json"
    with open(regen_path, "w") as f:
        json.dump(regen, f, indent=2)

    print(f"[done] elapsed {time.time()-t0:.1f}s", flush=True)
    print(f"       summary cells: {len(summary_df)}, deploy pass: {n_deploy_pass}", flush=True)

if __name__ == "__main__":
    main()
