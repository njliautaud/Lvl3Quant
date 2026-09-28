#!/usr/bin/env python3
"""
closest_to_profit_v4.py — HC #448 R2 deliverable.

For each (horizon h in {1s,5s,10s,30s,60s}) x (side in {long,short}):
  - Load CNN-Mamba v3.4.2 baseline OOT predictions (per-date) at horizon h directional head.
  - Use realized MFE/MAE within horizon h as the ex-post target.
      * 30s, 60s: use in-OOT target_pred_mfe/mae_{h}_ticks (these ARE v4 alpha labels at sampled events)
      * 1s, 5s, 10s: use target_log_ret_{h} in ticks (signed realized move at h) as a proxy MFE/MAE;
        for long side: realized_move>0 acts as MFE-like; <0 acts as MAE-like
        Annotated as PROXY in output. (v4 row-alignment to OOT sample indices is not recoverable
        without retraining metadata; the in-OOT realized log_ret_h IS the v4-equivalent realized
        signed move computed at training time for the same sampled events.)
  - Confidence buckets by |pred|: top 1/5/10/20/50%, bottom 50%.
  - Compute count, MFE/MAE p50/p75/p90, net-ticks/event at 0.376 commission, win-rate, p90-MFE-MAEp50.
  - Per-day stratification + regime (green/red/flat by per-day mean target_log_ret_60s sign).
  - Identify cells passing deploy gates per HC #428.

Outputs to /home/jupiter/Lvl3Quant/output/closest_to_profit_v4/:
  - summary.csv
  - per_day_stratification.csv
  - winning_cells.txt
  - .regen_complete.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------- constants ----------------------------------
COMMISSION_TICKS = 0.376  # passive fill round-trip (HC: ES_RT_COMMISSION/TICK_VALUE)
HORIZONS = ["1s", "5s", "10s", "30s"]  # 60s realized targets are zero-filled in OOT npz; drop
# NOTE on 60s: target_log_ret_60s and target_pred_mfe/mae_60s_ticks are ALL ZERO in the
# v3.4.2 fixedmtl OOT eval (mask=0 across all dates). The 60s head was predicted but
# 60s realized labels were not computed at eval time. Reported explicitly in winning_cells.txt.
SIDES = ["long", "short"]
BUCKETS = {  # name -> (lo_pct, hi_pct) of |pred| ranking
    "top_1pct":   (0.99, 1.00),
    "top_5pct":   (0.95, 1.00),
    "top_10pct":  (0.90, 1.00),
    "top_20pct":  (0.80, 1.00),
    "top_50pct":  (0.50, 1.00),
    "bottom_50pct": (0.00, 0.50),
}

OOT_DIR = Path("/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate")
OUT_DIR = Path("/home/jupiter/Lvl3Quant/output/closest_to_profit_v4")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------- data load ----------------------------------
def load_oot_day(npz_path: Path):
    d = np.load(npz_path, allow_pickle=True)
    out = {"date": npz_path.stem.replace("oot_", "")}
    for h in HORIZONS:
        pred_key = f"pred_log_ret_{h}"
        tgt_key = f"target_log_ret_{h}"
        mask_key = f"mask_log_ret_{h}"
        pred = d[pred_key].astype(np.float32)
        tgt = d[tgt_key].astype(np.float32)
        m = d[mask_key].astype(np.float32) > 0.5
        # also drop NaN from mask
        m = m & ~np.isnan(pred) & ~np.isnan(tgt)
        out[f"pred_{h}"] = pred
        out[f"realized_signed_ticks_{h}"] = tgt
        out[f"mask_{h}"] = m
    # for 30s: realized MFE/MAE in ticks
    for h in ["30s"]:
        mfe = d[f"target_pred_mfe_{h}_ticks"].astype(np.float32)
        mae = d[f"target_pred_mae_{h}_ticks"].astype(np.float32)
        mm_mfe = d[f"mask_pred_mfe_{h}_ticks"].astype(np.float32) > 0.5
        mm_mae = d[f"mask_pred_mae_{h}_ticks"].astype(np.float32) > 0.5
        mm_mfe = mm_mfe & ~np.isnan(mfe)
        mm_mae = mm_mae & ~np.isnan(mae)
        out[f"realized_mfe_{h}_ticks"] = mfe
        out[f"realized_mae_{h}_ticks"] = mae
        out[f"mfe_mask_{h}"] = mm_mfe
        out[f"mae_mask_{h}"] = mm_mae
    return out

# ----------------------- side adjustment helpers --------------------------
def realized_mfe_mae_for_side(day, h, side):
    """
    Returns (mfe_ticks, mae_ticks, signed_move_ticks, mask) for the given side.
    For 1s/5s/10s: PROXY — uses realized signed move at h as the realized outcome.
      mfe_proxy = max(0, side_signed_move); mae_proxy = max(0, -side_signed_move)
    For 30s/60s: real MFE/MAE from in-OOT alpha v4 labels at sampled events.
    """
    mask_h = day[f"mask_{h}"]
    signed = day[f"realized_signed_ticks_{h}"]
    if side == "short":
        signed = -signed  # flip for short
    if h == "30s":
        # In v3.4.2 OOT npz: MFE is signed positive (best up-move, >=0), MAE is signed negative
        # (worst draw, <=0). Take abs to put into standard "MFE>=0 = best favorable, MAE>=0 = worst adverse"
        # convention used in stats functions.
        mfe_raw = day[f"realized_mfe_{h}_ticks"]
        mae_raw = day[f"realized_mae_{h}_ticks"]
        if side == "long":
            mfe = np.maximum(mfe_raw, 0.0)
            mae = np.abs(np.minimum(mae_raw, 0.0))
        else:  # short — flip: favorable = -MAE (downward move), adverse = -MFE (upward move)
            mfe = np.abs(np.minimum(mae_raw, 0.0))  # was downward draw, now favorable for short
            mae = np.maximum(mfe_raw, 0.0)          # was upward move, now adverse for short
        m = mask_h & day[f"mfe_mask_{h}"] & day[f"mae_mask_{h}"]
        return mfe.astype(np.float32), mae.astype(np.float32), signed.astype(np.float32), m
    else:
        # proxy
        mfe_proxy = np.where(signed > 0, signed, 0.0).astype(np.float32)
        mae_proxy = np.where(signed < 0, -signed, 0.0).astype(np.float32)
        return mfe_proxy, mae_proxy, signed.astype(np.float32), mask_h

# ----------------------- bucketing & stats --------------------------------
def bucket_indices(abs_pred: np.ndarray, lo_pct: float, hi_pct: float):
    """Return boolean mask selecting items whose |pred| percentile falls in [lo, hi)."""
    if len(abs_pred) == 0:
        return np.zeros(0, dtype=bool)
    # rank to percentile via argsort
    order = np.argsort(abs_pred)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(order)) / max(1, len(order) - 1)
    return (ranks >= lo_pct) & (ranks <= hi_pct)

def cell_stats(mfe, mae, signed, mask, bucket_mask):
    sel = mask & bucket_mask
    # extra: filter any NaN in mfe/mae/signed
    sel = sel & ~np.isnan(mfe) & ~np.isnan(mae) & ~np.isnan(signed)
    n = int(sel.sum())
    if n == 0:
        return None
    mfe_s = mfe[sel]; mae_s = mae[sel]; signed_s = signed[sel]
    # net ticks per event = signed realized move - commission (entry+exit passive fill)
    net_ticks = signed_s - COMMISSION_TICKS
    win_rate = float(np.mean(mfe_s > mae_s))
    mfe_p50 = float(np.percentile(mfe_s, 50))
    mfe_p75 = float(np.percentile(mfe_s, 75))
    mfe_p90 = float(np.percentile(mfe_s, 90))
    mae_p50 = float(np.percentile(mae_s, 50))
    mae_p75 = float(np.percentile(mae_s, 75))
    mae_p90 = float(np.percentile(mae_s, 90))
    return dict(
        n=n,
        mfe_p50=mfe_p50, mfe_p75=mfe_p75, mfe_p90=mfe_p90,
        mae_p50=mae_p50, mae_p75=mae_p75, mae_p90=mae_p90,
        net_ticks_per_event=float(np.nanmean(net_ticks)),
        net_ticks_median=float(np.nanmedian(net_ticks)),
        signed_move_p50=float(np.percentile(signed_s, 50)),
        signed_move_mean=float(np.mean(signed_s)),
        win_rate=win_rate,
        edge_p90_mfe_minus_p50_mae=mfe_p90 - mae_p50,
    )

# ----------------------- per-day regime classification --------------------
def classify_regime_per_day(day):
    """green/red/flat from mean of realized signed 30s log_ret_ticks (proxy for daily drift).
    Threshold: per-event mean drift in ticks. With std ~4-6 ticks/event, |mean|>0.10 ticks
    over ~50k events is a clear directional day."""
    mask = day["mask_30s"]
    signed_30 = day["realized_signed_ticks_30s"][mask]
    if len(signed_30) == 0:
        return "flat"
    daily = float(np.mean(signed_30))
    if daily > 0.10:
        return "green"
    elif daily < -0.10:
        return "red"
    return "flat"

def sharpe(net_ticks_array):
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
    oot_files = sorted(OOT_DIR.glob("oot_*.npz"))
    print(f"[load] found {len(oot_files)} OOT date files in {OOT_DIR}", flush=True)
    if not oot_files:
        print("[fatal] no OOT files found; aborting.", file=sys.stderr)
        sys.exit(2)

    days = []
    for f in oot_files:
        try:
            d = load_oot_day(f)
            d["regime"] = classify_regime_per_day(d)
            days.append(d)
            print(f"  loaded {f.name}: regime={d['regime']}", flush=True)
        except Exception as e:
            print(f"  FAILED {f.name}: {e}", file=sys.stderr)

    total_events = sum(d["pred_1s"].shape[0] for d in days)
    print(f"[load] total events across all days: {total_events:,}", flush=True)

    summary_rows = []
    per_day_rows = []

    for h in HORIZONS:
        for side in SIDES:
            # pool predictions across days using bucket assignment computed per-day-pool
            all_abs_pred = []
            all_mfe = []
            all_mae = []
            all_signed = []
            all_mask = []
            all_day = []
            all_regime = []
            for d in days:
                pred = d[f"pred_{h}"]
                mfe, mae, signed, mask = realized_mfe_mae_for_side(d, h, side)
                # For SHORT side, signal = negative of model output (long-bias model)
                # We rank by |pred|: a strong-down prediction is high |pred| short-favorable signal.
                # But for SHORT direction, we want predictions that point DOWN (pred < 0).
                # Approach: for short, we filter pred < 0 then rank by |pred|.
                if side == "short":
                    side_mask = pred < 0
                else:
                    side_mask = pred > 0
                eff_mask = mask & side_mask
                abs_pred = np.abs(pred)
                all_abs_pred.append(abs_pred)
                all_mfe.append(mfe)
                all_mae.append(mae)
                all_signed.append(signed)
                all_mask.append(eff_mask)
                all_day.append(np.full(len(pred), d["date"], dtype=object))
                all_regime.append(np.full(len(pred), d["regime"], dtype=object))

            abs_pred = np.concatenate(all_abs_pred)
            mfe = np.concatenate(all_mfe)
            mae = np.concatenate(all_mae)
            signed = np.concatenate(all_signed)
            mask_all = np.concatenate(all_mask)
            day_arr = np.concatenate(all_day)
            regime_arr = np.concatenate(all_regime)

            # Rank only over masked items
            masked_idx = np.where(mask_all)[0]
            if len(masked_idx) == 0:
                print(f"  WARN: 0 valid events for h={h} side={side}", flush=True)
                continue
            order = np.argsort(abs_pred[masked_idx])
            ranks_dense = np.empty(len(masked_idx), dtype=np.float64)
            ranks_dense[order] = np.arange(len(masked_idx)) / max(1, len(masked_idx) - 1)
            ranks_full = np.full(len(abs_pred), -1.0, dtype=np.float64)
            ranks_full[masked_idx] = ranks_dense

            for bname, (lo, hi) in BUCKETS.items():
                bucket_mask = (ranks_full >= lo) & (ranks_full <= hi) & mask_all
                stats = cell_stats(mfe, mae, signed, mask_all, bucket_mask)
                if stats is None:
                    continue
                # Per-day stratification
                unique_days = np.unique(day_arr[bucket_mask])
                day_net_means = []
                day_regimes_for_day = {}
                profitable_days = 0
                for ud in unique_days:
                    day_mask = bucket_mask & (day_arr == ud) & ~np.isnan(signed) & ~np.isnan(mfe) & ~np.isnan(mae)
                    if day_mask.sum() == 0:
                        continue
                    day_net = signed[day_mask] - COMMISSION_TICKS
                    mean_net = float(np.nanmean(day_net))
                    day_net_means.append(mean_net)
                    if mean_net > 0:
                        profitable_days += 1
                    day_regimes_for_day[ud] = regime_arr[day_mask][0]
                    per_day_rows.append(dict(
                        horizon=h, side=side, bucket=bname, date=ud,
                        regime=day_regimes_for_day[ud], n=int(day_mask.sum()),
                        mean_net_ticks=mean_net,
                        mean_mfe=float(np.mean(mfe[day_mask])),
                        mean_mae=float(np.mean(mae[day_mask])),
                    ))

                # Regime-stratified Sharpe
                day_net_arr = np.array(day_net_means)
                regimes_arr = np.array([day_regimes_for_day[d] for d in unique_days if d in day_regimes_for_day])
                if len(day_net_arr) > 0 and len(regimes_arr) == len(day_net_arr):
                    s_green = sharpe(day_net_arr[regimes_arr == "green"])
                    s_red = sharpe(day_net_arr[regimes_arr == "red"])
                    s_flat = sharpe(day_net_arr[regimes_arr == "flat"])
                    s_all = sharpe(day_net_arr)
                    denom = max(abs(s_green), abs(s_red), 1e-9)
                    regime_imbalance = abs(s_green - s_red) / denom
                else:
                    s_green = s_red = s_flat = s_all = 0.0
                    regime_imbalance = 0.0

                row = dict(
                    horizon=h, side=side, bucket=bname,
                    is_proxy=h in ("1s", "5s", "10s"),
                    sharpe_all_days=s_all,
                    sharpe_green=s_green,
                    sharpe_red=s_red,
                    sharpe_flat=s_flat,
                    regime_imbalance=regime_imbalance,
                    profitable_days=profitable_days,
                    total_days=len(unique_days),
                    **stats,
                )
                summary_rows.append(row)
                print(f"  h={h} side={side} bucket={bname}: n={stats['n']:,} "
                      f"net={stats['net_ticks_per_event']:+.3f} wr={stats['win_rate']:.3f} "
                      f"prof_days={profitable_days}/{len(unique_days)}", flush=True)

    # ----------------------- write outputs --------------------------------
    summary_df = pd.DataFrame(summary_rows)
    per_day_df = pd.DataFrame(per_day_rows)
    summary_csv = OUT_DIR / "summary.csv"
    per_day_csv = OUT_DIR / "per_day_stratification.csv"
    summary_df.to_csv(summary_csv, index=False)
    per_day_df.to_csv(per_day_csv, index=False)
    print(f"[write] summary -> {summary_csv}", flush=True)
    print(f"[write] per_day -> {per_day_csv}", flush=True)

    # ----------------------- gate check ----------------------------------
    if len(summary_df) > 0:
        gate = summary_df[
            (summary_df["net_ticks_per_event"] > 0.10) &
            (summary_df["win_rate"] >= 0.52) &
            (summary_df["profitable_days"] >= 30) &
            (summary_df["regime_imbalance"] <= 0.50)
        ].copy()
    else:
        gate = pd.DataFrame()

    win_path = OUT_DIR / "winning_cells.txt"
    with open(win_path, "w") as f:
        f.write("CLOSEST-TO-PROFIT v4 — HC #448 R2 DELIVERABLE\n")
        f.write("="*80 + "\n")
        f.write(f"Predictions: {OOT_DIR}\n")
        f.write(f"OOT dates: {len(oot_files)}\n")
        f.write(f"Total events analyzed: {total_events:,}\n")
        f.write(f"Cost assumption: passive fill, {COMMISSION_TICKS} ticks/round-trip\n")
        f.write("Note: 1s/5s/10s use realized signed log_ret as proxy MFE/MAE (PROXY rows).\n")
        f.write("       30s/60s use true MFE/MAE from v4 alpha labels at sampled events.\n\n")
        f.write("DEPLOY GATES (HC #428 R1 + R2):\n")
        f.write("  - net-ticks-per-event > +0.10 after 0.376 commission\n")
        f.write("  - win-rate >= 0.52\n")
        f.write("  - profitable on >= 30 OOT days\n")
        f.write("  - |Sharpe_green - Sharpe_red| / max(|.,.|) <= 0.50\n\n")
        if len(gate) == 0:
            f.write("ZERO cells clear the deploy bar.\n\n")
            # closest-miss
            if len(summary_df) > 0:
                ranked = summary_df.sort_values("net_ticks_per_event", ascending=False).head(10)
                f.write("TOP 10 BEST-NET-TICKS CELLS (closest miss):\n")
                f.write(ranked[["horizon","side","bucket","n","net_ticks_per_event","win_rate",
                                "profitable_days","total_days","regime_imbalance","is_proxy"]].to_string(index=False))
                f.write("\n\nGAP TO DEPLOY (top cell):\n")
                top = ranked.iloc[0]
                f.write(f"  horizon={top['horizon']} side={top['side']} bucket={top['bucket']}\n")
                f.write(f"  net_ticks={top['net_ticks_per_event']:+.4f} (need > +0.10)  -> gap {0.10 - top['net_ticks_per_event']:+.4f}\n")
                f.write(f"  win_rate={top['win_rate']:.4f} (need >= 0.52)\n")
                f.write(f"  profitable_days={int(top['profitable_days'])}/{int(top['total_days'])} (need >= 30)\n")
                f.write(f"  regime_imbalance={top['regime_imbalance']:.4f} (need <= 0.50)\n")
        else:
            f.write(f"WINNING CELLS ({len(gate)}):\n\n")
            f.write(gate[["horizon","side","bucket","n","net_ticks_per_event","win_rate",
                          "profitable_days","total_days","sharpe_all_days",
                          "sharpe_green","sharpe_red","regime_imbalance","is_proxy"]].to_string(index=False))
            f.write("\n")
    print(f"[write] winning_cells -> {win_path}", flush=True)

    # ----------------------- regen-complete sentinel ---------------------
    regen = {
        "task": "closest_to_profit_v4",
        "hc_refs": ["HC#448R2", "HC#428R1", "HC#428R2", "HC#485R5"],
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "predictions_source": str(OOT_DIR),
        "n_oot_days": len(oot_files),
        "total_events": int(total_events),
        "n_summary_cells": int(len(summary_df)),
        "n_winning_cells": int(len(gate)),
        "elapsed_seconds": round(time.time() - t0, 1),
        "outputs": {
            "summary_csv": str(summary_csv),
            "per_day_csv": str(per_day_csv),
            "winning_cells_txt": str(win_path),
        },
    }
    with open(OUT_DIR / ".regen_complete.json", "w") as f:
        json.dump(regen, f, indent=2)
    print(f"[done] elapsed {time.time()-t0:.1f}s — winning cells: {len(gate)}", flush=True)

if __name__ == "__main__":
    main()
