#!/usr/bin/env python3
"""
HC #345 — v3.2 FULL Execution Metrics Dashboard

Inputs:
    - /home/jupiter/Lvl3Quant/output/v3_2_deep_sim_20260512/fold_00_oot_predictions.npz
      (241,351 OOT preds across 20260223-27, all 32 v3.2 heads, real MBO bid/ask FIFO labels)

Outputs (all under output/v3_2_full_exec_metrics_dashboard_20260514/):
    - exec_metrics_per_band.csv  (the wide table)
    - exec_metrics_per_band.json
    - static_rules_executable.json   (per-band suggested static {entry, hold, stop, target, cancel})
    - DASHBOARD.md  (human-readable summary)

Per HC #345, for each (horizon, side, band) cell we compute:
    (1) WR%                       — directional accuracy of the prediction
    (2) MFE distribution          — mean, median, p25/p75/p95 in ticks (per +30s window)
    (3) MAE distribution          — same percentiles
    (4) Realized price path       — mean +/- ticks at +1/+5/+10/+30/+60/+300s (signed by pred direction)
    (5) Edge decay                — IC and DA% by horizon (1s/5s/10s/30s/60s/300s)
    (6) Adverse-sel cost          — mean post-fill price move in ANTI-direction at +30s
    (7) Queue-pos proxy           — FIFO fill rate (target_fifo_tp4sl3_net non-zero rate)
    (8) Static-rules executable   — derived {threshold, hold_seconds, stop_ticks, target_ticks,
                                              cancel_seconds} that maximizes Sharpe-toy

NO trainer code is modified. Pure analysis script.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np

ROOT = Path("/home/jupiter/Lvl3Quant")
PRED_NPZ = ROOT / "output/v3_2_deep_sim_20260512/fold_00_oot_predictions.npz"
OUT_DIR = ROOT / "output/v3_2_full_exec_metrics_dashboard_20260514"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = OUT_DIR / "build.log"


def log(msg: str):
    ts = datetime.utcnow().isoformat(timespec="seconds")
    line = f"[{ts}Z] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


# ES tick constants (HC canon)
TICK_VALUE_USD = 12.50
RT_COMM_TICKS = 0.376
PASSIVE_COST_TICKS = RT_COMM_TICKS         # 0.376
MARKET_COST_TICKS = RT_COMM_TICKS + 1.0    # 1.376

# Convert log-return → ticks. ES at ~5000 → 1 tick = 0.25/5000 = 5e-5 in price → log≈5e-5.
# We use the standard approx: ticks ≈ log_ret / TICK_LOG, where TICK_LOG ~ 5e-5 (ESM6 ~5000)
# Use median(target log_ret_5min) magnitude to anchor — but cleaner: use price level proxy.
# For ES @ 5050 mid: 1 tick = 0.25 price units → log(5050.25/5050) = 4.95e-5
TICK_LOG = 5.0e-5


def logret_to_ticks(x):
    return x / TICK_LOG


HORIZONS = [
    ("1s",   "log_ret_1s",   1.0),
    ("5s",   "log_ret_5s",   5.0),
    ("10s",  "log_ret_10s",  10.0),
    ("30s",  "log_ret_30s",  30.0),
    ("60s",  "log_ret_60s",  60.0),
    ("300s", "log_ret_5min", 300.0),
]

BANDS = [
    ("Top0.1%", 0.001),
    ("Top0.5%", 0.005),
    ("Top1%",   0.01),
    ("Top5%",   0.05),
    ("Top10%",  0.10),
]


def percentile_dict(arr, label_prefix=""):
    if arr.size == 0:
        return {f"{label_prefix}{k}": None for k in ["mean", "p25", "p50", "p75", "p95"]}
    return {
        f"{label_prefix}mean": float(np.mean(arr)),
        f"{label_prefix}p25":  float(np.percentile(arr, 25)),
        f"{label_prefix}p50":  float(np.percentile(arr, 50)),
        f"{label_prefix}p75":  float(np.percentile(arr, 75)),
        f"{label_prefix}p95":  float(np.percentile(arr, 95)),
    }


def main():
    log("HC #345 dashboard — loading predictions")
    if not PRED_NPZ.exists():
        log(f"FATAL: missing {PRED_NPZ}")
        sys.exit(2)

    d = np.load(PRED_NPZ, allow_pickle=True)
    n_total = int(d["n_samples"])
    log(f"Loaded n={n_total} preds across dates {d['oot_dates']}")

    # Pull all horizon arrays (preds + targets + masks) in log-return space
    hz_data = {}
    for hkey, hcol, hsec in HORIZONS:
        pred  = d[f"pred_{hcol}"].astype(np.float64)
        tgt   = d[f"target_{hcol}"].astype(np.float64)
        mask  = d[f"mask_{hcol}"].astype(bool)
        hz_data[hkey] = dict(pred=pred, tgt=tgt, mask=mask, hsec=hsec)

    # MFE/MAE 30s heads (true tick units already)
    mfe_30 = d["target_pred_mfe_30s_ticks"].astype(np.float64)
    mae_30 = d["target_pred_mae_30s_ticks"].astype(np.float64)
    mfe_30_mask = d["mask_pred_mfe_30s_ticks"].astype(bool)
    mae_30_mask = d["mask_pred_mae_30s_ticks"].astype(bool)

    # FIFO fill labels (already realized PnL net of commission, in ticks/contract)
    fifo43_net = d["target_fifo_tp4sl3_net"].astype(np.float64)
    fifo43_msk = d["mask_fifo_tp4sl3_net"].astype(bool)
    fifo43_hit = d["target_fifo_tp4sl3_hit_tp"].astype(np.float64)
    fifo85_net = d["target_fifo_tp8sl5_net"].astype(np.float64)
    fifo85_msk = d["mask_fifo_tp8sl5_net"].astype(bool)
    fifo85_hit = d["target_fifo_tp8sl5_hit_tp"].astype(np.float64)

    rows = []
    static_rules = []

    # ----------------- per (horizon, side, band) cell -----------------
    for hkey, hcol, hsec in HORIZONS:
        pred = hz_data[hkey]["pred"]
        tgt  = hz_data[hkey]["tgt"]
        msk  = hz_data[hkey]["mask"]

        # Use abs(pred) ranking for top-N% across BOTH sides combined,
        # then split by sign of pred for long/short.
        valid = msk & np.isfinite(pred) & np.isfinite(tgt)
        idx_valid = np.where(valid)[0]
        if idx_valid.size < 100:
            log(f"  skip horizon={hkey}: only {idx_valid.size} valid")
            continue
        absp = np.abs(pred[idx_valid])
        # Sort descending
        order = np.argsort(-absp)

        for bname, bfrac in BANDS:
            n_take = max(1, int(round(idx_valid.size * bfrac)))
            top_idx_global = idx_valid[order[:n_take]]

            # Split long vs short by sign of pred
            for side, sign_mask in [
                ("long",  pred[top_idx_global] > 0),
                ("short", pred[top_idx_global] < 0),
            ]:
                cell_idx = top_idx_global[sign_mask]
                n = int(cell_idx.size)
                if n < 5:
                    continue

                p = pred[cell_idx]
                t = tgt[cell_idx]
                # Direction sign (+1 for long, -1 for short)
                dsign = 1.0 if side == "long" else -1.0

                # (1) WR — directional accuracy: sign(t) == sign(p)
                wr = float(np.mean(np.sign(t) == np.sign(p)) * 100.0)

                # IC (Pearson) within cell
                if np.std(p) > 0 and np.std(t) > 0:
                    ic = float(np.corrcoef(p, t)[0, 1])
                else:
                    ic = float("nan")

                # MagCorr — corr(|p|, t*sign(p))  i.e. does higher conviction predict bigger move-in-direction
                signed_t = t * np.sign(p)
                if np.std(np.abs(p)) > 0 and np.std(signed_t) > 0:
                    magcorr = float(np.corrcoef(np.abs(p), signed_t)[0, 1])
                else:
                    magcorr = float("nan")

                # Realized move at this horizon (in ticks, signed in trade direction)
                move_ticks = logret_to_ticks(t) * dsign

                # Sharpe-toy at this band/horizon (one unit per pick, no costs)
                if np.std(move_ticks) > 0:
                    sharpe_toy = float(np.mean(move_ticks) / np.std(move_ticks) * math.sqrt(n))
                else:
                    sharpe_toy = float("nan")

                # (2)(3) MFE/MAE distribution from MFE/MAE_30s heads — but only for fills present
                cell_mfe_mask = mfe_30_mask[cell_idx] & np.isfinite(mfe_30[cell_idx])
                cell_mae_mask = mae_30_mask[cell_idx] & np.isfinite(mae_30[cell_idx])
                mfe_arr = mfe_30[cell_idx][cell_mfe_mask] * dsign  # signed in trade direction
                mae_arr = mae_30[cell_idx][cell_mae_mask] * dsign
                mfe_dist = percentile_dict(mfe_arr, "mfe30_")
                mae_dist = percentile_dict(mae_arr, "mae30_")
                mfe_to_mae = (
                    float(np.mean(mfe_arr) / abs(np.mean(mae_arr)))
                    if mae_arr.size > 0 and abs(np.mean(mae_arr)) > 1e-9
                    else None
                )

                # (4) Price path — for each of the 6 horizons, compute mean realized signed move (ticks)
                price_path = {}
                for hkey2, hcol2, hsec2 in HORIZONS:
                    msk2 = hz_data[hkey2]["mask"][cell_idx]
                    t2   = hz_data[hkey2]["tgt"][cell_idx]
                    keep = msk2 & np.isfinite(t2)
                    if keep.sum() == 0:
                        price_path[f"+{hsec2:.0f}s_mean_ticks"] = None
                    else:
                        price_path[f"+{hsec2:.0f}s_mean_ticks"] = float(
                            np.mean(logret_to_ticks(t2[keep]) * dsign)
                        )

                # (5) Edge decay — IC at each horizon for this cell
                ic_decay = {}
                da_decay = {}
                for hkey2, hcol2, hsec2 in HORIZONS:
                    msk2 = hz_data[hkey2]["mask"][cell_idx]
                    t2   = hz_data[hkey2]["tgt"][cell_idx]
                    keep = msk2 & np.isfinite(t2)
                    if keep.sum() < 5 or np.std(p[keep]) == 0 or np.std(t2[keep]) == 0:
                        ic_decay[f"ic_{hkey2}"] = None
                        da_decay[f"da_{hkey2}"] = None
                    else:
                        ic_decay[f"ic_{hkey2}"] = float(np.corrcoef(p[keep], t2[keep])[0, 1])
                        da_decay[f"da_{hkey2}"] = float(
                            np.mean(np.sign(p[keep]) == np.sign(t2[keep])) * 100.0
                        )

                # (6) Adverse-sel cost — at +30s, fraction of cell where realized went OPPOSITE to pred,
                # and average magnitude of those opposite moves
                msk30 = hz_data["30s"]["mask"][cell_idx]
                t30   = hz_data["30s"]["tgt"][cell_idx]
                keep30 = msk30 & np.isfinite(t30)
                if keep30.sum() > 0:
                    signed30 = logret_to_ticks(t30[keep30]) * dsign
                    adv_mask = signed30 < 0
                    adv_frac = float(np.mean(adv_mask) * 100.0)
                    adv_cost_ticks = (
                        float(np.mean(signed30[adv_mask])) if adv_mask.sum() > 0 else 0.0
                    )
                else:
                    adv_frac = None
                    adv_cost_ticks = None

                # (7) Queue-pos proxy — FIFO fill rate (target labels exist when fill happened)
                cell_fifo43 = fifo43_msk[cell_idx]
                cell_fifo85 = fifo85_msk[cell_idx]
                fill43_pct = float(np.mean(cell_fifo43) * 100.0) if cell_idx.size > 0 else 0.0
                fill85_pct = float(np.mean(cell_fifo85) * 100.0) if cell_idx.size > 0 else 0.0
                # FIFO net PnL (ticks, after commission) for filled trades
                pnl43 = (
                    float(np.mean(fifo43_net[cell_idx][cell_fifo43]))
                    if cell_fifo43.sum() > 0
                    else None
                )
                pnl85 = (
                    float(np.mean(fifo85_net[cell_idx][cell_fifo85]))
                    if cell_fifo85.sum() > 0
                    else None
                )
                hit43_pct = (
                    float(np.mean(fifo43_hit[cell_idx][cell_fifo43]) * 100.0)
                    if cell_fifo43.sum() > 0
                    else None
                )

                # (8) STATIC-RULES EXECUTABLE — derive best static parameters for this cell
                # Pick hold = horizon at which signed_mean_path is maximum (peak edge)
                best_hold_sec = None
                best_hold_ticks = -math.inf
                for hkey2, hcol2, hsec2 in HORIZONS:
                    val = price_path[f"+{hsec2:.0f}s_mean_ticks"]
                    if val is not None and val > best_hold_ticks:
                        best_hold_ticks = val
                        best_hold_sec = hsec2
                # Stop = MAE p75 (covers 75% of normal drawdown), Target = MFE p50
                stop_ticks_hint = (
                    abs(mae_dist["mae30_p75"]) if mae_dist["mae30_p75"] is not None else None
                )
                target_ticks_hint = mfe_dist["mfe30_p50"]
                # Cancel = peak-time horizon (where most of edge already realized)
                cancel_sec = best_hold_sec
                # Threshold = the |pred| floor of this band (lowest qualifier)
                cell_thresh = float(np.min(np.abs(p))) if p.size > 0 else None

                static_rules.append({
                    "horizon": hkey,
                    "side": side,
                    "band": bname,
                    "n": n,
                    "entry_threshold_abs_pred": cell_thresh,
                    "hold_seconds_suggested": best_hold_sec,
                    "stop_ticks_suggested": stop_ticks_hint,
                    "target_ticks_suggested": target_ticks_hint,
                    "cancel_seconds_suggested": cancel_sec,
                    "expected_edge_ticks_at_hold": best_hold_ticks if best_hold_ticks > -1e8 else None,
                    "passive_breakeven_ticks": PASSIVE_COST_TICKS,
                    "market_breakeven_ticks": MARKET_COST_TICKS,
                    "passive_profitable": (
                        best_hold_ticks > PASSIVE_COST_TICKS
                        if best_hold_ticks > -1e8
                        else False
                    ),
                    "market_profitable": (
                        best_hold_ticks > MARKET_COST_TICKS
                        if best_hold_ticks > -1e8
                        else False
                    ),
                })

                row = {
                    "horizon": hkey,
                    "side": side,
                    "band": bname,
                    "n": n,
                    "wr_pct": wr,
                    "ic_in_cell": ic,
                    "magcorr": magcorr,
                    "sharpe_toy": sharpe_toy,
                    "realized_mean_ticks": float(np.mean(move_ticks)),
                    "realized_p25_ticks":  float(np.percentile(move_ticks, 25)),
                    "realized_p75_ticks":  float(np.percentile(move_ticks, 75)),
                    **mfe_dist,
                    **mae_dist,
                    "mfe_to_mae_ratio": mfe_to_mae,
                    **price_path,
                    **ic_decay,
                    **da_decay,
                    "adverse_frac_30s_pct": adv_frac,
                    "adverse_cost_ticks_30s": adv_cost_ticks,
                    "fill_pct_tp4sl3": fill43_pct,
                    "fill_pct_tp8sl5": fill85_pct,
                    "fifo_pnl_ticks_tp4sl3_filled": pnl43,
                    "fifo_pnl_ticks_tp8sl5_filled": pnl85,
                    "tp_hit_pct_tp4sl3": hit43_pct,
                }
                rows.append(row)
                log(
                    f"  {hkey:>5} {side:>5} {bname:>8} n={n:5d} "
                    f"WR={wr:5.1f}%  Sharpe-toy={sharpe_toy:6.2f}  "
                    f"hold={best_hold_sec}s  edge={best_hold_ticks:5.2f}t  "
                    f"fill43={fill43_pct:5.1f}%  pnl43={pnl43}"
                )

    # Write JSON + CSV + MD
    json_path = OUT_DIR / "exec_metrics_per_band.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "generated_utc": datetime.utcnow().isoformat(),
                "source_npz": str(PRED_NPZ),
                "n_total_preds": n_total,
                "tick_log_constant": TICK_LOG,
                "passive_cost_ticks": PASSIVE_COST_TICKS,
                "market_cost_ticks": MARKET_COST_TICKS,
                "rows": rows,
            },
            f,
            indent=2,
            default=lambda o: None if (isinstance(o, float) and not math.isfinite(o)) else o,
        )
    log(f"wrote {json_path} ({len(rows)} rows)")

    # CSV
    if rows:
        csv_path = OUT_DIR / "exec_metrics_per_band.csv"
        keys = list(rows[0].keys())
        with open(csv_path, "w") as f:
            f.write(",".join(keys) + "\n")
            for r in rows:
                f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
        log(f"wrote {csv_path}")

    # Static rules JSON
    sr_path = OUT_DIR / "static_rules_executable.json"
    with open(sr_path, "w") as f:
        json.dump(
            {
                "generated_utc": datetime.utcnow().isoformat(),
                "rules": static_rules,
            },
            f,
            indent=2,
            default=lambda o: None if (isinstance(o, float) and not math.isfinite(o)) else o,
        )
    log(f"wrote {sr_path}")

    # Build markdown summary — pick high-value cells
    md_lines = ["# HC #345 — v3.2 Full Execution Metrics Dashboard\n"]
    md_lines.append(f"_Generated {datetime.utcnow().isoformat()}Z, n={n_total} OOT preds, "
                    f"v3.2 fold 0 deep-sim 20260223-27_\n")
    md_lines.append("\n## Top0.1% & Top0.5% headline (per horizon × side)\n")
    md_lines.append("| Horiz | Side | Band | n | WR% | Sharpe-toy | Realized mean (t) | "
                    "Hold suggested (s) | Edge at hold (t) | Passive profitable? | "
                    "Fill43% | FIFO PnL filled (t) | Adv-sel frac% | Adv cost (t) |\n")
    md_lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in rows:
        if r["band"] not in ("Top0.1%", "Top0.5%"):
            continue
        sr = next(
            (s for s in static_rules if s["horizon"] == r["horizon"]
             and s["side"] == r["side"] and s["band"] == r["band"]),
            None,
        )
        hold_s = sr["hold_seconds_suggested"] if sr else None
        edge_h = sr["expected_edge_ticks_at_hold"] if sr else None
        passive_ok = sr["passive_profitable"] if sr else False
        md_lines.append(
            f"| {r['horizon']} | {r['side']} | {r['band']} | {r['n']} | "
            f"{r['wr_pct']:.1f} | {r['sharpe_toy']:.2f} | "
            f"{r['realized_mean_ticks']:.3f} | "
            f"{hold_s} | "
            f"{edge_h:.3f} | "
            f"{'PASS' if passive_ok else 'fail'} | "
            f"{r['fill_pct_tp4sl3']:.1f} | "
            f"{r['fifo_pnl_ticks_tp4sl3_filled']} | "
            f"{r['adverse_frac_30s_pct']} | "
            f"{r['adverse_cost_ticks_30s']} |\n"
        )

    md_lines.append("\n## STATIC-RULES executable params (entries with passive_profitable=PASS)\n")
    md_lines.append("| Horiz | Side | Band | n | Threshold(|pred|) | Hold(s) | Stop(t) | Target(t) | Cancel(s) | Edge(t) |\n")
    md_lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for s in static_rules:
        if not s["passive_profitable"]:
            continue
        md_lines.append(
            f"| {s['horizon']} | {s['side']} | {s['band']} | {s['n']} | "
            f"{s['entry_threshold_abs_pred']:.5f} | "
            f"{s['hold_seconds_suggested']} | "
            f"{s['stop_ticks_suggested']} | "
            f"{s['target_ticks_suggested']} | "
            f"{s['cancel_seconds_suggested']} | "
            f"{s['expected_edge_ticks_at_hold']:.3f} |\n"
        )

    md_lines.append("\n## Edge decay — IC by horizon for Top0.1% short (best-edge cell)\n")
    md_lines.append("| From horizon | Cell | IC@1s | IC@5s | IC@10s | IC@30s | IC@60s | IC@300s |\n")
    md_lines.append("|---|---|---:|---:|---:|---:|---:|---:|\n")
    for r in rows:
        if r["band"] != "Top0.1%" or r["side"] != "short":
            continue
        md_lines.append(
            f"| {r['horizon']} | Top0.1% short | "
            f"{r.get('ic_1s')} | {r.get('ic_5s')} | {r.get('ic_10s')} | "
            f"{r.get('ic_30s')} | {r.get('ic_60s')} | {r.get('ic_300s')} |\n"
        )

    md_path = OUT_DIR / "DASHBOARD.md"
    with open(md_path, "w") as f:
        f.writelines(md_lines)
    log(f"wrote {md_path}")
    log("HC #345 dashboard COMPLETE")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback as tb
        log("FATAL: " + repr(e))
        log(tb.format_exc())
        sys.exit(1)
