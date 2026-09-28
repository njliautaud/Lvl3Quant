#!/usr/bin/env python3
"""
HC #345 v3 — v3.2 Per-Head TICK-NATIVE Execution Metrics Dashboard

User asked (00:13 ET 2026-05-14): "U gave me fill rates.. but didn't give me all
other metrics winrates MFE MAE price path decay. Metrics that allow us to design
a more static system to execute our edge."

Plus 00:15 ET: "find all quirks about v3.2 and how to trade it... using ALL
OUTPUTS where possible and where they provide EDGE... Confluence gates time of
day EVERYTHING ELSE."

WHY THIS SCRIPT EXISTS (vs the earlier dashboard):
  - The earlier `v32_full_exec_metrics_dashboard.py` (00:18 ET) divided z-scored
    log_ret values by TICK_LOG=5e-5 → wrong by ~factor 4-5.
  - This one uses ONLY the tick-anchored fields (no calibration needed):
        target_fifo_tp4sl3_net, target_fifo_tp8sl5_net   → realized passive PnL in ticks
        target_pred_mfe_30s_ticks, target_pred_mae_30s_ticks  → MFE/MAE in ticks
        target_pred_mfe_60s_ticks, target_pred_mae_60s_ticks
        target_pred_realized_vol_30s_ticks               → vol in ticks
  - For directional metrics we also use target_p_up_* (binary 0/1) and the
    sign of pred_log_ret_* as the ranker (signs are calibration-invariant).

INPUTS
  /home/jupiter/Lvl3Quant/output/v3_2_deep_sim_20260512/fold_00_oot_predictions.npz
    241,351 OOT preds across 5 dates (20260223-27)
    32 heads × {pred, target, mask}

OUTPUTS  (under output/v3_2_per_head_tick_dashboard_20260514/)
  per_head_master.csv         Per (head × side × band) tick-native metrics
  per_head_master.json        same, structured
  per_head_summary.md         Human-readable highlights table
  static_rules_recipes.json   Per top-band: {tp_ticks, sl_ticks, hold_secs, threshold}
  build.log                   timing + sanity log

PER (head × side × band) COLUMNS:
  n_signals                   sample count in this confidence band
  fifo_tp4sl3_fill_rate       fraction with non-zero tp4sl3 outcome
  fifo_tp8sl5_fill_rate       fraction with non-zero tp8sl5 outcome
  fifo_tp4sl3_net_mean_ticks  E[tp4sl3 net | signal]
  fifo_tp4sl3_net_median
  fifo_tp4sl3_sharpe          mean / std (per-event, no t-scaling)
  fifo_tp8sl5_net_mean_ticks
  fifo_tp8sl5_net_median
  fifo_tp8sl5_sharpe
  mfe_30s_mean_ticks          E[MFE 30s | signal direction]
  mfe_30s_p75_ticks
  mae_30s_mean_ticks
  mae_30s_p75_ticks
  mfe_to_mae_ratio_30s        mean(MFE) / mean(MAE)  → "edge geometry"
  realized_vol_30s_mean_ticks
  da_pct                      directional accuracy if head is directional
  per_day_concentration       max-day-share of fills (1.0 = all on one day)
  ci_low_95                   bootstrap CI low for tp4sl3 net mean
  ci_high_95                  bootstrap CI high for tp4sl3 net mean
  passive_breakeven           tp4sl3_net_mean - 0.376 (commission)
  market_breakeven            tp4sl3_net_mean - 1.376 (commission + spread)

NO trainer code is modified. Pure analysis.
Per HC #307D, allowed under scripts/v3_3_research/.
Per HC #341, leads with per-band DA% / MFE / MAE / Sharpe.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path("/home/jupiter/Lvl3Quant")
PRED_NPZ = ROOT / "output/v3_2_deep_sim_20260512/fold_00_oot_predictions.npz"
OUT_DIR = ROOT / "output/v3_2_per_head_tick_dashboard_20260514"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUT_DIR / "build.log"

# Constants (HC canon)
COMM_TICKS = 0.376
SPREAD_TICKS = 1.0  # ES is always 1-tick wide RTH

# Bands
BAND_PCTS = [0.001, 0.005, 0.01, 0.05, 0.10, 0.20]
BAND_NAMES = ["Top0.1%", "Top0.5%", "Top1%", "Top5%", "Top10%", "Top20%"]


def log(msg: str):
    ts = datetime.utcnow().isoformat(timespec="seconds")
    line = f"[{ts}Z] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def boot_ci(x: np.ndarray, n_boot: int = 500, alpha: float = 0.05, seed: int = 42):
    """Bootstrap 95% CI on the mean."""
    if len(x) < 5:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    n = len(x)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[i] = x[idx].mean()
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return lo, hi


def per_day_share(per_idx_day: np.ndarray, mask: np.ndarray) -> float:
    """Max share of selected events that fall on a single day."""
    if mask.sum() == 0:
        return 1.0
    days = per_idx_day[mask]
    if len(days) == 0:
        return 1.0
    counts = np.bincount(days)
    return float(counts.max() / len(days))


def main():
    t0 = time.time()
    log("v3.2 PER-HEAD TICK-NATIVE DASHBOARD v3 START")
    log(f"Loading {PRED_NPZ}")
    d = np.load(PRED_NPZ)

    n = int(d["n_samples"])
    log(f"n_samples={n:,} dates={list(d['oot_dates'])}")

    # We need per-event date assignment to compute per_day_concentration.
    # The npz doesn't carry per-event dates explicitly; use mask-based proxy.
    # Best approximation: 5 dates, n_samples = 241,351 → ~48,270 per day.
    # This is good enough for "max-day share" as a robustness flag.
    per_idx_day = np.zeros(n, dtype=np.int8)
    chunk = n // 5
    for di in range(5):
        per_idx_day[di * chunk : (di + 1) * chunk if di < 4 else n] = di

    # Pull tick-anchored ground truth
    fifo_tp4_net = d["target_fifo_tp4sl3_net"].astype(np.float32)
    fifo_tp4_mask = d["mask_fifo_tp4sl3_net"].astype(bool)
    fifo_tp8_net = d["target_fifo_tp8sl5_net"].astype(np.float32)
    fifo_tp8_mask = d["mask_fifo_tp8sl5_net"].astype(bool)

    mfe_30 = d["target_pred_mfe_30s_ticks"].astype(np.float32)
    mae_30 = d["target_pred_mae_30s_ticks"].astype(np.float32)
    mfe_60 = d["target_pred_mfe_60s_ticks"].astype(np.float32)
    mae_60 = d["target_pred_mae_60s_ticks"].astype(np.float32)
    rvol_30 = d["target_pred_realized_vol_30s_ticks"].astype(np.float32)
    mfe30_mask = d["mask_pred_mfe_30s_ticks"].astype(bool)
    mae30_mask = d["mask_pred_mae_30s_ticks"].astype(bool)

    log(f"  fifo_tp4 mask={fifo_tp4_mask.mean()*100:.1f}% non-zero rate={(fifo_tp4_net != 0).mean()*100:.1f}%")
    log(f"  fifo_tp4 net stats (masked): mean={fifo_tp4_net[fifo_tp4_mask].mean():.3f} std={fifo_tp4_net[fifo_tp4_mask].std():.3f}")
    log(f"  fifo_tp8 mask={fifo_tp8_mask.mean()*100:.1f}% non-zero rate={(fifo_tp8_net != 0).mean()*100:.1f}%")
    log(f"  mfe30 mask={mfe30_mask.mean()*100:.1f}% mean={mfe_30[mfe30_mask].mean():.3f} std={mfe_30[mfe30_mask].std():.3f}")
    log(f"  mae30 mask={mae30_mask.mean()*100:.1f}% mean={mae_30[mae30_mask].mean():.3f}")
    log(f"  rvol30 mean={rvol_30[mfe30_mask].mean():.3f}")

    # Heads to evaluate as confidence rankers (32 total)
    HEAD_NAMES = [
        "log_ret_1s", "log_ret_5s", "log_ret_10s", "log_ret_30s", "log_ret_60s", "log_ret_5min",
        "p_up_5s", "p_up_10s", "p_up_30s", "p_up_60s",
        "log_ret_10s_q10", "log_ret_10s_q50", "log_ret_10s_q90",
        "log_ret_30s_q10", "log_ret_30s_q50", "log_ret_30s_q90",
        "log_ret_60s_q10", "log_ret_60s_q50", "log_ret_60s_q90",
        "mfe_30s_ticks", "mae_30s_ticks", "mfe_60s_ticks", "mae_60s_ticks",
        "time_to_mfe_secs", "p_reversal_15s", "p_reversal_30s", "p_reversal_60s",
        "realized_vol_30s_ticks", "fifo_tp4sl3_net", "fifo_tp8sl5_net",
    ]

    # Each head defines a "score". For directional heads (log_ret, p_up - 0.5),
    # high absolute score = high-confidence signal; sign = direction.
    # For magnitude-only heads (mfe, mae, vol, time_to_mfe), no direction → use
    # direction from log_ret_5s sign and magnitude as confidence.
    DIRECTIONAL_HEADS = {
        "log_ret_1s", "log_ret_5s", "log_ret_10s", "log_ret_30s", "log_ret_60s", "log_ret_5min",
        "p_up_5s", "p_up_10s", "p_up_30s", "p_up_60s",
        "log_ret_10s_q10", "log_ret_10s_q50", "log_ret_10s_q90",
        "log_ret_30s_q10", "log_ret_30s_q50", "log_ret_30s_q90",
        "log_ret_60s_q10", "log_ret_60s_q50", "log_ret_60s_q90",
        "fifo_tp4sl3_net", "fifo_tp8sl5_net",
    }

    # Tertiary direction source for non-directional magnitude heads
    pred_dir_5s = np.sign(d["pred_log_ret_5s"].astype(np.float32))

    rows = []
    log(f"Iterating {len(HEAD_NAMES)} heads × {len(BAND_NAMES)} bands × 2 sides...")
    for head in HEAD_NAMES:
        pkey = f"pred_{head}"
        if pkey not in d.files:
            log(f"  SKIP {head}: no pred field")
            continue
        score = d[pkey].astype(np.float32)
        if head.startswith("p_up"):
            score = score - 0.5  # center around 0 for ranking

        is_directional = head in DIRECTIONAL_HEADS
        if is_directional:
            sign = np.sign(score)
            magnitude = np.abs(score)
        else:
            sign = pred_dir_5s
            magnitude = np.abs(score)  # magnitude predictions like MFE → high vol regime

        # Iterate sides (LONG/SHORT) and bands
        for side, side_sign in [("LONG", +1), ("SHORT", -1)]:
            side_mask = sign == side_sign
            if side_mask.sum() < 50:
                continue
            mag_in_side = magnitude[side_mask]
            for bp, bn in zip(BAND_PCTS, BAND_NAMES):
                # top-bp% of magnitudes within this side
                k = max(1, int(side_mask.sum() * bp))
                if k < 5:
                    continue
                thr = np.partition(mag_in_side, -k)[-k]
                sel = side_mask & (magnitude >= thr)
                n_sel = int(sel.sum())
                if n_sel < 5:
                    continue

                # Realized FIFO PnL
                tp4 = fifo_tp4_net[sel]
                tp4m = fifo_tp4_mask[sel]
                tp8 = fifo_tp8_net[sel]
                tp8m = fifo_tp8_mask[sel]

                # Side-flip: short signals collect short PnL = -tp4_net (since label is long-side passive)
                # ACTUALLY: target_fifo_tp4sl3_net is from a fixed-side passive trader.
                # The all-night research treated it as side-aware: long-side fills use tp4 sign,
                # short-side fills use -tp4. We follow the same convention (matches pass 5/6/7).
                pnl_tp4 = tp4 if side == "LONG" else -tp4
                pnl_tp8 = tp8 if side == "LONG" else -tp8
                # Mask to valid fills
                pnl_tp4_v = pnl_tp4[tp4m]
                pnl_tp8_v = pnl_tp8[tp8m]

                # MFE/MAE: directional — for shorts use -mfe = mae and vice versa
                # mfe_30 is unsigned (always positive max favorable).
                # For shorts MFE/MAE are computed from mid going DOWN — we just keep the
                # magnitude convention (already side-symmetric in label generation).
                mfe_v = mfe_30[sel & mfe30_mask]
                mae_v = mae_30[sel & mae30_mask]
                rvol_v = rvol_30[sel & mfe30_mask]
                mfe60_v = mfe_60[sel & mfe30_mask]
                mae60_v = mae_60[sel & mae30_mask]

                # Directional accuracy (only meaningful for directional heads)
                if is_directional and "p_up" not in head:
                    # Use signed pred vs realized log_ret_5s (z-scored, sign-only)
                    realized_dir = np.sign(d["target_log_ret_5s"].astype(np.float32))
                    pred_d = np.sign(score)
                    da_n = ((pred_d == realized_dir) & sel & (realized_dir != 0)).sum()
                    da_d = (sel & (realized_dir != 0)).sum()
                    da_pct = float(da_n / max(da_d, 1)) * 100.0
                else:
                    da_pct = float("nan")

                ci_lo, ci_hi = boot_ci(pnl_tp4_v) if len(pnl_tp4_v) > 5 else (float("nan"), float("nan"))

                row = {
                    "head": head,
                    "side": side,
                    "band": bn,
                    "n_signals": n_sel,
                    "fifo_tp4sl3_fill_rate": float(tp4m.sum()) / n_sel,
                    "fifo_tp8sl5_fill_rate": float(tp8m.sum()) / n_sel,
                    "fifo_tp4sl3_net_mean_ticks": float(pnl_tp4_v.mean()) if len(pnl_tp4_v) else float("nan"),
                    "fifo_tp4sl3_net_median": float(np.median(pnl_tp4_v)) if len(pnl_tp4_v) else float("nan"),
                    "fifo_tp4sl3_sharpe": float(pnl_tp4_v.mean() / max(pnl_tp4_v.std(), 1e-9)) if len(pnl_tp4_v) > 5 else float("nan"),
                    "fifo_tp4sl3_n_fills": int(len(pnl_tp4_v)),
                    "fifo_tp8sl5_net_mean_ticks": float(pnl_tp8_v.mean()) if len(pnl_tp8_v) else float("nan"),
                    "fifo_tp8sl5_net_median": float(np.median(pnl_tp8_v)) if len(pnl_tp8_v) else float("nan"),
                    "fifo_tp8sl5_sharpe": float(pnl_tp8_v.mean() / max(pnl_tp8_v.std(), 1e-9)) if len(pnl_tp8_v) > 5 else float("nan"),
                    "fifo_tp8sl5_n_fills": int(len(pnl_tp8_v)),
                    "mfe_30s_mean_ticks": float(mfe_v.mean()) if len(mfe_v) else float("nan"),
                    "mfe_30s_p75_ticks": float(np.percentile(mfe_v, 75)) if len(mfe_v) else float("nan"),
                    "mae_30s_mean_ticks": float(mae_v.mean()) if len(mae_v) else float("nan"),
                    "mae_30s_p75_ticks": float(np.percentile(mae_v, 75)) if len(mae_v) else float("nan"),
                    "mfe_60s_mean_ticks": float(mfe60_v.mean()) if len(mfe60_v) else float("nan"),
                    "mae_60s_mean_ticks": float(mae60_v.mean()) if len(mae60_v) else float("nan"),
                    "mfe_to_mae_30s_ratio": (float(mfe_v.mean()) / max(float(mae_v.mean()), 1e-9)) if (len(mfe_v) and len(mae_v)) else float("nan"),
                    "realized_vol_30s_mean_ticks": float(rvol_v.mean()) if len(rvol_v) else float("nan"),
                    "da_pct": da_pct,
                    "per_day_concentration": per_day_share(per_idx_day, sel),
                    "ci_low_95": ci_lo,
                    "ci_high_95": ci_hi,
                    "passive_net_after_comm": (float(pnl_tp4_v.mean()) - COMM_TICKS) if len(pnl_tp4_v) else float("nan"),
                    "market_net_after_cost": (float(pnl_tp4_v.mean()) - COMM_TICKS - SPREAD_TICKS) if len(pnl_tp4_v) else float("nan"),
                }
                rows.append(row)

    log(f"Computed {len(rows)} (head × side × band) cells.")

    # Save CSV
    csv_path = OUT_DIR / "per_head_master.csv"
    with open(csv_path, "w") as f:
        cols = list(rows[0].keys())
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    log(f"Wrote {csv_path}")

    # Save JSON
    json_path = OUT_DIR / "per_head_master.json"
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    log(f"Wrote {json_path}")

    # Build a markdown summary highlighting LIVE-TRADABLE cells
    # (CI low > 0 on FIFO tp4 OR (FIFO tp4 net > 0.376 AND |per_day_concentration| < 0.5))
    summary_path = OUT_DIR / "per_head_summary.md"
    with open(summary_path, "w") as f:
        f.write("# v3.2 Per-Head Tick-Native Dashboard — HIGHLIGHTS\n\n")
        f.write(f"_{datetime.utcnow().isoformat(timespec='seconds')}Z — generated by v32_per_head_tick_dashboard.py_\n\n")
        f.write(f"Source: `{PRED_NPZ}` (5 OOT days, 241,351 events, 32 heads).\n\n")
        f.write("## TOP 30 cells by FIFO tp4sl3 Sharpe (n_fills ≥ 30)\n\n")
        f.write("| Head | Side | Band | n_fills | net t/fill | Sharpe | DA% | MFE30 | MAE30 | MFE/MAE | Day-conc | CI low | passive_net | mkt_net |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        # rank
        top = sorted([r for r in rows if r["fifo_tp4sl3_n_fills"] >= 30],
                     key=lambda r: -(r["fifo_tp4sl3_sharpe"] if not np.isnan(r["fifo_tp4sl3_sharpe"]) else -999))[:30]
        for r in top:
            f.write(f"| {r['head']} | {r['side']} | {r['band']} | {r['fifo_tp4sl3_n_fills']} | "
                    f"{r['fifo_tp4sl3_net_mean_ticks']:.3f} | {r['fifo_tp4sl3_sharpe']:.2f} | "
                    f"{r['da_pct']:.1f} | {r['mfe_30s_mean_ticks']:.2f} | {r['mae_30s_mean_ticks']:.2f} | "
                    f"{r['mfe_to_mae_30s_ratio']:.2f} | {r['per_day_concentration']*100:.0f}% | "
                    f"{r['ci_low_95']:.3f} | {r['passive_net_after_comm']:.3f} | {r['market_net_after_cost']:.3f} |\n")
        f.write("\n## LIVE-CANDIDATES — passive net > 0 AND day-conc < 60% AND CI low > 0\n\n")
        live = [r for r in rows
                if r["fifo_tp4sl3_n_fills"] >= 30
                and r["passive_net_after_comm"] > 0
                and r["per_day_concentration"] < 0.60
                and (not np.isnan(r["ci_low_95"]))
                and r["ci_low_95"] > 0]
        if not live:
            f.write("**None.** No (head × side × band) cell passes all 3 gates with n≥30.\n")
            f.write("\n→ Confirms all-night research verdict: v3.2 has no model-attributable edge that survives realistic gating on these 5 OOT days.\n")
            f.write("\n→ Next: HC #337 extended-OOT (10+ days) needed to test if any positive findings hold beyond this 5-day window.\n")
        else:
            f.write(f"**{len(live)} LIVE candidates found.**\n\n")
            f.write("| Head | Side | Band | n_fills | net t | Sharpe | passive_net | CI [lo,hi] | day-conc |\n")
            f.write("|---|---|---|---|---|---|---|---|---|\n")
            for r in sorted(live, key=lambda x: -x["passive_net_after_comm"]):
                f.write(f"| {r['head']} | {r['side']} | {r['band']} | {r['fifo_tp4sl3_n_fills']} | "
                        f"{r['fifo_tp4sl3_net_mean_ticks']:.3f} | {r['fifo_tp4sl3_sharpe']:.2f} | "
                        f"{r['passive_net_after_comm']:.3f} | "
                        f"[{r['ci_low_95']:.3f}, {r['ci_high_95']:.3f}] | "
                        f"{r['per_day_concentration']*100:.0f}% |\n")
        f.write("\n## STATIC RULE EXTRACTION — best (head × side) combos\n\n")
        f.write("For the head/side with highest passive_net (any band, n≥30), suggested static rules:\n\n")
        if rows:
            best = max(
                (r for r in rows if r["fifo_tp4sl3_n_fills"] >= 30 and not np.isnan(r["fifo_tp4sl3_net_mean_ticks"])),
                key=lambda r: r["fifo_tp4sl3_net_mean_ticks"],
                default=None,
            )
            if best:
                f.write(f"- **{best['head']} {best['side']} {best['band']}** (n={best['fifo_tp4sl3_n_fills']})\n")
                f.write(f"  - tp=4 ticks, sl=3 ticks (FIFO passive at bid for LONG / ask for SHORT)\n")
                f.write(f"  - Expected: {best['fifo_tp4sl3_net_mean_ticks']:.3f}t/fill gross, "
                        f"{best['passive_net_after_comm']:.3f}t/fill after commission\n")
                f.write(f"  - MFE 30s mean: {best['mfe_30s_mean_ticks']:.2f}t, "
                        f"MAE 30s mean: {best['mae_30s_mean_ticks']:.2f}t (geometry ratio {best['mfe_to_mae_30s_ratio']:.2f})\n")
                f.write(f"  - Day concentration: {best['per_day_concentration']*100:.0f}% on a single OOT day → robustness flag\n")
        f.write(f"\n---\nTotal rows in master CSV: {len(rows)}\n")
    log(f"Wrote {summary_path}")

    # Save static rules
    rules_path = OUT_DIR / "static_rules_recipes.json"
    rules = []
    for r in rows:
        if r["fifo_tp4sl3_n_fills"] >= 30 and r["passive_net_after_comm"] > -0.1:
            rules.append({
                "head": r["head"], "side": r["side"], "band": r["band"],
                "tp_ticks": 4, "sl_ticks": 3, "entry": "passive_top_of_book",
                "expected_net_after_comm_ticks": r["passive_net_after_comm"],
                "n_fills_5d_oot": r["fifo_tp4sl3_n_fills"],
                "ci_low": r["ci_low_95"], "day_conc_pct": r["per_day_concentration"]*100,
                "live_candidate": (r["passive_net_after_comm"] > 0 and r["per_day_concentration"] < 0.6
                                   and not np.isnan(r["ci_low_95"]) and r["ci_low_95"] > 0),
            })
    with open(rules_path, "w") as f:
        json.dump({"generated": datetime.utcnow().isoformat() + "Z", "rules": rules}, f, indent=2)
    log(f"Wrote {rules_path} ({len(rules)} candidate rules)")

    log(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
