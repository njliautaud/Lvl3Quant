#!/usr/bin/env python3
"""
ofi_edge_test_v1.py — Standalone OFI edge test + model × OFI intersection (HC #451 R3).

Stage 2: Bucket events by each OFI feature × horizon × side. Run the same
bracket-exit simulator as bracket_exit_v1.py. Apply all deploy gates.

Stage 3: For top-3 closest-miss cells from bracket_exit_v1, intersect the
model's prediction bucket with the OFI feature's same-side bucket and re-test.

Join policy (HC #451 R3):
  - OFI features are per-event over the raw 12.4M/day corpus.
  - v4 alpha labels are also per-event over the same raw corpus.
  - OOT model predictions are sampled at v4_idx = 1499 + k*250 (verified corr=1.0 in bracket_exit_v1).
  - For Stage 2 (standalone, no model) we ALIGN OFI features to the same stride-250 grid
    so we can apply the same deploy-gate machinery as bracket_exit_v1 head-to-head
    AND keep walltime manageable. This is a true standalone test — model preds are NOT used.
  - For Stage 3 we use the same grid for OFI + model preds.

Deploy gates (HC #428 R1+R2):
  - net_ticks > +0.10
  - WR >= 0.52
  - profitable_days >= 30
  - regime_imbalance <= 0.50
  - day_concentration <= 0.70
  - TP <= p90 of realized MFE in bucket
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
COMMISSION_TICKS = 0.376
HORIZONS = ["1s", "5s", "10s", "30s", "60s"]
SIDES = ["long", "short"]
BUCKETS = {
    "top_1pct":  (0.99, 1.00),
    "top_5pct":  (0.95, 1.00),
    "top_10pct": (0.90, 1.00),
    "top_20pct": (0.80, 1.00),
}
SL_GRID = [0.5, 1.0, 1.5, 2.0]
TP_QUANTILES = [0.25, 0.50, 0.75, 0.90]

JOIN_OFFSET = 1499
JOIN_STRIDE = 250

OOT_DIR = Path("/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate")
V4_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3_alpha_labels_v4")
OFI_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3_ofi_features")
BRACKET_V1_SUMMARY = Path("/home/jupiter/Lvl3Quant/output/bracket_exit_v1/summary.csv")
OUT_DIR = Path("/home/jupiter/Lvl3Quant/output/ofi_edge_v1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Which OFI features to test in Stage 2 (filter to non-MISSING per stats)
OFI_FEATURE_NAMES = (
    [f"ofi_aggressive_{w}s" for w in (1, 5, 10, 30)]
    + [f"ofi_book_{w}s" for w in (1, 5, 10, 30)]
)

WALLTIME_CAP_SEC = 75 * 60


def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{ts} {msg}", flush=True)


# --------------------------------------------------------------------------
def load_day(date: str):
    """Load OOT preds + joined v4 MFE/MAE + OFI features all at stride-250 grid."""
    oot_path = OOT_DIR / f"oot_{date}.npz"
    v4_path = V4_DIR / f"{date}_alpha_labels.npz"
    ofi_path = OFI_DIR / f"{date}_ofi.npz"
    if not (oot_path.exists() and v4_path.exists() and ofi_path.exists()):
        return None
    oot = np.load(oot_path, allow_pickle=True)
    v4 = np.load(v4_path, allow_pickle=True)
    ofi = np.load(ofi_path, allow_pickle=True)

    N_oot = oot["pred_log_ret_1s"].shape[0]
    N_v4 = v4["target_mfe_1s_ticks"].shape[0]
    v4_idx = JOIN_OFFSET + np.arange(N_oot) * JOIN_STRIDE
    in_bounds = v4_idx < N_v4
    if not in_bounds.all():
        N_oot = int(in_bounds.sum())
        v4_idx = v4_idx[in_bounds]

    out = {"date": date, "N": N_oot}
    for h in HORIZONS:
        pred = oot[f"pred_log_ret_{h}"].astype(np.float32)[:N_oot]
        signed = oot[f"target_log_ret_{h}"].astype(np.float32)[:N_oot]
        sig_mask = oot[f"mask_log_ret_{h}"].astype(np.float32)[:N_oot] > 0.5
        mfe_long = v4[f"target_mfe_{h}_ticks"][v4_idx].astype(np.float32)
        mae_long = v4[f"target_mae_{h}_ticks"][v4_idx].astype(np.float32)
        valid = sig_mask & ~np.isnan(pred) & ~np.isnan(signed) & ~np.isnan(mfe_long) & ~np.isnan(mae_long)
        out[f"pred_{h}"] = pred
        out[f"signed_ticks_{h}"] = signed
        out[f"mfe_long_{h}"] = mfe_long
        out[f"mae_long_{h}"] = mae_long
        out[f"mask_{h}"] = valid

    # OFI features at stride-250 grid (same v4_idx)
    for fn in OFI_FEATURE_NAMES:
        if fn in ofi.files:
            v = ofi[fn][v4_idx].astype(np.float32)
            out[f"ofi_{fn}"] = v
        else:
            out[f"ofi_{fn}"] = np.full(N_oot, np.nan, dtype=np.float32)
    return out


def regime_for_day(day):
    if "signed_ticks_30s" not in day:
        return "flat"
    m = day["mask_30s"]
    if m.sum() == 0:
        return "flat"
    daily = float(np.mean(day["signed_ticks_30s"][m]))
    if daily > 0.10: return "green"
    if daily < -0.10: return "red"
    return "flat"


def favorable_adverse_for_side(day, h, side):
    mfe_long = day[f"mfe_long_{h}"]
    mae_long = day[f"mae_long_{h}"]
    signed = day[f"signed_ticks_{h}"]
    mask = day[f"mask_{h}"]
    if side == "long":
        favorable = np.maximum(mfe_long, 0.0)
        adverse = np.maximum(-mae_long, 0.0)
        signed_side = signed
    else:
        favorable = np.maximum(-mae_long, 0.0)
        adverse = np.maximum(mfe_long, 0.0)
        signed_side = -signed
    return favorable.astype(np.float32), adverse.astype(np.float32), signed_side.astype(np.float32), mask


def bracket_pnl(fav, adv, sgn, tp, sl):
    sl_hit = adv >= sl
    tp_hit = (fav >= tp) & ~sl_hit
    tmo = ~sl_hit & ~tp_hit
    pnl = np.empty_like(fav)
    pnl[tp_hit] = tp
    pnl[sl_hit] = -sl
    pnl[tmo] = sgn[tmo]
    return pnl, tp_hit, sl_hit, tmo


def sharpe(arr):
    arr = np.asarray(arr, dtype=np.float64)
    if len(arr) < 2: return 0.0
    s = np.std(arr, ddof=1)
    if s == 0: return 0.0
    return float(np.mean(arr) / s * np.sqrt(252))


def evaluate_cell(bmask, fav, adv, sgn, dayarr, regarr, p_mfe, tp, sl):
    """Evaluate one (TP, SL) bracket on a bucket-mask. Returns a stats dict."""
    pnl_b, tp_hit, sl_hit, tmo = bracket_pnl(fav[bmask], adv[bmask], sgn[bmask], tp, sl)
    net = pnl_b - COMMISSION_TICKS
    day_b = dayarr[bmask]
    reg_b = regarr[bmask]
    unique_days = np.unique(day_b)
    day_net = []; day_tot = []; day_reg = []; profdays = 0
    for ud in unique_days:
        dm = day_b == ud
        m = float(np.mean(net[dm])); s = float(np.sum(net[dm]))
        day_net.append(m); day_tot.append(s); day_reg.append(reg_b[dm][0])
        if m > 0: profdays += 1
    day_net_arr = np.array(day_net); day_tot_arr = np.array(day_tot)
    reg_arr = np.array(day_reg)

    s_all = sharpe(day_net_arr)
    s_green = sharpe(day_net_arr[reg_arr == "green"])
    s_red = sharpe(day_net_arr[reg_arr == "red"])
    denom = max(abs(s_green), abs(s_red), 1e-9)
    regimb = abs(s_green - s_red) / denom
    abs_tot = np.abs(day_tot_arr)
    dayconc = float(abs_tot.max() / abs_tot.sum()) if abs_tot.sum() > 0 else 1.0

    net_mean = float(np.mean(net))
    wr = float(np.mean(net > 0))
    tp_within_p90 = tp <= p_mfe[0.90] + 1e-6

    gates = dict(
        gate_net=net_mean > 0.10, gate_wr=wr >= 0.52,
        gate_pdays=profdays >= 30, gate_regime=regimb <= 0.50,
        gate_dayconc=dayconc <= 0.70, gate_tp=tp_within_p90,
    )
    pass_all = all(gates.values())
    return dict(
        n=int(bmask.sum()),
        mfe_p25=p_mfe[0.25], mfe_p50=p_mfe[0.50],
        mfe_p75=p_mfe[0.75], mfe_p90=p_mfe[0.90],
        tp_ticks=round(tp, 4), sl_ticks=sl,
        tp_rate=float(np.mean(tp_hit)), sl_rate=float(np.mean(sl_hit)),
        timeout_rate=float(np.mean(tmo)),
        mean_pnl_pre_cost=float(np.mean(pnl_b)),
        net_ticks_per_event=net_mean, net_ticks_median=float(np.median(net)),
        win_rate=wr,
        profitable_days=profdays, total_days=int(len(unique_days)),
        sharpe_all_days=s_all, sharpe_green=s_green, sharpe_red=s_red,
        regime_imbalance=regimb, day_concentration=dayconc,
        tp_within_p90=tp_within_p90, pass_all_gates=pass_all,
        **gates,
    )


def rank_by_feature(feature_vals, msk):
    """Return per-event rank in [0,1] over masked items, NaN outside."""
    mi = np.where(msk)[0]
    if len(mi) == 0:
        return np.full(len(feature_vals), -1.0)
    order = np.argsort(feature_vals[mi])
    ranks = np.empty(len(mi), dtype=np.float64)
    ranks[order] = np.arange(len(mi)) / max(1, len(mi) - 1)
    out = np.full(len(feature_vals), -1.0)
    out[mi] = ranks
    return out


# --------------------------------------------------------------------------
def stage2(days):
    log(f"[stage2] OFI standalone edge test on {len(days)} days, "
        f"{len(OFI_FEATURE_NAMES)} OFI features × {len(HORIZONS)} horizons")
    rows = []

    for fn in OFI_FEATURE_NAMES:
        for h in HORIZONS:
            for side in SIDES:
                all_fav = []; all_adv = []; all_sgn = []; all_msk = []
                all_ofi = []; all_day = []; all_reg = []
                for d in days:
                    fav, adv, sgn, msk = favorable_adverse_for_side(d, h, side)
                    ofi_v = d[f"ofi_{fn}"]
                    eff = msk & np.isfinite(ofi_v)
                    all_fav.append(fav); all_adv.append(adv); all_sgn.append(sgn)
                    all_msk.append(eff); all_ofi.append(ofi_v)
                    all_day.append(np.full(len(fav), d["date"], dtype=object))
                    all_reg.append(np.full(len(fav), d["regime"], dtype=object))
                fav = np.concatenate(all_fav); adv = np.concatenate(all_adv)
                sgn = np.concatenate(all_sgn); msk = np.concatenate(all_msk)
                ofi_v = np.concatenate(all_ofi)
                dayarr = np.concatenate(all_day); regarr = np.concatenate(all_reg)

                if msk.sum() == 0:
                    continue

                # Long side bets on HIGH OFI (buy pressure). Short side bets on LOW OFI (sell pressure).
                # We rank ascending; long buckets pick top by raw value, short buckets pick bottom.
                # Easier: rank ascending so 1.0 = highest. For LONG we keep BUCKETS as top-X%.
                # For SHORT we invert: use (1 - rank) >= lo.
                ranks = rank_by_feature(ofi_v, msk)

                for bname, (lo, hi) in BUCKETS.items():
                    if side == "long":
                        bmask = (ranks >= lo) & (ranks <= hi) & msk
                    else:
                        bmask = (ranks >= 0) & (ranks <= 1.0 - lo) & msk  # bottom-X%
                    n_b = int(bmask.sum())
                    if n_b < 200:
                        continue
                    fav_b = fav[bmask]
                    p_mfe = {q: float(np.percentile(fav_b, q * 100)) for q in TP_QUANTILES}
                    for q in TP_QUANTILES:
                        tp = p_mfe[q]
                        if tp <= 0:
                            continue
                        for sl in SL_GRID:
                            stats = evaluate_cell(bmask, fav, adv, sgn, dayarr, regarr, p_mfe, tp, sl)
                            stats.update(dict(feature=fn, horizon=h, side=side, bucket=bname, tp_q=q))
                            rows.append(stats)
                log(f"  {fn} {h} {side}: examined buckets")
    return pd.DataFrame(rows)


def stage3(days, closest_cells):
    """closest_cells: list of dicts with horizon, side, bucket (model bucket) from bracket_exit_v1."""
    log(f"[stage3] model × OFI intersection on top {len(closest_cells)} closest-miss cells")
    rows = []
    for cell in closest_cells:
        h = cell["horizon"]; side = cell["side"]; mbname = cell["bucket"]
        mlo, mhi = BUCKETS[mbname]

        for fn in OFI_FEATURE_NAMES:
            all_fav = []; all_adv = []; all_sgn = []; all_msk = []
            all_pred = []; all_ofi = []; all_day = []; all_reg = []
            for d in days:
                fav, adv, sgn, msk = favorable_adverse_for_side(d, h, side)
                pred = d[f"pred_{h}"]
                if side == "long":
                    sd = pred > 0
                else:
                    sd = pred < 0
                eff = msk & sd & np.isfinite(d[f"ofi_{fn}"])
                all_fav.append(fav); all_adv.append(adv); all_sgn.append(sgn)
                all_msk.append(eff); all_pred.append(np.abs(pred))
                all_ofi.append(d[f"ofi_{fn}"])
                all_day.append(np.full(len(fav), d["date"], dtype=object))
                all_reg.append(np.full(len(fav), d["regime"], dtype=object))
            fav = np.concatenate(all_fav); adv = np.concatenate(all_adv)
            sgn = np.concatenate(all_sgn); msk = np.concatenate(all_msk)
            apr = np.concatenate(all_pred); ofi_v = np.concatenate(all_ofi)
            dayarr = np.concatenate(all_day); regarr = np.concatenate(all_reg)
            if msk.sum() == 0:
                continue

            # Model rank (by |pred|, descending = top)
            m_ranks = rank_by_feature(apr, msk)
            # OFI rank
            o_ranks = rank_by_feature(ofi_v, msk)

            # Model bucket: top X% by |pred|
            m_mask = (m_ranks >= mlo) & (m_ranks <= mhi) & msk

            # OFI same-side: long -> top, short -> bottom; try multiple OFI bucket sizes
            for obname, (olo, ohi) in BUCKETS.items():
                if side == "long":
                    o_mask = (o_ranks >= olo) & (o_ranks <= ohi) & msk
                else:
                    o_mask = (o_ranks >= 0) & (o_ranks <= 1.0 - olo) & msk
                inter = m_mask & o_mask
                n_b = int(inter.sum())
                if n_b < 200:
                    continue
                fav_b = fav[inter]
                p_mfe = {q: float(np.percentile(fav_b, q * 100)) for q in TP_QUANTILES}
                for q in TP_QUANTILES:
                    tp = p_mfe[q]
                    if tp <= 0:
                        continue
                    for sl in SL_GRID:
                        stats = evaluate_cell(inter, fav, adv, sgn, dayarr, regarr, p_mfe, tp, sl)
                        stats.update(dict(
                            feature=fn, horizon=h, side=side,
                            model_bucket=mbname, ofi_bucket=obname,
                            model_cell_orig_net=cell.get("net_ticks_per_event", np.nan),
                            tp_q=q,
                        ))
                        rows.append(stats)
            log(f"  cell {h}/{side}/{mbname} × {fn}: done")
    return pd.DataFrame(rows)


def pick_closest_cells():
    if not BRACKET_V1_SUMMARY.exists():
        log(f"[warn] {BRACKET_V1_SUMMARY} missing — fall back to top-3 by net")
        return []
    df = pd.read_csv(BRACKET_V1_SUMMARY)
    gates = ["gate_net", "gate_wr", "gate_pdays", "gate_regime", "gate_dayconc", "gate_tp"]
    for g in gates:
        if g not in df.columns:
            df[g] = False
    df["n_gates_passed"] = df[gates].sum(axis=1)
    top = df.sort_values(["n_gates_passed", "net_ticks_per_event"], ascending=False).head(3)
    return top.to_dict("records")


def main():
    t0 = time.time()
    log(f"[start] ofi_edge_test_v1")

    # Discover overlap dates
    oot = {p.stem.replace("oot_", "") for p in OOT_DIR.glob("oot_*.npz")}
    v4 = {p.name.split("_")[0] for p in V4_DIR.glob("*_alpha_labels.npz")}
    ofi = {p.name.split("_")[0] for p in OFI_DIR.glob("*_ofi.npz")}
    overlap = sorted(oot & v4 & ofi)
    log(f"[dates] overlap (oot ∩ v4 ∩ ofi) = {len(overlap)}")
    if len(overlap) < 5:
        log("[fatal] not enough dates; run ofi_features_v1.py first"); sys.exit(2)

    days = []
    for d in overlap:
        if time.time() - t0 > WALLTIME_CAP_SEC * 0.3:
            log(f"[walltime] load phase cap; stopping at {len(days)} days")
            break
        try:
            day = load_day(d)
            if day is None or day["N"] == 0:
                continue
            day["regime"] = regime_for_day(day)
            days.append(day)
            log(f"  loaded {d}: N={day['N']:,} regime={day['regime']}")
        except Exception as e:
            log(f"  FAILED {d}: {e}")

    if not days:
        log("[fatal] no usable days"); sys.exit(3)

    # ---- Stage 2 ----
    if time.time() - t0 < WALLTIME_CAP_SEC * 0.7:
        s2 = stage2(days)
        s2_csv = OUT_DIR / "standalone_summary.csv"
        s2.to_csv(s2_csv, index=False)
        log(f"[write] {s2_csv} ({len(s2)} rows)")
    else:
        s2 = pd.DataFrame()
        log("[skip] stage2 (walltime)")

    # ---- Stage 3 ----
    closest = pick_closest_cells()
    log(f"[stage3] closest cells from bracket_exit_v1: {[(c['horizon'], c['side'], c['bucket']) for c in closest]}")
    if closest and time.time() - t0 < WALLTIME_CAP_SEC * 0.95:
        s3 = stage3(days, closest)
        s3_csv = OUT_DIR / "combined_summary.csv"
        s3.to_csv(s3_csv, index=False)
        log(f"[write] {s3_csv} ({len(s3)} rows)")
    else:
        s3 = pd.DataFrame()
        log("[skip] stage3 (walltime or no closest cells)")

    # ---- Winners ----
    win_path = OUT_DIR / "winning_cells.txt"
    with open(win_path, "w") as f:
        f.write("OFI EDGE V1 — standalone + model×OFI intersection (HC #451 R3)\n")
        f.write("=" * 80 + "\n")
        f.write(f"OOT dates used: {len(days)}\n")
        f.write(f"Join key: v4_idx = {JOIN_OFFSET} + k * {JOIN_STRIDE} (verified corr=1.0 in bracket_exit_v1)\n")
        f.write(f"OFI features tested: {OFI_FEATURE_NAMES}\n")
        f.write(f"Commission: {COMMISSION_TICKS} ticks (passive)\n\n")
        f.write("DEPLOY GATES: net>+0.10, WR>=0.52, prof_days>=30, regime_imb<=0.50, day_conc<=0.70, TP<=p90 MFE\n\n")

        # Stage 2 results
        f.write("STAGE 2 — STANDALONE OFI EDGE\n")
        f.write("-" * 80 + "\n")
        if len(s2) > 0:
            win2 = s2[s2["pass_all_gates"]].copy()
            f.write(f"Total cells evaluated: {len(s2)}\n")
            f.write(f"Cells passing ALL gates: {len(win2)}\n\n")
            cols = ["feature", "horizon", "side", "bucket", "tp_ticks", "sl_ticks", "n",
                    "net_ticks_per_event", "win_rate", "profitable_days", "total_days",
                    "regime_imbalance", "day_concentration", "tp_within_p90",
                    "tp_rate", "sl_rate", "timeout_rate"]
            if len(win2) > 0:
                f.write("WINNING STANDALONE CELLS:\n")
                f.write(win2.sort_values("net_ticks_per_event", ascending=False)[cols].to_string(index=False))
                f.write("\n\n")
            top5 = s2.sort_values("net_ticks_per_event", ascending=False).head(5)
            f.write("TOP 5 STANDALONE BY NET_TICKS:\n")
            f.write(top5[cols].to_string(index=False))
            f.write("\n\n")
        else:
            f.write("(stage 2 not run)\n\n")

        # Stage 3 results
        f.write("STAGE 3 — MODEL × OFI INTERSECTION\n")
        f.write("-" * 80 + "\n")
        if len(s3) > 0:
            win3 = s3[s3["pass_all_gates"]].copy()
            f.write(f"Total intersection cells: {len(s3)}\n")
            f.write(f"Intersection cells passing ALL gates: {len(win3)}\n\n")
            cols3 = ["feature", "horizon", "side", "model_bucket", "ofi_bucket",
                     "tp_ticks", "sl_ticks", "n",
                     "net_ticks_per_event", "model_cell_orig_net",
                     "win_rate", "profitable_days", "total_days",
                     "regime_imbalance", "day_concentration"]
            if len(win3) > 0:
                f.write("WINNING INTERSECTION CELLS:\n")
                f.write(win3.sort_values("net_ticks_per_event", ascending=False)[cols3].to_string(index=False))
                f.write("\n\n")
            top3 = s3.sort_values("net_ticks_per_event", ascending=False).head(3)
            f.write("TOP 3 INTERSECTION CELLS (closest to deploy bar):\n")
            f.write(top3[cols3].to_string(index=False))
            f.write("\n\n")
        else:
            f.write("(stage 3 not run)\n\n")

        # Strategic note
        f.write("STRATEGIC NOTE\n")
        f.write("-" * 80 + "\n")
        if len(s2) > 0:
            best_standalone_net = float(s2["net_ticks_per_event"].max())
            if best_standalone_net > 0.30:
                f.write(f"!!! OFI HAS STRONG STANDALONE EDGE — best cell net={best_standalone_net:+.4f} ticks/event !!!\n")
                f.write("RECOMMEND: add OFI as model feature and retrain CNN-Mamba on combined feature set.\n\n")
            else:
                f.write(f"Best standalone OFI cell: net={best_standalone_net:+.4f} ticks/event.\n")
                f.write("Below the +0.30 strong-edge bar; OFI alone is not a deploy-ready signal.\n")
                if len(s3) > 0:
                    best_inter_net = float(s3["net_ticks_per_event"].max())
                    bracket_v1_best = -0.40
                    lift = best_inter_net - bracket_v1_best
                    f.write(f"Model × OFI best cell: net={best_inter_net:+.4f} ticks/event (lift over baseline={lift:+.4f}).\n")
                    if best_inter_net > 0.10:
                        f.write("RECOMMEND: deploy candidate identified via intersection — verify in fill sim before live.\n")
                    else:
                        f.write("RECOMMEND: structural alpha augmentation needed — add OFI as feature, retrain on combined.\n")
    log(f"[write] {win_path}")

    # Regen sentinel
    regen = {
        "task": "ofi_edge_test_v1",
        "hc_refs": ["HC#451R3", "HC#428R1", "HC#428R2", "HC#485R5", "HC#420"],
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_dates_used": len(days),
        "n_standalone_rows": int(len(s2)) if isinstance(s2, pd.DataFrame) else 0,
        "n_intersection_rows": int(len(s3)) if isinstance(s3, pd.DataFrame) else 0,
        "elapsed_seconds": round(time.time() - t0, 1),
        "outputs": {
            "standalone_csv": str(OUT_DIR / "standalone_summary.csv"),
            "combined_csv": str(OUT_DIR / "combined_summary.csv"),
            "winning_cells_txt": str(win_path),
        },
    }
    with open(OUT_DIR / ".regen_complete.json", "w") as f:
        json.dump(regen, f, indent=2)
    log(f"[done] elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
