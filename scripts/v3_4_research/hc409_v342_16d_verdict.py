"""
HC #409 strict-gate verdict on v3.4.2 16-day combined OOT.

Inputs (read-only):
  output/v342_fold_00_ep1_oot_inference.npz           (241,351 samples, 5d 02-23..02-27)
  output/v342_fold_00_ep1_oot_inference_extended.npz  (516,569 samples, 11d 03-01..03-15)

Outputs:
  output/hc409_v342_16d_verdict_<ts>/
    ic_by_horizon.csv             headline IC 1s/5s/10s/30s
    tier_ic_perf.csv              (horizon x side x tier) -> IC, n, gross_mfe, hit_rate, fifo_net
    promotion_summary.csv         cells passing HC #408 strict-gate (n>=50, day_conc<=0.20, CI_low_95>0)
    VERDICT.json                  headline + recommendation

HC #408 strict gate: n>=50 fills, day_conc<=0.20, CI_low_95(net_ticks)>0.
HC #405 cost: net_ticks computed as TP4SL3 fifo_net (already accounts for commission).
"""
from __future__ import annotations
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats as scstats

PROJ = Path("/home/jupiter/Lvl3Quant")
NPZ_5D = PROJ / "output" / "v342_fold_00_ep1_oot_inference.npz"
NPZ_11D = PROJ / "output" / "v342_fold_00_ep1_oot_inference_extended.npz"

DAYS_5D = ["2026-02-23", "2026-02-24", "2026-02-25", "2026-02-26", "2026-02-27"]
DAYS_11D = ["2026-03-01", "2026-03-02", "2026-03-03", "2026-03-04", "2026-03-05",
            "2026-03-08", "2026-03-09", "2026-03-10", "2026-03-11", "2026-03-12",
            "2026-03-15"]
ALL_DAYS = DAYS_5D + DAYS_11D  # 16

HORIZONS = ["1s", "5s", "10s", "30s"]
SIDES = ["long", "short"]
TIERS = [("Top10", 0.10), ("Top5", 0.05), ("Top1", 0.01), ("Top0.5", 0.005), ("Top0.1", 0.001)]

OUT = PROJ / "output" / f"hc409_v342_16d_verdict_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUT.mkdir(parents=True, exist_ok=True)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_and_concat():
    log(f"loading 5d  {NPZ_5D.name}")
    a = np.load(NPZ_5D, allow_pickle=True)
    log(f"loading 11d {NPZ_11D.name}")
    b = np.load(NPZ_11D, allow_pickle=True)
    n_a = a["pred_log_ret_1s"].shape[0]
    n_b = b["pred_log_ret_1s"].shape[0]
    log(f"5d samples={n_a}, 11d samples={n_b}, total={n_a+n_b}")
    # day-partition indices using np.array_split over the per-NPZ N
    # day_idx aligned to combined sample order
    parts_a = np.array_split(np.arange(n_a), len(DAYS_5D))
    parts_b = np.array_split(np.arange(n_b), len(DAYS_11D))
    day_idx_a = np.empty(n_a, dtype=np.int32)
    for i, p in enumerate(parts_a):
        day_idx_a[p] = i
    day_idx_b = np.empty(n_b, dtype=np.int32)
    for i, p in enumerate(parts_b):
        day_idx_b[p] = i + len(DAYS_5D)
    out = {}
    # concat per-sample arrays (shape match required, skip scalars)
    for k in a.files:
        arr_a = a[k]
        arr_b = b[k]
        if arr_a.ndim == 0 or arr_b.ndim == 0:
            continue
        if arr_a.shape[0] != n_a or arr_b.shape[0] != n_b:
            log(f"SKIP {k}: shape mismatch ({arr_a.shape} vs {arr_b.shape})")
            continue
        out[k] = np.concatenate([arr_a, arr_b], axis=0)
    out["__day_idx__"] = np.concatenate([day_idx_a, day_idx_b])
    return out, n_a + n_b


def safe_pearson(x, y):
    if len(x) < 3:
        return float("nan")
    try:
        r, _ = scstats.pearsonr(x, y)
        return float(r)
    except Exception:
        return float("nan")


def safe_spearman(x, y):
    if len(x) < 3:
        return float("nan")
    try:
        r, _ = scstats.spearmanr(x, y)
        return float(r)
    except Exception:
        return float("nan")


def main():
    data, N = load_and_concat()
    day_idx = data["__day_idx__"]

    # ----- Headline IC by horizon (Pearson + Spearman, masked) -----
    rows = []
    for h in HORIZONS:
        pred = data[f"pred_log_ret_{h}"]
        tgt = data[f"target_log_ret_{h}"]
        msk = data[f"mask_log_ret_{h}"].astype(bool)
        p = pred[msk]
        t = tgt[msk]
        finite = np.isfinite(p) & np.isfinite(t)
        p = p[finite]
        t = t[finite]
        rows.append({
            "horizon": h,
            "n_valid": int(msk.sum()),
            "ic_pearson": safe_pearson(p, t),
            "ic_spearman": safe_spearman(p, t),
        })
    df_ic = pd.DataFrame(rows)
    df_ic.to_csv(OUT / "ic_by_horizon.csv", index=False)
    log("IC by horizon:\n" + df_ic.to_string(index=False))

    # ----- Tier perf: horizon x side x tier -----
    # For each (horizon, side), rank by signed-confidence of that head:
    #   long: rank by pred (descending) → top X% strongest LONG signals
    #   short: rank by -pred (descending) → top X% strongest SHORT signals
    # Within tier, compute:
    #   n, mean_pred, ic_pearson, mean_target, hit_rate (sign(target)==side),
    #   mean_gross_mfe_ticks (signed, 30s window), mean_fifo_tp4sl3_net (target),
    #   day_conc (max share of fills concentrated in any single day)
    rows = []
    for h in HORIZONS:
        pred = data[f"pred_log_ret_{h}"]
        tgt = data[f"target_log_ret_{h}"]
        msk = data[f"mask_log_ret_{h}"].astype(bool)
        # gross 30s MFE (in ticks, model-target — proxy for realized excursion)
        gross_mfe_long = data["target_pred_mfe_30s_ticks"]  # already signed long-favorable
        gross_mae_long = data["target_pred_mae_30s_ticks"]  # signed long-unfavorable (negative)
        # FIFO realized: target_fifo_tp4sl3_net is the actual fifo replay result per sample
        fifo_net_long = data["target_fifo_tp4sl3_net"]
        # require finite tgt + finite fifo for ranking validity
        msk = msk & np.isfinite(tgt) & np.isfinite(pred) & np.isfinite(fifo_net_long)
        idx_all = np.where(msk)[0]
        if len(idx_all) == 0:
            continue
        for side in SIDES:
            sign = 1.0 if side == "long" else -1.0
            score = sign * pred  # higher = more confident in this side
            score_v = score[idx_all]
            tgt_v = tgt[idx_all] * sign  # signed target in side direction
            # gross MFE for the side: long uses MFE; short uses -MAE (favorable excursion downward)
            if side == "long":
                gross_v = gross_mfe_long[idx_all]
            else:
                gross_v = -gross_mae_long[idx_all]
            # FIFO net signed for side
            if side == "long":
                fifo_v = fifo_net_long[idx_all]
            else:
                fifo_v = -fifo_net_long[idx_all]  # mirror: short profits when long-fifo loses
            di = day_idx[idx_all]
            order = np.argsort(-score_v)  # descending
            for tier_name, frac in TIERS:
                n_tier = int(len(order) * frac)
                if n_tier < 1:
                    continue
                sel = order[:n_tier]
                p_t = score_v[sel] * sign  # back to original sign
                tg_t = tgt_v[sel] * sign
                gr_t = gross_v[sel]
                fi_t = fifo_v[sel]
                d_t = di[sel]
                # day_conc
                _, counts = np.unique(d_t, return_counts=True)
                day_conc = float(counts.max() / counts.sum()) if counts.size else 1.0
                # CI_low_95 on fifo_net mean (bootstrap-free: normal approx t-CI)
                if len(fi_t) >= 2:
                    se = float(np.std(fi_t, ddof=1) / np.sqrt(len(fi_t)))
                    ci_low = float(np.mean(fi_t) - 1.96 * se)
                else:
                    ci_low = float("nan")
                # hit rate (signed target direction matches side)
                hit = float((tgt_v[sel] > 0).mean())
                rows.append({
                    "horizon": h,
                    "side": side,
                    "tier": tier_name,
                    "n_fills": int(n_tier),
                    "ic_pearson": safe_pearson(p_t, tg_t * sign),  # tg_t*sign = signed actual target
                    "mean_target_signed": float(tg_t.mean()),
                    "hit_rate": hit,
                    "gross_mfe_ticks": float(gr_t.mean()),
                    "fifo_tp4sl3_net_mean": float(fi_t.mean()),
                    "fifo_ci_low_95": ci_low,
                    "day_conc": day_conc,
                    "pass_strict_gate": bool(n_tier >= 50 and day_conc <= 0.20 and ci_low > 0),
                })
    df_t = pd.DataFrame(rows)
    df_t.to_csv(OUT / "tier_ic_perf.csv", index=False)
    log(f"Tier rows: {len(df_t)} (sample below)")
    log(df_t.head(20).to_string(index=False))

    # ----- Promotion summary -----
    promoted = df_t[df_t["pass_strict_gate"]].copy().sort_values("fifo_tp4sl3_net_mean", ascending=False)
    promoted.to_csv(OUT / "promotion_summary.csv", index=False)
    log(f"\nPromoted cells (HC #408 strict-gate pass): {len(promoted)}")
    if len(promoted):
        log(promoted.to_string(index=False))

    # ----- VERDICT -----
    verdict = {
        "ts": datetime.now().isoformat(),
        "n_samples_total": int(N),
        "n_days": len(ALL_DAYS),
        "ic_headline": df_ic.set_index("horizon")["ic_pearson"].to_dict(),
        "ic_spearman": df_ic.set_index("horizon")["ic_spearman"].to_dict(),
        "promoted_count": int(len(promoted)),
        "promoted_cells": promoted[["horizon", "side", "tier", "n_fills",
                                     "fifo_tp4sl3_net_mean", "fifo_ci_low_95",
                                     "day_conc", "gross_mfe_ticks", "hit_rate"]].to_dict(orient="records")
            if len(promoted) else [],
        "book_gate_status": "DEAD (raw=0.002673, tanh~=0.003)",
        "hc402_min_15d_clear": len(ALL_DAYS) >= 15,
        "comparison_v33_15d_promoted": 21,  # per SESSION_STATE 21 v3.3 cells promoted on 15d
    }
    (OUT / "VERDICT.json").write_text(json.dumps(verdict, indent=2, default=str))
    log(f"\nVERDICT written to {OUT}/VERDICT.json")
    log(f"   promoted={verdict['promoted_count']} | hc402_clear={verdict['hc402_min_15d_clear']}")
    return verdict


if __name__ == "__main__":
    main()
