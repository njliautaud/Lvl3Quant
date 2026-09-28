#!/usr/bin/env python3
"""
v342_pred_asymmetry_v1
======================

Diagnostic: is CNN-Mamba v3.4.2 intrinsically biased at top-confidence,
or is the asymmetry regime-driven? Does the 30s head agree with the 5s head?

Inputs: /home/jupiter/Lvl3Quant/output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate/oot_YYYYMMDD.npz
Outputs: /home/jupiter/Lvl3Quant/output/v342_pred_asymmetry_v1/
MLflow exp: v342_pred_asymmetry_v1
"""
import os
import sys
import glob
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import mlflow

ROOT = Path("/home/jupiter/Lvl3Quant")
PRED_DIR = ROOT / "output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate"
MINBAR_DIR = ROOT / "data/processed/mbo_minute_bars_v1"
OUT_DIR = ROOT / "output/v342_pred_asymmetry_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HEADS = ["log_ret_1s", "log_ret_5s", "log_ret_10s", "log_ret_30s"]
QUANTILES = [0.99, 0.95, 0.90, 0.80, 0.50, 0.0]  # top-1%, top-5%, ..., all
QLABELS  = ["top-1%", "top-5%", "top-10%", "top-20%", "top-50%", "all"]

# -------------------- regime classification --------------------

def classify_day(date_str: str) -> dict:
    """Return {'close_to_close_pct': float, 'regime': 'green'/'red'/'flat'} for date.
    Uses ES minute bar close-to-close. Flat = |chg| < 0.10%.
    """
    fp = MINBAR_DIR / f"{date_str}.parquet"
    if not fp.exists():
        return {"close_to_close_pct": np.nan, "regime": "unknown"}
    try:
        df = pd.read_parquet(fp, columns=["close"])
        if len(df) < 5:
            return {"close_to_close_pct": np.nan, "regime": "unknown"}
        first_close = float(df["close"].iloc[0])
        last_close = float(df["close"].iloc[-1])
        chg_pct = (last_close - first_close) / first_close * 100.0
        if chg_pct > 0.10:
            r = "green"
        elif chg_pct < -0.10:
            r = "red"
        else:
            r = "flat"
        return {"close_to_close_pct": chg_pct, "regime": r}
    except Exception as e:
        return {"close_to_close_pct": np.nan, "regime": "unknown", "err": str(e)}

# -------------------- core analysis --------------------

def load_day(fp: Path):
    d = np.load(fp, allow_pickle=True)
    out = {}
    for h in HEADS:
        p = np.asarray(d[f"pred_{h}"], dtype=np.float64)
        t = np.asarray(d[f"target_{h}"], dtype=np.float64) if f"target_{h}" in d.files else None
        m = np.asarray(d[f"mask_{h}"], dtype=np.float64) if f"mask_{h}" in d.files else np.ones_like(p)
        out[h] = {"pred": p, "target": t, "mask": m}
    # p_up labels (binary) for 5s/10s/30s as auxiliary truth
    for k in ["p_up_5s", "p_up_10s", "p_up_30s"]:
        if f"target_{k}" in d.files:
            out[k + "_target"] = np.asarray(d[f"target_{k}"], dtype=np.float64)
    return out

def sign_dist_for_head(pred: np.ndarray, mask: np.ndarray = None) -> dict:
    """Per-quantile sign distribution table for a single head.
    For each Q in QUANTILES: pick samples where |pred| >= Q-quantile of |pred|.
    Return %long, %short, %zero, n.
    """
    if mask is not None:
        keep = (mask > 0.5) & np.isfinite(pred)
        pred = pred[keep]
    else:
        pred = pred[np.isfinite(pred)]
    if len(pred) == 0:
        return {ql: {"n": 0, "pct_long": np.nan, "pct_short": np.nan, "pct_zero": np.nan, "skew": np.nan} for ql in QLABELS}
    absp = np.abs(pred)
    rows = {}
    for q, ql in zip(QUANTILES, QLABELS):
        if q == 0.0:
            sel = np.ones_like(pred, dtype=bool)
        else:
            thr = np.quantile(absp, q)
            sel = absp >= thr
        sub = pred[sel]
        n = int(sel.sum())
        if n == 0:
            rows[ql] = {"n": 0, "pct_long": np.nan, "pct_short": np.nan, "pct_zero": np.nan, "skew": np.nan}
            continue
        n_long = int((sub > 0).sum())
        n_short = int((sub < 0).sum())
        n_zero = int((sub == 0).sum())
        rows[ql] = {
            "n": n,
            "pct_long": 100.0 * n_long / n,
            "pct_short": 100.0 * n_short / n,
            "pct_zero": 100.0 * n_zero / n,
            "skew": (n_long - n_short) / n,
        }
    return rows

def head_agreement_top_quantile(day_data: dict, q: float = 0.95) -> dict:
    """For top-(1-q)% samples by |pred_5s| (the canonical head),
    compare sign of 1s / 5s / 10s / 30s heads on the SAME indices.
    """
    p5 = day_data["log_ret_5s"]["pred"]
    m5 = day_data["log_ret_5s"]["mask"]
    valid = (m5 > 0.5) & np.isfinite(p5)
    p5_valid = p5[valid]
    if len(p5_valid) == 0:
        return {}
    thr = np.quantile(np.abs(p5_valid), q)
    sel_full = valid & (np.abs(p5) >= thr)
    n_sel = int(sel_full.sum())
    if n_sel == 0:
        return {"n": 0}

    signs = {}
    for h in HEADS:
        p = day_data[h]["pred"]
        s = np.sign(p[sel_full])
        signs[h] = s

    out = {"n": n_sel}
    # all-agree (5s, 30s)
    s5 = signs["log_ret_5s"]; s30 = signs["log_ret_30s"]
    out["pct_5s_pos"] = 100.0 * float((s5 > 0).mean())
    out["pct_5s_neg"] = 100.0 * float((s5 < 0).mean())
    out["pct_30s_pos"] = 100.0 * float((s30 > 0).mean())
    out["pct_30s_neg"] = 100.0 * float((s30 < 0).mean())
    out["pct_5s_30s_agree"] = 100.0 * float((s5 * s30 > 0).mean())
    out["pct_5s_30s_disagree"] = 100.0 * float((s5 * s30 < 0).mean())
    # 3-way 1s/5s/30s
    s1 = signs["log_ret_1s"]
    all3agree = ((s1 * s5 > 0) & (s5 * s30 > 0))
    out["pct_1s_5s_30s_agree"] = 100.0 * float(all3agree.mean())
    # 1s vs 5s
    out["pct_1s_5s_agree"] = 100.0 * float((s1 * s5 > 0).mean())
    out["pct_1s_30s_agree"] = 100.0 * float((s1 * s30 > 0).mean())
    return out

def calibration_top_quantile(day_data: dict, q: float = 0.95) -> dict:
    """For top-(1-q)% by |pred_5s|, what's the realized 5s and target_p_up_5s?
    Sign accuracy + magnitude calibration.
    """
    out = {}
    for h, p_up_key in [("log_ret_1s", None),
                         ("log_ret_5s", "p_up_5s_target"),
                         ("log_ret_10s", "p_up_10s_target"),
                         ("log_ret_30s", "p_up_30s_target")]:
        p = day_data[h]["pred"]
        m = day_data[h]["mask"]
        t = day_data[h]["target"]
        valid = (m > 0.5) & np.isfinite(p) & np.isfinite(t) if t is not None else (m > 0.5) & np.isfinite(p)
        if valid.sum() < 10:
            out[h] = {"n": 0}
            continue
        absp = np.abs(p[valid])
        thr = np.quantile(absp, q)
        sel_full = valid & (np.abs(p) >= thr)
        n = int(sel_full.sum())
        if n == 0:
            out[h] = {"n": 0}
            continue
        psub = p[sel_full]
        rec = {"n": n}
        if t is not None and np.isfinite(t[sel_full]).any():
            tsub = t[sel_full]
            ok = np.isfinite(tsub)
            psub_ok = psub[ok]; tsub_ok = tsub[ok]
            if len(psub_ok) > 5:
                sign_acc = float((np.sign(psub_ok) == np.sign(tsub_ok)).mean())
                # mean realized return for long-side & short-side predictions
                long_mask = psub_ok > 0
                short_mask = psub_ok < 0
                rec["sign_acc"] = sign_acc
                rec["realized_mean_when_long_pred"] = float(tsub_ok[long_mask].mean()) if long_mask.any() else np.nan
                rec["realized_mean_when_short_pred"] = float(tsub_ok[short_mask].mean()) if short_mask.any() else np.nan
                rec["realized_overall_mean"] = float(tsub_ok.mean())
                rec["pred_overall_mean"] = float(psub_ok.mean())
        # binary p_up labels for the longer horizons (always available)
        if p_up_key and p_up_key in day_data:
            pup = day_data[p_up_key][sel_full]
            ok = np.isfinite(pup)
            if ok.sum() > 5:
                psub_ok = psub[ok]; pup_ok = pup[ok]
                # convert predicted log-ret sign to predicted up=1/down=0
                pred_up = (psub_ok > 0).astype(float)
                rec["binary_acc_vs_p_up"] = float((pred_up == pup_ok).mean())
                # base rate among selected
                rec["pup_base_rate"] = float(pup_ok.mean())
                rec["pred_up_rate"] = float(pred_up.mean())
        out[h] = rec
    return out

# -------------------- run --------------------

def main():
    mlflow.set_tracking_uri("http://localhost:5000")
    try:
        mlflow.set_experiment("v342_pred_asymmetry_v1")
        mlflow_active = True
    except Exception as e:
        print(f"[warn] MLflow set_experiment failed: {e}")
        mlflow_active = False

    run_ctx = mlflow.start_run(run_name=f"asymmetry_{datetime.now():%Y%m%d_%H%M%S}") if mlflow_active else None

    files = sorted(glob.glob(str(PRED_DIR / "*.npz")))
    print(f"[info] found {len(files)} OOT prediction files")
    skipped = []

    # Per-day records
    per_day_rows = []          # one row per (day, head, quantile)
    per_day_agree = []         # one row per day (top-5% head agreement)
    per_day_calib = []         # one row per (day, head) calibration
    per_day_summary = []       # one row per day (regime + per-head mean sign skew)

    # Global accumulation (across-day pooled)
    global_concat = {h: {"pred": [], "target": [], "mask": []} for h in HEADS}
    global_concat["p_up_5s_target"] = []
    global_concat["p_up_10s_target"] = []
    global_concat["p_up_30s_target"] = []
    global_concat["dates"] = []
    global_concat["regimes"] = []

    skipped = []
    for fp in files:
        date_str = Path(fp).stem.replace("oot_", "")
        # skip files missing the heads we need (e.g. weekend stubs)
        try:
            _d = np.load(fp, allow_pickle=True)
            if not all(f"pred_{h}" in _d.files for h in HEADS):
                skipped.append(date_str)
                continue
        except Exception:
            skipped.append(date_str)
            continue
        regime_info = classify_day(date_str)
        regime = regime_info["regime"]
        c2c = regime_info["close_to_close_pct"]
        day = load_day(Path(fp))

        # per head per quantile
        for h in HEADS:
            rows = sign_dist_for_head(day[h]["pred"], day[h]["mask"])
            for ql, r in rows.items():
                per_day_rows.append({
                    "date": date_str, "regime": regime, "c2c_pct": c2c,
                    "head": h, "quantile": ql,
                    **r
                })

        # head agreement (top 5%)
        agree = head_agreement_top_quantile(day, q=0.95)
        agree.update({"date": date_str, "regime": regime, "c2c_pct": c2c, "q": "top-5%"})
        per_day_agree.append(agree)
        # head agreement (top 1%)
        agree1 = head_agreement_top_quantile(day, q=0.99)
        agree1.update({"date": date_str, "regime": regime, "c2c_pct": c2c, "q": "top-1%"})
        per_day_agree.append(agree1)

        # calibration top 5%
        calib = calibration_top_quantile(day, q=0.95)
        for h, rec in calib.items():
            row = {"date": date_str, "regime": regime, "c2c_pct": c2c, "head": h, "q": "top-5%"}
            row.update(rec)
            per_day_calib.append(row)

        # per-day summary
        s = {"date": date_str, "regime": regime, "c2c_pct": c2c}
        for h in HEADS:
            p = day[h]["pred"]; m = day[h]["mask"]
            v = (m > 0.5) & np.isfinite(p)
            s[f"mean_pred_{h}"] = float(p[v].mean()) if v.any() else np.nan
            s[f"median_pred_{h}"] = float(np.median(p[v])) if v.any() else np.nan
            s[f"pct_long_all_{h}"] = float((p[v] > 0).mean() * 100) if v.any() else np.nan
        per_day_summary.append(s)

        # accumulate for pooled (cross-day) analysis
        n = len(day[HEADS[0]]["pred"])
        for h in HEADS:
            global_concat[h]["pred"].append(day[h]["pred"])
            global_concat[h]["target"].append(day[h]["target"] if day[h]["target"] is not None else np.full(n, np.nan))
            global_concat[h]["mask"].append(day[h]["mask"])
        for k in ["p_up_5s_target", "p_up_10s_target", "p_up_30s_target"]:
            if k in day:
                global_concat[k].append(day[k])
            else:
                global_concat[k].append(np.full(n, np.nan))
        global_concat["dates"].append(np.array([date_str] * n))
        global_concat["regimes"].append(np.array([regime] * n))

    # ---------------- write per-day tables ----------------
    df_sign = pd.DataFrame(per_day_rows)
    df_agree = pd.DataFrame(per_day_agree)
    df_calib = pd.DataFrame(per_day_calib)
    df_sum = pd.DataFrame(per_day_summary)
    df_sign.to_csv(OUT_DIR / "per_day_sign_distribution.csv", index=False)
    df_agree.to_csv(OUT_DIR / "per_day_head_agreement.csv", index=False)
    df_calib.to_csv(OUT_DIR / "per_day_calibration.csv", index=False)
    df_sum.to_csv(OUT_DIR / "per_day_summary.csv", index=False)

    # ---------------- pooled cross-day ----------------
    pooled = {}
    for h in HEADS:
        pooled[h] = {
            "pred": np.concatenate(global_concat[h]["pred"]),
            "target": np.concatenate(global_concat[h]["target"]),
            "mask": np.concatenate(global_concat[h]["mask"]),
        }
    pooled_dates = np.concatenate(global_concat["dates"])
    pooled_regimes = np.concatenate(global_concat["regimes"])
    for k in ["p_up_5s_target", "p_up_10s_target", "p_up_30s_target"]:
        pooled[k] = np.concatenate(global_concat[k])

    # ---- pooled sign distribution per head per quantile ----
    pooled_sign_rows = []
    for h in HEADS:
        rows = sign_dist_for_head(pooled[h]["pred"], pooled[h]["mask"])
        for ql, r in rows.items():
            pooled_sign_rows.append({"head": h, "quantile": ql, **r})
    df_pooled_sign = pd.DataFrame(pooled_sign_rows)
    df_pooled_sign.to_csv(OUT_DIR / "pooled_sign_distribution.csv", index=False)

    # ---- regime-stratified pooled sign distribution ----
    reg_rows = []
    for reg in ["green", "red", "flat"]:
        idx = pooled_regimes == reg
        if idx.sum() == 0:
            continue
        for h in HEADS:
            p = pooled[h]["pred"][idx]
            m = pooled[h]["mask"][idx]
            rows = sign_dist_for_head(p, m)
            for ql, r in rows.items():
                reg_rows.append({"regime": reg, "head": h, "quantile": ql, **r})
    df_reg = pd.DataFrame(reg_rows)
    df_reg.to_csv(OUT_DIR / "regime_stratified_sign_distribution.csv", index=False)

    # ---- pooled head agreement (top 5% by |pred_5s|) ----
    p5 = pooled["log_ret_5s"]["pred"]; m5 = pooled["log_ret_5s"]["mask"]
    valid = (m5 > 0.5) & np.isfinite(p5)
    thr5 = np.quantile(np.abs(p5[valid]), 0.95)
    sel = valid & (np.abs(p5) >= thr5)
    pooled_agree = {"n": int(sel.sum()), "threshold_abs_pred_5s": float(thr5)}
    for h in HEADS:
        s = np.sign(pooled[h]["pred"][sel])
        pooled_agree[f"pct_{h}_pos"] = float((s > 0).mean() * 100)
        pooled_agree[f"pct_{h}_neg"] = float((s < 0).mean() * 100)
    s5 = np.sign(pooled["log_ret_5s"]["pred"][sel])
    s30 = np.sign(pooled["log_ret_30s"]["pred"][sel])
    s1 = np.sign(pooled["log_ret_1s"]["pred"][sel])
    pooled_agree["pct_5s_30s_agree"] = float((s5 * s30 > 0).mean() * 100)
    pooled_agree["pct_5s_30s_disagree"] = float((s5 * s30 < 0).mean() * 100)
    pooled_agree["pct_1s_5s_agree"] = float((s1 * s5 > 0).mean() * 100)
    pooled_agree["pct_1s_30s_agree"] = float((s1 * s30 > 0).mean() * 100)
    pooled_agree["pct_1s_5s_30s_all_agree"] = float(((s1 * s5 > 0) & (s5 * s30 > 0)).mean() * 100)
    with open(OUT_DIR / "pooled_head_agreement_top5pct.json", "w") as f:
        json.dump(pooled_agree, f, indent=2)

    # ---- pooled calibration top 5% per head ----
    pooled_calib = {}
    for h in HEADS:
        p = pooled[h]["pred"]; t = pooled[h]["target"]; m = pooled[h]["mask"]
        valid = (m > 0.5) & np.isfinite(p)
        if valid.sum() == 0:
            continue
        thr = np.quantile(np.abs(p[valid]), 0.95)
        sel = valid & (np.abs(p) >= thr)
        rec = {"n": int(sel.sum())}
        # log-ret target where finite
        tsub = t[sel]; psub = p[sel]
        ok = np.isfinite(tsub) & np.isfinite(psub)
        if ok.sum() > 10:
            po = psub[ok]; to = tsub[ok]
            rec["log_ret_sign_acc"] = float((np.sign(po) == np.sign(to)).mean())
            long_m = po > 0; short_m = po < 0
            rec["realized_mean_when_long_pred"] = float(to[long_m].mean()) if long_m.any() else None
            rec["realized_mean_when_short_pred"] = float(to[short_m].mean()) if short_m.any() else None
            rec["pred_mean"] = float(po.mean())
            rec["realized_mean"] = float(to.mean())
            rec["realized_pct_pos"] = float((to > 0).mean() * 100)
            rec["realized_pct_neg"] = float((to < 0).mean() * 100)
            # IC at top-5%
            rec["spearman_ic_top5pct"] = float(pd.Series(po).corr(pd.Series(to), method="spearman"))
        # p_up binary
        pup_key = {"log_ret_5s": "p_up_5s_target",
                   "log_ret_10s": "p_up_10s_target",
                   "log_ret_30s": "p_up_30s_target"}.get(h)
        if pup_key:
            pup = pooled[pup_key][sel]
            ok2 = np.isfinite(pup) & np.isfinite(psub)
            if ok2.sum() > 10:
                pred_up = (psub[ok2] > 0).astype(float)
                rec["binary_acc_vs_p_up"] = float((pred_up == pup[ok2]).mean())
                rec["pup_base_rate"] = float(pup[ok2].mean())
                rec["pred_up_rate"] = float(pred_up.mean())
        pooled_calib[h] = rec
    with open(OUT_DIR / "pooled_calibration_top5pct.json", "w") as f:
        json.dump(pooled_calib, f, indent=2)

    # ---- regime-stratified head agreement (top 5% by |pred_5s|) within each regime ----
    reg_agree_rows = []
    for reg in ["green", "red", "flat"]:
        idx = pooled_regimes == reg
        if idx.sum() == 0:
            continue
        p5r = pooled["log_ret_5s"]["pred"][idx]
        m5r = pooled["log_ret_5s"]["mask"][idx]
        validr = (m5r > 0.5) & np.isfinite(p5r)
        if validr.sum() == 0:
            continue
        thrr = np.quantile(np.abs(p5r[validr]), 0.95)
        selr = validr & (np.abs(p5r) >= thrr)
        if selr.sum() == 0:
            continue
        s5 = np.sign(p5r[selr])
        s30 = np.sign(pooled["log_ret_30s"]["pred"][idx][selr])
        s1 = np.sign(pooled["log_ret_1s"]["pred"][idx][selr])
        reg_agree_rows.append({
            "regime": reg, "n": int(selr.sum()),
            "pct_5s_pos": float((s5 > 0).mean() * 100),
            "pct_5s_neg": float((s5 < 0).mean() * 100),
            "pct_30s_pos": float((s30 > 0).mean() * 100),
            "pct_30s_neg": float((s30 < 0).mean() * 100),
            "pct_5s_30s_agree": float((s5 * s30 > 0).mean() * 100),
            "pct_1s_5s_agree": float((s1 * s5 > 0).mean() * 100),
            "pct_1s_30s_agree": float((s1 * s30 > 0).mean() * 100),
        })
    df_reg_agree = pd.DataFrame(reg_agree_rows)
    df_reg_agree.to_csv(OUT_DIR / "regime_stratified_head_agreement.csv", index=False)

    # ---- regime-stratified calibration top 5% ----
    reg_cal_rows = []
    for reg in ["green", "red", "flat"]:
        idx = pooled_regimes == reg
        if idx.sum() == 0:
            continue
        for h in HEADS:
            p = pooled[h]["pred"][idx]; t = pooled[h]["target"][idx]; m = pooled[h]["mask"][idx]
            v = (m > 0.5) & np.isfinite(p)
            if v.sum() == 0:
                continue
            thr = np.quantile(np.abs(p[v]), 0.95)
            sel = v & (np.abs(p) >= thr)
            psub = p[sel]; tsub = t[sel]
            ok = np.isfinite(psub) & np.isfinite(tsub)
            if ok.sum() < 10:
                continue
            po = psub[ok]; to = tsub[ok]
            reg_cal_rows.append({
                "regime": reg, "head": h, "n": int(ok.sum()),
                "pct_long_pred": float((po > 0).mean() * 100),
                "pct_short_pred": float((po < 0).mean() * 100),
                "sign_acc": float((np.sign(po) == np.sign(to)).mean()),
                "realized_mean": float(to.mean()),
                "realized_mean_long_pred": float(to[po > 0].mean()) if (po > 0).any() else np.nan,
                "realized_mean_short_pred": float(to[po < 0].mean()) if (po < 0).any() else np.nan,
            })
    df_reg_cal = pd.DataFrame(reg_cal_rows)
    df_reg_cal.to_csv(OUT_DIR / "regime_stratified_calibration.csv", index=False)

    # ---- VERDICT ----
    verdict = build_verdict(df_pooled_sign, df_reg, pooled_agree, pooled_calib,
                            df_reg_agree, df_reg_cal, df_sum)
    with open(OUT_DIR / "VERDICT.md", "w") as f:
        f.write(verdict)

    print("\n" + "=" * 70)
    print(verdict)
    print("=" * 70)

    if mlflow_active:
        try:
            for h in HEADS:
                row = df_pooled_sign[(df_pooled_sign["head"] == h) & (df_pooled_sign["quantile"] == "top-5%")].iloc[0]
                mlflow.log_metric(f"top5pct_pct_long_{h}", float(row["pct_long"]))
                mlflow.log_metric(f"top5pct_pct_short_{h}", float(row["pct_short"]))
            mlflow.log_metric("top5pct_5s_30s_agree_pct", float(pooled_agree["pct_5s_30s_agree"]))
            mlflow.log_metric("top5pct_1s_5s_agree_pct", float(pooled_agree["pct_1s_5s_agree"]))
            mlflow.log_metric("top5pct_1s_30s_agree_pct", float(pooled_agree["pct_1s_30s_agree"]))
            mlflow.log_metric("n_days", len(files))
            mlflow.log_artifacts(str(OUT_DIR))
            mlflow.end_run()
        except Exception as e:
            print(f"[warn] MLflow log failed: {e}")

    print(f"\n[done] outputs in {OUT_DIR}")
    return 0

def build_verdict(df_pooled_sign, df_reg, pooled_agree, pooled_calib,
                  df_reg_agree, df_reg_cal, df_sum) -> str:
    lines = []
    lines.append("# v3.4.2 PRED ASYMMETRY — VERDICT\n")
    lines.append(f"Run date: {datetime.now():%Y-%m-%d %H:%M}\n")
    lines.append(f"OOT days analyzed: {df_sum.shape[0]}  ")
    n_green = int((df_sum["regime"] == "green").sum())
    n_red = int((df_sum["regime"] == "red").sum())
    n_flat = int((df_sum["regime"] == "flat").sum())
    n_unk = int((df_sum["regime"] == "unknown").sum())
    lines.append(f"Regime mix: green={n_green}, red={n_red}, flat={n_flat}, unknown={n_unk}\n")

    # 1) Top-5% sign distribution per head (pooled)
    lines.append("\n## 1. Top-5% sign distribution by head (pooled across all days)\n")
    lines.append("| head | n | %long | %short |")
    lines.append("|------|---|-------|--------|")
    for h in HEADS:
        r = df_pooled_sign[(df_pooled_sign["head"] == h) & (df_pooled_sign["quantile"] == "top-5%")].iloc[0]
        lines.append(f"| {h} | {int(r['n'])} | {r['pct_long']:.1f}% | {r['pct_short']:.1f}% |")

    # All-quantile sweep for 5s and 30s
    lines.append("\n## 2. Sign skew across quantiles (5s and 30s heads)\n")
    lines.append("| quantile | 5s %long | 5s %short | 30s %long | 30s %short |")
    lines.append("|----------|----------|-----------|-----------|------------|")
    for ql in QLABELS:
        r5 = df_pooled_sign[(df_pooled_sign["head"] == "log_ret_5s") & (df_pooled_sign["quantile"] == ql)].iloc[0]
        r30 = df_pooled_sign[(df_pooled_sign["head"] == "log_ret_30s") & (df_pooled_sign["quantile"] == ql)].iloc[0]
        lines.append(f"| {ql} | {r5['pct_long']:.1f}% | {r5['pct_short']:.1f}% | {r30['pct_long']:.1f}% | {r30['pct_short']:.1f}% |")

    # 3) Head agreement at top-5%
    lines.append("\n## 3. Head agreement at top-5% (by |pred_5s|)\n")
    lines.append(f"- n = {pooled_agree['n']}")
    lines.append(f"- 5s sign: {pooled_agree['pct_log_ret_5s_pos']:.1f}% pos, {pooled_agree['pct_log_ret_5s_neg']:.1f}% neg")
    lines.append(f"- 30s sign: {pooled_agree['pct_log_ret_30s_pos']:.1f}% pos, {pooled_agree['pct_log_ret_30s_neg']:.1f}% neg")
    lines.append(f"- 1s sign: {pooled_agree['pct_log_ret_1s_pos']:.1f}% pos, {pooled_agree['pct_log_ret_1s_neg']:.1f}% neg")
    lines.append(f"- 10s sign: {pooled_agree['pct_log_ret_10s_pos']:.1f}% pos, {pooled_agree['pct_log_ret_10s_neg']:.1f}% neg")
    lines.append(f"- 5s ↔ 30s agree: **{pooled_agree['pct_5s_30s_agree']:.1f}%**")
    lines.append(f"- 1s ↔ 5s agree: {pooled_agree['pct_1s_5s_agree']:.1f}%")
    lines.append(f"- 1s ↔ 30s agree: {pooled_agree['pct_1s_30s_agree']:.1f}%")
    lines.append(f"- 1s/5s/30s ALL agree: **{pooled_agree['pct_1s_5s_30s_all_agree']:.1f}%**")

    # 4) Regime breakdown
    lines.append("\n## 4. Regime-stratified top-5% sign (5s head)\n")
    lines.append("| regime | n | %long | %short |")
    lines.append("|--------|---|-------|--------|")
    for reg in ["green", "red", "flat"]:
        rr = df_reg[(df_reg["regime"] == reg) & (df_reg["head"] == "log_ret_5s") & (df_reg["quantile"] == "top-5%")]
        if len(rr):
            r = rr.iloc[0]
            lines.append(f"| {reg} | {int(r['n'])} | {r['pct_long']:.1f}% | {r['pct_short']:.1f}% |")

    lines.append("\n## 4b. Regime-stratified top-5% sign (30s head)\n")
    lines.append("| regime | n | %long | %short |")
    lines.append("|--------|---|-------|--------|")
    for reg in ["green", "red", "flat"]:
        rr = df_reg[(df_reg["regime"] == reg) & (df_reg["head"] == "log_ret_30s") & (df_reg["quantile"] == "top-5%")]
        if len(rr):
            r = rr.iloc[0]
            lines.append(f"| {reg} | {int(r['n'])} | {r['pct_long']:.1f}% | {r['pct_short']:.1f}% |")

    # 5) Calibration
    lines.append("\n## 5. Calibration at top-5% (pooled)\n")
    lines.append("| head | n | sign_acc | realized_mean | when_long_pred | when_short_pred | binary_acc_vs_p_up |")
    lines.append("|------|---|----------|---------------|----------------|-----------------|--------------------|")
    for h in HEADS:
        c = pooled_calib.get(h, {})
        n = c.get("n", 0)
        sa = c.get("log_ret_sign_acc", float("nan"))
        rm = c.get("realized_mean", float("nan"))
        rml = c.get("realized_mean_when_long_pred")
        rms = c.get("realized_mean_when_short_pred")
        ba = c.get("binary_acc_vs_p_up", float("nan"))
        rml_s = f"{rml:.4f}" if isinstance(rml, float) else "-"
        rms_s = f"{rms:.4f}" if isinstance(rms, float) else "-"
        lines.append(f"| {h} | {n} | {sa:.3f} | {rm:.4f} | {rml_s} | {rms_s} | {ba:.3f} |")

    # ---- automated interpretation ----
    lines.append("\n## VERDICT (plain English)\n")

    # extract numbers
    r5 = df_pooled_sign[(df_pooled_sign["head"] == "log_ret_5s") & (df_pooled_sign["quantile"] == "top-5%")].iloc[0]
    r30 = df_pooled_sign[(df_pooled_sign["head"] == "log_ret_30s") & (df_pooled_sign["quantile"] == "top-5%")].iloc[0]
    r1 = df_pooled_sign[(df_pooled_sign["head"] == "log_ret_1s") & (df_pooled_sign["quantile"] == "top-5%")].iloc[0]
    sk5 = r5["pct_short"] - r5["pct_long"]
    sk30 = r30["pct_short"] - r30["pct_long"]

    # asymmetry call
    if abs(sk5) > 60:
        asym_call = f"**SEVERE asymmetry on 5s head**: top-5% predictions are {r5['pct_short']:.0f}% short / {r5['pct_long']:.0f}% long."
    elif abs(sk5) > 30:
        asym_call = f"**Moderate asymmetry on 5s head**: top-5% is {r5['pct_short']:.0f}% short / {r5['pct_long']:.0f}% long."
    else:
        asym_call = f"5s head is roughly balanced at top-5% ({r5['pct_short']:.0f}% short / {r5['pct_long']:.0f}% long)."

    # regime explanation
    reg_call = "Regime breakdown is INCONCLUSIVE."
    try:
        gr = df_reg[(df_reg["regime"] == "green") & (df_reg["head"] == "log_ret_5s") & (df_reg["quantile"] == "top-5%")].iloc[0]
        rd = df_reg[(df_reg["regime"] == "red") & (df_reg["head"] == "log_ret_5s") & (df_reg["quantile"] == "top-5%")].iloc[0]
        gr_skew = gr["pct_short"] - gr["pct_long"]
        rd_skew = rd["pct_short"] - rd["pct_long"]
        if gr_skew > 50 and rd_skew > 50:
            reg_call = (f"Top-5% is short-skewed in BOTH green ({gr['pct_short']:.0f}% short) "
                        f"and red ({rd['pct_short']:.0f}% short) regimes → asymmetry is **INTRINSIC to the model**, "
                        "NOT regime-driven.")
        elif gr_skew > 50 and rd_skew < -10:
            reg_call = (f"Top-5% is short on green days ({gr['pct_short']:.0f}% short) but long on red days "
                        f"({rd['pct_long']:.0f}% long) → model is **mean-reverting against the day's trend** "
                        "(regime-aware contrarian).")
        elif (gr_skew > 50) and (rd_skew > 0 and rd_skew < 50):
            reg_call = (f"Short skew is amplified on green days ({gr['pct_short']:.0f}% short vs red {rd['pct_short']:.0f}% short). "
                        "Mixed signal — model is mostly short but more so when the day is up.")
        else:
            reg_call = (f"Green-day short% = {gr['pct_short']:.0f}, Red-day short% = {rd['pct_short']:.0f}. "
                        "Mixed pattern — see regime table above.")
    except Exception:
        pass

    # head agreement call
    agree_pct = pooled_agree.get("pct_5s_30s_agree", float("nan"))
    if agree_pct < 30:
        head_call = (f"**5s and 30s heads STRONGLY DISAGREE** at top-5%: only {agree_pct:.1f}% agreement on sign. "
                     "These heads encode different (likely conflicting) horizons. "
                     "DO NOT fuse naively. Pick one or build an explicit multi-h model.")
    elif agree_pct < 60:
        head_call = (f"5s and 30s heads partially disagree at top-5% ({agree_pct:.1f}% agreement). "
                     "Confluence-style fusion (require sign-agreement) will throw away most signal. "
                     "Recommend: trade on the 5s head alone (IC is highest there), use 30s only as a "
                     "negative gate (skip when 30s strongly disagrees).")
    else:
        head_call = (f"5s and 30s heads broadly agree at top-5% ({agree_pct:.1f}%) — fusion is viable.")

    # base-rate edge check (the critical test)
    base_rate_check_lines = []
    for h_name, h in [("5s", "log_ret_5s"), ("10s", "log_ret_10s"), ("30s", "log_ret_30s")]:
        c = pooled_calib.get(h, {})
        if "binary_acc_vs_p_up" in c:
            ba = c["binary_acc_vs_p_up"]
            br = c["pup_base_rate"]
            pu = c["pred_up_rate"]
            # if model is ~constant, expected acc = max(br, 1-br); edge above is ba - max(br, 1-br)
            expected_if_constant = max(br, 1 - br) if (pu < 0.05 or pu > 0.95) else max(br, 1 - br)
            edge = ba - expected_if_constant
            sgn = "+" if edge >= 0 else ""
            base_rate_check_lines.append(
                f"  - {h_name}: binary_acc={ba:.3f}, base_rate_up={br:.3f}, pred_up_rate={pu:.3%}, "
                f"edge_above_constant_baseline={sgn}{edge:.3f}"
            )
    if base_rate_check_lines:
        lines.append("\n### Base-rate edge test (CRITICAL):")
        lines.append("If the model just always predicts the majority direction in top-5%, its accuracy = max(base_rate_up, 1 - base_rate_up). Edge over that baseline is what matters.")
        lines.extend(base_rate_check_lines)

    # calibration call
    cal5 = pooled_calib.get("log_ret_5s", {})
    cal_call = "Calibration data missing for 5s head."
    if "log_ret_sign_acc" in cal5:
        sa = cal5["log_ret_sign_acc"]
        rm = cal5.get("realized_mean")
        rml = cal5.get("realized_mean_when_long_pred")
        rms = cal5.get("realized_mean_when_short_pred")
        cond = []
        if sa >= 0.55:
            cond.append(f"sign accuracy at top-5% is {sa*100:.1f}% — **strong edge**")
        elif sa >= 0.51:
            cond.append(f"sign accuracy at top-5% is {sa*100:.1f}% — marginal edge")
        else:
            cond.append(f"sign accuracy at top-5% is {sa*100:.1f}% — **NO edge / coin flip**")
        if isinstance(rms, float) and isinstance(rml, float):
            if rms < 0 and rml > 0:
                cond.append("realized direction matches predicted side (longs go up, shorts go down) — model is CORRECT, just asymmetric")
            elif rms < 0 and rml < 0:
                cond.append("realized return is negative for BOTH long and short predictions — model is wrong on longs, right on shorts (one-sided edge)")
            elif rms > 0 and rml > 0:
                cond.append("realized return is positive for BOTH long and short predictions — model is right on longs, wrong on shorts")
            elif rms > 0 and rml < 0:
                cond.append("realized return is INVERTED — model is systematically wrong, possible sign flip somewhere")
        cal_call = " | ".join(cond)

    lines.append(f"- **Asymmetry:** {asym_call}")
    lines.append(f"- **Regime driver:** {reg_call}")
    lines.append(f"- **5s vs 30s heads:** {head_call}")
    lines.append(f"- **Calibration (5s, top-5%):** {cal_call}")

    # final recommendation
    lines.append("\n### Recommendation\n")
    use_5s = True
    use_30s_fuse = agree_pct >= 60
    sa = cal5.get("log_ret_sign_acc", 0.5)
    if sa < 0.51 and abs(sk5) > 60:
        lines.append("- **REPLACE v3.4.2 for execution research.** "
                     "Top-confidence predictions are heavily one-sided AND sign accuracy is at coin-flip. "
                     "This is consistent with the model latching onto a training-window artifact (likely a "
                     "downtrending training period), not a transferable edge.")
    elif sa >= 0.55 and abs(sk5) > 60 and "INTRINSIC" in reg_call:
        lines.append("- **KEEP v3.4.2 but use SHORT-ONLY execution.** "
                     "The model has a real edge but only on the short side. Don't force long-side trades against "
                     "a model that has no long-side signal. This matches HC #471 finding that shorts have more edge.")
    elif sa >= 0.55 and "regime-aware" in reg_call:
        lines.append("- **KEEP v3.4.2 with regime-aware execution.** "
                     "Model is mean-reverting against trend and the calls are sign-correct on average. "
                     "Use it symmetrically — long signals on red days, short signals on green days.")
    else:
        lines.append("- **KEEP v3.4.2 but gate execution by 1s + 5s head agreement only.** "
                     "Drop 30s from the inference path — it disagrees too often to add information. "
                     "Use top-5% by |pred_5s| with sign-agreement of 1s and 5s as the trade trigger.")

    lines.append(f"- **30s head usage:** "
                 f"{'fuse with 5s' if use_30s_fuse else 'do NOT fuse with 5s. Drop from execution path or use only as a veto when |pred_30s| is large and disagrees.'}")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
