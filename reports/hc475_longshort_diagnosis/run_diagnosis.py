#!/usr/bin/env python3
"""
HC #475 Long/Short Diagnosis — produces all 5 reports for the
diagnostic suite from the v3.4.2 multi-head OOT NPZ.

This is the SAME NPZ that the FIFO + adaptive-exit configs use as
input, so any asymmetry we find here propagates downstream to the
99.84% short fill ratio.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

LVL3 = Path("/home/jupiter/Lvl3Quant")
NPZ = LVL3 / "output/hc432_v342_47day_validation/fold_00_ep1_oot_inference_47day_hc432.npz"
OUT = LVL3 / "reports/hc475_longshort_diagnosis"
OUT.mkdir(parents=True, exist_ok=True)

print(f"[load] {NPZ}")
arrs = {k: v for k, v in np.load(NPZ, allow_pickle=False).items()}
sample_dates = arrs["sample_dates"].astype(str)
n_total = sample_dates.shape[0]
unique_dates = sorted(set(sample_dates.tolist()))
print(f"[load] n_samples={n_total:,} n_dates={len(unique_dates)}")

# ============================================================
# REPORT 1 — LABEL DISTRIBUTION
# ============================================================
print("\n[01] Label distribution analysis")

LABEL_HORIZONS = [
    ("target_log_ret_1s",  "mask_log_ret_1s",  "1s"),
    ("target_log_ret_5s",  "mask_log_ret_5s",  "5s"),
    ("target_log_ret_10s", "mask_log_ret_10s", "10s"),
    ("target_log_ret_30s", "mask_log_ret_30s", "30s"),
    ("target_log_ret_60s", "mask_log_ret_60s", "60s"),
]
P_UP_HORIZONS = [
    ("target_p_up_5s", "mask_p_up_5s", "5s"),
    ("target_p_up_10s", "mask_p_up_10s", "10s"),
    ("target_p_up_30s", "mask_p_up_30s", "30s"),
    ("target_p_up_60s", "mask_p_up_60s", "60s"),
]

label_summary = []
for tgt, msk, hor in LABEL_HORIZONS:
    y = arrs[tgt]
    m = arrs[msk].astype(bool)
    y = y[m]
    if y.size == 0:
        continue
    n_pos = int((y > 0).sum())
    n_neg = int((y < 0).sum())
    n_zero = int((y == 0).sum())
    frac_pos = n_pos / y.size
    frac_neg = n_neg / y.size
    mean_pos = float(y[y > 0].mean()) if n_pos else 0.0
    mean_neg = float(y[y < 0].mean()) if n_neg else 0.0
    med_pos = float(np.median(y[y > 0])) if n_pos else 0.0
    med_neg = float(np.median(y[y < 0])) if n_neg else 0.0
    label_summary.append({
        "horizon": hor,
        "n_valid": int(y.size),
        "n_pos": n_pos, "n_neg": n_neg, "n_zero": n_zero,
        "frac_pos": frac_pos, "frac_neg": frac_neg,
        "frac_zero": n_zero / y.size,
        "mean_pos": mean_pos, "mean_neg": mean_neg,
        "median_pos": med_pos, "median_neg": med_neg,
        "abs_mean_ratio_neg_over_pos": abs(mean_neg) / mean_pos if mean_pos > 0 else float("inf"),
    })
label_summary = pd.DataFrame(label_summary)

# p_up label distribution (these are binary labels — the model literally predicts P(up))
pup_summary = []
for tgt, msk, hor in P_UP_HORIZONS:
    y = arrs[tgt]; m = arrs[msk].astype(bool); y = y[m]
    if y.size == 0: continue
    # p_up labels can be {0,1} or in [0,1]; check
    pup_summary.append({
        "horizon": hor,
        "n_valid": int(y.size),
        "mean": float(y.mean()),
        "median": float(np.median(y)),
        "frac_above_0.5": float((y > 0.5).mean()),
        "frac_equals_1": float((y == 1.0).mean()),
        "frac_equals_0": float((y == 0.0).mean()),
    })
pup_summary = pd.DataFrame(pup_summary)

# Per-day asymmetry trajectory at 10s horizon (the canonical trading horizon)
y10 = arrs["target_log_ret_10s"]; m10 = arrs["mask_log_ret_10s"].astype(bool)
per_day = []
for d in unique_dates:
    mask = (sample_dates == d) & m10
    yy = y10[mask]
    if yy.size == 0: continue
    per_day.append({
        "date": d,
        "n": int(yy.size),
        "frac_pos": float((yy > 0).mean()),
        "frac_neg": float((yy < 0).mean()),
        "mean": float(yy.mean()),
        "median": float(np.median(yy)),
    })
per_day_df = pd.DataFrame(per_day)
per_day_df.to_parquet(OUT / "label_per_day_10s.parquet", index=False)

# Save and write report
label_summary.to_parquet(OUT / "label_summary.parquet", index=False)
pup_summary.to_parquet(OUT / "pup_summary.parquet", index=False)

# ============================================================
# REPORT 2 — RAW SIGNAL IC (LONG-SIDE vs SHORT-SIDE)
# ============================================================
print("[02] Raw-signal IC analysis (long-side vs short-side)")

def signed_ic_split(pred, tgt, mask):
    """Returns (overall_pearson, long_pearson, short_pearson, n_long_signal, n_short_signal)."""
    m = mask.astype(bool) & np.isfinite(pred) & np.isfinite(tgt)
    p = pred[m]; t = tgt[m]
    if p.size < 100:
        return float("nan"), float("nan"), float("nan"), 0, 0
    overall = float(np.corrcoef(p, t)[0, 1])
    long_idx = p > 0; short_idx = p < 0
    n_l = int(long_idx.sum()); n_s = int(short_idx.sum())
    ic_l = float(np.corrcoef(p[long_idx], t[long_idx])[0, 1]) if n_l >= 100 else float("nan")
    ic_s = float(np.corrcoef(p[short_idx], t[short_idx])[0, 1]) if n_s >= 100 else float("nan")
    return overall, ic_l, ic_s, n_l, n_s

# Map each horizon to its prediction key
HEADS = [
    ("pred_log_ret_1s",  "target_log_ret_1s",  "mask_log_ret_1s",  "1s"),
    ("pred_log_ret_5s",  "target_log_ret_5s",  "mask_log_ret_5s",  "5s"),
    ("pred_log_ret_10s", "target_log_ret_10s", "mask_log_ret_10s", "10s"),
    ("pred_log_ret_30s", "target_log_ret_30s", "mask_log_ret_30s", "30s"),
    ("pred_log_ret_60s", "target_log_ret_60s", "mask_log_ret_60s", "60s"),
]

ic_rows = []
mag_rows = []
for pk, tk, mk, hor in HEADS:
    p = arrs[pk]; t = arrs[tk]; m = arrs[mk].astype(bool)
    overall, ic_l, ic_s, n_l, n_s = signed_ic_split(p, t, m)
    # Magnitude distribution
    pv = p[m]
    pos = pv[pv > 0]; neg = pv[pv < 0]
    mag_rows.append({
        "horizon": hor,
        "n_pos_pred": int(pos.size),
        "n_neg_pred": int(neg.size),
        "frac_pred_pos": pos.size / pv.size if pv.size else float("nan"),
        "frac_pred_neg": neg.size / pv.size if pv.size else float("nan"),
        "mean_pos": float(pos.mean()) if pos.size else float("nan"),
        "mean_neg": float(neg.mean()) if neg.size else float("nan"),
        "p99_pos_mag": float(np.quantile(pos, 0.99)) if pos.size else float("nan"),
        "p99_neg_mag": float(-np.quantile(neg, 0.01)) if neg.size else float("nan"),  # positive number = mag of bottom 1%
        "max_pos": float(pos.max()) if pos.size else float("nan"),
        "max_neg_mag": float(-neg.min()) if neg.size else float("nan"),
        "asymmetry_mag_neg_over_pos": (float(-np.quantile(neg, 0.01)) / float(np.quantile(pos, 0.99))) if pos.size and neg.size else float("nan"),
    })
    ic_rows.append({
        "horizon": hor,
        "overall_IC": overall,
        "IC_long_side": ic_l,
        "IC_short_side": ic_s,
        "n_long_pred": n_l,
        "n_short_pred": n_s,
        "long_short_ratio": n_l / n_s if n_s else float("inf"),
        # Both-sides competency per HC #475 R2
        "best_side_IC": max(abs(ic_l) if not np.isnan(ic_l) else 0, abs(ic_s) if not np.isnan(ic_s) else 0),
        "competency_pass": (abs(ic_l) >= 0.5 * max(abs(ic_l), abs(ic_s)) if not (np.isnan(ic_l) or np.isnan(ic_s)) else False),
    })
ic_df = pd.DataFrame(ic_rows)
mag_df = pd.DataFrame(mag_rows)
ic_df.to_parquet(OUT / "ic_by_horizon.parquet", index=False)
mag_df.to_parquet(OUT / "magnitude_asymmetry.parquet", index=False)

# Per-day IC at 10s
per_day_ic = []
for d in unique_dates:
    dm = sample_dates == d
    p = arrs["pred_log_ret_10s"]; t = arrs["target_log_ret_10s"]
    m = arrs["mask_log_ret_10s"].astype(bool) & dm
    overall, ic_l, ic_s, n_l, n_s = signed_ic_split(p, t, m)
    per_day_ic.append({
        "date": d, "overall_IC": overall, "IC_long": ic_l, "IC_short": ic_s,
        "n_long": n_l, "n_short": n_s,
    })
per_day_ic_df = pd.DataFrame(per_day_ic)
per_day_ic_df.to_parquet(OUT / "per_day_ic_10s.parquet", index=False)

# ============================================================
# REPORT 3 — THRESHOLD ATTRIBUTION (TOP-3 TRIPLETS)
# ============================================================
print("[03] Threshold attribution for top-3 triplets")

# Top triplets from the FIFO survival report (best Sharpe, all profitable):
#   trip10_logret60s+logret30sq50+fifotp8sl5_top10 (top10 confidence)
#   trip07_logret60s+logret30sq50+fifotp8sl5      (top5 confidence)
#   trip09_logret60s+logret10sq50+fifotp8sl5      (top5 confidence)
NON_DIRECTIONAL = {"pred_fifo_tp4sl3_net", "pred_fifo_tp8sl5_net",
                   "pred_fifo_tp4sl3_hit_tp", "pred_fifo_tp8sl5_hit_tp",
                   "pred_pred_mfe_30s_ticks", "pred_pred_mae_30s_ticks",
                   "pred_pred_mfe_60s_ticks", "pred_pred_mae_60s_ticks",
                   "pred_pred_realized_vol_30s_ticks", "pred_pred_time_to_mfe_secs",
                   "pred_p_reversal_15s", "pred_p_reversal_30s", "pred_p_reversal_60s"}

def directional_signal(name, raw):
    """Match stream_continuation_backtest.directional_signal semantics."""
    if "p_up" in name:
        return raw - 0.5  # center binary prob
    if "q50" in name:
        return raw
    if name in NON_DIRECTIONAL:
        return raw  # magnitude-only, treated as confluence gate
    return raw

def confluence_count(arrs, heads, conf_top_pct):
    n = arrs[heads[0]].shape[0]
    long_mask = np.ones(n, dtype=bool)
    short_mask = np.ones(n, dtype=bool)
    head_long_indep = {}
    head_short_indep = {}
    for h in heads:
        raw = arrs[h].astype(np.float64)
        sig = directional_signal(h, raw)
        finite = np.isfinite(sig)
        if h in NON_DIRECTIONAL:
            mag = np.abs(sig)
            thr = np.quantile(mag[finite], 1.0 - conf_top_pct)
            keep = finite & (mag >= thr)
            head_long_indep[h] = int(keep.sum())
            head_short_indep[h] = int(keep.sum())
            long_mask &= keep
            short_mask &= keep
        else:
            pos = sig > 0; neg = sig < 0
            if pos.any():
                thr_pos = np.quantile(sig[pos], 1.0 - conf_top_pct)
                kp = pos & (sig >= thr_pos)
                head_long_indep[h] = int(kp.sum())
                long_mask &= kp
            else:
                head_long_indep[h] = 0
                long_mask &= False
            if neg.any():
                thr_neg = np.quantile(-sig[neg], 1.0 - conf_top_pct)
                kn = neg & (-sig >= thr_neg)
                head_short_indep[h] = int(kn.sum())
                short_mask &= kn
            else:
                head_short_indep[h] = 0
                short_mask &= False
    return int(long_mask.sum()), int(short_mask.sum()), head_long_indep, head_short_indep

triplets = [
    ("trip10", ["pred_log_ret_60s", "pred_log_ret_30s_q50", "pred_fifo_tp8sl5_net"], 0.10),
    ("trip07", ["pred_log_ret_60s", "pred_log_ret_30s_q50", "pred_fifo_tp8sl5_net"], 0.05),
    ("trip09", ["pred_log_ret_60s", "pred_log_ret_10s_q50", "pred_fifo_tp8sl5_net"], 0.05),
    ("pair01", ["pred_log_ret_1s", "pred_p_up_5s"], 0.05),
    ("trip03", ["pred_log_ret_5s", "pred_p_up_5s", "pred_log_ret_1s"], 0.05),
]

thr_rows = []
indep_rows = []
for name, heads, q in triplets:
    n_l, n_s, hl, hs = confluence_count(arrs, heads, q)
    thr_rows.append({
        "config": name, "heads": ",".join(heads), "conf_top_pct": q,
        "n_long_triggers_confluence": n_l,
        "n_short_triggers_confluence": n_s,
        "long_short_ratio": (n_l / n_s) if n_s else float("inf"),
        "short_share": n_s / (n_l + n_s) if (n_l + n_s) else float("nan"),
    })
    for h in heads:
        indep_rows.append({
            "config": name, "head": h,
            "independent_n_long": hl[h],
            "independent_n_short": hs[h],
        })
thr_df = pd.DataFrame(thr_rows)
indep_df = pd.DataFrame(indep_rows)
thr_df.to_parquet(OUT / "threshold_triggers.parquet", index=False)
indep_df.to_parquet(OUT / "threshold_per_head_independent.parquet", index=False)

# ============================================================
# REPORT 4 — EXECUTION FILTER ATTRIBUTION
# ============================================================
print("[04] Execution filter attribution (signal triggers -> fills)")

# Pull fills from the FIFO output
fills = pd.read_parquet(LVL3 / "output/stream_backtest_v2/surviving_canonical_fifo_fills.parquet")
print(f"  fills rows: {len(fills):,}; cols: {list(fills.columns)[:15]}")

# Count fills by side per config (also count signal triggers from above)
exec_rows = []
for name, heads, q in triplets:
    n_l_trig, n_s_trig, _, _ = confluence_count(arrs, heads, q)
    sub = fills[fills["config"].str.startswith(name)]
    if "direction" in sub.columns:
        n_l_fill = int((sub["direction"] == "long").sum())
        n_s_fill = int((sub["direction"] == "short").sum())
    elif "side" in sub.columns:
        n_l_fill = int((sub["side"] == "long").sum())
        n_s_fill = int((sub["side"] == "short").sum())
    else:
        n_l_fill = n_s_fill = -1
    exec_rows.append({
        "config_prefix": name,
        "signal_triggers_long": n_l_trig,
        "signal_triggers_short": n_s_trig,
        "fills_long": n_l_fill,
        "fills_short": n_s_fill,
        "fill_rate_long": (n_l_fill / n_l_trig) if n_l_trig else float("nan"),
        "fill_rate_short": (n_s_fill / n_s_trig) if n_s_trig else float("nan"),
    })
exec_df = pd.DataFrame(exec_rows)
exec_df.to_parquet(OUT / "execution_attribution.parquet", index=False)

# Aggregate totals across all surviving configs
total_long_trig = sum(r["signal_triggers_long"] for r in exec_rows)
total_short_trig = sum(r["signal_triggers_short"] for r in exec_rows)
total_long_fill = sum(r["fills_long"] for r in exec_rows if r["fills_long"] >= 0)
total_short_fill = sum(r["fills_short"] for r in exec_rows if r["fills_short"] >= 0)
print(f"  AGGREGATE (5 configs): triggers long={total_long_trig:,} short={total_short_trig:,} "
      f"fills long={total_long_fill:,} short={total_short_fill:,}")

# Also: at the RAW prediction level (no confluence, just sign of pred_log_ret_10s) what's the long/short split?
p10 = arrs["pred_log_ret_10s"]; m10 = arrs["mask_log_ret_10s"].astype(bool)
pv = p10[m10]
raw_long = int((pv > 0).sum()); raw_short = int((pv < 0).sum())
print(f"  RAW pred_log_ret_10s sign split (no threshold): long={raw_long:,} short={raw_short:,} "
      f"(frac_short={raw_short/(raw_long+raw_short):.3f})")

# ============================================================
# Save a consolidated machine-readable summary
# ============================================================
summary_json = {
    "n_total_events": int(n_total),
    "n_dates": len(unique_dates),
    "raw_pred_log_ret_10s_sign_split": {
        "n_long": raw_long, "n_short": raw_short,
        "short_share": raw_short / (raw_long + raw_short),
    },
    "label_summary": label_summary.to_dict("records"),
    "pup_summary": pup_summary.to_dict("records"),
    "ic_by_horizon": ic_df.to_dict("records"),
    "magnitude_asymmetry": mag_df.to_dict("records"),
    "threshold_triggers": thr_df.to_dict("records"),
    "execution_attribution": exec_df.to_dict("records"),
    "aggregate_triggers_vs_fills": {
        "triggers_long": total_long_trig, "triggers_short": total_short_trig,
        "fills_long": total_long_fill, "fills_short": total_short_fill,
        "trigger_short_share": total_short_trig / (total_long_trig + total_short_trig) if (total_long_trig + total_short_trig) else float("nan"),
        "fill_short_share": total_short_fill / (total_long_fill + total_short_fill) if (total_long_fill + total_short_fill) else float("nan"),
    },
}

(OUT / "diagnosis_summary.json").write_text(json.dumps(summary_json, indent=2, default=float))
print(f"\n[done] summary -> {OUT/'diagnosis_summary.json'}")
print(json.dumps({
    "raw_short_share_10s": summary_json["raw_pred_log_ret_10s_sign_split"]["short_share"],
    "trigger_short_share": summary_json["aggregate_triggers_vs_fills"]["trigger_short_share"],
    "fill_short_share":   summary_json["aggregate_triggers_vs_fills"]["fill_short_share"],
    "ic_10s": [r for r in summary_json["ic_by_horizon"] if r["horizon"] == "10s"],
}, indent=2, default=float))
