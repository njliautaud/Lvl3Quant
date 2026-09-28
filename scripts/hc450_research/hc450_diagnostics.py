#!/usr/bin/env python3
"""HC #450 R3+R4 — MFE/MAE x confidence + signal smoothness diagnostics.

Inputs: per-date OOT prediction NPZs already on Jupiter.
Outputs: CSVs + PNGs in output/hc450_diagnostics/ and SUMMARY.md.

Idempotent: re-running overwrites.

Note on MFE/MAE proxy: the prediction NPZs do NOT contain per-sample MFE/MAE
arrays (v3.4.2 has the fields but they're masked off for the OOT set). We
therefore use the realized move at horizon h (labels_h, in TICKS) as the
proxy for "directional excursion" within that horizon. This is a lower bound
on true MFE because actual MFE can be larger than the close-of-horizon move.
We label outputs `realized_move` to be honest about this.
"""

import os, sys, json, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/home/jupiter/Lvl3Quant"
OUT  = f"{BASE}/output/hc450_diagnostics"
os.makedirs(OUT, exist_ok=True)

COMMISSION_TICKS = 0.376         # ES_RT_COMMISSION / ES_TICK_VALUE
STRIDE_SEC = 0.25                # 250ms — confirmed in v2/patchtst metadata
HORIZONS = ["1s", "5s", "10s", "30s"]
CONF_BANDS = [0.005, 0.01, 0.05, 0.10, 0.20, 0.50]  # top X
CONF_LABELS = ["top0.5%", "top1%", "top5%", "top10%", "top20%", "top50%", "bottom50%"]


# ---------------------------------------------------------------------------
# Model adapters: produce dict {h: (pred, label_ticks)} per date
# ---------------------------------------------------------------------------

def load_v342(path):
    d = np.load(path, allow_pickle=True)
    out = {}
    for h in HORIZONS:
        p = d[f"pred_log_ret_{h}"]
        t = d[f"target_log_ret_{h}"]
        m = d[f"mask_log_ret_{h}"]
        f = np.isfinite(p) & np.isfinite(t) & (m > 0)
        if f.sum() < 10:
            continue
        out[h] = (p[f].astype(np.float32), t[f].astype(np.float32), f)
    return out

def load_v33(path):
    return load_v342(path)

def load_v2_pt(path):
    d = np.load(path, allow_pickle=True)
    preds = d["predictions"]; labels = d["labels"]
    # v2/PatchTST horizons: 1s, 5s, 10s only
    out = {}
    for i, h in enumerate(["1s", "5s", "10s"]):
        p, t = preds[:, i], labels[:, i]
        f = np.isfinite(p) & np.isfinite(t)
        if f.sum() < 10:
            continue
        out[h] = (p[f].astype(np.float32), t[f].astype(np.float32), f)
    return out


MODELS = {
    "cnn_mamba_v3_4_2": {
        "glob": f"{BASE}/output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate/oot_*.npz",
        "loader": load_v342,
        "horizons": HORIZONS,
    },
    "cnn_mamba_v3_3": {
        "glob": f"{BASE}/output/cnn_mamba_v3_3_uncertainty_weighted/oot_47day_perdate/oot_*.npz",
        "loader": load_v33,
        "horizons": HORIZONS,
    },
    "cnn_mamba_v2": {
        "glob": f"{BASE}/output/cnn_mamba_v2_bulk_oot_v2/*_predictions.npz",
        "loader": load_v2_pt,
        "horizons": ["1s", "5s", "10s"],
    },
    "patchtst": {
        "glob": f"{BASE}/output/patchtst_bulk_oot/*_predictions.npz",
        "loader": load_v2_pt,
        "horizons": ["1s", "5s", "10s"],
    },
}


# ---------------------------------------------------------------------------
# Diagnostic 1: MFE/MAE x confidence x side
# ---------------------------------------------------------------------------

def conf_band_indices(pred_abs):
    """Return dict band_label -> boolean mask."""
    n = len(pred_abs)
    order = np.argsort(-pred_abs)              # descending |pred|
    bands = {}
    prev = 0
    for label, frac in zip(CONF_LABELS[:-1], CONF_BANDS):
        k = max(1, int(np.ceil(frac * n)))
        m = np.zeros(n, dtype=bool)
        m[order[:k]] = True
        bands[label] = m
    # bottom 50%
    k = int(np.ceil(0.50 * n))
    m = np.zeros(n, dtype=bool); m[order[-k:]] = True
    bands["bottom50%"] = m
    return bands


def mfe_mae_table(model_name, per_date):
    """per_date: dict date -> {h: (pred, label_ticks, finite_mask)}"""
    rows = []
    # First concat all dates per horizon
    horizon_buckets = {}
    for date, hd in per_date.items():
        for h, (p, l, _) in hd.items():
            horizon_buckets.setdefault(h, []).append((p, l))
    for h, parts in horizon_buckets.items():
        p = np.concatenate([x[0] for x in parts])
        l = np.concatenate([x[1] for x in parts])
        ap = np.abs(p)
        bands = conf_band_indices(ap)
        for side_name, side_mask in [
            ("long",  p > 0),
            ("short", p < 0),
            ("all",   np.ones_like(p, dtype=bool)),
        ]:
            for band_label, band_mask in bands.items():
                m = band_mask & side_mask
                n = int(m.sum())
                if n < 20:
                    continue
                lm = l[m]
                # Direction-adjusted realized move (signed in the direction of trade)
                #   long  -> +label
                #   short -> -label
                #   all   -> +label (raw)
                if side_name == "short":
                    realized = -lm
                else:
                    realized = lm
                mean_real = float(np.mean(realized))
                p50 = float(np.percentile(realized, 50))
                p90 = float(np.percentile(realized, 90))
                p10 = float(np.percentile(realized, 10))
                wr  = float(np.mean(realized > 0))
                # "MFE" proxy = upside excursion (positive realized), "MAE" = downside
                mfe_p50 = float(np.percentile(realized[realized > 0], 50)) if (realized>0).any() else 0.0
                mfe_p90 = float(np.percentile(realized[realized > 0], 90)) if (realized>0).any() else 0.0
                mae_p50 = float(np.percentile(realized[realized < 0], 50)) if (realized<0).any() else 0.0
                mae_p90 = float(np.percentile(realized[realized < 0], 10)) if (realized<0).any() else 0.0  # 10th = worst 10%
                net = mean_real - COMMISSION_TICKS
                rows.append(dict(
                    side=side_name, conf_band=band_label, horizon=h, n=n,
                    mean_realized_tk=round(mean_real, 4),
                    p50_realized_tk=round(p50, 4),
                    p90_realized_tk=round(p90, 4),
                    p10_realized_tk=round(p10, 4),
                    mfe_p50_tk=round(mfe_p50, 4),
                    mfe_p90_tk=round(mfe_p90, 4),
                    mae_p50_tk=round(mae_p50, 4),
                    mae_p90_tk=round(mae_p90, 4),
                    win_rate=round(wr, 4),
                    net_after_comm_376_tk=round(net, 4),
                ))
    df = pd.DataFrame(rows).sort_values(["horizon", "side", "conf_band"])
    df.to_csv(f"{OUT}/{model_name}_mfe_mae_by_conf.csv", index=False)
    return df


def plot_mfe_mae(model_name, df):
    horizons = sorted(df["horizon"].unique(), key=lambda x: int(x.replace("s","")))
    fig, axes = plt.subplots(1, len(horizons), figsize=(5*len(horizons), 4.5), squeeze=False)
    for ax, h in zip(axes[0], horizons):
        sub = df[(df["horizon"] == h) & (df["side"].isin(["long","short"]))].copy()
        order = ["top0.5%","top1%","top5%","top10%","top20%","top50%","bottom50%"]
        sub["conf_band"] = pd.Categorical(sub["conf_band"], order)
        sub = sub.sort_values(["conf_band","side"])
        x = np.arange(len(order))
        w = 0.4
        longs  = sub[sub["side"]=="long"].set_index("conf_band").reindex(order)["mean_realized_tk"].values
        shorts = sub[sub["side"]=="short"].set_index("conf_band").reindex(order)["mean_realized_tk"].values
        ax.bar(x - w/2, longs,  w, label="long",  color="#2ecc71")
        ax.bar(x + w/2, shorts, w, label="short", color="#e74c3c")
        ax.axhline(COMMISSION_TICKS, ls="--", color="gray", lw=0.8, label="commission 0.376tk")
        ax.axhline(-COMMISSION_TICKS, ls="--", color="gray", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"{h} horizon")
        ax.set_ylabel("mean realized move (ticks, direction-adjusted)")
        ax.legend(fontsize=8)
    fig.suptitle(f"{model_name} — realized move by confidence band & side")
    fig.tight_layout()
    fig.savefig(f"{OUT}/{model_name}_mfe_mae_by_conf.png", dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Diagnostic 2: signal smoothness
# ---------------------------------------------------------------------------

def autocorr(x, lag):
    if lag <= 0 or lag >= len(x): return np.nan
    a = x[:-lag]; b = x[lag:]
    af = np.isfinite(a) & np.isfinite(b)
    if af.sum() < 100: return np.nan
    a = a[af]; b = b[af]
    a_ = a - a.mean(); b_ = b - b.mean()
    denom = np.sqrt((a_*a_).sum() * (b_*b_).sum())
    if denom == 0: return np.nan
    return float((a_*b_).sum() / denom)


def smoothness_per_date(pred_1s):
    """Return dict of smoothness stats for one day."""
    lags = [1, 2, 4, 8, 20, 40, 120]
    out = {"lag_ticks": lags,
           "lag_seconds": [l*STRIDE_SEC for l in lags],
           "autocorr_raw":  [autocorr(pred_1s, l) for l in lags],
           "autocorr_sign": [autocorr(np.sign(pred_1s).astype(np.float32), l) for l in lags]}
    # sign flips per second
    sign = np.sign(pred_1s)
    flips = int(np.sum(sign[1:] != sign[:-1]))
    dur_sec = len(pred_1s) * STRIDE_SEC
    out["sign_flips_per_sec"] = flips / dur_sec if dur_sec > 0 else np.nan
    # top-10% in/out
    thresh = np.percentile(np.abs(pred_1s), 90)
    in_top10 = np.abs(pred_1s) >= thresh
    transitions = int(np.sum(in_top10[1:] != in_top10[:-1]))
    out["top10_transitions_per_min"] = transitions / (dur_sec/60) if dur_sec > 0 else np.nan
    out["top10_threshold"] = float(thresh)
    # persistence: given in top 10% at t, fraction of next 40 ticks (10s) still in top 10%
    persistence_window = 40
    idx = np.where(in_top10)[0]
    idx = idx[idx + persistence_window < len(in_top10)]
    if len(idx) > 0:
        # build matrix of next-40 booleans
        future = np.stack([in_top10[i+1:i+1+persistence_window] for i in idx])
        # fraction of next-40 still in top 10%
        frac = future.mean(axis=1)
        out["persistence_frac_mean"] = float(frac.mean())
        out["persistence_frac_p50"]  = float(np.percentile(frac, 50))
        out["persistence_frac_p90"]  = float(np.percentile(frac, 90))
        out["persistence_frac_array"] = frac          # for CDF
    else:
        out["persistence_frac_mean"] = np.nan
        out["persistence_frac_p50"]  = np.nan
        out["persistence_frac_p90"]  = np.nan
        out["persistence_frac_array"] = np.array([])
    # mean run length in top 10% (ticks)
    runs = []
    cur = 0
    for v in in_top10:
        if v: cur += 1
        else:
            if cur > 0: runs.append(cur)
            cur = 0
    if cur > 0: runs.append(cur)
    out["top10_mean_run_ticks"] = float(np.mean(runs)) if runs else 0.0
    return out


def smoothness_table(model_name, per_date):
    # Pool stats: average per-day smoothness (weighted by n samples)
    lag_acc = {l: [] for l in [1,2,4,8,20,40,120]}
    lag_acc_sign = {l: [] for l in [1,2,4,8,20,40,120]}
    per_day_summary = []
    pers_array_pool = []
    for date, hd in per_date.items():
        if "1s" not in hd: continue
        p = hd["1s"][0]
        st = smoothness_per_date(p)
        for lag, ac in zip(st["lag_ticks"], st["autocorr_raw"]):
            if np.isfinite(ac): lag_acc[lag].append(ac)
        for lag, ac in zip(st["lag_ticks"], st["autocorr_sign"]):
            if np.isfinite(ac): lag_acc_sign[lag].append(ac)
        per_day_summary.append(dict(
            date=date,
            n=len(p),
            sign_flips_per_sec=st["sign_flips_per_sec"],
            top10_transitions_per_min=st["top10_transitions_per_min"],
            top10_mean_run_ticks=st["top10_mean_run_ticks"],
            persistence_frac_mean=st["persistence_frac_mean"],
        ))
        if len(st["persistence_frac_array"]) > 0:
            pers_array_pool.append(st["persistence_frac_array"])
    rows = []
    for lag in sorted(lag_acc.keys()):
        rows.append(dict(
            lag_ticks=lag,
            lag_seconds=lag*STRIDE_SEC,
            autocorr_raw=float(np.mean(lag_acc[lag])) if lag_acc[lag] else np.nan,
            autocorr_sign=float(np.mean(lag_acc_sign[lag])) if lag_acc_sign[lag] else np.nan,
            n_days=len(lag_acc[lag]),
        ))
    df_smooth = pd.DataFrame(rows)
    df_smooth.to_csv(f"{OUT}/{model_name}_smoothness.csv", index=False)
    df_day = pd.DataFrame(per_day_summary)
    df_day.to_csv(f"{OUT}/{model_name}_smoothness_perday.csv", index=False)
    pers_pool = np.concatenate(pers_array_pool) if pers_array_pool else np.array([])
    return df_smooth, df_day, pers_pool


def persistence_by_conf_table(model_name, per_date, horizons):
    """Per-confidence-band IC and mean run length stats, all horizons."""
    rows = []
    # Pool predictions per horizon
    for h in horizons:
        all_p = []
        all_l = []
        # also need per-day sign-flip rate within band
        sign_flip_in_band = []
        run_in_band = []
        for date, hd in per_date.items():
            if h not in hd: continue
            p, l, _ = hd[h]
            all_p.append(p); all_l.append(l)
        if not all_p: continue
        P = np.concatenate(all_p)
        L = np.concatenate(all_l)
        AP = np.abs(P)
        bands = conf_band_indices(AP)
        for band_label, m in bands.items():
            if m.sum() < 50: continue
            f = np.isfinite(P[m]) & np.isfinite(L[m])
            if f.sum() < 20: continue
            ic = float(np.corrcoef(P[m][f], L[m][f])[0,1])
            rows.append(dict(
                conf_band=band_label,
                horizon=h,
                n=int(m.sum()),
                IC=round(ic, 5),
            ))
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/{model_name}_persistence.csv", index=False)
    return df


def plot_smoothness(model_name, df_smooth, df_day, pers_pool, df_pers):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    # (a) autocorr decay
    ax = axes[0,0]
    ax.plot(df_smooth["lag_seconds"], df_smooth["autocorr_raw"], "o-", label="raw pred")
    ax.plot(df_smooth["lag_seconds"], df_smooth["autocorr_sign"], "s--", label="sign(pred)")
    ax.axhline(0.6, ls=":", color="green", lw=0.8, label="pressure-like (>0.6)")
    ax.axhline(0.3, ls=":", color="red",   lw=0.8, label="noisy (<0.3)")
    ax.set_xscale("log")
    ax.set_xlabel("lag (seconds)")
    ax.set_ylabel("autocorr of 1s-horizon pred")
    ax.set_title("(a) Autocorr decay")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    # (b) sign-flip rate per day
    ax = axes[0,1]
    if len(df_day) > 0:
        ax.bar(range(len(df_day)), df_day["sign_flips_per_sec"], color="#3498db")
        ax.set_xticks(range(len(df_day)))
        ax.set_xticklabels(df_day["date"], rotation=90, fontsize=6)
    ax.set_ylabel("sign-flips per second")
    ax.set_title("(b) Sign-flip rate per day (lower = smoother)")
    ax.grid(alpha=0.3)
    # (c) persistence CDF: given top10% at t, frac of next 10s still top10%
    ax = axes[1,0]
    if len(pers_pool) > 0:
        xs = np.sort(pers_pool)
        ys = np.arange(1, len(xs)+1) / len(xs)
        ax.plot(xs, ys, color="#9b59b6")
        ax.axvline(np.mean(pers_pool), ls="--", color="black", lw=0.8,
                   label=f"mean={np.mean(pers_pool):.3f}")
        ax.legend()
    ax.set_xlabel("fraction of next 10s also in top-10%")
    ax.set_ylabel("CDF")
    ax.set_title("(c) Top-10% persistence (next 10s)")
    ax.grid(alpha=0.3)
    # (d) per-conf-band IC bars at each horizon
    ax = axes[1,1]
    order = ["top0.5%","top1%","top5%","top10%","top20%","top50%","bottom50%"]
    horizons = sorted(df_pers["horizon"].unique(), key=lambda x: int(x.replace("s","")))
    w = 0.8 / len(horizons)
    x = np.arange(len(order))
    for i, h in enumerate(horizons):
        sub = df_pers[df_pers["horizon"]==h].set_index("conf_band").reindex(order)
        ax.bar(x + (i-len(horizons)/2)*w + w/2, sub["IC"].values, w, label=h)
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("IC")
    ax.set_title("(d) IC by confidence band & horizon")
    ax.axhline(0, color="black", lw=0.5)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle(f"{model_name} — signal smoothness diagnostics")
    fig.tight_layout()
    fig.savefig(f"{OUT}/{model_name}_smoothness.png", dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_model(model_name, info):
    files = sorted(glob.glob(info["glob"]))
    per_date = {}
    for fp in files:
        # skip empty placeholders by size
        try:
            sz = os.path.getsize(fp)
        except OSError:
            continue
        if sz < 50000:           # placeholder files for empty days
            continue
        try:
            hd = info["loader"](fp)
        except Exception as e:
            print(f"  ! skip {fp}: {e}")
            continue
        if not hd: continue
        # extract date from filename
        bn = os.path.basename(fp)
        date = "".join(ch for ch in bn if ch.isdigit())[:8]
        per_date[date] = hd
    print(f"  loaded {len(per_date)} valid dates for {model_name}")
    return per_date


def summary_paragraph(stats):
    """stats: dict model_name -> small dict for SUMMARY.md"""
    lines = []
    lines.append("# HC #450 R3+R4 Diagnostics Summary\n")
    lines.append(f"Generated: {pd.Timestamp.now()}  \n")
    lines.append(f"Commission used: 0.376 ticks (= $4.70 / $12.50)  \n")
    lines.append(f"Stride: {STRIDE_SEC*1000:.0f} ms  \n\n")
    lines.append("## Models processed\n")
    for m, s in stats.items():
        lines.append(f"- **{m}**: {s['n_days']} OOT days, {s['total_preds']:,} pooled predictions @ 1s horizon\n")
    lines.append("\n## 1. Are we trading pressure or noise?\n")
    lines.append("Criterion: autocorr at lag-4 (=1s after prediction). >0.6 = pressure-like (signal persists, model is predicting a slow-moving order-book imbalance). <0.3 = noisy (signal flips faster than we can act on it).\n\n")
    lines.append("| model | autocorr lag-1 (250ms) | autocorr lag-4 (1s) | autocorr lag-40 (10s) | sign-flips/sec | top-10% mean run (s) | verdict |\n")
    lines.append("|---|---|---|---|---|---|---|\n")
    for m, s in stats.items():
        a1 = s["autocorr"].get(1, np.nan)
        a4 = s["autocorr"].get(4, np.nan)
        a40 = s["autocorr"].get(40, np.nan)
        verdict = "PRESSURE-LIKE" if a4 > 0.6 else ("MIXED" if a4 > 0.3 else "NOISY")
        run_sec = s["top10_mean_run_ticks"] * STRIDE_SEC
        lines.append(f"| {m} | {a1:.3f} | {a4:.3f} | {a40:.3f} | {s['sign_flips_per_sec']:.3f} | {run_sec:.2f} | {verdict} |\n")
    lines.append("\n## 2. Is PatchTST smoother than CNN-Mamba v3.4.2?\n")
    if "patchtst" in stats and "cnn_mamba_v3_4_2" in stats:
        sp = stats["patchtst"]; sv = stats["cnn_mamba_v3_4_2"]
        lines.append(f"- autocorr lag-4 (1s): PatchTST = {sp['autocorr'].get(4,float('nan')):.3f} vs v3.4.2 = {sv['autocorr'].get(4,float('nan')):.3f}\n")
        lines.append(f"- sign-flips/sec: PatchTST = {sp['sign_flips_per_sec']:.3f} vs v3.4.2 = {sv['sign_flips_per_sec']:.3f}\n")
        lines.append(f"- top-10% mean run (s): PatchTST = {sp['top10_mean_run_ticks']*STRIDE_SEC:.2f} vs v3.4.2 = {sv['top10_mean_run_ticks']*STRIDE_SEC:.2f}\n")
        verdict = "YES, PatchTST is smoother" if sp['autocorr'].get(4,0) > sv['autocorr'].get(4,0) else "NO, v3.4.2 is at least as smooth"
        lines.append(f"- **verdict: {verdict}**\n")
    lines.append("\n## 3. Should we smooth our signal before trading?\n")
    for m, s in stats.items():
        a1 = s["autocorr"].get(1, np.nan)
        a4 = s["autocorr"].get(4, np.nan)
        decay = a1 - a4
        if a4 < 0.3:
            rec = "YES — heavy smoothing (EMA ~1-2s) before any trade decision."
        elif a4 < 0.6:
            rec = "YES — modest smoothing (EMA ~500ms) recommended; raw signal is mid-noise."
        else:
            rec = "NO — signal is already persistent, smoothing wastes information."
        lines.append(f"- **{m}**: lag-1 autocorr={a1:.3f}, lag-4={a4:.3f} (decay {decay:.3f}). → {rec}\n")
    lines.append("\n## 4. Top-10% signal persistence (how many seconds does it stay 'extreme'?)\n")
    lines.append("| model | mean run (ticks) | mean run (sec) | given top-10% at t, frac of next 10s also top-10% |\n")
    lines.append("|---|---|---|---|\n")
    for m, s in stats.items():
        lines.append(f"| {m} | {s['top10_mean_run_ticks']:.1f} | {s['top10_mean_run_ticks']*STRIDE_SEC:.2f} | {s.get('persistence_mean', float('nan')):.3f} |\n")
    lines.append("\n## 5. Top-10% short signal MFE/MAE headline (per HC #450 R3)\n")
    lines.append("Direction-adjusted mean realized move (ticks) for the top-10% short signals (pred<0, |pred| in top decile of |pred|):\n\n")
    lines.append("| model | horizon | n | mean realized (tk) | p90 realized | win rate | net after 0.376tk commission |\n")
    lines.append("|---|---|---|---|---|---|---|\n")
    for m, s in stats.items():
        for h in s.get("horizons", []):
            row = s.get("top10_short_row", {}).get(h)
            if row is None: continue
            lines.append(f"| {m} | {h} | {row['n']} | {row['mean_realized_tk']:.3f} | {row['p90_realized_tk']:.3f} | {row['win_rate']:.3f} | {row['net_after_comm_376_tk']:.3f} |\n")
    lines.append("\n## 6. Top-1% MFE/MAE headline\n")
    lines.append("| model | side | horizon | n | mean realized (tk) | net after commission |\n")
    lines.append("|---|---|---|---|---|---|\n")
    for m, s in stats.items():
        for side in ["long","short"]:
            for h in s.get("horizons", []):
                row = s.get(f"top1_{side}_row", {}).get(h)
                if row is None: continue
                lines.append(f"| {m} | {side} | {h} | {row['n']} | {row['mean_realized_tk']:.3f} | {row['net_after_comm_376_tk']:.3f} |\n")
    lines.append("\n## Caveat on MFE/MAE proxy\n")
    lines.append("The NPZs do not contain per-sample MFE/MAE arrays for the OOT set "
                "(v3.4.2 has the fields but they are mask=0). We use the realized "
                "close-of-horizon move (`labels_h`, in ticks) as the proxy. True MFE "
                "within the horizon is **larger** than what we report; true MAE "
                "(downside excursion) is **worse** than what we report. To get true "
                "MFE/MAE, regenerate the OOT NPZs from MBO replay with the bookkeeping "
                "enabled.\n")
    return "".join(lines)


def main():
    stats = {}
    for model_name, info in MODELS.items():
        print(f"\n=== {model_name} ===")
        per_date = load_model(model_name, info)
        if not per_date:
            print("  no data, skipping")
            continue
        df_mm = mfe_mae_table(model_name, per_date)
        plot_mfe_mae(model_name, df_mm)
        df_smooth, df_day, pers_pool = smoothness_table(model_name, per_date)
        df_pers = persistence_by_conf_table(model_name, per_date, info["horizons"])
        plot_smoothness(model_name, df_smooth, df_day, pers_pool, df_pers)
        # collect stats for summary
        autocorr_dict = dict(zip(df_smooth["lag_ticks"], df_smooth["autocorr_raw"]))
        # pooled top-10% mean run / sign flips: take across-day mean weighted by n
        if len(df_day) > 0:
            weights = df_day["n"] / df_day["n"].sum()
            sf = float((df_day["sign_flips_per_sec"] * weights).sum())
            run = float((df_day["top10_mean_run_ticks"] * weights).sum())
            pmean = float((df_day["persistence_frac_mean"] * weights).sum())
        else:
            sf = run = pmean = float("nan")
        # top-10% / top-1% short row for each horizon
        top10_short = {}; top1_long = {}; top1_short = {}
        for h in info["horizons"]:
            sh = df_mm[(df_mm["horizon"]==h)&(df_mm["side"]=="short")&(df_mm["conf_band"]=="top10%")]
            if len(sh): top10_short[h] = sh.iloc[0].to_dict()
            sl = df_mm[(df_mm["horizon"]==h)&(df_mm["side"]=="long")&(df_mm["conf_band"]=="top1%")]
            if len(sl): top1_long[h] = sl.iloc[0].to_dict()
            ss = df_mm[(df_mm["horizon"]==h)&(df_mm["side"]=="short")&(df_mm["conf_band"]=="top1%")]
            if len(ss): top1_short[h] = ss.iloc[0].to_dict()
        total_preds = sum(len(hd["1s"][0]) for hd in per_date.values() if "1s" in hd)
        stats[model_name] = dict(
            n_days=len(per_date),
            total_preds=total_preds,
            autocorr=autocorr_dict,
            sign_flips_per_sec=sf,
            top10_mean_run_ticks=run,
            persistence_mean=pmean,
            horizons=info["horizons"],
            top10_short_row=top10_short,
            top1_long_row=top1_long,
            top1_short_row=top1_short,
        )
    # Save SUMMARY
    md = summary_paragraph(stats)
    with open(f"{OUT}/SUMMARY.md", "w") as fh:
        fh.write(md)
    # Save stats json for re-use
    stats_serializable = {}
    for m, s in stats.items():
        s2 = {k: v for k, v in s.items() if k not in ("top10_short_row","top1_long_row","top1_short_row")}
        s2["autocorr"] = {int(k): float(v) if np.isfinite(v) else None for k, v in s["autocorr"].items()}
        stats_serializable[m] = s2
    with open(f"{OUT}/stats.json", "w") as fh:
        json.dump(stats_serializable, fh, indent=2, default=str)
    print(f"\n[done] outputs in {OUT}")


if __name__ == "__main__":
    main()
