"""
HC #410 — Three-way comparison: v2 vs v3.3 vs v3.4.2.

Builds a unified comparable matrix on:
  model x horizon x side x conf_tier
with metrics: n, mfe_mean_tk, mae_mean_tk, net_tk_per_fill, wr_pct,
day_conc, sharpe_per_fill, ci_low_95_net, promote.

Inputs:
  v2:    /home/jupiter/Lvl3Quant/output/cnn_mamba_v2_all_oot/*_predictions.npz
         (96 daily NPZs, horizons 1s/5s/10s; labels are {0,0.5,1} discrete)
  v3.3:  /home/jupiter/Lvl3Quant/output/v3_3_extended_oot_20260514/extended_oot_predictions.npz
  v3.4.2: concat of v342_5d.npz + v342_11d_ext.npz under
          /home/jupiter/Lvl3Quant/data/v342_oot_npz/

CPU-only Jupiter. Additive — does NOT modify HC #408 script.
"""
from __future__ import annotations

import glob
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path("/home/jupiter/Lvl3Quant")
V2_DIR = PROJ / "output" / "cnn_mamba_v2_all_oot"
V33_NPZ = PROJ / "output" / "v3_3_extended_oot_20260514" / "extended_oot_predictions.npz"
V342_5D = PROJ / "data" / "v342_oot_npz" / "v342_5d.npz"
V342_11D = PROJ / "data" / "v342_oot_npz" / "v342_11d_ext.npz"

HORIZONS_FULL = ["1s", "5s", "10s", "30s"]
HORIZONS_V2 = ["1s", "5s", "10s"]
SIDES = ["long", "short"]
CONF_TIERS = [
    ("Top0.5", 0.005),
    ("Top1",   0.01),
    ("Top5",   0.05),
    ("Top10",  0.10),
]

COMMISSION = 0.376
HONESTY_N_FILLS_MIN = 50
HONESTY_DAY_CONC_MAX = 0.20

TS = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = PROJ / "output" / f"hc410_three_way_comparison_{TS}"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def mean_ci_lower(arr, z=1.96):
    n = len(arr)
    if n < 2:
        return float("nan")
    se = arr.std(ddof=1) / np.sqrt(n)
    return float(arr.mean() - z * se)


def compute_cell_metrics(
    pred,        # 1d float (already sign-aligned: higher=stronger long OR stronger short conv depending on context)
    target_tk,   # 1d float, realized horizon-end log-ret (in ticks units); may be None for v2
    label_disc,  # 1d {0,0.5,1} or None — for v2 hit-rate
    side_sign,   # +1 long, -1 short
    mae_tk,      # 1d adverse-move ticks or None
    date_idx,    # 1d int per sample
    n_dates,     # int
    tier_frac,   # float
):
    """Return metrics dict for one (side, tier) cell. pred is RAW (unsigned)."""
    signed_pred = side_sign * pred
    ok = np.isfinite(signed_pred)
    if target_tk is not None:
        ok = ok & np.isfinite(target_tk)
    n_ok = int(ok.sum())
    if n_ok == 0:
        return None
    ok_idx = np.where(ok)[0]
    k = max(1, int(n_ok * tier_frac))
    signed_ok = signed_pred[ok_idx]
    if k >= len(signed_ok):
        sel = ok_idx
    else:
        part = np.argpartition(-signed_ok, k - 1)[:k]
        sel = ok_idx[part]
    n_cell = len(sel)
    if n_cell == 0:
        return None

    out = dict(n=n_cell)

    # day_conc
    counts = np.bincount(date_idx[sel], minlength=n_dates)
    day_conc = float(counts.max() / n_cell) if n_cell else 1.0
    out["day_conc"] = round(day_conc, 4)

    if target_tk is not None:
        realized = side_sign * target_tk[sel]
        net = realized - COMMISSION
        mean_mfe = float(realized.mean())
        mean_net = float(net.mean())
        std_net = float(net.std(ddof=1)) if n_cell > 1 else float("nan")
        sharpe = float(mean_net / std_net) if std_net and std_net > 0 else float("nan")
        ci_low = mean_ci_lower(net)
        wr = float((realized > 0).mean())
        out["mfe_mean_tk"] = round(mean_mfe, 4)
        out["net_tk_per_fill"] = round(mean_net, 4)
        out["wr_pct"] = round(100 * wr, 2)
        out["sharpe_per_fill"] = round(sharpe, 4)
        out["ci_low_95_net"] = round(ci_low, 4)
        out["promote"] = bool(
            n_cell >= HONESTY_N_FILLS_MIN
            and day_conc <= HONESTY_DAY_CONC_MAX
            and ci_low > 0
        )
    else:
        # v2 fallback: hit-rate using discrete labels
        out["mfe_mean_tk"] = float("nan")
        out["net_tk_per_fill"] = float("nan")
        out["sharpe_per_fill"] = float("nan")
        out["ci_low_95_net"] = float("nan")
        if label_disc is not None:
            lab = label_disc[sel]
            if side_sign > 0:  # long: want label == 1.0
                hits = (lab == 1.0)
                wrong = (lab == 0.0)
            else:              # short: want label == 0.0
                hits = (lab == 0.0)
                wrong = (lab == 1.0)
            # Hit rate excluding the "unclear" 0.5 bin
            decisive = hits | wrong
            n_dec = int(decisive.sum())
            wr = float(hits.sum() / n_dec) if n_dec else float("nan")
            out["wr_pct"] = round(100 * wr, 2) if np.isfinite(wr) else float("nan")
        else:
            out["wr_pct"] = float("nan")
        # No promote without net
        out["promote"] = False

    if mae_tk is not None:
        # MAE is reported as adverse move magnitude (positive ticks against position)
        # target_pred_mae_30s_ticks is the realized worst adverse in the next 30s
        mae_vals = mae_tk[sel]
        mae_vals = mae_vals[np.isfinite(mae_vals)]
        if mae_vals.size > 0:
            out["mae_mean_tk"] = round(float(mae_vals.mean()), 4)
        else:
            out["mae_mean_tk"] = float("nan")
    else:
        out["mae_mean_tk"] = float("nan")

    return out


def load_v2():
    """Load all 96 daily v2 NPZs. Returns dict of {horizon: (pred, label_disc, date_idx)} and n_dates."""
    import re
    all_files = sorted(glob.glob(str(V2_DIR / "*_predictions.npz")))
    # Keep only YYYYMMDD-prefixed files (skip broken fold_NN symlinks)
    files = [f for f in all_files if re.match(r"^\d{8}_predictions\.npz$", Path(f).name)]
    # Filter out broken symlinks
    files = [f for f in files if Path(f).exists()]
    _log(f"v2: {len(files)} valid daily files (filtered from {len(all_files)})")
    preds_all = {h: [] for h in HORIZONS_V2}
    labels_all = {h: [] for h in HORIZONS_V2}
    date_idx_all = {h: [] for h in HORIZONS_V2}
    dates_used = []
    for di, f in enumerate(files):
        d = np.load(f, allow_pickle=True)
        p = d["predictions"]  # (N, 3)
        l = d["labels"]       # (N, 3)
        n = p.shape[0]
        for hi, h in enumerate(HORIZONS_V2):
            preds_all[h].append(p[:, hi].astype(np.float32))
            labels_all[h].append(l[:, hi].astype(np.float32))
            date_idx_all[h].append(np.full(n, di, dtype=np.int32))
        dates_used.append(str(d["date"]))
    out = {}
    for h in HORIZONS_V2:
        out[h] = (
            np.concatenate(preds_all[h]),
            np.concatenate(labels_all[h]),
            np.concatenate(date_idx_all[h]),
        )
    n_dates = len(files)
    _log(f"v2: concatenated, n_samples={out['1s'][0].shape[0]}, n_dates={n_dates}")
    return out, n_dates, dates_used


def load_v33():
    _log(f"v3.3: loading {V33_NPZ}")
    d = np.load(V33_NPZ, allow_pickle=False)
    n = d["pred_log_ret_1s"].shape[0]
    oot_dates = list(d["oot_dates"])
    n_dates = len(oot_dates)
    samples_per_day = n // n_dates
    date_idx = np.minimum(np.arange(n) // samples_per_day, n_dates - 1).astype(np.int32)
    bundle = {}
    for h in HORIZONS_FULL:
        bundle[h] = (d[f"pred_log_ret_{h}"], d[f"target_log_ret_{h}"])
    mae = d["target_pred_mae_30s_ticks"] if "target_pred_mae_30s_ticks" in d.files else None
    _log(f"v3.3: n={n}, n_dates={n_dates}")
    return bundle, mae, date_idx, n_dates, oot_dates


def load_v342():
    _log("v3.4.2: loading 5d + 11d_ext")
    d1 = np.load(V342_5D, allow_pickle=False)
    d2 = np.load(V342_11D, allow_pickle=False)
    bundle = {}
    for h in HORIZONS_FULL:
        p = np.concatenate([d1[f"pred_log_ret_{h}"], d2[f"pred_log_ret_{h}"]])
        t = np.concatenate([d1[f"target_log_ret_{h}"], d2[f"target_log_ret_{h}"]])
        bundle[h] = (p, t)
    mae = None
    if "target_pred_mae_30s_ticks" in d1.files and "target_pred_mae_30s_ticks" in d2.files:
        mae = np.concatenate([d1["target_pred_mae_30s_ticks"], d2["target_pred_mae_30s_ticks"]])
    n1 = d1["pred_log_ret_1s"].shape[0]
    n2 = d2["pred_log_ret_1s"].shape[0]
    n = n1 + n2
    # Date attribution: 5 dates over n1 samples, 11 dates over n2 samples
    dates_5 = ["20260223", "20260224", "20260225", "20260226", "20260227"]
    dates_11 = ["20260301", "20260302", "20260303", "20260304", "20260305",
                "20260308", "20260309", "20260310", "20260311", "20260312", "20260315"]
    oot_dates = dates_5 + dates_11
    n_dates = len(oot_dates)
    sub1 = n1 // 5
    sub2 = n2 // 11
    di1 = np.minimum(np.arange(n1) // sub1, 4).astype(np.int32)
    di2 = np.minimum(np.arange(n2) // sub2, 10).astype(np.int32) + 5
    date_idx = np.concatenate([di1, di2])
    _log(f"v3.4.2: n={n} (5d={n1}, 11d={n2}), n_dates={n_dates}")
    return bundle, mae, date_idx, n_dates, oot_dates


def evaluate_model(model_name, horizons, bundle, mae, date_idx, n_dates, label_disc_by_h=None):
    rows = []
    for h in horizons:
        pred, target = bundle[h]
        label_disc = None
        if label_disc_by_h is not None:
            label_disc = label_disc_by_h.get(h)
        for side in SIDES:
            sign = 1.0 if side == "long" else -1.0
            for tier_name, tier_frac in CONF_TIERS:
                m = compute_cell_metrics(
                    pred=pred,
                    target_tk=target,
                    label_disc=label_disc,
                    side_sign=sign,
                    mae_tk=mae if h == "30s" else None,  # MAE is 30s-window in schema
                    date_idx=date_idx,
                    n_dates=n_dates,
                    tier_frac=tier_frac,
                )
                if m is None:
                    continue
                row = dict(
                    model=model_name,
                    horizon=h,
                    side=side,
                    conf_tier=tier_name,
                    **m,
                )
                rows.append(row)
    return rows


def main():
    t0 = time.time()
    all_rows = []

    # v2 — has only discrete labels; pred sign × label sign = directional hit
    # For mfe_mean_tk, target_tk for v2 is not in ticks; we set target=None and rely on hit-rate.
    v2_bundle_raw, v2_n_dates, v2_dates = load_v2()
    v2_bundle = {h: (v2_bundle_raw[h][0], None) for h in HORIZONS_V2}
    v2_label_disc = {h: v2_bundle_raw[h][1] for h in HORIZONS_V2}
    v2_date_idx = v2_bundle_raw["1s"][2]  # same for all horizons since same files
    all_rows += evaluate_model("v2", HORIZONS_V2, v2_bundle, None, v2_date_idx, v2_n_dates,
                                label_disc_by_h=v2_label_disc)

    # v2 IC (correlation pred vs discretized label)
    v2_ic = {}
    for h in HORIZONS_V2:
        p = v2_bundle_raw[h][0]
        l = v2_bundle_raw[h][1]
        ok = np.isfinite(p) & np.isfinite(l)
        if ok.sum() > 10:
            v2_ic[h] = float(np.corrcoef(p[ok], l[ok])[0, 1])
        else:
            v2_ic[h] = float("nan")
    _log(f"v2 IC: {v2_ic}")

    # v3.3
    v33_bundle, v33_mae, v33_di, v33_nd, _ = load_v33()
    all_rows += evaluate_model("v3.3", HORIZONS_FULL, v33_bundle, v33_mae, v33_di, v33_nd)

    # v3.4.2
    v342_bundle, v342_mae, v342_di, v342_nd, _ = load_v342()
    all_rows += evaluate_model("v3.4.2", HORIZONS_FULL, v342_bundle, v342_mae, v342_di, v342_nd)

    df = pd.DataFrame(all_rows)
    cols = ["model", "horizon", "side", "conf_tier", "n",
            "mfe_mean_tk", "mae_mean_tk", "net_tk_per_fill", "wr_pct",
            "day_conc", "sharpe_per_fill", "ci_low_95_net", "promote"]
    df = df[[c for c in cols if c in df.columns]]
    csv_path = OUT_DIR / "comparison_matrix.csv"
    df.to_csv(csv_path, index=False)
    _log(f"matrix: {csv_path} ({len(df)} rows)")

    # Top 10 promoted per model
    top_rows = []
    for m in ["v2", "v3.3", "v3.4.2"]:
        sub = df[(df["model"] == m) & (df["promote"] == True)].copy()
        sub = sub.sort_values("net_tk_per_fill", ascending=False).head(10)
        top_rows.append(sub)
    top_df = pd.concat(top_rows) if top_rows else pd.DataFrame()
    top_path = OUT_DIR / "top10_per_model.csv"
    top_df.to_csv(top_path, index=False)
    _log(f"top10 per model: {top_path}")

    # Verdict
    promoted_by_model = {m: int(((df["model"] == m) & (df["promote"] == True)).sum())
                         for m in ["v2", "v3.3", "v3.4.2"]}
    avg_net_by_model = {}
    for m in ["v3.3", "v3.4.2"]:
        sub = df[(df["model"] == m) & (df["promote"] == True)]
        avg_net_by_model[m] = float(sub["net_tk_per_fill"].mean()) if len(sub) else float("nan")

    # Per-cell winners (highest net_tk_per_fill across v3.3 vs v3.4.2 at each h × side × tier)
    winners = []
    for h in HORIZONS_FULL:
        for side in SIDES:
            for tier_name, _ in CONF_TIERS:
                sub = df[(df["horizon"] == h) & (df["side"] == side)
                         & (df["conf_tier"] == tier_name)
                         & (df["model"].isin(["v3.3", "v3.4.2"]))]
                if len(sub) == 0:
                    continue
                sub = sub.dropna(subset=["net_tk_per_fill"])
                if len(sub) == 0:
                    continue
                idx = sub["net_tk_per_fill"].idxmax()
                w = sub.loc[idx]
                winners.append((h, side, tier_name, w["model"],
                                float(w["net_tk_per_fill"]),
                                bool(w["promote"]), int(w["n"])))
    winner_counts = {"v3.3": 0, "v3.4.2": 0}
    for w in winners:
        winner_counts[w[3]] += 1

    # Best overall = most promoted + best avg net
    def headline_pick():
        v33_score = promoted_by_model["v3.3"] + (0 if np.isnan(avg_net_by_model.get("v3.3", float('nan'))) else avg_net_by_model["v3.3"])
        v342_score = promoted_by_model["v3.4.2"] + (0 if np.isnan(avg_net_by_model.get("v3.4.2", float('nan'))) else avg_net_by_model["v3.4.2"])
        if v33_score >= v342_score:
            return "v3.3"
        return "v3.4.2"
    headline = headline_pick()

    def top3(m):
        sub = df[(df["model"] == m) & (df["promote"] == True)].sort_values(
            "net_tk_per_fill", ascending=False).head(3)
        if len(sub) == 0:
            return [f"(no promoted cells for {m})"]
        return [
            f"{r['horizon']}/{r['side']}/{r['conf_tier']}: n={int(r['n'])}, "
            f"net={r['net_tk_per_fill']:.3f} tk, WR={r['wr_pct']:.1f}%, "
            f"day_conc={r['day_conc']:.2f}, Sharpe/fill={r['sharpe_per_fill']:.3f}"
            for _, r in sub.iterrows()
        ]

    lines = []
    lines.append(f"# HC #410 — Three-way comparison (v2 vs v3.3 vs v3.4.2)\n")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append(f"Output dir: `{OUT_DIR}`\n")
    lines.append("## Headline")
    lines.append(f"- **Recommended model: {headline}**")
    lines.append(f"- Promoted cells per model (n>=50, day_conc<=0.20, CI_low_95(net)>0):")
    for m in ["v2", "v3.3", "v3.4.2"]:
        lines.append(f"  - {m}: {promoted_by_model[m]} cells")
    lines.append(f"- Avg net_tk_per_fill across PROMOTED cells:")
    for m in ["v3.3", "v3.4.2"]:
        v = avg_net_by_model.get(m, float("nan"))
        lines.append(f"  - {m}: {v:.4f} tk/fill" if np.isfinite(v) else f"  - {m}: n/a")
    lines.append(f"- Per-cell wins (v3.3 vs v3.4.2 on net_tk_per_fill across all 16 cells = 4h x 2side x 4tier - lookups):")
    lines.append(f"  - v3.3 wins: {winner_counts['v3.3']}")
    lines.append(f"  - v3.4.2 wins: {winner_counts['v3.4.2']}")
    lines.append("")

    lines.append("## Top 3 promoted cells per model")
    for m in ["v2", "v3.3", "v3.4.2"]:
        lines.append(f"### {m}")
        for line in top3(m):
            lines.append(f"- {line}")
        lines.append("")

    lines.append("## v2 IC (continuous pred vs discrete label {0,0.5,1})")
    for h, v in v2_ic.items():
        lines.append(f"- IC_{h} = {v:.4f}")
    lines.append("")

    lines.append("## Per-cell winners (v3.3 vs v3.4.2)")
    lines.append("| horizon | side | conf_tier | winner | net (tk) | promoted | n |")
    lines.append("|---|---|---|---|---|---|---|")
    for h, side, tier, m, net, promo, n in winners:
        lines.append(f"| {h} | {side} | {tier} | **{m}** | {net:.4f} | {promo} | {n} |")
    lines.append("")

    lines.append("## Concrete next step")
    if promoted_by_model["v3.3"] == 0 and promoted_by_model["v3.4.2"] == 0:
        lines.append("- **No model has any cell passing the honesty gate** (n>=50, day_conc<=0.20, CI_low_95(net)>0).")
        lines.append("- DO NOT wire any model into live paper trader yet.")
        lines.append("- Investigate: either commission-only net is genuinely unprofitable on these OOT periods, OR confidence ranking is not concentrating fills (day_conc>0.20).")
    elif headline == "v3.4.2" and promoted_by_model["v3.4.2"] > promoted_by_model["v3.3"]:
        lines.append("- **Resume v3.4.2 training to convergence**, then wire the top-3 promoted cells into Razer paper trader.")
    elif headline == "v3.3":
        lines.append("- **Wire v3.3 (trial 278 / extended OOT) top promoted cells into Razer paper trader.**")
        lines.append("- v3.4.2 status: keep as candidate, do not deploy yet — fewer/lower promoted cells than v3.3 on overlapping methodology.")
    else:
        lines.append(f"- Headline pick is {headline} — proceed with that for paper-trader wiring.")
    lines.append("")

    lines.append("## Limitations & caveats")
    lines.append("- **v2 label encoding**: labels are discrete {0, 0.5, 1.0} (down / unclear / up), NOT continuous tick returns. Therefore:")
    lines.append("  - `mfe_mean_tk`, `net_tk_per_fill`, `sharpe_per_fill`, `ci_low_95_net` are **NaN** for v2.")
    lines.append("  - `wr_pct` for v2 is the directional hit-rate on decisive samples (excluding the 0.5 'unclear' bin) — NOT comparable apples-to-apples to v3.3/v3.4.2 wr_pct (which is sign of horizon-end log-return).")
    lines.append("  - `promote` is always False for v2 (no tick-level net available).")
    lines.append("- **v3.4.2 dates**: 5d NPZ covers 02-23..02-27, 11d_ext covers 03-01..03-15 (11 RTH days). Date attribution uses uniform chunking per the HC #408 convention since `oot_dates` is not stored in the v3.4.2 NPZs.")
    lines.append("- **MAE**: only available at the 30s horizon (from `target_pred_mae_30s_ticks`); other horizons report NaN.")
    lines.append("- **Cost model**: net = realized - 0.376 ticks (commission only, HC #405 fill-price framing; passive entry assumed). Market-cross would subtract additional 1.0 tick.")
    lines.append("- **OOT-period overlap**: v3.3 = 15 RTH days in March 2026; v3.4.2 = 16 RTH days (Feb 23 - Mar 15). v2 = 96 days. Periods are NOT identical, so cross-model numbers are indicative not strictly head-to-head.")
    lines.append("- **Confidence ranking**: top-k by |signed pred|, NOT by predicted realized vol. This is the simplest comparable ranking across all three models.")

    verdict_path = OUT_DIR / "verdict.md"
    verdict_path.write_text("\n".join(lines))
    _log(f"verdict: {verdict_path}")

    elapsed = time.time() - t0
    _log(f"DONE in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
